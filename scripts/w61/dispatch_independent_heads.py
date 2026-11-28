"""Continue each W61 hybrid head from its own immutable checkpoint, through the shared pair pool."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import dispatch_learning as previous
from dispatch_learning import campaign, pool
from hybrid import ROOT, SPEC
from s1_w59_learner_critic import atomic_json, file_digest

RUN, SMALL = campaign.RUN, campaign.SMALL
SELF = Path(__file__).resolve()
REGISTRY = SMALL / "independent-dispatch.json"
HANDOFF = SMALL / "independent-dispatch-handoff.json"
LIVE = SMALL / "independent-dispatch-live.json"
STATE = RUN / "independent-dispatch"
SEATS = STATE / "seats.json"
CURVES = RUN / "render-queue" / "curve-ready"
ORDER = [(m,s) for s in SPEC["seeds"] for m in ("E-score","PPO","SPO")]
ORDER += [(SPEC["geometry_ablation"],s) for s in SPEC["seeds"]]
METHOD = {campaign.head_name(m,s):(m,s) for m,s in ORDER}
PRIMARY = [k for k,(m,s) in METHOD.items() if m in SPEC["methods"]]
ABLATION = [k for k in METHOD if k not in PRIMARY]
BATCH = 2 * SPEC["batch_per_case"]
VERSIONS = SPEC["used_per_head"] // BATCH
NODES = [n // BATCH for n in SPEC["curve_nodes"]]
PREFIX = "w61-"+campaign.STUDY+"-"


def read(path):
    return json.loads(path.read_text())


def pid_record(pid):
    try:
        proc=Path(f"/proc/{pid}")
        fields=(proc/"stat").read_text().split(") ",1)[1].split()
        return dict(pid=pid,state=fields[0],start_ticks=int(fields[19]),
            argv=[arg.decode() for arg in (proc/"cmdline").read_bytes().split(b"\0") if arg])
    except FileNotFoundError:
        return None


def lock(path, stack, wait=False):
    stream=stack.enter_context(path.open("a"))
    fcntl.flock(stream,fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))


def head_job(head, version):
    return f"{PREFIX}{head}-train-v{version:03d}"


def history(head):
    """The durable commit boundary is campaign's train receipt, following its checkpoint write."""
    directory=SMALL/head
    if (directory/"terminal.json").exists():
        previous_terminal=read(directory/"terminal.json")
        terminal=previous_terminal if previous_terminal.get("role","train")=="train" else None
    else:
        terminal=None
    latest=0
    last=directory/"initial.json"
    for v in range(VERSIONS):
        receipt=directory/f"train-{v:03d}.json"
        if not receipt.is_file():
            break
        row=read(receipt)
        if row["status"]!="updated":
            terminal={"failed_version":v,"status":row["status"],"source":str(receipt)}
            break
        latest=v+1
        last=receipt
    return dict(head=head,version=latest,used=BATCH*latest,ready_at=last.stat().st_mtime,
        source=str(last),terminal=terminal,frozen=latest==VERSIONS)


def checkpoint(head,version):
    source=SMALL/head/(f"train-{version-1:03d}.json" if version else "initial.json")
    r=read(source)
    name="updated_checkpoint" if version else "checkpoint"
    key="updated_checkpoint_sha256" if version else "sha256"
    path=Path(r[name])
    if file_digest(path)!=r[key]:
        raise ValueError(f"Published checkpoint differs for {head} version {version}")
    return path,r[key],source


def deliver_nodes(head,latest):
    method,seed=METHOD[head]
    for v in NODES:
        if v>latest: break
        notice=CURVES/f"{head}-v{v:03d}.json"
        freeze=SMALL/head/"freeze-08192.json"
        if notice.exists() and (v!=VERSIONS or freeze.exists()):
            continue
        policy,sha,source=checkpoint(head,v)
        rec=dict(study=campaign.STUDY,head=head,method=method,seed=seed,version=v,used=BATCH*v,
            checkpoint=str(policy),checkpoint_sha256=sha,source_receipt=str(source),source_sha256=file_digest(source),
            policy_published_unix=source.stat().st_mtime,notified_unix=time.time(),
            evaluation_is_separate=True,training_waits_for_evaluation=False)
        if not notice.exists():
            atomic_json(notice,rec)
        if v==VERSIONS and not freeze.exists():
            atomic_json(freeze,dict(rec,status="independently_frozen",final_inputs_retained=True))


def active_jobs():
    rows=previous.inventory()
    def owns_work(row):
        processes=[row.get("owner_pid")]
        if row["state"]=="running":
            processes.append(row.get("pid"))
        return any(p and (rec:=pid_record(p)) and rec["state"]!="Z" for p in processes)
    return [r for r in rows if owns_work(r)]


def identify(row):
    name=row["job"]
    if not name.startswith(PREFIX) or "-train-v" not in name:return None
    head,version=name[len(PREFIX):].rsplit("-train-v",1)
    return (head,int(version)) if head in METHOD else None


def start_pair(head,hist):
    method,seed=METHOD[head]
    version=hist["version"]
    name=head_job(head,version)
    ck,sha,source=checkpoint(head,version)
    args=[sys.executable,str(previous.LEASE),"--runtime",str(ROOT),"--job",name,"--slots","2",
        "--cores","2","--ram-gib","5.5","--gpu-mib","1500","--scratch-gib","1.0","--",sys.executable,
        "-u",str(campaign.SELF),"paired",method,str(seed),str(version),"train"]
    log=STATE/"logs"/(name+".log")
    with log.open("a") as out:
        child=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    note=STATE/"admissions"/(name+".json")
    atomic_json(note,dict(study=campaign.STUDY,head=head,method=method,seed=seed,version=version,
        new_used_target=(version+1)*BATCH,train_job=name,pair_owner_pid=child.pid,
        checkpoint=str(ck),checkpoint_sha256=sha,previous_training_receipt=str(source),
        prior_published_unix=hist["ready_at"],requested_unix=time.time(),command=args))
    return child


def note_start(row,head,version):
    name=row["job"]
    path=STATE/"admissions"/(name+".json")
    if not path.exists() or not row.get("started"):return
    record=read(path)
    if record.get("lease_started_unix") is not None:return
    registered=datetime.fromisoformat(row["registered"]).timestamp()
    started=datetime.fromisoformat(row["started"]).timestamp()
    record.update(lease_registered_unix=registered,lease_started_unix=started,
        train_to_next_lease_s=started-record["prior_published_unix"],
        ready_to_feeder_request_s=record["requested_unix"]-record["prior_published_unix"],
        shared_queue_delay_s=started-registered,slots=row.get("slots"),paired_pid=row.get("pid"))
    atomic_json(path,record)


def third_permitted(jobs,resources,speed):
    ours=[r for r in jobs if identify(r) and r["state"]=="running"]
    nontraining=[r["job"] for r in jobs if not identify(r)]
    caps=read(pool.POOL/"capacity.json")
    enough=(resources["quota_cores"]-resources["total_cpu_cores"]>=caps["spare_cpu"]+2
        and resources["available_ram_gib"]>=caps["spare_ram_gib"]+5.5
        and resources["free_gpu_mib"]>=caps["spare_gpu_mib"]+1500
        and resources["io_pressure_pct"]<=caps["max_io_pressure_pct"]
        and resources["scratch_free_gib"]>=caps["spare_scratch_gib"]+1.
        and resources["scratch_free_inodes"]>=caps["spare_inodes"])
    matched=speed.compare()
    # The previous production feeder accepted a third pair when no other physical request
    # was competing and matched four/six-flow rates supported it. In-flight pairs keep their leases.
    if matched:
        from statistics import median
        favorable=max(median(x["event_ratio"] for x in matched),median(x["physics_ratio"] for x in matched))>1.
    else:
        favorable=True
    return len(ours)>=2 and not nontraining and enough and favorable, nontraining,matched


def quiesce(pid):
    original=pid_record(pid)
    if not original or not any(Path(arg).name=="dispatch_learning.py" for arg in original["argv"]) or Path(f"/proc/{pid}/cwd").resolve()!=ROOT:
        raise ValueError("Process is not this study's active legacy dispatcher")
    old=[r for r in active_jobs() if identify(r)]
    os.kill(pid,signal.SIGSTOP)
    while (r:=pid_record(pid)) and r["state"] not in ("T","t"):
        time.sleep(.05)
    children=[]
    for child in sorted(previous.children(pid)):
        record=pid_record(child)
        if record and record["state"]!="Z":
            if not any(Path(a).name=="lease.py" for a in record["argv"]):
                os.kill(pid,signal.SIGCONT)
                raise ValueError(f"Unexpected active child of old dispatcher: {record}")
            children.append(record)
    old=[r for r in active_jobs() if identify(r)]
    snapshot=dict(study=campaign.STUDY,previous_feeder=original,new_feeder=pid_record(os.getpid()),
        quiesced_unix=time.time(),inherited=[{k:r.get(k) for k in ("job","registered","started","owner_pid","pid","slots","state")} for r in old],
        original_children=children,old_dispatch_status=read(previous.OUT.with_name("dispatch-live.json")),
        status="old admission quiesced; original leases continue independently",wall_deadline_enabled=False)
    atomic_json(HANDOFF,snapshot)
    return snapshot


def retire_if_drained(handoff,stack):
    if not handoff or handoff.get("manager_lock_transferred_unix"):return False
    for old in handoff["original_children"]:
        p=pid_record(old["pid"])
        if p and p["start_ticks"]==old["start_ticks"] and p["state"]!="Z":
            return False
    before=handoff["previous_feeder"]
    live=pid_record(before["pid"])
    if live and live["start_ticks"]==before["start_ticks"] and live["state"]!="Z":
        if live["state"] not in ("T","t"):raise RuntimeError("Legacy admission manager unexpectedly resumed")
        os.kill(before["pid"],signal.SIGTERM)
        os.kill(before["pid"],signal.SIGCONT)
    lock(RUN/"manager.lock",stack,wait=True)
    handoff.update(manager_lock_transferred_unix=time.time(),status="sole independent feeder owns original manager.lock",
        inherited_lease_outcomes=[{k:r.get(k) for k in ("job","state","started","ended","exit_code","slots")}
            for r in [read(pool.POOL/"jobs"/(o["job"]+".json")) for o in handoff["inherited"]]])
    atomic_json(HANDOFF,handoff)
    return True


def refresh_slots(active,saved,heads,target):
    # Logical positions express preference for an on-policy continuation. They never own flocks on physical slots.
    seats=[h for h in saved if not (heads[h]["frozen"] or heads[h]["terminal"])]
    for h in active:
        if h not in seats:
            seats.append(h)
    while len(seats)<target:
        candidates=[h for h in PRIMARY if h not in seats and not (heads[h]["frozen"] or heads[h]["terminal"])]
        if not candidates:
            candidates=[h for h in ABLATION if h not in seats and not (heads[h]["frozen"] or heads[h]["terminal"])]
        if not candidates:break
        seats.append(candidates[0])
    return seats


def curves_closed(heads):
    """Late overall lifecycle preserves existing Release consumers after per-head training freezes."""
    for head,h in heads.items():
        for v in NODES:
            if v>h["version"]:break
            final=SMALL/head/f"evaluate-{v:03d}.json"
            if not final.is_file():return False
            if read(final).get("status") not in ("evaluated","technical_or_starved"):return False
            delivery=SMALL/head/f"evaluate-{v:03d}-media.json"
            if not delivery.is_file() or read(delivery).get("status") not in ("verified","technical_no_renderable_trajectory"):
                return False
    return True


def run(old_pid):
    STATE.mkdir(parents=True,exist_ok=True)
    for name in ("admissions","logs"): (STATE/name).mkdir(exist_ok=True)
    with ExitStack() as stack:
        lock(STATE/"owner.lock",stack)
        manifest=read(RUN/"manifest.json")
        runtime=campaign.verified_runtime(manifest)
        for method,seed in ORDER:campaign.prepare(method,seed)
        if old_pid and not HANDOFF.exists():
            handoff=quiesce(old_pid)
        elif HANDOFF.exists():
            handoff=read(HANDOFF)
        else:
            handoff=None
        if not handoff or handoff.get("manager_lock_transferred_unix"):
            lock(RUN/"manager.lock",stack)
        own_lock=not handoff or bool(handoff.get("manager_lock_transferred_unix"))
        jobs=active_jobs()
        seated=read(SEATS)["heads"] if SEATS.exists() else [identify(r)[0] for r in sorted(jobs,key=lambda r:r["registered"]) if identify(r)]
        if not REGISTRY.is_file():
            atomic_json(REGISTRY,dict(study=campaign.STUDY,registered_unix=time.time(),scientific_runtime=runtime,
                dispatcher=str(SELF),dispatcher_sha256=file_digest(SELF),physical_slots=6,atomic_pair_slots=2,
                initial_logical_positions=seated,primary_order=PRIMARY,ablation_order=ABLATION,
                allocation="An incumbent retains priority until its own 8192 or hard terminal. Every batch relinquishes its real pair lease before re-queuing. Unassigned active primary heads receive vacant logical positions ahead of the E-Fisher ablations.",
                third_position="At most three; unchanged shared resource and four/six-flow admission, no reserved physics cores while waiting.",
                no_cross_head_version_barrier=True,primary_budget=73728,geometry_ablation_budget=24576,
                curve_nodes=SPEC["curve_nodes"],curve_ready_queue=str(CURVES),final_matrix_primary_and_ablation=96,
                deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False))
        children={}
        speed=previous.Speed()
        while True:
            if not own_lock and retire_if_drained(handoff,stack):own_lock=True
            jobs=active_jobs()
            active={}
            for r in jobs:
                item=identify(r)
                if item:
                    h,v=item
                    active[h]=r["job"]
                    note_start(r,h,v)
            for name,child in list(children.items()):
                rc=child.poll()
                if rc is None:
                    h=name[len(PREFIX):].rsplit("-train-v",1)[0]
                    active[h]=name
                else:
                    del children[name]
                    h,vs=name[len(PREFIX):].rsplit("-train-v",1)
                    prior=SMALL/h/f"train-{int(vs):03d}.json"
                    if rc and (not prior.exists() or read(prior)["status"]!="updated"):
                        atomic_json(SMALL/h/"terminal.json",dict(job=name,role="train",returncode=rc,independent=True))
            heads={h:history(h) for h in METHOD}
            for h,rec in heads.items():deliver_nodes(h,rec["version"])
            resources=pool.resources()
            if any(r["job"].startswith(PREFIX) and not identify(r) for r in jobs):
                speed.last=None
            else:
                speed.record(jobs,resources)
            third,other,matched=third_permitted(jobs,resources,speed)
            target=3 if third else 2
            seated=refresh_slots(active,seated,heads,target)
            if not SEATS.exists() or read(SEATS)["heads"]!=seated:
                atomic_json(SEATS,dict(heads=seated,at=time.time()))
            # An existing third logical head can wait for a physical pair when the resource pool shrinks.
            ready=sorted((h for h in seated if h not in active and not heads[h]["frozen"] and not heads[h]["terminal"]),
                         key=lambda h:(heads[h]["ready_at"],seated.index(h)))
            for h in ready:
                if len(active)>=target:break
                v=heads[h]["version"]
                name=head_job(h,v)
                prior=pool.POOL/"jobs"/(name+".json")
                if prior.is_file():
                    rec=read(prior)
                    if rec["state"]=="ended":
                        atomic_json(SMALL/h/"terminal.json",dict(job=name,role="train",returncode=rec.get("exit_code"),
                            reason="Original lease ended without a published update",source=str(prior)))
                        continue
                children[name]=start_pair(h,heads[h])
                active[h]=name
            atomic_json(LIVE,dict(study=campaign.STUDY,feeder_pid=os.getpid(),sampled_unix=time.time(),
                manager_lock_transferred=own_lock,logical_positions=seated,current_max_pairs=target,
                training_admissions=active,heads=heads,primary_used=sum(heads[h]["used"] for h in PRIMARY),
                ablation_used=sum(heads[h]["used"] for h in ABLATION),other_physical_demand=other,
                resources=resources,matched_throughput=matched,curve_ready_directory=str(CURVES)))
            if all(h["frozen"] or h["terminal"] for h in heads.values()) and not active:
                end=SMALL/"training-lifecycle.json"
                if not end.is_file():
                    atomic_json(end,dict(status="independent_heads_naturally_closed",heads=heads,time=time.time()))
                if curves_closed(heads):
                    atomic_json(SMALL/"lifecycle.json",dict(status="naturally_ended",training=str(end),time=time.time()))
                    return
            time.sleep(3)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--takeover",type=int)
    run(p.parse_args().takeover)
