"""Dispatch the unchanged W61 frozen scenarios as soon as each 8192-event head closes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import evaluate_learning as native
import evaluation_physics as detached
from campaign import SMALL, RUN, head_name
from develop import admitted_call
from hybrid import ROOT, SPEC
from publish_learning import verified, DELIVERY
from s1_w59_learner_critic import atomic_json, file_digest
from dispatch_learning import pool as lease

LEDGER = native.LEDGER
WORK = RUN / "parallel-final"
PREFIX = "w61-e-final-"
BUDGETS = {"E30": 2.5, "E60": 4.5, "short": 12., "recovered": 20.}
HEADS = [(m,s) for s in SPEC["seeds"] for m in [*SPEC["methods"],SPEC["geometry_ablation"]]]
BASELINE = ("PPO",6001,0)
SH = ZoneInfo("Asia/Shanghai")


def now():
    return datetime.now(SH).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Episode:
    method: str
    seed: int
    version: int
    case: str
    kind: str
    condition: str

    @property
    def head(self):
        return head_name(self.method,self.seed)

    @property
    def name(self):
        return native.ident(self.method,self.seed,self.version,self.case,self.kind,self.condition)

    def path(self, suffix):
        return LEDGER/self.head/(self.name+suffix)

    def summary(self):
        return self.path("-summary.json")

    def physical_record(self):
        physical=self.path("-physical.json")
        return physical if physical.exists() else self.summary()

    def predecessor(self):
        return Episode(self.method,self.seed,self.version,self.case,"E30","nominal")

    def args(self):
        return (self.method,str(self.seed),str(self.version),self.case,self.kind,self.condition)


def programme():
    result=[]
    for method,seed,version in [BASELINE,*[(m,s,32) for m,s in HEADS]]:
        cases=SPEC["scenarios"]
        for case in cases:
            result.append(Episode(method,seed,version,case,"E30","nominal"))
        if version:
            # Alternate window lengths across incoming heads. Every combination keeps
            # its original identifier, specimen, duration and disturbance.
            for condition in native.PLAN["conditions"]:
                for i,case in enumerate(cases):
                    for kind in (("short","recovered") if i==0 else ("recovered","short")):
                        result.append(Episode(method,seed,version,case,kind,condition["id"]))
        for case in cases:
            result.append(Episode(method,seed,version,case,"E60","nominal"))
    return result


def registration(episodes):
    return dict(study=native.PLAN["id"],source_evaluator_sha256=file_digest(native.SELF),
        frozen_contract_sha256=file_digest(native.CONFIG),backend_sha256=file_digest(native.BACKEND),
        long_count=sum(e.kind in ("short","recovered") for e in episodes),primary=72,geometry_ablation=24,
        maximum_shared_physical_slots=6,training_pair_slots=2,frozen_episode_slots=1,
        trigger="own train-031 updated receipt at 8192 plus the matching immutable policy-032 SHA256",
        publication="verified after the physical lease is released",deadline=None,wall_deadline_enabled=False,
        kill_on_conversation_end=False,episodes=[dict(id=e.name,method=e.method,seed=e.seed,version=e.version,
        case=e.case,stage=e.kind,condition=e.condition,predecessor=e.predecessor().name if e.kind=="E60" else None) for e in episodes])


def ready_heads(cache):
    for method,seed,version in [BASELINE,*[(m,s,32) for m,s in HEADS]]:
        key=(method,seed,version)
        if key in cache:continue
        h=head_name(method,seed)
        src=SMALL/h/("train-031.json" if version else "initial.json")
        if not src.exists():continue
        row=native.read(src)
        if version and (row.get("status")!="updated" or row.get("used")!=8192):continue
        path=RUN/h/f"policy-{version:03d}.pt"
        expected=row["updated_checkpoint_sha256" if version else "sha256"]
        if (row["method"],row["seed"],row["version"])!=(method,seed,version-1 if version else 0) or file_digest(path)!=expected:
            raise ValueError(f"Frozen actor differs from the training receipt: {h} v{version}")
        dst=LEDGER/h/f"parallel-freeze-{version:03d}.json"
        if dst.exists():
            frozen=native.read(dst)
            if frozen["sha256"]!=expected:raise ValueError(f"Frozen receipt changed: {h}")
        else:
            frozen=dict(head=h,version=version,used=row["used"],checkpoint=str(path),sha256=expected,
                        training_receipt=str(src),training_receipt_unix=src.stat().st_mtime,registered=now())
            atomic_json(dst,frozen)
        cache[key]=frozen


def process(pid):
    try:
        st=Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()
    except FileNotFoundError:
        return None
    return None if st[0]=="Z" else (int(st[19]),st[0])


def is_serial(pid,ticks):
    state=process(pid) if pid else None
    if not state or state[0]!=ticks:return False
    proc=Path(f"/proc/{pid}")
    try:
        argv=proc.joinpath("cmdline").read_bytes().split(b"\0")
        return (proc.joinpath("cwd").resolve()==ROOT and argv[-2:]==[b"watch",b""] and
                any(x.rsplit(b"/",1)[-1]==b"evaluate_learning.py" for x in argv))
    except FileNotFoundError:
        return False


def transfer_legacy(pid,ticks,zero_settled,stack,ownership):
    if ownership.get("acquired"):return True
    if is_serial(pid,ticks):
        # The old owner finishes the complete reference cohort, including upload.
        # Its idle sleep between scans has no physical worker or publisher child.
        if not zero_settled:return False
        proc=Path(f"/proc/{pid}")
        with open(proc/"wchan") as channel: sleeping="nanosleep" in channel.read()
        if sleeping and not any(f.read_text().strip() for f in (proc/"task").glob("*/children")):
            # This D57 Python exposes POSIX kill but no pidfd wrappers. The four
            # reference receipts are closed and the owner has no worker or upload
            # child. Recheck its recorded start ticks, argv and cwd before signalling.
            if is_serial(pid,ticks):
                try:
                    os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
        return False
    handle=(RUN/"final-evaluator.lock").open("a")
    try:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return False
    stack.enter_context(handle)
    ownership.update(acquired=True,finished=now())
    atomic_json(WORK/"ownership.json",ownership)
    return True


def physical(ep):
    cmd=[sys.executable,"-u",str(Path(detached.__file__).resolve()),"native",*ep.args()]
    return admitted_call(ep.name,1,cmd,ram=5.,gpu=800,disk=BUDGETS[ep.kind])


def compact_commit(paths, message):
    names=sorted({str(p.relative_to(ROOT)) for p in paths if p.exists()})
    git=["git","-C",str(ROOT)]
    if not names:return
    subprocess.run([*git,"add","--sparse","--",*names],check=True)
    if subprocess.run([*git,"diff","--cached","--quiet","--",*names]).returncode:
        subprocess.run([*git,"commit","--only","-qm",message,"--",*names],check=True)


def record_throughput(previous, active, episodes, hardware):
    at=time.time()
    if previous and at-previous["time"]<25:return previous
    clocks={}
    for job in active:
        if job["state"]!="running" or not job["job"].startswith(PREFIX):continue
        p=native.DATA/job["job"]/"live.json"
        if p.exists():
            obj=native.read(p)
            clocks[job["job"]]=obj["physical_time_s"]
    if previous and clocks and clocks.keys()==previous["clocks"].keys():
        elapsed=at-previous["time"]
        simulated=sum(v-previous["clocks"][name] for name,v in clocks.items())
        row=dict(shanghai=now(),seconds=elapsed,episodes=list(clocks),width=len(clocks),
                 endpoint_long=sum(episodes[n].version==32 and episodes[n].kind in ("short","recovered") for n in clocks),
                 physical_s=simulated,physical_s_per_wall_s=simulated/elapsed,resources=hardware)
        with (WORK/"throughput.jsonl").open("a") as f:f.write(json.dumps(row,separators=(",",":"))+"\n")
    return dict(time=at,clocks=clocks)


def run(legacy):
    WORK.mkdir(parents=True,exist_ok=True)
    episodes=programme()
    names={e.name:e for e in episodes}
    plan=LEDGER/"parallel-plan.json"
    config=registration(episodes)
    if len(names)!=len(episodes) or config["long_count"]!=96:raise ValueError("Frozen evaluation matrix changed")
    if plan.exists() and native.read(plan)!=config:raise ValueError("Registered frozen evaluation identities changed")
    atomic_json(plan,config)
    record=WORK/"ownership.json"
    ownership=native.read(record) if record.exists() else {}
    if legacy:
        st=process(legacy)
        if ownership and ownership.get("pid")!=legacy:raise ValueError("Another legacy handoff is already registered")
        if not ownership:
            if not st or not is_serial(legacy,st[0]):raise ValueError("Specified PID is not the existing serial evaluation watcher")
            ownership=dict(pid=legacy,start_ticks=st[0],registered=now(),
                policy="leave every existing worker, reference run and upload alive; retire only the old idle watcher")
            atomic_json(record,ownership)
    pid,ticks=ownership.get("pid",0),ownership.get("start_ticks",0)
    ownership.pop("acquired",None)  # Every process lifetime must acquire the kernel lock again.
    heads={}; physics={}; uploads={}; retry={}; previous=None; signature=None
    summary_path=LEDGER/"parallel-progress.json"
    with ExitStack() as stack:
        service=stack.enter_context((WORK/"dispatcher.lock").open("a"))
        fcntl.flock(service,fcntl.LOCK_EX|fcntl.LOCK_NB)
        workers=stack.enter_context(ThreadPoolExecutor(max_workers=6))
        sender=stack.enter_context(ThreadPoolExecutor(max_workers=1))
        while True:
            ready_heads(heads)
            for ep,future in list(physics.items()):
                if not future.done():continue
                rc=future.result();del physics[ep]
                if rc or not ep.physical_record().exists():
                    atomic_json(ep.path("-technical.json"),dict(attempt=ep.name,exit_code=rc,status="technical_interruption",
                        reason="native evaluator did not close its physics receipt; originals retained",used=0))
            for ep,future in list(uploads.items()):
                if not future.done():continue
                del uploads[ep]
                try:
                    future.result();retry.pop(ep,None)
                except (RuntimeError,subprocess.SubprocessError,OSError) as err:
                    retry[ep]=time.time()+120
                    atomic_json(WORK/(ep.name+"-delivery-pending.json"),dict(attempt=ep.name,at=now(),error=str(err)))
            jobs=[native.read(p) for p in (lease.POOL/"jobs").glob("*.json")]
            active=[r for r in jobs if r["state"] in ("waiting","running")]
            by_name={r["job"]:r for r in jobs}
            hardware=lease.resources()
            previous=record_throughput(previous,active,names,hardware)
            states={};candidates=[];send=[];counts=Counter()
            for ep in episodes:
                key=(ep.method,ep.seed,ep.version)
                if key not in heads:
                    states[ep]="awaiting_8192";continue
                summary=ep.physical_record()
                if summary.exists():
                    row=native.read(summary)
                    if row["actor_sha256"]!=heads[key]["sha256"] or row["case"]!=ep.case or row["stage"]!=ep.kind or row["condition"]!=ep.condition:
                        raise ValueError("Frozen episode receipt mismatch: "+ep.name)
                    if ep.summary().exists():
                        states[ep]="verified" if verified(ep.name) else "publication_pending"
                    elif row.get("physical_samples")==0:
                        states[ep]="technical_interruption"
                    else:
                        states[ep]="physical_closed_render_pending"
                    if states[ep]=="publication_pending":send.append(ep)
                    counts[key]+=1;continue
                if ep.path("-technical.json").exists():
                    states[ep]="technical_interruption";continue
                if ep.kind=="E60":
                    pred=ep.predecessor()
                    if pred.path("-technical.json").exists() and not pred.physical_record().exists():
                        states[ep]="E30_did_not_qualify";continue
                    if not pred.physical_record().exists():
                        states[ep]="waiting_for_E30";continue
                    prior=native.read(pred.physical_record())
                    if (prior["actor_sha256"],prior["case"],prior["head"])!=(heads[key]["sha256"],ep.case,ep.head):
                        raise ValueError("E30 exact head or case mismatch")
                    if prior["stage_status"]!="passed" or not prior["qualification_usable"]:
                        states[ep]="E30_did_not_qualify";continue
                rec=by_name.get(ep.name)
                if rec and rec["state"] in ("waiting","running"):
                    if process(rec["owner_pid"]) or (rec.get("pid") and process(rec["pid"])):
                        states[ep]=rec["state"];counts[key]+=1;continue
                    if rec["state"]=="running":
                        states[ep]="orphaned_lease_needs_reconciliation";continue
                if ep in physics:
                    states[ep]="submitted";counts[key]+=1;continue
                if rec and rec["state"]=="ended":
                    if ep.path(".json").exists() and (native.DATA/ep.name/"trajectory.npz").exists():
                        detached.native_closure(ep.method,ep.seed,ep.version,ep.case,ep.kind,ep.condition)
                        states[ep]="physical_closed_render_pending";continue
                    atomic_json(ep.path("-technical.json"),dict(attempt=ep.name,exit_code=rec["exit_code"],
                        status="technical_interruption",used=0,reason="closed original lease without a native physics result"))
                    states[ep]="technical_interruption";continue
                states[ep]="ready";candidates.append(ep)

            ref_settled=all(states[e] in ("verified","E30_did_not_qualify","technical_interruption") or
                (e.kind=="E60" and states.get(e.predecessor())=="technical_interruption") for e in episodes if not e.version)
            owns=transfer_legacy(pid,ticks,ref_settled,stack,ownership)
            # The legacy watcher retains publication for its live reference cohort.
            # The shared lease and stable job name join an already-running physical case.
            if owns and not uploads:
                next_upload=next((ep for ep in send if retry.get(ep,0)<=time.time()),None)
                if next_upload:
                    uploads[next_upload]=sender.submit(native.deliver,next_upload.name,next_upload.method,next_upload.seed)
            reserved=sum(len(r["slots"]) for r in active if r["state"]=="running")
            others=sum(r["slots_requested"] for r in active if r["state"]=="waiting" and not r["job"].startswith(PREFIX))
            queued=sum(r["state"]=="waiting" and r["job"].startswith(PREFIX) for r in active)
            in_transit=sum(e.name not in by_name for e in physics)
            capacity=min(6,native.read(lease.POOL/"capacity.json")["physical_slots"])
            # One standby at a full pool establishes a fair place for the first new
            # checkpoint; additional admissions consume only actual spare slots.
            offers=max(0,max(1,capacity-reserved-others)-queued-in_transit)
            for _ in range(min(offers,capacity-len(physics),len(candidates))):
                ep=min(candidates,key=lambda e:(counts[e.method,e.seed,e.version],episodes.index(e)))
                candidates.remove(ep);counts[ep.method,ep.seed,ep.version]+=1
                physics[ep]=workers.submit(physical,ep);states[ep]="submitted"
            long=Counter(states[e] for e in episodes if e.kind in ("short","recovered"))
            all_counts=Counter(states.values())
            end=all_counts["verified"]+all_counts["E30_did_not_qualify"]+all_counts["technical_interruption"]==len(episodes)
            snapshot=dict(shanghai=now(),pid=os.getpid(),registered_long=96,primary_long=72,ablation_long=24,
                long=long,all_cases=all_counts,checkpoint_heads_8192=sum(k[2]==32 for k in heads),
                legacy_pid=pid,legacy_active=is_serial(pid,ticks),single_dispatch_owner=owns,capacity=capacity,
                active_pool={r["job"]:r["slots"] for r in active if r["state"]=="running"},
                other_waiting_slots=others,resources=hardware,evaluations={e.name:s for e,s in states.items() if s!="awaiting_8192"},
                completed=end,deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
            atomic_json(WORK/"live.json",snapshot)
            changed=(len(heads),all_counts["verified"],all_counts["publication_pending"]+all_counts["verified"],
                     all_counts["technical_interruption"],all_counts["physical_closed_render_pending"],owns,end)
            if changed!=signature:
                atomic_json(summary_path,snapshot)
                files=[plan,summary_path,*LEDGER.glob("*/parallel-freeze-*.json"),
                       *LEDGER.glob("*/*-physical.json"),*LEDGER.glob("*/*-summary.json"),*LEDGER.glob("*/*-technical.json")]
                files.extend(DELIVERY/(e.name+".json") for e in episodes if states[e]=="verified")
                compact_commit(files,"Record W61 concurrent frozen evaluations and verified episode identities")
                signature=changed
                print(snapshot["shanghai"],"8192 heads",snapshot["checkpoint_heads_8192"],"long counts",dict(long),
                      "reference",[(e.name,states[e]) for e in episodes if e.version==0],"legacy",snapshot["legacy_active"],flush=True)
            if end:return
            time.sleep(8)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-pid",type=int)
    run(parser.parse_args().legacy_pid)
