"""Restore the saved 5902 event barrier and resume its frozen W59 learning/qualification chain."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from threading import Event, Thread
from zoneinfo import ZoneInfo

from s1_w59_learner_critic import ROOT, RELEASE, atomic_json, file_digest, json_rows
from s1_w59_learning import read_json, append

DATA=ROOT/RELEASE
WORK=ROOT/"work/w59"
RAW=ROOT/"outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12"
SEED=5902
SUFFIX="PPO-full-fallback"
RUN=WORK/"learning"/"seed-5902"/SUFFIX
TAG="w59-learning-5902-ppo-full-fallback-08192"
POOL=WORK/"pool"
RUNTIME=Path((WORK/"runtime-research.txt").read_text().strip())
EXPECTED="4ad3a5f28b99ad9ccc84455915fdffa9dfcb57af"
MARKER=WORK/"archive-process-20261127/RESUME_AUTHORIZED.json"
RECEIPT=DATA/"learning/recovery-20261127.json"
LIVE=DATA/"learning/resume-20261127.json"
BG=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
TOKEN="20261127/w59/5902/enospc-v19"
SH=ZoneInfo("Asia/Shanghai")


def timestamp():
    return datetime.now(SH).isoformat(timespec="seconds")


@contextmanager
def exclusive(p):
    with p.open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def raw_name(case):
    return f"w59-r3-5902-ppo-full-fallback-{case.lower()}-e60-0020"


def original_run():
    rev=subprocess.check_output(["git","-C",str(RUNTIME),"rev-parse","HEAD"],text=True).strip()
    if rev!=EXPECTED:raise ValueError(f"Frozen execution commit changed: {rev}")
    return rev


def validate_events():
    import torch
    from s1_w59_learning import RequestActor, TrueTaskCritic
    from s1_w59_learner_critic import file_digest
    original_run()
    pub=read_json(RUN/"published.json")
    if (pub["version"],pub["used_events"],pub["mode"])!=(19,4864,"acquire"):
        raise ValueError(f"Not the interrupted v19 state: {pub}")
    weight=Path(pub["checkpoint"])
    sha=file_digest(weight)
    ck=torch.load(weight,map_location="cpu",weights_only=True)
    if (ck["version"],ck["used_events"],ck["seed"],ck["arm"],ck["verified_first_update"])!=(19,4864,5902,"PPO-full",True):
        raise ValueError("Frozen actor/critic do not belong to the interrupted version")
    actor=RequestActor();actor.load_state_dict(ck["actor"]);actor.eval()
    critic=TrueTaskCritic();critic.load_state_dict(ck["critic"]);critic.eval()
    sealed=json_rows(RUN/"used-event-ids.jsonl")
    if [v["version"] for v in sealed]!=list(range(19)) or any(len(v["ids"])!=256 for v in sealed):
        raise ValueError("Previously optimized versions are not complete 2x128 batches")
    used={key for block in sealed for key in block["ids"]}
    if len(used)!=4864:raise ValueError("Duplicate previously optimized event")
    all_new=set();by_case={}
    for case in "AB":
        path=RUN/f"events-019-{case}.jsonl"
        rows=json_rows(path);name=raw_name(case)
        if len(rows)!=59 or (RUN/f"ready-019-{case}.json").exists():
            raise ValueError(f"Unexpected open batch size or barrier publication in {case}")
        ids=set()
        for i,row in enumerate(rows):
            if ((row["case"],row["position"],row["serial"],row["version"],row["seed"],row["arm"],row["sampling"],
                row["versions"]["actor_sha256"])!=(case,i,19*128+i,19,5902,"PPO-full","on_policy",sha)
                or row["dt_s"]<=0 or abs(row["end_s"]-row["start_s"]-row["dt_s"])>1e-8):
                raise ValueError(f"Unusable open event {row['event_id']}")
            ids.add(row["event_id"])
        if len(ids)!=59 or ids & (used|all_new):raise ValueError("Duplicate IDs in partial/saved batches")
        all_new.update(ids)
        source_counts={}
        for attempt in dict.fromkeys(r["attempt"] for r in rows):
            expected=[r for r in rows if r["attempt"]==attempt]
            original=Path(expected[0]["source_dir"])/"w59-actor-events.jsonl"
            matching=[r for r in json_rows(original) if r["version"]==19 and r["sampling"]=="on_policy"]
            if matching!=expected:raise ValueError(f"Completed physical successor differs from open ledger: {original}")
            source_counts[attempt]=len(expected)
        tensor=lambda key:torch.as_tensor([r[key] for r in rows],dtype=torch.float32)
        with torch.inference_mode():
            mean=actor(tensor("state"),tensor("base"))
            lp=actor.logp(mean,tensor("raw"))
            dv=critic(tensor("critic_state"))-tensor("value")
            nxt=critic(tensor("next_critic_state"))
            nv=torch.where(torch.tensor([r["terminated"] for r in rows]),0.,nxt)
            residuals=dict(gaussian_mean=float((mean-tensor("mean_raw")).abs().max()),
                logp=float((lp-tensor("logp_raw")).abs().max()),critic=float(dv.abs().max()),
                next_critic=float((nv-tensor("next_value")).abs().max()))
        if max(residuals.values())>1e-4:raise ValueError(f"Behavior or successor mismatch {residuals}")
        req_file=RAW/name/"w55-f/handover-requests.jsonl"
        req=json_rows(req_file)
        end=req[-1]
        if (end["stage"],end["end_reason"])!=("censored","execution_error"):
            raise ValueError("Original control terminal lacks confirmed technical interruption")
        nrequests=sum(r["stage"]=="request" for r in req)
        if nrequests!=source_counts[name]+1 or end["decision_id"]!=nrequests:
            raise ValueError("Could not identify the unsaved pending request")
        t=float(end["physical_end_s"])
        rt=read_json(DATA/"attempts"/(name+".json"))
        for pid in (rt["pid"],rt["runner_pid"]):
            if Path(f"/proc/{pid}").exists():
                raise RuntimeError(f"Old physics PID exists; verify exact ownership before any correction ({pid})")
        outlog=RUN/(name+".log")
        if "No space left on device" not in outlog.read_text():raise ValueError("Missing original ENOSPC traceback")
        by_case[case]=dict(attempt=name,on_policy_rows=len(rows),physical_attempt_rows=source_counts[name],
            source_rows=source_counts,open_event_file=str(path),ledger_sha256=file_digest(path),
            end_physical_s=t,physical_substeps=round(t/.0002),requested=nrequests,
            open_pending_excluded=1,last_closed_successor_s=rows[-1]["end_s"],
            last_closed_id=rows[-1]["event_id"],censored_request=nrequests,first_unused_serial=19*128+len(rows),
            remaining_to_barrier=128-len(rows),log_sha256=file_digest(outlog),censor_log_sha256=file_digest(req_file),
            last_traceback_unix_s=outlog.stat().st_mtime,previous_status=rt["status"],
            exited_run_pid=rt["runner_pid"],exited_physics_pid=rt["pid"],max_behaviour_replay_error=residuals)
    return dict(event_id=TOKEN,prepared_at=timestamp(),runtime=str(RUNTIME),source_commit=EXPECTED,
        policy_checkpoint=str(weight),checkpoint_sha256=sha,seed=SEED,version=19,used_distinct_before=4864,
        validated_open_distinct=len(all_new),new_optimization_steps=0,cases=by_case,
        gae_cut="At the new virgin A/B attempt ID: bootstrap from each last saved successor; no backward carry across the technical episode boundary.",
        optimizer_state_preserved=True,**BG)


def register_cut(info):
    idx=DATA/"episodes/index.json"
    with exclusive(idx.with_name("index.lock")):
        original=read_json(idx)
        for case,proof in info["cases"].items():
            name=proof["attempt"];t=proof["end_physical_s"]
            item=next(r for r in original["attempts"] if r["attempt"]==name)
            if item.get("stage_status")!="technical_interruption" or "No space left on device" not in item.get("error",""):
                raise ValueError(f"Missing interrupted official episode index evidence for {name}")
            path=DATA/"attempts"/(name+".json");attempt=read_json(path)
            final=DATA/"episodes"/(name+".json")
            if final.exists() and read_json(final).get("recovery_event_id")!=TOKEN:
                raise ValueError(f"Refusing to replace an unrelated final result: {final}")
            notes=dict(recovery_event_id=TOKEN,simulation_s=t,physical_samples=proof["physical_substeps"],
                reason="storage_enospc",stage_status="technical_interruption",success=None,
                physical_started_unix_s=item["physical_started_unix_s"],qualification_usable=False)
            # This lost runner never produced a normal terminal result. Record only the fields
            # witnessed in its original trace, including completed successors and the censored action.
            stats=dict(requested=proof["requested"],completed=proof["physical_attempt_rows"],
                used_barrier_rows=proof["physical_attempt_rows"],tail_excluded=0,data_cut=1,
                evaluation_only=0,zero_duration=0,task_absorbed=False,technical=True)
            atomic_json(final,dict(attempt=name,case_key=item["case_key"],evaluation_stage=item["evaluation_stage"],
                source_commit=EXPECTED,execution_fingerprint=item["execution_fingerprint"],error=item["error"],
                w59_learning=stats,status="interrupted",reconstructed_from=["episode index","handover censor","closed actor events"],
                **notes))
            attempt.update(notes,status="technical_interruption",exit_code=1,error=item["error"],
                ended_unix_s=proof["last_traceback_unix_s"],end_wall_source="stored launcher traceback mtime")
            atomic_json(path,attempt)
            item.update(simulation_s=t,physical_samples=proof["physical_substeps"],
                        reason=notes["reason"],recovery_event_id=TOKEN)
            ends=RUN/"physics-ends.jsonl"
            old=[r for r in json_rows(ends) if r["attempt"]==name]
            if not any(r.get("recovery_event_id")==TOKEN for r in old):
                append(ends,dict(attempt=name,case=case,exit_code=1,status="technical_interruption",
                    physical_s=t,reason="storage_enospc",unix_s=proof["last_traceback_unix_s"],recovery_event_id=TOKEN,
                    superseded_provisional_status=old[-1]["status"] if old else None))
        atomic_json(idx,original)
    job=POOL/"jobs"/(TAG+".json")
    with exclusive(job.with_suffix(".lock")):
        prior=read_json(job)
        if prior.get("recovery_event_id")!=TOKEN:
            if prior["state"]!="running" or any(Path(f"/proc/{prior[k]}").exists() for k in ("owner_pid","pid")):
                raise ValueError("Prior two-slot campaign is not the known lost lease")
            info["previous_lease"]={k:prior[k] for k in ("job","state","owner_pid","pid","start_ticks","slots","registered_at")}
            prior.update(state="ended",exit_code=1,recovery_event_id=TOKEN,ended_at=timestamp(),
                end_wall_source="retrospective closure after PID/ENOSPC reconciliation",reason="storage_enospc")
            atomic_json(job,prior)
    atomic_json(RECEIPT,info)


def prepare():
    if RECEIPT.exists():
        info=read_json(RECEIPT)
        if info.get("event_id")==TOKEN and file_digest(Path(info["policy_checkpoint"]))==info["checkpoint_sha256"]:
            return info
        raise ValueError("Conflicting W59 recovery receipt")
    import torch
    torch.set_num_threads(1)
    with exclusive(RUN/"learner.lock"):
        info=validate_events()
        register_cut(info)
    return info


def stage(state,**facts):
    if LIVE.is_file():
        old=read_json(LIVE)
        if (old.get("state"),old.get("facts"),old.get("pid"))==(state,facts,os.getpid()):return
    atomic_json(LIVE,dict(state=state,pid=os.getpid(),shanghai=timestamp(),facts=facts,**BG))
    print(timestamp(),state,json.dumps(facts,ensure_ascii=False),flush=True)


def metrics_source():
    """Supply fresh cgroup resource samples for the shared lease after the lost monitor."""
    from observe_s1_w59 import sample
    before,_=sample()
    time.sleep(1.)
    now,_=sample()
    now["total_cpu_cores"]=(now["cgroup_cpu_seconds"]-before["cgroup_cpu_seconds"])/(now["unix_s"]-before["unix_s"])
    atomic_json(WORK/"last-resource.json",now)


def recovery_series(info):
    from s1_w59_learner_series import Series
    args=SimpleNamespace(runtime=RUNTIME,admission=DATA/"learning/final-action-gate.json",
             first_states=WORK/"learner/first-states",critic=WORK/"learner/historical-task-critic.pt")
    series=Series(args)
    if series.role!="fallback-ppo":raise ValueError("Unexpected algorithm identity")
    original_run()
    seed1=series.frozen(5901,"PPO-full",4096)
    if not seed1 or read_json(series.run_path(5903,"PPO-full")/"published.json")["mode"]!="rejected":
        raise ValueError("Previously selected scientific branches changed")
    # Adopt the 8192 node only; prior nodes are immutable, and the stopped 5901/5903
    # training paths must not be dispatched again. New collector attempts start A21/B21.
    pub=read_json(RUN/"published.json")
    old=read_json(POOL/"jobs"/(TAG+".json"))
    if (pub["version"]<19 or pub["used_events"]<4864 or not Path(old["runtime"]).samefile(RUNTIME)):
        raise ValueError("Current run no longer extends verified v19")
    node_file=series.compact(SEED,"PPO-full")/"node-08192.json"
    def continuation():
        if node_file.is_file():return 0
        if pub["mode"] in ("rejected","incomplete"):return 1
        return series.leases(TAG+"-resume-enospc-20261127",2,old["command"])
    stage("resumed",runtime=EXPECTED,actor_sha256=info["checkpoint_sha256"],partial_source_events=118,
          prior_scientific_stops={"5901":"WBC after 5888; evaluate own closed 4096","5903":"first parameter change rejected; no eligible node"})
    # Only original qualifications may overlap the remaining 5902 learning. Heldout data
    # stay closed until all final per-seed checkpoints and budgets have been fixed.
    with ThreadPoolExecutor(max_workers=3) as pool:
        train=pool.submit(continuation)
        q={case:pool.submit(series.qualify,(seed1,case)) for case in "AB"}
        code=train.result()
        for future in q.values():future.result()
    state=read_json(RUN/"published.json")
    latest=next((n for n in (8192,4096,2048) if (series.compact(SEED,"PPO-full")/f"node-{n:05d}.json").is_file()
        and (RUN/f"freeze-{n:05d}.json").is_file()),None)
    if latest is None:raise RuntimeError("5902 has no eligible frozen acquisition node")
    if latest==8192 and not code:
        series.node_decision(SEED,8192,read_json(node_file))
    elif state["mode"] not in ("incomplete","rejected"):
        raise RuntimeError(f"Continuation stopped without a terminal learning record: exit={code}, state={state}")
    selected={5901:4096,5902:latest,5903:None}
    series.budgets=selected
    for s in selected:
        for n in (2048,4096,8192):
            src=series.compact(s,"PPO-full")/f"node-{n:05d}.json"
            if src.exists():series.denominators[f"{s}/PPO-full/{n}"]=read_json(src)
    selection=[series.frozen(s,"PPO-full",n) for s,n in selected.items() if n]
    if any(x is None for x in selection):raise RuntimeError("Incomplete frozen head registration")
    freeze=series.dest/"pre-heldout-frozen.json"
    if freeze.exists() and read_json(freeze)["selections"]!=selection:
        raise ValueError("Heldout selector was previously frozen with different checkpoints")
    atomic_json(freeze,dict(role=series.role,display_seed=5901,N_s=selected,selections=selection,
        heldout_protocol_sha256=file_digest(ROOT/"experiments/terrain_memory/s1-w59-footprints.json")))
    stage("frozen_evaluation",selected=selected,continuation_exit=code)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[(r,pool.submit(series.eval_head,r)) for r in selection]
        for r,future in futures:series.evaluations[f"{r['seed']}/PPO-full"]=future.result()
    series.finish("Frozen original qualification, eligible independent mechanism and conditional long envelopes ended")
    stage("scientific_chain_ended",selected=selected,outcome=str(series.dest/"learning-series.json"))


def watch():
    with exclusive(WORK/"resume-enospc-20261127.lock"):
        info=prepare()
        stage("waiting_for_verified_retirement",marker=str(MARKER),version=19,used=4864,open_by_case={"A":59,"B":59})
        while not MARKER.is_file():time.sleep(5)
        auth=read_json(MARKER)
        if (auth.get("authorized") is not True or auth.get("phase")!="H_verified_D57_retired" or not auth.get("cutoff")
            or not re.fullmatch(r"[a-fA-F0-9]{64}",str(auth.get("source_manifest_sha256","")))):
            raise ValueError("Retirement authorization receipt is incomplete")
        stage("authorized",marker_sha256=file_digest(MARKER),cutoff=auth["cutoff"])
        # run_s1_go2_r12 imports the approved MuJoCo MPC Python binding from the frozen environment.
        os.environ["PYTHONPATH"]=os.pathsep.join([str(RUNTIME/"src"),str(RUNTIME/"work/upstreams/mujoco_mpc/python")])
        metrics_source()
        stopped=Event()
        def refresh_resources():
            sample=WORK/"last-resource.json"
            while not stopped.is_set():
                if not sample.exists() or time.time()-read_json(sample)["unix_s"]>=20:
                    metrics_source()
                stopped.wait(10)
        monitor=Thread(target=refresh_resources,name="W59 lease resource samples",daemon=True)
        monitor.start()
        try:recovery_series(info)
        finally:
            stopped.set()
            monitor.join()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("prepare","watch"))
    a=p.parse_args()
    if a.mode=="prepare":
        print(json.dumps(prepare(),ensure_ascii=False))
    else:
        try:watch()
        except BlockingIOError:
            raise SystemExit("An existing W59 recovery owner holds the lock")
        except Exception as e:
            stage("blocked",reason=repr(e),receipt=str(RECEIPT))
            raise

