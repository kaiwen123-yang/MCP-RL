"""Schedule the existing E matrix at natural batch boundaries, sharing the six physical leases."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"learning"))
import campaign
from develop import LEASE, admitted_call
from hybrid import CACHE, ROOT
from s1_w59_learner_critic import atomic_json, file_digest

spec=importlib.util.spec_from_file_location("w61_common_pool",LEASE)
pool=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)
RECORD=campaign.RUN/"concurrency.jsonl"
OUT=campaign.SMALL/"dispatch-v2.json"
PREFIX="w61-"+campaign.STUDY+"-"


def inventory():
    records=[]
    for file in (pool.POOL/"jobs").glob("*.json"):
        job=json.loads(file.read_text())
        if job["state"] in ("waiting","running"):
            records.append(job)
    return records


def snapshots(jobs):
    active=[j for j in jobs if j["job"].startswith(PREFIX) and j["state"]=="running"]
    streams={}
    for job in active:
        name=job["job"].removeprefix(PREFIX)
        head,role,vers=name.rsplit("-",2)
        version=int(vers.removeprefix("v"))
        for case in ("A","B"):
            src=campaign.RUN/head/f"{role}-{version:03d}-{case}"
            for file in sorted(src.glob("*/live.json")):
                final=file.with_name("result.json")
                row=json.loads((final if final.is_file() else file).read_text())
                streams[file.parent.name]=[row["observed"],row["physical_s"],row.get("phase",row.get("terminal_phase"))]
    return dict(at=time.time(),slots=sum(len(j["slots"]) for j in active),jobs=[j["job"] for j in active],streams=streams)


class Speed:
    def __init__(self):
        self.last=None
        self.windows=[]
        if RECORD.exists():
            self.windows=[json.loads(l) for l in RECORD.read_text().splitlines() if l.strip()][-180:]

    def record(self,jobs,resources):
        snap=snapshots(jobs)
        if self.last and snap["at"]-self.last["at"]<28.:
            return
        last=self.last
        self.last=snap
        if last is None or set(snap["streams"])!=set(last["streams"]) or not snap["streams"] or snap["jobs"]!=last["jobs"]:
            return
        interval=snap["at"]-last["at"]
        phases=Counter(s[2] for s in snap["streams"].values())
        phase,count=phases.most_common(1)[0]
        row=dict(unix=snap["at"],seconds=interval,E_physical_slots=snap["slots"],phase_majority=phase,
            majority_share=count/len(snap["streams"]),events_s=sum(s[0]-last["streams"][name][0] for name,s in snap["streams"].items())/interval,
            physical_s_per_wall_s=sum(s[1]-last["streams"][name][1] for name,s in snap["streams"].items())/interval,
            other_physical_demand=any(not j["job"].startswith(PREFIX) for j in jobs),
            cpu_total_cores=resources["total_cpu_cores"],available_ram_gib=resources["available_ram_gib"],
            gpu_util_percent=resources["gpu_util_percent"],gpu_free_mib=resources["free_gpu_mib"],
            io_pressure_pct=resources["io_pressure_pct"])
        self.windows.append(row)
        self.windows=self.windows[-180:]
        with RECORD.open("a") as f: f.write(json.dumps(row,separators=(",",":"))+"\n")

    def compare(self):
        # Compare 4/6-flow evidence within the same predominant task phase, overlapping
        # external CPU pressure, and no non-E physical demand, never interrupting a run.
        import statistics
        effects=[]
        for phase in {w["phase_majority"] for w in self.windows}:
            for load in range(5):
                def cells(n):
                    return [w for w in self.windows if w["E_physical_slots"]==n and w["phase_majority"]==phase
                        and w["majority_share"]>=.5 and not w["other_physical_demand"]
                        and min(4,int(max(0.,w["cpu_total_cores"]-n)//5))==load and w["physical_s_per_wall_s"]>0]
                four,six=cells(4),cells(6)
                if len(four)>=3 and len(six)>=3:
                    rates=[statistics.median(x["events_s"] for x in arm) for arm in (four,six)]
                    physical=[statistics.median(x["physical_s_per_wall_s"] for x in arm) for arm in (four,six)]
                    if rates[0]>0 and physical[0]>0:
                        effects.append(dict(phase=phase,other_cpu_bin=load,windows=[len(four),len(six)],
                            four_six_events_s=rates,four_six_physics_s=physical,
                            event_ratio=rates[1]/rates[0],physics_ratio=physical[1]/physical[0]))
        return effects


def one(job):
    method,seed,role,version=job
    head=campaign.head_name(method,seed)
    label=f"{PREFIX}{head}-{role}-v{version:03d}"
    cmd=[sys.executable,"-u",campaign.SELF,"paired",method,str(seed),str(version),role]
    code=admitted_call(label,2,cmd,ram=5.5,gpu=1500,disk=1.)
    return None if code==0 else (head,dict(job=label,role=role,returncode=code))


def dispatch(jobs,terminal):
    if not jobs: return
    speed=Speed()
    pending=list(jobs)
    capacity=json.loads((pool.POOL/"capacity.json").read_text())
    with ThreadPoolExecutor(max_workers=3) as workers:
        live={}
        while pending or live:
            routes=inventory()
            hardware=pool.resources()
            speed.record(routes,hardware)
            other=[r["job"] for r in routes if not r["job"].startswith(PREFIX)]
            own=[r for r in routes if r["state"]=="running" and r["job"].startswith(PREFIX)]
            matched=speed.compare()
            import statistics
            measured_gain=statistics.median(m["event_ratio"] for m in matched) if matched else None
            measured_physics=statistics.median(m["physics_ratio"] for m in matched) if matched else None
            enough=(hardware["quota_cores"]-hardware["total_cpu_cores"]>=capacity["spare_cpu"]+2
                and hardware["available_ram_gib"]>=capacity["spare_ram_gib"]+5.5 and hardware["free_gpu_mib"]>=capacity["spare_gpu_mib"]+1500
                and hardware["io_pressure_pct"]<=capacity["max_io_pressure_pct"] and hardware["scratch_free_gib"]>capacity["spare_scratch_gib"]+1.)
            # First available fully physical six-flow window is charged to the preregistered
            # work. Matched evidence subsequently decides whether to repeat that expansion.
            third=not other and len(own)>=2 and enough and (measured_gain is None or max(measured_gain,measured_physics)>1.)
            target=3 if third else 2
            while pending and len(live)<target:
                job=pending.pop(0)
                live[workers.submit(one,job)]=job
            status=dict(study=campaign.STUDY,feeder_pid=os.getpid(),updated_unix=time.time(),target_active_pairs=target,
                physically_running_pairs=len(own),other_physical_demand=other,third_eligible=third,actual_resources=hardware,
                measured_matched_throughput=matched,registered_job_count=len(jobs),unadmitted_in_wave=len(pending),
                launched=[list(j) for j in live.values()],follows_up_without_additional_study=True)
            atomic_json(OUT.with_name("dispatch-live.json"),status)
            if live:
                done,_=wait(live,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    live.pop(future)
                    ended=future.result()
                    if ended:
                        name,reason=ended
                        terminal[name]=reason
                        atomic_json(campaign.SMALL/name/"terminal.json",reason)


def children(pid):
    result=set()
    for tid in Path(f"/proc/{pid}/task").glob("*/children"):
        try: result.update(int(v) for v in tid.read_text().split())
        except FileNotFoundError: pass
    return result


def move_feeder(pid):
    proc=Path(f"/proc/{pid}")
    if proc.exists():
        argv=proc.joinpath("cmdline").read_bytes().split(b"\0")
        if argv[1:4]!=[b"-u",b"scripts/w61/learning/campaign.py",b"matrix"] or proc.joinpath("cwd").resolve()!=ROOT:
            raise ValueError("Selected process is not the original dedicated E matrix feeder")
        # Freeze this Python feeder alone. Its children own the real leases, physical
        # collectors and update. Their process groups, tmux and checkpoints are untouched.
        os.kill(pid,signal.SIGSTOP)
        while proc.exists() and "T" not in proc.joinpath("stat").read_text().split(") ")[1].split()[0]:
            time.sleep(.05)
        existing=children(pid)
        record=dict(previous_feeder=pid,held_registrations=sorted(existing),state="waiting_for_natural_pair_boundaries",
            created_unix=time.time(),new_scheduler=str(Path(__file__).resolve()),new_scheduler_sha256=file_digest(Path(__file__)),
            physics_slots_ceiling=6,maximum_dual_campaigns=3,old_pool_and_all_existing_scientific_sources_retained=True)
        atomic_json(OUT,record)
        while any(Path(f"/proc/{p}/stat").exists() and Path(f"/proc/{p}/stat").read_text().split(") ")[1][0]!="Z" for p in existing):
            time.sleep(15)
        if proc.exists():
            os.kill(pid,signal.SIGTERM)
            os.kill(pid,signal.SIGCONT)
        # Wait for the old process to release its single manager.lock before re-entry.
        with (campaign.RUN/"manager.lock").open("a") as gate:
            fcntl.flock(gate,fcntl.LOCK_EX)
        atomic_json(OUT,dict(record,state="single_feeder_resumed",resumed_unix=time.time(),old_feeder_exited=True))
    campaign.run_jobs=dispatch
    campaign.matrix()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replace-feeder",type=int)
    a=p.parse_args()
    if a.replace_feeder:
        move_feeder(a.replace_feeder)
    else:
        campaign.run_jobs=dispatch
        campaign.matrix()
