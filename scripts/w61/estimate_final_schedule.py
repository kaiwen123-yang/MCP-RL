"""Estimate W61 makespan using clocked, disjoint physics windows and whole training leases."""
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

BASE=Path("/root/autodl-tmp")
LEARNING=BASE/"MCP-RL-w61-learning"
OLD=BASE/"MCP-RL-w59"
REL=Path("results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
E=LEARNING/REL/"w61/learning/e61-hybrid-m1-01"
WORK=LEARNING/"work/w61/learning/W61-E-M1-hybrid-v1/e61-hybrid-m1-01/parallel-final"
POOL=BASE/"MCP-RL-w61-joint/work/w61/pool/jobs"
SH=ZoneInfo("Asia/Shanghai")


def read(path):
    return json.loads(path.read_text())


def seconds(job):
    return (datetime.fromisoformat(job["ended"])-datetime.fromisoformat(job["started"])).total_seconds()


def old_windows(long_names):
    log=OLD/"work/w59/resource-samples.jsonl"
    streams=defaultdict(list);start=None;last=None;prior=None
    for line in log.open():
        row=json.loads(line)
        t=row["unix_s"];names=frozenset(row["slots"])
        if names!=prior or last is None or t-last>90:
            start=None
        if not names or not names<=long_names or not names<=row["clock"].keys():
            start=None
        elif start is None:
            start=row
        elif t-start["unix_s"]>=180:
            wall=t-start["unix_s"]
            sim=sum(row["clock"][n][0]-start["clock"][n][0] for n in names)
            if sim>=0:streams[len(names)].append((wall,sim))
            start=row
        prior=names;last=t
    return {str(k):dict(windows=len(v),wall_s=sum(w for w,_ in v),physical_s=sum(x for _,x in v),
                        aggregate_sim_s_per_wall_s=sum(x for _,x in v)/sum(w for w,_ in v)) for k,v in streams.items()}


def estimate():
    now=datetime.now(SH)
    plan=read(E/"final-evaluation/parallel-plan.json");registered=plan["episodes"]
    protocol=read(LEARNING/"experiments/terrain_memory/w61/learning/final-evaluator.json")
    hybrid=read(LEARNING/"experiments/terrain_memory/w61/learning/hybrid.json")
    curves=hybrid["curve_nodes"]
    old=OLD/REL/"w59";old_clocks=OLD/"work/w59/resource-samples.jsonl"
    material={f.stem:o for f in (old/"attempts").glob("*.json") if
              (o:=read(f)).get("task_mode") in ("short","recovered") and o.get("simulation_s",0)>165}
    windows=old_windows(set(material))
    old_sizes=defaultdict(list);upload_rates=[]
    for name,attempt in material.items():
        assets=read(old/"delivery"/(name+".json"))["assets"]
        arr=sorted((x["verified_unix_s"],x["size"]) for x in assets)
        old_sizes[attempt["task_mode"]].append(sum(n for _,n in arr))
        upload_rates.append(sum(n for _,n in arr[1:])/(arr[-1][0]-arr[0][0]))
    study=read(WORK/"live.json")
    summaries=E/"final-evaluation"
    remaining=[e for e in registered if not (summaries/head(e)/(e["id"]+"-summary.json")).is_file()]
    long_count={k:sum(e["stage"]==k for e in remaining) for k in ("short","recovered")}
    long_sim=sum(n*protocol["mechanism"][k+"_horizon_s"] for k,n in long_count.items())
    frozen_bytes=sum(n*median(old_sizes[k]) for k,n in long_count.items())
    heads=[(m,s) for s in hybrid["seeds"] for m in [*hybrid["methods"],hybrid["geometry_ablation"]]]
    budget=hybrid["used_per_head"];batch=2*hybrid["batch_per_case"]
    training={}
    train_times=[]
    for method,seed in heads:
        h=f"{method.lower()}-s{seed}";p=E/h
        recs=[read(f) for f in p.glob("train-*.json")]
        passed=[a for a in recs if a.get("status")=="updated"]
        training[h]=max((a["used"] for a in passed),default=0)
        for a in passed:
            j=read(POOL/f"w61-e61-hybrid-m1-01-{h}-train-v{a['version']:03d}.json")
            if j.get("state")=="ended":train_times.append(seconds(j))
    ref_case=next(p for p in (summaries/"ppo-s6001").glob("*v000-a-e30-nominal-summary.json"))
    ref=read(ref_case)
    name=ref_case.name.removesuffix("-summary.json");reference_wall=seconds(read(POOL/(name+".json")))
    wall_ratio=reference_wall/ref["physical_s"]
    speeds_live=defaultdict(list)
    recorded=WORK/"throughput.jsonl"
    if recorded.is_file():
        for line in recorded.open():
            w=json.loads(line)
            if w["endpoint_long"]==w["width"] and w["seconds"]>0:speeds_live[w["width"]].append(w)
    forecasts={}; train_slot_h=sum(budget-v for v in training.values())/batch*mean(train_times)*2/3600
    nominal_sim=sum(int(x["stage"][1:]) for x in remaining if x["stage"] in ("E30","E60"))
    curve_sim=sum(2*60 for m,s in heads for n in curves if
                  not (E/f"{m.lower()}-s{s}"/f"evaluate-{n//batch:03d}.json").exists())
    nominal_slot_hours=(nominal_sim+curve_sim)*wall_ratio/3600
    for width in (4,6):
        # Use actual matched W61 endpoint windows when available; otherwise the measured
        # same-width W59 material-route cohort supplies the parallel-rate proxy.
        live=speeds_live.get(width,[])
        duration=sum(v["seconds"] for v in live)
        if duration>=180 and sum(v["physical_s"] for v in live)>0:
            rate=sum(v["physical_s"] for v in live)/duration
            source="W61 current 8192-only windows"
        else:
            rate=windows[str(width)]["aggregate_sim_s_per_wall_s"]
            source="W59 matched all-long occupancy"
        long_hours=long_sim/rate/3600
        bundle_hours=frozen_bytes/median(upload_rates)/3600
        # A slot-hour is charged to exactly one task. This scenario lets subsequent
        # training overlap with eligible frozen heads; it does not allocate six slots twice.
        compute=train_slot_h/width+nominal_slot_hours/width+long_hours
        primary_left=sum(budget-training[f"{m.lower()}-s{s}"] for m in hybrid["methods"] for s in hybrid["seeds"])
        primary_h=primary_left/batch*mean(train_times)*2/width/3600
        forecasts[str(width)]=dict(effective_slots=width,source=source,training_all_slot_hours=train_slot_h,
            remaining_primary_training_net_h=primary_h,long_eval_net_h=long_hours,actual_long_rate=rate,
            original_and_curve_upper_slot_hours=nominal_slot_hours,parallel_upload_estimated_h=bundle_hours,
            total_effective_work_h=compute,from_now_full_allocation_finish=(now+timedelta(hours=compute)).isoformat(timespec="minutes"))
    return dict(as_of_shanghai=now.isoformat(timespec="seconds"),scope="nine formal and three E-Fisher heads, all 96 long episodes plus original qualifications/learning curve nodes, incremental publication",
        primary_used=sum(training[f"{m.lower()}-s{s}"] for m in hybrid["methods"] for s in hybrid["seeds"]),
        registered_long=plan["long_count"],remaining_long=long_count,full_horizon_physical_s=long_sim,
        frozen_8192_heads=study["checkpoint_heads_8192"],head_used=training,completed_training_batch_wall_s=train_times,
        E30_reference_physical_s=ref["physical_s"],E30_reference_wall_s=reference_wall,
        historical_complete_long_episodes=len(material),historical_raw_clock_log=str(old_clocks),
        historical_clock_sha256=hashlib.sha256(old_clocks.read_bytes()).hexdigest(),
        window_rule="non-overlapping >=180s with identical W59 long job set throughout, no other project-lane physics in slots; reset on membership change/missing clock/monitor gap >90s; zero progress retained",
        measured_historical_concurrency=windows,current_W61_endpoint_window_counts={k:len(v) for k,v in speeds_live.items()},
        median_historical_active_upload_MiB_s=median(upload_rates)/2**20,
        estimated_long_asset_GiB=frozen_bytes/2**30,scenarios=forecasts,
        interpretation="Full-duration scenarios, with all E30 cases allowed their E60. Curve nodes use their maximum 60 simulated seconds. Slot sharing with remaining A–F jobs, release tail, setup gaps and fluctuating external load can shift the calendar; a natural physical failure ends that episode early. No wall deadlines.")


def head(ep):
    return f"{ep['method'].lower()}-s{ep['seed']}"


if __name__=="__main__":
    r=estimate();p=E/"final-evaluation/parallel-throughput-estimate.json"
    tmp=p.with_suffix(".pending");tmp.write_text(json.dumps(r,indent=2)+"\n");tmp.replace(p)
    print(json.dumps({k:r[k] for k in ("as_of_shanghai","primary_used","remaining_long","completed_training_batch_wall_s","scenarios")}))
