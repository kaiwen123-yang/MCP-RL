"""Estimate W61 makespan using clocked, disjoint physics windows and whole training leases."""
from collections import defaultdict
from datetime import datetime, timedelta
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo
from independent_timeline import simulate

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
    # Physics completion and video completion now release different resources.
    source_jobs=[read(p) for p in POOL.glob("*.json")]
    pool_jobs={j["job"]:j for j in source_jobs}
    native_tasks=[];done_native=set();actual_frozen={}
    pending_native=[];render_pending={"native":0.,"curve":0.};upload_pending=0.
    for episode in registered:
        name,kind,h=episode["id"],episode["stage"],head(episode)
        root=summaries/h
        physics=root/(name+"-physical.json");video=root/(name+"-summary.json")
        rec=read(physics) if physics.is_file() else (read(video) if video.is_file() else None)
        horizon=protocol["mechanism"][kind+"_horizon_s"] if kind in ("short","recovered") else float(kind[1:])
        raw=WORK.parent/"final-originals"/name
        remaining_seconds=horizon
        if rec is None and name in pool_jobs and pool_jobs[name]["state"]=="running" and (raw/"live.json").is_file():
            remaining_seconds=max(0.,horizon-read(raw/"live.json")["physical_time_s"])
        task=dict(episode,head=h,duration=remaining_seconds,recorded_horizon=horizon)
        if rec is not None:
            done_native.add(name);actual_frozen[name]=rec
            if not video.is_file():render_pending["native"]+=rec["physical_s"]
            delivery=E.parent/"delivery"/(name+".json")
            if not delivery.is_file() or read(delivery).get("status")!="verified":pending_native.append((kind,rec["physical_s"]))
        native_tasks.append(task)
    # A failed E30 retains that case's original gate and its eight independent long episodes.
    included=[]
    for task in native_tasks:
        pred=actual_frozen.get(task.get("predecessor"))
        if pred is not None and (pred["stage_status"]!="passed" or not pred.get("qualification_usable",False)):continue
        included.append(task)
    native_tasks=included
    remaining=[j for j in native_tasks if j["id"] not in done_native]
    long_count={k:sum(e["stage"]==k for e in remaining) for k in ("short","recovered")}
    long_sim=sum(j["duration"] for j in remaining if j["stage"] in long_count)
    frozen_bytes=sum(n*median(old_sizes[k]) for k,n in long_count.items())
    heads=[(m,s) for s in hybrid["seeds"] for m in [*hybrid["methods"],hybrid["geometry_ablation"]]]
    budget=hybrid["used_per_head"];batch=2*hybrid["batch_per_case"]
    training={}
    train_times=[]; inflight={}; current_seats=[]; wait_seconds=[]
    for method,seed in heads:
        h=f"{method.lower()}-s{seed}";p=E/h
        recs=[read(f) for f in p.glob("train-[0-9][0-9][0-9].json")]
        passed=[a for a in recs if a.get("status")=="updated"]
        training[h]=max((a["used"] for a in passed),default=0)
        for a in passed:
            j=read(POOL/f"w61-e61-hybrid-m1-01-{h}-train-v{a['version']:03d}.json")
            leases=[j]
            continuation=a.get("numerical_continuation")
            if continuation:
                resumed=pool_jobs.get(continuation["continuation_job"])
                if resumed is None or resumed.get("state")!="ended":continue
                leases.append(resumed)
            if all(job.get("state")=="ended" for job in leases):
                # One update, including the cost of both parts of a recovered on-policy batch.
                train_times.append(sum(seconds(job) for job in leases))
                wait_seconds.append(sum((datetime.fromisoformat(job["started"])-datetime.fromisoformat(job["registered"])).total_seconds() for job in leases))
        v=training[h]//batch
        live=pool_jobs.get(f"w61-e61-hybrid-m1-01-{h}-train-v{v:03d}")
        if live and live.get("state")=="running":
            inflight[h,v]=max(0.,now.timestamp()-datetime.fromisoformat(live["started"]).timestamp())
    status=E/"independent-dispatch-live.json"
    if status.is_file():current_seats=read(status).get("logical_positions",[])
    ref_case=next(p for p in (summaries/"ppo-s6001").glob("*v000-a-e30-nominal-summary.json"))
    ref=read(ref_case)
    name=ref_case.name.removesuffix("-summary.json");reference_wall=seconds(read(POOL/(name+".json")))
    reference_data=[];native_wall={};native_sizes=defaultdict(list)
    for path in (summaries/"ppo-s6001").glob("*v000-*-summary.json"):
        v=read(path);name=path.name.removesuffix("-summary.json");job=pool_jobs[name]
        base=path.with_name(name+".json");raw=read(base)
        # The four pre-cutover reference leases included the original full renderer.
        # Use the checked original receipt creation clock only for these legacy cases.
        render=max(0.,min(seconds(job)-raw["wall_s"],path.stat().st_mtime-base.stat().st_mtime))
        reference_data.append(dict(episode=name,stage=v["stage"],simulation_s=v["physical_s"],original_lease_s=seconds(job),
            postphysics_inside_legacy_lease_s=render,physical_lease_equivalent_s=seconds(job)-render))
        pub=E.parent/"delivery"/(name+".json")
        if pub.is_file() and read(pub)["status"]=="verified":
            native_sizes[v["stage"]].append(sum(x["size"] for x in read(pub)["assets"]))
    for kind in ("E30","E60"):
        native_wall[kind]=mean(r["physical_lease_equivalent_s"] for r in reference_data if r["stage"]==kind)
    wall_ratio=sum(r["physical_lease_equivalent_s"] for r in reference_data)/sum(r["simulation_s"] for r in reference_data)
    legacy_render=sum(r["postphysics_inside_legacy_lease_s"] for r in reference_data)/sum(r["simulation_s"] for r in reference_data)
    replay_path=E/"media-boundary-recorded-replay.json";render_rates={"native":legacy_render,"curve":legacy_render}
    if replay_path.is_file():
        for item in read(replay_path)["evidence"]:
            source=Path(item["physical_receipt"])
            if source.is_file():
                t=read(source);duration=t.get("simulation_s",t.get("physical_s",0.))
                if duration>0:render_rates[item["kind"]]=item["render_wall_s"]/duration
    # Prefer actual newly closed off-lease movies over replay when present.
    for kind,paths in (("native",summaries.glob("*/*-summary.json")),("curve",E.glob("*-s6*/evaluate-*-media.json"))):
        observed=[]
        for filepath in paths:
            obj=read(filepath)
            media=obj.get("media")
            if isinstance(media,dict) and media.get("render_wall_s") and obj.get("physical_s",0.)>0:
                observed.append(media["render_wall_s"]/obj["physical_s"])
        if observed:render_rates[kind]=median(observed)
    byte_budget={k:median(values) for k,values in (dict(old_sizes)|dict(native_sizes)).items()}
    curve_rates=[]
    for path in (E.parent/"delivery").glob("w61-E-interface-*.json"):
        receipt=read(path)
        if receipt["status"]!="verified":continue
        episode_root=receipt.get("roots",{}).get("episode")
        if episode_root and (Path(episode_root)/"result.json").is_file():
            sec=read(Path(episode_root)/"result.json").get("physical_s",0)
            if sec>0:curve_rates.append(sum(a["size"] for a in receipt["assets"])/sec)
    curve_bytes_per_phys=median(curve_rates) if curve_rates else 0.
    for kind,physical in pending_native:upload_pending+=byte_budget.get(kind,0.)
    speeds_live=defaultdict(list); reference_windows=defaultdict(list)
    recorded=WORK/"throughput.jsonl"
    if recorded.is_file():
        for line in recorded.open():
            w=json.loads(line)
            if w["endpoint_long"]==w["width"] and w["seconds"]>0:
                speeds_live[w["width"]].append(w)
            elif all("-v000-" in name for name in w["episodes"]) and w["seconds"]>0:
                reference_windows[w["width"]].append(w)
    observed_reference={str(k):dict(windows=len(v),wall_s=sum(x["seconds"] for x in v),
        simulated_s=sum(x["physical_s"] for x in v),
        aggregate_sim_s_per_wall_s=sum(x["physical_s"] for x in v)/sum(x["seconds"] for x in v))
        for k,v in reference_windows.items()}
    forecasts={}; batch_wall=mean(train_times)
    spent=sum(min(s,batch_wall) for s in inflight.values())
    train_slot_h=(sum(budget-v for v in training.values())/batch*batch_wall-spent)*2/3600
    nominal_sim=sum(j["duration"] for j in remaining if j["stage"] in ("E30","E60"))
    curves_done=set()
    for m,seed in heads:
        h=f"{m.lower()}-s{seed}"
        for node in curves:
            version=node//batch;f=E/h/f"evaluate-{version:03d}.json"
            if not f.is_file():continue
            curves_done.add((h,version))
            if not (E/h/f"evaluate-{version:03d}-media.json").is_file():
                render_pending["curve"]+=sum(ep.get("physical_s",0) for c in read(f).get("cases",[]) for ep in c.get("episodes",[]))
    curve_sim=2*60*(len(heads)*len(curves)-len(curves_done))
    nominal_slot_hours=(sum(native_wall[j["stage"]]*j["duration"]/float(j["stage"][1:])
                            for j in remaining if j["stage"] in native_wall)+curve_sim*wall_ratio)/3600
    head_priority=[f"{m.lower()}-s{s}" for s in hybrid["seeds"] for m in ("E-score","PPO","SPO")]
    head_priority.extend(f"{hybrid['geometry_ablation'].lower()}-s{s}" for s in hybrid["seeds"])
    # Per physical slot-hour occupied in the last hour; requests waiting to start do not consume slots.
    share=defaultdict(float)
    for j in source_jobs:
        if "started" not in j:continue
        a=max(now.timestamp()-3600,datetime.fromisoformat(j["started"]).timestamp())
        b=min(now.timestamp(),datetime.fromisoformat(j["ended"]).timestamp()) if "ended" in j else now.timestamp()
        k="training_and_curves" if j["job"].startswith("w61-e61-hybrid-m1-01-") else ("frozen_evaluations" if j["job"].startswith("w61-e-final-") else "other_jobs")
        share[k]+=max(0.,b-a)*j["slots_requested"]/3600
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
        timeline=simulate(now=now,width=width,heads=head_priority,seats=current_seats,versions={h:v//batch for h,v in training.items()},
            nodes=[i//batch for i in curves],last=budget//batch,batch_wall={h:batch_wall for h in head_priority},
            in_flight=dict(inflight),curves_done=curves_done,curve_wall=60*wall_ratio,native_jobs=native_tasks,done_native=done_native,
            native_wall=native_wall,render_rate=render_rates,asset_bytes=byte_budget,upload_rate=median(upload_rates),long_rate=rate,
            nominal_render_backlog=sum(render_pending[k]*render_rates[k] for k in render_pending),
            upload_backlog=upload_pending/median(upload_rates),curve_bytes_per_s=curve_bytes_per_phys)
        forecasts[str(width)]=dict(effective_slots=width,source=source,independent_head_timeline=timeline,training_all_slot_hours=train_slot_h,
            remaining_primary_training_net_h=primary_h,long_eval_net_h=long_hours,actual_long_rate=rate,
            original_and_curve_upper_slot_hours=nominal_slot_hours,parallel_upload_estimated_h=bundle_hours,
            total_effective_work_h=compute,from_now_full_allocation_finish=(now+timedelta(hours=compute)).isoformat(timespec="minutes"))
    return dict(as_of_shanghai=now.isoformat(timespec="seconds"),scope="nine formal and three E-Fisher heads, all 96 long episodes plus original qualifications/learning curve nodes, incremental publication",
        primary_used=sum(training[f"{m.lower()}-s{s}"] for m in hybrid["methods"] for s in hybrid["seeds"]),
        registered_long=plan["long_count"],remaining_long=long_count,full_horizon_physical_s=long_sim,
        frozen_8192_heads=study["checkpoint_heads_8192"],head_used=training,completed_training_batch_wall_s=train_times,
        independent_logical_heads=current_seats,closed_batch_queue_wait_s=wait_seconds,last_hour_occupied_slot_share=share,
        GPU_material_benchmark=material_status(), independently_frozen_targets_without_retry_budget_reduction=True,
        render_separate_from_physics=dict(original_references=reference_data,legacy_render_inside_s_per_sim_s=legacy_render,
            paced_media_wall_s_per_sim_s=render_rates,open_native_and_curve_render_physical_s=render_pending,
            stage_wall_without_render=native_wall,curve_upload_proxy_bytes_per_phys_s=curve_bytes_per_phys),
        first_head_and_pipeline_method="Coupled event projection: one ordered chain per head, 256 used per atomic pair, frozen/curve tasks become eligible at that head's own nodes. Each E30 precedes only its E60; native and curve movies plus uploads progress on separate lanes. Four/six are effective E capacities, not guarantees of future external load.",
        material_backend_projection="Matched real CPU production leases. A cross-collector GPU probe does not enter these timings unless the corresponding backend actually processes a production batch.",
        E30_reference_physical_s=ref["physical_s"],E30_reference_wall_s=reference_wall,
        W61_reference_windows=observed_reference,historical_complete_long_episodes=len(material),historical_raw_clock_log=str(old_clocks),
        historical_clock_sha256=hashlib.sha256(old_clocks.read_bytes()).hexdigest(),
        window_rule="non-overlapping >=180s with identical W59 long job set throughout, no other project-lane physics in slots; reset on membership change/missing clock/monitor gap >90s; zero progress retained",
        measured_historical_concurrency=windows,current_W61_endpoint_window_counts={k:len(v) for k,v in speeds_live.items()},
        median_historical_active_upload_MiB_s=median(upload_rates)/2**20,
        estimated_long_asset_GiB=frozen_bytes/2**30,scenarios=forecasts,
        interpretation="Full-duration scenarios, with all E30 cases allowed their E60. Curve nodes use their maximum 60 simulated seconds. Slot sharing with remaining A–F jobs, release tail, setup gaps and fluctuating external load can shift the calendar; a natural physical failure ends that episode early. No wall deadlines.")


def material_status():
    folder=BASE/"MCP-RL-w61-material"/REL/"w61/material/w61-e-packet-throughput-v1"
    p=folder/"index.json"
    if not p.is_file():return dict(production_backend="cpu",measured_groups=0)
    result=read(p)
    return dict(stage=result["stage"],production_backend=result["production_backend"],production_eligible=result["production_eligible"],
                measurements=result.get("comparisons",{}),receipt=str(p))


def head(ep):
    return f"{ep['method'].lower()}-s{ep['seed']}"


def write_forecast(path):
    r=estimate();tmp=path.with_suffix(".pending")
    tmp.write_text(json.dumps(r,indent=2)+"\n");tmp.replace(path)
    print(json.dumps({k:r[k] for k in ("as_of_shanghai","primary_used","remaining_long","completed_training_batch_wall_s","scenarios")}),flush=True)


def watch():
    with (WORK/"forecast.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signature=None
        while True:
            status=read(WORK/"live.json")
            log=WORK/"throughput.jsonl"
            sizes=defaultdict(float)
            if log.is_file():
                for line in log.read_text().splitlines():
                    row=json.loads(line)
                    if row["endpoint_long"]==row["width"]:sizes[row["width"]]+=row["seconds"]
            # Recharge the estimate at real checkpoints, completed episodes, or
            # sustained new evidence about concurrent endpoint performance.
            nodes=sum(len(list(h.glob("train-[0-9][0-9][0-9].json"))) for h in E.glob("*-s6*"))
            mat=material_status()
            key=(datetime.now(SH).date(),nodes//3,status["checkpoint_heads_8192"],status["completed"],
                 mat.get("stage"),len(mat.get("measurements",{})),
                 status["long"].get("verified",0),tuple(sorted((n,int(sec>=180),int(sec//1800)) for n,sec in sizes.items())))
            if key!=signature:
                write_forecast(WORK/"eta-live.json");signature=key
            if status["completed"]:return
            time.sleep(60)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--watch",action="store_true")
    if p.parse_args().watch:watch()
    else:write_forecast(E/"final-evaluation/parallel-throughput-estimate.json")
