"""Bind the predeclared R3 task-gradient comparison to the physical W59 P3 action matrix."""
import argparse
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import torch

from score_s1_w59_interventions import SIGNED, starting_state, summarize
from run_s1_w59_interventions import LARGE, SMALL, job_name
from s1_w59_learning import NOMINAL, TrueTaskCritic, atomic_json, read_json
from s1_w59_learner_critic import BETA, ROOT, file_digest
from s1_w59_model_fields import DT, INTERVAL, PHASES

HORIZON = 32
PLANT_DT = .0002
RULE = ROOT / "experiments/terrain_memory/s1-w59-gradient-admission.json"


def actual_credit(raw,result,snap,horizon=HORIZON):
    """Original task reward, all cost integrals from every applied 0.2 ms plant step."""
    trajectory=raw/"trajectory.npz"
    with np.load(trajectory) as a:
        t,q,feet,grf,ph,cap=[a[k] for k in ("time","qpos","feet","actual_grf","phase","requested_speed_cap")]
    with np.load(raw/"physical-000.npz") as a:
        plant=a["physical"]
    fragments=[]
    for p in sorted(raw.glob("material-[0-9][0-9][0-9][0-9].npz")):
        with np.load(p) as a:
            if len(a["foot"]):fragments.append(a["foot"])
    mat=np.concatenate(fragments)[:len(plant)] if fragments else np.empty((0,0))
    if len(plant)!=len(mat) or not np.allclose(plant[:,0],mat[:,0],atol=1e-8,rtol=0.):
        raise ValueError(f"Original material and load clocks differ at {raw}")
    start=float(t[0]);n=len(plant)
    k=np.rint((plant[:,0]-start)/PLANT_DT).astype(int)
    if not np.array_equal(k,np.arange(n)):
        raise ValueError(f"Gaps in physical branch prefix {raw}")
    cost=np.zeros(n)
    duration=min(float(result["branch"]["elapsed_s"]),2.)
    failure=result["stage_status"]=="physical_failure"
    # A task completion is set by all original 500 Hz hold criteria rather than by interpolation.
    completed=np.flatnonzero(ph==PHASES.index("complete"))
    absorbed=float(t[completed[0]]) if len(completed) else float("inf")
    live=plant[:,0]<absorbed-1.e-8
    force=plant[:,32:42:3];loaded=force>1.
    soft=(mat[:,32:42:3]>1.) & mat[:,2:6].astype(bool)
    bottom=mat[:,8:18:3]
    joint_slip_squared=(mat[:,18:30].reshape(-1,4,3)[:,:,:2]**2).sum(-1)/.25**2
    net_indent=(((np.maximum(-bottom,0)/.001)**2)*soft).sum(1)/np.maximum(soft.sum(1),1)
    net_slip=(joint_slip_squared*loaded).sum(1)/np.maximum(loaded.sum(1),1)
    cost[live]=(-1-net_indent[live]-.1*net_slip[live])*PLANT_DT/60.
    r=deepcopy(snap["joint"]["legal_route"])
    lastcap=float(snap["joint"]["applied_cap"])
    ctrl=np.rint((t-start)/PLANT_DT).astype(int)
    initial_progress=None
    for i,(tm,ix) in enumerate(zip(t,ctrl)):
        if tm>absorbed+1.e-8:break
        phase=PHASES[int(ph[i])]
        mission=SimpleNamespace(phase=phase)
        if "laps" in snap["mission"]:
            mission.laps=snap["mission"]["laps"]+sum(tr["phase"]=="complete" and
                start-1e-8<=tr["time_s"]<=tm+1e-8 for tr in result["transitions"])
        oldprogress,oldlaps=r.progress,r.completed_laps
        r.update(float(tm),mission,q[i,:2],feet[i],grf[i])
        if i:
            credit=5*(r.progress-oldprogress)+10*(r.completed_laps-oldlaps)
            cost[min(ix,n)-1]+=credit
        else:initial_progress=r.progress
        if ix<n and phase in ("outbound","return"):
            change=((float(cap[i])-lastcap)/.035)**2
            cost[ix]-=.01*change
            lastcap=float(cap[i])
    if failure and t[-1]<=absorbed+1.e-8:
        # The recorded result includes original LegalRoute.settle_physical_terminal at the actual endpoint.
        cost[-1]+=5*(float(result["legal_route"]["progress"])-r.progress)-10.
    usable=min(n,round(horizon*DT/PLANT_DT))
    if usable<round(horizon*DT/PLANT_DT) and not failure:
        raise ValueError(f"Nonterminal .64 s branch record missing at {raw}")
    blocks=np.arange(usable)//round(DT/PLANT_DT)
    discounted=float(np.sum(cost[:usable]*np.exp(-BETA*DT*blocks)))
    # Condense modes without equating mere shifts in transition time with a mode switch.
    bits=np.array([1,2,4,8])
    modes=loaded[:usable].astype(int)@bits
    seq=modes[np.r_[True,np.diff(modes)!=0]].tolist() if len(modes) else []
    gate=[]
    for trace in sorted((raw/"w55-f").glob("control-*.npz")):
        with np.load(trace) as z:
            take=(z["time"]>=start-1.e-8)&(z["time"]<=start+horizon*DT+1.e-8)
            zmode=z["load_bearing_contact"][take].astype(int)@bits
            gate.extend(zmode.tolist())
    ordered=[m for i,m in enumerate(gate) if i==0 or m!=gate[i-1]]
    return dict(loaded_contact_sequence=seq,bearing_sequence=ordered,measured_horizon_s=horizon*DT,discounted_w38=discounted,undiscounted_w38_2s=float(cost.sum()),n_substeps=n,
        hard_failure=failure,physical_s=duration,route_start=initial_progress,
        route_end=float(result["legal_route"]["progress"]),end_before_2s=duration<2.-1e-7)


def predictions(wm,crit,state,world,requests,device,material,source,horizon):
    from s1_w59_model import critic_features, ExecutionEnsemble
    from s1_w59_planning import select_batch
    source_arm=source.get("source_arm","B3")
    source_opts=source.get("source_settings",{})
    source_planner=None
    if source_arm.startswith("P3"):
        members=source_opts["model_paths"]
        if [file_digest(Path(p)) for p in members]!=source_opts["model_sha256"]:
            raise ValueError("Stored follow-on P3 model identity changed")
        source_planner=ExecutionEnsemble.load(members,device).eval()
        source_planner.planning_substeps=source_opts["planning_substeps"]
    actions=torch.as_tensor(requests,dtype=torch.float32,device=device).requires_grad_(True)
    n=len(actions)
    x=torch.as_tensor(state,dtype=torch.float32,device=device).expand(n,-1)
    w=torch.as_tensor(world,dtype=torch.float32,device=device).expand(n,-1,-1)
    held=actions
    alive=torch.ones(n,device=device,dtype=torch.bool)
    flow=actions.new_zeros(n)
    boot=actions.new_zeros(n)
    continuation=actions.new_tensor([0.,.060,.05] if source_arm=="Fixed-action" else NOMINAL)
    for k in range(horizon):
        x,y,w=wm.step_with_world(x,held,w,material=material,private=False)
        reward=y[:,INTERVAL["task_reward"]].squeeze(1)
        weight=math.exp(-BETA*k*DT)
        flow=flow+weight*reward*alive
        absorb=(y[:,INTERVAL["complete"]]>.5).flatten() | (y[:,INTERVAL["hard_failure"]]>.5).flatten()
        trunc=(y[:,INTERVAL["truncated"]]>.5).flatten() & ~absorb
        boot=boot+weight*math.exp(-BETA*DT)*alive*trunc*crit(critic_features(x))
        alive=alive & ~absorb & ~trunc
        # First perturbation is absolute. Every later legal request uses the real source's
        # frozen release policy, including a new P3 solve on that predicted future world.
        issue=alive & (y[:,INTERVAL["request"]]>.5).flatten()
        if source_planner is not None and issue.any():
            where=torch.where(issue)[0]
            with torch.no_grad():
                next_action,_=select_batch(source_planner,x[where].detach(),w[where].detach(),
                    private=source_arm=="P3-private",material=source_opts.get("material","M1"))
            held=held.index_copy(0,where,next_action)
        elif source_planner is None:
            held=torch.where(issue[:,None],continuation,held)
    values=flow+boot+math.exp(-BETA*horizon*DT)*alive*crit(critic_features(x))
    deriv,=torch.autograd.grad(values[0],actions)
    return flow.detach().cpu().numpy(),values.detach().cpu().numpy(),deriv[0].detach().cpu().numpy()


def rank_gate(rows):
    """Exactly the original state-bootstrap comparator, substituting its finite-candidate rank sign."""
    ranked=deepcopy(rows)
    for state in ranked:
        if "comparison" in state:
            for d in state["comparison"]["directions"]:
                d["gradient_correct"]=d["rank_correct"]
    return summarize(ranked)


def physical_only(a):
    reg=read_json(a.registry)
    if a.role!="development":
        raise ValueError("The sealed set is only opened with its model, recipe, and registered outcomes")
    out=[]; seen=0; num_physics=0
    for state in reg["states"][a.role]:
        records={}
        with gzip.open(state["snapshot"],"rb") as stream:snap=pickle.load(stream)
        for sign in reg["signed"]:
            key=job_name(state,a.role,sign)
            receipt=SMALL/"episodes"/(key+".json")
            if not receipt.exists():
                records[sign]=dict(status="unrun")
                continue
            result=read_json(receipt)
            if result.get("error") or not result.get("branch",{}).get("applied"):
                records[sign]=dict(status="technical",reason=result.get("reason"))
                continue
            locked=result["branch"]["registration"]
            if locked["snapshot_id"]!=state["snapshot_id"] or locked["sha256"]!=state["sha256"]:
                raise ValueError("Branch did not originate in registered development state")
            measured=actual_credit(LARGE/key,result,snap)
            records[sign]=dict(status="measured",**measured)
            seen+=1;num_physics+=measured["n_substeps"]
        group=dict(state_id=state["snapshot_id"],stratum=state["stratum"],records=records)
        if all(r["status"]=="measured" for r in records.values()):
            nominal=[records[s]["discounted_w38"] for s in ("nominal1","nominal2")]
            floor=abs(nominal[1]-nominal[0])
            mean=sum(nominal)/2
            signed={k:records[k]["discounted_w38"]-mean for k in SIGNED}
            group.update(nominal_replay_floor=floor,signed=signed,
                resolvable=sum(abs(delta)>max(1e-12,floor) for delta in signed.values()))
        out.append(group)
    complete=[r for r in out if "signed" in r]
    report=dict(scope="Actual W38 development action outcomes",registry_sha256=file_digest(a.registry),
        rules_sha256=file_digest(RULE),registered_states=len(out),registered_branches=8*len(out),
        expected_signed_directions=6*len(out),complete_state_count=len(complete),measured_branches=seen,
        fully_paired_directions=6*len(complete),resolved_true_directions=sum(r["resolvable"] for r in complete),
        physical_substeps=num_physics,actual_simulated_seconds=num_physics*PLANT_DT,
        maximum_abs_discounted_w38_contrast=max((abs(v) for r in complete for v in r["signed"].values()),default=None),
        signed_axis_mean={k:float(np.mean([r["signed"][k] for r in complete])) for k in SIGNED} if complete else {},
        states=out,model_gradient_not_evaluated=True)
    atomic_json(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!="states"}),flush=True)


def make(a):
    from s1_w59_model import ExecutionEnsemble
    reg=read_json(a.registry);base=read_json(a.planning)
    if base["role"]!=a.role or base["registry_sha256"]!=file_digest(a.registry) or reg["status"]!="locked":
        raise ValueError("P3 scoring, physical registration and split disagree")
    if len(reg["states"][a.role])!=12 or len(reg["signed"])!=8:
        raise ValueError("W59 action admission requires the complete 12 x 8 registered states and interventions")
    fp=file_digest(a.freeze) if a.freeze else None
    horizon=a.substeps
    if a.role=="sealed_admission" and a.freeze:
        horizon=read_json(a.freeze).get("r3_substeps",32)
    if horizon not in (0,1,5,17,32):raise ValueError("Invalid locked R3 horizon")
    models=base["checkpoints"]
    if len(models)!=3 or any(file_digest(Path(p["path"]))!=p["sha256"] for p in models):
        raise ValueError("The three frozen checkpoints do not match the candidate scoring")
    if a.role=="sealed_admission":
        if not a.freeze:raise ValueError("Missing immutable pre-outcome recipe")
        frozen=read_json(a.freeze)
        if frozen.get("learner_rule_sha256")!=file_digest(RULE) or frozen.get("critic_sha256")!=file_digest(a.critic):
            raise ValueError("Real reward rule or terminal critic differs from the sealed recipe")
        if not a.reuse_for_online and frozen.get("wm_sha256",frozen.get("model_sha256"))!=[p["sha256"] for p in models]:
            raise ValueError("A changed model cannot reuse the initial unopened action admission")
        if a.reuse_for_online and frozen.get("online_recipe_sha256")!=file_digest(ROOT/"experiments/terrain_memory/s1-w59-online-wm.json"):
            raise ValueError("Online model recipe was not preregistered for the reused action outcomes")
    wm=ExecutionEnsemble.load([p["path"] for p in models],a.device).eval()
    wm.requires_grad_(False)
    critic=TrueTaskCritic().to(a.device).eval()
    critic.load_state_dict(torch.load(a.critic,map_location=a.device,weights_only=True)["critic"])
    critic.requires_grad_(False)
    rows=[];floor2=[]
    signed=list(reg["signed"]);requests=np.array([reg["signed"][s] for s in signed])
    if signed[:2]!=["nominal1","nominal2"]:raise ValueError("Unexpected nominal action ordering")
    for st in reg["states"][a.role]:
        rr=dict(state_id=st["snapshot_id"],stratum=st["stratum"],records={},comparison_missing=[])
        raw_nominal=None
        p=Path(st["snapshot"])
        if file_digest(p)!=st["sha256"]:raise ValueError("Registered nominal state changed")
        with gzip.open(p,"rb") as stream:snap=pickle.load(stream)
        for sign in signed:
            key=job_name(st,a.role,sign)
            result_path=SMALL/"episodes"/(key+".json")
            if not result_path.is_file():
                rr["comparison_missing"].append([sign,"unrun"]);continue
            result=read_json(result_path)
            b=result.get("branch") or {}
            locked=b.get("registration") or {}
            if not b.get("applied") or result.get("error"):
                rr["comparison_missing"].append([sign,str(result.get("reason"))]);continue
            if locked.get("sha256")!=st["sha256"] or locked["signed"]!=sign or locked["source_build"]!=st["source_build"]:
                raise ValueError("Physical continuation changed registered state, controller or perturbation")
            if a.role=="sealed_admission" and (not fp or locked.get("model_recipe_sha256")!=fp):
                raise ValueError("Sealed results require the matching pre-outcome model/recipe registration")
            rr["records"][sign]=actual_credit(LARGE/key,result,snap,horizon or HORIZON)
            if sign=="nominal1":raw_nominal=LARGE/key
        if not rr["comparison_missing"]:
            state,world=starting_state(raw_nominal,st,snap)
            if not horizon:
                floor2.append(abs(rr["records"]["nominal1"]["undiscounted_w38_2s"]-rr["records"]["nominal2"]["undiscounted_w38_2s"]))
                rows.append(rr)
                continue
            immediate,pred,grad=predictions(wm,critic,state,world,requests,a.device,a.material,st,horizon)
            actual=np.array([rr["records"][s]["discounted_w38"] for s in signed])
            ref=actual[:2].mean();floor=float(abs(actual[1]-actual[0]))
            f2=abs(rr["records"]["nominal1"]["undiscounted_w38_2s"]-rr["records"]["nominal2"]["undiscounted_w38_2s"])
            floor2.append(f2); ds=[]
            nominal_fail=any(rr["records"][k]["hard_failure"] for k in signed[:2])
            for idx,s in enumerate(signed):
                if s not in SIGNED:continue
                observed=float(actual[idx]-ref);different=abs(observed)>max(floor,1e-12)
                local=float(grad@(requests[idx]-requests[0]));dr=float(pred[idx]-pred[0])
                hard=rr["records"][s]["hard_failure"]
                same=all(rr["records"][s][k]==rr["records"][nom][k] for nom in ("nominal1","nominal2")
                         for k in ("loaded_contact_sequence","bearing_sequence"))
                ds.append(dict(signed=s,same_measured_mode_sequence=same,observed=observed,predicted=dr,linearized=local,resolvable=different,
                    rank_correct=bool(different and dr*observed>0),gradient_correct=bool(different and local*observed>0),
                    hard_failure=hard,excess_hard=hard and not nominal_fail))
            winner=int(np.argmax(pred))
            rr["comparison"]=dict(directions=ds,nominal_gradient=grad.tolist(),model_reward_only=immediate.tolist(),
                model_reward_plus_terminal_value=pred.tolist(),best=signed[winner],best_real_gain=float(actual[winner]-ref),
                nominal_replay_floor=floor,nominal_2s_w38_replay_floor=f2,
                best_new_hard_fail=rr["records"][signed[winner]]["hard_failure"] and not nominal_fail)
        rows.append(rr)
    summary=summarize(rows)
    directions=[d for row in rows if "comparison" in row for d in row["comparison"]["directions"]]
    by_mode={}
    for mode,flag in (("same_mode_sequence",True),("mode_sequence_changed",False)):
        bucket=[d for d in directions if d["same_measured_mode_sequence"]==flag]
        by_mode[mode]=dict(denominator=len(bucket),resolvable=sum(d["resolvable"] for d in bucket),
            locally_correct=sum(d["gradient_correct"] for d in bucket),rank_correct=sum(d["rank_correct"] for d in bucket))
    ranking=rank_gate(base["states"])
    physical_axes=[]
    for trial in base["states"]:
        effects=trial.get("physical_effects") or {}
        if not effects:continue
        floor=effects["full_nominal_repeat_difference"]
        for sign,delta in effects["full_delta_from_nominal"].items():
            motion=trial["records"][sign].get("measured") or {}
            physical_axes.append(dict(state=trial["state_id"],direction=sign,
                above_repeat_without_failure=bool(abs(delta)>max(floor,1e-12) and not motion.get("hard_failure",True))))
    by_stratum={k:summarize([r for r in rows if r["stratum"]==k]) for k in
                  ("rigid_steady","early_touch_or_search","cross_foot_reload")}
    ncomplete=summary.get("evaluated_states",0)
    mean_floor=float(np.mean([r["comparison"]["nominal_replay_floor"] for r in rows if "comparison" in r])) if ncomplete else None
    mean_gain=float(np.mean([r["comparison"]["best_real_gain"] for r in rows if "comparison" in r])) if ncomplete else None
    signed_complete=base["summary"]["measured_branches"]==96 and len(floor2)==12
    status=(a.role=="sealed_admission" and signed_complete)
    configuration=read_json(a.freeze) if a.freeze else {}
    out=dict(role=a.role,r3_scored_substeps=horizon,
        action_evidence="reused locked initial measurements, no independent new action evidence" if a.reuse_for_online else "registered frozen comparison",
        online_reused_actions=bool(a.reuse_for_online),registry_path=str(a.registry.resolve()),freeze_path=str(a.freeze.resolve()) if a.freeze else None,
        planning_substeps=configuration.get("planning_substeps",32),registry_sha256=base["registry_sha256"],rules_sha256=file_digest(RULE),
        freeze_sha256=fp,checkpoint_source=a.material,wm_checkpoints=[str(Path(p["path"]).resolve()) for p in models],
        wm_sha256=[p["sha256"] for p in models],material=a.material,critic_sha256=file_digest(a.critic),
        ppo_action_authority=dict(registered_directions=72,observed_directions=len(physical_axes),
            above_repeat_without_failure=sum(z["above_repeat_without_failure"] for z in physical_axes)),
        ppo_eligible=bool(status and len(physical_axes)==72 and any(z["above_repeat_without_failure"] for z in physical_axes)),
        planning_eligible=bool(status and ranking.get("admitted")),p3_rank=ranking,p3_original=base["summary"],
        r3_gradient_eligible=bool(status and summary.get("admitted")),summary=summary,by_stratum=by_stratum,by_mode=by_mode,states=rows,
        mean_nominal_2s_return_replay_error=float(np.mean(floor2)) if len(floor2)==12 else None,
        calibration_snapshot_ids=[s["snapshot_id"] for split in reg["states"].values() for s in split],
        gradient_admission=dict(states=ncomplete,branches=8*ncomplete,direction_lcb=summary.get("gradient_state_bootstrap_lower95"),
            mean_nominal_replay_error=mean_floor,mean_predicted_improvement_real_effect=mean_gain,
            new_hard_failures=summary.get("directional_excess_hard_failures"),trusted_substeps=horizon if summary.get("admitted") else 0))
    atomic_json(a.output,out)
    if status and a.first_output:
        for seed,source in reg["first_update"].items():
            states=[dict(r,rng_key=r["sha256"]+"/stored") for r in source]
            atomic_json(a.first_output/f"seed-{seed}.json",dict(seed=int(seed),states=states,registry_sha256=base["registry_sha256"]))
    print(json.dumps(dict(role=a.role,complete=ncomplete,summary=summary,planning_eligible=out["planning_eligible"],
                          r3_gradient_eligible=out["r3_gradient_eligible"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry",type=Path,required=True);p.add_argument("--planning",type=Path)
    p.add_argument("--role",choices=("development","sealed_admission"),required=True)
    p.add_argument("--material",choices=("M1","M3"),default="M1")
    p.add_argument("--freeze",type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--first-output",type=Path);p.add_argument("--device",default="cuda")
    p.add_argument("--critic",type=Path,default=ROOT/"work/w59/learner/historical-task-critic.pt")
    p.add_argument("--substeps",type=int,choices=(1,5,17,32),default=32)
    p.add_argument("--physical-only",action="store_true")
    p.add_argument("--reuse-for-online",action="store_true")
    a=p.parse_args()
    if a.physical_only:physical_only(a)
    elif a.planning:make(a)
    else:p.error("--planning is required for the model comparison")

