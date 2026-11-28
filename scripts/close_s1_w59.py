"""Close W59 after earned physical evaluation, head ledgers and D57 Release evidence agree."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import fcntl
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
from urllib.error import URLError
import subprocess
from zoneinfo import ZoneInfo

from collect_s1_w59 import collect
from publish_s1_go2_w54 import write
from publish_s1_w59 import ROOT, STATE, WORK, release
from run_s1_w59_interventions import job_name

ZONE = ZoneInfo("Asia/Shanghai")
BG = dict(deadline=None, wall_deadline_enabled=False, kill_on_conversation_end=False)
ARMS = ("PPO-full","Diff-full","Diff-private","Diff-fixedWM")
SEEDS = (5901,5902,5903)
FOUR = {f"{c}/{m}" for c in "AB" for m in ("short","recovered")}


def read(path):
    return json.loads(path.read_text())


@lru_cache(maxsize=1024)
def file_hash(path,size,mtime_ns):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def digest(path):
    st=path.stat()
    return file_hash(path,st.st_size,st.st_mtime_ns)


def stamp():
    return datetime.now(ZONE).isoformat(timespec="seconds")


def wait_state(state, items):
    path = WORK/"closeout-status.json"
    old = read(path) if path.is_file() else {}
    if (old.get("state"),old.get("pending")) != (state,items):
        write(path,dict(state=state,pending=items,shanghai=stamp(),**BG))
        print(stamp(),state,str(items)[:1400],flush=True)


def models():
    result = {}
    for kind in ("blackbox","structured"):
        result[kind] = {}
        for seed in SEEDS:
            key=f"{kind}-{seed}"
            fit=read(STATE/"model"/(key+".json"))
            if not fit["terminal"]:
                raise ValueError("An execution-model fit still has an open optimizer: "+key)
            err=read(STATE/"model"/(key+"-open-loop.json"))
            result[kind][seed]=dict(nodes=fit["nodes"], evaluated_checkpoint_step=err["backbone_step"],
                development_force_error=[dict(horizon_s=v["horizon_s"],origins=v["origins"],finite=v["finite"],
                mean_Fz_MAE_N=v["metrics"]["foot_fz_MAE_N"]["mean"]) for v in err["per_horizon"]])
    rec=read(STATE/"model/recovery.json")
    prior=read(STATE/"model/fit-technical.json")
    return dict(fits=result,bank=read(STATE/"model/bank-progress.json"),
        technical_preparation_attempts=prior["attempts"],
        material=dict(offline_candidate_accepted=rec["offline_candidate_accepted"],
        accepted_physical=rec["accepted_physical"],fitted=rec["fitted"],fit_rows=rec["fit_rows"],parameters=rec["parameters"],
        by_recording={k:{field:v[field] for field in ("m1_foot_force_error_N","m3_foot_force_error_N","short_M1","short_M3")}
                      for k,v in rec["per_episode"].items()}))


def ledgers(series, freeze, roster):
    heads={}
    for seed in SEEDS:
        for arm in ARMS:
            key=f"{seed}/{arm}"
            base=series["heads"][key]
            group=arm if series["matched"] else "PPO-full-fallback"
            run=WORK/"learning"/f"seed-{seed}"/group
            compact=STATE/"learning"/f"seed-{seed}"/group
            if base["mode"]=="unrun":
                heads[key]=dict(algorithm=arm,state="unrun",observed=0,used=0,tail_excluded=0,
                    imagined_transitions=0,accepted_actor_steps=0)
                continue
            updates=[read(p) for p in sorted(compact.glob("update-*.json"))]
            paid=set()
            if (run/"used-event-ids.jsonl").exists():
                with (run/"used-event-ids.jsonl").open() as stream:
                    for line in stream:
                        ids=json.loads(line)["ids"]
                        if len(set(ids))!=len(ids) or paid.intersection(ids):
                            raise ValueError("Repeated optimization event in "+key)
                        paid.update(ids)
            if len(paid)!=base["used"]:
                raise ValueError(f"{key} used {base['used']} but {len(paid)} event IDs are verified")
            launches=[json.loads(v) for v in (run/"physics-attempts.jsonl").read_text().splitlines()]
            train={l["attempt"] for l in launches if l["task"]=="training"}
            stats=Counter(); missing=[]
            for name in sorted(train):
                p=STATE/'episodes'/(name+'.json')
                ep=read(p) if p.is_file() else {}
                info=ep.get("w59_learning")
                if info:
                    stats.update({k:int(info[k]) for k in ("requested","completed","tail_excluded","used_barrier_rows","data_cut")})
                else:
                    missing.append(name)
            selected=next((x for x in freeze["selections"] if x["seed"]==seed and x["arm"]==arm),None)
            first=compact/"first-update-real.json"
            f=read(first) if first.is_file() else {}
            heads[key]=dict(algorithm=series["role"] if "role" in series else ("matched" if series["matched"] else "real-PPO-fallback"),
                arm=arm,state=base["mode"],actual_used=len(paid),used=len(paid),requested=stats["requested"],
                observed=stats["completed"],tail_excluded=stats["tail_excluded"],eligible_barrier_rows=stats["used_barrier_rows"],
                technical_data_cut=stats["data_cut"],unknown_observation_counts=missing,
                observations_known_through_receipts=sum(n in roster for n in train),
                imagined_transitions=sum(x["imagined_substeps"] for x in updates),
                real_critic_steps=sum(x.get("real_critic_steps",0) for x in updates),imagined_critic_steps=sum(x.get("imagined_critic_steps",0) for x in updates),
                accepted_actor_steps=sum(x["actual_accepted_actor_steps"] for x in updates),
                optimizer_batches=len(updates),frozen_node=selected["node"] if selected else None,
                frozen_actor_sha256=selected["actor_sha256"] if selected else None,
                first_update={k:f.get(k) for k in ("accepted","marginal_mean_delta","mean_replay_floor","before_failures","after_failures")},
                closed_nodes=[read(p)["used_events"] for p in sorted(compact.glob("node-*.json")) if read(p).get("complete_common_node")])
    return heads


def effective_head(attempt):
    cfg=attempt.get("w59",{}).get("learner_config") or {}
    return (f'{cfg["seed"]}/{cfg["head"]}' if cfg else attempt.get("arm"))


def real_cost_and_results(roster):
    cost=defaultdict(Counter);stages=defaultdict(list); mechanisms=[];pairs={}
    qualifications=defaultdict(dict)
    for name,a in roster.items():
        role=a["task_mode"]
        p=STATE/"episodes"/(name+".json"); ep=read(p) if p.is_file() else {}
        cost_role=role
        if role=="branch":
            cost_role=("development_signed_action" if name.startswith("w59-action-development-")
                       else "sealed_signed_action" if name.startswith("w59-action-sealed-") else "first_actor_update")
        t=cost[cost_role];t["attempts"]+=1;t[a["status"]]+=1
        rows=ep.get("physical_samples",a.get("physical_samples"))
        if rows is None:t["unknown_physical_cost"]+=1
        else:t["actual_plant_substeps"]+=rows
        if role=="original":
            stages[(effective_head(a),a["case"])].append(dict(attempt=name,level=a["level"],status=a["status"],reason=a.get("reason"),controller=a["controller"]))
        if role not in ("short","recovered","original"):continue
        gait=STATE/"gait/new"/(name+".json");scores=read(gait) if gait.is_file() else {}
        summary=dict(attempt=name,head=effective_head(a),controller=a["controller"],case=a["case"],split=a["w59"]["split"],mode=role,
            plant_s=ep.get("simulation_s"),status=a["status"],slip_per_m=scores.get("moving_point_slip_per_m"),
            positive_mechanical_CoT=scores.get("moving_positive_CoT"),observation_complete=False)
        m=ep.get("mechanism")
        if m:
            summary.update(requested=m["requested"],achieved=m["achieved"],home_stable=m["home_stable"],body_traversals=m["body_traversals"])
            summary["observation_complete"]=a["status"] in ("passed","mechanism_criteria_unmet") and ep["simulation_s"]+1.e-7>=ep["evaluation_stage"]["duration_s"]
            mechanisms.append(summary)
        elif role=="original":
            summary["observation_complete"]=a["status"]=="passed"
        if summary["observation_complete"]:
            pairs[(effective_head(a),a["case"],a["w59"]["split"],role,a["level"])]=summary
    comparisons=[]
    for (who,case,split,mode,level),v in pairs.items():
        if who=="B3":continue
        base=pairs.get(("B3",case,split,mode,level))
        if not base or abs(v["plant_s"]-base["plant_s"])>1.e-5:continue
        comparisons.append(dict(head=who,case=case,split=split,task=mode,level=level,attempt=v["attempt"],baseline=base["attempt"],
            achieved_target_delta=v["achieved"]-base["achieved"] if "achieved" in v else None,
            moving_contact_point_slip_delta_per_m=v["slip_per_m"]-base["slip_per_m"] if None not in (v["slip_per_m"],base["slip_per_m"]) else None,
            moving_positive_joint_CoT_delta=v["positive_mechanical_CoT"]-base["positive_mechanical_CoT"] if None not in (v["positive_mechanical_CoT"],base["positive_mechanical_CoT"]) else None))
    for (who,c),values in stages.items():
        passed=[v["level"] for v in values if v["status"]=="passed"]
        qualifications[who][c]=dict(highest=max(passed,key=lambda s:int(s[1:])) if passed else None,attempts=values)
    for t in cost.values():
        t["actual_simulated_seconds"]=t["actual_plant_substeps"]*.0002
    exposure={}
    for row in mechanisms:
        idx=f"{row['head']}/{row['split']}/{row['mode']}"
        val=exposure.setdefault(idx,dict(episodes=0,completed_windows=0,targets_achieved=0,targets_requested=0,home_stable=0))
        val["episodes"]+=1
        val["completed_windows"]+=int(row["observation_complete"])
        val["targets_achieved"]+=row["achieved"]
        val["targets_requested"]+=row["requested"]
        val["home_stable"]+=int(row["home_stable"])
    return dict(cost_by_task=cost,originals=qualifications,mechanisms=mechanisms,exposure_by_head_split_history=exposure,paired_comparisons=comparisons)


def resource_use():
    groups={}
    path=WORK/"resource-samples.jsonl"
    with path.open() as file:
        for line in file:
            if not line.endswith("\n"):continue
            sample=json.loads(line)
            widths=[len(slots) for slots in sample["slots"].values()]
            width=sum(widths)
            q=groups.setdefault(width,dict(samples=0,project_cores=[],other_cores=[],gpu_util=[],max_dual_campaigns=0))
            q["samples"]+=1
            q["max_dual_campaigns"]=max(q["max_dual_campaigns"],widths.count(2))
            if "W59_cpu_cores" in sample:
                q["project_cores"].append(sample["W59_cpu_cores"]);q["other_cores"].append(sample["other_cpu_cores"])
            q["gpu_util"].append(sample["gpu_util_used_total_MiB"][0])
    from statistics import median
    return {width:dict(samples=q["samples"],max_dual_campaigns=q["max_dual_campaigns"],
        median_W59_cores=median(q["project_cores"]) if q["project_cores"] else None,
        median_other_cores=median(q["other_cores"]) if q["other_cores"] else None,
        median_whole_board_gpu_pct=median(q["gpu_util"])) for width,q in groups.items()}


def evaluation_ready(series,freeze,controls,roster):
    pending=[]
    if series.get("state") not in ("partial","blocked","complete_earned_scope"):pending.append("learning-series terminal decision")
    nodes={int(k):v for k,v in freeze["N_s"].items()}
    selected={(i["seed"],i["arm"]) for i in freeze["selections"]}
    for head,record in series["heads"].items():
        if record["mode"]=="unrun":continue
        seed,arm=head.split("/",1);seed=int(seed)
        suffix=arm if series["matched"] else "PPO-full-fallback"
        directory=STATE/"learning"/f"seed-{seed}"/suffix
        earned=[read(p)["used_events"] for p in directory.glob("node-*.json") if read(p).get("complete_common_node")]
        if earned and ((seed,arm) not in selected or nodes.get(seed) not in earned):
            pending.append(f"{head} earned original evaluation at a selected closed node; available {sorted(earned)}")
    for info in freeze["selections"]:
        seed,arm=info["seed"],info["arm"]
        key=f"{seed}/{arm}";d=series["evaluations"].get(key,{})
        if not d or d.get("actor_sha256")!=info["actor_sha256"] or nodes[seed]!=info["node"]:
            pending.append(key+" frozen evaluation");continue
        qok=True
        for case in "AB":
            q=d["qualification"][case]
            for level in ("E30","E60"):
                if level=="E60" and q["E30"].get("status")!="passed":continue
                cell=q.get(level) or {}
                if cell.get("attempt") not in roster:pending.append(f"{key} {case}/{level}")
                qok &= cell.get("status")=="passed"
        for slot in sorted(FOUR):
            cell=d["heldout"].get(slot,{})
            if cell.get("ran"):
                if cell.get("attempt") not in roster:pending.append(key+" independent "+slot)
            elif qok or not cell.get("reason"):
                pending.append(key+" independent "+slot)
        for case,long in d.get("long_original",{}).items():
            if long["target_exposure"] and not long["both_costs_deteriorated"]:
                if not (long["E300"] or {}).get("attempt") in roster:pending.append(key+" "+case+"/E300")
                if (long["E300"] or {}).get("status")=="passed" and not (long["E600"] or {}).get("attempt") in roster:pending.append(key+" "+case+"/E600")
    for arm in controls["arms"]:
        if arm["status"]=="original_qualification_failed":continue
        name=f'{arm["arm"].lower()}-{controls["material"].lower()}-heldout'
        p=STATE/"mechanism"/(name+".json")
        if not p.exists():pending.append(name);continue
        v=read(p)
        cells={(roster[e["attempt"]]["case"],roster[e["attempt"]]["task_mode"]) for e in v["episodes"] if e["attempt"] in roster}
        if cells!={(c,m) for c in "AB" for m in ("short","recovered")}:pending.append(name+" incomplete 4-case denominator")
    return pending


def labels_and_gaps(roster,model_results):
    labels={"w59-causal-observation-bank","w59-prespecified-action-data","w59-fp64-material-recovery",
            "w59-execution-model-open-loop","w59-prior-route-opportunities","w59-historical-task-critic"}
    for kind, members in model_results["fits"].items():
        for seed,member in members.items():
            for node in member["nodes"]:
                labels.add(f"w59-wm-{kind}-{seed}-node-{node['updates']:d}")
            labels.add(f"w59-wm-action-{kind}-{seed}-node-10000")
    reg=read(STATE/"action/registry-v3.json")
    branch_names=set()
    for role in ("development","sealed_admission"):
        for state in reg["states"][role]:
            prefix=job_name(state,role,"nominal1").removesuffix("-nominal1-a01")
            labels.add(prefix+"-branches")
            branch_names.update(job_name(state,role,sign) for sign in reg["signed"])
    missing=[]
    if not branch_names.issubset(roster):missing.append("calibration attempt receipts")
    inv_path=STATE/"delivery/learning-inventory.json"
    inv=read(inv_path) if inv_path.is_file() else {}
    if inv.get("status")!="verified":missing.append("learning-inventory verified")
    needed=set()
    for name,row in roster.items():
        if row["task_mode"]=="training" or (row["task_mode"]=="branch" and name not in branch_names):
            linked=inv.get("attempt_receipts",{}).get(name)
            if not linked:missing.append("learning raw "+name)
            else:labels.update(linked)
        elif row["task_mode"]!="branch":
            labels.add(name)
            if row.get("case")=="A" and row["task_mode"]=="original" and row.get("level") in ("E60","E300","E600") and row["status"]!="technical_interruption":
                labels.add(name+"-gait-excerpts")
    for pat in ("seed-*/*/policy-*.pt","seed-*/*/proposal-*.pt"):
        for p in (WORK/'learning').glob(pat):needed.add(digest(p))
    for h in needed:
        refs=inv.get("checkpoint_receipts",{}).get(h)
        if refs:labels.update(refs)
        else:missing.append("learning checkpoint "+h)
    expected={}
    for label in sorted(labels):
        p=STATE/"delivery"/(label+".json");q=read(p) if p.is_file() else {}
        if q.get("status")!="verified" or not q.get("assets"):
            missing.append("release "+label);continue
        if label in roster and roster[label]["task_mode"] in ('original','short','recovered') and roster[label]["status"]!="technical_interruption":
            if not any(a["name"]==label+"-full.mp4" for a in q["assets"]):missing.append("full video "+label)
        for a in q["assets"]:
            prior=expected.get(a["name"])
            if prior and any(prior[k]!=a[k] for k in ("asset_id","size","digest")):
                raise ValueError("Conflicting local asset receipts for "+a["name"])
            expected[a["name"]]=a
    return sorted(labels),expected,missing


def resolve(archive):
    prereq=["learning/learning-series.json","learning/pre-heldout-frozen.json","mechanism/planning-holdout.json","model/fit-completion.json",
            "action/registry-v3.json","action/sealed-scored.json","learning/final-action-gate.json","mechanism/planning-series.json"]
    if not archive.is_file():
        return "waiting_verified_archive_retirement",[str(archive)]
    a=read(archive)
    if not a.get("manifest_sha256") or not a.get("files"):
        return "waiting_verified_archive_retirement",[str(archive)]
    gaps=[k for k in prereq if not (STATE/k).is_file()]
    if gaps:return "waiting_terminal_science",gaps
    series=read(STATE/prereq[0]);freeze=read(STATE/prereq[1]);controls=read(STATE/prereq[2])
    roster={p.stem:read(p) for p in (STATE/"attempts").glob("*.json")}
    pending=[k for k,row in roster.items() if not row.get("ended_unix_s")]
    pending+=evaluation_ready(series,freeze,controls,roster)
    if pending:return "waiting_earned_evaluations",pending
    reports=models()
    labels,assets,pending=labels_and_gaps(roster,reports)
    if pending:return "waiting_release_receipts",pending
    try:
        info,remote=release.release_assets()
    except (RuntimeError,subprocess.CalledProcessError,URLError) as error:
        return "waiting_remote_release_metadata",[str(error)[:1000]]
    mismatches=[]
    for name,src in assets.items():
        item=remote.get(name,{})
        ok=(item.get('state')=='uploaded' and item.get('id')==src['asset_id'] and item.get('size')==src['size']
            and str(src['digest']).startswith('sha256:') and item.get('digest')==src['digest'] and item.get('browser_download_url')==src['url'])
        if not ok:mismatches.append(name)
    if mismatches:return "remote_asset_verification_pending",mismatches
    gate=read(STATE/'learning/final-action-gate.json');sealed=read(STATE/'action/sealed-scored.json')
    physical=real_cost_and_results(roster)
    heads=ledgers(series,freeze,roster)
    head_trials=sum(x.get("ran",False) for d in series["evaluations"].values() for x in d["heldout"].values())
    observed_status={"passed","mechanism_criteria_unmet","physical_material_failure","physical_failure"}
    control_entries=[v for v in roster.values() if v["task_mode"] in ("short","recovered") and
                     v.get("w59",{}).get("split")=="heldout" and not v.get("w59",{}).get("learner_config")]
    actual_ctrl=len(control_entries)
    learned_natural=sum(t.get("status") in observed_status for d in series["evaluations"].values() for t in d["heldout"].values())
    scientific_outcomes={key:dict(qualification=entry["qualification"],
        heldout={slot:{k:v for k,v in cell.items() if k in ("ran","status","physical_s","reason","attempt")} for slot,cell in entry["heldout"].items()},
        long_qualification=entry["long_original"]) for key,entry in series["evaluations"].items()}
    checked=stamp()
    evidence=dict(status="remote_metadata_checked",release=info["url"],checked_shanghai=checked,labels=labels,
        assets=[dict(name=v["name"],asset_id=v["asset_id"],size=v["size"],digest=v["digest"],url=v["url"]) for _,v in sorted(assets.items())])
    end=dict(status="verified",checked_shanghai=checked,release=info["url"],verified_bundles=len(labels),
        verified_assets=len(assets),verified_bytes=sum(v['size'] for v in assets.values()),remote_proof="delivery/final-assets.json",**BG)
    finals=dict(schema="W59_terminal_scientific_results_v1",finished_shanghai=checked,state="earned_chain_terminal",
        learning_scope=series["state"],learning_stop_reason=series["reason"],matched_four_arms=series["matched"],
        sample_budget_by_seed=freeze["N_s"],learning_heads=heads,qualifications_and_cost=physical,
        earned_evaluations=scientific_outcomes,
        independent_mechanism_counts=dict(learned_attempts=head_trials,learned_natural_terminal=learned_natural,learned_matched_contract=48,
            controls_attempts=actual_ctrl,controls_natural_terminal=sum(v["status"] in observed_status for v in control_entries),
            controls_matched_contract=16,optional_extra_material=68-64,
            added_material_generalization_claim=controls["extra_M3_generalization_claim"]),
        action=dict(development=read(STATE/'action/development-scored.json')["summary"], independent_sealed=sealed["summary"],
            frozen_prediction_substeps=sealed["frozen_planning_substeps"],ranking_admitted=gate["planning_eligible"],
            r3_gradient_admitted=gate["r3_gradient_eligible"],r3_gradient_test_substeps=gate["r3_scored_substeps"],
            true_ppo_admitted=gate["ppo_eligible"],true_action_authority=gate["ppo_action_authority"]),
        model=reports, gait_source="gait/phases",resources_by_total_leased_slots=resource_use(),resource_source="dispatch/mechanism-overlap.json",
        inference=dict(material_history_increment="No full/private history effect was established by the admitted physical fallback.",
           M3_closed_loop_accepted=reports["material"]["accepted_physical"],
           scope="Two fixed Go2/TM02 cases; paired costs report changes at the same complete physical window."),
        publication=end,**BG)
    with (WORK/'collect.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        collect()
        write(STATE/'delivery/final-assets.json',evidence)
        idx=read(STATE/'index.json')
        idx.update(stage="terminal",terminal_science="outcome.json",delivery_receipt="delivery-ended.json",finished_shanghai=checked)
        write(STATE/'index.json',idx)
        log=ROOT/'docs/S1_DAILY_WORKLOG.md'
        text=log.read_text();event=datetime.now(ZONE).strftime('%Y%m%d')+'/w59/terminal-physical-release'
        if event not in text:
            body=(f"### {event}｜W59 科学终局与原件交付\n\n{checked}。独立机制实际完成学习身份 {head_trials} 条、规划对照 {actual_ctrl} 条。"
                  f"真实学习使用 {sum(x['used'] for x in heads.values())} 个不同事件，四臂匹配比较为 {series['matched']}。"
                  f"封存模型排序与路径梯度准入分别为 {gate['planning_eligible']}、{gate['r3_gradient_eligible']}。"
                  f"按 D57 原件逐项核实 GitHub Release 的 {end['verified_assets']} 个资产、{end['verified_bytes']} 字节。逐头节点、资格、未运行原因和配对结果见 w59/outcome.json。\n\n")
            marker='## 以后每天如何追加'
            if marker not in text:raise ValueError("Missing daily worklog insertion anchor")
            if '## '+checked[:10] not in text:body='## '+checked[:10]+'｜W59 执行\n\n'+body
            tmp=log.with_name(log.name+'.w59-close.tmp');tmp.write_text(text.replace(marker,body+marker,1));tmp.replace(log)
        write(STATE/'delivery-ended.json',end)
        write(STATE/'outcome.json',finals)
    return "verified_terminal",[]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-final',type=Path,default=WORK/'archive-process-20261127/D57-final.json')
    p.add_argument('--once',action='store_true')
    a=p.parse_args()
    with (WORK/'closeout.lock').open('a') as l:
        fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if (STATE/'outcome.json').exists() and (STATE/'delivery-ended.json').exists():return
            state,items=resolve(a.archive_final)
            wait_state(state,items)
            if state=="verified_terminal" or a.once:return
            time.sleep(45)


if __name__=="__main__":
    main()
