"""Match the four W59 acquisition arms, then run each eligible frozen head's originals and holdouts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from s1_w59_learning import read_json, atomic_json
from s1_w59_learner_critic import ROOT, RELEASE, SPEC, file_digest, json_rows

OUT=ROOT/RELEASE
RAW=ROOT/"outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12"
ARMS=SPEC["arms"];NODES=SPEC["nodes"]


def parallel(fun,jobs,limit):
    items=list(jobs)
    if not items:return []
    with ThreadPoolExecutor(max_workers=limit) as workers:
        return list(workers.map(fun,items))


class Series:
    def __init__(self,a):
        self.a=a;self.gate=read_json(a.admission)
        self.root=ROOT/"work/w59/learning"
        self.dest=OUT/"learning"
        self.role="matched" if self.gate["r3_gradient_eligible"] else "fallback-ppo"
        self.arms=ARMS if self.role=="matched" else ["PPO-full"]
        self.budgets={};self.denominators={};self.evaluations={}

    def suffix(self,arm):
        return arm if self.role=="matched" else "PPO-full-fallback"

    def run_path(self,seed,arm):
        return self.root/f"seed-{seed}"/self.suffix(arm)

    def compact(self,seed,arm):
        return self.dest/f"seed-{seed}"/self.suffix(arm)

    def leases(self,name,slots,cmd):
        root=self.a.runtime
        command=[sys.executable,str(ROOT/"scripts/s1_w59_lease.py"),"--runtime",str(root),
                 "--slots",str(slots),"--job",name,"--",*cmd]
        code=subprocess.run(command,cwd=root).returncode
        rec=ROOT/"work/w59/pool/jobs"/(name+".json")
        if not rec.is_file():return code
        while True:
            state=read_json(rec)
            if state["state"]=="ended":return state["exit_code"]
            owner=state["owner_pid"]
            if not Path(f"/proc/{owner}").exists():
                raise RuntimeError(f"Unclosed physical lease {name}: recorded owner PID {owner} is absent")
            # A previous exact identity already owns this job. Follow its actual lease to its natural end.
            time.sleep(5)

    def planning_development(self):
        # A strict target miss is kept as an observed outcome. A missing or technically
        # incomplete controller must not be called a finished four-arm comparison.
        if self.role=="fallback-ppo" and not self.gate["planning_eligible"]:
            for case in "ab":
                for level in ("e30","e60"):
                    paths=(OUT/"episodes").glob(f"w59-b3-{case}-original-{level}-a[0-9][0-9].json")
                    if not any(read_json(p)["stage_status"]=="passed" for p in paths):
                        raise ValueError(f"Missing B3 nominal fallback original qualification {case}/{level}")
            return
        controls=("B3","Fixed-action","P3-full","P3-private") if self.role=="matched" else ("P3-full",)
        for arm in controls:
            material="M1" if arm in ("B3","Fixed-action") else self.gate["material"]
            path=OUT/"mechanism"/f"{arm.lower()}-{material.lower()}-development.json"
            if not path.is_file():raise ValueError(f"Missing actual development comparison {path}")
            row=read_json(path)
            if row["denominator"]!=4 or len(row["episodes"])!=4:raise ValueError(f"Incomplete denominator at {path}")
            for item in row["episodes"]:
                ep=read_json(OUT/"episodes"/(item["attempt"]+".json"))
                if ep.get("error") or ep.get("stage_status")=="technical_interruption":
                    raise ValueError(f"Technical development interruption {item['attempt']}")

    def run_arm(self,job):
        seed,arm,node=job;run=self.run_path(seed,arm)
        final=self.compact(seed,arm)/f"node-{node:05d}.json"
        if final.exists():return read_json(final)
        if (run/"published.json").exists() and read_json(run/"published.json")["mode"] in ("rejected","incomplete"):
            return {"complete_common_node":False,"reason":read_json(run/"published.json")["mode"]}
        cmd=["taskset","-c","20",sys.executable,"-u","scripts/s1_w59_learner_campaign.py","--arm",arm,
             "--seed",str(seed),"--node",str(node),"--admission",str(self.a.admission),"--role",self.role,
             "--first-states",str(self.a.first_states/f"seed-{seed}.json"),"--critic",str(self.a.critic)]
        dec=self.dest/f"common-budget-{seed}.json"
        if node>2048:cmd.extend(["--common-decision",str(dec)])
        rc=self.leases(f"w59-learning-{seed}-{self.suffix(arm).lower()}-{node:05d}",2,cmd)
        if final.is_file():return read_json(final)
        state=read_json(run/"published.json") if (run/"published.json").exists() else {}
        return dict(complete_common_node=False,reason=state.get("error",state.get("mode","no valid head receipt")),
                    actual_used=state.get("used_events",0),exit_code=rc)

    def node_decision(self,seed,node,reference):
        # Decisions come from Diff-full training, never from matched controls or held-out episodes.
        arm="Diff-full" if self.role=="matched" else "PPO-full"
        run=self.run_path(seed,arm)
        reason=[];by_case={c:dict(n=0,success=0,hard_failure=0) for c in "AB"}
        if (run/"physics-attempts.jsonl").is_file():
            for a in json_rows(run/"physics-attempts.jsonl"):
                if a["task"]!="training":continue
                path=OUT/"episodes"/(a["attempt"]+".json")
                if not path.is_file():continue
                ep=read_json(path);q=by_case[a["case"]]
                q["n"]+=1;q["success"]+=int(ep["stage_status"]=="passed")
                q["hard_failure"]+=int(ep["stage_status"]=="physical_material_failure")
        if not reference.get("complete_common_node"):reason.append("reference did not complete this node")
        if reference.get("verified_actor_steps",0)<=0:reason.append("no independently accepted actor update")
        if any(v["success"]<1 or v["hard_failure"] for v in by_case.values()):
            reason.append("a reference case lacks an original training task completion or has an added hard failure")
        allowed=not reason and node<8192
        decision=dict(seed=seed,node=node,source=arm,role=self.role,heldout_used=False,original_training=by_case,
            allow_next=allowed,reason=reason or (["highest registered node"] if node==8192 else []),
            authorized_nodes=NODES[:NODES.index(node)+1+int(allowed)],source_report=reference,
            recipe_sha256=file_digest(ROOT/"experiments/terrain_memory/s1-w59-learner-nodes.json"))
        atomic_json(self.dest/f"decision-{seed}-{node:05d}.json",decision)
        atomic_json(self.dest/f"common-budget-{seed}.json",decision)
        return decision

    def cohort(self,seed,nodes=None):
        selected=self.budgets.get(seed);go=False
        if not (self.a.first_states/f"seed-{seed}.json").is_file():
            self.budgets[seed]=None
            self.denominators[f"{seed}/all"]=dict(reason="Missing independent eight-state first-update registration")
            return False
        for node in (nodes or NODES):
            results=parallel(self.run_arm,[(seed,h,node) for h in self.arms],3)
            for h,r in zip(self.arms,results):
                self.denominators[f"{seed}/{h}/{node}"]=r
            ref=results[self.arms.index("Diff-full" if self.role=="matched" else "PPO-full")]
            dec=self.node_decision(seed,node,ref)
            if ref.get("complete_common_node"):selected=node
            if node==2048:go=dec["allow_next"]
            if not dec["allow_next"]:break
        self.budgets[seed]=selected
        return go

    def frozen(self,seed,arm,node):
        run=self.run_path(seed,arm)
        path=run/f"freeze-{node:05d}.json"
        report=self.compact(seed,arm)/f"node-{node:05d}.json"
        if not path.is_file() or not report.is_file() or not read_json(report)["complete_common_node"]:return None
        f=read_json(path);ck=Path(f["checkpoint"])
        if file_digest(ck)!=read_json(report)["checkpoint_sha256"]:raise ValueError(str(ck))
        # This is the model which actually collected at the frozen actor's node; a pending
        # next-node model is not silently paired with a different policy or evaluation.
        sha=read_json(report)["wm_sha256"]
        initial=read_json(run/"campaign.json")
        candidates=[(initial["initial_wm_sha256"],initial["initial_wm"])]
        for p in (run/"online").glob("publish-*/candidate.json"):
            candidate=read_json(p)
            if candidate.get("current_version_eligible"):candidates.append((candidate["sha256"],candidate["models"]))
        models=next(m for s,m in candidates if s==sha)
        return dict(actor_config=str(path),actor_sha256=file_digest(ck),wm=models,wm_sha256=sha,
                    seed=seed,arm=arm,node=node,algo_identity=self.role)

    def physical(self,info,case,level,mode):
        name=f"w59-{self.suffix(info['arm']).lower()}-{info['seed']}-{case.lower()}-{mode}-{level.lower()}-a01"
        receipt=OUT/"episodes"/(name+".json")
        if not receipt.exists():
            base_nominal=self.role=="fallback-ppo" and not self.gate["planning_eligible"]
            arm="B3" if base_nominal else ("P3-private" if info["arm"]=="Diff-private" else "P3-full")
            cmd=[sys.executable,"-u","scripts/run_s1_w59_research.py","--arm",arm,"--material","M1" if base_nominal else self.gate["material"],
                "--case",case,"--level",level,"--task",mode,"--split","original" if mode=="original" else "heldout",
                "--learner-config",info["actor_config"],"--attempt",name]
            if self.gate["freeze_path"]:cmd.extend(["--wm-recipe",self.gate["freeze_path"]])
            for p in info["wm"]:cmd.extend(["--wm-checkpoint",p])
            self.leases(name,1,cmd)
        if not receipt.exists():return dict(attempt=name,ran=False,reason="no final original receipt")
        ep=read_json(receipt)
        command=read_json(RAW/"jobs"/name/"execution-contract.json")
        attached=command["w59"]["learner_config"]
        if (file_digest(Path(attached["checkpoint"]))!=info["actor_sha256"]
                or command["w59"]["model_sha256"]!=info["wm_sha256"]):
            raise ValueError(f"Head/WM mismatch at {name}")
        return dict(attempt=name,ran=True,status=ep["stage_status"],physical_s=ep["simulation_s"],
            success=ep.get("success"),case_key=ep["case_key"],mechanism=ep.get("mechanism"))

    def qualify(self,job):
        info,case=job
        q30=self.physical(info,case,"E30","original")
        q60=self.physical(info,case,"E60","original") if q30.get("status")=="passed" else None
        return case,dict(E30=q30,E60=q60)

    def cost(self,attempt):
        path=OUT/"gait/new"/(attempt+".json")
        if not path.is_file():
            from s1_w59_contact import score
            ep=read_json(OUT/"episodes"/(attempt+".json"))
            atomic_json(path,score(RAW/attempt,ep))
        return read_json(path)

    def eval_head(self,info):
        hold={}
        qualification=dict(parallel(self.qualify,[(info,c) for c in "AB"],2))
        ready=all(v["E60"] and v["E60"].get("status")=="passed" for v in qualification.values())
        if ready:
            keys=[(c,m) for c in "AB" for m in ("short","recovered")]
            obs=parallel(lambda p:self.physical(info,p[0],"E60",p[1]),keys,4)
            hold={c+"/"+m:r for (c,m),r in zip(keys,obs)}
        else:
            hold={c+"/"+m:dict(ran=False,reason="Both frozen A/B original E30 and E60 are required")
                  for c in "AB" for m in ("short","recovered")}
        long={}
        if ready and info["seed"]==5901:
            for case in "AB":
                observed=[hold[case+"/"+m] for m in ("short","recovered")]
                from s1_w59_geometry import mechanism_spec
                windows=mechanism_spec("heldout")["observation_s"]
                exposure=all((r.get("mechanism") or {}).get("achieved",0)>0 and r.get("status") in
                    ("passed","mechanism_criteria_unmet") and r["physical_s"]>=windows[m]-1e-7
                    for r,m in zip(observed,("short","recovered")))
                candidate=self.cost(qualification[case]["E60"]["attempt"])
                reference=self.cost(f"w59-b3-{case.lower()}-original-e60-a01")
                worse=all(candidate[k]>reference[k] for k in ("moving_positive_CoT","moving_point_slip_per_m"))
                if exposure and not worse:
                    e300=self.physical(info,case,"E300","original")
                    e600=self.physical(info,case,"E600","original") if e300.get("status")=="passed" else None
                else:e300=e600=None
                long[case]=dict(E300=e300,E600=e600,target_exposure=exposure,both_costs_deteriorated=worse)
        result=dict(**info,qualification=qualification,heldout_denominator=4,heldout=hold,long_original=long)
        atomic_json(self.compact(info["seed"],info["arm"])/"frozen-outcome.json",result)
        return result

    def execute(self):
        if not self.gate["planning_eligible"] and not self.gate["ppo_eligible"]:
            return self.finish("No usable true action effect or qualified model choice was established")
        self.planning_development()
        go=self.cohort(5901,[2048])
        if go:
            # Once 5901 establishes the common initial milestone, later seeds start without
            # waiting for 5901 to exhaust 8192. Leases enforce three dual campaigns globally.
            with ThreadPoolExecutor(max_workers=3) as workers:
                pending=[workers.submit(self.cohort,5901,[4096,8192]),
                         workers.submit(self.cohort,5902),workers.submit(self.cohort,5903)]
                for task in pending:task.result()
        frozen=[]
        for seed,node in sorted(self.budgets.items()):
            if node is None:continue
            for arm in self.arms:
                info=self.frozen(seed,arm,node)
                if info:frozen.append(info)
        atomic_json(self.dest/"pre-heldout-frozen.json",dict(role=self.role,display_seed=5901,N_s=self.budgets,
            selections=frozen,heldout_protocol_sha256=file_digest(ROOT/"experiments/terrain_memory/s1-w59-footprints.json")))
        for info in frozen:
            self.evaluations[f"{info['seed']}/{info['arm']}"]=self.eval_head(info)
        return self.finish("Eligible training, frozen qualification, heldout and conditional long envelopes ended")

    def finish(self,reason):
        attempts={}
        for seed in SPEC["seeds"]:
            for h in ARMS:
                run=self.run_path(seed,h) if h in self.arms else None
                path=(run/"published.json") if run else None
                if path and path.exists():
                    s=read_json(path);attempts[f"{seed}/{h}"]=dict(role=self.role,used=s["used_events"],mode=s["mode"])
                else:attempts[f"{seed}/{h}"]=dict(role=self.role,used=0,mode="unrun")
        out=dict(gate=file_digest(self.a.admission),reason=reason,matched=self.role=="matched",
            common_N_s=self.budgets,heads=attempts,nodes=self.denominators,evaluations=self.evaluations,
            nominal_matched_heldout_denominator=48,
            actual_learned_heldout_attempts=sum(bool(t.get("ran")) for r in self.evaluations.values() for t in r["heldout"].values()),
            actual_learned_heldout_natural=sum(t.get("status") in ("passed","mechanism_criteria_unmet","physical_material_failure")
                for r in self.evaluations.values() for t in r["heldout"].values()),
            independent_learned_trials_planned_per_head=4,
            display_seed=5901,background=SPEC["background"])
        atomic_json(self.dest/"learning-series.json",out)
        return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime",required=True,type=Path)
    p.add_argument("--admission",required=True,type=Path)
    p.add_argument("--first-states",required=True,type=Path)
    p.add_argument("--critic",type=Path,default=ROOT/"work/w59/learner/historical-task-critic.pt")
    a=p.parse_args();result=Series(a).execute()
    print(json.dumps({k:v for k,v in result.items() if k not in ("heads","nodes","evaluations")}),flush=True)

