"""Run one W59 arm/seed through a common cumulative node on an atomic two-slot lease."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from s1_w59_learner_critic import ROOT, RELEASE, SPEC, file_digest, json_rows, atomic_json, atomic_weights
from s1_w59_learning import append, checkpoint, init_head, read_json, batch_file
from s1_w59_learner_update import batch_update, first_update_verdict, load_batch

ATTEMPTS=ROOT/RELEASE/"attempts"
EPISODES=ROOT/RELEASE/"episodes"


def verify_scope(a):
    gate=read_json(a.admission)
    standalone=gate["role"]=="standalone_true_ppo" and a.role=="fallback-ppo" and a.arm=="PPO-full"
    if (gate["role"]!="sealed_admission" and not standalone) or gate.get("online_reused_actions"):
        raise ValueError("A new learner requires the matching physics admission/standalone allocation")
    if gate["critic_sha256"]!=file_digest(a.critic):
        raise ValueError("Collector critic differs from the critic used in gradient validation")
    models=list(map(lambda p:str(Path(p).resolve()), gate["wm_checkpoints"]))
    sha=[file_digest(Path(p)) for p in models]
    if (len(models)!=0 if standalone else len(models)!=3) or sha!=gate["wm_sha256"]:
        raise ValueError("The scored physical-action model ensemble changed")
    nominal_fallback=a.role=="fallback-ppo" and not gate["planning_eligible"]
    if nominal_fallback:
        if not gate["ppo_eligible"]:raise ValueError("No measured usable three-axis action effect for standalone PPO")
        models,sha=[],[]
    elif not gate["planning_eligible"]:
        raise ValueError("Matched model-ranked planning has not been established")
    if a.role!="fallback-ppo":
        g=gate["gradient_admission"]
        if (not gate["r3_gradient_eligible"] or g["states"]!=12 or g["branches"]!=96 or g["direction_lcb"]<=.5
                or g["mean_predicted_improvement_real_effect"]<=g["mean_nominal_replay_error"]
                or g["new_hard_failures"]>0):
            raise ValueError("Independent action calibration has not admitted model policy gradients")
    elif a.arm!="PPO-full":
        raise ValueError("A real-data fallback requires its own explicit PPO identity")
    registry=read_json(a.first_states)
    ids={r["snapshot_id"] for r in registry["states"]}
    if len(ids)!=8 or len(registry["states"])!=8 or ids.intersection(gate["calibration_snapshot_ids"]):
        raise ValueError("Register eight legal states disjoint from action development and admission")
    if registry["seed"]!=a.seed or gate["material"] not in ("M1","M3"):
        raise ValueError("Frozen material and first-check seed must match the campaign")
    names=["s1_w59_learning.py","s1_w59_learner_critic.py","s1_w59_learner_update.py","s1_w59_learner_campaign.py",
           "s1_w59_learner_online.py"]
    return dict(admission_path=str(a.admission.resolve()), planning_recipe=gate["freeze_path"],
        planning_substeps=gate["planning_substeps"],online_rule_sha256=file_digest(ROOT/"experiments/terrain_memory/s1-w59-online-wm.json"),
        initial_wm=models,initial_wm_sha256=sha,material="M1" if nominal_fallback else gate["material"],
        nominal_fallback=nominal_fallback,
        admission_sha256=file_digest(a.admission),first_states_sha256=file_digest(a.first_states),
        recipe_sha256=file_digest(ROOT/"experiments/terrain_memory/s1-w59-learner.json"),
        sources={p:file_digest(ROOT/"scripts"/p) for p in names},science_role=a.role,
        gradient_admission=gate.get("gradient_admission"),first_states=str(a.first_states.resolve()),
        first_replay_floor=float(gate["mean_nominal_2s_return_replay_error"]))


def leased_groups():
    groups=json.loads(os.environ["S1_W59_PAIR_CPUS"])
    fds=tuple(int(n) for n in os.environ["S1_W58_LEASE_FDS"].split(","))
    if len(groups)!=2 or len(fds)!=2 or set(groups[0])&set(groups[1]):
        raise ValueError("The dual campaign requires two already acquired disjoint physical slots")
    return groups,fds


class Campaign:
    def __init__(self,a):
        self.a=a
        scope=verify_scope(a)
        self.scope=scope
        suffix=a.arm if a.role=="matched" else "PPO-full-fallback"
        self.run=a.runs/f"seed-{a.seed}"/suffix
        self.compact=ROOT/RELEASE/"learning"/f"seed-{a.seed}"/suffix
        self.compact.mkdir(parents=True,exist_ok=True)
        init_head(self.run,a.arm,a.seed,a.critic,scope)
        self.ledger=self.run/"physics-attempts.jsonl"
        self.ends=self.run/"physics-ends.jsonl"
        self.groups,self.fds=leased_groups()
        self.children={}
        self.configs={}
        self.prefix=f"w59-r3-{a.seed}-{suffix.lower()}"
        history=json_rows(self.ledger) if self.ledger.exists() else []
        ended=json_rows(self.ends) if self.ends.exists() else []
        closed={r["attempt"] for r in ended}
        for row in history:
            if row["attempt"] not in closed:
                receipt=ATTEMPTS/(row["attempt"]+".json")
                if not receipt.is_file() or read_json(receipt)["status"] in ("running","declared"):
                    raise RuntimeError(f"Exact unfinished process or receipt requires reconciliation: {row['attempt']}")
                self.record_end(row["attempt"],row["case"],read_json(receipt).get("exit_code",1))
        self.episodes=Counter(r["case"] for r in history if r["task"]=="training")
        for case in "AB":
            path=self.run/f"collector-{case}.json"
            atomic_json(path,dict(mode="collect",run_dir=str(self.run.resolve()),case=case,
                                  head=a.arm,seed=a.seed,science_role=a.role,nominal_fallback=scope["nominal_fallback"],
                                  learning_sources=scope["sources"],recipe=scope["recipe_sha256"]))
            self.configs[case]=path

    def publication(self,**changes):
        path=self.run/"published.json"
        state=dict(read_json(path),**changes)
        atomic_json(path,state)
        return state

    def launch(self,case,attempt,task,config,models,slot,branch=None):
        if (ATTEMPTS/(attempt+".json")).exists():
            raise FileExistsError(f"Existing W59 physical identity {attempt}")
        arm="B3" if self.scope["nominal_fallback"] else ("P3-private" if self.a.arm=="Diff-private" else "P3-full")
        cmd=["taskset","-c",",".join(map(str,self.groups[slot])),sys.executable,"-u",
             str(ROOT/"scripts/run_s1_w59_research.py"),"--arm",arm,"--material",self.scope["material"],
             "--case",case,"--level","E30" if task=="branch" else "E60","--task",task,
             "--split","development" if task=="branch" else "training","--attempt",attempt,"--learner-config",str(config)]
        if self.scope["planning_recipe"]:cmd.extend(["--wm-recipe",self.scope["planning_recipe"]])
        for model in models:
            cmd.extend(["--wm-checkpoint",str(model)])
        if branch:
            cmd.extend(["--branch",str(branch)])
        with (self.run/f"{attempt}.log").open("a") as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=self.fds)
        append(self.ledger,dict(case=case,attempt=attempt,task=task,cmd=cmd,pid=proc.pid,slot=slot,unix_s=time.time()))
        return proc

    def record_end(self,attempt,case,code):
        receipt=ATTEMPTS/(attempt+".json")
        event=read_json(receipt) if receipt.is_file() else {}
        append(self.ends,dict(attempt=attempt,case=case,exit_code=code,status=event.get("status"),
                              physical_s=event.get("simulation_s"),reason=event.get("reason"),unix_s=time.time()))
        if event.get("task_mode")=="training" and read_json(self.run/"published.json")["mode"]=="acquire":
            episode=EPISODES/(attempt+".json")
            if episode.exists() and not read_json(episode).get("w59_learning",{}).get("used_barrier_rows"):
                raise RuntimeError(f"Original virgin E60 had no new sealed-eligible events: {attempt}")
        if code or event.get("status") in (None,"running","declared","technical_interruption"):
            raise RuntimeError(f"Physical attempt {attempt} ended without a natural result: {event.get('reason')}")

    def finish_collector(self,case):
        identifier,proc=self.children.pop(case)
        self.record_end(identifier,case,proc.wait())

    def drain(self,mode):
        # A free slot can only be reused after its already-running process reaches a natural tail.
        self.publication(mode=mode)
        for case in list(self.children):
            self.finish_collector(case)

    def collect_batch(self,state):
        version=state["version"]
        while True:
            done=[]
            for index,case in enumerate("AB"):
                child=self.children.get(case)
                if child and child[1].poll() is not None:
                    self.finish_collector(case)
                    child=None
                ready=(self.run/f"ready-{version:03d}-{case}.json").is_file()
                done.append(ready)
                if child or ready:
                    continue
                self.episodes[case]+=1
                num=self.episodes[case]
                key=f"{self.prefix}-{case.lower()}-e60-{num:04d}"
                proc=self.launch(case,key,"training",self.configs[case],state["wm_checkpoints"],index)
                self.children[case]=(key,proc)
            if all(done):return
            time.sleep(.5)

    def first_branches(self,version,proposal,initial):
        record=self.run/f"first-update-request-{version+1:03d}.json"
        registry=read_json(self.a.first_states)
        scope=read_json(self.run/"published.json")
        entries=[]
        for i,source in enumerate(registry["states"]):
            for role,path in [("before",checkpoint(self.run,version)),("after",proposal)]:
                sha=file_digest(path)
                actor_cfg=self.run/f"frozen-{role}-{version:03d}.json"
                atomic_json(actor_cfg,dict(mode="evaluate",head=self.a.arm,seed=self.a.seed,
                         checkpoint=str(path),science_role=self.a.role,nominal_fallback=self.scope["nominal_fallback"],learning_sources=self.scope["sources"],
                         recipe=self.scope["recipe_sha256"]))
                branch=dict(source,role=role,seed=self.a.seed,actor_digest=sha,duration_s=2.,replacement=None)
                branch_path=self.run/f"first-{i+1:02d}-{role}.json"
                atomic_json(branch_path,branch)
                key=f"{self.prefix}-first-{i+1:02d}-{role}"
                entries.append(dict(case=source["case"],key=key,branch=branch_path,config=actor_cfg))
        atomic_json(record,dict(registry_sha256=file_digest(self.a.first_states),before_sha256=initial,
                                proposed_sha256=file_digest(proposal),jobs=[x["key"] for x in entries],
                                freeze_unix_s=time.time(),branches=16,simulated_seconds_ceiling=32))
        self.drain("first_check_tail")
        results=[]
        # The same atomic two-slot lease is now empty. The system never runs extra collectors
        # beside these two branch processes, even with three simultaneously active campaigns.
        for start in range(0,len(entries),2):
            jobs=[]
            for slot,entry in enumerate(entries[start:start+2]):
                result_file=EPISODES/(entry["key"]+".json")
                if result_file.exists():
                    results.append(result_file)
                    continue
                proc=self.launch(entry["case"],entry["key"],"branch",entry["config"],scope["wm_checkpoints"],slot,entry["branch"])
                jobs.append((entry,proc,result_file))
            errors=[]
            for entry,proc,path in jobs:
                try:
                    self.record_end(entry["key"],entry["case"],proc.wait())
                    results.append(path)
                except RuntimeError as exc:
                    errors.append(str(exc))
            if errors:raise RuntimeError(" | ".join(errors))
        verdict=first_update_verdict(results,registry,initial,file_digest(proposal),self.scope["first_replay_floor"])
        atomic_json(self.compact/"first-update-real.json",verdict)
        return verdict

    def fit(self,state):
        version=state["version"];next_version=version+1
        path=self.run/f"proposal-{next_version:03d}.pt";mp=self.run/f"proposed-{next_version:03d}.json"
        if path.is_file() and mp.is_file():
            metrics=read_json(mp); _,rows=load_batch(self.run,version); keys=[r["event_id"] for r in rows]
        else:
            with (ROOT/"work/w59/pool/learner-gpu.lock").open("a") as device_lock:
                fcntl.flock(device_lock,fcntl.LOCK_EX)
                path,metrics,keys=batch_update(self.run,version,self.a.device)
        proposed=torch.load(path,map_location="cpu",weights_only=True)
        rejected=False
        if metrics["first_actor_direction_needed"]:
            old=checkpoint(self.run,version)
            check=self.first_branches(version,path,file_digest(old))
            rejected=not check["accepted"]
            if rejected:
                # Revert both parameter sets and both optimizers; the used sealed data and
                # attempted computation are still charged to this head and never recycled.
                proposed=torch.load(old,map_location="cpu",weights_only=True)
                proposed.update(version=next_version,used_events=metrics["cumulative_used"],
                                verified_first_update=False)
            else:
                proposed["verified_first_update"]=True
        metrics["first_update_rejected"]=rejected
        metrics["actual_accepted_actor_steps"]=0 if rejected else metrics["actor_steps_accepted_by_KL"]
        metrics["actual_parameter_change_l2"]=0. if rejected else metrics["delta_parameter_l2"]
        metrics["actual_final_KL"]=0. if rejected else metrics["final_KL"]
        installed=checkpoint(self.run,next_version)
        atomic_weights(installed,proposed)
        event_index=self.run/"used-event-ids.jsonl"
        previous=[r for r in (json_rows(event_index) if event_index.exists() else []) if r["version"]==version]
        if previous:
            if len(previous)!=1 or previous[0]["ids"]!=keys:raise ValueError("Conflicting event ledger for sealed version")
        else:append(event_index,dict(version=version,ids=keys))
        atomic_json(self.compact/f"update-{next_version:03d}.json",metrics)
        n=256*next_version
        node=n in SPEC["nodes"]
        mode="rejected_tail" if rejected else "node_tail" if node else "acquire"
        state=self.publication(version=next_version,mode=mode,used_events=n,checkpoint=str(installed))
        if rejected or node:
            self.drain(mode)
            state=self.publication(mode="rejected" if rejected else "closed_node")
            self.node_report(state,rejected)
        if state["mode"]=="acquire":
            from s1_w59_learner_online import launch_if_due
            launch_if_due(self.run,state,self.a.device)
        return state

    def node_report(self,state,rejected):
        files=sorted(self.compact.glob("update-*.json"))
        updates=[read_json(p) for p in files]
        # All actual episode events are read after natural tails. Traces from direction checks
        # are opt-in physical branches and never counted as collector learning samples.
        launches=json_rows(self.ledger)
        completed={r["attempt"]:r for r in json_rows(self.ends)}
        physical=Counter(); counts=Counter(); cases=Counter()
        for ep in launches:
            name=ep["attempt"];kind=ep["task"]
            if name not in completed:continue
            physical[kind]+=float(completed[name]["physical_s"] or 0) if kind=="training" else 0.
            path=EPISODES/(name+".json")
            if path.is_file():
                row=read_json(path)
                if kind=="branch":physical[kind]+=row["branch"]["elapsed_s"]
                else:
                    stats=row["w59_learning"]
                    for k in ("requested","completed","used_barrier_rows","tail_excluded","data_cut"):counts[k]+=stats[k]
                    cases[ep["case"]]+=stats["used_barrier_rows"]
        n=state["used_events"]
        report=dict(arm=self.a.arm,seed=self.a.seed,science_role=self.a.role,used_events=n,
            observed_completed=counts["completed"],on_policy_rows=counts["used_barrier_rows"],by_case=dict(cases),
            excluded_natural_tail=counts["tail_excluded"],requested=counts["requested"],technical_unclosed=counts["data_cut"],
            physical_seconds=dict(physical),imagined_20ms_transitions=sum(u["imagined_substeps"] for u in updates),
            verified_actor_steps=sum(u["actual_accepted_actor_steps"] for u in updates),rejected_first_update=rejected,
            complete_common_node=(not rejected and n in SPEC["nodes"]),checkpoint=state["checkpoint"],
            checkpoint_sha256=file_digest(Path(state["checkpoint"])),wm_sha256=state["wm_sha256"],unix_s=time.time(),
            background=SPEC["background"])
        atomic_json(self.compact/(f"node-{n:05d}.json" if not rejected else f"stopped-{n:05d}.json"),report)
        if not rejected:
            atomic_json(self.run/f"freeze-{n:05d}.json",dict(mode="evaluate",head=self.a.arm,seed=self.a.seed,
                checkpoint=state["checkpoint"],science_role=self.a.role,nominal_fallback=self.scope["nominal_fallback"],learning_sources=self.scope["sources"],
                recipe=self.scope["recipe_sha256"]))

    def charge_head(self):
        # All trees share work/w59 through the common D57 execution directory.
        ledger=ROOT/"work/w59/pool/learning-heads.json"
        lock=ledger.with_suffix(".lock")
        key=f"{self.a.seed}/{self.a.arm}"
        with lock.open("a") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            record=read_json(ledger) if ledger.exists() else {}
            identity=dict(arm=self.a.arm,seed=self.a.seed,role=self.a.role,run=str(self.run.resolve()),
                initial_model=self.scope["initial_wm_sha256"],admission=self.scope["admission_sha256"])
            if key in record and record[key]!=identity:
                raise ValueError("A previously registered arm/seed already owns this learning allocation")
            record[key]=identity
            if len(record)>12:raise ValueError("The W59 contract has twelve head allocations")
            atomic_json(ledger,record)

    def execute(self):
        self.charge_head()
        state=read_json(self.run/"published.json")
        n=state["used_events"]
        if n==self.a.node and state["mode"]=="closed_node":return
        if state["mode"] in ("rejected","incomplete"):
            raise RuntimeError(f"Stopped campaign {self.run} is {state['mode']}")
        previous={2048:0,4096:2048,8192:4096}[self.a.node]
        if n<previous or n>=self.a.node:
            raise ValueError("The requested cumulative node is not adjacent to this head's actual ledger")
        if n and state["mode"]=="closed_node":
            from s1_w59_learner_online import publish_pending
            state=publish_pending(self.run,state,self.compact)
        if not self.a.node==2048 and not self.a.common_decision:
            raise ValueError("A later common node requires the pre-holdout Diff-full development decision")
        if self.a.common_decision:
            gate=read_json(self.a.common_decision)
            if gate["seed"]!=self.a.seed or self.a.node not in gate["authorized_nodes"] or gate["heldout_used"]:
                raise ValueError("Invalid common matched-budget decision")
        state=self.publication(mode="acquire")
        while state["used_events"]<self.a.node:
            self.collect_batch(state)
            state=self.fit(state)
            if state["mode"]=="rejected":
                break


def main(a):
    torch.set_num_threads(1)
    lane=Campaign(a)
    lock=(lane.run/"learner.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        lane.execute()
    except Exception as exc:
        lane.publication(mode="incomplete",error=repr(exc))
        for case in list(lane.children):
            lane.finish_collector(case)
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=SPEC["arms"],required=True)
    p.add_argument("--seed",type=int,choices=SPEC["seeds"],required=True)
    p.add_argument("--node",type=int,choices=SPEC["nodes"],required=True)
    p.add_argument("--role",choices=("matched","fallback-ppo"),default="matched")
    p.add_argument("--admission",type=Path,required=True)
    p.add_argument("--first-states",type=Path,required=True)
    p.add_argument("--common-decision",type=Path)
    p.add_argument("--critic",type=Path,default=ROOT/"work/w59/learner/historical-task-critic.pt")
    p.add_argument("--runs",type=Path,default=ROOT/"work/w59/learning")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    main(p.parse_args())

