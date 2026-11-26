"""Prepare and publish head-local execution models only at eligible natural boundaries."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from s1_w59_learner_critic import ROOT, RELEASE, file_digest, json_rows, atomic_json
from s1_w59_learning import batch_file,read_json
from s1_w59_model_fields import STATE, INTERVAL

CHOICES={1024:2048,3072:4096,6144:8192}
RULE=ROOT/"experiments/terrain_memory/s1-w59-online-wm.json"


def launch_if_due(run,state,device):
    scope=read_json(run/"campaign.json")
    target=CHOICES.get(state["used_events"])
    if not target or scope["arm"]=="Diff-fixedWM" or scope.get("nominal_fallback"):return
    root=run/"online"/f"publish-{target:05d}"
    root.mkdir(parents=True,exist_ok=True)
    if (root/"request.json").exists():return
    # Freeze paid samples and their initial WM before the collector keeps going.
    req=dict(seed=scope["seed"],arm=scope["arm"],version=state["version"],used_ceiling=state["used_events"],
             target_node=target,parents=state["wm_checkpoints"],parent_sha256=state["wm_sha256"],
             rule_sha256=file_digest(RULE),state="pending",requested_unix_s=time.time())
    atomic_json(root/"request.json",req)
    log=(root/"trainer.log").open("a")
    try:
        cmd=["taskset","-c","20",sys.executable,"-u",__file__,"--run",str(run),"--node",str(target),"--device",device]
        subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                         start_new_session=True,close_fds=True)
    finally:log.close()


def real_windows(run,req,cache):
    from s1_w59_model_cells import CausalCellMap
    from s1_w59_planning import prepare_private
    from s1_w59_model_rigid import RigidLinearization
    import s1_w59_model_bank as bank
    import s1_w59_model_supervision as labels
    bank.WORK=cache;labels.WORK=cache
    for sub in ("bank","aligned"):(cache/sub).mkdir(parents=True,exist_ok=True)
    ids={key for r in json_rows(run/"used-event-ids.jsonl") if r["version"]<req["version"] for key in r["ids"]}
    by_episode=defaultdict(list)
    for v in range(req["version"]):
        for case in "AB":
            for row in json_rows(batch_file(run,v,case)):
                if row["event_id"] not in ids:raise ValueError("Missing sealed online WM event")
                if (row["arm"],row["seed"],row["sampling"])!=(req["arm"],req["seed"],"on_policy"):
                    raise ValueError("Online execution model received other-arm or excluded-tail data")
                by_episode[row["attempt"]].append(row)
    if len(ids)!=req["used_ceiling"] or sum(map(len,by_episode.values()))!=len(ids):
        raise ValueError("Online model replay would exceed paid macro-event accounting")
    arrays=[];manifest=[];offset=0;starts=[]
    for name,rows in sorted(by_episode.items()):
        p=Path(rows[0]["source_dir"])
        result=ROOT/RELEASE/"episodes"/(name+".json")
        if not result.is_file() or not (p/"trajectory.npz").is_file():
            manifest.append(dict(episode=name,eligible_events=len(rows),windows=0,reason="episode not closed at extraction"))
            continue
        summary=read_json(result)
        if summary.get("error"):
            manifest.append(dict(episode=name,eligible_events=len(rows),windows=0,reason="technical interruption"))
            continue
        row=dict(root=str(p),episode=name,version=3,case=rows[0]["case"],split="arm_local_online_fit")
        af=cache/"aligned"/(name+".npz")
        if not af.exists():
            bank.extract(row);labels.enrich(row)
        with np.load(af) as z:
            x,u,y,w,ts=(z[k] for k in ("state","action","interval","valid","time"))
        n=len(y);paid=np.zeros(n,bool)
        for item in rows:
            # Never allow the leading/trailing partial interval of an unpaid action.
            paid|=(ts[:-1]>=item["start_s"]-1e-8)&(ts[1:]<=item["end_s"]+1e-8)
        if req["arm"]=="Diff-private":
            with np.load(p/"trajectory.npz") as z:feet=z["feet"];grf=z["actual_grf"]
            table=CausalCellMap()
            for tick in range(10*n+1):
                table.observe(tick*.002,feet[tick],grf[tick])
                if tick%10==0:
                    k=tick//10
                    x[k],_,_=prepare_private(x[k],table.table)
        good=np.flatnonzero(np.convolve(paid.astype(int),np.ones(7,int),"valid")==7) if n>=7 else np.empty(0,int)
        if not len(good):
            manifest.append(dict(episode=name,eligible_events=len(rows),windows=0,reason="no fully paid 7-transition support"))
            continue
        # Only materialize rigid maps at valid sequence starts. The fitted member takes
        # the first measured configuration's frozen local linearization.
        engine=RigidLinearization(p/"model.mjb")
        maps=np.zeros((len(x),18,25),np.float32)
        for k in good:maps[k]=engine(x[k])
        starts.extend((good+offset).tolist());offset+=len(x)
        arrays.append((x,u,np.vstack((y,np.zeros_like(y[:1]))),np.vstack((w,np.zeros_like(w[:1]))),maps))
        manifest.append(dict(episode=name,eligible_events=len(rows),windows=int(len(good)),true_20ms_used=int(paid.sum()),
                             first_event=rows[0]["event_id"],last_event=rows[-1]["event_id"]))
    if not starts:raise ValueError("No fully measured within-budget online sequences")
    data=tuple(np.concatenate([d[i] for d in arrays]) for i in range(5))
    return data,np.asarray(starts),manifest


def updated_evidence(run,root,models,scope):
    source=read_json(Path(scope["admission_path"]))
    registry=source["registry_path"];freeze=source["freeze_path"]
    lock=read_json(Path(freeze))
    if lock.get("online_recipe_sha256")!=file_digest(RULE):
        return dict(current_version_eligible=False,reason="Online adaptation was not preregistered in the original recipe freeze")
    common=["--registry",registry,"--role","sealed_admission","--material",scope["material"]]
    p3=root/"rescore-p3.json";r3=root/"rescore-r3.json"
    cmd=[sys.executable,str(ROOT/"scripts/score_s1_w59_interventions.py"),*common,"--output",str(p3)]
    for p in models:cmd.extend(["--wm-checkpoint",str(p)])
    subprocess.run(cmd,cwd=ROOT,check=True)
    cmd=[sys.executable,str(ROOT/"scripts/s1_w59_learner_admission.py"),*common,"--planning",str(p3),
         "--critic",scope["initial_critic_source"],"--freeze",freeze,"--reuse-for-online","--output",str(r3)]
    subprocess.run(cmd,cwd=ROOT,check=True)
    res=read_json(r3)
    ok=res["planning_eligible"] and (scope["arm"]=="PPO-full" or res["r3_gradient_eligible"])
    return dict(current_version_eligible=bool(ok),new_physical_branches=0,independent_new_evidence=False,
                label_source="Reused locked real continuations, predeclared update recipe and no fitting on branch labels",
                calibration_file=str(r3),calibration_sha256=file_digest(r3))


def update(a):
    from s1_w59_model import ExecutionMember
    from s1_w59_model_train import sequence_loss,window
    run=a.run;root=run/"online"/f"publish-{a.node:05d}";scope=read_json(run/"campaign.json")
    req=read_json(root/"request.json")
    if (root/"candidate.json").exists():return
    if scope["arm"]=="Diff-fixedWM":raise ValueError("Fixed-WM head has no online model update")
    try:
        start=time.perf_counter()
        data,starts,source=real_windows(run,req,run/"online"/"recordings")
        atomic_json(root/"sources.json",source)
        lock=(ROOT/"work/w59/pool/learner-gpu.lock").open("a")
        with lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
            tensors=tuple(torch.as_tensor(d,device=a.device) for d in data)
            eligible=torch.as_tensor(starts,device=a.device)
            paths=[];fits=[]
            nsteps={2048:128,4096:128,8192:256}[a.node]
            for index,path in enumerate(req["parents"]):
                if file_digest(Path(path))!=req["parent_sha256"][index]:
                    raise ValueError("Source model bytes differ from the head's published node")
                member=ExecutionMember.load(path,a.device)
                previous=torch.load(path,map_location=a.device,weights_only=False)
                opt=torch.optim.AdamW(member.parameters(),lr=.0003,weight_decay=.0001,
                                      fused=a.device.startswith("cuda"),capturable=a.device.startswith("cuda"))
                opt.load_state_dict(previous["optimizer"])
                rng=torch.Generator(device=a.device).manual_seed(req["seed"]*100000+a.node*3+index)
                begin=time.perf_counter();losses=[]
                for _ in range(nsteps):
                    ids=eligible[torch.randint(len(eligible),(64,),generator=rng,device=a.device)]
                    loss=sequence_loss(member,*window(tensors,ids))
                    opt.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(),1.)
                    opt.step();losses.append(loss.detach())
                final=root/f"member-{index+1}.pt"
                avg=float(torch.stack(losses).mean())
                member.save(final,opt,dict(parent_sha256=req["parent_sha256"][index],seed=req["seed"],
                    arm=req["arm"],published_node_target=a.node,used_event_ceiling=req["used_ceiling"],updates=nsteps))
                fits.append(dict(member=index+1,loss=avg,optimizer_steps=nsteps,batch=64,elapsed_s=time.perf_counter()-begin))
                paths.append(str(final))
            verdict=updated_evidence(run,root,paths,scope)
        atomic_json(root/"candidate.json",dict(status="fit",models=paths,sha256=[file_digest(Path(p)) for p in paths],
                    arm=req["arm"],seed=req["seed"],parent_sha256=req["parent_sha256"],source_windows=len(starts),
                    fits=fits,elapsed_s=time.perf_counter()-start,**verdict))
    except Exception as exc:
        atomic_json(root/"candidate.json",dict(status="failed",reason=repr(exc),current_version_eligible=False))
        raise


def publish_pending(run,state,compact):
    node=state["used_events"];scope=read_json(run/"campaign.json")
    if state["mode"]!="closed_node":
        raise ValueError("WM replacement requires the sealed node and both natural collector tails")
    root=run/"online"/f"publish-{node:05d}"
    file=root/"candidate.json"
    reason="fixed or model-free execution control" if scope["arm"]=="Diff-fixedWM" or scope.get("nominal_fallback") else "candidate has not completed its current-version validation"
    note=dict(node=node,previous_sha256=state["wm_sha256"],published=False,reason=reason)
    if scope["arm"]!="Diff-fixedWM" and file.is_file():
        candidate=read_json(file)
        if candidate.get("current_version_eligible") and candidate["parent_sha256"]==state["wm_sha256"]:
            state=dict(state,wm_checkpoints=candidate["models"],wm_sha256=candidate["sha256"])
            atomic_json(run/"published.json",state)
            note.update(published=True,reason="passed registered current-version measurement",sha256=candidate["sha256"])
        else:note.update(reason=candidate.get("reason","registered validation did not pass"))
    atomic_json(compact/f"model-publication-{node:05d}.json",note)
    return state


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,type=Path);p.add_argument("--node",required=True,type=int)
    p.add_argument("--device",default="cuda");update(p.parse_args())

