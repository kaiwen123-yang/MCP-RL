"""Version-balanced W61 nine-head study. Each paired real-physics batch takes an atomic lease."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from collect import BASE, BASE_REL, SOURCE_REL
from develop import admitted_call, COLLECT
from hybrid import CACHE, CONFIG, RESULT, ROOT, SPEC, new_heads, weight_digest
from s1_w59_learner_critic import atomic_json, atomic_weights, file_digest
from update import OPT, RESPONSE_CONFIG, update

STUDY="e61-hybrid-m1-01"
METHODS=SPEC["methods"]
CASES=SPEC["scenarios"]
BATCH=SPEC["batch_per_case"]
TOTAL=SPEC["used_per_head"]
RUN=CACHE/STUDY
SMALL=RESULT/STUDY
SELF=Path(__file__).resolve()
BUILD_FILES=sorted(SELF.parent.glob("*.py"))


def read(path):
    return json.loads(path.read_text())


def head_name(method,seed):
    return f"{method.lower()}-s{seed}"


def head_runtime(method,seed):
    return RUN/head_name(method,seed)


def head_result(method,seed):
    return SMALL/head_name(method,seed)


def batch_rows(folder):
    events=folder/"events.jsonl"
    return [json.loads(row) for row in events.read_text().splitlines()] if events.exists() else []


def source_identities():
    sources={str(p.relative_to(ROOT)):file_digest(p) for p in [CONFIG,RESPONSE_CONFIG,*BUILD_FILES]}
    for case in CASES:
        ref=f"w59-b3-{case.lower()}-original-e30-a01"
        original=BASE/SOURCE_REL/"jobs"/ref/"execution-contract.json"
        frozen_runtime=BASE/SOURCE_REL/"jobs"/f"w59-r3-5902-ppo-full-fallback-{case.lower()}-e60-0001"/"execution-contract.json"
        sources[str(frozen_runtime)]=file_digest(frozen_runtime)
        for name,expected in read(frozen_runtime)["w57_execution_sources"].items():
            observed=file_digest(ROOT/name)
            if observed!=expected:
                raise ValueError(f"Original physical dependency changed: {name}")
            sources[name]=expected
        parent=read(BASE/BASE_REL/"w56/f"/f"F-{case}-r60/job.json")
        p=BASE/parent["checkpoint"]
        if file_digest(p)!=parent["checkpoint_sha256"]:
            raise ValueError(f"W44 {case} original checkpoint differs")
        sources[str(p)]=parent["checkpoint_sha256"]
        for p in [original,BASE/BASE_REL/"w59/episodes"/(ref+"-resolved.json"),BASE/SOURCE_REL/ref/"model/G1-project-go2.xml"]:
            sources[str(p)]=file_digest(p)
    return sources


def prepare(method,seed):
    directory=head_runtime(method,seed)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/"policy-000.pt"
    if path.exists():
        return
    actor,critic=new_heads(seed)
    actopt=torch.optim.Adam(actor.parameters(),lr=OPT["learning_rate"])
    critopt=torch.optim.Adam(critic.parameters(),lr=OPT["learning_rate"])
    atomic_weights(path,dict(study=STUDY,method=method,seed=seed,version=0,used=0,
       actor_sha=weight_digest(actor),critic_sha=weight_digest(critic),actor=actor.state_dict(),critic=critic.state_dict(),
       actor_optimizer=actopt.state_dict(),critic_optimizer=critopt.state_dict(),per_case_cursor={k:0 for k in CASES},
       rng_schema="BLAKE2b(seed,role,case,version,episode,request); immutable per-request full key/state hash"))
    atomic_json(head_result(method,seed)/"initial.json",dict(study=STUDY,method=method,seed=seed,version=0,used=0,
        checkpoint=str(path),sha256=file_digest(path),actor=weight_digest(actor),critic=weight_digest(critic)))


def case_rollouts(method,seed,version,case,role,group):
    folder=head_runtime(method,seed)
    quota=BATCH if role=="train" else 10000
    base=folder/f"{role}-{version:03d}-{case}"
    base.mkdir(exist_ok=True)
    records,rows=[],[]
    blank=0
    for episode in range(quota+3):
        out=base/f"{STUDY}-{head_name(method,seed)}-{role}-v{version:03d}-{case}-e{episode:03d}"
        raw=out/"result.json"
        if not raw.is_file() and out.exists():
            # Completed events stay on-policy. An unclosed event has no successor and cannot enter the loss.
            observed=batch_rows(out)
            if observed:
                observed[-1].update(truncated=True,end_kind="external_batch_cut")
            rows.extend(observed)
            records.append(dict(attempt=out.name,status="interrupted",observed=len(observed),open_cut=True))
            if len(rows)>=quota:
                break
            continue
        if not raw.exists():
            command=["taskset","-c",",".join(map(str,group)),sys.executable,"-u",str(COLLECT),
                "--out",str(out),"--weights",str(folder/f"policy-{version:03d}.pt"),"--method",method,"--seed",str(seed),
                "--version",str(version),"--episode",str(episode),"--case",case,"--role",role,
                "--offset",str(len(rows)),"--quota",str(quota-len(rows)),"--duration","60"]
            with (base/(out.name+".log")).open("a") as log:
                rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
            if rc or not raw.is_file():
                return dict(status="technical",case=case,attempt=out.name,raw=str(out),observed=len(rows)+len(batch_rows(out)),
                    reason=read(raw).get("error") if raw.is_file() else "see collector log")
        result=read(raw)
        if role=="evaluate" and result["physical_steps"] and not (out/"video.json").is_file():
            from render import render
            render(out)
        batch=batch_rows(out)
        rows.extend(batch)
        records.append({k:result[k] for k in ("run","reason","requested","observed","open_cut","physical_s","physical_steps",
            "wall_s","cpu_h","max_rss_gib","cuda_peak_allocated_bytes","progress","complete","physical_failure",
            "absolute_mechanical_work_J","force_peak_percentiles","control_full_ms_p50_p95_p99","actor_sha")})
        if (out/"video.json").is_file():
            records[-1]["video"]=read(out/"video.json")
        if result["reason"]=="technical":
            return dict(status="technical",case=case,attempt=out.name,raw=str(out),observed=len(rows),reason=result["error"])
        if role=="evaluate" or len(rows)==quota:
            break
        blank=0 if batch else blank+1
        if blank>=3:
            return dict(status="collection_starved",case=case,episodes=records,observed=len(rows))
    return dict(status="observed",case=case,episodes=records,observed=len(rows),rows=rows)


def paired(method,seed,version,role):
    manifest=read(RUN/"manifest.json")
    if source_identities()!=manifest["sources"]:
        raise ValueError("Immutable W61 study source/dataset changed, register a new runtime before sampling")
    groups=json.loads(os.environ["W61_PAIR_CPUS"])
    if len(groups)!=2 or len(os.environ["W61_SLOTS"].split(","))!=2:
        raise ValueError("A physical A/B collection requires two atomic shared leases")
    folder=head_runtime(method,seed)
    prefix=f"{role}-{version:03d}"
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures=[workers.submit(case_rollouts,method,seed,version,c,role,g) for c,g in zip(CASES,groups)]
        outcomes=[f.result() for f in futures]
    compact=[{k:v for k,v in o.items() if k!="rows"} for o in outcomes]
    receipt=dict(study=STUDY,role=role,method=method,seed=seed,version=version,requested_per_case=BATCH if role=="train" else None,
        cases=compact,slots=os.environ["W61_SLOTS"],groups=groups,checkpoint_sha=file_digest(folder/f"policy-{version:03d}.pt"))
    if any(o["status"]!="observed" for o in outcomes):
        receipt["status"]="technical_or_starved"
        atomic_json(head_result(method,seed)/(prefix+".json"),receipt)
        return 1
    if role=="evaluate":
        receipt.update(status="evaluated",used=0,distinct_observed=sum(o["observed"] for o in outcomes))
        atomic_json(head_result(method,seed)/(prefix+".json"),receipt)
        return 0
    streams=[o["rows"] for o in outcomes]
    checkpoint=torch.load(folder/f"policy-{version:03d}.pt",map_location="cpu",weights_only=True)
    for case,rows in zip(CASES,streams):
        if len(rows)!=BATCH or any((r["position"],r["version"],r["case"],r["method"],r["seed"],r["role"],r["actor_sha"])!=
                (i,version,case,method,seed,"train",checkpoint["actor_sha"]) for i,r in enumerate(rows)):
            raise ValueError("Distinct physical actor/position/case identities do not match this update")
    actor,critic=new_heads(seed)
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    actopt=torch.optim.Adam(actor.parameters(),lr=OPT["learning_rate"])
    critopt=torch.optim.Adam(critic.parameters(),lr=OPT["learning_rate"])
    actopt.load_state_dict(checkpoint["actor_optimizer"])
    critopt.load_state_dict(checkpoint["critic_optimizer"])
    info=update(actor,critic,actopt,critopt,streams,method,seed,version)
    event_ids=[r["event_id"] for rows in streams for r in rows]
    old=set()
    for k in range(version):
        old.update(read(folder/f"used-{k:03d}.json")["event_ids"])
    if old.intersection(event_ids):
        raise ValueError("Real transitions consumed twice")
    atomic_json(folder/f"used-{version:03d}.json",dict(version=version,event_ids=event_ids))
    used=checkpoint["used"]+len(event_ids)
    next_ck=folder/f"policy-{version+1:03d}.pt"
    atomic_weights(next_ck,dict(study=STUDY,method=method,seed=seed,version=version+1,used=used,
        actor=actor.state_dict(),critic=critic.state_dict(),actor_sha=weight_digest(actor),critic_sha=weight_digest(critic),
        actor_optimizer=actopt.state_dict(),critic_optimizer=critopt.state_dict(),
        per_case_cursor={c:(version+1)*BATCH for c in CASES},rng_schema=checkpoint["rng_schema"]))
    receipt.update(status="updated",used=used,data_hash=hashlib.sha256("\n".join(event_ids).encode()).hexdigest(),
        updated_checkpoint=str(next_ck),updated_checkpoint_sha256=file_digest(next_ck),algorithm=info)
    atomic_json(head_result(method,seed)/(prefix+".json"),receipt)
    return 0


def matrix():
    proof=RESULT/"interface-v1.json"
    numerical=RESULT/"development-update-v2.json"
    if read(proof)["status"]!="interface_observed" or read(numerical)["status"]!="verified":
        raise ValueError("Physical mode/trajectory authority or causal update diagnostics incomplete")
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/"manager.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        register=RUN/"manifest.json"
        sources=source_identities()
        if register.exists():
            manifest=read(register)
            if manifest["sources"]!=sources:
                raise ValueError("Saved study source hashes changed")
        else:
            manifest=dict(study=STUDY,sources=sources,source_commit=subprocess.check_output(
                ["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),contract_sha256=file_digest(CONFIG),
                interface_sha256=file_digest(proof),update_diagnostic_sha256=file_digest(numerical),
                primary_methods=METHODS,seed=SPEC["seeds"],used_per_primary_head=TOTAL,primary_planned=3*3*TOTAL,
                geometry_ablation=dict(method="E-Fisher",seeds=SPEC["seeds"],separate_planned=3*TOTAL),
                freeze_nodes=SPEC["curve_nodes"],leased_atomic_per_head=2,maximum_simultaneous_heads=2,
                wall_deadline_enabled=False,deadline=None,kill_on_conversation_end=False)
            atomic_json(register,manifest)
            atomic_json(SMALL/"manifest.json",manifest)
        term={}
        for methods in [METHODS,[SPEC["geometry_ablation"]]]:
            heads=[(m,s) for s in SPEC["seeds"] for m in (["E-score","PPO","SPO"] if methods==METHODS else methods)]
            for m,s in heads:
                prepare(m,s)
            for version in range(TOTAL//(2*BATCH)):
                jobs=[]
                for m,s in heads:
                    rec=head_result(m,s)/f"train-{version:03d}.json"
                    if rec.exists():
                        if read(rec)["status"]!="updated": term[head_name(m,s)]=read(rec)["status"]
                        continue
                    if head_name(m,s) not in term:
                        jobs.append((m,s,"train",version))
                run_jobs(jobs,term)
                node=(version+1)*BATCH*2
                if node in SPEC["curve_nodes"]:
                    jobs=[(m,s,"evaluate",version+1) for m,s in heads if head_name(m,s) not in term and
                        not (head_result(m,s)/f"evaluate-{version+1:03d}.json").exists()]
                    run_jobs(jobs,term)
        atomic_json(SMALL/"lifecycle.json",dict(status="naturally_ended",technical_or_starved=term,completed_at_unix=time.time()))


def run_jobs(jobs,term):
    def one(job):
        m,s,role,ver=job
        name=f"w61-{STUDY}-{head_name(m,s)}-{role}-v{ver:03d}"
        cmd=[sys.executable,"-u",SELF,"paired",m,str(s),str(ver),role]
        rc=admitted_call(name,2,cmd,ram=5.5,gpu=1500,disk=1.0)
        if rc:
            return head_name(m,s),dict(job=name,role=role,returncode=rc)
        return None
    with ThreadPoolExecutor(max_workers=2) as workers:
        for item in as_completed([workers.submit(one,j) for j in jobs]):
            stopped=item.result()
            if stopped:
                name,reason=stopped
                term[name]=reason
                atomic_json(SMALL/name/"terminal.json",reason)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("matrix")
    pair=sub.add_parser("paired")
    pair.add_argument("method")
    pair.add_argument("seed",type=int)
    pair.add_argument("version",type=int)
    pair.add_argument("role",choices=["train","evaluate"])
    a=p.parse_args()
    if a.command=="matrix":
        matrix()
    else:
        raise SystemExit(paired(a.method,a.seed,a.version,a.role))
