"""Resume a confirmed numerical episode interruption from closed on-policy events and a fresh registered reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"learning"))
import campaign
import collect
from hybrid import ROOT, SPEC, key
from s1_w59_learner_critic import atomic_json,file_digest


def read(path):
    return json.loads(path.read_text())


def immutable_copy(original, target):
    if target.exists():
        if file_digest(target)!=file_digest(original):
            raise ValueError(f"Immutable evidence mismatch: {target}")
    else:
        # The canonical JSON is always atomically replaced by campaign. Its original inode stays reachable here.
        os.link(original,target)


def witness(out):
    raw=out/"result.json"
    meta=out/"w55-f/solver-failure.json"
    if not raw.is_file() or not meta.is_file():return None
    r=read(raw)
    s=read(meta)
    kind=s.get("kind")
    end=float(r["physical_s"])
    if (r.get("reason")!="technical" or kind not in ("wbc","mpc") or s.get("status_val")!=7 or
            s.get("status")!="maximum iterations reached" or abs(s["time_s"]-end)>1.e-6 or
            not (r.get("error") or "").rstrip().endswith(f"{kind}: maximum iterations reached at physical time {end:.6f}")):
        return None
    inputs=[out/"w55-f"/f"failed-{kind}-{end:.6f}{suffix}.npz" for suffix in ("-P","-A","")]
    if not all(p.exists() for p in inputs):return None
    hashes={p.name:file_digest(p) for p in inputs}
    return dict(kind=kind,status=s["status"],iterations=s["iterations"],physical_s=end,
        primal_residual=s["primal_residual"],dual_residual=s["dual_residual"],raw=str(out),
        raw_result_sha256=file_digest(raw),solver_sha256=file_digest(meta),qp_sha256=hashes,
        fingerprint=hashlib.sha256(json.dumps([kind,end,*hashes.values()]).encode()).hexdigest())


def recoverable(receipt):
    if receipt.get("status")!="technical_or_starved":return False
    failed=[c for c in receipt.get("cases",[]) if c.get("status")!="observed"]
    return bool(failed) and all(c.get("status")=="technical" and c.get("raw") and witness(Path(c["raw"])) for c in failed)


def episode_rows(out,method,seed,version,case,checkpoint_sha,offset):
    source=read(out/"result.json")
    ident=read(out/"identity.json")
    rows=campaign.batch_rows(out)
    if (ident["method"],ident["seed"],ident["version"],ident["case"],ident["role"],ident["actor_sha"],ident["offset"]) != (
            method,seed,version,case,"train",checkpoint_sha,offset):
        raise ValueError(f"Original actor, case or sample interval differs at {out}")
    if len(rows)!=source["observed"] or any((r["position"],r["actor_sha"],r["episode"],r["attempt"],r["rng_key"]) !=
            (offset+i,checkpoint_sha,ident["episode"],out.name,key(seed,"train","explore",case,version,ident["episode"],i))
            for i,r in enumerate(rows)):
        raise ValueError(f"Closed events or action RNG sequence differs at {out}")
    for row in rows:
        if not np.isfinite([row["value"],row["reward"],row["logp"],row["next_value"],row["dt_s"]]).all() or row["dt_s"]<=0:
            raise ValueError("Nonfinite or zero-duration event cannot be recovered")
    keep={k:source[k] for k in ("run","reason","requested","observed","open_cut","physical_s","physical_steps",
          "wall_s","cpu_h","max_rss_gib","cuda_peak_allocated_bytes","progress","complete","physical_failure",
          "absolute_mechanical_work_J","force_peak_percentiles","control_full_ms_p50_p95_p99","actor_sha")}
    evidence=None
    if source["reason"]=="technical":
        evidence=witness(out)
        if evidence is None:
            raise ValueError(f"Numerical interruption is outside the registered recovery class: {out}")
        requests=[json.loads(line) for line in (out/"requests.jsonl").read_text().splitlines()] if (out/"requests.jsonl").exists() else []
        if source["open_cut"] not in (0,1) or len(requests)!=len(rows)+source["open_cut"]:
            raise ValueError("Cannot identify the unclosed request and last real successor")
        cut=None
        if rows:
            last=rows[-1]
            if last["terminated"]:
                raise ValueError("An absorbing event must not be converted to a data cut")
            next_request=requests[len(rows)] if source["open_cut"] else None
            if next_request and abs(next_request["start_s"]-last["end_s"])>1.e-8:
                raise ValueError("Next request does not start at the old actor's measured bootstrap boundary")
            cut=dict(event_id=last["event_id"],time_s=last["end_s"],next_value=last["next_value"],
                true_successor_request=next_request["event_id"] if next_request else None,
                raw_next_obs_persisted=bool((out/"closed-successors.jsonl").exists()),
                next_obs_source="original live true successor at event close; its critic value is retained; no observation from reset",
                previous_truncated=last["truncated"],truncated=True,bootstrap=last["next_value"])
            last.update(truncated=True,end_kind="technical_episode_cut",recovery_boundary=cut)
        evidence.update(archived_events_sha256=file_digest(out/"events.jsonl") if rows else None,
            open_excluded=[r["event_id"] for r in requests[len(rows):]],boundary=cut)
        keep["recovery_evidence"]=evidence
    return rows,keep,evidence


class SuccessorRecorder(collect.EventSession):
    """For new recovery episodes persist the true critic observation already passed to settle."""
    def settle(self,end,obs,kind):
        before=len(self.rows)
        super().settle(end,obs,kind)
        if len(self.rows)>before:
            row=self.rows[-1]
            collect.append(self.out/"closed-successors.jsonl",dict(event_id=row["event_id"],end_s=row["end_s"],
                obs=obs.detach().cpu().tolist(),next_value=row["next_value"],truncated=row["truncated"],terminated=row["terminated"]))


def case_continuation(method,seed,version,case,role,group):
    if role!="train":raise ValueError("Numerical continuation is limited to training")
    folder=campaign.head_runtime(method,seed)
    base=folder/f"train-{version:03d}-{case}"
    expected=torch.load(folder/f"policy-{version:03d}.pt",map_location="cpu",weights_only=True)["actor_sha"]
    rows=[]
    records=[]
    seen_fingerprints=set()
    blank=0
    episode=0
    original_count=0
    for out in sorted((p for p in base.iterdir() if p.is_dir()),key=lambda p:read(p/"identity.json")["episode"]):
        if not (out/"result.json").is_file():
            raise ValueError(f"Cannot take over a physically live or unresolved episode: {out}")
        ident=read(out/"identity.json")
        if ident["episode"]!=episode:
            raise ValueError("Non-contiguous independent reset index")
        part,rec,e=episode_rows(out,method,seed,version,case,expected,len(rows))
        if e:
            if e["fingerprint"] in seen_fingerprints:raise ValueError("Repeated identical retained failed QP")
            seen_fingerprints.add(e["fingerprint"])
        rows.extend(part)
        records.append(rec)
        blank=0 if part else blank+1
        episode+=1
        original_count+=len(part)
    while len(rows)<campaign.BATCH:
        if blank>=3 or episode>130:
            return dict(status="collection_starved",case=case,episodes=records,observed=len(rows))
        out=base/f"{campaign.STUDY}-{campaign.head_name(method,seed)}-train-v{version:03d}-{case}-e{episode:03d}-solver-continuation"
        cmd=["taskset","-c",",".join(map(str,group)),sys.executable,"-u",__file__,"collect","--out",str(out),
            "--weights",str(folder/f"policy-{version:03d}.pt"),"--method",method,"--seed",str(seed),
            "--version",str(version),"--episode",str(episode),"--case",case,"--role","train",
            "--offset",str(len(rows)),"--quota",str(campaign.BATCH-len(rows)),"--duration","60"]
        with (base/(out.name+".log")).open("a") as log:
            code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        if not (out/"result.json").is_file() or code and witness(out) is None:
            return dict(status="technical",case=case,attempt=out.name,raw=str(out),observed=len(rows),reason="new interruption outside saved numerical recovery class")
        part,rec,e=episode_rows(out,method,seed,version,case,expected,len(rows))
        if e:
            if e["fingerprint"] in seen_fingerprints:
                return dict(status="technical_terminal",case=case,attempt=out.name,raw=str(out),observed=len(rows)+len(part),reason="same saved QP fingerprint on a fresh episode")
            seen_fingerprints.add(e["fingerprint"])
        records.append(rec)
        rows.extend(part)
        episode+=1
        blank=0 if part else blank+1
    return dict(status="observed",case=case,episodes=records,observed=len(rows),rows=rows,
        closed_events_retained=original_count,new_real_episodes=len(records)-sum("solver-continuation" not in rec["run"] for rec in records),
        real_added=len(rows)-original_count)


def resume(method,seed,version):
    # Existing sealed actor, optimizer, tensors and the original on-policy updater are reused unchanged.
    folder=campaign.head_result(method,seed)
    canonical=folder/f"train-{version:03d}.json"
    original=folder/f"train-{version:03d}-original-technical.json"
    retained=read(canonical)
    if not recoverable(retained):
        raise ValueError("Can only resume a closed registered max-iterations episode")
    immutable_copy(canonical,original)
    pool_dir=Path("/root/autodl-tmp/MCP-RL-w61-joint/work/w61/pool/jobs")
    head=campaign.head_name(method,seed)
    old_job=f"w61-{campaign.STUDY}-{head}-train-v{version:03d}"
    lease=read(pool_dir/(old_job+".json"))
    if lease["state"]!="ended" or not lease["exit_code"]:
        raise ValueError("Original paired lease has not reached its natural failed boundary")
    archive=folder/f"train-{version:03d}-solver-continuation.json"
    previous_sha=file_digest(original)
    original_write=campaign.atomic_json
    def write(path,value):
        if path==canonical:
            value=dict(value,scientific_update="unchanged campaign.paired and update kernel",
                numerical_continuation=dict(original_receipt=str(original),original_sha256=previous_sha,
                    old_job=old_job,original_lease_sha256=file_digest(pool_dir/(old_job+".json")),
                    continuation_job=old_job+"-solver-continuation",canonical_updated_receipt=str(archive),
                    input_checkpoint_sha256=file_digest(campaign.head_runtime(method,seed)/f"policy-{version:03d}.pt"),
                    recovery_wrapper_sha256=file_digest(Path(__file__)),hard_bounds_changed=False,train_seed_changed=False,numerical_iterations_changed=False))
            original_write(archive,value)
        original_write(path,value)
    campaign.case_rollouts=case_continuation
    campaign.atomic_json=write
    return campaign.paired(method,seed,version,"train")


if __name__=="__main__":
    if sys.argv[1]=="collect":
        collect.EventSession=SuccessorRecorder
        # Use the exact argument contract of the immutable collector without rewriting its source.
        p=argparse.ArgumentParser()
        p.add_argument("mode");p.add_argument("--out",required=True,type=Path);p.add_argument("--weights",required=True,type=Path)
        p.add_argument("--method",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--case",required=True)
        p.add_argument("--role",required=True);p.add_argument("--variant",default="");p.add_argument("--version",type=int,required=True)
        p.add_argument("--episode",type=int,required=True);p.add_argument("--offset",type=int,required=True)
        p.add_argument("--quota",type=int,required=True);p.add_argument("--duration",type=float,required=True)
        collect.collect(p.parse_args())
    else:
        raise SystemExit(resume(sys.argv[2],int(sys.argv[3]),int(sys.argv[4])))
