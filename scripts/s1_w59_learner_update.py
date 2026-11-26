"""One sealed W59 actor version and independent first-update physical admission."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from s1_w59_learning import (BATCH_PER_CASE, RequestActor, TrueTaskCritic, atomic_json, atomic_weights,
    batch_file, checkpoint, file_digest, json_rows, project_tensor, read_json)
from s1_w59_learner_critic import BETA, SPEC
from s1_w59_model_fields import DT, INTERVAL


def load_batch(run, version):
    docs = [json_rows(batch_file(run,version,case)) for case in "AB"]
    seen_path = run / "used-event-ids.jsonl"
    old = {key for group in (json_rows(seen_path) if seen_path.exists() else [])
           if group["version"]<version for key in group["ids"]}
    rows = [r for stream in docs for r in stream]
    keys = [r["event_id"] for r in rows]
    if len(rows)!=256 or len(set(keys))!=256 or old.intersection(keys):
        raise ValueError("A sealed update must have 256 distinct physical events absent from earlier optimized batches")
    for case,stream in zip("AB", docs):
        if len(stream)!=BATCH_PER_CASE:
            raise ValueError("Unequal A/B collector batch")
        for pos,row in enumerate(stream):
            if (row["case"],row["position"],row["version"],row["serial"],row["sampling"]) != (
                    case,pos,version,version*BATCH_PER_CASE+pos,"on_policy"):
                raise ValueError("Wrong event identity or sampling law at the shared policy barrier")
    return docs, rows


def task_targets(streams):
    returns, advantages = [], []
    for stream in streams:
        adv = np.empty(len(stream))
        carry = 0.
        for i in range(len(stream)-1,-1,-1):
            row=stream[i]
            dec=math.exp(-BETA*row["dt_s"])
            lam=SPEC["gae_lambda_per_half_second"]**(row["dt_s"]/.5)
            continuing=(i+1<len(stream) and stream[i+1]["attempt"]==row["attempt"]
                        and not row["truncated"] and not row["terminated"])
            td=row["reward"]+dec*(not row["terminated"])*row["next_value"]-row["value"]
            carry=td+continuing*dec*lam*carry
            adv[i]=carry
        advantages.extend(adv)
        returns.extend([float(a+r["value"]) for a,r in zip(adv,stream)])
    return np.asarray(returns,np.float32),np.asarray(advantages,np.float32)


def worlds(rows,device):
    data=np.zeros((len(rows),4200,5),np.float32)
    data[:,:,1:4]=-1
    for i,row in enumerate(rows):
        keys=row["world"]["keys"]
        if keys:
            data[i,keys]=row["world"]["measured"]
    return torch.tensor(data,device=device)


def imagination(actor,critic,wm,device,state,base,world,material,private,steps):
    from s1_w59_model import critic_features, private_state
    from s1_w59_planning import select_batch
    size=state.shape[0]
    live=torch.ones(size,dtype=torch.bool,device=device)
    request=torch.ones_like(live)
    held=base
    discount=math.exp(-BETA*DT)
    weight=1.
    total=state.new_zeros(size)
    trajectory=[]
    simulated=0
    events=0
    for k in range(steps):
        if not live.any():
            break
        issued=request & live
        if issued.any():
            positions=torch.where(issued)[0]
            states=state.index_select(0,positions)
            tables=world.index_select(0,positions)
            if k==0:
                plans=base.index_select(0,positions)
            else:
                with torch.no_grad():
                    plans,_=select_batch(wm,states.detach(),tables.detach(),private=private,material=material)
            inputs=private_state(states,tables) if private else states
            issued_actions=project_tensor(plans.detach(),actor(inputs,plans.detach()))
            held=held.index_copy(0,positions,issued_actions)
            events+=len(positions)
        current=critic_features(state)
        # Discrete modes are hard-forward, fixed for the local derivative. World table is carried
        # forward by this particular candidate and used at the next request.
        step_state,interval,step_world=wm.step_with_world(state,held,world,material=material,private=private)
        interval=interval.reshape(size,-1)
        reward=interval[:,INTERVAL["task_reward"]].squeeze(-1)
        is_absorbed=((interval[:,INTERVAL["complete"]]>0.5) | (interval[:,INTERVAL["hard_failure"]]>0.5)).flatten()
        truncated=(interval[:,INTERVAL["truncated"]]>0.5).flatten() & ~is_absorbed
        request=(interval[:,INTERVAL["request"]]>0.5).flatten()
        successor=critic_features(step_state)
        # Time-limit truncation has a critic bootstrap; actual task completion/failure has zero.
        terminal_bootstrap=critic(successor)*truncated
        active_reward=reward+discount*terminal_bootstrap
        total=total+weight*live*active_reward
        trajectory.append((current.detach(),successor.detach(),reward.detach(),live.detach(),
                           is_absorbed.detach(),truncated.detach()))
        simulated+=int(live.sum().detach())
        live=live & ~is_absorbed & ~truncated
        state,world=step_state,step_world
        weight*=discount
    total=total+weight*live*critic(critic_features(state))
    # Detached, separately counted value targets, never relabelled as measured transitions.
    imagined_features,imagined_targets=[],[]
    with torch.no_grad():
        boot=critic(critic_features(state))
        lam=SPEC["imagination"]["lambda_per_half_second"]**(DT/.5)
        for cur,nxt,r,active,absorb,trunc in reversed(trajectory):
            nv=critic(nxt)
            y=r+discount*(~absorb)*torch.where(trunc,nv,(1-lam)*nv+lam*boot)
            imagined_features.append(cur[active])
            imagined_targets.append(y[active])
            boot=y
    return total,torch.cat(imagined_features),torch.cat(imagined_targets),simulated,events


def first_update_verdict(receipt_paths, registry, old_sha, new_sha, floor):
    """Eight preregistered initial states, both real continuations including failures."""
    locked=registry["states"]
    if len(locked)!=8 or len({r["snapshot_id"] for r in locked})!=8:
        raise ValueError("A first-update comparison needs eight separately registered legal requests")
    by_id={}
    for path in receipt_paths:
        result=read_json(path)
        branch=result["branch"]; reg=branch["registration"]; cont=result["w59_learning"]["continuation"]
        if result.get("error") or cont["technical"] or not branch["applied"] or reg["replacement"] is not None:
            raise ValueError(f"Invalid first-update physical continuation at {path}")
        if reg["role"] not in ("before","after"):
            raise ValueError(f"Unexpected branch role at {path}")
        sha=old_sha if reg["role"]=="before" else new_sha
        if reg["actor_digest"]!=sha or cont["actor_digest"]!=sha or reg["duration_s"]!=2:
            raise ValueError(f"Wrong checkpoint or continuation length at {path}")
        key=(reg["snapshot_id"],reg["role"])
        if key in by_id:
            raise ValueError(f"Duplicate actor direction outcome {key}")
        if not cont["physical_failure"] and abs(branch["elapsed_s"]-2)>1e-6:
            raise ValueError(f"Unclosed physical comparison at {path}")
        by_id[key]=(reg,cont,path)
    pairs=[]
    for entry in locked:
        key=entry["snapshot_id"]
        before,a,p=by_id[key,"before"];after,b,q=by_id[key,"after"]
        if not (before["sha256"]==after["sha256"]==entry["sha256"]
                and before["rng_key"]==after["rng_key"]==entry["rng_key"]
                and before["request_id"]==after["request_id"]==entry["request_id"]):
            raise ValueError("Counterfactuals changed initial state, request or random stream")
        pairs.append(dict(snapshot_id=key,before=str(p),after=str(q),before_return=a["return_w38"],
            after_return=b["return_w38"],delta=b["return_w38"]-a["return_w38"],
            before_hard_failure=a["physical_failure"],after_hard_failure=b["physical_failure"]))
    if len(by_id)!=16:
        raise ValueError("Every first-update branch must belong to the 8x2 comparison")
    diff=np.array([p["delta"] for p in pairs])
    n0=sum(p["before_hard_failure"] for p in pairs); n1=sum(p["after_hard_failure"] for p in pairs)
    rng=np.random.default_rng(5901)
    bootstrap=diff[rng.integers(0,8,size=(4096,8))].mean(axis=1)
    return dict(pairs=pairs,marginal_mean_delta=float(diff.mean()),mean_replay_floor=floor,
                paired_95pct_mean=np.quantile(bootstrap,[.025,.975]).tolist(),
                signs_resolvable=int((np.abs(diff)>floor).sum()),before_failures=n0,after_failures=n1,
                accepted=bool(diff.mean()>floor and n1<=n0))


def batch_update(run,version,device):
    scope=read_json(run/"campaign.json")
    publication=read_json(run/"published.json")
    payload=torch.load(checkpoint(run,version),map_location=device,weights_only=True)
    seed,head=payload["seed"],payload["arm"]
    streams,rows=load_batch(run,version)
    if any((r["arm"],r["seed"],r["versions"]["model_sha256"])!=(head,seed,publication["wm_sha256"]) for r in rows):
        raise ValueError("New head data crossed arm, seed or published model-version boundaries")
    actor,critic=RequestActor().to(device),TrueTaskCritic().to(device)
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor_opt=torch.optim.Adam(actor.parameters(),lr=SPEC["actor"]["lr"])
    critic_opt=torch.optim.Adam(critic.parameters(),lr=SPEC["critic"]["standalone_fit"]["lr"])
    actor_opt.load_state_dict(payload["actor_optimizer"]);critic_opt.load_state_dict(payload["critic_optimizer"])
    def field(key):
        return torch.as_tensor(np.asarray([r[key] for r in rows]),dtype=torch.float32,device=device)
    obs,base,raw,critic_state=field("state"),field("base"),field("raw"),field("critic_state")
    ret,adv=task_targets(streams)
    ret,adv=(torch.as_tensor(a,device=device) for a in (ret,adv))
    oldlogp=field("logp_raw")
    with torch.no_grad():
        old_mean=actor(obs,base).detach()
        old_mismatch=(actor.logp(old_mean,raw)-oldlogp).abs().max().item()
        if old_mismatch>1e-4 or (old_mean-field("mean_raw")).abs().max()>1e-5:
            raise ValueError("Stored original Gaussian sample law disagrees with the sealed actor")
    recipe=SPEC["actor"]
    start=time.perf_counter()
    generator=torch.Generator(device="cpu").manual_seed(seed*100000+version)
    raw_signal=float(adv.std(unbiased=False))
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    real_losses=[]
    for epoch in range(recipe["epochs"]):
        for ids in torch.randperm(len(rows),generator=generator).split(recipe["minibatch"]):
            ids=ids.to(device)
            loss=(critic(critic_state[ids])-ret[ids]).square().mean()
            critic_opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),recipe["max_grad_norm"])
            critic_opt.step();real_losses.append(loss.item())
    diff=head!="PPO-full"
    wm=None
    if diff:
        from s1_w59_model import ExecutionEnsemble
        wm=ExecutionEnsemble.load(publication["wm_checkpoints"],device)
        wm.planning_substeps=scope["planning_substeps"]
        for parameter in wm.parameters():
            parameter.requires_grad_(False)
        wm.eval()
        horizon=scope["gradient_admission"]["trusted_substeps"]
        if not 1<=horizon<=SPEC["imagination"]["substeps"]:
            raise ValueError("Imagined horizon exceeds the independently calibrated interval")
    proposed=accepted=imagined=imagined_requests=0
    max_attempt_kl=0.;triggered=None;last_kl=0.;image_x=[];image_y=[]
    orig=[p.detach().clone() for p in actor.parameters()]
    critic.requires_grad_(False)
    for epoch in range(recipe["epochs"]):
        if triggered is not None:break
        for indices in torch.randperm(len(rows),generator=generator).split(recipe["minibatch"]):
            ids=indices.to(device)
            if diff:
                obj,ix,iy,n,k=imagination(actor,critic,wm,device,obs[ids],base[ids],worlds([rows[i] for i in indices.tolist()],device),
                                           scope["material"],head=="Diff-private",horizon)
                loss=-obj.mean()
                image_x.append(ix.cpu());image_y.append(iy.cpu());imagined+=n;imagined_requests+=k
            else:
                ratio=(actor.logp(actor(obs[ids],base[ids]),raw[ids])-oldlogp[ids]).exp()
                loss=-torch.minimum(ratio*adv[ids],ratio.clamp(1-recipe["clip_epsilon"],1+recipe["clip_epsilon"])*adv[ids]).mean()
            actor_snapshot=deepcopy(actor.state_dict());opt_snapshot=deepcopy(actor_opt.state_dict())
            actor_opt.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),recipe["max_grad_norm"])
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite policy gradient")
            actor_opt.step();proposed+=1
            with torch.no_grad():
                # Diagonal Gaussian with fixed std: this is the exact forward KL, performed after the real Adam step.
                kl=(.5*((actor(obs,base)-old_mean)/actor.std).square().sum(-1)).mean().item()
            max_attempt_kl=max(max_attempt_kl,kl)
            if not math.isfinite(kl) or kl>recipe["analytic_kl_max"]:
                actor.load_state_dict(actor_snapshot);actor_opt.load_state_dict(opt_snapshot)
                triggered=kl;break
            last_kl=kl;accepted+=1
    critic.requires_grad_(True)
    imag_losses=[]
    if image_x:
        xs,ys=torch.cat(image_x),torch.cat(image_y)
        for ids in torch.randperm(len(xs),generator=generator).split(256):
            il=(critic(xs[ids].to(device))-ys[ids].to(device)).square().mean()
            loss=SPEC["imagination"]["imagined_critic_weight"]*il
            critic_opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),recipe["max_grad_norm"])
            critic_opt.step();imag_losses.append(il.item())
    weight_change=float(torch.sqrt(sum((p.detach()-q).square().sum() for p,q in zip(actor.parameters(),orig))).cpu())
    metrics=dict(arm=head,seed=seed,version_input=version,version_output=version+1,observed_batch=len(rows),
        distinct_used=len(rows),cumulative_used=256*(version+1),return_mean=float(ret.mean()),advantage_std=raw_signal,
        real_critic_steps=len(real_losses),real_critic_loss=float(np.mean(real_losses)),imagined_critic_steps=len(imag_losses),
        imagined_critic_loss=float(np.mean(imag_losses)) if imag_losses else None,imagined_substeps=imagined,
        imagined_actor_calls=imagined_requests,first_actor_direction_needed=(weight_change>0 and not payload.get("verified_first_update",False)),
        actor_steps_proposed=proposed,actor_steps_accepted_by_KL=accepted,trigger_KL=triggered,max_proposed_KL=max_attempt_kl,
        final_KL=last_kl,delta_parameter_l2=weight_change,old_raw_logp_max_error=old_mismatch,
        success_absorptions=sum(r["task_complete"] for r in rows),physical_failures=sum(r["physical_failure"] for r in rows),
        physical_truncations=sum(r["truncated"] for r in rows),saturated_by_axis=np.sum([r["saturated"] for r in rows],axis=0).tolist(),
        model_sha256=publication["wm_sha256"],optimization_wall_s=time.perf_counter()-start)
    proposal=dict(payload,version=version+1,actor=actor.state_dict(),critic=critic.state_dict(),
        actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),used_events=metrics["cumulative_used"],
        accepted_actor_steps=payload.get("accepted_actor_steps",0)+accepted)
    path=run/f"proposal-{version+1:03d}.pt"
    atomic_weights(path,proposal)
    atomic_json(run/f"proposed-{version+1:03d}.json",metrics)
    return path,metrics,[r["event_id"] for r in rows]

