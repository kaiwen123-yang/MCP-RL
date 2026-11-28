"""Finite, measured updates of a common event-time critic and mixed policy."""
from copy import deepcopy
import json
import math
import time

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from hybrid import ACTIVE, CONFIG, DIMS, SIGMA, SPEC, joint_logp, mixed_kl, stream
from orbits import nominal_response
from s1_w59_model_fields import STATE

OPT = SPEC["algorithm"]
RESPONSE_CONFIG=CONFIG.with_name("response-update.json")
RESPONSE=json.loads(RESPONSE_CONFIG.read_text())


def targets(streams):
    advantages,returns=[],[]
    for rows in streams:
        adv=np.zeros(len(rows))
        carry=0.
        for i in range(len(rows)-1,-1,-1):
            r=rows[i]
            discount=SPEC["per_half_second_discount"]**(r["dt_s"]/.5)
            lam=SPEC["per_half_second_gae"]**(r["dt_s"]/.5)
            linked=(i+1<len(rows) and rows[i+1]["attempt"]==r["attempt"] and not r["truncated"] and not r["terminated"])
            delta=r["reward"]+discount*r["next_value"]*(not r["terminated"])-r["value"]
            carry=delta+linked*discount*lam*carry
            adv[i]=carry
        advantages.extend(adv)
        returns.extend(adv+np.array([r["value"] for r in rows]))
    return torch.tensor(advantages,dtype=torch.float32),torch.tensor(returns,dtype=torch.float32)


def geometry_subset(streams):
    found=[]
    for case,rows in enumerate(streams):
        phases={}
        for i,row in enumerate(rows):
            phase=int(np.argmax(np.asarray(row["obs"])[STATE["phase"]]))
            phases.setdefault(phase,[]).append(i)
        counts={s:1 for s in phases}
        while sum(counts.values()) < min(32,len(rows)):
            s=max(phases,key=lambda k:len(phases[k])/counts[k])
            counts[s]+=1
        chosen=[]
        for s,ns in counts.items():
            indices=np.linspace(0,len(phases[s])-1,ns,dtype=int)
            chosen.extend(phases[s][j] for j in indices)
        found.extend(sum(len(st) for st in streams[:case])+int(j) for j in sorted(set(chosen)))
    return torch.tensor(found)


class ResponseMap:
    """Within-chart finite differences of the same frozen C2 and WBC load-reference code."""
    def __init__(self,actor,obs,rows,ids,seed,version):
        self.obs=obs[ids]
        self.contexts=[rows[i]["response_context"] for i in ids.tolist()]
        self.events=[rows[i]["event_id"] for i in ids.tolist()]
        oldlog,means=actor(self.obs)
        self.noise=SIGMA*torch.randn(means.shape,generator=stream(seed,"geometry",version))*ACTIVE
        self.center=(means+self.noise).detach()
        x=self.center.numpy()
        self.queries=0
        self.old_response=self.actual(self.center)
        self.baseline=(oldlog.exp().detach().unsqueeze(-1)*self.old_response).sum(1,keepdim=True)
        derivatives=np.zeros((*self.old_response.shape,DIMS))
        discrepancies=[]
        for i,ctx in enumerate(self.contexts):
            for m in range(2):
                if self.obs[i,-2+m] < .5:
                    continue
                for j in np.flatnonzero(ACTIVE[m].numpy()):
                    diffs=[]
                    for h in (.004,.002):
                        d=np.zeros(DIMS)
                        d[j]=h
                        diffs.append((nominal_response(ctx,m,x[i,m]+d)-nominal_response(ctx,m,x[i,m]-d))/(2*h))
                        self.queries+=2
                    derivatives[i,m,:,j]=diffs[-1]
                    discrepancies.append(np.linalg.norm(diffs[0]-diffs[1]))
        self.d=torch.tensor(derivatives,dtype=torch.float32)
        self.halving_rmse=float(np.sqrt(np.mean(np.square(discrepancies))))
        self.max_singular=float(torch.linalg.svdvals(self.d).max())
        sv=torch.linalg.svdvals(self.d)
        self.active_action_rank=float((sv>1.e-5).sum(-1).sum()/self.obs[:,-2:].sum())
        self.phi0=self.embed(oldlog.detach(),self.old_response).detach()

    def actual(self,raw):
        x=raw.detach().cpu().numpy()
        out=np.array([[nominal_response(ctx,m,x[i,m]) for m in range(2)] for i,ctx in enumerate(self.contexts)])
        self.queries+=2*len(self.contexts)
        return torch.tensor(out,dtype=raw.dtype)

    def embed(self,lp,response):
        rootp=(.5*lp).exp().unsqueeze(-1)
        return torch.cat((rootp,rootp*(response-self.baseline)),dim=-1)

    def deviation(self,actor,direct=False):
        lp,mu=actor(self.obs)
        raw=mu+self.noise
        y=(self.actual(raw) if direct else self.old_response + torch.einsum("nmkd,nmd->nmk",self.d,raw-self.center))
        return .5*(self.embed(lp,y)-self.phi0).square().sum((1,2)).mean()


def flatgrad(value,parameters,create=False):
    return torch.cat([v.reshape(-1) for v in torch.autograd.grad(value,parameters,create_graph=create,retain_graph=True)])


def conjugate(operator,g):
    estimate=torch.zeros_like(g)
    residual=g.clone()
    direction=residual.clone()
    norm=residual.dot(residual)
    it=0
    for it in range(1,OPT["cg_steps"]+1):
        product=operator(direction)
        denom=direction.dot(product)
        if not torch.isfinite(denom) or denom<=0.:
            break
        alpha=norm/denom
        estimate+=alpha*direction
        residual-=alpha*product
        nxt=residual.dot(residual)
        if nxt <= max(1.e-14,float(g.dot(g))*1.e-12):
            norm=nxt
            break
        direction=residual+(nxt/norm)*direction
        norm=nxt
    return estimate,it,math.sqrt(float(norm))


def update(actor,critic,actopt,critopt,streams,method,seed,version):
    begin=time.perf_counter()
    rows=[r for s in streams for r in s]
    ids=[r["event_id"] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError("Duplicated real event in one update")
    obs=torch.tensor([r["obs"] for r in rows])
    mode=torch.tensor([r["mode"] for r in rows])
    raw=torch.tensor([r["raw"] for r in rows])
    oldlog=torch.tensor([r["oldlog"] for r in rows])
    oldmean=torch.tensor([r["oldmean"] for r in rows])
    oldp=torch.tensor([r["logp"] for r in rows])
    with torch.no_grad():
        lp,mu=actor(obs)
        if (lp-oldlog).abs().max()>2.e-5 or (mu-oldmean).abs().max()>2.e-5 or (joint_logp(lp,mu,mode,raw)-oldp).abs().max()>2.e-5:
            raise ValueError("Original actor law and ledger disagree")
    adv,ret=targets(streams)
    advstd=float(adv.std(unbiased=False))
    adv=(adv-adv.mean())/(advstd+1.e-8)
    actor_rng=stream(seed,"actor_permutation",version)
    critic_rng=stream(seed,"critic_permutation",version)
    losses=[]
    for epoch in range(OPT["epochs"]):
        for idx in torch.randperm(len(rows),generator=critic_rng).split(OPT["minibatch"]):
            value=(critic(obs[idx])-ret[idx]).square().mean()
            critopt.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),OPT["max_grad_norm"],error_if_nonfinite=True)
            critopt.step()
            losses.append(float(value.detach()))
    accepted=0
    proposed=0
    info={}
    def evaluate_law():
        lp,mu=actor(obs)
        ratio=(joint_logp(lp,mu,mode,raw)-oldp).exp()
        return (ratio*adv).mean(),mixed_kl(oldlog,oldmean,lp,mu)
    if method in ("PPO","SPO"):
        for epoch in range(OPT["epochs"]):
            stopped=False
            for ii in torch.randperm(len(rows),generator=actor_rng).split(OPT["minibatch"]):
                lp,mu=actor(obs[ii])
                ratio=(joint_logp(lp,mu,mode[ii],raw[ii])-oldp[ii]).exp()
                surrogate=ratio*adv[ii]
                eps=OPT["epsilon"]
                if method=="PPO":
                    obj=torch.minimum(surrogate,ratio.clamp(1.-eps,1.+eps)*adv[ii])
                else:
                    obj=surrogate-(ratio-1.).square()*adv[ii].abs()/(2.*eps)
                saved=deepcopy(actor.state_dict())
                savedopt=deepcopy(actopt.state_dict())
                actopt.zero_grad()
                (-obj.mean()).backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(),OPT["max_grad_norm"],error_if_nonfinite=True)
                actopt.step()
                proposed+=1
                with torch.no_grad():
                    _,kl=evaluate_law()
                if not torch.isfinite(kl) or kl>OPT["kl_max"]:
                    actor.load_state_dict(saved)
                    actopt.load_state_dict(savedopt)
                    stopped=True
                    break
                accepted+=1
            if stopped:
                break
    else:
        geo_start=time.perf_counter()
        geometry=ResponseMap(actor,obs,rows,geometry_subset(streams),seed,version) if method=="E-score" else None
        params=list(actor.parameters())
        snapshot=parameters_to_vector(params).detach().clone()
        obj,kl=evaluate_law()
        g=flatgrad(obj,params).detach()
        fgrad=flatgrad(kl,params,True)
        cgrad=flatgrad(geometry.deviation(actor),params,True) if geometry else None
        def F(v): return flatgrad(fgrad.dot(v),params).detach()
        def C(v): return flatgrad(cgrad.dot(v),params).detach() if cgrad is not None else torch.zeros_like(v)
        best=None
        candidates=[]
        for multiplier in OPT["geometry_multipliers"] if geometry else [0.]:
            metric=RESPONSE["base_response_metric_weight"]+multiplier if geometry else 0.
            def op(v): return F(v)+metric*C(v)+OPT["damping"]*v
            v,it,res=conjugate(op,g)
            fg=max(0.,float(v.dot(F(v))))
            cg=max(0.,float(v.dot(C(v))))
            scale=min(math.sqrt(2*OPT["kl_max"]/max(fg,1.e-20)),
                math.sqrt(2*OPT["response_budget"]/max(cg,1.e-20)) if geometry else 1.e9)
            d=scale*v
            gain=float(g.dot(d))
            candidates.append(dict(multiplier=multiplier,effective_response_weight=metric,predicted_gain=gain,cg_iterations=it,residual=res,fisher_quadratic=fg,response_quadratic=cg))
            if best is None or gain>best[0]: best=gain,d,multiplier
        gain,d,multiplier=best
        info.update(geometry_candidates=candidates,chosen_multiplier=multiplier,
                    effective_response_weight=RESPONSE["base_response_metric_weight"]+multiplier if geometry else 0.,gradient_norm=float(g.norm()),
                    score_direction_cosine=float(torch.nn.functional.cosine_similarity(g[None],d[None])[0]))
        before=float(obj.detach())
        for k in range(OPT["backtracks"]):
            proposed+=1
            vector_to_parameters(snapshot+(0.5**k)*d,params)
            with torch.no_grad():
                objective,new_kl=evaluate_law()
                change=float(geometry.deviation(actor,True)) if geometry else 0.
            if (torch.isfinite(objective) and torch.isfinite(new_kl) and objective>before
                    and new_kl <= OPT["kl_max"] and change <= OPT["response_budget"]):
                accepted=1
                info.update(line_search_scale=.5**k,response_change=change,surrogate_improvement=float(objective)-before)
                break
        if not accepted:
            vector_to_parameters(snapshot,params)
        if geometry:
            info.update(geometry_events=geometry.events,query_calls=geometry.queries,halving_rmse=geometry.halving_rmse,
                max_response_singular_value=geometry.max_singular,mean_action_rank=geometry.active_action_rank)
        info["geometry_solver_wall_s"]=time.perf_counter()-geo_start
    with torch.no_grad():
        result_obj,result_kl=evaluate_law()
    return dict(method=method,seed=seed,old_version=version,distinct_real_used=len(ids),
        requested=sum(len(s) for s in streams),accepted_actor_steps=accepted,proposed_actor_steps=proposed,critic_updates=len(losses),
        final_kl=float(result_kl),final_surrogate=float(result_obj),original_advantage_std=advstd,
        critic_mse=float(np.mean(losses)),elapsed_mean_s=float(np.mean([r["dt_s"] for r in rows])),
        hold_count=sum(r["mode"]==1 for r in rows),physical_failures=sum(r["end_kind"]=="physical_failure" for r in rows),
        imagined=0,raw_event_reward_mean=float(np.mean([r["reward"] for r in rows])),optimizer_wall_s=time.perf_counter()-begin,**info)
