"""Closed-form calibration of executed W59 macro request responses at the common 10k node."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from s1_w59_model_fields import STATE,INTERVAL,NS,NI,BG,MASS
from s1_w59_model_bank import ROOT,WORK,OUT,write

RIDGE=.01


def design(state,action):
    incoming=state[...,STATE['incoming']]
    sign=torch.where(incoming.cumsum(-1)<=1,incoming,-incoming)
    fz=state[...,STATE['measured_force']].unflatten(-1,(4,3))[...,2]
    imbalance=(sign*fz).sum(-1)/(MASS*9.81)
    context=torch.stack((torch.ones_like(imbalance),state[...,STATE['nominal_due_in']].squeeze(-1).div(.25).clamp(-1,1),
        state[...,STATE['transfer']].squeeze(-1),state[...,STATE['memory_valid']][...,:4].mean(-1),imbalance),dim=-1)
    nominal=action.new_tensor([0.,.05,0.]);scale=action.new_tensor([.05,.03,.10])
    u=(action-nominal)/scale
    return (context.unsqueeze(-1)*torch.cat((u,u.square()),dim=-1).unsqueeze(-2)).flatten(-2)


def valid_state(states):
    mask=np.ones_like(states,dtype=bool)
    for k in ('time','version','request_valid','control_valid','route_valid','memory','memory_valid','target_valid','target_xy','target_in','incoming','nominal_due_in','request_age','active','phase','goal','command'):
        mask[:,STATE[k]]=False
    return mask


def predict_data(member,episode):
    with np.load(WORK/'action-development'/(episode+'.npz')) as z:
        state,action,interval,valid,rigid,mixed=(z[k] for k in ('state','action','interval','valid','rigid','interior_request'))
    x=torch.from_numpy(state[:-1]);u=torch.from_numpy(action[:-1]);j=torch.from_numpy(rigid[:-1])
    nexts=[];intervals=[]
    with torch.no_grad():
        for sx,su,sj in zip(x.split(256),u.split(256),j.split(256)):
            n,y=member(sx,su,sj);nexts.append(n);intervals.append(y)
    pred=torch.cat((torch.cat(nexts),torch.cat(intervals)),dim=1).numpy()
    yn=np.column_stack((state[1:],interval))
    mask=np.column_stack((valid_state(state[1:]),valid))
    # Absorbing/technical termination and discrete observation horizon are not action residuals.
    for k in ('truncated','hard_failure','complete'):
        mask[:,NS+INTERVAL[k].start:NS+INTERVAL[k].stop]=False
    return dict(x=design(x,u).numpy().astype(float),action=action[:-1],phase=state[:-1,STATE['phase']].argmax(1),
        age=state[:-1,STATE['request_age']].ravel(),residual=yn-pred,pred=pred,truth=yn,valid=mask,legal=~mixed)


def match(s,n,fixed=False):
    k=min(len(s['x']),len(n['x']));ix=np.arange(k)
    sign=np.any(abs(s['action'][:k]-np.array([0.,.05,0.]))>1e-6,axis=1)
    if not fixed:
        # All direct exposures are from the first frozen signed request. Once a
        # new nominal was issued the later state remains available for the separate rollout check.
        sign &= np.all(abs(s['action'][:k]-s['action'][0])<1e-7,axis=1)
    keep=sign&s['legal'][:k]&n['legal'][:k]&(s['phase'][:k]==n['phase'][:k])
    ii=ix[keep]
    return s['x'][ii],s['residual'][ii]-n['residual'][ii],s['valid'][ii]&n['valid'][ii]


def calibrate(kind,seed,registry):
    from s1_w59_model import ExecutionMember,BINARY
    base=WORK/'checkpoints'/kind/str(seed)/'node-10000.pt'
    member=ExecutionMember.load(base,'cpu').eval()
    groups={}
    for r in registry['development']:groups.setdefault(r['source'],{})[r['signed']]=r['job']
    XX=[];YY=[];MM=[];WW=[];ledger=[]
    def add(rows,name):
        x,y,m=(np.concatenate([a[j] for a in rows]) for j in range(3))
        ledger.append(dict(group=name,rows=len(x)))
        if not len(x):return
        XX.append(x);YY.append(y);MM.append(m);WW.append(np.full(len(x),1/len(x)))
    for key,group in sorted(groups.items()):
        a=predict_data(member,group['nominal1']);b=predict_data(member,group['nominal2'])
        repeats=float(np.abs(a['truth']-b['truth']).max())
        a['residual']=(a['residual']+b['residual'])/2;a['valid']&=b['valid'];a['legal']&=b['legal']
        add([match(predict_data(member,v),a) for k,v in sorted(group.items()) if not k.startswith('nominal')],key)
        ledger[-1]['maximum_nominal_repeat_difference_mixed_units']=repeats
    aa=predict_data(member,registry['originals']['fixed-action/a'])
    ab=predict_data(member,registry['originals']['b3/a'])
    add([match(aa,ab,True)],'Fixed-A / B3-A E60')
    x,y,valid,w=np.concatenate(XX),np.concatenate(YY),np.concatenate(MM),np.concatenate(WW)
    coeff=np.zeros((x.shape[1],y.shape[1]));masks={};factorizations=0
    for i in range(y.shape[1]):masks.setdefault(np.packbits(valid[:,i]).tobytes(),[]).append(i)
    for idx in masks.values():
        mask=valid[:,idx[0]]
        if not mask.any():continue
        f=x[mask];weights=w[mask];wx=weights[:,None]*f
        cov=f.T@wx+RIDGE*np.eye(x.shape[1]);coeff[:,idx]=np.linalg.solve(cov,wx.T@y[mask][:,idx])
        factorizations+=1
    fixed=registry['originals']['fixed-action/b']
    raw=predict_data(member,fixed)
    member.install_action_coefficients(torch.from_numpy(coeff).float())
    new=predict_data(member,fixed)
    metrics={}
    use=raw['legal'];ns=NS
    for k,sl in INTERVAL.items():
        ix=slice(NS+sl.start,NS+sl.stop);mask=raw['valid'][:,ix]&use[:,None]
        if mask.any():metrics[k]=dict(observed_elements=int(mask.sum()),base_MAE=float(np.mean(np.abs(raw['residual'][:,ix][mask]))),adapted_MAE=float(np.mean(np.abs(new['residual'][:,ix][mask]))))
    for k in ('qpos','qvel','measured_force','force_reference','force_solved','feet','foot_velocity','progress','hold'):
        sl=STATE[k]
        mask=use
        metrics[k]=dict(observed_boundaries=int(mask.sum()),base_MAE=float(abs(raw['residual'][mask,sl]).mean()),adapted_MAE=float(abs(new['residual'][mask,sl]).mean()))
    srcsha=hashlib.sha256(base.read_bytes()).hexdigest()
    output=WORK/'action-checkpoints'/kind/f'{seed}-node-10000.pt'
    member.save(output,metadata=dict(source_checkpoint=str(base.relative_to(ROOT)),source_sha256=srcsha,
        action_calibration='W59-real-action-residual-calibration-v1',fit_data='complete B3-A 96 branches and B3-A/Fixed-A E60',model_gradient_updates=10000))
    write(OUT/f'actions-{kind}-{seed}.json',dict(source_checkpoint=str(base.relative_to(ROOT)),source_sha256=srcsha,
        calibrated_checkpoint=str(output.relative_to(ROOT)),calibrated_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        base_gradient_updates=10000,extra_gradient_updates=0,non_gradient_fits=1,FP64_masked_linear_factorizations=factorizations,
        features=x.shape[1],observed_rows=len(x),weighted_design_singular_values=np.linalg.svd(x*np.sqrt(w[:,None]),compute_uv=False).tolist(),
        entire_snapshot_group_weights=ledger,development_validation=dict(source=fixed,match='whole fixed-B E60; independent case after calibration, no refit',metrics=metrics),
        no_sealed_or_actor_check_input=True,**BG))
    print('fitted action residual',kind,seed,len(x),factorizations,flush=True)


def main():
    torch.set_num_threads(1)
    while not (OUT/'action-bank.json').is_file():time.sleep(5)
    registry=json.loads((OUT/'action-bank.json').read_text())
    todo={(k,s) for k in ('blackbox','structured') for s in (5901,5902,5903)}
    while todo:
        for kind,seed in sorted(todo):
            path=WORK/'checkpoints'/kind/str(seed)/'node-10000.pt';dest=OUT/f'actions-{kind}-{seed}.json'
            if not path.is_file():continue
            if not dest.is_file():calibrate(kind,seed,registry)
            todo.remove((kind,seed))
        if todo:time.sleep(10)

if __name__=='__main__':main()
