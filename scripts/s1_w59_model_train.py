"""Matched FP32 B/C model fitting on whole-case, cross-episode deduplicated banks."""
import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from s1_w59_model import BINARY, CONT, ExecutionMember
from s1_w59_model_bank import ROOT, WORK, OUT, write
from s1_w59_model_fields import STATE,INTERVAL,NS,NI,BG

H=7
SEEDS=(5901,5902,5903)
KINDS=('blackbox','structured')
NODES=(5000,10000,30000,50000)
BATCH=64


def create_bank():
    reviewed=json.loads((OUT/'aligned-bank.json').read_text())['episodes']
    inventory={r['episode']:r for r in json.loads((OUT/'source-inventory.json').read_text())['attempts']}
    X=[];U=[];Y=[];W=[];J=[];usable=[];offset=0;allfit=set();seen={'fit':set(),'development_validation':set()}
    candidates={};source_rows=[]
    # Include every available continuous recording as an individually traceable episode.
    for row in sorted(reviewed,key=lambda x:(x['split']!='fit',x['episode'])):
        name=row['episode'];split=row['split']
        op=WORK/'rigid'/(name+'.npy')
        if not op.is_file():source_rows.append(dict(episode=name,split=split,used=0,reason='recording-specific model binary missing'));continue
        with np.load(WORK/'aligned'/(name+'.npz')) as z:
            x,u,y,w,h=(z[k] for k in ('state','action','interval','valid','fingerprint'))
        # Old W57/W58 validation was available for this development only.
        # No W59 sealed admission, first-update or held-out states enter the bank.
        if inventory[name]['version']>2:raise ValueError(name)
        n=len(y);keep=np.ones(n,bool);matched_prefix=0;duplicate=0
        for i,fp in enumerate(h.tolist()):
            if split!='fit' and fp in allfit:keep[i]=False;matched_prefix+=1
            # Preserve different causal requests on the same measured path.
            key=(fp,u[i].tobytes(),inventory[name]['version'],bool(x[i,STATE['request_valid']][0]))
            if key in seen[split]:
                keep[i]=False;duplicate+=1
            seen[split].add(key)
        if split=='fit':allfit.update(h.tolist())
        # Each start has an entire seven-transition sequence in one episode.
        # Exclude all duplicate/shared constituent intervals, not just the first.
        ids=np.flatnonzero(np.convolve(keep.astype(int),np.ones(H,int),'valid')==H)
        flags=np.any(y[:,INTERVAL['touch']]>0,axis=1)|np.any(y[:,INTERVAL['bearing_on']]>0,axis=1)|(y[:,INTERVAL['request']].ravel()>0)
        candidates.setdefault(split,[]).append((ids+offset,ids[flags[ids]]+offset))
        source_rows.append(dict(episode=name,split=split,intervals=n,usable_starts=len(ids),
            exact_cross_case_intervals_excluded=matched_prefix,within_group_duplicate_intervals=duplicate,source_group=inventory[name]['source_group']))
        X.append(x);U.append(u);Y.append(np.vstack((y,np.zeros((1,NI),y.dtype))));W.append(np.vstack((w,np.zeros((1,NI),bool))))
        J.append(np.load(op,mmap_mode='r'));offset+=len(x)
    x=np.concatenate(X);u=np.concatenate(U);y=np.concatenate(Y);w=np.concatenate(W)
    lists={k:[np.concatenate([p[j] for p in v]) for j in range(2)] for k,v in candidates.items()}
    fit=lists['fit'][0]
    mean_s=x[fit].mean(axis=0,dtype=np.float64).astype('f4')
    floor=np.full(NS,.01,dtype='f4')
    floor[STATE['memory']]=np.tile([.01,2e-6,.02,.01],8)
    floor[STATE['request_age']]=.002;floor[STATE['nominal_due_in']]=.002
    scale_s=np.maximum(x[fit].std(axis=0,dtype=np.float64),floor)
    delta_s=np.maximum((x[fit+1]-x[fit]).std(axis=0,dtype=np.float64),floor*.15)
    mean_y=np.zeros(NI,'f4');scale_y=np.ones(NI,'f4');event_bias=np.zeros(NI,'f4')
    for j in range(NI):
        valid=w[fit,j];a=y[fit,j][valid]
        if len(a):
            mean_y[j]=a.mean(dtype=np.float64);scale_y[j]=max(a.std(dtype=np.float64),1e-6)
            if j in BINARY:
                rate=(float(a.sum())+.5)/(len(a)+1.)
                event_bias[j]=np.log(rate/(1-rate))
    stats=dict(mean_s=mean_s,scale_s=scale_s,delta_s=delta_s,mean_y=mean_y,scale_y=scale_y,event_bias=event_bias)
    # Put the shared preparation once on the GPU; each member has its own optimizer and sampled starts.
    cuda=lambda a:torch.as_tensor(np.asarray(a),device='cuda')
    rigid=torch.empty((len(x),18,25),device='cuda',dtype=torch.float32)
    off=0
    for mat in J:
        rigid[off:off+len(mat)].copy_(torch.from_numpy(np.array(mat)));off+=len(mat)
    tensors=tuple(cuda(a) for a in (x,u,y,w))+(rigid,)
    write(OUT/'split.json',dict(episodes=source_rows,starts={k:len(v[0]) for k,v in lists.items()},event_starts={k:len(v[1]) for k,v in lists.items()},normalization='fit windows only',**BG))
    return tensors,{k:[cuda(v) for v in a] for k,a in lists.items()},stats


def sequence_terms(member,states,actions,intervals,valid,rigid):
    state=states[:,0];weights=state.new_ones(len(state));loss=state.new_zeros(());event=state.new_zeros(());total=state.new_zeros(())
    for j in range(actions.shape[1]):
        nxt,target=states[:,j+1],intervals[:,j]
        ns,yp=member(state,actions[:,j],rigid)
        known=valid[:,j]; mask=known.index_select(1,member.cont_indices).to(state.dtype)
        er=(yp.index_select(1,member.cont_indices)-target.index_select(1,member.cont_indices))/member.scale_y.index_select(0,member.cont_indices)
        science=(F.smooth_l1_loss(er,torch.zeros_like(er),reduction='none')*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        bm=known.index_select(1,member.binary_indices).to(state.dtype)
        probability=yp.index_select(1,member.binary_indices)
        binary=(F.binary_cross_entropy(probability.clamp(1e-6,1-1e-6),target.index_select(1,member.binary_indices),reduction='none')*bm).sum(-1)/bm.sum(-1).clamp_min(1)
        brier=(((probability-target.index_select(1,member.binary_indices))**2)*bm).sum(-1)/bm.sum(-1).clamp_min(1)
        # The current action can change only at the next legal event. An
        # unissued future action never enters this open-loop supervision.
        available=weights*(1.-target[:,INTERVAL['request']].squeeze(-1)*known[:,INTERVAL['request']].squeeze(-1))
        e=(ns-nxt)/member.scale_s
        smask=torch.ones_like(e)
        for group,flag in [('held_command','control_valid'),('force_reference','control_valid'),('force_solved','control_valid'),('bearing','control_valid'),('confirmed','control_valid'),('transfer','control_valid'),('incoming','request_valid'),('target_xy','request_valid'),('target_in','request_valid'),('progress','route_valid'),('hold','route_valid')]:
            smask[:,STATE[group]]=nxt[:,STATE[flag]]
        state_err=(F.smooth_l1_loss(e,torch.zeros_like(e),reduction='none')*smask).sum(-1)/smask.sum(-1)
        loss=loss+(available*(state_err+science)+weights*binary).sum()
        event=event+(weights*brier).sum();total=total+weights.sum()
        state=ns;weights=available
    return loss/total.clamp_min(1),event/total.clamp_min(1)


def sequence_loss(member,states,actions,intervals,valid,rigid):
    return sequence_terms(member,states,actions,intervals,valid,rigid)[0]


def window(data,starts):
    x,u,y,w,rigid=data
    ids=starts[:,None]+torch.arange(H+1,device=starts.device)
    f=ids[:,:-1]
    return x[ids],u[f],y[f],w[f],rigid[starts]


@torch.no_grad()
def evaluate(member,data,starts):
    rows=[sequence_terms(member,*window(data,t)) for t in starts.split(256)]
    counts=[len(t) for t in starts.split(256)]
    return [sum(float(r[i])*c for r,c in zip(rows,counts))/sum(counts) for i in (0,1)]


def fit_member(kind,seed,data,ids,stats,val,recipe,through=50000,live_name="fit-live.json"):
    directory=WORK/'checkpoints'/kind/str(seed);directory.mkdir(parents=True,exist_ok=True)
    rec=OUT/f'{kind}-{seed}.json'
    if rec.exists() and json.loads(rec.read_text()).get('terminal'):return
    torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    member=ExecutionMember(kind,stats).cuda();opt=torch.optim.AdamW(member.parameters(),lr=.0003,weight_decay=.0001,fused=True,capturable=True)
    trace=[];count=0;actuals=[];start=time.time()
    latest=sorted(directory.glob('node-*.pt'),key=lambda p:int(p.stem.split('-')[-1]))
    if latest:
        checkpoint=torch.load(latest[-1],map_location='cuda',weights_only=False)
        member.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        torch.cuda.set_rng_state(checkpoint['metadata']['cuda_rng'].cpu());count=int(checkpoint['metadata']['updates'])
        if rec.exists():trace=json.loads(rec.read_text())['nodes']
    allrows,eventrows=ids
    def update():
        opt.zero_grad(set_to_none=True)
        a=allrows[torch.randint(len(allrows),(BATCH//2,),device='cuda')]
        b=eventrows[torch.randint(len(eventrows),(BATCH//2,),device='cuda')]
        err=sequence_loss(member,*window(data,torch.cat((a,b))))
        err.backward()
        torch.nn.utils.clip_grad_norm_(member.parameters(),2.)
        opt.step()
        return err
    steps=lambda:int(opt.state[next(member.parameters())]['step'].item())
    interval_wall=time.time();interval_count=count
    for node in NODES:
        if node<=count:continue
        while count<node:
            target=min(node,count+250)
            for _ in range(target-count):err=update()
            torch.cuda.synchronize();count=steps();loss=float(err)
            if not np.isfinite(loss):raise FloatingPointError(f'nonfinite fit {kind}/{seed} at step {count}')
            wall=time.time();actuals.append(dict(updates=count,wall_s=wall-start,updates_per_s=(count-interval_count)/(wall-interval_wall)))
            interval_count=count;interval_wall=wall
            if count%1000==0 or count==node:
                print(kind,seed,count,loss,actuals[-1]['updates_per_s'],flush=True)
                write(OUT/live_name,dict(kind=kind,seed=seed,accepted_updates=count,loss=loss,updates_per_s=actuals[-1]['updates_per_s'],
                    trainer_wall_s=wall-start,GPU_allocated_bytes=torch.cuda.memory_allocated(),**BG))
        vl,ev=evaluate(member,data,val)
        last=trace[-1] if trace else None
        keep=node==5000 or (node<50000 and last is not None and vl<last['validation_loss'] and ev<last['validation_event_brier'])
        meta=dict(kind=kind,seed=seed,updates=count,validation_loss=vl,validation_event_brier=ev,
            continue_next_node=keep,wall_s=time.time()-start,mean_updates_per_s=(count-(latest and checkpoint['metadata']['updates'] or 0))/max(time.time()-start,1e-6))
        path=directory/f'node-{node}.pt'
        member.save(path,opt,dict(**meta,cuda_rng=torch.cuda.get_rng_state()))
        trace.append(meta|dict(checkpoint=str(path.relative_to(ROOT)),bytes=path.stat().st_size))
        write(rec,dict(nodes=trace,terminal=not keep,stop='current validation loss and event Brier must both improve over preceding registered comparison' if not keep else None,
            batch=BATCH,supervised_substeps=H,window_duration_s=H*.02,precision='FP32, TF32 disabled',resources=actuals,recipe=recipe,**BG))
        if not keep or node>=through:break


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='historical');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    recipe=dict(seed=SEEDS,kinds=KINDS,width=128,optimizer='AdamW',lr=.0003,weight_decay=.0001,global_grad_norm=2,
        batch=BATCH,frames=H+1,transitions=H,starts='half uniform unique, half touch/bearing/request; cross-case duplicates removed',
        fixed_whole_episode_validation='case B historical model development; deterministic 4096 evenly spaced starts across eligible unique case-B intervals',
        advance='10000 vs 5000, 30000 vs 10000: strict improvement in both common validation loss and Brier, otherwise stop each seed/structure separately; 50000 terminal',
        operator='structured uses known current articulated 18x25 inverse dynamic map frozen across 7-step training window, total constrained held PD, learned actual contact impulse and acceleration residual; neither variant uses future real requests',
        private_sealed_states_used=0)
    write(OUT/'recipe.json',recipe|BG)
    while not (OUT/'aligned-bank.json').exists() or not (OUT/'rigid.json').exists():time.sleep(5)
    data,starts,stats=create_bank()
    val=starts['development_validation'][0]
    val=val[torch.linspace(0,len(val)-1,min(len(val),4096),device='cuda').long()]
    for seed in SEEDS:
        for kind in KINDS:fit_member(kind,seed,data,starts['fit'],stats,val,recipe)
    write(OUT/'fit-completion.json',dict(heads=[json.loads((OUT/f'{k}-{s}.json').read_text())['nodes'][-1] for s in SEEDS for k in KINDS],**BG))

if __name__=='__main__':main()
