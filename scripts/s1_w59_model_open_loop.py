"""Open-loop historical development evaluation by controller identity, phase and physical horizon."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from numba import njit
from s1_w59_model import ExecutionMember,advance_world,query_world
from s1_w59_model_bank import OUT,WORK,ROOT,write
from s1_w59_model_cells import update_cells,NX,NY
from s1_w59_model_fields import STATE,INTERVAL,PHASES,DT,NI,BG

TIMES=(.02,.10,.35,.64,1.30,2.)

@njit(cache=True)
def origin_tables(feet,fz,indices):
    table=np.zeros((NX*NY,5));table[:,1:4]=-1.;current=np.full((4,9),-1,np.int32);j=0
    result=np.empty((len(indices),NX*NY,5),np.float32)
    for i in range(indices[-1]*10+1):
        current=update_cells(table,current,i*.002,feet[i],fz[i])
        if i==10*indices[j]:
            result[j]=table
            j+=1
            if j==len(indices):break
    return result


def absmean(x):return float(np.mean(abs(x)))


def rollpitch(q):
    w,x,y,z=q[:4];return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1))])


def prepare():
    dest=WORK/'historical-origins';dest.mkdir(exist_ok=True)
    rows=json.loads((OUT/'bank.json').read_text())['episodes']
    fit=set()
    for row in rows:
        if 'intervals'in row and row['split']=='fit':
            with np.load(WORK/'aligned'/(row['episode']+'.npz')) as d:fit.update(d['fingerprint'].tolist())
    totals=[];ana=[]
    for row in rows:
        if 'intervals' not in row or row['split']!='development_validation':continue
        ep=row['episode'];p=Path(row['root']);r=WORK/'rigid'/(ep+'.npy')
        if not r.exists():continue
        with np.load(WORK/'aligned'/(ep+'.npz')) as z:
            x=z['state'];u=z['action'];y=z['interval'];w=z['valid'];ts=z['time'];h=z['fingerprint']
        valid=~np.isin(h,np.fromiter(fit,dtype=np.uint64));time_ix=[]
        for phase in np.unique(x[:,STATE['phase']].argmax(1)):
            options=np.flatnonzero(valid&(x[:-1,STATE['phase']].argmax(1)==phase))
            if len(options):time_ix.extend(options[np.linspace(0,len(options)-1,min(len(options),3),dtype=int)].tolist())
        if not time_ix:continue
        ix=np.asarray(sorted(set(time_ix)))
        with np.load(p/'trajectory.npz') as z:
            feet=z['feet'];grf=z['actual_grf'];q=z['qpos'];rpy=z['rpy']
        world=origin_tables(feet,grf[:,:,2],ix)
        refs=[];solves=p/'w55-f/mpc-solves.jsonl'
        if solves.exists():refs=[json.loads(v) for v in solves.open()]
        call_times=np.array([t['time_s'] for t in refs]);metrics=np.full((len(ix),len(TIMES),4),np.nan)
        phys=np.full((len(ix),len(TIMES),20),np.nan)
        interval=np.full((len(ix),len(TIMES),NI),np.nan)
        for j,i in enumerate(ix):
            k=np.searchsorted(call_times,ts[i]+1e-8,side='right')-1
            plan=p/'w55-f'/refs[k]['file'] if k>=0 else None
            source=np.load(plan) if plan is not None and plan.is_file() else None
            if source is not None:ptime=source['times'];feet_plan=source['foot_center_plan'].reshape(-1,12);fplan=source['forces']
            for d,sec in enumerate(TIMES):
                end=10*i+int(round(sec/.002));horizon=ts[i]+sec
                if end>=len(q):continue
                phys[j,d]=np.r_[feet[end].ravel(),grf[end,:,2],rpy[end,:2],q[end,2],q[end,0]]
                b=round(sec/.02)
                if abs(b*.02-sec)<1e-8 and i+b<=len(y):
                    vv=w[i:i+b].all(0);interval[j,d,vv]=y[i:i+b].sum(0)[vv]
                if source is None or horizon>ptime[-1]+1e-7:continue
                f=np.array([np.interp(horizon,ptime,feet_plan[:,c]) for c in range(12)])
                stage=int(np.clip(np.searchsorted(ptime,horizon-1e-8,side='right')-1,0,len(fplan)-1))
                orient=np.array([np.interp(horizon,ptime,np.unwrap(source['states'][:,c])) for c in (0,1)])
                metrics[j,d,:3]=[absmean(f-phys[j,d,:12]),absmean(fplan[stage,:,2]-phys[j,d,12:16]),absmean(orient-phys[j,d,16:18])]
            if source is not None:source.close()
        each=dest/(ep+'.npz')
        with each.open('wb') as f:np.savez(f,state=x[ix],action=u[ix],world=world,rigid=np.load(r,mmap_mode='r')[ix],physical=phys,interval=interval,analytic=metrics,index=ix)
        ident=row['source_identity'].get('controller_action_identity') or ep.rsplit('-a0',1)[0]
        totals.append(dict(episode=ep,controller=ident,source_tree=row['source_tree'],n=len(ix),source_category=row['source_group']))
        for phase in sorted(set(x[ix,STATE['phase']].argmax(1).tolist())):
            use=x[ix,STATE['phase']].argmax(1)==phase
            for j,horizon in enumerate(TIMES):
                z=metrics[use,j];valid=np.isfinite(z[:,0])
                ana.append(dict(episode=ep,controller=ident,phase=PHASES[phase],horizon_s=horizon,origins=int(use.sum()),archived_MPC_supported=int(valid.sum()),
                    values=(dict(foot_xyz_MAE_m=float(z[valid,0].mean()),foot_fz_MAE_N=float(z[valid,1].mean()),roll_pitch_MAE_rad=float(z[valid,2].mean())) if valid.any() else None)))
    write(OUT/'historical-origins.json',dict(episodes=totals,rules='At most three evenly spaced independent-case development origins per observed phase per source, exclude any exact complete physical start transition seen on A; no W59 intervention or heldout data.',input_version='4-vector identifies only W56/W57/W58/W59 generation; exact controller/action identity indexed as metadata; old qualification is not transferred',**BG))
    import gzip
    with gzip.open(WORK/'analytic-original-M1.json.gz','wt') as f:json.dump(ana,f,allow_nan=False)
    summary=[]
    for sec in TIMES:
        ss=[r for r in ana if r['horizon_s']==sec];n=sum(r['archived_MPC_supported'] for r in ss)
        summary.append(dict(horizon_s=sec,origins=sum(r['origins'] for r in ss),supported=n,metrics={k:sum(r['archived_MPC_supported']*r['values'][k] for r in ss if r['values'])/n for k in ('foot_xyz_MAE_m','foot_fz_MAE_N','roll_pitch_MAE_rad')} if n else None))
    write(OUT/'analytic-baseline.json',dict(archived_original_force_MPC=summary,material_reference='recovery.json: M1 exact same 5 kHz observed trajectories; free-horizons.json for separate release scores',limit='Only the latest current source plan, 14x25ms; missing archived full plans and unsupported horizons are counted explicitly.',**BG))


def run_member(kind,seed,origins):
    member=ExecutionMember.load(WORK/'checkpoints'/kind/str(seed)/'node-10000.pt','cuda').eval();details=[];start=time.time()
    tensor=lambda a:torch.as_tensor(a,device='cuda',dtype=torch.float32)
    with torch.inference_mode():
        for row in origins:
            with np.load(WORK/'historical-origins'/(row['episode']+'.npz')) as f:
                state=tensor(f['state']);world=tensor(f['world']);action=tensor(f['action']);rigid=tensor(f['rigid'])
                truth=f['physical'];interval=f['interval'];ph=state[:,STATE['phase']].argmax(-1).cpu().numpy()
            n=len(state);series=[state];ys=[];alive=torch.ones(n,device='cuda',dtype=torch.bool);valid=[alive]
            for _ in range(100):
                nxt,y=member(query_world(state,world),action,rigid);advanced=advance_world(nxt,world,y[...,INTERVAL['load_mean']])
                world=torch.where(alive[:,None,None],advanced,world);state=torch.where(alive[:,None],query_world(nxt,world),state)
                series.append(state);ys.append(torch.where(alive[:,None],y,0.))
                issued=y[...,INTERVAL['request']].squeeze(-1)>=.5
                action=torch.where(issued[:,None],action.new_tensor([0.,.05,0.]),action)
                alive=alive&(y[...,INTERVAL['hard_failure']].squeeze(-1)<.5)&(y[...,INTERVAL['complete']].squeeze(-1)<.5);valid.append(alive)
            trajectory=torch.stack(series).cpu().numpy();yi=torch.stack(ys).cpu().numpy();live=torch.stack(valid).cpu().numpy()
            for i in range(n):
                for j,sec in enumerate(TIMES):
                    pos=sec/.02;lo=int(np.floor(pos+1e-7));hi=int(np.ceil(pos-1e-7));frac=pos-lo
                    rec=dict(episode=row['episode'],controller=row['controller'],phase=PHASES[ph[i]],horizon_s=sec,full_origin=1,
                        has_observation=bool(np.isfinite(truth[i,j,0])),prediction_active=bool(live[lo,i]))
                    pred=(1-frac)*trajectory[lo,i]+frac*trajectory[hi,i]
                    rec['finite']=bool(np.isfinite(pred).all())
                    if rec['has_observation'] and rec['finite']:
                        actual=truth[i,j]
                        rec.update(foot_xyz_MAE_m=absmean(pred[STATE['feet']]-actual[:12]),foot_fz_MAE_N=absmean(pred[STATE['measured_force']].reshape(4,3)[:,2]-actual[12:16]),roll_pitch_MAE_rad=absmean(rollpitch(pred[STATE['qpos']][3:7])-actual[16:18]),height_abs_error_m=float(abs(pred[STATE['qpos']][2]-actual[18])))
                        if lo==hi:
                            aggregate=yi[:lo,i].sum(0)
                            for name in ('work_J','slip_m','task_reward','touch','rebound','bearing_on','request'):
                                sl=INTERVAL[name];v=interval[i,j,sl]
                                if np.isfinite(v).all() and np.isfinite(aggregate[sl]).all():rec[name+'_integral_abs_error']=absmean(aggregate[sl]-v)
                    details.append(rec)
    import gzip
    file=WORK/f'{kind}-{seed}-historical-open-loop.json.gz'
    with gzip.open(file,'wt') as f:json.dump(details,f,allow_nan=False)
    groups=[]
    for h in TIMES:
        subset=[d for d in details if d['horizon_s']==h];metrics={}
        for key in ('foot_xyz_MAE_m','foot_fz_MAE_N','roll_pitch_MAE_rad','height_abs_error_m','work_J_integral_abs_error','slip_m_integral_abs_error','task_reward_integral_abs_error','touch_integral_abs_error','bearing_on_integral_abs_error'):
            vals=[r[key] for r in subset if key in r]
            metrics[key]=dict(n=len(vals),mean=float(np.mean(vals))) if vals else None
        groups.append(dict(horizon_s=h,origins=len(subset),observed=sum(r['has_observation'] for r in subset),predicted_active=sum(r['prediction_active'] for r in subset),finite=sum(r['finite'] for r in subset),metrics=metrics))
    write(OUT/f'{kind}-{seed}-open-loop.json',dict(kind=kind,seed=seed,backbone_step=10000,scope='Historical B model-development: absolute clocks from recordings, original phase/finite masks/whole origins retained. Successor nominal for learned unobserved future requests. At 0.35s interpolate adjacent predicted 20ms states; real endpoint at 500Hz.',per_horizon=groups,by_controller_and_phase=str(file.relative_to(ROOT)),elapsed_s=time.time()-start,**BG))


def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if not (OUT/'analytic-baseline.json').is_file():prepare()
    origins=json.loads((OUT/'historical-origins.json').read_text())['episodes']
    rest={(k,s) for k in ('blackbox','structured') for s in (5901,5902,5903)}
    while rest:
        for k,s in sorted(rest):
            if not (WORK/'checkpoints'/k/str(s)/'node-10000.pt').is_file():continue
            if not (OUT/f'{k}-{s}-open-loop.json').is_file():run_member(k,s,origins)
            rest.remove((k,s));print(k,s,'complete',flush=True)
        if rest:time.sleep(15)
    from publish_s1_w59 import ensure_release,publish
    ensure_release()
    outputs=[WORK/'analytic-original-M1.json.gz',*sorted(WORK.glob('*-historical-open-loop.json.gz'))]
    publish('w59-execution-model-open-loop',[(f.name,f) for f in outputs])

if __name__=='__main__':main()
