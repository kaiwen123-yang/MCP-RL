"""Align pre-decision controls and original W38 reward to the 5 kHz subwindows."""
import json
import time
from pathlib import Path

import numpy as np
from s1_w59_model_bank import WORK, OUT, ROOT, REL, asset_identity, write, read_requests
from s1_w59_model_fields import STATE, INTERVAL, NI, BG
from s1_w59_contact import contact_velocity_xy


def enrich(row):
    p=Path(row['root']);bank=WORK/'bank'/f"{row['episode']}.npz"
    with np.load(bank) as f:
        state,action,old,mask,ts=(f[k] for k in ('state','action','interval','valid','time'))
    m=len(old);out=np.zeros((m,NI),np.float32);ok=np.zeros((m,NI),bool)
    out[:,:old.shape[1]]=old;ok[:,:old.shape[1]]=mask
    # The request is made before today's WBC solve. Use only commands and
    # solved forces from the last completed 2ms tick.
    with np.load(p/'trajectory.npz') as z:
        q=z['qpos'];prev=np.maximum(np.arange(len(state))*10-1,0)
        state[:,STATE['held_command']]=z['q_target'][prev]
        state[0,STATE['held_command']]=q[0,7:];cap=z['requested_speed_cap']
        phase=z['phase'];active=np.isin(phase,[1,3]);n500=10*m
    state[:,STATE['control_valid']]=0
    for k in ('force_reference','force_solved','transfer'):state[:,STATE[k]]=0
    for f in sorted((p/'w55-f').glob('control-*.npz')):
        with np.load(f) as z:
            ticks=np.rint(z['time']/.002).astype(int);sel=(ticks%10==9)&(ticks<10*m)
            ix=(ticks[sel]+1)//10
            if not len(ix):continue
            for src,k in [('redistributed_force_target','force_reference'),('force','force_solved'),('transfer_fraction','transfer'),('load_bearing_contact','bearing'),('confirmed_contact','confirmed')]:
                if src in z:state[ix,STATE[k]]=z[src][sel].reshape(len(ix),-1)
            state[ix,STATE['control_valid']]=int('redistributed_force_target' in z)
    for k in ('incoming','target_xy','target_in','target_valid','request_age','nominal_due_in','active'):
        state[:,STATE[k]]=0
    action,_,_,_=read_requests(p/'w55-f/handover-requests.jsonl',ts,state)
    # Compute cap changes exactly on moving ticks, as in Joint.before_physics.
    last=.25;delta=np.zeros(n500)
    for i in np.flatnonzero(active[:n500]):
        if cap[i]!=last:
            delta[i]=((cap[i]-last)/.035)**2;last=cap[i]
    cap_delta=delta.reshape(m,10).sum(axis=1)
    out[:,INTERVAL['cap_change']]=cap_delta[:,None];ok[:,INTERVAL['cap_change']]=True
    # Mark missing segments individually; the technical cut is not a fall.
    for k in ('slip_m','indentation_sq_s','old_slip_sq_s','task_reward'):
        ok[:,INTERVAL[k]]=False
    phys=None;phys_idx=-1
    for path in sorted(p.glob('material-[0-9][0-9][0-9][0-9].npz')):
        with np.load(path) as z:mat=z['foot']
        if not len(mat):continue
        first=int(round(mat[0,0]/.0002))
        if first%100:raise ValueError('material transition is not at its declared source clock')
        sub=first//150000;src=p/f'physical-{sub:03d}.npz'
        if sub!=phys_idx:
            phys_idx=sub
            if src.exists():
                with np.load(src) as z:phys=z['physical']
            else:phys=None
        if phys is None or phys.shape[1]<85:continue
        idx=first-int(round(phys[0,0]/.0002))
        count=min(len(mat),len(phys)-idx,100*m-first)//100*100
        if count<=0:continue
        mat=mat[:count];xx=phys[idx:idx+count]
        if abs(mat[-1,0]-xx[-1,0])>1e-7:raise ValueError('physical and applied material boundary clocks disagree')
        zpos=mat[:,8:18:3];fz=xx[:,32:42:3];loaded=fz>1.;soft=loaded&mat[:,2:6].astype(bool)
        indent=(np.maximum(-zpos,0.)/.001)**2
        slip2=(mat[:,18:30].reshape(-1,4,3)[:,:,:2]**2).sum(axis=2)/.25**2
        cp=contact_velocity_xy(mat,xx[:,73:85].reshape(-1,4,3),np.full(4,.022))
        s=slice(first//100,(first+count)//100)
        for key,value in [('indentation_sq_s',(indent*soft).sum(axis=1)/np.maximum(soft.sum(axis=1),1)),('old_slip_sq_s',(slip2*loaded).sum(axis=1)/np.maximum(loaded.sum(axis=1),1))]:
            out[s,INTERVAL[key]]=value.reshape(-1,100).sum(axis=1)[:,None]*.0002;ok[s,INTERVAL[key]]=True
        out[s,INTERVAL['slip_m']]=(np.linalg.norm(cp,axis=2)*loaded).reshape(-1,100,4).sum(axis=1)*.0002
        ok[s,INTERVAL['slip_m']]=True
    ident=asset_identity(p)
    final=ident['final_reason']
    if final:
        out[:,INTERVAL['truncated']]=0.;out[:,INTERVAL['hard_failure']]=0.
        # horizon is an observation boundary. Absorbed task completion is separate.
        if final in ('stage_horizon_reached','stage_passed','mission_complete','completed'):
            if phase[-1]!=5:out[-1,INTERVAL['truncated']]=1.
        elif final in ('material_domain_exit','body_fall','nonfoot_ground_contact','controller_infeasible','forbidden_ground_contact','controller_numeric_failure'):
            out[-1,INTERVAL['hard_failure']]=1.
        else:
            ok[-1,INTERVAL['hard_failure']]=False;ok[-1,INTERVAL['truncated']]=False
    dlap=np.diff(state[:,STATE['laps']],axis=0)
    rew=(5*out[:,INTERVAL['progress_delta']]-.02/60-out[:,INTERVAL['indentation_sq_s']]/60
         -.1*out[:,INTERVAL['old_slip_sq_s']]/60-.01*out[:,INTERVAL['cap_change']]+10*dlap-10*out[:,INTERVAL['hard_failure']])
    mask=np.logical_and.reduce([ok[:,INTERVAL[k]] for k in ('progress_delta','indentation_sq_s','old_slip_sq_s','hard_failure')])
    at=(phase[:n500].reshape(m,10)==5)
    rew[at.all(axis=1)]=0.;mask[at.all(axis=1)]=True
    # A straddling interval needs the exact absorbing clock; no full-window reward inserted.
    mask[at.any(axis=1)&~at.all(axis=1)]=False
    out[:,INTERVAL['task_reward']]=rew;ok[:,INTERVAL['task_reward']]=mask
    tmp=WORK/'aligned'/bank.name
    with np.load(bank) as z:
        with tmp.with_suffix('.tmp').open('wb') as dst:
            np.savez(dst,state=state,action=action,interval=out,valid=ok,fingerprint=z['fingerprint'],time=ts)
    tmp.with_suffix('.tmp').replace(tmp)
    return dict(episode=row['episode'],controller_identity=ident,intervals=m,
        reward_valid=int(ok[:,INTERVAL['task_reward']].sum()),slip5k_intervals=int(ok[:,INTERVAL['slip_m'].start].sum()),
        solved_previous_boundaries=int(state[:,STATE['control_valid']].sum()),split=row['split'])


def main():
    (WORK/'aligned').mkdir(exist_ok=True,parents=True);out=[];src=OUT/'source-inventory.json'
    inventory=json.loads(src.read_text())['attempts']
    start=time.time()
    for row in inventory:
        if not row['trajectory']:continue
        entry=WORK/'bank'/f"{row['episode']}.json"
        while not entry.is_file():time.sleep(5)
        receipt=WORK/'aligned'/entry.name
        if receipt.exists():out.append(json.loads(receipt.read_text()))
        else:
            print(row['episode'],flush=True)
            record=enrich(row);write(receipt,record);out.append(record)
        write(OUT/'aligned-progress.json',dict(processed=len(out),intervals=sum(r['intervals'] for r in out),
             reward=sum(r['reward_valid'] for r in out),current=row['episode'],elapsed_s=time.time()-start,**BG))
    write(OUT/'aligned-bank.json',dict(episodes=out,wall_s=time.time()-start,source_clock='5kHz contact point velocity and W38 material/cap reward; previous completed 2 ms WBC buffers as inputs',**BG))

if __name__=='__main__':main()
