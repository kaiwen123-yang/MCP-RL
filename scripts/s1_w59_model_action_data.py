"""Physical 20 ms development sequences preserving the actual starting belief and mission."""
import gzip
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from s1_go2_legal_route import LegalRoute
from s1_w59_contact import contact_velocity_xy
from s1_w59_model_bank import ROOT, OUT, WORK, REL, read_requests, write
from s1_w59_model_cells import CausalCellMap
from s1_w59_model_fields import STATE,INTERVAL,PHASES,NS,NI,DT
from s1_w59_model_rigid import RigidLinearization

RAW=ROOT/'outputs'/REL


def episode(p,snapshot=None):
    with np.load(p/'trajectory.npz') as z:
        t,q,v,feet,vel,grf,cmd,contact,angle,phase,goal,direction,torque,cap=(z[k] for k in
            ('time','qpos','qvel','feet','feet_velocity','actual_grf','q_target','contact','rpy','phase','body_target','command_velocity','actuator_force','requested_speed_cap'))
    t0=t[0];m=(len(t)-1)//10;n=m+1;ts=t[np.arange(n)*10];n500=m*10
    state=np.zeros((n,NS),'f4');out=np.zeros((m,NI),'f4');valid=np.ones((m,NI),bool)
    choose=np.arange(n)*10
    for k,a in [('qpos',q),('qvel',v),('feet',feet),('foot_velocity',vel),('goal',goal),('command',direction)]:
        state[:,STATE[k]]=a[choose].reshape(n,-1)
    state[:,STATE['time']]=ts[:,None];state[:,STATE['version'].stop-1]=1
    state[np.arange(n),STATE['phase'].start+phase[choose]]=1
    action,request,execution,_=read_requests(p/'w55-f/handover-requests.jsonl',ts,state)
    early_request=[]
    for k in range(m):
        early_request.append(bool(np.any((request>ts[k]+1e-8)&(request<ts[k+1]-1e-8))))
    cc={}
    for path in sorted((p/'w55-f').glob('control-*.npz')):
        with np.load(path) as z:
            for k in ('time','w59_measured_grf','w59_requested_wbc_force','force','load_bearing_contact','confirmed_contact','transfer_fraction','w59_request_id','w59_preference_active'):
                cc.setdefault(k,[]).append(z[k])
    c={k:np.concatenate(a) for k,a in cc.items()}
    if not np.allclose(c['time'][:n500+1],t[:n500+1],atol=1e-7,rtol=0):
        raise ValueError(f'{p.name}: controller and trajectory clocks differ')
    grf=c['w59_measured_grf']
    state[:,STATE['measured_force']]=grf[choose].reshape(n,12)
    km=np.maximum(choose-1,0)
    for k,a in [('held_command',cmd),('force_reference',c['w59_requested_wbc_force']),('force_solved',c['force']),
                ('confirmed',c['confirmed_contact']),('bearing',c['load_bearing_contact']),('transfer',c['transfer_fraction'])]:
        state[:,STATE[k]]=a[km].reshape(n,-1)
    state[:,STATE['control_valid']]=1.
    if snapshot is None:
        world=CausalCellMap();route=LegalRoute();laps=0;previous=-1
        hold_from=None;lastcap=.25
        state[0,STATE['held_command']]=q[0,7:];state[0,STATE['control_valid']]=0
    else:
        ctrl=snapshot['controller'];world=pickle.loads(pickle.dumps(ctrl['causal_cells']))
        route=pickle.loads(pickle.dumps(snapshot['joint']['legal_route']))
        laps=snapshot['mission'].get('laps',0);previous=PHASES.index(snapshot['mission']['phase'])
        hold_from=snapshot['mission'].get('stable_since');lastcap=snapshot['joint']['applied_cap']
        src=RAW/snapshot['source_attempt']/'trajectory.npz';i=snapshot['step']
        with np.load(src) as z:state[0,STATE['held_command']]=z['q_target'][i-1]
        state[0,STATE['force_reference']]=ctrl['w59_force_ref'].ravel()
        state[0,STATE['force_solved']]=ctrl['last_force'].ravel()
        state[0,STATE['confirmed']]=ctrl['confirmed'];state[0,STATE['bearing']]=ctrl['load_bearing']
        state[0,STATE['transfer']]=ctrl['transfer_fraction']
    hold=np.zeros(n500+1);delta_cap=np.zeros(n500+1)
    for i in range(n500+1):
        ph=int(phase[i]);now=t[i]
        if ph in (5,8) and ph!=previous:laps+=1
        route.update(now,SimpleNamespace(phase=PHASES[ph],laps=laps),q[i,:2],feet[i],grf[i])
        endpoint=np.array([3.,0.]) if ph==6 else np.zeros(2)
        stable=ph in (4,5,6,7) and contact[i].all() and np.linalg.norm(q[i,:2]-endpoint)<.15 and np.linalg.norm(v[i,:3])<.05 and np.linalg.norm(v[i,3:6])<.1 and np.max(np.abs(angle[i,:2]))<.15
        if not stable:hold_from=None
        elif hold_from is None:hold_from=now
        hold[i]=0 if hold_from is None else now-hold_from
        if ph in (1,3):
            delta_cap[i]=((cap[i]-lastcap)/.035)**2;lastcap=cap[i]
        previous=ph;world.observe(now,feet[i],grf[i])
        if i%10==0:
            j=i//10
            mem,mv=world.query(now,feet[i],state[j,STATE['target_xy']].reshape(4,2),state[j,STATE['target_valid']])
            state[j,STATE['memory']]=mem.ravel();state[j,STATE['memory_valid']]=mv
            for k,val in [('hold',hold[i]),('progress',route.progress),('laps',route.completed_laps),('route_valid',1)]:
                state[j,STATE[k]]=val
    def label(k,values):out[:,INTERVAL[k]]=np.asarray(values).reshape(m,-1)
    def bins(a):return a[:n500].reshape((m,10)+a.shape[1:])
    label('bearing_on',bins(c['load_bearing_contact'][1:]&~c['load_bearing_contact'][:-1]).any(1))
    label('work_J',bins(np.maximum(torque*v[:,6:],0).sum(1)).sum(1)*.002)
    label('z_min',bins(q[1:,2]).min(1))
    label('tilt_max_rad',bins(np.abs(angle[1:,:2]).max(1)).max(1))
    label('hold_broken',bins((hold[:-1]>0)&(hold[1:]==0)).any(1))
    label('complete',bins(phase[1:]==5).any(1))
    label('progress_delta',np.diff(state[:,STATE['progress']],axis=0))
    label('cap_change',bins(delta_cap).sum(1))
    for name,clock in [('request',request),('handover',execution)]:
        ix=(np.rint((np.asarray(clock)-t0)/.0002).astype(int)-1)//100
        use=(ix>=0)&(ix<m);out[ix[use],INTERVAL[name]]=1
        if name=='handover':
            valid[:,INTERVAL['handover_at']]=False
            out[ix[use],INTERVAL['handover_at']]=(np.asarray(clock)[use]-ts[ix[use]])[:,None]
            valid[ix[use],INTERVAL['handover_at']]=True
    pmap={};last=grf[0,:,2]>1
    for path in sorted(p.glob('physical-*.npz')):
        with np.load(path) as z:
            xx=z['physical']
        start=int(round((xx[0,0]-t0)/.0002));pmap[start]=xx
    material=[]
    for f in sorted(p.glob('material-[0-9][0-9][0-9][0-9].npz')):
        with np.load(f) as z:material.append(z['foot'])
    xx=np.concatenate(list(pmap.values()))[:100*m];foot=np.concatenate(material)[:100*m]
    if abs(xx[0,0]-ts[0])>1e-7 or not np.allclose(xx[:,0],foot[:,0],rtol=0,atol=1e-7):raise ValueError('unmatched material physics clocks')
    fz=xx[:,32:42:3];loaded=fz>1.;prev=np.vstack((last,loaded[:-1]))
    soft=foot[:,2:6].astype(bool);active=soft&loaded
    vel=xx[:,73:85].reshape(-1,4,3)
    cp=contact_velocity_xy(foot,vel,np.full(4,.022))
    def physical(a):return a.reshape((m,100)+a.shape[1:])
    label('load_peak',physical(fz).max(1));label('load_mean',physical(fz).mean(1))
    label('torque_peak_Nm',physical(np.abs(xx[:,2:14]).max(1)).max(1))
    zload=np.maximum(-foot[:,8:18:3],0.)
    label('sink_peak',physical(zload*active).max(1))
    label('touch',physical(loaded&~prev).any(1))
    b=np.repeat(c['load_bearing_contact'][:n500],10,axis=0)
    label('rebound',physical(prev&~loaded&~b).any(1))
    label('slip_m',physical(np.linalg.norm(cp,axis=2)*loaded).sum(1)*.0002)
    ii=((zload/.001)**2*active).sum(1)/np.maximum(active.sum(1),1)
    so=(((foot[:,18:30].reshape(-1,4,3)[:,:,:2]/.25)**2).sum(2)*loaded).sum(1)/np.maximum(loaded.sum(1),1)
    label('indentation_sq_s',physical(ii).sum(1)*.0002);label('old_slip_sq_s',physical(so).sum(1)*.0002)
    meta=OUT.parent/'episodes'/(p.name+'.json')
    record=json.loads(meta.read_text())
    out[-1,INTERVAL['hard_failure']]=float(record.get('stage_status')=='physical_failure')
    out[-1,INTERVAL['truncated']]=float(phase[-1]!=5 and not out[-1,INTERVAL['hard_failure']][0])
    reward=5*out[:,INTERVAL['progress_delta']]-DT/60-out[:,INTERVAL['indentation_sq_s']]/60-.1*out[:,INTERVAL['old_slip_sq_s']]/60-.01*out[:,INTERVAL['cap_change']]+10*np.diff(state[:,STATE['laps']],axis=0)-10*out[:,INTERVAL['hard_failure']]
    done=bins(phase==5)
    reward[done.all(1)]=0;valid[done.any(1)&~done.all(1),INTERVAL['task_reward']]=False
    label('task_reward',reward)
    engine=RigidLinearization(p/'model.mjb')
    rigid=np.stack([engine(x) for x in state]).astype('f4')
    return dict(state=state,action=action,interval=out,valid=valid,rigid=rigid,interior_request=np.array(early_request),time=ts)

def prepare():
    import time
    from s1_w59_model_fields import BG
    directory=WORK/'action-development'
    directory.mkdir(exist_ok=True,parents=True)
    contract=ROOT/'experiments/terrain_memory/w59-execution-action-calibration.json'
    while True:
        desc=sorted((ROOT/'work/w59/interventions/development').glob('*.json'))
        completed=[]
        for path in desc:
            reg=json.loads(path.read_text())
            if reg['role']!='development' or reg['case']!='A' or not reg['snapshot_id'].startswith('w59-b3-a-original-e30-'):raise ValueError('Only the prespecified B3-A development source is authorized')
            job=reg['job'];raw=RAW/job;dest=directory/(job+'.npz')
            result=OUT.parent/'episodes'/(job+'.json')
            if not result.exists():continue
            detail=json.loads(result.read_text())
            if not detail.get('branch',{}).get('applied'):continue
            if not dest.exists():
                with gzip.open(reg['snapshot'],'rb') as f:state=pickle.load(f)
                values=episode(raw,state)
                with dest.with_suffix('.pending').open('wb') as f:np.savez(f,**values)
                dest.with_suffix('.pending').replace(dest)
                print(job,len(values['interval']),flush=True)
            completed.append(dict(job=job,source=reg['snapshot_id'],stratum=reg['stratum'],request=reg['replacement'],signed=reg['signed'],case='A'))
        originals={}
        for arm in ('b3','fixed-action'):
            for case in ('a','b'):
                names=sorted(RAW.glob(f'w59-{arm}-{case}-original-e60-a*'))
                done=[p for p in names if (OUT.parent/'episodes'/(p.name+'.json')).is_file() and (p/'trajectory.npz').is_file()]
                if not done:continue
                raw=done[-1];dest=directory/(raw.name+'.npz')
                if not dest.exists():
                    values=episode(raw)
                    with dest.with_suffix('.pending').open('wb') as f:np.savez(f,**values)
                    dest.with_suffix('.pending').replace(dest)
                originals[arm+'/'+case]=raw.name
        write(OUT/'action-bank-progress.json',dict(leaves=len(completed),source_count=len({r['source'] for r in completed}),originals=originals,recipe=str(contract.relative_to(ROOT)),**BG))
        if len(completed)==96 and len(originals)==4:
            write(OUT/'action-bank.json',dict(development=completed,originals=originals,
                groups={'fit':'B3-A development legal snapshots share one trajectory; Fixed-A and B3-A full E60 are a paired nominal case, their E30 prefixes excluded',
                        'development_validation':'entire Fixed-B/B3-B E60, no refit','sealed':'no data loaded','before_after':'no data loaded'},
                shared_prefix='B3 nominal A/B and W57 M1/B original E30 histories coincide; these states are development only, never used as unseen admission',**BG))
            return
        time.sleep(20)


if __name__=='__main__':prepare()

