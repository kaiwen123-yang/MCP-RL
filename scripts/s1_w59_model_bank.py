"""Build W59 20 ms causal sequences from retained complete W56–W58 raw clocks."""
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numba import njit

from s1_go2_legal_route import LegalRoute
from s1_w59_model_cells import series
from s1_w59_model_fields import BG, DT, INTERVAL, NI, NS, NOMINAL, PHASES, STATE

ROOT=Path(__file__).resolve().parents[1]
REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
WORK=ROOT/'work/w59/model'
OUT=ROOT/'results'/REL/'w59/model'


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def sources():
    for version, name in enumerate(('MCP-RL-w56-r60','MCP-RL-w57','MCP-RL-w58')):
        raw=ROOT.parent/name/'outputs'/REL
        for p in sorted(raw.iterdir()):
            if not p.is_dir() or not p.name.startswith(('job-','w57-','w58-')):continue
            match=re.search(r'-(a|b)-(?:r60-|original-|short|recovered|development-|validation-|e[0-9]+-)',p.name,re.I)
            if not match:continue
            req=p/'w55-f/handover-requests.jsonl'
            ctrl=p/'w55-f/control-00001.npz'
            case=match[1].upper()
            group=('E600' if 'e600' in p.name.lower() else 'training' if 'pilot' in p.name else
                   'old_validation' if 'validation' in p.name else 'mechanism' if any(x in p.name for x in ('short','recovered')) else 'qualification')
            yield dict(episode=p.name,root=str(p.resolve()),source_tree=name,version=version,case=case,
                       source_group=group,split='fit' if case=='A' else 'development_validation',
                       trajectory=(p/'trajectory.npz').is_file(),physical=(p/'physical-000.npz').is_file(),
                       controls=ctrl.is_file(),requests=req.is_file(),material_events=(p/'material-events.jsonl').is_file())


def read_requests(p, timestamps, state):
    n=len(timestamps);action=np.broadcast_to(NOMINAL,(n,3)).copy()
    times=[]; ends=[]; records=[]; compact=[]
    if not p.is_file():return action,np.empty(0),np.empty(0),compact
    for line in p.open():
        v=json.loads(line)
        if v['stage']=='request':
            m=v.get('macro_action',v.get('macro_request',v.get('w59_request',{})))
            if isinstance(m,list): m=dict(zip(('due_offset_s','transfer_ramp_s','rho'),m))
            nominal=v.get('original_due_s',v['nominal_due_s'])
            a=np.asarray([m.get('due_offset_s',v['requested_due_s']-nominal),m.get('transfer_ramp_s',.05),m.get('rho',0.)])
            records.append((float(v['sample_time_s']),float(nominal),v.get('incoming_pair',sorted(set(range(4))-set(v['planned_pair']))),v.get('plan_publication',[]),a))
            times.append(v['sample_time_s'])
            compact.append([v['decision_id'],v['sample_time_s'],nominal,*a])
        elif v['stage']=='executed' and v.get('actual_liftoff_s') is not None:
            ends.append(float(v['actual_liftoff_s']))
    times=np.array(times);ends=np.array(ends)
    state[:,STATE['request_valid']]=1
    for k,(t,due,pair,targets,a) in enumerate(records):
        lo=np.searchsorted(timestamps,t-1e-8)
        hi=np.searchsorted(timestamps,times[k+1]-1e-8) if k+1<len(times) else n
        s=slice(lo,hi);state[s,STATE['active']]=1
        state[s,STATE['incoming'].start+np.asarray(pair)]=1
        state[s,STATE['request_age']]=(timestamps[s]-t)[:,None]
        state[s,STATE['nominal_due_in']]=(due-timestamps[s])[:,None]
        action[s]=a
        for row in targets:
            if not row.get('valid') or row.get('xy') is None:continue
            leg=int(row['leg']);base=STATE['target_xy'].start+leg*2
            state[s,base:base+2]=row['xy']
            state[s,STATE['target_in'].start+leg]=row['target_time_s']-timestamps[s]
            state[s,STATE['target_valid'].start+leg]=1
    # An executed pending request is not retained as active after the last
    # handover when no next pair was published (e.g. stopping or endpoint hold).
    standing=state[:,STATE['phase']][:,[0,4,5,6,7]].sum(axis=1)>.5
    state[standing,STATE['active']]=0
    return action,times,ends,compact


def event_values(timestamps, events):
    # End-inclusive intervals: (t,t+.020]. Excludes just-issued request.
    ix=np.searchsorted(timestamps-1e-8,events,side='left')-1
    # floating drift in plant clocks is below 1e-7, convert timestamps to ticks.
    ix=(np.rint(np.asarray(events)/.0002).astype(np.int64)-1)//100
    use=(ix>=0)&(ix<len(timestamps)-1)
    return ix[use],np.asarray(events)[use]


@njit(cache=True)
def route_replay(q,qvel,phase,feet,grf,contact,rpy):
    n=len(q);result=np.zeros((n,3));progress=0.;credit=0.;laps=0
    valid=True;inp=False;entered=False;done=False;loaded=False;last_pass=False
    stable_since=-1.;raw_laps=0
    for i in range(n):
        ph=phase[i];prev=phase[i-1] if i else -1
        if i and prev in (1,3) and valid:
            dx=(q[i,0]-q[i-1,0])/6*(1 if prev==1 else -1);progress+=dx;credit+=dx
        if inp and ph!=prev:
            if not done and valid:progress-=max(credit,0.);credit-=max(credit,0.);valid=False
            if prev==1:last_pass=done
            inp=False
        if (ph==8 and prev!=8) or (ph==5 and prev!=5):
            laps+=int(valid and last_pass and done);raw_laps+=1;credit=0.;valid=True;last_pass=False
        if ph in (1,3):
            if not inp:inp=True;entered=False;done=False;loaded=False
            x=q[i,0];y=q[i,1];half=np.sqrt(.001)/2
            entered=entered or (x>=.8 if ph==1 else x<=2.2)
            if entered and not done:
                if abs(y)>.6 and valid:progress-=max(credit,0.);credit-=max(credit,0.);valid=False
                for l in range(4):
                    if .8+half<=feet[i,l,0]<=2.2-half and abs(feet[i,l,1])<=.6-half and grf[i,l,2]>0:loaded=True
                if x>=2.2 if ph==1 else x<=.8:
                    if loaded and valid:done=True
                    elif valid:progress-=max(credit,0.);credit-=max(credit,0.);valid=False
        endpoint=3. if ph==6 else 0.
        stable=(ph in (4,5,6,7) and contact[i].all() and
                (q[i,0]-endpoint)**2+q[i,1]**2<.15**2 and np.linalg.norm(qvel[i,:3])<.05 and
                np.linalg.norm(qvel[i,3:6])<.1 and np.max(np.abs(rpy[i,:2]))<.15)
        if stable:
            if stable_since<0 or prev!=ph:stable_since=i*.002
            hold=i*.002-stable_since
        else:stable_since=-1.;hold=0.
        result[i,0]=progress;result[i,1]=laps;result[i,2]=hold
    return result


def matrix_intervals(v,n):
    return v[:n*10].reshape((n,10)+v.shape[1:])


def asset_identity(p):
    manifest=p/'w55-f/manifest.json'
    v=json.loads(manifest.read_text()) if manifest.exists() else {}
    return {k:v.get(k) for k in ('schema','backend','w57_planner','w57_planner_history','w58_planner','final_reason','final_physical_time_s')} | {'controller_action_identity':v.get('config',{}).get('identity'),'mode':v.get('config',{}).get('controller')}


def extract(row):
    p=Path(row['root'])
    with np.load(p/'trajectory.npz') as f:
        q=f['qpos'];v=f['qvel'];feet=f['feet'];vel=f['feet_velocity'];phase=f['phase']
        t=f['time'];grf=f['actual_grf'];cmd=f['q_target'];contact=f['contact'];rpy=f['rpy']
        goal=f['body_target'];direction=f['command_velocity'];torque=f['actuator_force']
    # Retain only fully sampled consecutive plant intervals. A partial terminal
    # row may have a previous diagnostic clock and is not another force frame.
    m=min((len(t)-1)//10,int(round(t[-1]/.002))//10);take=slice(0,10*m+1,10)
    nt=m+1;n500=10*m;ts=t[take];state=np.zeros((nt,NS),np.float32)
    for k,d in [('qpos',q),('qvel',v),('feet',feet),('foot_velocity',vel),('measured_force',grf),('held_command',cmd),('goal',goal),('command',direction)]:
        state[:,STATE[k]]=d[take].reshape(nt,-1)
    state[np.arange(nt),STATE['phase'].start+phase[take]]=1
    state[:,STATE['version'].start+row['version']]=1;state[:,STATE['time']]=ts[:,None]
    mission=route_replay(q,v,phase,feet,grf,contact,rpy)
    for j,k in enumerate(('progress','laps','hold')):state[:,STATE[k]]=mission[take,j,None]
    state[:,STATE['route_valid']]=1
    action,req,execution,requests=read_requests(p/'w55-f/handover-requests.jsonl',ts,state)
    target=state[:,STATE['target_xy']].reshape(nt,4,2).astype(float)
    tv=state[:,STATE['target_valid']]
    mem,mv=series(feet[:n500+1],grf[:n500+1,:,2],target,tv,10)
    state[:,STATE['memory']]=mem.reshape(nt,-1);state[:,STATE['memory_valid']]=mv
    outcome=np.zeros((m,NI),np.float32);valid=np.zeros_like(outcome,dtype=bool)
    def label(k,value):
        outcome[:,INTERVAL[k]]=np.asarray(value).reshape(m,-1)
        valid[:,INTERVAL[k]]=True
    # A missing streamed field has its own validity mask and does not become a zero target.
    control_coverage=np.zeros(nt,bool);bearing_seq=np.zeros((n500+1,4),bool);cb_valid=np.zeros(n500+1,bool)
    for f in sorted((p/'w55-f').glob('control-*.npz')):
        with np.load(f) as z:
            idx=np.rint(z['time']/.002).astype(int);good=(idx>=0)&(idx<=n500)
            sample=good&(idx%10==0);i=idx[sample]//10
            for src,k in [('redistributed_force_target','force_reference'),('force','force_solved'),('confirmed_contact','confirmed'),('load_bearing_contact','bearing'),('transfer_fraction','transfer')]:
                if src in z and len(i):state[i,STATE[k]]=z[src][sample].reshape(len(i),-1)
            if 'load_bearing_contact' in z:
                bearing_seq[idx[good]]=z['load_bearing_contact'][good];cb_valid[idx[good]]=True
                control_coverage[i]=True
    state[:,STATE['control_valid']]=control_coverage[:,None]
    edge=bearing_seq[1:]&~bearing_seq[:-1]
    label('bearing_on',matrix_intervals(edge,m).any(axis=1))
    valid[:,INTERVAL['bearing_on']]=matrix_intervals(cb_valid[:-1]&cb_valid[1:],m).all(axis=1)[:,None]
    label('work_J',matrix_intervals(np.maximum(torque*v[:,6:],0).sum(axis=1),m).sum(axis=1)*.002)
    label('z_min',matrix_intervals(q[1:,2],m).min(axis=1))
    label('tilt_max_rad',matrix_intervals(abs(rpy[1:,:2]).max(axis=1),m).max(axis=1))
    label('progress_delta',np.diff(mission[take,0]))
    hold=mission[:,2];label('hold_broken',matrix_intervals((hold[:-1]>0)&(hold[1:]<=0),m).any(axis=1))
    label('complete',matrix_intervals(phase[1:]==5,m).any(axis=1))
    if row['requests']:
        for name,events in [('request',req),('handover',execution)]:
            idx,times=event_values(ts,events);label(name,np.zeros(m));outcome[idx,INTERVAL[name]]=1
            if name=='handover':
                outcome[idx,INTERVAL['handover_at']]=(times-ts[idx])[:,None];valid[idx,INTERVAL['handover_at']]=True
    # Only fields emitted by the physical recorder enter these 5 kHz labels.
    last=np.zeros(4,bool);physical_done=0
    for f in sorted(p.glob('physical-*.npz')):
        with np.load(f) as z:a=z['physical']
        first=int(round(a[0,0]/.0002))
        if first%100:raise ValueError(f'non-aligned raw chunk: {f}')
        start=first//100;length=min(len(a)//100,m-start)
        if length<=0:continue
        data=a[:length*100].reshape(length,100,-1);end=start+length
        fz=data[:,:,32:42:3]
        loaded=fz>1.;prior=np.concatenate((last.reshape(1,4),loaded.reshape(-1,4)[:-1])).reshape(length,100,4)
        last=loaded[-1,-1]
        for k,values in [('load_mean',fz.mean(axis=1)),('load_peak',fz.max(axis=1)),('touch',(loaded&~prior).any(axis=1)),('torque_peak_Nm',np.abs(data[:,:,2:14]).max(axis=(1,2))[:,None])]:
            outcome[start:end,INTERVAL[k]]=values;valid[start:end,INTERVAL[k]]=True
        # Rebound: measured loaded->off edge while the corresponding foot
        # has yet to establish confirmed bearing. Stance release excluded.
        b=bearing_seq[start*10:end*10].reshape(length,10,4).repeat(10,axis=1)
        outcome[start:end,INTERVAL['rebound']]=(prior&~loaded&~b).any(axis=1)
        valid[start:end,INTERVAL['rebound']]=cb_valid[start*10:end*10].reshape(length,10).all(axis=1)[:,None]
        if data.shape[2]>=85:
            xyz=data[:,:,61:73].reshape(length,100,4,3)
            interior=(xyz[:,:,:,0]>=.8+np.sqrt(.001)/2)&(xyz[:,:,:,0]<=2.2-np.sqrt(.001)/2)&(abs(xyz[:,:,:,1])<=.6-np.sqrt(.001)/2)
            outcome[start:end,INTERVAL['sink_peak']]=np.where(interior&loaded,np.maximum(.022-xyz[:,:,:,2],0),0).max(axis=1)
            valid[start:end,INTERVAL['sink_peak']]=True
        physical_done+=length
    # Rolling-point velocity is directly logged at 500 Hz. Flat terrain is
    # stationary. The 5 kHz material/rigid point reconstruction is joined
    # separately for the new W59 evidence and remains distinguishable here.
    slip=np.zeros((n500,4));slip_ok=np.zeros(n500,bool)
    for f in sorted((p/'w55-f').glob('control-*.npz')):
        with np.load(f) as z:
            if 'rolling_contact_velocity' not in z:continue
            idx=np.rint(z['time']/.002).astype(int);ok=(idx>=0)&(idx<n500)
            vel_cp=z['rolling_contact_velocity'][ok,:,:2];load=z['measured_grf'][ok,:,2]>1.
            slip[idx[ok]]=np.linalg.norm(vel_cp,axis=2)*load
            slip_ok[idx[ok]]=True
    label('slip_m',matrix_intervals(slip,m).sum(axis=1)*.002)
    valid[:,INTERVAL['slip_m']]=matrix_intervals(slip_ok,m).all(axis=1)[:,None]
    ident=asset_identity(p)
    reason=ident['final_reason']
    # Physics faults and scheduler interruptions never receive success labels.
    if reason:
        label('hard_failure',np.zeros(m));label('truncated',np.zeros(m))
        if reason not in ('stage_horizon_reached','mission_complete','completed'):
            if 'interrupt' in reason or 'wall' in reason:outcome[-1,INTERVAL['truncated']]=1
            else:outcome[-1,INTERVAL['hard_failure']]=1
    # Split rows retain the complete episode. Cross-case duplicated physical
    # starts are removed by window fingerprint when constructing fit/val batches.
    ids=q[take].astype('<f8').copy().view('uint64')
    import hashlib
    fingerprint=np.array([int.from_bytes(hashlib.blake2b(x.tobytes()+y.tobytes(),digest_size=8).digest(),'little') for x,y in zip(ids[:-1],ids[1:])],dtype=np.uint64)
    target=WORK/'bank'/f'{p.name}.npz'
    with target.with_suffix('.tmp').open('wb') as out:
        np.savez(out,state=state,action=action,interval=outcome,valid=valid,fingerprint=fingerprint,requests=np.asarray(requests),time=ts)
    target.with_suffix('.tmp').replace(target)
    return dict(intervals=m,simulated_s=float(ts[-1]),bytes=target.stat().st_size,source_identity=ident,
        physical_intervals=physical_done,control_boundaries=int(control_coverage.sum()),
        actions_observed=int(len(req)),action_min=action.min(axis=0).tolist(),action_max=action.max(axis=0).tolist(),
        targets_per_field={k:int(vv.sum()) for k,sl in INTERVAL.items() if (vv:=valid[:,sl]).size},
        final_legal_progress=float(mission[take,0][-1]),route_laps=int(mission[take,1][-1]))


def main():
    wall=time.time();cpu=time.process_time();(WORK/'bank').mkdir(exist_ok=True,parents=True)
    rows=list(sources());write(OUT/'source-inventory.json',dict(attempts=rows,
        historical_migration_catalog='H:/MPC+RL/00_catalog/migration-20261124-process/archive-index.sqlite',
        split='whole case A fit, whole case B model-development validation. W58 historical validation stays W59 development; sealed W59 action states excluded.',
        missing='D57 source paths are inspected per member. Missing fields have explicit masks; raw data never moved by this program.',**BG))
    for row in rows:
        print(row['episode'],flush=True)
        status=WORK/'bank'/f"{row['episode']}.json"
        if not row['trajectory']:
            row['use']='no synchronized 500 Hz trajectory on execution mount; original path and available partial streams retained'
        elif status.exists():
            row.update(json.loads(status.read_text()))
        else:
            row.update(extract(row));write(status,row)
        write(OUT/'bank-progress.json',dict(processed=sum(('intervals'in v or 'use'in v) for v in rows),total=len(rows),
            episodes_with_windows=sum('intervals'in v for v in rows),intervals=sum(v.get('intervals',0) for v in rows),current=row['episode'],elapsed_s=time.time()-wall,**BG))
    write(OUT/'bank.json',dict(episodes=rows,total_intervals=sum(v.get('intervals',0) for v in rows),
         source_sha='f11cd7c89654ff74fddc9d70dcb18169c2634094',clock_s=DT,wall_s=time.time()-wall,cpu_core_s=time.process_time()-cpu,**BG))


if __name__=='__main__': main()
