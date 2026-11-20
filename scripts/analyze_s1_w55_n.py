"""Summarize the complete recorded N task, preserving the first native constraint."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from publish_s1_go2_w54 import LARGE,write
from run_s1_w55_n import SMALL

def records(path):
    with path.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)

def read_policy(path):
    """Version 55.1 full installed policy, K and accepted trajectory."""
    with Path(path).open('rb') as handle:
        header=np.fromfile(handle,dtype='<i8',count=10)
        magic,version,call,h,nx,ndx,nu,nr,ntrace,rep=map(int,header)
        if (magic,version)!=(55,1):raise ValueError('Unsupported N policy record')
        meta=np.fromfile(handle,dtype='<f8',count=4)
        shapes={'reference_times':(h,),'reference_states':(h,nx),'reference_actions':(h,nu),
            'feedback_gain':(h,nu,ndx),'action_improvement':(h,nu),'accepted_times':(h,),
            'accepted_states':(h,nx),'accepted_actions':(h,nu),'accepted_residual':(h,nr),
            'accepted_costs':(h,),'accepted_trace':(h,ntrace,3),'applied_actions':(h,nu),
            'execution_diagnostics':(h,4,24)}
        arrays={key:np.fromfile(handle,dtype='<f8',count=int(np.prod(shape))).reshape(shape) for key,shape in shapes.items()}
    return dict(call=call,representation=rep,time_s=meta[0],feedback_scaling=meta[1],
                reference_J=meta[2],accepted_J=meta[3],**arrays)

def measure(case,stage):
    stage_file=case/(stage+'.json');state=json.loads(stage_file.read_text());attempt=state['attempt']
    out=LARGE/attempt;result=json.loads((SMALL/(attempt+'.json')).read_text())
    native=out/'w54-n-native';calls=list(records(native/'calls.jsonl'));nominal=list(records(native/'nominal-candidates.jsonl'))
    last=calls[-1];causes=collections.Counter()
    for row in nominal:
        if not row['feasible']:causes[(row['generator'],row['first_constraint_code'])]+=1
    for call in calls:
        for row in call['candidates']:
            if row['generator']=='action_search' and row['failure']:
                causes[(row['generator'],row['first_constraint_code'])]+=1
    evidence=dict(attempt=attempt,simulation_s=result['simulation_s'],physical_steps=result['physical_samples'],
        original_success=result['success'],reason=result['reason'],error=result.get('error'),
        qualification=result.get('qualification'),stage_progress=result.get('stage_progress'),
        zero_PPO_samples_updates_donor=[0,0,0],material_memory_causal_benefit_evaluated=False,
        legal_route=result.get('legal_route'),peak_recorded_strain=result.get('material',{}).get('max_abs_principal_strain'),
        nominal_recovery_feasible=sum(x['feasible'] for x in nominal if x['generator']=='support_recovery'),
        recovery_installs=sum(x.get('selection_source')=='support_recovery' for x in calls),
        count_by_generator_and_first_constraint=[dict(generator=k[0],code=int(k[1]),count=v) for k,v in causes.items()],
        solver_calls=len(calls),last_call=dict(call_id=last['call_id'],time_s=last['time_s'],controller_infeasible=last['controller_infeasible'],
            nominal=[{k:r[k] for k in ('slot','generator','feasible','first_constraint_code','first_constraint_time_s')} for r in nominal if r['call_id']==last['call_id']],
            actions=[{k:r[k] for k in ('slot','generator','failure','first_constraint_code','first_constraint_time_s')} for r in last['candidates']]))
    execution=list(records(out/'w54-n-execution.jsonl'))
    if execution:
        times=np.array([r['time_s'] for r in execution]);d=np.array([r['diagnostic'] for r in execution]);raw=np.array([r['raw_action'] for r in execution]);applied=np.array([r['applied_action'] for r in execution])
        with np.load(out/'trajectory.npz') as z:
            t=z['time'];q=z['qpos'];grf=z['actual_grf'];vel=z['feet_velocity']
            idx=np.searchsorted(t,times);idx=np.clip(idx,0,len(t)-1)
            active=d[:,:,4]>0;active_tick=active.any(axis=1)
            loaded=grf[idx,:,2]>10
            leg_change=np.max(np.abs(raw-applied).reshape(-1,4,3),axis=2)
            nearest=idx+1
            match=(nearest<len(t)) & np.isclose(t[np.minimum(nearest,len(t)-1)]-times,.002,atol=1.e-7,rtol=0)
            masks=active&match[:,None];speed_diff=vel[np.minimum(nearest,len(t)-1),:,2]-d[:,:,12]
            foot_braking=d[:,:,4]>0
            minimum=lambda a:float(np.min(a)) if a.size else None
            maximum=lambda a:float(np.max(a)) if a.size else None
            action_masks=active_tick[:,None] & np.broadcast_to(True,loaded.shape)
            # Receding-horizon policies are replaced every 20 ms. Only their
            # first executed interval is a prediction/actual error; the full
            # 350 ms prediction is reported as the direction the optimizer chose.
            windows={}
            for call in calls:
                now=call['time_s'];key=int(now)
                if now<2 or not call['installed'] or now+.02>t[-1]+1e-8:continue
                pi=read_policy(native/f"policy-{call['call_id']:06d}.bin")
                if not np.isclose(pi['accepted_times'][0],now,atol=1e-8,rtol=0):continue
                dt_index=np.argmin(np.abs(pi['accepted_times']-now-.02))
                i=np.argmin(abs(t-now));j=np.argmin(abs(t-now-.02))
                if abs(t[j]-now-.02)>1e-7:continue
                px=pi['accepted_states'][:,0]
                span=px[-1]-px[0]
                dx=px[dt_index]-px[0];actual=q[j,0]-q[i,0]
                feasible=[row['dx_m'] for row in call['candidates'] if not row['failure']]
                windows.setdefault(key,[]).append((dx,actual,dx-actual,span,len(feasible),sum(v>0 for v in feasible)))
            evidence['same_clock_progress_by_physical_second']=[]
            for key,rows in sorted(windows.items()):
                w=np.array(rows)
                evidence['same_clock_progress_by_physical_second'].append(dict(
                    start_s=key,installed_intervals=len(w),mean_predicted_20ms_dx_m=float(w[:,0].mean()),
                    mean_actual_20ms_dx_m=float(w[:,1].mean()),prediction_20ms_dx_RMS_error_m=float(np.sqrt(np.mean(w[:,2]**2))),
                    mean_installed_full_horizon_dx_m=float(w[:,3].mean()),feasible_recorded_candidates=int(w[:,4].sum()),
                    feasible_positive_x_candidates=int(w[:,5].sum())))
            evidence.update(max_x_m=float(np.max(q[:,0])),final_position_m=q[-1,:3].tolist(),min_base_z_m=float(np.min(q[:,2])),
                first_below_0p28_s=float(t[np.flatnonzero(q[:,2]<.28)[0]]) if (q[:,2]<.28).any() else None,
                stance_changes_per_foot=np.count_nonzero(np.diff((grf[:,:,2]>10).astype(int),axis=0),axis=0).tolist(),
                sim_actual_physical_interval=[result.get('physical_started_unix_s'),result.get('physical_ended_unix_s')],
                execution=dict(holds=len(d),braking_ticks=int(active_tick.sum()),braking_feet=int(foot_braking.sum()),
                  actual_above_10N_legs_changed_over_1e_6_rad=int(np.count_nonzero(loaded&action_masks&(leg_change>1e-6))),
                  max_action_change_support_rad=maximum(leg_change[action_masks&loaded]),
                  max_action_change_other_rad=maximum(leg_change[action_masks&~loaded]),
                  projected_minus_baseline_support_min_N=minimum((d[:,:,17]-d[:,:,16])[(d[:,:,10]>0)&action_masks]),
                  projected_minus_baseline_body_z_acc_min_m_s2=minimum((d[:,0,15]-d[:,0,14])[active_tick]),
                  peak_unattained_brake_m_s2=maximum(d[:,:,13][foot_braking]),
                  actual_next_foot_vz_minus_predicted_rms_m_s=float(np.sqrt(np.mean(speed_diff[masks]**2))) if masks.any() else None,
                  valid_next_foot_velocity_pairs=int(masks.sum())))
    write(case/(stage+'-actual-analysis.json'),evidence)
    return evidence

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('stage');a=p.parse_args();measure(a.case,a.stage)
