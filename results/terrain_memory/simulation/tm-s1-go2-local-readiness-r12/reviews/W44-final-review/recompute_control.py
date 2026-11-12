#!/usr/bin/env python3
"""Recompute selected W44 control evidence from byte-verified saved originals.

Requires Python 3 and NumPy, never imports MuJoCo or runs the controller.
Download names/URLs/digests are listed in control-originals.json. Keep its 21
additional originals in --originals-dir and its four trajectory/call originals
in --raw-dir. No network access or simulation is performed by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path

"""Read-only alignment of W44 saved physical and planner records."""
import csv,gzip,json
from pathlib import Path
import numpy as np



def load_rows(name):
    return [json.loads(x) for x in gzip.open(HERE/name,'rt') if x.strip()]

def maxabs(x):return float(np.max(np.abs(x)))

def align_stream(stream):
    z=np.load(RAW/(stream+'-trajectory.npz'));tr={k:z[k] for k in z.files};t=tr['time']
    plans=load_rows(stream+'-planner-records.jsonl.gz')
    fb=load_rows(stream+'-feedback-actions.jsonl.gz')
    ex=load_rows(stream+'-action-execution.jsonl.gz')
    links={x['query_id']:x for x in ex if x['query_kind']=='applied_hold'}
    feedback=[]
    for x in fb:
        if x['query_id'] not in links:continue
        e=links[x['query_id']];at=e['time_s'];ix=int(np.argmin(abs(t-at)))
        feedback.append(dict(t=at,index=ix,plan_id=x['plan_id'],query=x['query_id'],
          reference_state_position_error=maxabs(x['state_delta'][:18]),
          reference_state_velocity_error=maxabs(x['state_delta'][18:]),
          raw_feedback=maxabs(x['raw_K_delta_x']),effective_feedback=abs(x['feedback_scaling'])*maxabs(x['raw_K_delta_x']),projection=maxabs(x['projection_delta']),
          nominal_target_gap=maxabs(np.array(x['u_held'])-tr['nominal_q_target'][ix]),
          u_matches_trajectory=maxabs(np.array(x['u_held'])-tr['q_target'][ix]),
          metric_valid=x['metric_valid'],qp_converged=x['qp_converged'],
          restoration=x['restoration'],feedback_scaling=x['feedback_scaling'],
          kkt=x['kkt_scaled_residual'],primal=x['primal_residual_rad'],
          serialized_torque=maxabs(x['serialization_delta_tau']),
          action_delta_from_previous=(maxabs(np.array(x['u_held'])-tr['q_target'][ix-1]) if ix else 0.)))
    prediction=[]
    for x in plans:
        origin=x['origin'];ix=int(np.argmin(abs(t-origin)))
        ns={n['node']:n for n in x.get('nodes',[])}
        if 0 not in ns or 10 not in ns:continue
        p0=ns[0];p=ns[10];it=int(np.argmin(abs(t-p['t'])))
        if abs(t[it]-p['t'])>1e-7:continue
        a=np.array(p['accepted_state']);s=np.r_[tr['qpos'][it],tr['qvel'][it]]
        zero=np.array(p0['accepted_state'])-np.r_[tr['qpos'][ix],tr['qvel'][ix]]
        rec=dict(origin=origin,t=p['t'],plan_id=x['plan_id'],call_id=x.get('source_call_id'),
          source=x.get('candidate_source'),alpha=x.get('action_step_alpha'),
          zero_q_error=maxabs(zero[:19]),zero_v_error=maxabs(zero[19:]),
          base_position_error=maxabs(a[:3]-s[:3]),joint_position_error=maxabs(a[7:19]-s[7:19]),
          velocity_error=maxabs(a[19:]-s[19:]),base_velocity_error=maxabs(a[19:25]-s[19:25]),
          predicted_base_z=a[2],actual_base_z=s[2],
          K0_max=maxabs(p0['K']),K10_max=maxabs(p['K']),
          u0_error=maxabs(np.array(p0['accepted_action'])-tr['q_target'][ix]),
          pred_u20_vs_actual=maxabs(np.array(p['accepted_action'])-tr['q_target'][it]),
          native_cost=x['accepted_cost'],trunk_violation=x.get('accepted_trunk_violation_m',x.get('accepted_trunk_violation')),
          native_reference_cost=x.get('feedback_reference_cost'),
          headroom_20ms=float(np.min(tr['qpos'][ix:it+1,2])),
          soft_xy_all_outside=bool(np.all((tr['feet'][ix:it+1,:,0]<.8158113883008419)|(tr['feet'][ix:it+1,:,0]>2.1841886116991582)|(abs(tr['feet'][ix:it+1,:,1])>.5841886116991581))))
        prediction.append(rec)
    phys=np.load(HERE/(stream+'-physical-000.npz'))['physical']
    force_range=np.tile(np.array([23.7,23.7,45.43]),4)
    raw=phys[:,49:61];excess=np.maximum(abs(raw)-force_range,0)
    first=np.flatnonzero(excess.max(1)>1e-4)
    result=dict(feedback_rows=len(feedback),prediction_rows=len(prediction),
        first_held_saturation_s=float(phys[first[0],0]) if len(first) else None,
        held_saturation_fraction=float(np.mean(excess.max(1)>1e-4)),
        held_saturation_peak_Nm=float(excess.max()),
        physical_clamp_error=maxabs(phys[:,14:26]-np.clip(raw,-force_range,force_range)),
        actual_query_vs_saved_target_max=max(x['u_matches_trajectory'] for x in feedback),
        qp_failed=sum(not x['qp_converged'] for x in feedback),restoration_count=sum(x['restoration'] for x in feedback),
        max_primal=max(x['primal'] for x in feedback),max_kkt=max(x['kkt'] for x in feedback),
        max_serialized_torque=max(x['serialized_torque'] for x in feedback),
        bins=[])
    for lo,hi in [(0,2),(2,2.4),(2.4,2.6),(2.6,2.8),(2.8,3.0),(3,4),(4,5),(5,8),(8,10),(10,14)]:
        f=[x for x in feedback if lo<=x['t']<hi];p=[x for x in prediction if lo<=x['origin']<hi]
        bin=dict(window=[lo,hi],n=len(f))
        for key in ['raw_feedback','effective_feedback','projection','action_delta_from_previous']:
            if f:bin[key+'_p50_p95_max']=np.quantile([x[key] for x in f],[.5,.95,1]).tolist()
        for key in ['base_position_error','joint_position_error','velocity_error','K0_max','u0_error']:
            if p:bin[key+'_p50_p95_max']=np.quantile([x[key] for x in p],[.5,.95,1]).tolist()
        result['bins'].append(bin)
    result['first_raw_feedback_gt_point5']=next((x for x in feedback if x['raw_feedback']>.5),None)
    result['first_effective_feedback_gt_point5']=next((x for x in feedback if x['effective_feedback']>.5),None)
    result['first_prediction_v_error_gt_point1']=next((x for x in prediction if x['origin']>=2 and x['velocity_error']>.1),None)
    result['first_prediction_joint_error_gt_point001']=next((x for x in prediction if x['origin']>=2 and x['joint_position_error']>.001),None)
    return result


"""Existing full-plan and physical-prefix alignment, without dynamics rerun."""
import gzip,json
from pathlib import Path
import numpy as np
from xml.etree import ElementTree as ET

def rows(path):return [json.loads(x) for x in gzip.open(path,'rt') if x.strip()]
def m(x):return float(np.max(np.abs(x)))
def prefix(raw):
    out=[];i=0
    while i<len(raw):
        h=raw[i:i+5];i+=5;nc=int(h[4]);d={'node':int(h[0]),'sub':int(h[1]),'before':h[2],'after':h[3],'ncontact':nc}
        for name,n in [('pre',37),('post',37),('raw_pd',12),('actuator',12),('passive',18),('residual',101)]:
            d[name]=raw[i:i+n];i+=n
        d['contact']=raw[i:i+8*nc].reshape(nc,8);i+=8*nc;out.append(d)
    assert i==len(raw)
    return out

def analyze_saved_plan(s,n):
    z=np.load(P/f'{s}-accepted-reference-gain-H-plan{n:06d}.npz')
    t0=z['accepted_times'][0]
    tr=np.load(R/(s+'-trajectory.npz'));t=tr['time'];ti=np.argmin(abs(t-t0))
    phy=np.load(P/(s+'-physical-000.npz'))['physical']
    pre=prefix(z['accepted_prefix_substeps'])
    feedback=rows(P/(s+'-feedback-actions.jsonl.gz'));execution=rows(P/(s+'-action-execution.jsonl.gz'))
    mapping={x['query_id']:x for x in execution if x['query_kind']=='applied_hold'}
    f={round(mapping[x['query_id']]['time_s'],5):x for x in feedback if x['query_id'] in mapping}
    limits=np.tile([23.7,23.7,45.43],4)
    result={'stream':s,'plan_id':n,'origin':t0,'header':z['header'].tolist(),
      'accepted_reference_state_max':m(z['accepted_states']-z['reference_states']),
      'accepted_reference_action_max':m(z['accepted_actions']-z['reference_actions']),
      'accepted_reference_time_max':m(z['accepted_times']-z['reference_times']),
      'Kmax':m(z['K']),'first_hold':[], 'control_nodes':[]}
    for x in pre[:10]:
        pi=int(np.argmin(abs(phy[:,0]-x['before'])));a=phy[pi]
        result['first_hold'].append(dict(sub=x['sub'],t=x['before'],native_contacts=x['contact'][:,:3].tolist(),
         predicted_pd=x['raw_pd'].tolist(),actual_pd=a[49:61].tolist(),pd_max_difference=m(x['raw_pd']-a[49:61]),
         predicted_actuator=x['actuator'].tolist(),actual_actuator=a[14:26].tolist(),actuator_max_difference=m(x['actuator']-a[14:26]),
         native_clamp_error=m(x['actuator']-np.clip(x['raw_pd'],-limits,limits)),
         physical_clamp_error=m(a[14:26]-np.clip(a[49:61],-limits,limits)),
         physical_force_z=a[32:42:3].tolist()))
    for j in range(11):
        tm=z['accepted_times'][j];ix=int(np.argmin(abs(t-tm)));ff=f.get(round(float(tm),5));
        if ff is None:continue
        ref=np.array(ff['nominal_unclipped']);states=np.r_[tr['qpos'][ix],tr['qvel'][ix]]
        nominal_pd=60*(ref-tr['qpos'][ix,7:19])-5*tr['qvel'][ix,6:18]
        du=z['accepted_actions'][j]-tr['q_target'][ix]
        ib=ff['interpolation_nodes'];alpha=ff['interpolation_fraction']
        interpolated_K=(1-alpha)*z['K'][ib[0]]+alpha*z['K'][ib[1]] if ff['plan_id']==n else None
        Kdelta_error=m(interpolated_K@np.asarray(ff['state_delta'])-ff['raw_K_delta_x']) if interpolated_K is not None else None
        result['control_nodes'].append(dict(j=j,time=float(tm),actual_plan_id=ff['plan_id'],scale=ff['feedback_scaling'],
          position_error=m(z['accepted_states'][j,:19]-states[:19]),velocity_error=m(z['accepted_states'][j,19:]-states[19:]),
          predicted_actual_target_max=m(du),raw_feedback_max=m(ff['raw_K_delta_x']),effective_feedback_max=abs(ff['feedback_scaling'])*m(ff['raw_K_delta_x']),projection_max=m(ff['projection_delta']),
          nominal_same_actual_state_PD=nominal_pd.tolist(),nominal_same_actual_state_PD_excess=m(np.maximum(abs(nominal_pd)-limits,0)),
          actual_contact=tr['contact'][ix].astype(int).tolist(),actual_grf_z=tr['actual_grf'][ix,:,2].tolist(),
          logged_Kdelta_reconstruction_max=Kdelta_error,
          Kdelta=ff['raw_K_delta_x'],delta_x=ff['state_delta'],nominal=ff['nominal_unclipped'],actual=ff['u_held']))
    derivative=z['r31_native_soft_substeps'].reshape(175,10,124)
    result['native_derivative_nominal_successor_difference']=m(derivative[:,:,50:87]-derivative[:,:,87:124])
    predheld=np.array([x['raw_pd'] for x in pre]);result['native_hold_PD_max_excess']=float(np.maximum(abs(predheld)-limits,0).max())
    result['all_native_prefix_clamp_max_error']=max(m(x['actuator']-np.clip(x['raw_pd'],-limits,limits)) for x in pre)
    holds={}
    for x in pre:
        target=(x['raw_pd']+60*x['pre'][7:19]+5*x['pre'][25:37])/60
        holds.setdefault(x['node'],[]).append(target)
    result['native_reconstructed_target_within_hold_max_change']=max(m(np.asarray(us)-us[0]) for us in holds.values())
    return result


"""Recompute W44 numerical-tie/complete-policy evidence from saved originals.

This is array/log analysis only: no MuJoCo stepping, planner execution or training.
The reported numerical bins are descriptive, not an acceptance tolerance.
"""
import csv
import gzip
import json
from pathlib import Path

import numpy as np




def rows(p):
    return [json.loads(x) for x in gzip.open(p, 'rt') if x.strip()]


def ma(x):
    return float(np.max(np.abs(x)))


def quantiles(values):
    return np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist() if values else []


def selected_candidates(stream, call_id, plan_id):
    z = np.load(P / f'{stream}-r33-candidates-call{call_id:06d}.npz')
    h = {int(k[9:12]): z[k] for k in z.files if k.endswith('_header')}
    nominal = [i for i, a in h.items() if a[2] == 0 and a[13] == 0 and a[14] == 1]
    best = min(nominal, key=lambda i: (h[i][12], h[i][11]))
    alpha = [i for i, a in h.items() if a[2] == 1 and a[13] == 0 and a[14] == 1]
    winner = min(alpha, key=lambda i: (h[i][12], h[i][11]))
    positive = [i for i in alpha if h[i][4] > 0]
    def a(i, name): return z[f'candidate{i:03d}_{name}']
    inst = np.load(P / f'{stream}-accepted-reference-gain-H-plan{plan_id:06d}.npz')
    names = ['accepted_states', 'accepted_actions', 'accepted_times', 'accepted_residual',
             'reference_states', 'reference_actions', 'reference_times', 'reference_residual',
             'K', 'H', 'base_Qu', 'base_Qxu', 'base_active']
    return {
        'stream': stream, 'call_id': call_id, 'plan_id': plan_id,
        'best_nominal_candidate': best, 'selected_candidate': winner,
        'best_nominal_header': h[best].tolist(), 'selected_header': h[winner].tolist(),
        'native_decrease': float(h[best][11] - h[winner][11]),
        'best_positive_alpha_cost': float(min(h[i][11] for i in positive)),
        'best_positive_alpha_candidate': min(positive, key=lambda i: h[i][11]),
        'selected_reference_equals_baseline_accepted': {
            k: ma(a(winner, 'reference_' + k) - a(best, 'accepted_' + k))
            for k in ['states', 'actions', 'times', 'residual']},
        'baseline_vs_selected_maxabs': {k: ma(a(winner, k)-a(best, k)) for k in names},
        'selected_vs_installed_maxabs': {k: ma(a(winner, k)-inst[k]) for k in names},
        'all_candidate_headers': [dict(index=i, header=v.tolist()) for i, v in h.items()],
    }


def analyze_zero_alpha(stream):
    calls = rows(R / f'{stream}-planner-calls.jsonl.gz')
    plans = rows(P / f'{stream}-planner-records.jsonl.gz')
    execution = rows(P / f'{stream}-action-execution.jsonl.gz')
    feedback = rows(P / f'{stream}-feedback-actions.jsonl.gz')
    links = {e['query_id']: e for e in execution if e['query_kind'] == 'applied_hold'}
    fb = []
    for f in feedback:
        if f['query_id'] not in links: continue
        fb.append({**f, 'physical_time': links[f['query_id']]['time_s']})
    by_plan = {}
    for f in fb: by_plan.setdefault(f['plan_id'], []).append(f)
    by_call = {p.get('source_call_id'): p for p in plans}
    alpha_calls = [c for c in calls if c['candidate_source'].startswith('alpha_') and c['policy_updated']]
    zero = [c for c in alpha_calls if c['chosen_alpha'] == 0]
    zero_plan_ids = {c['installed_plan_id'] for c in zero}
    events = []
    for c in zero:
        p = by_call[c['call_id']]
        assert p['plan_id'] == c['installed_plan_id']
        assert p['action_step_alpha'] == 0
        fs = by_plan.get(p['plan_id'], [])
        dc = c['chosen_actual']
        j = p['nominal_cost']
        actual = lambda f: abs(f['feedback_scaling']) * ma(f['raw_K_delta_x'])
        positive = [x for a in c['search_attempts'] for x in a.get('native_candidates', [])
                    if not x['failed'] and x['alpha'] > 0]
        row = dict(stream=stream, call_id=c['call_id'], plan_id=p['plan_id'], origin=c['origin'],
                   source=c['candidate_source'], nominal_cost=j, accepted_cost=p['accepted_cost'],
                   nominal_v=p['nominal_trunk_violation_m'], accepted_v=p['accepted_trunk_violation_m'],
                   native_decrease=dc, decrease_relative=dc/abs(j) if j else None,
                   decrease_ulps_at_nominal=dc/np.spacing(j) if j else None,
                   nominal_feedback_scale=p['nominal_feedback_scale'],
                   installed_feedback_scale=p['installed_feedback_scaling'],
                   feedback_scale_change=p['installed_feedback_scaling']-p['nominal_feedback_scale'],
                   regularization_used=p['regularization_used'],
                   best_positive_alpha_cost=min((x['cost'] for x in positive), default=None),
                   actual_applied_holds=len(fs),
                   actual_max_effective_feedback=max((actual(f) for f in fs),default=0),
                   actual_max_projection=max((ma(f['projection_delta']) for f in fs),default=0),
                   actual_max_reference_position_delta=max((ma(f['state_delta'][:18]) for f in fs),default=0),
                   actual_max_reference_velocity_delta=max((ma(f['state_delta'][18:]) for f in fs),default=0),
                   first_effective_feedback_gt_point5=next((f['physical_time'] for f in fs if actual(f)>.5),None))
        events.append(row)
    large = [f for f in fb if abs(f['feedback_scaling'])*ma(f['raw_K_delta_x'])>.5]
    bins = [1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-6,1e-3]
    return dict(stream=stream, planner_calls=len(calls), new_alpha_policy_calls=len(alpha_calls),
                chosen_alpha_zero_all_sources=sum(c['chosen_alpha']==0 for c in calls),
                zero_alpha_new_installs=len(zero),
                zero_alpha_positive_cost_decrease=sum(e['native_decrease']>0 for e in events),
                zero_alpha_equal_violation=sum(e['nominal_v']==e['accepted_v'] for e in events),
                zero_alpha_feedback_scale_changed=sum(e['feedback_scale_change']!=0 for e in events),
                zero_alpha_from_exact_zero_scale=sum(e['nominal_feedback_scale']==0 for e in events),
                zero_alpha_from_less_than_one_scale=sum(e['nominal_feedback_scale']<1 for e in events),
                zero_alpha_scale_histogram={str(a):sum(e['nominal_feedback_scale']==a for e in events)
                                           for a in sorted({e['nominal_feedback_scale'] for e in events})},
                zero_alpha_delta_quantiles=quantiles([e['native_decrease'] for e in events]),
                zero_alpha_relative_delta_quantiles=quantiles([e['decrease_relative'] for e in events]),
                zero_alpha_ulp_delta_quantiles=quantiles([e['decrease_ulps_at_nominal'] for e in events]),
                descriptive_positive_absolute_delta_le={str(b):sum(0<e['native_decrease']<=b for e in events) for b in bins},
                actual_applied_holds=len(fb), actual_holds_effective_feedback_gt_point5=len(large),
                of_these_zero_alpha_installed_policy=sum(f['plan_id'] in zero_plan_ids for f in large),
                first_actual_effective_feedback_gt_point5={k:large[0][k] for k in
                    ['physical_time','time','plan_id','feedback_scaling','nominal_feedback_scale','action_step_alpha',
                     'state_delta','raw_K_delta_x','projection_delta','u_held','nominal_unclipped']},
                events=events)




def verify_originals(manifest_path):
    data=json.loads(manifest_path.read_text())
    checks=[]
    for x in data['originals']:
        folder=R if x['group']=='previously_verified_trajectory_or_calls' else P
        path=folder/x['local_name']
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        size=path.stat().st_size
        assert digest==x['sha256'], f"SHA256 mismatch: {path}"
        assert size==x['downloaded_bytes'], f"Byte count mismatch: {path}"
        checks.append(dict(local_name=x['local_name'],sha256=digest,bytes=size))
    return checks


def contact_runs(times, contact, min_duration=0.):
    mask=~np.any(np.asarray(contact)>0,axis=1)
    edges=np.diff(np.r_[False,mask,False].astype(int))
    dt=float(np.median(np.diff(times)))
    result=[]
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        end=float(times[b]) if b<len(times) else float(times[-1]+dt)
        if end-times[a]+1e-10>=min_duration:
            result.append(dict(start=float(times[a]),end=end,duration=end-float(times[a]),
                               first_index=int(a),end_index=int(b)))
    return result


def early_contact_context():
    answer={}
    for stream in ['A','B']:
        tr=np.load(R/(stream+'-trajectory.npz'))
        phys=np.load(P/(stream+'-physical-000.npz'))['physical']
        calls=rows(R/(stream+'-planner-calls.jsonl.gz'))
        run=next(x for x in contact_runs(phys[:,0],phys[:,26:30],.020)
                 if 2.<x['start']<3.2)
        endpoint=next(x for x in contact_runs(tr['time'],tr['contact']) if x['start']>2.)
        ix=int(np.searchsorted(tr['time'],run['start']))
        sample_time=float(tr['time'][ix])
        call=max((c for c in calls if c['origin']<=sample_time+1e-10),key=lambda c:c['origin'])
        g=call['gait_context'];params=call['task_parameters']
        phase=g[8]+(sample_time-g[9])*g[10]
        angle=(np.fmod(phase+np.pi-2*np.pi*np.array([0,.5,.5,0]),2*np.pi)-np.pi)*.5/(1-params[4])
        answer[stream]=dict(first_physical_no_foot_contact_run_at_least_20ms=run,
            first_500Hz_no_contact_run=endpoint,actual_call_id=call['call_id'],
            actual_plan_id=call['installed_plan_id'],sample_time=sample_time,
            actual_task_parameters_cadence_Hz=params[2],actual_task_parameters_amplitude_m=params[3],
            actual_task_parameters_duty=params[4],private_phase_start=g[8],
            private_phase_start_time=g[9],private_phase_velocity=g[10],
            phase_at_sample_mod_2pi=float(phase%(2*np.pi)),
            nominal_phase_stance_FL_FR_RL_RR=(np.abs(angle)>=np.pi/2).tolist(),
            nominal_phase_stance_is_not_a_hard_contact_constraint=True,
            actual_base_z=float(tr['qpos'][ix,2]),actual_base_vz=float(tr['qvel'][ix,2]),
            physical_grf_max_in_long_run=ma(phys[run['first_index']:run['end_index'],30:42]))
    z=np.load(P/'B-r33-candidates-call000158.npz')
    h={int(k[9:12]):z[k] for k in z.files if k.endswith('_header')}
    nominal=min((i for i,a in h.items() if a[2]==0 and a[13]==0 and a[14]==1),
                key=lambda i:(h[i][12],h[i][11]))
    chosen=19
    native=[]
    tr=np.load(R/'B-trajectory.npz')
    for i in [nominal,chosen]:
        feet=z[f'candidate{i:03d}_foot_rows'].reshape(-1,29)
        s=z[f'candidate{i:03d}_accepted_states'];ts=z[f'candidate{i:03d}_accepted_times']
        first=contact_runs(feet[:,0],feet[:,[7,14,21,28]])[0]
        j=first['first_index'];ix=int(np.argmin(abs(tr['time']-ts[j])))
        native.append(dict(candidate=i,header=h[i].tolist(),first_native_foot_contact_absence=first,
            native_grf_max_in_run=ma(feet[j:first['end_index']][:,[5,12,19,26]]),
            sample_time=float(ts[j]),predicted_base_z=float(s[j,2]),actual_base_z=float(tr['qpos'][ix,2]),
            predicted_base_vz=float(s[j,21]),actual_base_vz=float(tr['qvel'][ix,2]),
            note='Foot rows after mj_step retain pre-integration geometry/force; physical output may replan after 20 ms.'))
    answer['B_first_flight_native_call158']=dict(candidates=native,
        baseline_selected_state_max=ma(z[f'candidate{nominal:03d}_accepted_states']-z['candidate019_accepted_states']),
        native_decrease=float(h[nominal][11]-h[chosen][11]),
        conclusion='The best nominal as well as chosen alpha0 already predicts the early flight; not solely later off-reference amplification.')
    answer['A_prediction_coverage']='Call154 candidate archive is not present in the exact attempt manifest; candidate archive begins at call159 after this first long flight. No exact first-flight prediction-contact comparison claimed.'
    return answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals-dir',type=Path,required=True)
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('control-originals.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    global P,R,HERE,RAW
    P=HERE=args.originals_dir
    R=RAW=args.raw_dir
    checked=verify_originals(args.manifest)
    result=dict(schema='w44-control-array-recomputation-v1',
        audit_basis_commit='69d5a6ff2aa9803fddc8ac3a78a906008ea526c6',
        scope='Final W44 E30 attempts only. Saved-array/log arithmetic, no dynamics rerun.',
        numeric_bins_are_descriptive_not_an_acceptance_tolerance=True,
        input_checks=checked,
        alignment={s:align_stream(s) for s in ['A','B']},
        zero_alpha={s:analyze_zero_alpha(s) for s in ['A','B']},
        early_contact=early_contact_context(),
        candidate_comparisons=[selected_candidates('A',163,156),selected_candidates('B',173,161)],
        saved_plans=[analyze_saved_plan(s,n) for s,n in
                     [('A',151),('A',156),('A',234),('B',140),('B',143),('B',144),('B',161)]])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(f"Verified {len(checked)} originals; wrote {args.output}")


if __name__=='__main__':
    main()
