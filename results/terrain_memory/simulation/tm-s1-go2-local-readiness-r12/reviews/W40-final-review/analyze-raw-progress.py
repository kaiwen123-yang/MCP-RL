#!/usr/bin/env python3
"""Inspect selected published W40 raw data; never simulates a new trajectory.

Usage: python analyze-raw-progress.py --raw-root /path/to/extracted/w40-final-raw
The directory uses ZIP member paths w40-a/epNNNNN and w40-b/evaluation-E30.
Missing requested originals are recorded explicitly. Re-running after a download
finishes recomputes the report, without modifying any research source data.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import math
import re
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[5]
BASE=REPO/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
READ={}
MISSING=[]
TIMER_FIELDS=['nominal_rollout','model_derivatives','cost_derivatives','backward_pass','candidate_rollouts','policy_update']


def summary(values):
    a=np.asarray(values,dtype=float)
    if not a.size:return {'n':0}
    return {'n':int(a.size),'min':float(a.min()),'median':float(np.median(a)),
            'p95':float(np.percentile(a,95)),'max':float(a.max()),'mean':float(a.mean())}


def recorded(path):
    if not path.exists():MISSING.append(str(path));return False
    READ[str(path)]={'bytes':path.stat().st_size}
    return True


def readjson(path):
    if not recorded(path):return None
    return json.loads(path.read_text())


def readjsonl(path):
    if not recorded(path):return []
    with path.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def npz(path):
    if not recorded(path):return None
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def result_for(branch,label):
    attempt=(f'job-20261106T103028Z-w40-{branch}-eval-PANN_PPO_S5-s{1 if branch=="a" else 2}-E30'
        if label=='evaluation-E30' else f'train-job-20261106T103028Z-w40-{branch}-PANN_PPO_S5-s{1 if branch=="a" else 2}-{label}')
    return attempt,json.loads((BASE/(attempt+'.json')).read_text())


def feedback_audit(directory,branch,label):
    attempt,result=result_for(branch,label)
    resolved=json.loads((BASE/(attempt+'-resolved.json')).read_text())
    ranges=np.asarray(resolved['actuator_ctrlrange'])
    limits=np.asarray(resolved['actuator_forcerange'])
    startup=result['startup']['iterations']
    f=directory/'feedback-actions.jsonl'
    if not f.exists():recorded(f);return None
    counts=Counter(); statuses=Counter(); refinements=Counter(); first={}; peaks={}
    rawpeaks=[]; projection=[]; cert=[]; regularization=[]; refine_examples=[]
    actual_ids=[]; actual_targets=[]; applied_times=[]
    def event(name,flag,row,value=None):
        if not flag:return
        counts[name]+=1
        first.setdefault(name,{'time_s':row['time'],'query_id':row['query_id'],'plan_id':row['plan_id']})
        if value is not None and (name not in peaks or value>peaks[name]['value']):
            peaks[name]={'time_s':row['time'],'query_id':row['query_id'],'plan_id':row['plan_id'],'value':float(value)}
    for i,row in enumerate(readjsonl(f)):
        counts['all_queries']+=1
        if i<startup:
            counts['startup_diagnostic_queries']+=1
            continue
        counts['applied_query_attempts']+=1
        if not (row['metric_valid'] and row['qp_converged']):
            event('rejected_before_physics',True,row)
            statuses[row['projection_status']]+=1
            if row.get('qp_refinement_attempted'):refinements[row.get('qp_refinement_status','unknown')]+=1
            continue
        counts['applied_certified_queries']+=1
        statuses[row['projection_status']]+=1
        u=np.asarray(row['u_star']);held=np.asarray(row['u_held']);c=np.asarray(row['center_before_projection'])
        nom=np.clip(row['nominal_unclipped'],ranges[:,0],ranges[:,1]);pd=np.asarray(row['PD_u_star'])
        pdnom=pd+60*(nom-u);pdcenter=pd+60*(c-u)
        satnom=(pdnom<limits[:,0]-1e-8)|(pdnom>limits[:,1]+1e-8)
        satcenter=(pdcenter<limits[:,0]-1e-8)|(pdcenter>limits[:,1]+1e-8)
        raw=float(np.max(np.abs(row['raw_K_delta_x'])));delta=float(np.max(np.abs(row['projection_delta'])))
        rawpeaks.append(raw);projection.append(delta);cert.append(row['kkt_scaled_residual']);regularization.append(row['metric_regularization'])
        event('raw_feedback_gt_half_rad',raw>.5,row,raw)
        event('nominal_PD_saturation',satnom.any(),row,float(np.max(np.maximum(limits[:,0]-pdnom,pdnom-limits[:,1]))))
        event('feedback_added_PD_saturation',(satcenter & ~satnom).any(),row)
        event('target_clip',np.any(np.asarray(row['action_before_clip'])!=c),row)
        event('projection_changed',row['projection_changed'],row,delta)
        event('actuator_restoration',row['restoration'],row)
        event('same_state_star_PD_over_1e_4',np.max(row['PD_violation_Nm'])>1e-4,row,float(np.max(row['PD_violation_Nm'])))
        pdheld=np.asarray(row['PD_u_held_same_state']);violation=np.maximum(limits[:,0]-pdheld,pdheld-limits[:,1])
        event('same_state_held_PD_over_1e_4',np.max(violation)>1e-4,row,float(np.max(violation)))
        if row.get('qp_refinement_attempted'):
            event('native_certificate_failed_refined',True,row)
            refinements[row.get('qp_refinement_status','unknown')]+=1
            if len(refine_examples)<6:
                refine_examples.append({k:row.get(k) for k in ['time','query_id','plan_id','interpolation_nodes','interpolation_fraction',
                  'qp_refinement_iterations','qp_refinement_status','qp_native_kkt_scaled_residual','kkt_scaled_residual']})
        if label in ('ep00314','evaluation-E30') and (branch=='a' or label=='evaluation-E30'):
            actual_ids.append(row['query_id']);actual_targets.append(held);applied_times.append(row['time'])
    calls=list(readjsonl(directory/'planner-calls.jsonl'))
    callcounts=Counter(x['return_reason'] for x in calls)
    timings=Counter()
    for call in calls:
        timers=call.get('timers_us',[])
        timings.update(dict(zip(TIMER_FIELDS,timers)) if isinstance(timers,list) else timers)
    call_summary={'count':len(calls),'return_reasons':dict(callcounts),
                  'policy_not_updated':sum(not x['policy_updated'] for x in calls),
                  'reuse_age_s':summary([x['reused_policy_age_s'] for x in calls]),
                  'wall_s':summary([x['wall_s'] for x in calls]),'timers_us_sum':dict(timings)} if calls else None
    out={'attempt':attempt,'stream':branch,'label':label,'count':dict(counts),'projection_status':dict(statuses),
         'refinement_status':dict(refinements),'first':first,'peaks':peaks,'raw_feedback_max_axis_rad':summary(rawpeaks),
         'projection_max_axis_rad':summary(projection),'kkt_scaled_residual':summary(cert),
         'metric_regularization':summary(regularization),'refined_success_examples':refine_examples,
         'first_material_load_s':None,'planner_calls':call_summary,
         'query_count_matches_runtime':counts['applied_query_attempts']==result['timing']['feedback']['count']}
    evpath=BASE/'jobs'/f'job-20261106T103028Z-w40-{branch}'/'numeric-evidence/episodes'/(attempt+'-events.json')
    ev=json.loads(evpath.read_text())
    out['first_material_load_s']=ev.get('first_material_events_by_type',{}).get('load',{}).get('event',{}).get('time_s')
    if out['first_material_load_s'] is not None:
        out['large_feedback_before_load']=first.get('raw_feedback_gt_half_rad',{}).get('time_s',math.inf)<out['first_material_load_s']
        out['nominal_or_feedback_PD_saturation_before_load']=min(first.get(k,{}).get('time_s',math.inf) for k in ['nominal_PD_saturation','feedback_added_PD_saturation'])<out['first_material_load_s']
    return out,{'query_ids':actual_ids,'targets':actual_targets,'times':applied_times}


def special_physical(directory,branch,label,feedback):
    attempt,result=result_for(branch,label)
    trajectory=npz(directory/'trajectory.npz');physical=npz(directory/'physical-000.npz')
    events=readjson(directory/'r29-event-manifest.json')
    if trajectory is None or physical is None:return None
    t=trajectory['time'];q=trajectory['qpos'];vel=trajectory['qvel'];cmd=trajectory['command_velocity']
    columns=result['physical_columns'];p=physical['physical'];pc={k:i for i,k in enumerate(columns)}
    raw=p[:,[pc[f'unclipped_pd_request_{i}'] for i in range(12)]]
    effort=p[:,[pc[f'torque_{i}'] for i in range(12)]]
    limits=np.array([23.7,23.7,45.43]*4);excess=np.maximum(abs(raw)-limits,0)
    # Physical rows are exactly ten 0.2 ms substeps per successful held action.
    assert len(p)%10==0
    first=np.arange(len(p))%10==0
    first_mx=excess.max(axis=1)
    periods=[]
    for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,float(t[-1]))]:
        ids=np.where((t>=lo)&(t<=hi))[0]
        if len(ids)<2:continue
        periods.append({'start_s':lo,'end_s':hi,'dx_m':float(q[ids[-1],0]-q[ids[0],0]),
          'z_min_m':float(q[ids,2].min()),'z_median_m':float(np.median(q[ids,2])),
          'command_x_m_s':summary(cmd[ids,0]),'actual_vx_m_s':summary(vel[ids,0]),
          'nonfoot_force_over_1N_fraction':float(np.mean(np.linalg.norm(trajectory['nonfoot_ground_force'][ids],axis=1)>1)),
          'contact_count':summary(trajectory['contact'][ids].sum(axis=1))})
    eventsout=events['events'] if events else []
    eventposes=[]
    for e in eventsout:
        i=int(np.argmin(abs(t-e['time_s'])))
        eventposes.append({'kind':e['kind'],'time_s':e['time_s'],'trajectory_time_s':float(t[i]),
            'xyz_m':q[i,:3].tolist(),'rpy_rad':trajectory['rpy'][i].tolist(),
            'base_velocity':vel[i,:6].tolist(),'command':cmd[i].tolist(),
            'nonfoot_force_N':trajectory['nonfoot_ground_force'][i].tolist(),
            'actual_foot_grf_z_N':trajectory['actual_grf'][i,:,2].tolist()})
    identities={}
    applied=list(readjsonl(directory/'action-execution.jsonl'))
    applied=[x for x in applied if x.get('applied_action_id') is not None and x.get('query_kind')=='applied_hold']
    if applied and feedback['targets']:
        ids=[x['query_id'] for x in applied]
        targets=np.asarray([x['held_target'] for x in applied]);ft=np.asarray(feedback['targets'])
        nt=min(len(targets),len(trajectory['q_target']))
        identities={'linked_applied_actions':len(applied),'feedback_certified_actions':len(ft),
          'query_ids_identical':ids==feedback['query_ids'],
          'feedback_to_link_target_max_abs_rad':float(abs(ft-targets).max()) if ft.shape==targets.shape else None,
          'links_to_trajectory_q_target_max_abs_rad':float(abs(targets[:nt]-trajectory['q_target'][:nt]).max()),
          'trajectory_rows':len(t),'extra_terminal_rows':len(t)-len(applied)}
    final_window=np.where(t>=t[-1]-10)[0]
    return {'attempt':attempt,'physical_samples':len(p),'trajectory_samples':len(t),
         'events':eventposes,'periods':periods,'action_identity':identities,
         'first_z_below_0_25_s':next((float(tt) for tt,zz in zip(t,q[:,2]) if zz<.25),None),
         'first_z_below_0_20_s':next((float(tt) for tt,zz in zip(t,q[:,2]) if zz<.20),None),
         'last_10s_dx_m':float(q[-1,0]-q[final_window[0],0]),
         'last_10s_command_x_m_s':summary(cmd[final_window,0]),
         'first_hold_substep_PD_max_excess_Nm':float(excess[first].max()),
         'later_nine_hold_substeps_PD_max_excess_Nm':float(excess[~first].max()),
         'first_hold_substep_rows_excess_over_1e_4':int(np.sum(first_mx[first]>1e-4)),
         'later_nine_hold_substeps_rows_excess_over_1e_4':int(np.sum(first_mx[~first]>1e-4)),
         'later_nine_hold_substep_rows':int(np.sum(~first)),
         'physical_PD_clipped_to_effort_max_abs_Nm':float(abs(np.clip(raw,-limits,limits)-effort).max()),
         'hold_violation_first_time_s':float(p[np.where(first_mx>1e-4)[0][0],pc['force_evaluation_time_s']]) if np.any(first_mx>1e-4) else None,
         'hold_violation_peak_time_s':float(p[np.argmax(first_mx),pc['force_evaluation_time_s']]),
         'nonfoot_physical_rows_over_1N':int(np.sum(np.linalg.norm(p[:,[pc[f'nonfoot_grf_{i}'] for i in 'xyz']],axis=1)>1)),
         'physical_time_step_max_abs_error':float(np.max(abs(np.diff(p[:,0])-.0002))),
         'note':'A valid projection is for the held-action beginning state; the later nine plant steps use the unchanged target. Effort clipping is still physically active.'}


def rollout_audit(rawroot,branch):
    root=BASE/'jobs'/f'job-20261106T103028Z-w40-{branch}'/'numeric-evidence/training'
    ts={r['transition_serial']:r for p in sorted(root.glob('*-transitions.jsonl')) for r in readjsonl(p)}
    vs={r['transition_serial']:r for p in sorted(root.glob('*-pann-values.jsonl')) for r in readjsonl(p)}
    arrays=[];serial=[];files=[]
    expected=3 if branch=='a' else 4
    for path in sorted((rawroot/f'w40-{branch}'/'training').glob('rollout-*.npz')):
        data=npz(path);a,b=map(int,re.search(r'rollout-(\d+)-(\d+)\.npz$',path.name).groups())
        arrays.append(data);serial.extend(range(a,b+1));files.append(str(path))
    if not arrays:return {'status':'missing','expected_rollout_files':expected}
    data={k:np.concatenate([d[k] for d in arrays]) for k in arrays[0]}
    obs=data['obs'].reshape(-1,658);nxt=data['next_obs'].reshape(-1,658)
    logs=[ts[s] for s in serial];values=[vs[s] for s in serial]
    chain=np.array([i for i in range(len(logs)-1) if logs[i]['episode']==logs[i+1]['episode']],int)
    scalar={}
    for key in ['raw_action','reward','log_prob','value','next_value','terminated','truncated']:
        target=[v[key] if key in ('value','next_value') else t[key] for t,v in zip(logs,values)]
        scalar[key]=float(np.max(abs(data[key].ravel().astype(float)-np.asarray(target))))
    query=obs[:,94:102].reshape(-1,4,2).astype(float)*3
    mask=obs[:,102:106].astype(bool)
    # Ground.interior in src/tm02/s1_contact.py uses the complete .001 m2
    # equivalent sole, not just its center. Keep the frozen hard-priority edge.
    footprint_half=.5*math.sqrt(.001)
    inside=(query[:,:,0]>=.8+footprint_half)&(query[:,:,0]<=2.2-footprint_half)&(abs(query[:,:,1])<=.6-footprint_half)
    phase=np.argmax(obs[:,80:89],axis=1)
    return {'status':'audited' if len(files)==expected else 'partial','files':files,
      'expected_rollout_files':expected,'rows':len(obs),'serial_range':[serial[0],serial[-1]],
      'all_arrays_finite':{k:bool(np.isfinite(v).all()) for k,v in data.items()},
      'scalar_compact_max_abs_difference':scalar,
      'current_pann_compact_max_abs_m':float(abs(obs[:,-12:]*np.float32(.001)-np.asarray([v['pann_recovery_m'] for v in values])).max()),
      'next_pann_compact_max_abs_m':float(abs(nxt[:,-12:]*np.float32(.001)-np.asarray([v['next_pann_recovery_m'] for v in values])).max()),
      'same_episode_next_obs_to_next_obs_pairs':len(chain),
      'same_episode_next_obs_to_next_current_max_abs':float(abs(nxt[chain]-obs[chain+1]).max()) if len(chain) else None,
      'query_mask_geometry_mismatches':int(np.sum(mask!=inside)),
      'query_mask_geometry':'Ground.interior; .001 m2 whole-footprint hard-priority edge',
      'outside_query_nonzero_pann_feet':int(np.sum(np.any(obs[:,-12:].reshape(-1,4,3)!=0,axis=2)&~mask)),
      'phase_one_hot_max_error':float(abs(obs[:,80:89].sum(axis=1)-1).max()),
      'finite_task_channels_all_zero':bool(np.all(obs[:,91:93]==0)),
      'quaternion_norm_max_error':float(abs(np.linalg.norm(obs[:,3:7],axis=1)-1).max()),
      'current_history_abs':summary(abs(obs[:,106:646])),
      'current_pann_normalized_abs':summary(abs(obs[:,646:658])),
      'logical_plan_age_s':summary(obs[:,93]*.35),
      'tail_note':'A last 147 rows are only checked in compact saved-state exports; no full resume tensor downloaded.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-root',type=Path,required=True);args=ap.parse_args()
    rows=[];special={}
    for branch,low in [('a',285),('b',281)]:
        labels=[f'ep{ep:05d}' for ep in range(low,315)]
        if branch=='b':labels.append('evaluation-E30')
        for label in labels:
            directory=args.raw_root/f'w40-{branch}'/label
            found=feedback_audit(directory,branch,label)
            if found is None:continue
            row,links=found;rows.append(row)
            if (branch,label) in [('a','ep00314'),('b','evaluation-E30')]:
                special[f'{branch}/{label}']=special_physical(directory,branch,label,links)
    training=[x for x in rows if x['label']!='evaluation-E30']
    counts=Counter();statuses=Counter();refine=Counter();returns=Counter()
    for row in training:
        counts.update(row['count']);statuses.update(row['projection_status']);refine.update(row['refinement_status'])
        if row['planner_calls']:returns.update(row['planner_calls']['return_reasons'])
    report={'schema':'W40_final_selected_raw_audit_v1','review_baseline':'c438f5463659ab7c8d3a7f5eced251639e14b2ef',
      'scope':'All W40 training/E30 feedback and planner-call logs, selected A314/B E30 physical/execution records and seven completed PPO rollout arrays. No new physical experiment.',
      'training_episodes_read':len(training),'training_counts':dict(counts),'training_projection_status':dict(statuses),
      'training_refinement_status':dict(refine),'training_planner_return_reasons':dict(returns),
      'loaded_training_episodes_read':sum(x['first_material_load_s'] is not None for x in training),
      'loaded_with_large_feedback_before_load':sum(x.get('large_feedback_before_load',False) for x in training),
      'loaded_with_nominal_or_feedback_saturation_before_load':sum(x.get('nominal_or_feedback_PD_saturation_before_load',False) for x in training),
      'all_query_counts_match_runtime':all(x['query_count_matches_runtime'] for x in rows),
      'episodes':rows,'physical_cases':special,
      'rollouts':{branch:rollout_audit(args.raw_root,branch) for branch in ['a','b']},
      'limitations':['All event thresholds are descriptive diagnostics, not new termination criteria.',
         'Timing order supports hypotheses; it does not isolate the causal effect of projection, feedback or model differences.',
         'Raw K delta is a request before original target clipping and H projection, never an executed joint angle.',
         'The three technical rejected queries do not advance physics; successful prior actions remain valid samples.',
         'Observation preservation is not a network activation/gradient check or proof of learned causal memory benefit.']}
    report['coverage']={'read_files':len(READ),'read_bytes':sum(x['bytes'] for x in READ.values()),'files':READ,'missing':MISSING}
    path=HERE/'raw-progress-audit.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['training_episodes_read','training_counts','training_projection_status',
        'training_refinement_status','training_planner_return_reasons','loaded_training_episodes_read',
        'loaded_with_large_feedback_before_load','loaded_with_nominal_or_feedback_saturation_before_load',
        'all_query_counts_match_runtime']},ensure_ascii=False,indent=2))
    print('Read',len(READ),'files; missing',len(MISSING),'output',path)


if __name__=='__main__':main()
