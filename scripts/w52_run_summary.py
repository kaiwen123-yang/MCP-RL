"""Compact actual W52 attempt evidence, derived only from recorded same-attempt files."""
import json
from pathlib import Path
import numpy as np
from publish_s1_go2_w52 import ROOT,SMALL,LARGE,write
def summarize_attempt(attempt):
 out=LARGE/attempt;dest=SMALL/'w52'/'original-evidence'/(attempt+'.json')
 rp=SMALL/(attempt+'.json');result=json.loads(rp.read_text()) if rp.exists() else None
 calls=[json.loads(x) for x in (out/'planner-calls.jsonl').read_text().splitlines()] if (out/'planner-calls.jsonl').exists() else []
 boundaries=[json.loads(x) for x in (out/'w50-planner-boundaries.jsonl').read_text().splitlines()] if (out/'w50-planner-boundaries.jsonl').exists() else []
 events=json.loads((out/'w50-event-manifest.json').read_text()) if (out/'w50-event-manifest.json').exists() else {}
 positive=[x for x in calls if x['policy_updated'] and x['candidate_source'].startswith(('alpha_','k_guided_','h_guided_','homotopy_','w50_','B_homotopy_','S_homotopy_','B_bootstrap_','S_bootstrap_')) and x['chosen_alpha']>0]
 matched=[x for x in boundaries if x['previous_prediction']['status']=='available']
 first=next((x for x in matched if any(v!=0 for v in x['previous_prediction']['position_tangent']+x['previous_prediction']['velocity_difference'])),None)
 keys=['native_new_rollouts','candidate_records','derivative_passes','transition_fd_calls','risk_geometry_fd_calls','backward_attempts']
 summary=dict(attempt=attempt,status=result['stage_status'] if result else 'technical_interruption',
  original_result={k:result.get(k) for k in ['success','reason','error','simulation_s','wall_s','physical_samples','legal_route','stage_progress','qualification','max_x_m','min_body_height_m','completed_round_trips']} if result else None,
  planner_calls=len(calls),positive_installs=len(positive),positive_same_problem_cost_descent=sum(x['chosen_actual']>0 for x in positive),
  positive_risk_priority_cost_increase=sum(x['chosen_actual']<0 for x in positive),
  internal_stagnation=sum(x['stop_reason']=='native_search_stagnation' for x in calls),
  work_completed_calls={k:sum(x[k] for x in calls) for k in keys},
  work_scope='Only completed native call records; interrupted calls with no record have unknown unrecorded native rollout work, never inferred zero.',new_PPO_samples=0,new_PPO_updates=0,
  plant_physics_substeps=result.get('physical_samples') if result else None,
  prediction_pairs=len(matched),prediction_unavailable=len(boundaries)-len(matched),
  first_nonzero_same_time_prediction_error=first,
  diagnostic_events=events.get('events',[]),object_bytes=events.get('actual_bytes'),
  same_attempt_record_readback=json.loads((out/'w50-first-record-readback.json').read_text()) if (out/'w50-first-record-readback.json').exists() else None,
  causal_conclusion='Original task status above; nonzero prediction error and diagnostic thresholds do not by themselves identify the cause of failure.')
 if matched:
  summary['prediction_error_extrema']={key:float(max(np.max(abs(np.asarray(x['previous_prediction'][key]))) for x in matched)) for key in
    ['base_position_m','base_orientation_rad','joint_position_rad','base_linear_velocity_m_s','base_angular_velocity_rad_s','joint_velocity_rad_s']}
 summary['stop_reason_counts']={reason:sum(x['stop_reason']==reason for x in calls) for reason in sorted({x['stop_reason'] for x in calls})}
 summary['return_reason_counts']={reason:sum(x['return_reason']==reason for x in calls) for reason in sorted({x['return_reason'] for x in calls})}
 summary['incumbent_retained_calls']=sum(x['return_reason']=='incumbent_retained' for x in calls)
 summary['no_improving_candidate_calls']=sum(not x['policy_updated'] for x in calls)
 summary['backward_failed_calls']=sum(x['stop_reason']=='backward_failed' for x in calls)
 summary['support_scope']='Descriptive gait/support evidence, not altered original task qualification. Survival is not stable locomotion.'
 tr=out/'trajectory.npz'
 if tr.exists():
  with np.load(tr) as z:
   t=z['time'];q=z['qpos'];v=z['qvel'];fz=z['actual_grf'][:,:,2];nf=z['nonfoot_ground_force'][:,2]
   contacts=z['contact'].astype(bool);idx=np.flatnonzero(nf>1)
   summary['gait_and_support']=dict(physical_duration_s=float(t[-1]),
    xyz_initial_m=q[0,:3].tolist(),xyz_final_m=q[-1,:3].tolist(),max_x_m=float(q[:,0].max()),
    body_z_min_m=float(q[:,2].min()),body_z_final_m=float(q[-1,2]),
    vertical_velocity_range_m_s=[float(v[:,2].min()),float(v[:,2].max())],
    foot_order=['FL','FR','RL','RR'],per_foot_Fz_quantiles_N=np.quantile(fz,[0,.1,.5,.9,1],axis=0).tolist(),
    total_foot_Fz_quantiles_N=np.quantile(fz.sum(axis=1),[0,.1,.5,.9,1]).tolist(),
    nonfoot_Fz_quantiles_N=np.quantile(nf,[0,.1,.5,.9,1]).tolist(),
    first_nonfoot_over_1N_s=next((e['time_s'] for e in events.get('events',[]) if e['kind']=='first_nonfoot_load'),None),
    first_sampled_nonfoot_over_1N_s=float(t[idx[0]]) if len(idx) else None,
    force_sample_clock='trajectory 500Hz same-time mj_forward force; first_nonfoot_load event retains 5000Hz detection time',
    foot_load_over_1N_fraction=(fz>1).mean(axis=0).tolist(),
    nonfoot_load_over_1N_fraction=float((nf>1).mean()),
    diagnostic_load_threshold_N=1,
    contact_transitions_per_foot=np.count_nonzero(np.diff(contacts.astype(int),axis=0),axis=0).tolist(),
    contact_fraction_per_foot=contacts.mean(axis=0).tolist(),
    full_record_observed=result.get('stage_progress',{}).get('full_observation') if result else None,
    full_record_observed_source='original evaluator stage_progress.full_observation; early-ended result is not automatically full observation',
    stable_locomotion_claim=False,
    qualification_only_from_original_result=True)
 summary["windows"]=windows(out,boundaries)
 write(dest,summary);return dest

def windows(out,boundaries):
 if not (out/'trajectory.npz').exists():return []
 # Same physical states; phase reconstructed only from each call's recorded
 # private gait clock and original parameter values, not from force thresholds.
 result=[]
 events=json.loads((out/'w50-event-manifest.json').read_text())
 context=[]
 for row in events['event_calls']:
  p=out/row['objects']['context']['file']
  with np.load(p) as z:
   g=z['private_gait'];context.append((row['boundary']['origin_s'],g.copy()))
 context.sort(key=lambda x:x[0])
 with np.load(out/'trajectory.npz') as z:
  t=z['time'];fz=z['actual_grf'][:,:,2];nf=z['nonfoot_ground_force'][:,2]
  q=z['qpos'];v=z['qvel'];feet=z['feet'];fv=z['feet_velocity'];rpy=z['rpy']
  import mujoco
  model=mujoco.MjModel.from_binary_path(str(out/'model.mjb'))
  radii=np.array([model.geom_size[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name),0] for name in ('FL','FR','RL','RR')])
  bottom=feet[:,:,2]-radii[None,:]
  for lo,hi in [(0,2),(2.002,4.3),(4.3,8)]:
   ids=np.flatnonzero((t>=lo-1e-9)&(t<=hi+1e-9))
   if not len(ids):result.append(dict(requested_s=[lo,hi],status='not_reached'));continue
   k=ids;clock=None
   prior=[g for tt,g in context if tt<=t[k[0]]+1e-9]
   if prior:clock=prior[-1]
   swing=None
   if lo>2 and clock is not None:
    phase=clock[8]+(t[k]-clock[9])*clock[10]
    # plant order FL,FR,RL,RR vs native FL,RL,FR,RR (same Trot shifts).
    w=(phase[:,None]/(2*np.pi)-np.array([0,.5,.5,0])[None,:]+.5)%1-.5
    swing=np.abs(w)<.275
   loads=fz[k]>1;row=dict(requested_s=[lo,hi],observed_s=[float(t[k[0]]),float(t[k[-1]])],
    status='complete_window' if t[k[-1]]>=hi-.0021 else 'partial_physical_window',
    body_dx_m=float(q[k[-1],0]-q[k[0],0]),body_x_last_m=float(q[k[-1],0]),
    z_min_m=float(q[k,2].min()),z_median_m=float(np.median(q[k,2])),z_last_m=float(q[k[-1],2]),
    vz_range_m_s=[float(v[k,2].min()),float(v[k,2].max())],
    roll_pitch_max_abs_rad=np.max(abs(rpy[k,:2]),axis=0).tolist(),
    nonfoot_mean_Fz_N=float(nf[k].mean()),nonfoot_over_1N_fraction=float((nf[k]>1).mean()),
    four_feet_over_1N_fraction=float(loads.all(axis=1).mean()),feet_Fz_mean_N=fz[k].mean(axis=0).tolist(),
    feet_world_tangent_speed_mean_m_s=np.linalg.norm(fv[k,:,:2],axis=2).mean(axis=0).tolist(),
    feet_bottom_z_range_m=np.stack([bottom[k].min(axis=0),bottom[k].max(axis=0)]).tolist(),
    feet_z_semantics='recorded geom centers minus original compiled foot radii; relative to original z=0 plane, not unobserved deformed terrain',
    diagnostic_threshold_N=1,qualification_threshold_changed=False)
   pattern=loads.astype(np.int64)@np.array([1,2,4,8])
   row['actual_support_pattern_fractions']={str(mask):float((pattern==mask).mean()) for mask in np.unique(pattern)}
   row['support_pattern_bits']='FL=1, FR=2, RL=4, RR=8; load threshold is descriptive 1N'
   row['load_transitions_per_foot']=np.count_nonzero(np.diff(loads.astype(int),axis=0),axis=0).tolist()
   if swing is not None:
    row['loaded_stance_world_tangent_speed_mean_m_s']=[float(np.linalg.norm(fv[k,i,:2],axis=1)[(~swing[:,i])&loads[:,i]].mean()) if ((~swing[:,i])&loads[:,i]).any() else None for i in range(4)]
    row['target_swing_bottom_z_range_m']=[[float(bottom[k,i][swing[:,i]].min()),float(bottom[k,i][swing[:,i]].max())] if swing[:,i].any() else None for i in range(4)]
    row['target_swing_loaded_fraction']=[float(loads[:,i][swing[:,i]].mean()) if swing[:,i].any() else None for i in range(4)]
    row['target_stance_Fz_mean_N']=[float(fz[k,i][~swing[:,i]].mean()) if (~swing[:,i]).any() else None for i in range(4)]
   else:row['target_swing_loaded_fraction']=None
   costs=[b['current_solve']['cost_components'] for b in boundaries if lo-1e-9<=b['origin_s']<=hi+1e-9]
   row['installed_native_cost_mean']={name:float(np.mean([c[name] for c in costs])) for name in costs[0]} if costs else None
   result.append(row)
 return result
