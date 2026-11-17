#!/usr/bin/env python3
"""Analyze the already completed N task; saved arrays only, no model calls."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from s1_go2_w53_n_records import read_policy
ROOT=Path(__file__).resolve().parents[1]
AT='job-20261117T023608Z-w53-n-a-eval-PANN_PPO_S5-s1-E30-attempt02'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
OUT=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/AT
LEGS=['FL','FR','RL','RR']
def val(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
def first(t,flag):
 ids=np.flatnonzero(flag);return float(t[ids[0]]) if len(ids) else None
def intervals(t,flag,dt):
 d=np.diff(np.r_[False,flag,False].astype(int));a=np.flatnonzero(d==1);b=np.flatnonzero(d==-1)
 return [dict(start_s=float(t[i]),end_s=float(t[j-1]+dt),duration_s=float((j-i)*dt)) for i,j in zip(a,b)]
def main():
 result=json.loads((SMALL/(AT+'.json')).read_text())
 z=dict(np.load(OUT/'trajectory.npz'));p=np.concatenate([np.load(f)['physical'] for f in sorted(OUT.glob('physical-*.npz'))])
 cols={k:i for i,k in enumerate(result['physical_columns'])}
 mat=np.concatenate([np.load(f)['foot'] for f in sorted(OUT.glob('material-[0-9]*.npz'))])
 mc={k:i for i,k in enumerate(result['material']['foot_columns'])};mp=mat[:,:2];mf=mat[:,30:42].reshape(-1,4,3)
 events=[json.loads(l) for l in (OUT/'material-events.jsonl').read_text().splitlines()]
 failure=json.loads((OUT/'material-failures.jsonl').read_text().splitlines()[-1])
 t0=first(mp[:,0],(mf[:,:,2]>0).any(axis=1));first_i=np.flatnonzero((mf[:,:,2]>0).any(axis=1))[0]
 applied=mp[:,1]<=p[-1,1]+1e-10;rejected=~applied
 force_cols=[cols[f'grf_{leg}_z'] for leg in LEGS];fz=p[:,force_cols]
 loaded=fz>1;nonfoot=p[:,[cols[f'nonfoot_grf_{a}'] for a in 'xyz']]
 windows={}
 for label,a,b in [('moving_before_material',2.002,t0),('established_motion_before_material',3.,t0),('after_first_material_load',t0,float(z['time'][-1]))]:
  m=(p[:,0]>=a-1e-9)&(p[:,0]<b-1e-9);h=(z['time']>=a-1e-9)&(z['time']<b-1e-9)
  pp=p[m];ll=loaded[m];zz=z['qpos'][h];rr=z['rpy'][h];codes=np.sum(ll*(1<<np.arange(4)),axis=1)
  un={leg:[r for r in intervals(pp[:,0],~ll[:,j],.0002) if r['duration_s']>=.05] for j,leg in enumerate(LEGS)}
  windows[label]=dict(interval_s=[a,b],duration_s=b-a,body_start_m=zz[0,:3],body_last_sample_m=zz[-1,:3],net_dx_m=float(zz[-1,0]-zz[0,0]),
   min_body_z_m=float(zz[:,2].min()),max_body_z_m=float(zz[:,2].max()),max_abs_roll_pitch_yaw_rad=abs(rr).max(axis=0),
   mean_vx_m_s=float(z['qvel'][h,0].mean()),fraction_positive_vx=float(np.mean(z['qvel'][h,0]>0)),
   max_abs_y_m=float(abs(zz[:,1]).max()),max_nonfoot_force_N=float(np.linalg.norm(nonfoot[m],axis=1).max()),
   foot_clearance_max_m=(z['feet'][h,:,2]-.022).max(axis=0),actual_loaded_fraction=ll.mean(axis=0),
   four_feet_loaded_fraction=float(np.mean(codes==15)),diagonal_FL_RR_only_fraction=float(np.mean(codes==9)),diagonal_FR_RL_only_fraction=float(np.mean(codes==6)),
   no_feet_above_1N_fraction=float(np.mean(codes==0)),unloaded_intervals_at_least_50ms=un,
   raw_physical_peak_foot_Fz_N=fz[m].max(axis=0),threshold_note='1 N used only for analysis, never changes actual contact or termination.')
 bins=[]
 for a in np.arange(2.5,6.,.5):
  b=min(a+.5,t0);i=np.argmin(abs(z['time']-a));j=np.argmin(abs(z['time']-b));bins.append(dict(start_s=a,end_s=b,dx_m=z['qpos'][j,0]-z['qpos'][i,0]))
 first_by_leg={leg:dict(soft_footprint_boundary_s=first(mp[:,0],mat[:,2+j]>0),positive_force_boundary_s=first(mp[:,0],mf[:,j,2]>0),
   positive_force_applied_steps=int(np.sum((mf[:,j,2]>0)&applied)),peak_applied_Fz_N=float(mf[applied,j,2].max())) for j,leg in enumerate(LEGS)}
 final=dict(np.load(OUT/'material-final.npz'));imax=np.unravel_index(np.argmax(abs(final['e'])),final['e'].shape)
 same_peak=np.flatnonzero(np.max(abs(final['e']),axis=1)>=.02)
 # Every installed policy, same-time held action and accepted forecast are read.
 calls=[json.loads(l) for l in (OUT/'w53-n-native/calls.jsonl').read_text().splitlines()]
 policies={int(f.stem.split('-')[-1]):read_policy(f) for f in sorted((OUT/'w53-n-native').glob('policy-*.bin'))}
 held=[json.loads(l) for l in (OUT/'w53-n-held.jsonl').read_text().splitlines()]
 resolved=json.loads((SMALL/(AT+'-resolved.json')).read_text());limits=np.asarray(resolved['actuator_ctrlrange'])
 raw_errors=[];wire_errors=[];clip_deltas=[];feedback_norm=[];expiry=[];policy_age=[]
 for row in held:
  pp=policies[row['policy_call_id']];tt=np.asarray(pp['reference_times']);t=float(np.float32(row['time_s']));H=len(tt)
  i=min(max(int(np.searchsorted(tt,t,side='right')-1),0),H-2);w=np.clip((t-tt[i])/(tt[i+1]-tt[i]),0,1)
  def mix(a):return (1-w)*a[i]+w*a[i+1]
  ref=mix(pp['reference_states']);ref[3:7]/=np.linalg.norm(ref[3:7]);act=np.r_[row['qpos'],row['qvel']]
  rot=(Rotation.from_quat(ref[[4,5,6,3]]).inv()*Rotation.from_quat(act[[4,5,6,3]])).as_rotvec()
  dx=np.r_[act[:3]-ref[:3],rot,act[7:19]-ref[7:19],act[19:]-ref[19:]]
  gain=mix(pp['feedback_gain']);fb=pp['feedback_scaling']*gain@dx;raw=mix(pp['reference_actions'])+fb;clipped=np.clip(raw,limits[:,0],limits[:,1])
  raw_errors.append(float(np.max(abs(clipped-np.asarray(row['action'])))));wire_errors.append(float(np.max(abs(clipped.astype(np.float32).astype(float)-np.asarray(row['action'])))));clip_deltas.append(float(np.max(abs(clipped-raw))))
  feedback_norm.append(float(np.max(abs(fb))));expiry.append(t-tt[-1]);policy_age.append(t-pp['time_s'])
 # Forecast alignment uses only recorded x; no FK, model evaluation or replay.
 forecasts={}
 for ahead in [.02,.10,.35]:
  data={k:[] for k in ['before_material','crosses_first_material_load','already_material_exposed']}
  for call,pp in policies.items():
   source=float(pp['time_s']);target=source+ahead
   if source<2.002-1e-9 or target>z['time'][-1]+1e-9:continue
   tt=pp['accepted_times'];st=pp['accepted_states'];pred=np.array([np.interp(target,tt,st[:,j]) for j in range(st.shape[1])])
   actual=np.r_[np.array([np.interp(target,z['time'],z['qpos'][:,j]) for j in range(19)]),np.array([np.interp(target,z['time'],z['qvel'][:,j]) for j in range(18)])]
   category='before_material' if target<t0 else ('crosses_first_material_load' if source<t0 else 'already_material_exposed')
   data[category].append([source,target,*abs(pred[:3]-actual[:3]),np.linalg.norm(pred[:3]-actual[:3]),np.sqrt(np.mean((pred[7:19]-actual[7:19])**2)),np.linalg.norm(pred[19:22]-actual[19:22])])
  forecasts[str(ahead)]= {k:(dict(count=len(v),body_position_rmse_m=float(np.sqrt(np.mean(np.array(v)[:,5]**2))),
   max_body_position_error_m=float(np.max(np.array(v)[:,5])),joint_rmse_rad=float(np.sqrt(np.mean(np.array(v)[:,6]**2))),
   body_velocity_rmse_m_s=float(np.sqrt(np.mean(np.array(v)[:,7]**2)))) if v else dict(count=0)) for k,v in data.items()}
 last=policies[calls[-1]['policy_call_id']];lt=last['accepted_times'];ls=last['accepted_states']
 force_limits=np.asarray(resolved['actuator_forcerange']);requests=p[:,[cols[f'unclipped_pd_request_{j}'] for j in range(12)]]
 excess=np.maximum(np.maximum(requests-force_limits[:,1],force_limits[:,0]-requests),0.)
 actualforce=p[:,[cols[f'actuator_force_{j}'] for j in range(12)]]
 rec=json.loads((OUT/'w53-n-recording.json').read_text());runtime=json.loads((OUT.parent/'jobs/job-20261117T023608Z-w53-n-a/runtime.json').read_text());stage=next(v for v in runtime['stages'] if v['attempt']==AT)
 usage=stage['resource_usage'];wall=stage['ended_unix_s']-stage['started_unix_s'];cpu=usage['user_cpu_s']+usage['system_cpu_s']
 resource_rows=[json.loads(l) for l in (OUT.parent/'jobs/job-20261117T023608Z-w53-n-a/resources.jsonl').read_text().splitlines()]
 resource_rows=[r for r in resource_rows if r['root_pid']==stage['pid']]
 totals={k:sum(v[k] for v in calls) for k in ('full_rollouts','transition_fd_calls')}
 answer=dict(schema='W53_N_existing_actual_arrays_v1',attempt=AT,source_commit=result['source_commit'],execution_fingerprint=result['execution_fingerprint'],
  conclusion=dict(implementation_completed=True,observed_motion_recovered_before_material=True,
   observed_motion_scope='Repeated actual foot unloading/lift and positive body progression with bounded attitude, before first material force. Parent full-video inspection is separate. Not a full E30 pass or general stability claim.',
   original_E30_success=False,termination='material_domain_exit',physical_end_s=float(z['time'][-1]),material_state_rejected_time_s=failure['material_state_time_s'],
   original_E60_E300_E600='unrun; N-A E30 failed',material_memory_benefit_evaluated=False,new_PPO_samples=0,new_PPO_updates=0,new_native_or_plant_from_analysis=0),
  coverage=dict(trajectory_rows=len(z['time']),physical_applied_steps=len(p),material_force_evaluations=len(mat),rejected_material_evaluations=int(rejected.sum()),
   native_calls=len(calls),installed_full_policies_read=len(policies),actual_held_rows=len(held),material_event_rows=len(events),
   leg_order=LEGS,terminal_duplicate_diagnostic_row=True,measurement_note='Terminal qpos/qvel/time is actual; final copied foot diagnostics are not counted as a fresh force sample.'),
  motion_windows=windows,half_second_dx=bins,
  actual_material=dict(first_by_leg=first_by_leg,first_positive_force_boundary_s=t0,first_material_state_time_s=float(mat[first_i,1]),
   first_positive_force_N=mf[first_i],first_bottom_m=mat[first_i,6:18].reshape(4,3),first_velocity_m_s=mat[first_i,18:30].reshape(4,3),
   first_base_position_interpolated_m=[float(np.interp(t0,z['time'],z['qpos'][:,j])) for j in range(3)],
   distinct_loaded_cells=int(np.sum(final['ever_loaded'])),world_cells=len(final['e']),events_by_type={k:sum(e['type']==k for e in events) for k in ('load','revisit','release')},
   cross_leg_revisit_events=sum(e.get('cross_leg',False) for e in events),same_leg_revisit_events=sum(e['type']=='revisit' and not e.get('cross_leg',False) for e in events),
   completed_legal_outbound_passages=0,base_reached_strip_entry=False,
   revisit_count_note='64 same-leg spatial-cell reactivation events, not64 steps, cross-leg revisits or independent episodes; no return exposure.',
   route_counter_note='The base ended x=.723845 before .8, so legal passage material_loaded=false does not mean no foot material load.',
   material_exposure_duration_s=float(z['time'][-1]-t0)),
  failure=dict(physical_reason=result['reason'],diagnostic=failure,physical_failure_pose=result['final_qpos'][:7],
   last_applied_force_boundary_s=float(mp[applied][-1,0]),last_applied_postintegration_s=float(p[-1,1]),
   last_applied_material_force_N=mf[applied][-1],rejected_not_applied_force_N=mf[rejected][-1],
   max_strain_after_last_applied_material_update=float(mat[applied,mc['max_strain']].max()),max_strain_rejected_update=float(mat[rejected,mc['max_strain']][-1]),
   strain_limit=.02,exceeding_world_keys=final['keys'][same_peak],exceeding_e=final['e'][same_peak],exceeding_q=final['q'][same_peak],
   origin='FL touchdown on previously virgin cells at material_state t=6.4642, then continuous-loaded indentation; principal e11 crosses .02 at the rejected 6.4658 material update.',
   last_FL_touchdown_sequence=[dict(physical_boundary_s=float(row[0]),material_state_s=float(row[1]),bottom_z_m=float(row[8]),vz_m_s=float(row[20]),Fz_N=float(row[32]),running_max_strain=float(row[46]),applied_to_plant=bool(flag)) for row,flag in zip(mat,applied) if row[0]>=6.464-1e-9],
   rate_extrema=result['material']['rate_extrema'],
   application_note='Ground updates/logs material before checking domain; on violation JointMaterial.before_physics returns False before applying candidate force or advancing MuJoCo. The last material state and candidate force are a rejected update. Physical qpos at6.4656 is the boundary that imposed >1mm indentation.',
   max_nonfoot_force_N=float(np.linalg.norm(nonfoot,axis=1).max()),invalid_contact_or_body_fall=False),
  native_policy=dict(**totals,installed=rec['installed_policies'],risk_fd_calls=0,full_candidate_calls=rec['full_candidate_calls'],
   actual_action_double_reconstruction_vs_received_max_abs_error_rad=max(raw_errors),actual_action_after_response_float32_reconstruction_max_abs_error_rad=max(wire_errors),ctrlrange_clipped_hold_count=int(np.sum(np.array(clip_deltas)>1e-10)),
   max_ctrlrange_clip_rad=max(clip_deltas),max_full_state_feedback_rad=max(feedback_norm),
   physical_total_PD_request_overlimit_gt_0p01Nm_steps=int(np.sum(np.max(excess,axis=1)>.01)),
   first_physical_total_PD_request_overlimit_gt_0p01Nm_s=first(p[:,0],np.max(excess,axis=1)>.01),max_physical_total_PD_request_overlimit_Nm=float(excess.max()),
   max_actual_actuator_force_limit_excess_Nm=float(np.maximum(np.maximum(actualforce-force_limits[:,1],force_limits[:,0]-actualforce),0.).max()),genuine_reference_tail_expiry_count=int(np.sum(np.array(expiry)>1e-9)),
   action_time_semantics='GetActionRequest.time and GetActionResponse.action are protobuf float32; request and response rounding included separately. SetState and stored physical times remain double.',
   fd_count_semantics=dict(unit='mjd_transitionFD C API invocations, not scalar perturbations or native rollouts',per_call=351,formula='175 nodes x 2 held 1ms transitionFD calls + terminal sensor-C call',horizon=176,derivative_skip=0,physical_planner_calls=344,total=120744,measurement='Count inferred from restored fixed derivative path and complete calls; C++ record uses2*(H-1)+1. No independent scalar-perturbation counter.',comparison_to_W52='W52 increments r33_fd_counts immediately before mjd_transitionFD, so its transitionFD unit is also API invocations. Different per-call Jacobian outputs, predictor roles, substeps and risk calls prevent equal-work or speedup inference.'),
   forecast_error_note='Accepted same-time forecasts compared to actual future closed-loop state; later replanning and control changes contribute. These are not isolated model-error measurements.',
   max_policy_age_s=max(policy_age),last_policy_source_s=last['time_s'],last_accepted_horizon_s=[float(lt[0]),float(lt[-1])],
   last_accepted_body_z_range_m=[float(ls[:,2].min()),float(ls[:,2].max())],last_accepted_body_x_end_m=float(ls[-1,0]),
   prediction_error_by_horizon_and_material_exposure=forecasts,prediction_material_state='Absent by design; no TM02 future material dynamics in N.',
   foot_force_prediction_boundary='N records full policy and accepted state/action/residual/trace; no per-foot native GRF stream was exported. Direct native-vs-material force mismatch is not numerically claimed.'),
  resources=dict(stage_wall_including_process_start_exit_s=wall,reported_runtime_wall_s=result['wall_s'],user_cpu_s=usage['user_cpu_s'],system_cpu_s=usage['system_cpu_s'],cpu_core_hours=cpu/3600,
   equivalent_average_CPU_cores=cpu/wall,affinity_max_cores=10,children_maxrss_KiB=usage['children_maxrss_KiB'],
   sampled_process_tree_peak_rss_bytes=max(r['rss_bytes'] for r in resource_rows),sampled_process_tree_last_IO={k:resource_rows[-1][k] for k in ('read_bytes','write_bytes')},
   resource_sample_count=len(resource_rows),IO_note='Last live /proc sample precedes process exit; not full final filesystem IO.',
   runtime_timing=result['timing'],recorder=rec,stream_record_timing=result['record_timing'],material_wall_s=result['material']['wall_s'],observer=result['policy']['observer']),
  sources=[str((SMALL/(AT+'.json')).relative_to(ROOT)),str((SMALL/(AT+'-resolved.json')).relative_to(ROOT)),str(OUT.relative_to(ROOT))])
 path=SMALL/'w53/n-actual-analysis.json';path.write_text(json.dumps(answer,indent=2,default=val,allow_nan=False)+'\n')
 print(json.dumps(dict(output=str(path),first_load_s=t0,final_x_m=float(z['qpos'][-1,0]),normal_motion=answer['conclusion']['observed_motion_recovered_before_material'],
  last_applied_strain=answer['failure']['max_strain_after_last_applied_material_update'],rejected_strain=answer['failure']['max_strain_rejected_update'],
  reconstruction_error=max(raw_errors),clip_count=answer['native_policy']['ctrlrange_clipped_hold_count'],expiry=answer['native_policy']['genuine_reference_tail_expiry_count'],cpu_core_hours=cpu/3600),indent=2))
if __name__=='__main__':main()
