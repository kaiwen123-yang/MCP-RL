#!/usr/bin/env python3
"""Read the entire actual F record without new physics, optimization or FD."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-local-readiness-r12'
def analyze(attempt):
 small=ROOT/'results/terrain_memory/simulation'/RUN/'w54'
 out=ROOT/'outputs/terrain_memory/simulation'/RUN/attempt;folder=out/'w54-f'
 result=json.loads((small/(attempt+'.json')).read_text())
 resolved=json.loads((small/(attempt+'-resolved.json')).read_text())
 cfg=result['w54_f_recording']['config'];build=cfg['identity']
 z=np.load(out/'trajectory.npz')
 chunks=[np.load(p) for p in sorted(folder.glob('control-*.npz'))]
 c={k:np.concatenate([a[k] for a in chunks]) for k in chunks[0].files}
 physical=np.concatenate([np.load(p)['physical'] for p in sorted(out.glob('physical-*.npz'))])
 cols=result['physical_columns'];ix={name:i for i,name in enumerate(cols)}
 events=[json.loads(line) for line in (folder/'contact-events.jsonl').read_text().splitlines()]
 planlist=sorted(folder.glob('mpc-*.npz'))
 # Read every complete horizon, preserving actual source and no re-solves.
 plans=[np.load(p) for p in planlist]
 moving=c['time']>=2.002-1e-8
 missing=c['planned_contact'] & c['observed_contact'] & ~c['active_wbc_contact']
 rpy=z['rpy'];pos=z['qpos'][:,:3]
 windows=[]
 for lo,hi in [(0,2),(2,float(z['time'][-1]))]:
  m=(z['time']>=lo-1e-8)&(z['time']<=hi+1e-8)
  if not m.any():continue
  windows.append(dict(start_s=lo,end_s=hi,first_xyz_m=pos[m][0].tolist(),last_xyz_m=pos[m][-1].tolist(),
   z_min_m=float(pos[m,2].min()),z_max_m=float(pos[m,2].max()),max_abs_y_m=float(abs(pos[m,1]).max()),
   max_abs_roll_pitch_rad=abs(rpy[m,:2]).max(0).tolist()))
 requested=physical[:,[ix['unclipped_pd_request_'+str(i)] for i in range(12)]]
 actual=physical[:,[ix['actuator_force_'+str(i)] for i in range(12)]]
 limits=np.asarray(resolved['actuator_forcerange'])
 force=physical[:,[ix['grf_'+l+'_'+a] for l in ('FL','FR','RL','RR') for a in 'xyz']].reshape(-1,4,3)
 nonfoot=physical[:,[ix['nonfoot_grf_'+a] for a in 'xyz']]
 actual_qacc_res=np.linalg.norm(c['feet_acceleration_residual'],axis=2)
 liftoff=[e for e in events if e['kind']=='liftoff_committed']
 missing_examples=[]
 for at in [0.04,3.6,4.0]:
  i=int(np.argmin(abs(c['time']-at)))
  missing_examples.append(dict(time_s=float(c['time'][i]),measured_Fz_N=c['measured_grf'][i,:,2].tolist(),
   observed_contact=c['observed_contact'][i].tolist(),confirmed_contact=c['confirmed_contact'][i].tolist(),
   active_contact=c['active_wbc_contact'][i].tolist(),body_target_acceleration=c['body_acceleration_reference'][i].tolist()))
 physical_tau_violation=np.maximum(np.maximum(limits[:,0]-actual,actual-limits[:,1]),0)
 ans=dict(schema='w54_f_actual_analysis_v1',attempt=attempt,build=build,
  status=result['stage_status'],reason=result['reason'],success=result['success'],
  physical_end_s=result['simulation_s'],physical_samples=len(physical),trajectory_samples=len(z['time']),
  all_control_samples=len(c['time']),all_horizon_samples=len(plans),eval_wall_s=result['wall_s'],
  solver_failure=json.loads((folder/'solver-failure.json').read_text()) if (folder/'solver-failure.json').exists() else None,
  new_analysis_physics_rollout_FD_QP=0,windows=windows,
  max_x_m=float(pos[:,0].max()),final_xyz_m=pos[-1].tolist(),
  material_loaded=result['stage_progress']['actual_material_loaded'],
  maximum_nonfoot_force_N=float(np.linalg.norm(nonfoot,axis=1).max()),
  liftoff_events=liftoff,all_contact_events=events,
  support=dict(moving_ticks=int(moving.sum()),zero_active_moving_ticks=int((~c['active_wbc_contact'][moving].any(1)).sum()),
   zero_measured_moving_ticks=int((~c['observed_contact'][moving].any(1)).sum()),
   measured_non_swing_but_forced_zero_force_foot_ticks=int(missing.sum()),
   measured_non_swing_but_forced_zero_force_time_s_by_foot=(missing.sum(0)*.002).tolist(),
   saved_counterexamples=missing_examples),
  task_execution=dict(max_PD_mapping_error_Nm=float(abs(c['requested_pd']-c['tau_total']).max()),
   max_floating_dynamics_residual=float(abs(c['dynamics_residual']).max()),
   max_WBC_torque_violation_Nm=float(max(np.max(c['torque_lower']-c['tau_total']),np.max(c['tau_total']-c['torque_upper']),0)),
   max_actual_actuator_torque_violation_Nm=float(physical_tau_violation.max()),
   max_requested_minus_actual_torque_Nm=float(abs(requested-actual).max()),
   foot_accel_residual_RMS_m_s2=np.sqrt(np.mean(actual_qacc_res**2,axis=0)).tolist(),
   first_motion_foot_accel_residual_RMS_m_s2=np.sqrt(np.mean(actual_qacc_res[(c['time']>=2)&(c['time']<=2.3)]**2,axis=0)).tolist(),
   estimated_actual_friction_sum_RMS_Nm=np.sqrt(np.mean((c['friction_compensation']+c['actual_friction_constraint'])**2,axis=0)).tolist()),
  complete_recording=dict(actual_physics_columns=cols,
   has_5000Hz_foot_velocity=any('foot_velocity' in k or 'feet_velocity' in k for k in cols),
   preserved_500Hz_foot_velocity=True,full_trajectory_physics_control_plans_read=True),
  interpretation='Real steps and delayed load handovers occurred, but no sustained normal walking or original task qualification was demonstrated.',
  material_memory_evaluated=False,highest_original_qualification=None,
  E60_E300_E600='unrun; E30 did not qualify',new_PPO_samples=0,new_PPO_updates=0,donor=0,
  timing=result['w54_f_recording']['timing'])
 if build=='S5-r45':
  ans['new_evidence_and_revision']=json.loads((ROOT/'experiments/terrain_memory/s1-w54-f-r46.json').read_text())['revision']
  ans['video_review']=dict(reviewer='root agent',full_published_frames_viewed=125,frame_rate_Hz=30,
   motion_frames_enlarged_viewed=65,coverage='all125frames plus all65motionframes2.0-4.134s enlarged',
   observation='Real steps with repeated crouching/height variation, then lateral displacement, leg spreading and marked height loss.',
   boundary='30fps video cannot resolve sub-millisecond contact errors; exact mechanisms come from the complete arrays.')
 path=small/('f-'+build.removeprefix('S5-')+'-actual-analysis.json');path.write_text(json.dumps(ans,indent=2)+'\n')
 print(path);print(json.dumps({k:ans[k] for k in ('build','physical_end_s','max_x_m','final_xyz_m','support','task_execution')},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--attempt',required=True);analyze(ap.parse_args().attempt)
