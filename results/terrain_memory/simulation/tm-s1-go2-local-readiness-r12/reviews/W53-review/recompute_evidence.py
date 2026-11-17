"""Independent W53 saved-stream analysis; no simulation, solving or training."""
import argparse
import json
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[6])
parser.add_argument('--originals',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path(__file__).with_name('evidence.json'))
args=parser.parse_args()
R=args.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
O=args.originals
LEGS=['FL','FR','RL','RR']
DT=.0002

def clean(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 return v

def load_npz(p):
 with np.load(p) as f:return {k:f[k] for k in f.files}

def intervals(t,mask,dt):
 edges=np.diff(np.r_[False,mask,False].astype(int));st=np.flatnonzero(edges==1);en=np.flatnonzero(edges==-1)
 return [dict(start_s=float(t[i]),end_s=float(t[j-1]+dt),duration_s=float((j-i)*dt)) for i,j in zip(st,en)]

def summaries(tr,ph,cols,start,end):
 t=tr['time'];m=(t>=start-1e-9)&(t<end-1e-9)
 p=(ph[:,0]>=start-1e-9)&(ph[:,0]<end-1e-9)
 fz=ph[p][:,[cols.index('grf_'+l+'_z') for l in LEGS]];b=fz>1
 clearance=tr['feet'][m,:,2]-.022
 return dict(start_s=start,end_s=end,n_500Hz=int(m.sum()),n_physical=int(p.sum()),dx_m=float(tr['qpos'][m][-1,0]-tr['qpos'][m][0,0]),mean_vx_m_s=float(tr['qvel'][m,0].mean()),fraction_positive_vx=float(np.mean(tr['qvel'][m,0]>0)),body_z_minmax_m=[float(tr['qpos'][m,2].min()),float(tr['qpos'][m,2].max())],max_abs_rpy_rad=np.abs(tr['rpy'][m]).max(axis=0),max_abs_y_m=float(np.abs(tr['qpos'][m,1]).max()),max_nonfoot_N=float(np.linalg.norm(ph[p][:,[cols.index('nonfoot_grf_'+a) for a in 'xyz']],axis=1).max()),max_clearance_m=clearance.max(axis=0),foot_Fz_max_N=fz.max(axis=0),foot_Fz_mean_N=fz.mean(axis=0),four_loaded_fraction=float(np.mean(b.all(axis=1))),none_loaded_fraction=float(np.mean(~b.any(axis=1))),none_loaded_intervals=intervals(ph[p,0],~b.any(axis=1),DT),unload_intervals_50ms={l:[r for r in intervals(ph[p,0],~b[:,i],DT) if r['duration_s']>=.05-1e-10] for i,l in enumerate(LEGS)})

result={'scope':'Independent calculations on saved actual arrays only. No new physics, solver, native rollout, FD, or learning.'}
for arm in ['N','F']:
 p=next(O.glob('job-*w53-'+arm.lower()+'-a*'));tr=load_npz(p/'trajectory.npz');ph=np.concatenate([load_npz(x)['physical'] for x in sorted(p.glob('physical-*.npz'))]);raw=json.loads((R/(p.name+'.json')).read_text());cols=raw['physical_columns'];ft=[cols.index('grf_'+l+'_z') for l in LEGS]
 s=dict(attempt=p.name,trajectory_rows=len(tr['time']),physical_rows=len(ph),end_s=tr['time'][-1],last_diagnostic_time_s=tr['diagnostic_time'][-1],last_force_time_s=ph[-1,0],last_integrated_time_s=ph[-1,1],max_x_m=tr['qpos'][:,0].max(),last_xyz_m=tr['qpos'][-1,:3],last_rpy_rad=tr['rpy'][-1],max_nonfoot_N=np.linalg.norm(ph[:,[cols.index('nonfoot_grf_'+a) for a in 'xyz']],axis=1).max(),status=raw['status'],reason=raw['reason'])
 lim=np.tile([23.7,23.7,45.43],4);req=ph[:,[cols.index('unclipped_pd_request_'+str(i)) for i in range(12)]];ex=np.max(np.abs(req)-lim,axis=1);act=ph[:,[cols.index('actuator_force_'+str(i)) for i in range(12)]]
 s['torque']=dict(requested_over_limit_gt_0p01_steps=int((ex>.01).sum()),first_excess_s=float(ph[np.flatnonzero(ex>.01)[0],0]) if (ex>.01).any() else None,max_requested_excess_Nm=ex.max(),max_actual_excess_Nm=max(0.,float((np.abs(act)-lim).max())))
 mat=np.concatenate([load_npz(x)['foot'] for x in sorted(p.glob('material-*.npz')) if 'final' not in x.name]);mcols=raw['material']['foot_columns'];mfz=mat[:,[mcols.index('force_'+str(i)+'_z') for i in range(4)]];loads=np.flatnonzero((mfz>0).any(axis=1));first=float(mat[loads[0],0]) if len(loads) else None
 s['material']=dict(evaluation_rows=len(mat),first_positive_boundary_s=first,max_recorded_strain=float(mat[:,mcols.index('max_strain')].max()),last_material_boundary_s=float(mat[-1,0]),last_material_state_s=float(mat[-1,1]),foot_force_max_including_rejected_N=mfz.max(axis=0))
 if first is not None:
  s['windows']={'moving_before_material':summaries(tr,ph,cols,2.002,first),'established_before_material':summaries(tr,ph,cols,3.0,first),'after_material':summaries(tr,ph,cols,first,float(tr['time'][-1]))}
  last=mat[mat[:,0]>=6.464-1e-9];s['last_FL_impact']=[dict(force_time_s=x[0],material_time_s=x[1],bottom_z_m=x[mcols.index('bottom_0_z')],vz_m_s=x[mcols.index('velocity_0_z')],fz_N=x[mcols.index('force_0_z')],max_strain=x[mcols.index('max_strain')],applied=bool(x[0]<tr['time'][-1]-1e-9)) for x in last]
  applied=mat[:,0]<tr['time'][-1]-1e-9
  s['material'].update(last_applied_strain=mat[applied,-2][-1],peak_applied_FL_N=mfz[applied,0].max(),rejected_evaluations=int((~applied).sum()),first_base_x_interpolated_m=float(np.interp(first,tr['time'],tr['qpos'][:,0])))
  ev=[json.loads(x) for x in (p/'material-events.jsonl').read_text().splitlines()];s['material']['events']={k:sum(x['type']==k for x in ev) for k in ['load','release','revisit']};s['material']['cross_leg_revisits']=sum(x['type']=='revisit' and x.get('cross_leg',False) for x in ev)
  final=load_npz(p/'material-final.npz');s['material']['ever_loaded_cells']=int(final['ever_loaded'].sum());s['material']['world_cells']=len(final['keys'])
  calls=[json.loads(x) for x in (p/'w53-n-native/calls.jsonl').read_text().splitlines()];s['native']={'calls':len(calls),'full_rollouts':sum(x['full_rollouts'] for x in calls),'transition_FD_api_calls':sum(x['transition_fd_calls'] for x in calls),'installed':sum(x['installed'] for x in calls),'wall_s':{k:sum(x['wall_s'][k] for x in calls) for k in calls[0]['wall_s']},'cpu_s':np.sum([x['cpu_s'] for x in calls],axis=0),'full_candidate_call_ids':[x['call_id'] for x in calls if x['full_candidates']]}
 else:
  s['windows']={'moving':summaries(tr,ph,cols,2.002,float(tr['time'][-1]))}
  chunks=[load_npz(x) for x in sorted((p/'w53-f').glob('control-*.npz'))];ct={k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]};tt=ct['time'];s['WBC']={'records':len(tt),'zero_observed_ticks_after_2s':int(((tt>=2.002-1e-9)&~ct['observed_contact'].any(axis=1)).sum()),'zero_active_ticks_after_2s':int(((tt>=2.002-1e-9)&~ct['active_wbc_contact'].any(axis=1)).sum()),'max_PD_map_error_Nm':np.abs(ct['tau_total']-ct['requested_pd']).max(),'max_float_base_residual':np.abs(ct['dynamics_residual']).max()}
  touch=[]
  for i,l in enumerate(LEGS):
   rises=np.flatnonzero(np.diff(ct['planned_contact'][:,i].astype(int))==1)+1
   for row in rises:
    if tt[row]<2:continue
    observed=np.flatnonzero((tt>=tt[row]-1e-9)&ct['observed_contact'][:,i]);physical=np.flatnonzero((ph[:,0]>=tt[row]-1e-9)&(ph[:,ft[i]]>1))
    touch.append(dict(leg=l,planned_s=tt[row],control_observed_s=tt[observed[0]] if len(observed) else None,observed_delay_s=tt[observed[0]]-tt[row] if len(observed) else None,first_physical_force_s=ph[physical[0],0] if len(physical) else None,physical_delay_s=ph[physical[0],0]-tt[row] if len(physical) else None,foot_bottom_at_plan_m=float(np.interp(tt[row],tr['time'],tr['feet'][:,i,2]-.022))))
  s['touchdown']=touch
  hold_rows=np.clip(np.searchsorted(tt,ph[:,0]+1e-9,side='right')-1,0,len(tt)-1)
  delta=req-ct['tau_total'][hold_rows]
  s['WBC']['physical_hold_windows']={}
  for label,start,end in [('first_swing',2.052,2.254),('first_swing_until_both_touchdown',2.052,2.332),('moving_before_final_tick',2.002,3.118)]:
   mask=(ph[:,0]>=start-1e-9)&(ph[:,0]<end-1e-9)
   d=delta[mask];where=np.unravel_index(np.abs(d).argmax(),d.shape)
   s['WBC']['physical_hold_windows'][label]=dict(start_s=start,end_s=end,physical_steps=int(mask.sum()),max_actuator_minus_requested_Nm=float(np.abs(act[mask]-req[mask]).max()),max_requested_torque_fraction=float((np.abs(req[mask])/lim).max()),max_within_hold_minus_tick_WBC_Nm=float(np.abs(d).max()),peak_time_s=float(ph[mask,0][where[0]]),peak_joint_index=int(where[1]),per_joint_rms_hold_delta_Nm=np.sqrt(np.mean(d*d,axis=0)))
 stage=json.loads((R/'w53'/f'{arm}-A.json').read_text())['stages'][-1];wall=stage['ended_unix_s']-stage['started_unix_s'];cpu=stage['resource_usage']['user_cpu_s']+stage['resource_usage']['system_cpu_s'];s['resource']={'start_unix_s':stage['started_unix_s'],'end_unix_s':stage['ended_unix_s'],'stage_wall_s':wall,'eval_wall_s':raw['wall_s'],'user_plus_system_cpu_s':cpu,'average_cores_stage':cpu/wall,'eval_wall_per_physical_s':raw['wall_s']/s['end_s'],'workers_argument':stage['command'][stage['command'].index('--workers')+1]}
 result[arm]=s
n=result['N']['resource'];f=result['F']['resource'];result['actual_valid_job_overlap_s']=max(0,min(n['end_unix_s'],f['end_unix_s'])-max(n['start_unix_s'],f['start_unix_s']));result['gap_F_end_to_N_start_s']=n['start_unix_s']-f['end_unix_s']
result['counts']={'valid_physical_starts':2,'zero_plant_initialization_errors':2,'material_failure':1,'controller_execution_error_after_physical_start':1,'full_E30_passes':0,'plant_steps':result['N']['physical_rows']+result['F']['physical_rows'],'new_PPO_samples':0,'new_PPO_updates':0,'donor':0}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(clean({k:v for k,v in result.items() if k not in ['N','F']}),indent=2))
for arm in ['N','F']:
 d=result[arm];print(arm,json.dumps(clean({k:v for k,v in d.items() if k not in ['windows','last_FL_impact','native']}),indent=2)[:9000])
