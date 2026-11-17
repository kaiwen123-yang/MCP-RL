"""Independent NumPy checks of saved W54 arrays. Never steps a model or runs a solver."""
import argparse,collections,json,re
from pathlib import Path
import numpy as np
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[6]);P.add_argument('--originals',type=Path,required=True);P.add_argument('--output',type=Path,required=True);a=P.parse_args();a.output.mkdir(parents=True,exist_ok=True)
R=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w54';legs=['FL','FR','RL','RR'];dt=.0002
def clean(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 return x
def npz(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def first(t,mask):
 ix=np.flatnonzero(mask);return float(t[ix[0]]) if len(ix) else None
def spans(t,m,d=dt):
 edges=np.diff(np.r_[False,m,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
 return [dict(start_s=t[i],end_s=t[j-1]+d,duration_s=(j-i)*d) for i,j in zip(starts,ends)]
def window(tr,ph,cols,start,end):
 m=(tr['time']>=start-1e-8)&(tr['time']<end-1e-8);q=(ph[:,0]>=start-1e-8)&(ph[:,0]<end-1e-8)
 if not m.any() or not q.any():return None
 pos=tr['qpos'][m];vel=tr['qvel'][m];force=ph[q][:,[cols.index('grf_'+l+'_z') for l in legs]];support=(force>1).sum(axis=1);nf=np.linalg.norm(ph[q][:,[cols.index('nonfoot_grf_'+c) for c in 'xyz']],axis=1)
 return dict(start_s=start,end_s=end,trajectory_rows=m.sum(),physical_rows=q.sum(),dx_m=pos[-1,0]-pos[0,0],mean_vx_m_s=vel[:,0].mean(),command_vx_mean_m_s=tr['command_velocity'][m,0].mean(),fraction_vx_positive=np.mean(vel[:,0]>0),z_min=pos[:,2].min(),z_max=pos[:,2].max(),z_mean=pos[:,2].mean(),max_abs_rpy_rad=np.max(np.abs(tr['rpy'][m]),axis=0),four_foot_support_fraction=np.mean(support==4),zero_foot_support_duration_s=np.sum(support==0)*dt,zero_foot_support_fraction=np.mean(support==0),zero_foot_support_max_interval_s=max([v['duration_s'] for v in spans(ph[q,0],support==0)]+[0]),max_nonfoot_N=nf.max(),max_foot_Fz_N=force.max(axis=0))
phase=json.loads((R/'phase.json').read_text());stages={c['stage']['attempt']:c['stage'] for c in phase['cases']}
out={'schema':'w55_saved_w54_independent_evidence_v1','source_head':'1ae25272c86f551f8c6b3a65073bc05b24e3d8f6','no_new_physics_FD_QP_training':True,'runs':{}}
for p in sorted(a.originals.glob('job-*w54*attempt01')):
 rev=int(re.search(r'-a-r(\d+)-',p.name).group(1));arm='N' if '-n-a-' in p.name else 'F';key=arm+'-r'+str(rev);raw=json.loads((R/(p.name+'.json')).read_text());tr=npz(p/'trajectory.npz');ph=np.concatenate([npz(x)['physical'] for x in sorted(p.glob('physical-*.npz'))]);cols=raw['physical_columns'];mc=raw['material']['foot_columns'];t=ph[:,0]
 chunks=[npz(x) for x in sorted(p.glob('material-*.npz')) if x.name!='material-final.npz'];mat=np.concatenate([v['foot'] for v in chunks]);applied=mat[:,0]<tr['time'][-1]-1e-8;mfz=mat[:,[mc.index('force_'+str(i)+'_z') for i in range(4)]];firstmat=first(mat[:,0],applied&(mfz>0).any(axis=1));nf=np.linalg.norm(ph[:,[cols.index('nonfoot_grf_'+c) for c in 'xyz']],axis=1);fz=ph[:,[cols.index('grf_'+l+'_z') for l in legs]];sup=(fz>1).sum(axis=1)
 events=[json.loads(x) for x in (p/'material-events.jsonl').read_text().splitlines()];final=npz(p/'material-final.npz');stage=stages[p.name];stage_wall=stage['ended_unix_s']-stage['started_unix_s'];cpu=stage['resource_usage']['user_cpu_s']+stage['resource_usage']['system_cpu_s']
 ztimes={str(level):first(tr['time'],(tr['time']>=2.002-1e-8)&(tr['qpos'][:,2]<level)) for level in [.28,.27,.25,.2,.12]};limits=np.tile([23.7,23.7,45.43],4);req=ph[:,[cols.index('unclipped_pd_request_'+str(i)) for i in range(12)]];actual=ph[:,[cols.index('actuator_force_'+str(i)) for i in range(12)]];excess=np.max(np.abs(req)-limits,axis=1)
 d=dict(attempt=p.name,revision=rev,reason=raw['reason'],stage_status=raw['stage_status'],success=raw['success'],sim_s=tr['time'][-1],trajectory_rows=len(tr['time']),physical_rows=len(ph),physical_force_end_s=ph[-1,0],physical_integrated_end_s=ph[-1,1],last_diagnostic_s=tr['diagnostic_time'][-1],max_x_m=tr['qpos'][:,0].max(),final_xyz_m=tr['qpos'][-1,:3],final_rpy_rad=tr['rpy'][-1],height_first_below_after_motion_s=ztimes,first_nonfoot_force_gt_1N_s=first(t,nf>1),max_nonfoot_N=nf.max(),material=dict(foot_rows=len(mat),cells_rows=sum(len(x['cells']) for x in chunks),rates_rows=sum(len(x['rates']) for x in chunks),first_actual_positive_force_s=firstmat,first_actual_positive_force_by_foot_s={l:first(mat[:,0],applied&(mfz[:,i]>0)) for i,l in enumerate(legs)},exposure_s=float(tr['time'][-1])-firstmat if firstmat is not None else 0.,applied_foot_force_max_N=mfz[applied].max(axis=0),recorded_force_max_including_rejected_N=mfz.max(axis=0),max_recorded_strain=mat[:,-2].max(),max_applied_strain=mat[applied,-2].max(),rejected_candidate_evaluations=int((~applied).sum()),events=dict(collections.Counter(x['type'] for x in events)),cross_leg_revisit_count=sum(x['type']=='revisit' and x.get('cross_leg',False) for x in events),first_cross_leg_revisit_s=next((x['time_s'] for x in events if x['type']=='revisit' and x.get('cross_leg',False)),None),ever_loaded_cells=final['ever_loaded'].sum(),world_cells=len(final['keys'])),route=dict(transitions=raw['transitions'],arrivals=raw['arrivals'],completed_round_trips=raw['completed_round_trips'],original_mission_success=raw['original_mission_success'],first_completion_s=raw['first_completion_s']),torque=dict(requested_over_limit_gt_0p01_steps=np.sum(excess>.01),first_excess_s=first(t,excess>.01),max_requested_excess_Nm=excess.max(),max_actual_excess_Nm=max(0.,(np.abs(actual)-limits).max())),resource=dict(stage_start_unix_s=stage['started_unix_s'],stage_end_unix_s=stage['ended_unix_s'],physical_start_unix_s=raw['physical_started_unix_s'],physical_end_unix_s=raw['physical_ended_unix_s'],physical_clock=raw['physical_interval_clock'],stage_wall_s=stage_wall,eval_wall_s=raw['wall_s'],CPU_user_plus_system_s=cpu,average_stage_cores=cpu/stage_wall,workers=stage['command'][stage['command'].index('--workers')+1]),recording_timing_reported=raw['record_timing'])
 end=float(tr['time'][-1]);d['windows']={'moving':window(tr,ph,cols,2.002,end),'pre_material':window(tr,ph,cols,2.002,firstmat if firstmat else end),'post_material':window(tr,ph,cols,firstmat,end) if firstmat else None,'established_before_material':window(tr,ph,cols,3.,firstmat if firstmat else end),'last_second':window(tr,ph,cols,max(2.002,end-1),end)}
 if rev==50:d['windows'].update({f'interval_{start}_{endd}':window(tr,ph,cols,start,endd) for start,endd in [(2.002,4),(4,5),(5,10),(10,20),(20,30)]})
 d['zero_support_intervals_after_2s']=spans(t,(t>=2.002-1e-8)&(sup==0));d['material_last_evaluations']=mat[-10:]
 if arm=='N':
  calls=[json.loads(x) for x in next(p.rglob('calls.jsonl')).read_text().splitlines()];exc=[json.loads(x) for x in (p/'w54-n-execution.jsonl').read_text().splitlines()];diag=np.asarray([x['diagnostic'] for x in exc]);et=np.asarray([x['time_s'] for x in exc]);gate=diag[:,:,exc[0]['fields'].index('gate')];active=(gate>0).any(axis=1)
  d['native']=dict(calls=len(calls),installed=sum(v['installed'] for v in calls),full_rollouts=sum(v['full_rollouts'] for v in calls),transition_FD_reported_evaluations=sum(v['transition_fd_calls'] for v in calls),wall_s={k:sum(v['wall_s'][k] for v in calls) for k in calls[0]['wall_s']},cpu_s=np.sum([v['cpu_s'] for v in calls],axis=0),selection_counts=dict(collections.Counter(v.get('selection_source','unrecorded') for v in calls)),backward_failures=sum(v['backward_failed'] for v in calls) if all('backward_failed' in v for v in calls) else None,E_records=len(exc),E_first_active_s=first(et,active),E_active_ticks=active.sum(),native_bin_not_reconstructed=True,FD_count_semantics='W54 formula (horizon-1)*(1+state_derivative_dimension+action_dimension)+(1+state_derivative_dimension)=8612 per call; differs from W53 API-level 351 per call')
 else:
  control=[npz(x) for x in sorted(p.rglob('control-*.npz'))];ct={k:np.concatenate([x[k] for x in control]) for k in control[0]};tt=ct['time'];moving=tt>=2.002-1e-8
  d['control']=dict(records=len(tt),zero_observed_ticks=int(np.sum(moving&~ct['observed_contact'].any(axis=1))),zero_active_ticks=int(np.sum(moving&~ct['active_wbc_contact'].any(axis=1))))
  for k in ['load_bearing_contact','confirmed_contact']:
   if k in ct:d['control']['zero_'+k+'_ticks']=int(np.sum(moving&~ct[k].any(axis=1)))
  if 'tau_total' in ct and 'requested_pd' in ct:d['control']['max_PD_map_error_Nm']=np.abs(ct['tau_total']-ct['requested_pd']).max()
 out['runs'][key]=d
 print(key,round(end,4),d['reason'],round(float(d['max_x_m']),5),'z',round(float(d['final_xyz_m'][2]),5),'firstmaterial',firstmat,'nonfoot',d['first_nonfoot_force_gt_1N_s'],flush=True)
intervals=[(v['resource']['physical_start_unix_s'],v['resource']['physical_end_unix_s'],k) for k,v in out['runs'].items()];events=sorted([(st,1,k) for st,en,k in intervals]+[(en,-1,k) for st,en,k in intervals]);active=set();prev=None;periods=[]
for t,change,k in events:
 if prev is not None and t>prev and active:periods.append(dict(start_unix_s=prev,end_unix_s=t,duration_s=t-prev,active=sorted(active),concurrency=len(active)))
 if change>0:active.add(k)
 else:active.remove(k)
 prev=t
out['resource_overlap']=dict(physical_intervals=intervals,periods=periods,at_least_two_physical_s=sum(x['duration_s'] for x in periods if x['concurrency']>=2),at_least_three_physical_s=sum(x['duration_s'] for x in periods if x['concurrency']>=3),union_active_physical_s=sum(x['duration_s'] for x in periods),elapsed_first_start_to_last_end_s=max(v[1] for v in intervals)-min(v[0] for v in intervals),total_CPU_seconds=sum(v['resource']['CPU_user_plus_system_s'] for v in out['runs'].values()))
out['counts']=dict(valid_physical_starts=len(out['runs']),physical_steps=sum(v['physical_rows'] for v in out['runs'].values()),sim_s=sum(float(v['sim_s']) for v in out['runs'].values()),reasons=dict(collections.Counter(v['reason'] for v in out['runs'].values())),E30_passes=sum(v['success'] for v in out['runs'].values()),new_PPO_samples=0,new_PPO_updates=0,donor=0)
(a.output/'evidence.json').write_text(json.dumps(clean(out),indent=2,ensure_ascii=False)+'\n')
print(json.dumps(clean(out['counts']),ensure_ascii=False),flush=True)

gpu={'scope':'Saved optimizations only; independent recurrence/cost/friction checks; no QP or GPU rerun.','batches':{}}
for p in sorted((a.originals/'w54-C').glob('*')):
 if not (p/'result.json').exists():continue
 cfg=json.loads((p.parent/(p.name+'-inputs')/'configuration.json').read_text());raw=json.loads((p/'result.json').read_text());cases=[];mu=cfg['friction_coefficient'];slope=cfg.get('force_pyramid_slope',mu)
 for f in sorted(p.glob('case-*.npz')):
  q=npz(f);arms={}
  for name in ['GPU','CPU_QP']:
   force=q[name+'_forces'];state=q[name+'_states'];n=len(force);truth=np.zeros_like(state);truth[0]=q['x0']
   for i in range(n):truth[i+1]=q['stage_A'][i]@truth[i]+q['stage_B'][i]@force[i].reshape(-1)+q['stage_c'][i]
   weight=np.tile(cfg['mpc']['state_weights'],(n+1,1)).astype(float);weight[-1]*=2
   difference=np.diff(np.concatenate([q['last_force'][None],force]),axis=0);cost=float(np.sum(weight*(truth-q['reference'])**2)+cfg['mpc']['force_regularization']*np.sum(force**2)+cfg['mpc']['force_difference_weight']*np.sum(difference**2));fz=force[:,:,2]
   arms[name]=dict(objective=cost,dynamics_saved_vs_recurrence_max=np.abs(state-truth).max(),physical_cone_violation_N=max(0.,float(np.max(np.linalg.norm(force[:,:,:2],axis=-1)-mu*fz))),planning_pyramid_violation_N=max(0.,float(np.max(np.abs(force[:,:,:2])-slope*fz[:,:,None]))),inactive_force_max_N=np.max(np.abs(force*(~q['contact'])[:,:,None])),normal_lower_violation_N=max(0.,float(-fz.min())),normal_upper_violation_N=max(0.,float((fz-q['force_normal_upper']).max())))
  cases.append(dict(file=f.name,**arms,reported_GPU_objective_error=abs(float(q['GPU_objective'])-arms['GPU']['objective']),relative_objective_gap=(arms['GPU']['objective']-arms['CPU_QP']['objective'])/max(1.,abs(arms['CPU_QP']['objective'])),first_force_difference_N=np.abs(q['GPU_forces'][0]-q['CPU_QP_forces'][0]).max(),input_contact_patterns=np.unique(q['contact'],axis=0)))
 tim=raw['timing'];cpu=raw.get('CPU_reference_timing', {k:sum(v['CPU_QP'].get(k,0) for v in raw['cases']) for k in ['assembly_wall_s','setup_and_solve_wall_s']});cpu_total=cpu['assembly_wall_s']+cpu['setup_and_solve_wall_s'];warm=tim['GPU_path_without_initialization_compile_s'];cold=tim['GPU_path_including_initialization_compile_transfer_s']
 gpu['batches'][p.name]=dict(cases=cases,complete_horizons=len(cases),mu=mu,planning_slope=slope,max_relative_objective_gap=max(v['relative_objective_gap'] for v in cases),max_first_force_difference_N=max(v['first_force_difference_N'] for v in cases),max_GPU_physical_cone_violation_N=max(v['GPU']['physical_cone_violation_N'] for v in cases),max_CPU_physical_cone_violation_N=max(v['CPU_QP']['physical_cone_violation_N'] for v in cases),max_GPU_recurrence_error=max(v['GPU']['dynamics_saved_vs_recurrence_max'] for v in cases),timing_reported=tim,CPU_reference_timing_reported=cpu,CPU_assembly_plus_solve_s=cpu_total,GPU_warm_path_s=warm,GPU_cold_path_s=cold,CPU_over_warm_GPU_ratio=cpu_total/warm,CPU_over_cold_GPU_ratio=cpu_total/cold,input_preparation_timing_reported=raw.get('input_preparation_timing'),equal_optimization_budget=raw['equal_optimization_budget'],online_installed=raw['online_installed'],GPU_memory_stats_reported=raw.get('GPU_memory_stats'),started_unix_s=raw['started_unix_s'],ended_unix_s=raw['ended_unix_s'])
 print('GPU',p.name,len(cases),'gap',gpu['batches'][p.name]['max_relative_objective_gap'],flush=True)
(a.output/'gpu-evidence.json').write_text(json.dumps(clean(gpu),indent=2)+'\n')
