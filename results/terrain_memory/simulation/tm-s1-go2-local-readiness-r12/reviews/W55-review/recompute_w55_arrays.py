"""Independent arithmetic on downloaded W55 recordings. No simulation/training."""
from pathlib import Path
import argparse,json,numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--originals',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
W=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55'
PHASES=['initial_stand','outbound','turn','return','terminal_stand','complete','far_hold','home_hold','turn_home']
cases={};array_cache={}
def native(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def passage(a,start,end,phase):
 t=a['time'];xy=a['qpos'][:,:2];ii=np.arange(start,end)
 entry=ii[xy[ii,0]>=.8] if phase=='outbound' else ii[xy[ii,0]<=2.2]
 en=int(entry[0]) if len(entry) else None
 ii=np.arange(en,end) if en is not None else np.array([],int)
 exits=ii[xy[ii,0]>=2.2] if phase=='outbound' else ii[xy[ii,0]<=.8]
 ex=int(exits[0]) if len(exits) else None
 seg=np.arange(en,(ex+1) if ex is not None else end) if en is not None else np.array([],int)
 feet=a['feet'][seg,:,:2];h=.015811388300841896
 inside=(feet[:,:,0]>=.8+h)&(feet[:,:,0]<=2.2-h)&(np.abs(feet[:,:,1])<=.6-h)
 loaded=bool(np.any(inside&(a['actual_grf'][seg,:,2]>0)))
 side=seg[np.abs(xy[seg,1])>.6]
 return dict(direction=phase,entry_s=float(t[en]) if en is not None else None,exit_s=float(t[ex]) if ex is not None else None,
  lateral_exit_s=float(t[side[0]]) if len(side) else None,material_loaded=loaded,
  completed=bool(en is not None and ex is not None and not len(side) and loaded),
  max_abs_base_y_m=float(np.max(np.abs(xy[seg,1]))) if len(seg) else None)
def window(d,lo,hi):
 s=(d['time']>=lo-1e-8)&(d['time']<=hi+1e-8)
 if not s.any():return None
 q=d['qpos'][s];v=d['qvel'][s];r=d['rpy'][s]
 return dict(samples=int(s.sum()),start_s=float(d['time'][s][0]),end_s=float(d['time'][s][-1]),
  min_z_m=float(q[:,2].min()),max_z_m=float(q[:,2].max()),max_abs_roll_pitch_rad=float(np.abs(r[:,:2]).max()),
  max_origin_distance_m=float(np.linalg.norm(q[:,:2],axis=1).max()),max_speed_m_s=float(np.linalg.norm(v[:,:3],axis=1).max()),
  max_omega_rad_s=float(np.linalg.norm(v[:,3:6],axis=1).max()),all_four_contact_fraction=float(d['contact'][s].all(axis=1).mean()),
  max_abs_torque_Nm=float(np.abs(d['torque'][s]).max()),max_nonfoot_ground_contacts=int(d['nonfoot_ground_contacts'][s].max()))
for batch in sorted(a.originals.glob('*-e*')):
 paths=list(batch.glob('job-*/trajectory.npz'))
 if not paths:continue
 f=paths[0];attempt=f.parent.name
 rp=next((q for q in W.rglob(attempt+'.json') if 'delivery' not in str(q)),None)
 if rp is None:continue
 r=json.loads(rp.read_text())
 with np.load(f,allow_pickle=False) as z:
  d={k:z[k] for k in ['time','qpos','qvel','rpy','phase','contact','feet','actual_grf','nonfoot_ground_contacts','torque','actuator_force','command_velocity','feedback_s','planner_s']}
 t=d['time'];phase=d['phase'];bounds=np.r_[0,np.flatnonzero(phase[1:]!=phase[:-1])+1,len(t)]
 segs=[dict(phase=PHASES[int(phase[s])],start_s=float(t[s]),end_s=float(t[e-1]),samples=int(e-s)) for s,e in zip(bounds[:-1],bounds[1:])]
 passages=[passage(d,s,e,PHASES[int(phase[s])]) for s,e in zip(bounds[:-1],bounds[1:]) if phase[s] in [1,3]]
 checks=dict(contact=d['contact'].all(axis=1),position=np.linalg.norm(d['qpos'][:,:2],axis=1)<.15,
  linear_speed=np.linalg.norm(d['qvel'][:,:3],axis=1)<.05,angular_speed=np.linalg.norm(d['qvel'][:,3:6],axis=1)<.1,
  roll_pitch=np.max(np.abs(d['rpy'][:,:2]),axis=1)<.15)
 good=np.logical_and.reduce(list(checks.values()));first=r.get('first_completion_s')
 hs=(t>=first-1e-8) if first is not None else np.zeros(len(t),bool)
 hold=None
 if hs.any():
  valid=good[hs];times=t[hs];tail=(times[-1]-times[np.flatnonzero(~valid)[-1]+1]) if valid[-1] and (~valid).any() else times[-1]-times[0] if valid[-1] else 0.
  hold=dict(all_criteria_fraction=float(valid.mean()),violation_samples=int((~valid).sum()),
   endpoint_position_violation_samples=int((~checks['position'][hs]).sum()),final_stable_s=float(tail),
   criterion_pass_fraction={k:float(v[hs].mean()) for k,v in checks.items()},**window(d,float(first),float(t[-1])))
 phys=[];columns=r.get('physical_columns',[])
 for pf in sorted(f.parent.glob('physical-*.npz')):
  with np.load(pf,allow_pickle=False) as z:x=z['physical']
  torque=x[:,[columns.index('torque_'+str(i)) for i in range(12)]];act=x[:,[columns.index('actuator_force_'+str(i)) for i in range(12)]]
  lim=np.array([23.7,23.7,45.43]*4)
  nonfoot=x[:,columns.index('nonfoot_ground_contacts')]>0
  nf=x[nonfoot][:,[columns.index('nonfoot_grf_'+ax) for ax in 'xyz']]
  phys.append(dict(file=pf.name,samples=int(len(x)),first_force_time_s=float(x[0,0]),last_force_time_s=float(x[-1,0]),
   max_abs_torque_by_joint=np.abs(torque).max(axis=0),max_abs_actuator_force_by_joint=np.abs(act).max(axis=0),
   torque_limit_exceed_values=int((np.abs(torque)>lim+1e-7).sum()),actuator_limit_exceed_values=int((np.abs(act)>lim+1e-7).sum()),
   torque_at_limit_values=int((np.abs(torque)>=lim-1e-5).sum()),nonfoot_contact_steps=int(nonfoot.sum()),
   first_nonfoot_contact_s=float(x[nonfoot,0][0]) if nonfoot.any() else None,
   max_nonfoot_grf_norm_N=float(np.linalg.norm(nf,axis=1).max()) if nonfoot.any() else 0.,
   max_nonfoot_grf_Fz_N=float(nf[:,2].max()) if nonfoot.any() else 0.))
  del x
 reported=r.get('legal_route',{}).get('passages',[])
 pass_match=len(reported)==len(passages) and all(all(p0.get(k)==p1.get(k) for k in ['direction','entry_s','exit_s','lateral_exit_s','material_loaded','completed']) for p0,p1 in zip(passages,reported))
 out=dict(attempt=attempt,trajectory=str(f),result=str(rp.relative_to(a.repo)),samples_500Hz=len(t),last_time_s=float(t[-1]),
  sample_deltas_s=dict(min=float(np.diff(t).min()),max=float(np.diff(t).max())),
  status=r['stage_status'],reason=r['reason'],first_completion_s=first,
  phase_segments=segs,passages=passages,passages_exactly_match_reported=bool(pass_match),
  legal_outbound_traversals=sum(p['direction']=='outbound' and p['completed'] for p in passages),
  legal_return_traversals=sum(p['direction']=='return' and p['completed'] for p in passages),
  complete_phase_transitions=sum(s['phase']=='complete' for s in segs),
  roundtrip_completion_times_s=[cur['start_s'] for prev,cur in zip(segs[:-1],segs[1:]) if prev['phase'] in ['home_hold','terminal_stand'] and cur['phase'] in ['turn_home','complete']],
  reported_completed_roundtrips=r.get('legal_route',{}).get('completed_roundtrips'),
  max_x_m=float(d['qpos'][:,0].max()),min_z_m=float(d['qpos'][:,2].min()),max_abs_y_m=float(np.abs(d['qpos'][:,1]).max()),
  max_abs_roll_pitch_rad=float(np.abs(d['rpy'][:,:2]).max()),final_qpos=d['qpos'][-1],final_qvel=d['qvel'][-1],
  hold_recomputed=hold,windows={f'{lo}-{hi}':window(d,lo,hi) for lo,hi in [(0,36),(36,40),(40,float(t[-1])),(float(t[-1])-10,float(t[-1]))]},
  physical_recomputed=phys,feedback_ms={str(q):float(np.percentile(d['feedback_s']*1000,q)) for q in [0,50,95,99,100]},
  reported_feedback_misses=r.get('timing',{}).get('feedback',{}).get('deadline_misses'))
 if batch.name=='n-a-r54-e30':
  out['progress_windows']=[dict(t_s=float(t[np.searchsorted(t,tm)]),x_m=float(d['qpos'][np.searchsorted(t,tm),0]),
   command_speed_m_s=float(np.linalg.norm(d['command_velocity'][np.searchsorted(t,tm),:2])),
   actual_speed_m_s=float(np.linalg.norm(d['qvel'][np.searchsorted(t,tm),:2]))) for tm in [0,2,5,10,15,20,25,29.99]]
 cases[batch.name]=out
 if batch.name in ['f-a-r53-e60','f-a-r55-e60','f-b-r53-e60','f-b-r55-e60']:array_cache[batch.name]=d
comparisons={}
for seed in ['a','b']:
 old=array_cache.get(f'f-{seed}-r53-e60');new=array_cache.get(f'f-{seed}-r55-e60')
 if old is None or new is None:continue
 n=min(len(old['time']),len(new['time']));diff=np.max(np.abs(old['qpos'][:n]-new['qpos'][:n]),axis=1)
 changed=np.flatnonzero(diff>1e-12)
 comparisons[seed]=dict(common_samples=n,first_qpos_difference_over_1e12_s=float(old['time'][changed[0]]) if len(changed) else None,
  prefix_qpos_max_abs_difference=float(diff[:changed[0]].max()) if len(changed) and changed[0]>0 else None,
  max_qpos_difference_before_36s=float(diff[old['time'][:n]<36].max()),
  r53_final=window(old,float(old['time'][-1])-2,float(old['time'][-1])),
  r55_matched_time=window(new,float(old['time'][-1])-2,float(old['time'][-1])) )
a.output.write_text(json.dumps(dict(cases=cases,r53_r55_comparisons=comparisons),default=native,indent=2)+'\n')
for k,v in cases.items():print(k,json.dumps({n:v[n] for n in ['samples_500Hz','last_time_s','status','max_x_m','min_z_m','max_abs_y_m','max_abs_roll_pitch_rad','passages_exactly_match_reported','legal_outbound_traversals','legal_return_traversals','first_completion_s']}))
print('pair_comparisons',json.dumps(comparisons,default=native))
