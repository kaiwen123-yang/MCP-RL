"""Analyze the saved W36 onset and the non-plant planner replay."""
import json,math
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
LARGE=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
JOB=LARGE/'job-20261101T133954Z'
source=LARGE/'train-PANN_PPO_S5-s0-ep00118'
a=np.load(source/'trajectory.npz')
m=mujoco.MjModel.from_xml_path(str(source/'model/G1-project-go2.xml'))

# Same-state/input force comparison; no plant integration.
prediction=mujoco.MjModel.from_xml_path(str(source/'model/G1-project-go2.xml'))
prediction.jnt_solimp[:,0]=0
prediction.geom_solimp[:,0]=0
prediction.pair_solimp[:,0]=0
feet=[m.geom(n).id for n in ['FL','FR','RL','RR']]
contact_records=[]
for time_s in (2.002,2.004,2.006,2.008,2.010,2.020):
 i=int(np.argmin(abs(a['time']-time_s)));row=dict(t=float(a['time'][i]))
 for label,model in [('native',m),('differentiable',prediction)]:
  d=mujoco.MjData(model);d.qpos[:]=a['qpos'][i];d.qvel[:]=a['qvel'][i];d.ctrl[:]=a['q_target'][i]
  mujoco.mj_forward(model,d);force=np.zeros((4,3))
  for k,c in enumerate(d.contact):
   for leg,geom in enumerate(feet):
    if (c.geom1==geom and model.geom_bodyid[c.geom2]==0) or (c.geom2==geom and model.geom_bodyid[c.geom1]==0):
     local=np.zeros(6);mujoco.mj_contactForce(model,d,k,local)
     force[leg]+=c.frame.reshape(3,3).T@local[:3]*(1 if c.geom2==geom else -1)
  row[label]=dict(foot_Fz_N=force[:,2].tolist(),sum_Fz_N=float(force[:,2].sum()),
    floor_solimp=model.geom_solimp[model.geom('floor').id].tolist(),
    foot_solimp=model.geom_solimp[feet[0]].tolist())
 contact_records.append(row)
(JOB/'regression/contact-at-saved-state.json').write_text(json.dumps(dict(
 plant_integration_steps=0,scope='Fresh forward solves with identical saved qpos,qvel,ctrl;only the MakeDifferentiable solimp d0 change differs. These are model comparisons,not relabelled logged force samples.',
 records=contact_records),indent=2)+'\n')

replay=json.loads((JOB/'regression/r25_saved_state/result.json').read_text())
plans=[json.loads(x) for x in (JOB/'regression/r25_saved_state/planner.jsonl').read_text().splitlines()]
plan=min(plans,key=lambda r:abs(r['origin']-2.002))
sid=m.sensor('MovingSupport').id;adr=int(m.sensor_adr[sid]);dim=int(m.sensor_dim[sid])
qa=m.jnt_qposadr[m.actuator_trnid[:,0]];va=m.jnt_dofadr[m.actuator_trnid[:,0]]
def diff(x,y):
 d=np.zeros(2*m.nv+m.na)
 mujoco.mj_differentiatePos(m,d[:m.nv],1.,x[:m.nq],y[:m.nq])
 d[m.nv:2*m.nv]=y[m.nq:m.nq+m.nv]-x[m.nq:m.nq+m.nv]
 return d
records=[]
for initial_node in plan['nodes']:
 t=initial_node['t']
 if t>2.020001:continue
 i=int(np.argmin(abs(a['time']-t)))
 if abs(a['time'][i]-t)>1e-6:continue
 active_plan=max((x for x in plans if x['origin']<=t+1e-9),key=lambda x:x['origin'])
 node=min(active_plan['nodes'],key=lambda x:abs(x['t']-t))
 query_time=float(np.float32(t))
 ts=np.array([x['t'] for x in active_plan['nodes']])
 upper=int(np.clip(np.searchsorted(ts,query_time,side='right'),1,len(ts)-1));lower=upper-1
 blend=float(np.clip((query_time-ts[lower])/(ts[upper]-ts[lower]),0,1))
 def interpolated(key):
  return (1-blend)*np.asarray(active_plan['nodes'][lower][key])+blend*np.asarray(active_plan['nodes'][upper][key])
 K=interpolated('K').reshape(m.nu,2*m.nv+m.na)
 ref=interpolated('executed_reference_state');ref[3:7]/=np.linalg.norm(ref[3:7])
 nom=interpolated('executed_nominal_action')
 actual=np.r_[a['qpos'][i],a['qvel'][i]]
 dx=diff(ref,actual);du=plan['feedback_scaling']*(K@dx)
 target=np.clip(nom+du,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
 accepted=np.asarray(node['accepted_state'])
 accepted_action=np.clip(np.asarray(node['executed_nominal_action'])+active_plan['feedback_scaling']*(np.asarray(node['K']).reshape(m.nu,-1)@diff(np.asarray(node['executed_reference_state']),accepted)),
                         m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
 C=np.asarray(node['C']).reshape(plan['ns'],-1)[adr:adr+dim]
 D=np.asarray(node['D']).reshape(plan['ns'],-1)[adr:adr+dim]
 raw_tau=60*(target-a['qpos'][i,qa])-5*a['qvel'][i,va]
 row=dict(time_s=t,active_plan_origin_s=active_plan['origin'],query_time_float32=query_time,feedback_interpolation_fraction=blend,state_error=dx.tolist(),K=K.tolist(),K_delta_x=du.tolist(),
     unclipped_target=(nom+du).tolist(),clipped_target=target.tolist(),
     recorded_target=a['q_target'][i].tolist(),
     recorded_target_reconstruction_max_error=float(np.max(abs(target-a['q_target'][i]))),
     accepted_rollout_action_reconstruction_max_error=float(np.max(abs(accepted_action-node['accepted_action']))),
     max_feedback_correction_rad=float(np.max(abs(target-nom))),
     max_raw_K_delta_x_rad=float(np.max(abs(du))),max_abs_K=float(np.max(abs(K))),
     torque_saturated_joints=int(np.sum((raw_tau<m.actuator_forcerange[:,0])|(raw_tau>m.actuator_forcerange[:,1]))),
     moving_support_linearization_residual=node['linearization_residual'][adr:adr+dim],
     moving_support_C=C.tolist(),moving_support_D=D.tolist(),
     moving_support_C_max_abs=float(np.max(abs(C))),moving_support_D_max_abs=float(np.max(abs(D))))
 records.append(row)
phase=np.linspace(0,2*np.pi,100001);duty=.45;cadence=2.;amp=.005
steps=[]
for fp in (0,np.pi,np.pi,0):
 angle=((phase+np.pi-fp)%(2*np.pi)-np.pi)*.5/(1-duty)
 swing=(abs(angle)<np.pi/2)
 t=(angle+np.pi/2)/np.pi*(1-duty)/cadence
 rise=(1-duty)/cadence-.2
 rising=t<rise
 u=np.where(rising,t/rise,(t-rise)/.2);u=np.clip(u,0,1)
 smooth=u**3*(10+u*(-15+6*u))
 steps.append(np.where(swing,np.where(rising,amp*smooth,amp*(1-smooth)),0))
steps=np.asarray(steps);weights=np.exp(-steps/.001).sum(axis=0)
flight=np.all(steps>0,axis=0)
directions=[]
for t in (2.002,2.604,3.,5.,10.):
 i=int(np.argmin(abs(a['time']-t)))
 directions.append(dict(time_s=float(a['time'][i]),x=float(a['qpos'][i,0]),
   body_target_x=float(a['body_target'][i,0]),vx=float(a['qvel'][i,0]),
   commanded_vx=float(a['command_velocity'][i,0])))
stopped=json.loads((LARGE/'training/PANN_PPO_S5-seed0/stopped.json').read_text())
counts={}
for last in (141,142,144):
 reasons={}
 for ep in range(118,last+1):
  p=SMALL/f'train-PANN_PPO_S5-s0-ep{ep:05d}.json'
  if p.exists():
   r=json.loads(p.read_text());reason=r['reason'].split(':')[0]
   reasons[reason]=reasons.get(reason,0)+1
 counts[f'ep118_to_{last}']=reasons
result=dict(source_episode=source.name,replay_build='S5-r25-inspect',plant_integration_steps=0,
 replay_max_action_difference_from_original=max(x['action_difference_from_recorded'] for x in replay['observations']),
 first_trot_plan_origin_s=plan['origin'],winner=plan['winner'],
 accepted_cost=plan['accepted_cost'],policy_reference_cost=plan['executed_reference_cost'],
 action_consistency='Reconstruct nominal+K*StateDiff then original actuator target clipping. A policy reference need not itself be a free rollout.',
 onset=records,contact_model_comparison=json.loads((JOB/'regression/contact-at-saved-state.json').read_text()),
 support_schedule=dict(min_sum_exp=float(weights.min()),max_sum_exp=float(weights.max()),
   min_availability=float(np.minimum(weights/2,1).min()),max_availability=float(np.minimum(weights/2,1).max()),
   nominal_flight_fraction=float(flight.mean()),
   min_availability_during_flight=float(np.minimum(weights[flight]/2,1).min()),
   meaning='The claimed availability factor is always1 for this swing;it does not reduce requested support during nominal flight.'),
 direction=directions,failure_counts=counts,
 effective_translation_semantics='effective_translation_s and the historical effective sample flag count time/transitions in outbound or return phase;they do not establish positive route progress.',
 stopped=dict(samples=stopped['actual_transitions'],updates=stopped['updates'],tail=stopped['unupdated_fragment']),
 unused_repair_drafts=dict(status='not_built_not_applied_user_cancelled_repairs',
   files=['experiments/terrain_memory/s1-go2-r26-path-functions.inc','experiments/terrain_memory/s1-go2-r26-path-derivatives.inc']),
 root_cause_scope='Observed force-model inconsistency, onset amplification and ineffective phase availability are concrete defects/risks. No claim that lead initiated the regression or that C/D ordering is wrong. No repaired controller has been executed.')
(JOB/'regression/analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(stopped=result['stopped'],replay_error=result['replay_max_action_difference_from_original'],
 winner=plan['winner'],onset=[{k:x[k] for k in ('time_s','max_feedback_correction_rad','max_raw_K_delta_x_rad','max_abs_K','torque_saturated_joints','recorded_target_reconstruction_max_error','accepted_rollout_action_reconstruction_max_error','moving_support_C_max_abs','moving_support_D_max_abs')} for x in records],
 support_schedule=result['support_schedule'],failure_counts=counts)))
