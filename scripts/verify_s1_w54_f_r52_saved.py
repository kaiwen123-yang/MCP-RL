#!/usr/bin/env python3
"""Re-solve one actual r51 horizon/current action with the inner friction square."""
from pathlib import Path
from types import SimpleNamespace
import json,mujoco,numpy as np
from s1_w54_f_r52_backend import FController,assemble_centroidal,assemble_force_horizon
ROOT=Path(__file__).resolve().parents[1]
ATTEMPT='job-20261117T053419Z-w54-f-a-r51-eval-PANN_PPO_S5-s1-E30-attempt01'
def mission(z,i):
 return SimpleNamespace(phase='outbound',v=z['command_velocity'][i],w=z['command_omega'][i],
  task={'return_xy_m':[0,0],'turnaround_xy_m':[3,0]})
def main():
 old=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/ATTEMPT
 z=np.load(old/'trajectory.npz');parts=[np.load(p) for p in sorted((old/'w54-f').glob('control-*.npz'))]
 ctrl={k:np.concatenate([a[k] for a in parts]) for k in parts[0].files}
 m=mujoco.MjModel.from_binary_path(str(old/'model.mjb'));d=mujoco.MjData(m);legs=('FL','FR','RL','RR')
 feet=np.array([m.body_geomadr[m.body(l+'_foot').id] for l in legs]);base=m.body('trunk').id
 d.qpos[:]=z['qpos'][0];d.qvel[:]=z['qvel'][0];mujoco.mj_forward(m,d)
 out=ROOT/'work/w54-f/r52-saved-cone';out.mkdir(exist_ok=False);c=FController(m,d,feet,base,out)
 i=int(np.argmin(abs(ctrl['time']-4.884)));j=int(np.argmin(abs(z['time']-ctrl['time'][i])))
 events=[json.loads(line) for line in (old/'w54-f/contact-events.jsonl').read_text().splitlines()]
 for leg in range(4):
  if ctrl['event_mode'][i,leg]=='swing':
   e=[e for e in events if e['kind']=='liftoff_committed' and legs[leg] in e['legs'] and e['time_s']<=ctrl['time'][i]][-1]
   k=int(np.argmin(abs(z['time']-e['time_s'])))
   d.qpos[:]=z['qpos'][k];d.qvel[:]=z['qvel'][k];d.time=float(z['time'][k]);mujoco.mj_forward(m,d)
   mq=c.model_quantities(d);c._start_swing(d.time,leg,d,mq,mission(z,k),np.zeros(4,bool),{})
 d.qpos[:]=z['qpos'][j];d.qvel[:]=z['qvel'][j];d.ctrl[:]=z['q_target'][j];d.time=float(z['time'][j]);mujoco.mj_forward(m,d)
 c.event_mode=ctrl['event_mode'][i].copy();c.confirmed=ctrl['confirmed_contact'][i].copy();c.load_bearing=ctrl['load_bearing_contact'][i].copy()
 c.anchor=ctrl['stance_anchor'][i].copy();c.measured=ctrl['observed_contact'][i].copy();c.moving=True;c.last_force=ctrl['force'][i-1].copy()
 c.next_pair=np.array([k for k in range(4) if k not in ctrl['transfer_support_pair'][i]])
 c.model_data_root=d.xpos[base].copy();c.control_time=d.time;mq=c.model_quantities(d)
 call=int(ctrl['plan_call'][i]);a=np.load(old/'w54-f'/f'mpc-{call:06d}.npz')
 inputs={key:a[key] for key in ('stage_A','stage_B','stage_c','x0','reference','last_force','contact','feet','force_normal_upper')}
 qp=assemble_force_horizon(inputs,c.cfg);sol,stat=c._solve(qp['P'],qp['q'],qp['A'],qp['lower'],qp['upper'],'saved-inner-mpc',d.time)
 n=len(a['contact']);nx=(n+1)*12
 blocks=assemble_centroidal(c.mass,mq['inertia'],mq['com'],mq['contact_force_point'],mq['rpy'],mq['omega'],friction=c.cfg['force_pyramid_slope'])
 c.plan_count=call;c.last_plan=dict(time=float(a['times'][0]),times=a['times'],state=sol[:nx].reshape(n+1,12),
  reference=a['reference'],forces=sol[nx:].reshape(n,4,3),feet=a['foot_center_plan'],contact=a['contact'],blocks=blocks)
 ref_error=[]
 for leg in range(4):
  if c.event_mode[leg]=='swing':
   p,v,acc=c._swing_reference(d.time,leg);ref_error.append(float(max(abs(p-ctrl['feet_position_reference'][i,leg]).max(),abs(v-ctrl['feet_velocity_reference'][i,leg]).max())))
 action=c.wbc(d.time,d,mq,mission(z,j),c.event_mode!='swing',c.measured,ctrl['measured_grf'][i],np.zeros(4,bool),{})
 r=c.rows[-1];newf=c.last_plan['forces'];mu=.5;slope=c.cfg['force_pyramid_slope']
 def violation(f):return float(np.maximum(np.linalg.norm(f[...,:2],axis=-1)-mu*f[...,2],0).max())
 ans=dict(schema='w54_f_r52_same_input_inner_cone_v1',source_attempt=ATTEMPT,time_s=d.time,source_mpc_call=call,
 physical_mu=mu,planning_axis_slope=slope,physical_cone=int(m.opt.cone),
 original_MPC_physical_cone_violation_N=violation(a['forces']),new_MPC_physical_cone_violation_N=violation(newf),
 original_WBC_physical_cone_violation_N=violation(ctrl['force'][i]),new_WBC_physical_cone_violation_N=violation(r['force']),
 original_RL_force=ctrl['force'][i,2].tolist(),new_RL_force=r['force'][2].tolist(),
 reconstructed_swing_reference_max_error=max(ref_error,default=0),
 max_torque_violation_Nm=float(max(np.max(r['torque_lower']-r['tau_total']),np.max(r['tau_total']-r['torque_upper']),0)),
 max_input_violation_rad=float(max(np.max(m.actuator_ctrlrange[:,0]-action),np.max(action-m.actuator_ctrlrange[:,1]),0)),
 max_predicted_substep_torque_violation_Nm=float(max(np.max(m.actuator_forcerange[:,0]-r['predicted_hold_torque']),np.max(r['predicted_hold_torque']-m.actuator_forcerange[:,1]),0)),
 MPC_status=stat,new_SRB_solves=1,new_WBC_solves=1,new_physics_steps=0,new_rollouts=0,new_native_FD=0,
 qualification='same saved mathematical/actuator contract only; no original E30 qualification')
 c.finish(d.time,'r52_saved_inner_cone_checked')
 p=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w54/f-r52-saved-cone.json';p.write_text(json.dumps(ans,indent=2)+'\n')
 print(json.dumps(ans,indent=2))
if __name__=='__main__':main()
