#!/usr/bin/env python3
"""Two actual r45 saved counterexamples for r46; no plant step or rollout."""
from pathlib import Path
from types import SimpleNamespace
import json,mujoco,numpy as np
from s1_w54_f_r46_backend import FController
ROOT=Path(__file__).resolve().parents[1]
ATTEMPT='job-20261117T044400Z-w54-f-a-r45-eval-PANN_PPO_S5-s1-E30-attempt01'
def main():
 old=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/ATTEMPT
 traj=np.load(old/'trajectory.npz')
 parts=[np.load(p) for p in sorted((old/'w54-f').glob('control-*.npz'))]
 ctrl={k:np.concatenate([z[k] for z in parts]) for k in parts[0].files}
 m=mujoco.MjModel.from_binary_path(str(old/'model.mjb'))
 legs=('FL','FR','RL','RR')
 feet=np.array([m.body_geomadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,l+'_foot')] for l in legs])
 base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'trunk')
 rows=[]
 for ti in (0.,3.6):
  d=mujoco.MjData(m);d.qpos[:]=traj['qpos'][0];d.qvel[:]=traj['qvel'][0];mujoco.mj_forward(m,d)
  out=ROOT/'work/w54-f'/('r46-saved-'+str(ti));out.mkdir(exist_ok=False)
  c=FController(m,d,feet,base,out)
  i=int(np.argmin(abs(ctrl['time']-ti)));j=int(np.argmin(abs(traj['time']-ti)))
  d.qpos[:]=traj['qpos'][j];d.qvel[:]=traj['qvel'][j];d.ctrl[:]=traj['q_target'][j];d.time=float(traj['time'][j]);mujoco.mj_forward(m,d)
  mission=SimpleNamespace(phase='initial_stand' if ti==0 else 'outbound',
   v=traj['command_velocity'][j],w=traj['command_omega'][j],task={'return_xy_m':[0,0],'turnaround_xy_m':[3,0]})
  c.event_mode=ctrl['event_mode'][i].copy();c.confirmed=ctrl['confirmed_contact'][i].copy()
  c.load_bearing=ctrl['load_bearing_contact'][i].copy();c.anchor=ctrl['stance_anchor'][i].copy()
  c.measured=ctrl['observed_contact'][i].copy();c.moving=ti>2
  c.last_force=ctrl['force'][max(0,i-1)].copy()
  for leg in range(4):
   if c.event_mode[leg]=='search':
    c.swing[leg]=dict(start=3.324,end=3.524,apex=3.424,landing=ctrl['feet_position_reference'][i,leg].copy())
  c.model_data_root=d.xpos[base].copy();c.control_time=d.time
  mq=c.model_quantities(d);c.current_mq=mq;c.current_mission=mission
  c.replan(d.time,d,mq,mission)
  action=c.wbc(d.time,d,mq,mission,c.event_mode!='swing',c.measured,ctrl['measured_grf'][i],np.zeros(4,bool),{})
  r=c.rows[-1]
  rows.append(dict(saved_time_s=float(ctrl['time'][i]),measured_Fz=traj['actual_grf'][j,:,2].tolist(),
   confirmation=c.confirmed.tolist(),r45_active=ctrl['active_wbc_contact'][i].tolist(),r46_active=r['active_wbc_contact'].tolist(),
   r45_body_acceleration=ctrl['body_acceleration_reference'][i].tolist(),r46_body_acceleration=r['body_acceleration_reference'].tolist(),
   r45_joint_acceleration_peak=float(abs(ctrl['qacc'][i,c.va]).max()),r46_joint_acceleration_peak=float(abs(r['qacc'][c.va]).max()),
   r46_foot_mode=r['mode'].tolist(),
   max_torque_violation_Nm=float(max(np.max(r['torque_lower']-r['tau_total']),np.max(r['tau_total']-r['torque_upper']),0)),
   max_mapped_input_violation_rad=float(max(np.max(m.actuator_ctrlrange[:,0]-action),np.max(action-m.actuator_ctrlrange[:,1]),0)),
   pd_inverse_error_Nm=float(abs(r['requested_pd']-r['tau_total']).max()),
   floating_residual=float(abs(r['dynamics_residual']).max())))
  c.finish(d.time,'r46_saved_counterexample_checked')
 ans=dict(schema='w54_f_r46_saved_counterexamples_v1',source_attempt=ATTEMPT,rows=rows,
  new_physics_steps=0,new_rollout=0,new_FD=0,SRB_solves=2,WBC_solves=2,
  scope='same saved actual states; current-contract changes only, not alternate physical trajectories or qualification')
 path=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w54/f-r46-saved-counterexamples.json'
 path.write_text(json.dumps(ans,indent=2)+'\n');print(json.dumps(ans,indent=2))
if __name__=='__main__':main()
