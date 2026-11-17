#!/usr/bin/env python3
"""One original r50 stalled MPC/WBC input, measured-contact preload; no new SRB/physics."""
from pathlib import Path
from types import SimpleNamespace
import json,mujoco,numpy as np
from s1_w54_f_r51_backend import FController,assemble_centroidal
ROOT=Path(__file__).resolve().parents[1]
ATTEMPT='job-20261117T051942Z-w54-f-a-r50-eval-PANN_PPO_S5-s1-E30-attempt01'
def main():
 old=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/ATTEMPT
 traj=np.load(old/'trajectory.npz');parts=[np.load(p) for p in sorted((old/'w54-f').glob('control-*.npz'))]
 ctrl={k:np.concatenate([z[k] for z in parts]) for k in parts[0].files}
 m=mujoco.MjModel.from_binary_path(str(old/'model.mjb'));d=mujoco.MjData(m)
 legs=('FL','FR','RL','RR')
 feet=np.array([m.body_geomadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,l+'_foot')] for l in legs])
 base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'trunk')
 d.qpos[:]=traj['qpos'][0];d.qvel[:]=traj['qvel'][0];mujoco.mj_forward(m,d)
 out=ROOT/'work/w54-f/r51-saved-preload';out.mkdir(exist_ok=False);c=FController(m,d,feet,base,out)
 i=int(np.argmin(abs(ctrl['time']-15.0)));j=int(np.argmin(abs(traj['time']-15.0)))
 d.qpos[:]=traj['qpos'][j];d.qvel[:]=traj['qvel'][j];d.ctrl[:]=traj['q_target'][j];d.time=float(traj['time'][j]);mujoco.mj_forward(m,d)
 mission=SimpleNamespace(phase='outbound',v=traj['command_velocity'][j],w=traj['command_omega'][j],task={'return_xy_m':[0,0],'turnaround_xy_m':[3,0]})
 c.event_mode=ctrl['event_mode'][i].copy();c.confirmed=ctrl['confirmed_contact'][i].copy()
 c.load_bearing=ctrl['load_bearing_contact'][i].copy();c.anchor=ctrl['stance_anchor'][i].copy()
 c.measured=ctrl['observed_contact'][i].copy();c.moving=True;c.last_force=ctrl['force'][i-1].copy()
 c.next_pair=np.array([j for j in range(4) if j not in ctrl['transfer_support_pair'][i]])
 for leg in range(4):
  if c.event_mode[leg]=='search':c.swing[leg]=dict(landing=ctrl['feet_position_reference'][i,leg].copy())
 c.model_data_root=d.xpos[base].copy();c.control_time=d.time;mq=c.model_quantities(d)
 k=int(ctrl['plan_call'][i]);a=np.load(old/'w54-f'/f'mpc-{k:06d}.npz')
 blocks=assemble_centroidal(c.mass,mq['inertia'],mq['com'],mq['contact_force_point'],mq['rpy'],mq['omega'])
 c.plan_count=k;c.last_plan=dict(time=float(a['times'][0]),times=a['times'],state=a['states'],
  reference=a['reference'],forces=a['forces'],feet=a['foot_center_plan'],contact=a['contact'],blocks=blocks)
 action=c.wbc(d.time,d,mq,mission,c.event_mode!='swing',c.measured,ctrl['measured_grf'][i],np.zeros(4,bool),{})
 r=c.rows[-1];exact=(a['states'][1,9:12]-a['states'][0,9:12])/.025
 ans=dict(schema='w54_f_r51_measured_touch_preload_v1',source_attempt=ATTEMPT,physical_state_time_s=d.time,original_mpc_call=k,
  original_mpc_motion_acceleration=exact.tolist(),r50_executed_body_target=ctrl['body_acceleration_reference'][i].tolist(),
  r51_executed_body_target=r['body_acceleration_reference'].tolist(),
  r51_MPC_feedforward=r['mpc_body_acceleration_feedforward'].tolist(),
  feedforward_error=float(abs(exact-r['mpc_body_acceleration_feedforward']).max()),
  r50_body_task_residual=ctrl['body_acceleration_residual'][i].tolist(),r51_body_task_residual=r['body_acceleration_residual'].tolist(),
  max_torque_violation_Nm=float(max(np.max(r['torque_lower']-r['tau_total']),np.max(r['tau_total']-r['torque_upper']),0)),
  max_mapped_u_violation_rad=float(max(np.max(m.actuator_ctrlrange[:,0]-action),np.max(action-m.actuator_ctrlrange[:,1]),0)),
  floating_residual=float(abs(r['dynamics_residual']).max()),
  original_measured_Fz_N=ctrl['measured_grf'][i,:,2].tolist(),
  original_solved_Fz_N=ctrl['force'][i,:,2].tolist(),new_solved_Fz_N=r['force'][:,2].tolist(),
  preload_requested=r['measured_touch_preload'].tolist(),preload_normal_lower_N=r['preload_normal_lower'].tolist(),
  contact_confirmation_threshold_N=c.cfg['contact_events']['touch_on_N'],
  confirmation_unchanged=c.confirmed.tolist(),bearing_unchanged=c.load_bearing.tolist(),
  mean_of10_samples_error_Nm=float(abs(r['hold_torque_mean_error']).max()),
  max_predicted_substep_torque_violation_Nm=float(max(np.max(m.actuator_forcerange[:,0]-r['predicted_hold_torque']),np.max(r['predicted_hold_torque']-m.actuator_forcerange[:,1]),0)),
  new_physics_steps=0,new_rollouts=0,new_FD=0,new_SRB_solves=0,new_WBC_solves=1,
  qualification='same saved current execution check only; no original-stage qualification')
 c.finish(d.time,'r51_saved_preload_checked')
 p=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w54/f-r51-saved-preload.json'
 p.write_text(json.dumps(ans,indent=2)+'\n');print(json.dumps(ans,indent=2))
if __name__=='__main__':main()
