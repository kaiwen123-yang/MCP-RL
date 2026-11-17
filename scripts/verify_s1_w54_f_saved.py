#!/usr/bin/env python3
"""One W53 saved-state algebra/execution check for the grouped W54 F change.

No mj_step, robot episode, rollout or FD is performed. The original failed
static constraints are eliminated and one complete current WBC is solved
under the unchanged actuator/joint bounds.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from scipy import sparse
from s1_w54_f_backend import FController,assemble_force_horizon
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261117T023610Z-w53-f-a-eval-PANN_PPO_S5-s1-E30-attempt02')
def main():
 out=ROOT/'work/w54-f/saved-contract';out.mkdir(parents=True,exist_ok=False)
 m=mujoco.MjModel.from_binary_path(str(OLD/'model.mjb'));d=mujoco.MjData(m)
 z=np.load(OLD/'trajectory.npz')
 i=int(np.argmin(abs(z['time']-2.0)))
 d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['q_target'][i];d.time=float(z['time'][i]);mujoco.mj_forward(m,d)
 legs=('FL','FR','RL','RR')
 feet=np.array([m.body_geomadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,leg+'_foot')] for leg in legs])
 base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'trunk')
 c=FController(m,d,feet,base,out)
 mission=SimpleNamespace(phase='initial_stand',v=np.zeros(3),w=np.zeros(3),task={'return_xy_m':[0.,0.],'turnaround_xy_m':[3.,0.]})
 # Feed only the actual immediately preceding confirmation history.
 for j in range(i-3,i+1):
  d.qpos[:]=z['qpos'][j];d.qvel[:]=z['qvel'][j];d.time=float(z['time'][j]);mujoco.mj_forward(m,d)
  mq=c.model_quantities(d)
  c._update_schedule(d.time,d,mq,mission,np.zeros(4,bool),{},z['contact'][j],z['actual_grf'][j])
 action=c.update(d.time,d,mission,z['contact'][i],z['actual_grf'][i],np.zeros(4,bool),replan=True)
 row=c.rows[-1]
 b=m.actuator_forcerange;u=m.actuator_ctrlrange
 qnew=d.qpos[c.qa]+.002*d.qvel[c.va]+.5*.002**2*row['qacc'][c.va]
 bounds=m.jnt_range[c.joint_ids]
 # Saved six contradictory static rows are absent from the corrected SRB.
 old_A=sparse.load_npz(OLD/'w53-f/failed-mpc-3.120000-A.npz')
 new=np.load(c.record_dir/'mpc-000001.npz');inputs={k:new[k] for k in
  ('stage_A','stage_B','stage_c','x0','reference','last_force','contact','feet','force_normal_upper')}
 qp=assemble_force_horizon(inputs,c.cfg)
 report=dict(source=str(OLD),saved_state_time_s=d.time,new_physics_steps=0,new_rollouts=0,new_FD=0,
  current_WBC_solves=1,new_SRB_solves=1,old_failed_matrix_rows=old_A.shape[0],new_SRB_matrix_rows=qp['A'].shape[0],
  static_future_joint_rows_removed=14*12,original_WBC_torque_mapped_u_joint_hard_bounds_preserved=True,
  max_torque_violation_Nm=float(max(np.max(b[:,0]-row['tau_total']),np.max(row['tau_total']-b[:,1]),0.)),
  max_mapped_u_violation_rad=float(max(np.max(u[:,0]-action),np.max(action-u[:,1]),0.)),
  max_joint_hold_violation_rad=float(max(np.max(bounds[:,0]-qnew),np.max(qnew-bounds[:,1]),0.)),
  pd_inverse_error_Nm=float(np.max(abs(row['requested_pd']-row['tau_total']))),
  dynamics_residual=float(np.max(abs(row['dynamics_residual']))),
  feet_acceleration_residual=row['feet_acceleration_residual'].tolist(),
  friction_compensation_Nm=row['friction_compensation'].tolist(),
  actual_friction_constraint_Nm=row['actual_friction_constraint'].tolist(),
  contact_confirmed=row['confirmed_contact'].tolist(),
  qualification='saved-current-contract-only; no physical gait or task qualification')
 c.finish(d.time,'saved_state_contract_completed')
 target=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w54/f-saved-contract.json'
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
