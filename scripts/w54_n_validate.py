#!/usr/bin/env python3
"""Finite same-saved-input E/2ms/FD validation; never starts a plant episode."""
import json,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from s1_go2_w54_n_execution import PrecontactE
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'work/w54-n/validation'
OLD=Path('/root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261117T023608Z-w53-n-a-eval-PANN_PPO_S5-s1-E30-attempt02')
def main():
 OUT.mkdir(exist_ok=True)
 deps=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2/build-autodl/_deps'
 exe=OUT/'same-input'
 command=['g++','-O2','-std=c++17','-I'+str(ROOT/'work/w54-n/source'),'-I'+str(deps/'mujoco-src/include'),'-I'+str(deps/'abseil-cpp-src'),
  str(ROOT/'scripts/w54_n_same_input.cc'),'-L'+str(ROOT/'work/r11-frozen/S5-r44/lib'),'-Wl,-rpath,'+str(ROOT/'work/r11-frozen/S5-r44/lib'),'-l:libmujoco.so.3.2.3','-o',str(exe)]
 if not exe.exists():subprocess.run(command,check=True)
 tree=ET.parse(OLD/'model/G1-project-go2.xml');ET.SubElement(tree.getroot().find('custom'),'numeric',name='w54_precontact_E',data='1')
 xml=OUT/'saved-model.xml';tree.write(xml,encoding='unicode')
 m=mujoco.MjModel.from_xml_path(str(xml));m.opt.timestep=.001;m.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_WARMSTART
 m.jnt_solimp[:,0]=0;m.geom_solimp[:,0]=0;m.pair_solimp[:,0]=0
 E=PrecontactE(m);z=np.load(OLD/'trajectory.npz');rows=[]
 for time_s in (5.990,6.020,6.440,6.460):
  i=int(np.argmin(abs(z['time']-time_s)));d=mujoco.MjData(m)
  d.time=float(z['time'][i]);d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['q_target'][i]
  d.mocap_pos[:]=z['mocap_pos'][i];d.mocap_quat[:]=z['mocap_quat'][i];d.userdata[:]=z['support_reference_userdata'][i]
  raw=d.ctrl.copy();original_q=d.qpos.copy();original_v=d.qvel.copy()
  inp=OUT/f'input-{i}.bin';np.concatenate([[d.time],d.qpos,d.qvel,d.ctrl,d.mocap_pos.ravel(),d.mocap_quat.ravel(),d.userdata]).tofile(inp)
  out=OUT/f'output-{i}.bin'
  if not out.exists():subprocess.run([str(exe),str(xml),str(inp),str(out)],check=True)
  a=np.fromfile(out);native_u=a[:12];native_diag=a[12:44];nextq=a[44:63];nextv=a[63:81];A=a[81:81+1296].reshape(36,36);B=a[81+1296:].reshape(36,12)
  applied,diag=E.apply(d,raw);d.ctrl[:]=applied
  mujoco.mj_step(m,d);mujoco.mj_step(m,d)
  baseline_q=d.qpos.copy();baseline_v=d.qvel.copy()
  state_error=max(np.max(abs(d.qpos-nextq)),np.max(abs(d.qvel-nextv)))
  d.qpos[:]=original_q;d.qvel[:]=original_v;d.time=float(z['time'][i])
  direction=np.sin(np.arange(36)+1);direction/=np.linalg.norm(direction)
  du=np.cos(np.arange(12)+1);du/=np.linalg.norm(du)
  eps=1e-6;mujoco.mj_integratePos(m,d.qpos,direction[:18],eps);d.qvel[:]+=eps*direction[18:]
  perturbed,_=E.apply(d,raw+eps*du);d.ctrl[:]=perturbed;mujoco.mj_step(m,d);mujoco.mj_step(m,d)
  deltaq=np.empty(18);mujoco.mj_differentiatePos(m,deltaq,eps,baseline_q,d.qpos)
  observed=np.r_[deltaq,(d.qvel-baseline_v)/eps];linear=A@direction+B@du
  rows.append(dict(source_time_s=float(z['time'][i]),raw_to_E_max_rad=float(np.max(abs(raw-applied))),
    E_python_native_max_rad=float(np.max(abs(applied-native_u))),
    E_diagnostic_python_native_max=float(np.max(abs(diag.ravel()-native_diag))),
    held_transition_python_native_max=float(state_error),
    composite_fd_direction_max=float(np.max(abs(observed-linear))),
    composite_fd_direction_relative=float(np.linalg.norm(observed-linear)/max(np.linalg.norm(observed),1e-12)),
    FL_gap_m=float(diag[0,0]),FL_vz_m_s=float(diag[0,1]),FL_lead_m=float(diag[0,3]),
    FL_requested_acceleration_m_s2=float(diag[0,5]),
    E_applied_torque_max_Nm=float(np.max(abs(60*(applied-original_q[7:])-5*original_v[6:])))))
 result=dict(status='completed',source=str(OLD),rows=rows,physical_plant_steps=0,new_PPO_samples=0,new_PPO_updates=0,donor=0,
  native_composite_FD_evaluations=4*49,native_predictor_substeps=4*(49*2+2),python_predictor_substeps=4*4,
  check_scope='same saved input: two independent M/J assembly paths, shared E core, exact 2ms hold, independent mixed-direction FD check',
  caveat='rigid contact predictor with original R8 differentiable settings; no actual TM02 safety or original-task qualification')
 path=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w54/n-same-input.json';path.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
