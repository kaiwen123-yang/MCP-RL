#!/usr/bin/env python3
"""Timestamp-aligned contact windows from completed trajectories; no integration."""
import argparse
import csv
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-soft-landing-r10'


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
    out=ROOT/'outputs/terrain_memory/simulation'/RUN/a.attempt
    small=ROOT/'results/terrain_memory/simulation'/RUN
    result=json.loads((small/(a.attempt+'.json')).read_text())
    with np.load(out/'trajectory.npz') as archive:
        tr={key:archive[key] for key in archive.files}
    m=mujoco.MjModel.from_binary_path(str(out/'model.mjb'));d=mujoco.MjData(m)
    feet=[m.geom(n).id for n in ['FL','FR','RL','RR']]
    foot=np.concatenate([np.load(out/n)['foot'] for n in result['material']['chunks']])
    end=float(tr['time'][-1]);rows=[]
    for file in sorted(out.glob('prediction-*.npz')):
        with np.load(file) as archive:
            pr={key:archive[key] for key in archive.files}
        for i,origin in enumerate(pr['plan_origin']):
            if origin<end-.4:continue
            aligned={column:np.interp(pr['plan_times'][i],foot[:,0],foot[:,column])
                for column in [8,11,14,17,20,23,26,29,32,35,38,41]}
            for j,tm in enumerate(pr['plan_times'][i]):
                # Future beyond the real terminal state remains unobserved.
                if tm>end+1e-8:break
                state=pr['plan_states'][i,j]
                d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:m.nq+m.nv]
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d)
                for leg,geom in enumerate(feet):
                    spatial=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_GEOM,geom,spatial,0)
                    rows.append(dict(origin_s=float(origin),query_s=float(tm),horizon_s=float(tm-origin),leg=leg,
                        predicted_bottom_m=float(d.geom_xpos[geom,2]-m.geom_size[geom,0]),
                        predicted_vz_m_s=float(spatial[5]),
                        actual_bottom_m=float(aligned[8+3*leg][j]),
                        actual_vz_m_s=float(aligned[20+3*leg][j]),
                        actual_material_Fz_N=float(aligned[32+3*leg][j])))
    with (small/(a.attempt+'-contact-window.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    m.opt.timestep=.001;m.opt.integrator=3
    m.jnt_solimp[:,0]=0;m.geom_solimp[:,0]=0;m.pair_solimp[:,0]=0
    force_rows=[]
    for i in np.flatnonzero(tr['time']>=end-.4):
        d.qpos[:]=tr['qpos'][i];d.qvel[:]=tr['qvel'][i];d.ctrl[:]=tr['q_target'][i]
        d.time=tr['time'][i];mujoco.mj_forward(m,d)
        predicted=np.zeros(4)
        for k in range(d.ncon):
            con=d.contact[k];wrench=np.zeros(6);mujoco.mj_contactForce(m,d,k,wrench)
            for leg,geom in enumerate(feet):
                if con.geom2==geom and m.geom_bodyid[con.geom1]==0:
                    predicted[leg]+=(con.frame.reshape(3,3).T@wrench[:3])[2]
                elif con.geom1==geom and m.geom_bodyid[con.geom2]==0:
                    predicted[leg]-=(con.frame.reshape(3,3).T@wrench[:3])[2]
        for leg in range(4):
            force_rows.append(dict(time_s=float(d.time),leg=leg,
                native_prediction_Fz_N=float(predicted[leg]),
                material_Fz_N=float(np.interp(d.time,foot[:,0],foot[:,32+3*leg]))))
    with (small/(a.attempt+'-same-state-force.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(force_rows[0]));w.writeheader();w.writerows(force_rows)
    loaded=foot[:,32:42:3]>1
    stats=dict(attempt=a.attempt,simulation_s=end,reason=result['reason'],
        max_abs_strain=result['material']['max_abs_principal_strain'],
        peak_loaded_downward_m_s=float(np.max(np.where(loaded,-foot[:,20:30:3],0))),
        peak_attempted_material_Fz_N=float(foot[:,32:42:3].max()),
        final_boundary_bottom_mm=(1000*foot[-1,8:18:3]).tolist(),
        prediction_window_scope='Closed-loop plans vs actual at absolute time; feedback changes remain, not an identical-input counterfactual.',
        predicted_min_bottom_m=min(r['predicted_bottom_m'] for r in rows),
        native_force_error_on_material_load_N=max(abs(r['native_prediction_Fz_N']-r['material_Fz_N'])
            for r in force_rows if r['material_Fz_N']>1),
        max_bottom_error_m=max(abs(r['predicted_bottom_m']-r['actual_bottom_m']) for r in rows))
    print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
