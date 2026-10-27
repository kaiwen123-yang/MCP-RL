#!/usr/bin/env python3
"""R8 source-rate input and support calculations from retained R7 arrays."""
import csv,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from run_s1_go2_wb_baseline import dump,index_file,contact_state
ROOT=Path(__file__).resolve().parents[1]
CONFIG='experiments/terrain_memory/s1-go2-wb-support.json'
OLD='tm-s1-go2-wb-equilibrium-r7'

def replay(job):
    name,origin,state,actions,times,actual,mp,mq,path=job
    rows=[]
    # 0.348s is the common endpoint divisible by 1ms and 2ms inside R7's 0.349s span.
    for schedule in ['plan_1ms','plan_2ms_hold','feedback_2ms_hold','old_plan_10ms_hold']:
        values=[]
        for variant in ['plant','prediction']:
            m=mujoco.MjModel.from_binary_path(path)
            if variant=='prediction':
                m.opt.timestep=.001;m.opt.integrator=3
                if name.startswith('g1'):
                    m.jnt_solimp[:,0]=0;m.geom_solimp[:,0]=0;m.pair_solimp[:,0]=0
            # Author's native plant is 2ms: use its existing integrator at 1ms only
            # for the explicitly counterfactual 1ms-action schedule.
            if schedule=='plan_1ms' and m.opt.timestep>.001:m.opt.timestep=.001
            d=mujoco.MjData(m);d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:m.nq+m.nv]
            d.time=origin;d.mocap_pos[:]=mp;d.mocap_quat[:]=mq
            hold=.001 if schedule=='plan_1ms' else .01 if schedule=='old_plan_10ms_hold' else .002
            n=int(round(.348/m.opt.timestep));next_update=-1.; held=None
            for k in range(n):
                elapsed=k*m.opt.timestep
                if elapsed>=next_update-1e-9:
                    query=np.floor((elapsed+1e-9)/hold)*hold
                    if schedule=='feedback_2ms_hold': held=actual[min(round(query/.002),len(actual)-1)]
                    else: held=np.array([np.interp(origin+query,times[:len(actions)],actions[:,j]) for j in range(m.nu)])
                    next_update=query+hold
                d.ctrl[:]=held;mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
            values.append(np.r_[d.qpos,d.qvel])
        p,v=values
        angle=(Rotation.from_quat(p[[4,5,6,3]]).inv()*Rotation.from_quat(v[[4,5,6,3]])).magnitude()
        rows.append(dict(attempt=name,origin_s=origin,schedule=schedule,horizon_s=.348,
            position_difference_m=float(np.linalg.norm(p[:3]-v[:3])),orientation_difference_rad=float(angle),
            velocity_difference_m_s=float(np.linalg.norm(p[19:22]-v[19:22])),
            omega_difference_rad_s=float(np.linalg.norm(p[22:25]-v[22:25]))))
    return rows

def main():
    c=json.loads((ROOT/CONFIG).read_text());small=ROOT/c['outputs']['root'];small.mkdir(parents=True,exist_ok=True)
    all_rows=[];summary={};jobs=[]
    for name in ['g0-n-01','g1-nsh-nominal-03','g1-osh-nominal-04','g1-oshw-nominal-05']:
        source=ROOT/'outputs/terrain_memory/simulation'/OLD/name
        with np.load(source/'trajectory.npz') as z:
            a={k:z[k] for k in ['time','qpos','qvel','q_target','torque','rpy','contact','feet','physical','mocap_pos','mocap_quat','plan_origin','plan_times','plan_actions','plan_states']}
        m=mujoco.MjModel.from_binary_path(str(source/'model.mjb'));base=m.body('trunk').id
        joint=m.actuator_trnid[:,0];qa=m.jnt_qposadr[joint];va=m.jnt_dofadr[joint]
        request=60*(a['q_target']-a['qpos'][:,qa])-5*a['qvel'][:,va]
        lim=np.array([23.7,23.7,45.43]*4)
        saturation=(np.abs(request)>=lim-1e-6)
        rot=Rotation.from_quat(a['qpos'][:,[4,5,6,3]])
        comz=a['qpos'][:,2]+rot.apply(np.broadcast_to(m.body_ipos[base],(len(a['time']),3)).copy())[:,2]
        rel=comz-a['feet'][:,:,2].mean(axis=1)-.27
        ph=a['physical'];start=34 if name=='g1-osh-nominal-04' else 0
        for left in np.arange(start,min(45 if start else 60,a['time'][-1]),1):
            mask=(a['time']>=left)&(a['time']<left+1);pm=(ph[:,0]>=left)&(ph[:,0]<left+1)
            if not mask.any() or not pm.any():continue
            row=dict(attempt=name,start_s=float(left),end_s=float(min(left+1,a['time'][-1])),
                base_z_mean_m=float(a['qpos'][mask,2].mean()),base_z_min_m=float(a['qpos'][mask,2].min()),
                com_z_mean_m=float(comz[mask].mean()),relative_height_residual_mean_m=float(rel[mask].mean()),
                all_feet_contact_fraction=float(np.all(ph[pm,26:30]>0,axis=1).mean()),
                foot_Fz_mean_N=float(ph[pm,32:42:3].sum(axis=1).mean()),
                nonfoot_Fz_mean_N=float(ph[pm,47].mean()) if ph.shape[1]==49 else None,
                max_roll_pitch_rad=float(np.max(np.abs(a['rpy'][mask,:2]))),
                omega_rms_rad_s=float(np.sqrt(np.mean(a['qvel'][mask,3:6]**2))))
            for j in range(12):row['sat_fraction_joint_'+str(j)]=float(saturation[mask,j].mean())
            for j in range(4):
                row['foot_z_mean_'+str(j)]=float(a['feet'][mask,j,2].mean())
                row['Fz_mean_'+str(j)]=float(ph[pm,32+3*j].mean())
            all_rows.append(row)
        # Same nominal policy queried on its true 1ms grid and projected to 2ms holds.
        actions=a['plan_actions'];diff=actions[:,1:-1:2]-actions[:,:-2:2]
        per=[]
        for j in range(12):
            idx=np.flatnonzero(saturation[:,j])
            per.append(dict(joint=m.joint(int(joint[j])).name,project_limit_Nm=float(lim[j]),
                actual_limited=bool(m.actuator_forcelimited[j]),first_request_at_limit_s=float(a['time'][idx[0]]) if len(idx) else None,
                request_at_limit_samples=int(len(idx)),sample_duration_s=float(len(idx)*.002),
                peak_request_Nm=float(abs(request[:,j]).max()),peak_actual_Nm=float(abs(a['torque'][:,j]).max())))
        summary[name]=dict(plan_nodes=int(actions.shape[1]),plan_dt_s=float(np.median(np.diff(a['plan_times']))),
            odd_node_projection_rms_rad=float(np.sqrt(np.mean(diff**2))),odd_node_projection_max_rad=float(abs(diff).max()),
            joints=per,foot_radius_m=[float(m.geom_size[m.geom(n).id,0]) for n in ['FL','FR','RL','RR']])
        desired=[0.,1.,2.,10.,20.,34.,36.,38.,39.6,41.,44.,50.]
        for t in desired:
            i=int(np.searchsorted(a['plan_origin'],t));j=int(np.searchsorted(a['time'],t))
            if i>=len(actions) or j+175>=len(a['time']):continue
            origin=float(a['plan_origin'][i]);j=int(np.searchsorted(a['time'],origin-1e-8))
            jobs.append((name,origin,a['plan_states'][i,0],actions[i],a['plan_times'][i],
                a['q_target'][j:j+175],a['mocap_pos'][j],a['mocap_quat'][j],str(source/'model.mjb')))
    with ProcessPoolExecutor(2) as pool: replay_rows=[row for rows in pool.map(replay,jobs) for row in rows]
    for filename,rows in [('existing-support-windows.csv',all_rows),('existing-input-schedules.csv',replay_rows)]:
        with (small/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with index_file(small,c) as idx:
        idx['existing_data_calculations']=dict(source_run=OLD,attempts=summary,replay_cases=len(replay_rows),
            tables=['existing-support-windows.csv','existing-input-schedules.csv'],new_physical_starts=0,
            semantics='1ms saved plan; 2ms sampled linear nominal policy held; recorded 500Hz feedback held; old10ms counterfactual, each input identical across compared plant/predictor.',
            limitations='R7 omitted qacc_warmstart/integration state; replicas reset auxiliaries equally, no bitwise replay. Author1ms schedule also uses1ms integration for its otherwise2ms plant. Saturation is request reaching project limits; author actuators remain unbounded. Feedback-minus-nominal includes Clamp, not raw K feedback.')
    print(json.dumps(dict(attempts=summary,replay_cases=len(replay_rows))),flush=True)
if __name__=='__main__':main()
