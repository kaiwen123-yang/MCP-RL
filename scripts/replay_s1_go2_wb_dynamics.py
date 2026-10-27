#!/usr/bin/env python3
"""Compare explicit prediction dynamics under identical saved-plan target inputs."""
import argparse,csv,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'experiments/terrain_memory/s1-go2-wb-equilibrium.json'
VARIANTS=[('plant',None,None,False),('author',.01,0,True),
    ('native_coarse',.01,0,False),('native_fine_euler',.001,0,False),
    ('native_fine_implicitfast',.001,3,False)]


def evaluate(job):
    origin,state,targets,offsets,mocap_pos,mocap_quat,model_path=job
    records=[]; warnings=[]
    for name,dt,integrator,smooth in VARIANTS:
        m=mujoco.MjModel.from_binary_path(model_path)
        if dt is not None: m.opt.timestep=dt; m.opt.integrator=integrator
        if smooth:
            m.jnt_solimp[:,0]=0.;m.geom_solimp[:,0]=0.;m.pair_solimp[:,0]=0.
        d=mujoco.MjData(m); d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:m.nq+m.nv]
        if m.na:d.act[:]=state[m.nq+m.nv:]
        d.time=origin;d.mocap_pos[:]=mocap_pos;d.mocap_quat[:]=mocap_quat
        feet=[m.geom(n).id for n in ['FL','FR','RL','RR']]
        rows=[]
        steps=round(.01/m.opt.timestep)
        for k,offset in enumerate(offsets):
            d.ctrl[:]=targets[min(k,len(targets)-1)]
            mujoco.mj_forward(m,d)
            force=np.zeros((4,3))
            for z in range(d.ncon):
                con=d.contact[z];g1=int(con.geom1);g2=int(con.geom2)
                for j,foot in enumerate(feet):
                    if (g1==foot and m.geom_bodyid[g2]==0) or (g2==foot and m.geom_bodyid[g1]==0):
                        f=np.zeros(6);mujoco.mj_contactForce(m,d,z,f)
                        force[j]+=con.frame.reshape(3,3).T@f[:3]*(1 if foot==g2 else -1)
            rows.append(np.r_[d.qpos,d.qvel,d.actuator_force,force.ravel()])
            if k<len(offsets)-1:
                for _ in range(steps):mujoco.mj_step(m,d)
        records.append(rows);warnings.append([int(w.number) for w in d.warning])
    values=np.asarray(records);ref=values[0];rows=[]
    rr=Rotation.from_quat(ref[:,[4,5,6,3]])
    for j,(name,*_) in enumerate(VARIANTS[1:],1):
        vv=values[j];orientation=(rr.inv()*Rotation.from_quat(vv[:,[4,5,6,3]])).magnitude()
        for k,offset in enumerate(offsets):
            rows.append([origin,name,float(offset),float(np.linalg.norm(vv[k,:3]-ref[k,:3])),float(orientation[k]),
                float(np.linalg.norm(vv[k,19:22]-ref[k,19:22])),float(np.linalg.norm(vv[k,22:25]-ref[k,22:25])),
                float(np.linalg.norm(vv[k,37:49]-ref[k,37:49])),float(np.linalg.norm(vv[k,49:61]-ref[k,49:61]))])
    return origin,values,rows,warnings


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',default='tm-s1-go2-wb-startup-r6')
    p.add_argument('--attempt',default='g0-01');p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    c=json.loads(CFG.read_text());small=ROOT/c['outputs']['root'];small.mkdir(parents=True,exist_ok=True)
    out=ROOT/'outputs/terrain_memory/simulation'/c['run_id'];out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'outputs/terrain_memory/simulation'/args.source_run/args.attempt
    a=np.load(source/'trajectory.npz');origins=a['plan_origin'];times=a['plan_times'];states=a['plan_states'];actions=a['plan_actions']
    result=json.loads((ROOT/'results/terrain_memory/simulation'/args.source_run/(args.attempt+'.json')).read_text())
    indices=set(range(0,len(origins),25))
    indices.update(min(int(np.searchsorted(origins,x['time_s'])),len(origins)-1) for x in result['transitions'])
    jobs=[]; selected=[]
    for i in sorted(indices):
        horizon=times[i,-1]-times[i,0]
        if horizon<.34:continue
        offsets=np.arange(round(horizon/.01)+1)*.01
        action_indices=np.searchsorted(times[i]-times[i,0],offsets+1e-8,side='right')-1
        action_indices=np.minimum(action_indices,len(actions[i])-1)
        targets=actions[i,action_indices]
        j=min(int(np.searchsorted(a['time'],origins[i])),len(a['time'])-1)
        jobs.append((float(origins[i]),states[i,0],targets,offsets,a['mocap_pos'][j],a['mocap_quat'][j],str(source/'model.mjb')))
        selected.append(i)
    results=[]
    with ProcessPoolExecutor(args.workers) as pool:
        for value in pool.map(evaluate,jobs):results.append(value)
    name=args.source_run+'-'+args.attempt+'-matched-input'
    header=['origin_s','variant','horizon_s','position_error_m','orientation_error_rad','linear_velocity_error_m_s',
        'angular_velocity_error_rad_s','actuator_force_error_norm_Nm','foot_force_error_norm_N']
    rows=[row for value in results for row in value[2]]
    with (small/(name+'.csv')).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
    np.savez_compressed(out/(name+'.npz'),selected_plan_indices=selected,origins=np.asarray([v[0] for v in results]),
        variants=np.asarray([v[0] for v in VARIANTS]),states_forces=np.asarray([v[1] for v in results]),
        warnings=np.asarray([v[3] for v in results]),targets=np.asarray([v[2] for v in jobs]),offsets=jobs[0][3])
    summary=dict(source_run=args.source_run,attempt=args.attempt,origins=len(jobs),
        original_plan_dt_s=float(np.median(np.diff(times,axis=1))),
        common_input='Same saved-plan targets sampled at 0.01s and held identically on every replica; no feedback or replanning.',
        reference='Original actual plant under the same counterfactual input, not the historical future.',
        missing_state='qacc_warmstart and integration auxiliary state not recorded in R6. All replicas consistently reset auxiliary state, then restore saved qpos/qvel/act, clock and mocap. No bitwise history replay claim.',
        max_warning_counts=np.max(np.asarray([v[3] for v in results]),axis=0).tolist(),variants={})
    for variant,*_ in VARIANTS[1:]:
        tail=np.asarray([[float(x) for x in row[2:]] for row in rows if row[1]==variant and abs(row[2]-.35)<1e-7])
        summary['variants'][variant]={metric:dict(mean=float(tail[:,j].mean()),max=float(tail[:,j].max())) for j,metric in enumerate(header[3:],1)}
    (small/(name+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
