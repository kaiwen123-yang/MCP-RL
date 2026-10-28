"""Read frozen R10 C trajectories; no new physics or native-only force substitute."""
import json,csv
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/root/autodl-tmp/MCP-RL-r10-gnc-r3')
RUN='tm-s1-go2-soft-landing-r10'
OUT=ROOT/'results/terrain_memory/simulation/tm-s1-go2-material-continuation-r11'
OUT.mkdir(parents=True,exist_ok=True)
summaries=[]
for name in ['dev10-GNC-r3-cap025','dev11-GNC-r3-cap0215','dev12-GNC-r3-cap018']:
    directory=OLD/'outputs/terrain_memory/simulation'/RUN/name
    result=json.loads((OLD/'results/terrain_memory/simulation'/RUN/(name+'.json')).read_text())
    resolved=json.loads((OLD/'results/terrain_memory/simulation'/RUN/(name+'-resolved.json')).read_text())
    config=resolved['r10_configuration']
    with np.load(directory/'trajectory.npz') as z: tr={k:z[k] for k in z.files}
    foot=np.concatenate([np.load(directory/n)['foot'] for n in result['material']['chunks']])
    physical=np.concatenate([np.load(x)['physical'] for x in sorted(directory.glob('physical-*.npz'))])
    model=mujoco.MjModel.from_binary_path(str(directory/'model.mjb')); data=mujoco.MjData(model)
    feet=[model.geom(n).id for n in ['FL','FR','RL','RR']]
    stiffness=config['local_contact_model']['instantaneous_stiffness_N_m']
    height=config['normal_landing']['material_support_height']['target_m']
    end=float(tr['time'][-1]); lateral_i=int(np.argmax(abs(tr['qpos'][:,1])))
    lateral_t=float(tr['time'][lateral_i])
    windows={'terminal':(end-.5,end)}
    if name.startswith('dev11'):windows['lateral']=(lateral_t-.25,lateral_t+.25)
    prior=OUT/(name+'-C-plan-contact.csv')
    if prior.exists():
        with prior.open() as f: rows=list(csv.DictReader(f))
        for row in rows:
            for key in row:
                if key in ('attempt',):continue
                if key in ('predicted_soft','actual_observed'):row[key]=row[key]=='True'
                else:row[key]=float(row[key]) if row[key] else None
    else:
        rows=[]
        for file in sorted(directory.glob('prediction-*.npz')):
            with np.load(file) as z: pr={k:z[k] for k in z.files}
            for i,origin in enumerate(pr['plan_origin']):
                selected=[w for w,(lo,hi) in windows.items() if lo-.35<=origin<=hi]
                if not selected:continue
                ti=int(np.argmin(abs(tr['time']-origin)))
                anchors=tr['support_reference_userdata'][ti,62:78].reshape(4,4)
                for j,tm in enumerate(pr['plan_times'][i]):
                    state=pr['plan_states'][i,j]
                    data.qpos[:]=state[:model.nq];data.qvel[:]=state[model.nq:model.nq+model.nv]
                    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_comVel(model,data)
                    for leg,geom in enumerate(feet):
                        point=data.geom_xpos[geom].copy();point[2]-=model.geom_size[geom,0]
                        vel=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_GEOM,geom,vel,0)
                        soft=.8+.015811388300841896<=point[0]<=2.2-.015811388300841896 and abs(point[1])<=.6-.015811388300841896
                        overlap=np.prod(np.maximum(1-abs(point[:2]-anchors[:,1:3])/.03162277660168379,0),axis=1)
                        rest=float(np.dot(anchors[:,0]*overlap,anchors[:,3]))
                        fz=stiffness*max(rest-point[2],0) if soft else np.nan
                        gate=.5*(1-np.tanh((point[2]-.005)/.002))
                        desired=-.05*np.tanh(max(point[2],0)/.001)
                        vr=gate*(min(vel[5],0)-desired)/.05 if soft else 0
                        over=-point[2]-.00019889981664
                        ir=.5*(over+np.sqrt(over*over+1e-12))/.001 if soft else 0
                        actual=tm<=end+1e-9
                        row=dict(attempt=name,origin_s=float(origin),time_s=float(tm),leg=leg,
                            predicted_x_m=float(point[0]),predicted_y_m=float(point[1]),
                            predicted_bottom_m=float(point[2]),predicted_vz_m_s=float(vel[5]),predicted_soft=bool(soft),
                            predicted_C_Fz_N=float(fz),anchor_rest_m=rest,anchor_overlap_sum=float(np.dot(anchors[:,0],overlap)),
                            predicted_base_z_m=float(state[2]),normal_velocity_residual=float(vr),indentation_residual=float(ir),
                            normal_velocity_quadratic_contribution=float(.5*.02*vr*vr),
                            indentation_quadratic_contribution=float(.5*.1*ir*ir),
                            material_height_quadratic_contribution=float(.5*((state[2]-height)/.04)**2),
                            actual_observed=actual)
                        for label,col in [('bottom_m',8+3*leg),('vz_m_s',20+3*leg),('Fz_N',32+3*leg)]:
                            row['actual_'+label]=float(np.interp(tm,foot[:,0],foot[:,col])) if actual else None
                        rows.append(row)
    with (OUT/(name+'-C-plan-contact.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(attempt=name,reason=result['reason'],end_s=end,stiffness_N_m=stiffness,
        terminal_xyz=tr['qpos'][-1,:3].tolist(),terminal_rpy=tr['rpy'][-1].tolist(),
        maximum_lateral=dict(time_s=lateral_t,xyz=tr['qpos'][lateral_i,:3].tolist(),
            rpy=tr['rpy'][lateral_i].tolist(),soft_mask=foot[np.argmin(abs(foot[:,0]-lateral_t)),2:6].tolist(),
            feet_xy=tr['feet'][lateral_i,:,:2].tolist()),
        applied_settings=resolved['applied_settings'],windows={})
    for window,(lo,hi) in windows.items():
        select=(foot[:,0]>=lo)&(foot[:,0]<=hi)
        tsel=(tr['time']>=lo)&(tr['time']<=hi)
        psel=(physical[:,0]>=lo)&(physical[:,0]<=hi)
        ss=foot[select];lim=np.tile([23.7,23.7,45.43],4)
        summary['windows'][window]=dict(start_s=lo,end_s=hi,
            peak_downward_m_s=(-ss[:,20:30:3]).max(axis=0).tolist(),
            peak_Fz_N=ss[:,32:42:3].max(axis=0).tolist(),
            min_bottom_mm=(1000*ss[:,8:18:3].min(axis=0)).tolist(),
            saturated_joint_seconds=(np.sum(abs(physical[psel,49:61])>=lim-1e-6,axis=0)*.0002).tolist(),
            feedback_nominal_delta_max_rad=np.max(abs(tr['q_target'][tsel]-tr['nominal_q_target'][tsel]),axis=0).tolist(),
            min_body_z_m=float(tr['qpos'][tsel,2].min()),landings=[])
        for leg in range(4):
            loading=(foot[:,32+3*leg]>1)&(foot[:,2+leg]>0)
            starts=np.flatnonzero(loading & ~np.r_[False,loading[:-1]])
            starts=[k for k in starts if lo<=foot[k,0]<=hi][-1:]
            for k in starts:
                tm=float(foot[k,0])
                if not lo<=tm<=hi:continue
                origins=sorted(set(r['origin_s'] for r in rows if r['origin_s']<=tm and r['leg']==leg))
                evidence=[]
                for origin in origins[-3:]:
                    nodes=[r for r in rows if r['origin_s']==origin and r['leg']==leg and abs(r['time_s']-tm)<.012]
                    if not nodes:continue
                    near=min(nodes,key=lambda r:abs(r['time_s']-tm))
                    crossing=[r for r in rows if r['origin_s']==origin and r['leg']==leg and r['time_s']>=origin and r['predicted_C_Fz_N']>1]
                    evidence.append(dict(origin_s=origin,at_actual_contact=near,
                        next_predicted_loaded=crossing[0] if crossing else None))
                summary['windows'][window]['landings'].append(dict(leg=leg,time_s=tm,
                    actual_bottom_m=float(foot[k,8+3*leg]),actual_vz_m_s=float(foot[k,20+3*leg]),
                    actual_Fz_N=float(foot[k,32+3*leg]),plans=evidence))
    summaries.append(summary)
out=ROOT/'work/r11-contact-evidence.json'
out.write_text(json.dumps(summaries,indent=2,allow_nan=True,default=lambda x:x.item())+'\n')
for s in summaries:
    print(s['attempt'], {k:len(v['landings']) for k,v in s['windows'].items()}, flush=True)
