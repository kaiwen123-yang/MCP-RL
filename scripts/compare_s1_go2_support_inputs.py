#!/usr/bin/env python3
"""Same-input native/prediction calculations on completed R8 episode states."""
import argparse,json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from run_s1_go2_wb_baseline import ROOT,index_file
from summarize_s1_go2_wb_support import table,CONFIG

def compare(name):
    c=json.loads((ROOT/CONFIG).read_text());small=ROOT/c['outputs']['root']
    out=ROOT/'outputs/terrain_memory/simulation'/c['run_id']/name
    r=json.loads((small/(name+'.json')).read_text());resolved=json.loads((small/(name+'-resolved.json')).read_text())
    with np.load(out/'trajectory.npz') as z:a={k:z[k] for k in z.files}
    # Original author retains its 349ms horizon: common 348ms endpoint.
    duration=.348 if r['stage']=='G0' else .350;n=round(duration/.002)
    desired=[0.,1.,2.,10.,20.,40.,55.]+[float(t['time_s'])+1 for t in r['transitions'] if t['phase'] in ('terminal_stand','far_hold','home_hold','complete')]
    if r['mission']=='long':desired += [270.,290.]
    origins=[];targets=[];traces=[];rows=[]
    for chunk in r['chunks']:
        if chunk['stream']!='prediction':continue
        with np.load(chunk['path']) as z:
            po=z['plan_origin'];pt=z['plan_times'];actions=z['plan_actions'];states=z['plan_states']
            selected=sorted(set(int(np.searchsorted(po,t-1e-8)) for t in desired if po[0]-1e-8<=t<=po[-1]+1e-8))
            for i in selected:
                origin=float(po[i]);j=int(np.searchsorted(a['time'],origin-1e-8))
                if origin+duration>a['time'][-1]+1e-8:continue
                nominal=np.stack([np.interp(origin+np.arange(n)*.002,pt[i,:len(actions[i])],actions[i,:,k]) for k in range(12)],axis=1)
                for schedule,u in [('nominal_2ms_hold',nominal),('recorded_feedback_2ms_hold',a['q_target'][j:j+n])]:
                    variants=[]
                    for variant in ['native','prediction']:
                        m=mujoco.MjModel.from_binary_path(str(out/'model.mjb'))
                        if variant=='prediction':
                            p=resolved['prediction_settings']['numerics'];m.opt.timestep=p['agent_timestep'];m.opt.integrator=int(p['agent_integrator'])
                            if p['agent_differentiable']:
                                m.jnt_solimp[:,0]=0;m.geom_solimp[:,0]=0;m.pair_solimp[:,0]=0
                        d=mujoco.MjData(m);d.qpos[:]=a['qpos'][j];d.qvel[:]=a['qvel'][j];d.time=origin
                        d.mocap_pos[:]=a['mocap_pos'][j];d.mocap_quat[:]=a['mocap_quat'][j]
                        if r['arm']=='S':d.userdata[:]=a['support_reference_userdata'][j]
                        trace=[np.r_[d.qpos,d.qvel]]
                        for action in u:
                            d.ctrl[:]=action
                            for _ in range(round(.002/m.opt.timestep)):mujoco.mj_step(m,d)
                            trace.append(np.r_[d.qpos,d.qvel])
                        variants.append(np.asarray(trace))
                    native,pred=variants;dq=native[-1,:3]-pred[-1,:3]
                    ori=(Rotation.from_quat(native[-1,[4,5,6,3]]).inv()*Rotation.from_quat(pred[-1,[4,5,6,3]])).magnitude()
                    rows.append(dict(origin_s=origin,schedule=schedule,horizon_s=duration,
                        position_difference_m=float(np.linalg.norm(dq)),orientation_difference_rad=float(ori),
                        velocity_difference_m_s=float(np.linalg.norm(native[-1,19:22]-pred[-1,19:22])),
                        omega_difference_rad_s=float(np.linalg.norm(native[-1,22:25]-pred[-1,22:25])),
                        native_vs_recorded_position_difference_m=float(np.linalg.norm(native[-1,:3]-a['qpos'][j+n,:3])) if schedule.startswith('recorded') else None))
                    origins.append(origin);targets.append(u);traces.append(variants)
    table(small/(name+'-same-input.csv'),rows)
    np.savez_compressed(out/'same-input.npz',origin_s=origins,targets=targets,native_prediction_traces=traces)
    with index_file(small,c) as idx:
        idx.setdefault('same_input_calculations',{})[name]=dict(table=name+'-same-input.csv',cases=len(rows),
            semantics='Identical2ms held target sequences in native plant and actual predictor settings; native rate retained, two1ms prediction steps. Nominal and recorded feedback inputs separate.',
            limitation='Original qacc_warmstart and other integrator auxiliaries not recorded; both calculations initialize auxiliaries equally. No policy feedback/replanning in nominal replay. This is saved-data calculation, not new physical start.')
    print(json.dumps(dict(attempt=name,cases=len(rows))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args();compare(args.attempt)
