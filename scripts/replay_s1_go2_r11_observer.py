#!/usr/bin/env python3
"""Causal PANN replay of retained dev11 geometry and completed plans; no robot integration."""
import hashlib,json,os,time
from pathlib import Path
import mujoco,numpy as np
from s1_go2_r10_observer import MaterialObserver
from s1_go2_r11_protocol import ROOT,CONTRACT
def main():
    old=Path('/root/autodl-tmp/MCP-RL-r10-gnc-r3')
    run='tm-s1-go2-soft-landing-r10';name='dev11-GNC-r3-cap0215'
    directory=old/'outputs/terrain_memory/simulation'/run/name
    result=json.loads((old/'results/terrain_memory/simulation'/run/(name+'.json')).read_text())
    weights=Path(os.environ['MPC_PANN_WEIGHTS'])
    weight_hash=hashlib.file_digest(weights.open('rb'),'sha256').hexdigest()
    if weight_hash!=CONTRACT['joint_learning']['pann_sha256']:raise ValueError('Frozen B ideal identity differs')
    params=json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
    observer=MaterialObserver(params,weights)
    tr=np.load(directory/'trajectory.npz')
    # Only measured foot kinematics are read. True material cells/q/e are never loaded.
    foot=np.concatenate([np.load(directory/n)['foot'] for n in result['material']['chunks']])
    packets=[np.load(p) for p in sorted(directory.glob('prediction-*.npz'))]
    origins=np.concatenate([p['plan_origin'] for p in packets])
    states=np.concatenate([p['plan_states'] for p in packets])
    plan_times=np.concatenate([p['plan_times'] for p in packets])
    model=mujoco.MjModel.from_binary_path(str(directory/'model.mjb'));data=mujoco.MjData(model)
    feet=[model.geom(n).id for n in ['FL','FR','RL','RR']]
    begin=time.perf_counter();last_tick=-1;records=[]
    for t in tr['time']:
        j=int(np.searchsorted(foot[:,0],t))
        if j==len(foot):j-=1
        if j>0 and abs(foot[j-1,0]-t)<abs(foot[j,0]-t):j-=1
        observer.observe(float(t),foot[j,6:18].reshape(4,3),foot[j,18:30].reshape(4,3))
        tick=int(np.floor((t+1e-8)/.5))
        terminal=bool(t==tr['time'][-1])
        if tick==last_tick and not terminal:continue
        # The policy samples before the same-time planner update. At t=0 no
        # completed MPC trajectory exists yet.
        pi=int(np.searchsorted(origins,t-1e-8,side='left'))-1
        query=np.zeros((4,2));mask=np.zeros(4,bool);feature=np.zeros((4,3))
        if pi>=0:
            node=int(np.argmin(abs(plan_times[pi]-(t+.2))))
            data.qpos[:]=states[pi,node,:model.nq];mujoco.mj_kinematics(model,data)
            query=data.geom_xpos[feet,:2].copy()
            feature,mask=observer.recovery(query)
        records.append(dict(time_s=float(t),completed_plan_origin_s=float(origins[pi]) if pi>=0 else None,
            query_xy_m=query.tolist(),soft=mask.tolist(),recovery_m=feature.tolist(),
            kind='final_observation' if terminal and tick==last_tick else 'actor_tick'))
        last_tick=tick
        if tick%10==0:print(json.dumps(dict(replayed_s=float(t),wall_s=time.perf_counter()-begin,world_cells=len(observer.ground.e))),flush=True)
    out=ROOT/CONTRACT['outputs']['large_root']/'observer-retained-dev11'
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'observer-final.npz',**observer.states())
    (out/'recovery.json').write_text(json.dumps(records,indent=2)+'\n')
    summary=dict(kind='causal_retained_replay_not_closed_loop_or_training',source_run=run,source_attempt=name,
        observed_through_s=float(tr['time'][-1]),source_kinematics='0.2ms force-boundary foot positions/velocities sampled at recorded2ms control times and final endpoint',
        query_rule='Latest strictly earlier completed plan, t+.20s footprint; no same-tick future planner update',
        weights_sha256=weight_hash,new_robot_physical_steps=0,training_transitions=0,
        max_estimated_strain=observer.ground.max_strain,estimated_domain_exit=observer.ground.domain_exit,
        total_wall_s=time.perf_counter()-begin,**observer.timing())
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
