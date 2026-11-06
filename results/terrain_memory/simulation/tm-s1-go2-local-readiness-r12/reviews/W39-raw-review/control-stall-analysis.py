#!/usr/bin/env python3
"""Read B272's saved controls and plans to characterize its nonprogress tail."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('control_analysis',HERE/'control-analysis.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);args=parser.parse_args()
    attempt='train-job-20261105T032354Z-w39-b-PANN_PPO_S5-s2-ep00272'
    index=json.loads((args.raw/'raw-index.json').read_text());files={r['name']:r for r in index if r['attempt']==attempt}
    hashes=[];tr=audit.load(files['trajectory.npz'],hashes);plans=audit.load(files['planner-records.jsonl'],hashes)
    acts=audit.load(files['feedback-actions.jsonl'],hashes);packets=audit.load(files['reference-packets.json'],hashes)['packets']
    result=json.loads((audit.SMALL/(attempt+'.json')).read_text());t=tr['time'];valid=(abs(t-tr['diagnostic_time'])<1e-10)&(t<result['simulation_s']-1e-10)
    byclock={float(np.float32(t[i])):i for i in np.flatnonzero(valid)};amap={}
    for r in acts:
        i=byclock.get(r['time'])
        if i is not None and np.array_equal(np.asarray(r['action_after_clip'],dtype=np.float32),tr['q_target'][i]):amap[i]=r
    limits=np.tile([23.7,23.7,45.43],4);sat=np.any(abs(tr['unclipped_pd_request'])>limits,axis=1)
    windows=[]
    for lo,hi in [(2,9.966),(9.966,11),(11,15),(15,20),(20,60)]:
        mask=valid&(t>=lo-1e-9)&(t<hi-1e-9);ii=np.flatnonzero(mask)
        pp=[p for p in plans if p['origin']>=lo-1e-9 and p['origin']<hi-1e-9]
        aa=[amap[i] for i in ii if i in amap];k=[max(abs(v) for v in r['raw_K_delta_x']) for r in aa]
        windows.append(dict(window_s=[lo,hi],control_rows=len(ii),PD_saturated_control_rows=int(sat[mask].sum()),
            large_raw_K_gt_05_rows=int(np.sum(np.array(k)>.5)),max_raw_K_delta_x_rad=max(k),
            planning_calls=len(pp),accepted_trunk_infeasible_plans=sum(p['accepted_trunk_violation_m']>0 for p in pp),
            nominal_trunk_infeasible_plans=sum(p['nominal_trunk_violation_m']>0 for p in pp),
            all_ten_candidates_trunk_feasible_plans=sum(p['feasible_candidates']==10 for p in pp),
            accepted_cost=audit.stat([p['accepted_cost'] for p in pp]),alpha=audit.stat([p['action_step'] for p in pp]),
            base_height_m=audit.stat(tr['qpos'][mask,2]),vx_m_s=audit.stat(tr['qvel'][mask,0]),
            command_vx_m_s=audit.stat(tr['command_velocity'][mask,0]),cap_m_s=audit.stat(tr['requested_speed_cap'][mask]),
            max_abs_roll_rad=float(np.max(abs(tr['rpy'][mask,0]))),
            right_rear_target_at_original_bound_fraction={name:float(np.mean(abs(tr['q_target'][mask,j]-b)<1e-6)) for name,j,b in [('hip_upper',9,1.0472),('thigh_upper',10,4.5379),('calf_upper',11,-.83776)]}))
    samples=[]
    for tm in [9.966,11.344,12.532,15,20,30,40,50,60]:
        i=int(np.argmin(abs(t-tm)));a=amap.get(i);p=next((p for p in plans if a and p['plan_id']==a['plan_id']),None)
        packet=max((r for r in packets if r['new_optimized_action_time_s']<=t[i]+1e-9),key=lambda r:r['new_optimized_action_time_s'])
        samples.append(dict(time_s=float(t[i]),qpos=tr['qpos'][i].tolist(),qvel=tr['qvel'][i].tolist(),rpy=tr['rpy'][i].tolist(),
            feet=tr['feet'][i].tolist(),contact=tr['contact'][i].tolist(),q_target=tr['q_target'][i].tolist(),
            target_was_applied_to_following_physics=bool(valid[i]),packet=packet['packet'],
            max_raw_K_delta_x_rad=None if a is None else max(abs(v) for v in a['raw_K_delta_x']),
            plan=None if p is None else {k:p[k] for k in ['origin','plan_id','winner','action_step','accepted_cost','nominal_trunk_violation_m','accepted_trunk_violation_m','regularization_used','cost_surprise']}))
    out=dict(attempt=attempt,new_physics_steps=0,reason=result['reason'],sources=hashes,windows=windows,samples=samples,
        interpretation='Initial feedback/actuation instability is followed by a badly configured but nonterminal near-equilibrium. The controller continues to plan and receives a nonzero outbound packet. Low late saturation and survival alone do not establish route capability.',
        evidence_limit='Accepted full horizon and K outside conditional nodes 0/+20ms were never logged. This report does not prove a unique local minimum or claim a controlled causal experiment.')
    (HERE/'control-stall-B272.json').write_text(json.dumps(out,indent=2,ensure_ascii=False,default=audit.scalar)+'\n')
    print(json.dumps(dict(attempt=attempt,windows=windows),ensure_ascii=False,default=audit.scalar))


if __name__=='__main__':main()
