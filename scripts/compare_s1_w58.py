"""Pair the preregistered W58 physical episodes, including absent and unrealized target events."""
import json
from pathlib import Path

import numpy as np

from inspect_s1_w58_predecessor import REL, read
from publish_s1_go2_w54 import write

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'/REL/'w58'
RAW=ROOT/'outputs'/REL
ARMS=('B2','P2-private','P2-full','Fixed-wait')


def metrics(name):
    raw=RAW/name
    target=R/'mechanics'/(name+'.json')
    if target.exists():return read(target)
    path=R/'episodes'/(name+'.json')
    if not path.exists():return dict(attempt=name,status='not_run_or_no_physical_result',targets=[])
    episode=read(path)
    if not (raw/'trajectory.npz').is_file():
        return dict(attempt=name,status=episode.get('stage_status','technical_interruption'),reason=episode.get('reason'),targets=[])
    mech=episode.get('mechanism',{})
    targets=mech.get('all_target_rows',[])
    groups={}
    with np.load(raw/'trajectory.npz') as data:
        pos=data['qpos'][:,:2];clock=data['time']
        phase=data['phase']; torque=data['actuator_force'];v=data['qvel'][:,6:]
        feet=data['feet_velocity'][:,:,:2];forces=data['actual_grf'][:,:,2]
        dt=np.diff(clock,append=clock[-1])
        ds=np.r_[np.linalg.norm(np.diff(pos,axis=0),axis=1),0.]
        motion=np.isin(phase,[1,2,3,8])
        mass=read(raw/'w55-f/manifest.json')['model_mass_kg']
        for key,mask in [('all',np.ones(len(clock),bool)),('motion',motion),
                         ('material',motion&(pos[:,0]>=.8)&(pos[:,0]<=2.2)&(abs(pos[:,1])<=.6)),
                         ('turn',np.isin(phase,[2,8])),('hold',~motion)]:
            length=float(ds[mask].sum())
            work=float((dt[mask,None]*np.maximum(torque[mask]*v[mask],0.)).sum())
            shear=float((dt[mask,None]*(forces[mask]>1)*np.linalg.norm(feet[mask],axis=2)).sum())
            groups[key]=dict(duration_s=float(dt[mask].sum()),path_m=length,positive_joint_work_J=work,
                             mechanical_CoT=work/(mass*9.81*length) if length>0 else None,
                             loaded_center_slip_m=shear,loaded_center_slip_per_m=shear/length if length>0 else None)
    r=dict(attempt=name,status=episode['stage_status'],reason=episode['reason'],physics_s=episode['simulation_s'],
           first_home_s=episode['first_completion_s'],physical_samples=episode['physical_samples'],
           full_window=episode['simulation_s']>=episode['evaluation_stage']['duration_s']-1e-7,
           controller=episode['case_identity']['controller_build'],groups=groups,targets=targets,
           preparation_station_s=mech.get('preparation_station_s'),
           max_principal_strain=episode['material']['max_abs_principal_strain'],
           slip_integral_s=episode['material']['integral_slip_sq_s'],
           logical_timing=episode['timing'],wbc_timing=episode.get('w54_f_recording',{}).get('timing'),
           learning_timing=episode.get('w57_timing'))
    if targets:
        from score_s1_w58_p2 import score_episode
        r['P2']=score_episode(raw,targets)
    handovers=raw/'w55-f/w58-handover.jsonl'
    if handovers.exists():
        rr=[json.loads(l) for l in handovers.open()]
        r['P2_requested_wait_s']=sum(x['selected_wait_s'] or 0. for x in rr)
        r['candidate_p95_ms']=float(np.quantile([x['planning_wall_s']*1000 for x in rr],.95)) if rr else None
    write(target,r)
    return r


def matched_matrix(jobs, originals, split):
    out=[]
    for j in jobs:
        info=j['data']
        ledger=metrics(j['id'])
        out.append(dict(arm=info['arm'],case=info['case'],history=info['task'],seed=info['learning_seed'],
            attempt=j['id'],status=ledger['status'],own_E30=originals.get((info['arm'],info['case'],info['learning_seed'])),
            physics_s=ledger.get('physics_s'),full_window=ledger.get('full_window',False),home_s=ledger.get('first_home_s'),
            targets_realized=[t['target_id'] for t in ledger['targets'] if t['realized']],
            targets_declared=12,target_ledger=str(R.relative_to(ROOT)/'mechanics'/(j['id']+'.json')) if 'groups' in ledger else None,
            motion=ledger.get('groups',{}).get('motion'),material=ledger.get('groups',{}).get('material'),
            original_slip_integral_s=ledger.get('slip_integral_s'),preparation_s=ledger.get('preparation_station_s'),
            requested_wait_s=ledger.get('P2_requested_wait_s'),history_action_evidence=ledger.get('P2',{}).get('history_action_evidence',False)))
    write(R/(split+'-matrix.json'),dict(role=split,episodes=out,planned=len(jobs),
        completed_observation_windows=sum(x['full_window'] for x in out),
        passed=sum(x['status']=='passed' for x in out)))
    return out


def decide(rows, qualifiers):
    look={(r['case'],r['history'],r['arm']):r for r in rows}
    comparisons={}
    for case in 'AB':
        measures={}
        for history in ('short','recovered'):
            group={arm:look[case,history,arm] for arm in ARMS}
            common=sorted(set.intersection(*(set(r['targets_realized']) for r in group.values())))
            observed=all(r['full_window'] and r['home_s'] is not None and bool(r['motion']) for r in group.values())
            diffs={}
            if observed:
                for arm in ('B2','P2-private','Fixed-wait'):
                    full,other=group['P2-full'],group[arm]
                    diffs[arm]=dict(slip=full['motion']['loaded_center_slip_per_m']-other['motion']['loaded_center_slip_per_m'],
                        cot=full['motion']['mechanical_CoT']-other['motion']['mechanical_CoT'],
                        home_s=full['home_s']-other['home_s'],
                        preparation_s=full['preparation_s']-other['preparation_s'])
            measures[history]=dict(complete_observation_and_home=observed,common_targets=common,
                realized_by_arm={a:v['targets_realized'] for a,v in group.items()},full_minus_other=diffs,
                full_history_action=group['P2-full']['history_action_evidence'])
        comparisons[case]=measures
    exposure=all(m['common_targets'] and m['complete_observation_and_home'] for c in comparisons.values() for m in c.values())
    action=all(c['short']['full_history_action'] for c in comparisons.values())
    def improves(metric):
        for c in comparisons.values():
            for arm in ('P2-private','Fixed-wait'):
                a,b=c['short']['full_minus_other'][arm],c['recovered']['full_minus_other'][arm]
                if not (a[metric]<0 and a[metric]<b[metric] and a['home_s']<=.002):return False
        return True
    signs={metric:exposure and improves(metric) for metric in ('slip','cot')}
    eligible=bool(all(qualifiers.values()) and exposure and action and any(signs.values()))
    decision=dict(joint_eligible=eligible,own_original_E60=qualifiers,realized_paired_exposure=bool(exposure),
        same_state_action_on_actual_short_target=action,short_benefit_beyond_private_and_fixed=signs,
        comparisons=comparisons,criterion='Consistent direction in A and B for moving loaded-foot slip per metre or positive mechanical CoT, relative to private and fixed-wait. Require a larger effect under short history than after recovery, with no slower home arrival beyond one 2ms control tick. Report both costs. No significance claim.',
        validation_split_used=False)
    write(R/'development-decision.json',decision)
    return decision

