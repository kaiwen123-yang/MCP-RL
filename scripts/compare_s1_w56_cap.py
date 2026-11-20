"""Pair each W56 original stage with its frozen r55 case at task phases and events."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from publish_s1_go2_w54 import write
from s1_go2_r11_runtime import PHASES

RUN='tm-s1-go2-local-readiness-r12'
REL=Path('results/terrain_memory/simulation')/RUN


def read(path):
    return json.loads(path.read_text())


def describe(worktree,compact,case,level,mode):
    state=read(case/(level+'.json'))
    attempt=state['attempt']
    result=read(compact/(attempt+'.json'))
    raw=worktree/'outputs/terrain_memory/simulation'/RUN/attempt
    rows=read(raw/'pann-action-dependence.json')
    selected_key='zero_feature_cap' if mode=='zero_recovery_features' else 'with_pann_cap'
    actor_caps=np.array([r[selected_key] for r in rows])
    with np.load(raw/'trajectory.npz') as z:
        t=z['time'];caps=z['requested_speed_cap'];phase=z['phase'];q=z['qpos'];v=z['qvel'];rpy=z['rpy']
        decision_t=np.array([r['time_s'] for r in rows])
        tick=np.searchsorted(t,decision_t-1.e-8)
        present=(tick<len(t)) & (decision_t<=t[-1]+1.e-8)
        tick=tick[present]
        # A failed solve can terminate before its decision is written as a 500-Hz command.
        aligned=abs(t[tick]-decision_t[present])<1.e-8
        j=np.flatnonzero(present)[aligned]
        mismatch=float(abs(caps[tick[aligned]]-actor_caps[j]).max()) if len(j) else None
        if mismatch is not None and mismatch>0:
            raise ValueError('Trajectory command differs from the selected frozen-actor cap')
        phases={}
        for p in np.unique(phase):
            use=phase==p
            phases[PHASES[int(p)]]=dict(observation_s=[float(t[use][0]),float(t[use][-1])],
                z_min_max_m=[float(q[use,2].min()),float(q[use,2].max())],
                max_abs_roll_pitch_rad=abs(rpy[use,:2]).max(0).tolist(),
                body_linear_speed_RMS_m_s=float(np.mean((v[use,:3]**2).sum(axis=1))**.5),
                requested_cap_min_max_mean_m_s=[float(f(caps[use])) for f in (np.min,np.max,np.mean)])
    events=defaultdict(list)
    with (raw/'material-events.jsonl').open() as stream:
        for line in stream:
            e=json.loads(line)
            if e['time_s']<=result['simulation_s']+1e-9:
                events[e['type']].append(e)
    groups={}
    for kind,entries in events.items():
        groups[kind]=dict(count=len(entries),first_s=entries[0]['time_s'],
            max_abs_strain_after=max(max(map(abs,x['e'])) for x in entries),
            max_abs_q_before=max(max(map(abs,x['q_before_step'])) for x in entries),
            max_abs_q_after=max(max(map(abs,x['q'])) for x in entries))
    residuals=defaultdict(float)
    for file in sorted((raw/'w55-f').glob('control-*.npz')):
        with np.load(file) as z:
            for key in ('solver_primal_residual','solver_dual_residual','dynamics_residual'):
                residuals[key]=max(residuals[key],float(abs(z[key]).max()))
            torque_violation=np.maximum(z['torque_lower']-z['tau_total'],z['tau_total']-z['torque_upper'])
            residuals['max_start_torque_violation_Nm']=max(residuals['max_start_torque_violation_Nm'],float(torque_violation.max()))
    cap_delta=max((abs(r['with_pann_cap']-r['zero_feature_cap']) for r in rows),default=None)
    return dict(attempt=attempt,status=state['status'],reason=result['reason'],checkpoint_sha256=result['case_identity']['checkpoint_sha256'],
        normalizer_sha256=result['case_identity']['normalizer_sha256'],workers=result['qualification']['solver_workers'],
        simulation_s=result['simulation_s'],roundtrips=result['legal_route']['completed_roundtrips'],
        passages=result['legal_route']['passages'],phase_transitions=[{k:r[k] for k in ('time_s','phase')} for r in result['transitions']],
        final_stable_s=result['final_stable_s'],slip_integral=result['material']['integral_slip_sq_s'],
        peak_material_strain=result['material']['max_abs_principal_strain'],max_joint_limit_excess_rad=result['max_joint_limit_excess_rad'],
        original_unit_residuals=dict(residuals),material_events=groups,by_phase=phases,
        actor=dict(selected_mode=mode,total_decisions=len(rows),command_readback_matches=len(j),max_cap_readback_error_m_s=mismatch,
                   max_same_input_recovery_effect_m_s=cap_delta,executed_cap_range_m_s=[float(actor_caps.min()),float(actor_caps.max())] if len(rows) else None))


def compare(case,level):
    root=Path(__file__).resolve().parents[1]
    label=case.name
    letter=label.split('-')[1]
    mode=read(case/'job.json')['material_feature_mode']
    base=root.parent/'MCP-RL-w55-f'
    original=base/REL/'w55/f'
    off=describe(root,case.parent/label.split('-')[-1],case,level,mode)
    on=describe(base,original,original/f'F-{letter}-r55',level,'learned_history')
    for k in ('checkpoint_sha256','normalizer_sha256','workers'):
        if on[k]!=off[k]:
            raise ValueError('Paired case changed '+k)
    difference=dict(roundtrips=off['roundtrips']-on['roundtrips'],slip_integral=off['slip_integral']-on['slip_integral'],
        peak_material_strain=off['peak_material_strain']-on['peak_material_strain'])
    passage=lambda d:{(r['lap'],r['direction']):r for r in d['passages'] if r.get('completed')}
    ref,change=passage(on),passage(off)
    difference['matched_passages']=[dict(lap=k[0],direction=k[1],entry_time_change_s=change[k]['entry_s']-ref[k]['entry_s'],
        exit_time_change_s=change[k]['exit_s']-ref[k]['exit_s']) for k in sorted(ref.keys()&change.keys())]
    destination=case/(level+'-paired.json')
    write(destination,dict(case=letter,stage=level,baseline=on,new=off,difference_new_minus_baseline=difference,
        intervention=('Twelve explicit actor recovery inputs zeroed before frozen actor' if mode=='zero_recovery_features' else
                      'Causal observer window and material normal execution reference together'),
        scope='One deterministic original case per stage. Event and phase comparisons after closed-loop divergence.'))
    return destination
