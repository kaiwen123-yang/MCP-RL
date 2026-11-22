"""Counterfactual protected-history action only on a physically bound P/A target, WBC unchanged."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from analyze_s1_w57_planner_outcomes import analyze
from qualify_s1_w57_planning import Saved, ROOT, REL, PARAM, CFG, solve
from replay_s1_w57_force_execution import replay_wbc
from s1_w54_f_r52_backend import assemble_centroidal
from s1_w57_material import W57MaterialObserver
from s1_w57_planning import Forecast,rank


class Actual(Saved):
    def __init__(self,raw,estimator):
        self.root=raw;self.raw=raw/'w55-f';self.tick=0
        self.obs=W57MaterialObserver(PARAM,ROOT/'work/assets/B_random1000--ideal-potentials.npz',estimator)


def nominal_qp(plan,wbc,src):
    inp={k:plan[k].copy() for k in ('x0','reference','last_force','contact','feet','force_normal_upper','stage_A','stage_B','stage_c')}
    points=plan['foot_center_plan'].copy();points[:,:,2]-=.022;points[0]=wbc['contact_force_point']
    inp['feet']=points;n=len(inp['contact']);dt=CFG['mpc']['dt_s']
    centers=plan.get('moment_linearization_com',plan['reference'][:-1,3:6])
    orig=[assemble_centroidal(float(plan['mass']),plan['inertia'],centers[j],points[j],
                              plan['x0'][:3],plan['x0'][6:9],dt,CFG['force_pyramid_slope']) for j in range(n)]
    inp['stage_B']=np.stack([s['B'] for s in orig])
    # Re-solve nominal states for a possible nominal winner.
    from s1_w54_f_r52_backend import assemble_force_horizon
    z,_=solve(assemble_force_horizon(inp,CFG));nx=inp['reference'].size
    nominal_force=src['nominal_force'];nominal_state=z[:nx].reshape(inp['reference'].shape)
    return inp,nominal_force,nominal_state


def evidence(attempt):
    r12=ROOT/'results'/REL/'w57';file=r12/'p/physical'/(attempt+'.json')
    dest=r12/'p/physical'/(attempt+'-private-action.json')
    if dest.exists():
        answer=json.loads(dest.read_text())
        return {k:answer[k] for k in ('attempt','distinct_target_events','distinct_plans','above_reconstruction')}
    if not file.exists():analyze(attempt)
    closure=json.loads(file.read_text());row=json.loads((r12/'attempts'/(attempt+'.json')).read_text())
    if row['arm']!='P' or row['case']!='A' or row['task_mode'] not in ('short','original'):
        raise ValueError('The protected-history closed-loop screen was preselected for P, case A, original/short.')
    raw=ROOT/'outputs'/REL/attempt
    saved=Actual(raw,row['material'])
    entries=closure['first_matching_publication_per_true_crossfoot_target']
    grouped={}
    for target in entries:grouped.setdefault(target['call'],[]).append(target)
    solved=[];base=ROOT/'work/w57-p/closed-action-replay'/attempt
    for call,targets in sorted(grouped.items()):
        with np.load(saved.raw/f'mpc-{call:06d}.npz') as z:plan=dict(z)
        with np.load(saved.raw/targets[0]['data_file']) as z:src=dict(z)
        t=float(plan['times'][0]);tick=round(t/.002);saved.state(t)
        with np.load(saved.raw/f'control-{tick//500+1:05d}.npz') as z:
            wbc={key:z[key][tick%500] for key in z.files}
        fc=Forecast(saved.obs.ground,plan['times'],src['segment_grid_s'],src['xy'],src['active'],
                    saved.obs.published_bottom,wbc['observed_contact'],plan['foot_center_plan'][:-1,:,:2])
        if not np.array_equal(fc.keys,src['world_keys']):raise ValueError('Saved and replayed planned world keys differ')
        replay_error=float(np.abs(fc.initial-src['initial']).max(initial=0))
        reset_mismatch=int(np.count_nonzero(fc.reset!=src['detached_history']))
        # Intervene only on the exact published branch. Current-reconstruction discrepancy stays visible.
        fc.initial=src['initial'];fc.reset=src['detached_history'];fc.protected=src['protected']
        fc.backend='equal_columns';fc.workers=4
        inp,nominal_force,nominal_state=nominal_qp(plan,wbc,src)
        nominated,_=replay_wbc(saved,plan,nominal_state,nominal_force,t,wbc,base/f'{call:06d}-nominal')
        first={};winners={};scores={};candidate_forces={}
        for history in ('full','private_reset'):
            val=rank(fc,inp,plan['inertia'],CFG,nominal_force,solve,.0002,history)
            winner=val['winner'];winners[history]=winner;scores[history]=val['cost']
            u=val['corrected'] if winner else nominal_force
            x=val['qp_states'] if winner else nominal_state
            directory=base/f'{call:06d}-{history}'
            # Each new execution of this analysis retains its small WBC source records.
            if directory.exists():
                directory=base/f'{call:06d}-{history}-{time.time_ns()}'
            first[history],_=replay_wbc(saved,plan,x,u,t,wbc,directory)
            candidate_forces[history]=u[0]
        err=float(np.max(abs(first['full']-wbc['mapped_u'])))
        change=float(np.max(abs(first['private_reset']-first['full'])))
        ferror=float(np.max(abs(candidate_forces['full']-plan['forces'][0])))
        dforce=float(np.max(abs(candidate_forces['full']-candidate_forces['private_reset'])))
        physical_winner=targets[0]['recorded_selected']
        resolved=(change>err and not reset_mismatch and ('material' if winners['full'] else 'nominal')==physical_winner)
        solved.append(dict(call=call,publication_s=t,target_event_ids=[q['target_event_id'] for q in targets],
            associated_source_target_world_pairs=[q['matched'] for q in targets],world_state_recorded_reconstructed_max_error=replay_error,
            protected_reset_mask_mismatches=reset_mismatch,recorded_winner=physical_winner,winners=winners,objectives=scores,
            full_first_force_vs_recorded_N=ferror,private_vs_full_first_force_N=dforce,
            reproduced_full_mapped_vs_installed_rad=err,private_vs_full_mapped_rad=change,
            nominated_vs_full_first_mapped_rad=float(np.abs(nominated-first['full']).max()),
            distinct_mapped_above_same_state_reconstruction=bool(resolved)))
    res=dict(attempt=attempt,estimator=row['material'],case='A',task=row['task_mode'],level=row['level'],
        selection='First published P1 plan with an observed, previously released source, true cross-leg reload at the same cell, actual held footprint overlap, actual target event ID and a detached reset-eligible world key per destination event. All matching target IDs retained; plans de-duplicated.',
        distinct_target_events=len(entries),distinct_plans=len(solved),above_reconstruction=sum(x['distinct_mapped_above_same_state_reconstruction'] for x in solved),
        observations=solved,scope='True subsequent loads are from the installed full plan. A private counterfactual next physical outcome requires its own closed-loop episode.')
    temp=dest.with_suffix('.tmp')
    temp.write_text(json.dumps(res,indent=2,allow_nan=False)+'\n');temp.replace(dest)
    return {k:res[k] for k in ('attempt','distinct_target_events','distinct_plans','above_reconstruction')}

def first_actual(attempt):
    r12=ROOT/'results'/REL/'w57';raw=ROOT/'outputs'/REL/attempt
    dest=r12/'p/physical'/(attempt+'-first-selected.json')
    if dest.exists():return json.loads(dest.read_text())
    row=json.loads((r12/'attempts'/(attempt+'.json')).read_text())
    saved=Actual(raw,row['material']);logged=next(json.loads(line) for line in
        (saved.raw/'material-candidates.jsonl').open() if '"selected": "material"' in line)
    call=logged['call'];t=logged['time_s'];tick=round(t/.002);saved.state(t)
    with np.load(saved.raw/f'mpc-{call:06d}.npz') as z:plan=dict(z)
    with np.load(saved.raw/logged['data']) as z:src=dict(z)
    with np.load(saved.raw/f'control-{tick//500+1:05d}.npz') as z:wbc={k:z[k][tick%500] for k in z.files}
    inp,f,x=nominal_qp(plan,wbc,src)
    base=ROOT/'work/w57-p/first-action'/attempt
    nominal,base_s=replay_wbc(saved,plan,x,f,t,wbc,base/'nominal')
    actual,full_s=replay_wbc(saved,plan,plan['states'],plan['forces'],t,wbc,base/'installed')
    audit=dict(attempt=attempt,call=call,published_s=t,case=row['case'],estimator=row['material'],
        observer_time_s=logged['observer_time_s'],objectives=logged['original_objective'],
        first_force_change_N=float(np.max(np.abs(f[0]-plan['forces'][0]))),
        first_mapped_vs_nominal_rad=float(np.max(np.abs(actual-nominal))),
        saved_full_vs_reconstructed_mapped_rad=float(np.max(np.abs(actual-wbc['mapped_u']))),
        nominal_wbc_wall_s=base_s,installed_wbc_wall_s=full_s,selected=logged['selected'],
        actual_target_ids=[x['target_event_id'] for x in logged['target_events'] if x['within_material_horizon']])
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(audit,indent=2)+'\n');return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--first-selected',action='store_true');x=p.parse_args()
    print(first_actual(x.attempt) if x.first_selected else evidence(x.attempt))
