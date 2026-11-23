"""Separate source-state, unloaded-propagation and contact-law error on W56 development events."""
import json
import time

import numpy as np

from collect_s1_w58_material import BG, OUT, ROOT, SPLITS, WEIGHTS, read_gz
from score_s1_w57_material import batch, material_response, truth_at
from s1_w55_c_data import DT, R12, write
from s1_w60_cpu_observer import CPUColumns


def moments(x):
    a=np.asarray(x,dtype=float)
    return dict(n=len(a), bias=float(np.mean(a)), mae=float(np.mean(np.abs(a))),
                abs_p95=float(np.quantile(np.abs(a),.95)))


def decompose(case, episode, law):
    archive=ROOT.parent/'MCP-RL-w57'
    base=ROOT.parent/'MCP-RL-w56-r60/outputs'/R12/episode
    blocks=sorted(base.glob('material-[0-9]*.npz'))
    foot=np.concatenate([np.load(p)['foot'] for p in blocks])
    starts=read_gz(archive/'outputs'/R12/'w57/m'/f'{case}-M0-starts.json.gz')
    old=read_gz(archive/'results'/R12/'w57/m'/f'{case}-held-forecasts.json.gz')
    original={(v['mode'],v['event_id']):v['errors'] for v in old['release_to_reload']}
    rows=[q for q in starts if 'release_state' in q and q['event']['cross_leg']]
    ids=[r['event_id'] for r in rows]
    queries=[dict(event_id=q['event_id'],target=q['release_publication_tick'],keys=[q['key']]) for q in rows]
    initial_truth=truth_at(blocks,queries)
    lengths=np.array([q['event_tick']-q['release_publication_tick'] for q in rows],dtype=np.int32)
    x0=np.array([initial_truth[k][0,:4] for k in ids])
    # Truth enters only these offline counterfactual initializations.
    xp,resources=batch(law.model,x0,np.zeros(1),lengths,True,
        f'{case}/release-law attribution from contemporaneous plant state',
        np.unique([int(np.argmin(lengths)),int(np.argmax(lengths))]))
    decomp={'M0':[],'M1':[]}
    for i,q in enumerate(rows):
        e=q['event']
        xt=np.array([e['e_before_step']+e['q_before_step']])
        k=q['event_tick']
        z=foot[k,6+3*q['leg']+2]
        area=np.array([e['area_m2']])
        actual=e['normal_N']
        propagated=material_response(law,xp[i:i+1,:4],z,area)[0]
        exact_prestate=material_response(law,xt,z,area)[0]
        force_free=propagated-exact_prestate
        force_contact=exact_prestate-actual
        q_free=xp[i,2]-xt[0,2]
        for mode in decomp:
            old_error=original[mode,e['event_id']]
            total=old_error['next_cell_force_N']
            source=total-(propagated-actual)
            decomp[mode].append(dict(event_id=e['event_id'],gap_s=float(lengths[i]*DT),
                force_error_N=total,initial_state_component_N=source,
                free_evolution_component_N=force_free,final_contact_component_N=force_contact,
                exact_source_remaining_force_error_N=propagated-actual,
                q11_error=old_error['preload_internal_q11'],
                initial_state_component_q11=old_error['preload_internal_q11']-q_free,
                free_evolution_component_q11=q_free))
    summary={}
    for mode,vals in decomp.items():
        summary[mode]={k:moments([v[k] for v in vals]) for k in vals[0] if k not in ('event_id','gap_s')}
        summary[mode]['opposite_sign_initial_vs_free_force']=sum(
            v['initial_state_component_N']*v['free_evolution_component_N']<0 for v in vals)
    return dict(episode=episode,events=decomp,summary=summary,resources=resources)


if __name__=='__main__':
    wall,cpu=time.time(),time.process_time()
    law=CPUColumns(WEIGHTS)
    data=[decompose(case,episode,law) for case,episode in zip(('a','b'),SPLITS['development'])]
    write(OUT/'development-free-propagation.json',dict(by_case=data,
        decomposition='F(observer at release) - F(truth at release) after common learned free propagation; '
          'then F(truth at release propagated by the learned law) - F(truth at reload); '
          'then F(truth at reload) - measured cell force. Force evaluations share the learned contact solve, '
          'actual next sole height and area. Signed components sum per event. Their absolute errors need not sum.',
        scope='Diagnostics on development episodes using independent plant states offline.',
        wall_s=time.time()-wall,cpu_core_s=time.process_time()-cpu, **BG))
