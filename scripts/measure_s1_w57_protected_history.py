"""All actual material replans of P/A E30, one protected-history comparison per nonempty intervention."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
from analyze_s1_w57_planner_outcomes import ROOT,REL,q
from replay_s1_w57_confirmed_history_actions import Actual,nominal_inputs,nominal_qp
from replay_s1_w57_force_execution import replay_wbc
from qualify_s1_w57_planning import CFG,solve
from s1_w57_planning import Forecast,footprint,rank
from s1_w60_cpu_observer import grow_cells


@lru_cache(maxsize=2)
def control(path):
    keys=('time','planned_contact','observed_contact','confirmed_contact','load_bearing_contact','event_mode',
          'transfer_support_pair','transfer_fraction','feet_position_reference','feet_velocity_reference',
          'feet_acceleration_reference','J_contact_force','J_rolling','contact_force_point',
          'measured_grf','mode','mapped_u','stance_anchor')
    with np.load(path) as z:return {k:z[k] for k in keys}


def one(item):
    raw,record,end=item
    call=record['call'];t=record['time_s'];tick=round(t/.002)
    base=dict(call=call,time_s=t,following_physics=t<end-1.e-7,recorded_winner=record['selected'],
              detached=record['detached_history'])
    if not base['following_physics']:
        return dict(**base,private_winner=None,comparison='terminal_no_following_command')
    if not base['detached']:
        return dict(**base,private_winner=record['selected'],comparison='identical_empty_intervention',
                    private_first_force_difference_N=0.,private_first_mapped_difference_rad=0.)
    wall=time.perf_counter();cpu=time.process_time();folder=raw/'w55-f'
    with np.load(folder/record['data']) as z:
        src={k:z[k] for k in ('world_keys','initial','segment_grid_s','xy','active','detached_history','protected','nominal_force')}
    with np.load(folder/f'mpc-{call:06d}.npz') as z:plan=dict(z)
    row={k:a[tick%500] for k,a in control(folder/f'control-{tick//500+1:05d}.npz').items()}
    saved=Actual(raw,'M1');obs=saved.obs;g=obs.ground
    clock=round(t/.0002)-1;s=clock//5000
    bottom=saved.material(s)[clock%5000,6:18].reshape(4,3).copy()
    obs.time=t;obs.packet_start=t-.002;obs.packet_last=t-.0002;obs.published_bottom=bottom
    keys=[tuple(int(i) for i in k) for k in src['world_keys']]
    grow_cells(g,keys);g.e[:]=src['initial'][:,:2];g.q[:]=src['initial'][:,2:4];g.stress[:]=src['initial'][:,4:]
    needed={k for leg in np.flatnonzero(row['mode']==3) for k in footprint(g,bottom[leg,:2])}
    missing=sorted(needed-set(keys))
    fc=Forecast(g,plan['times'],src['segment_grid_s'],src['xy'],src['active'],bottom,row['observed_contact'],plan['foot_center_plan'][:-1,:,:2])
    if keys!=fc.keys:raise ValueError(f'World-key mismatch in plan {call}')
    fc.reset=src['detached_history'];fc.protected=src['protected'];fc.backend='equal_columns'
    # Four independent archived publications run concurrently; each uses one native worker.
    inputs=nominal_inputs(plan,row)
    val=rank(fc,inputs,plan['inertia'],CFG,src['nominal_force'],solve,.0002,'private_reset')
    winner='material' if val['winner'] else 'nominal'
    private=val['corrected'] if val['winner'] else src['nominal_force']
    delta=float(np.abs(private[0]-plan['forces'][0]).max())
    measured=dict(private_first_mapped_difference_rad=None,full_WBC_reconstruction_error_rad=None,
                  above_same_state_WBC_error=None)
    if not missing:
        with tempfile.TemporaryDirectory(dir=ROOT/'work/w57-p') as tmp:
            tmp=Path(tmp)
            full,_=replay_wbc(saved,plan,plan['states'],plan['forces'],t,row,tmp/'actual')
            if winner=='nominal' and record['selected']=='nominal':
                altered=full
            else:
                state=(val['qp_states'] if val['winner'] else nominal_qp(plan,row,src)[2])
                altered,_=replay_wbc(saved,plan,state,private,t,row,tmp/'protected')
        error=float(np.abs(full-row['mapped_u']).max());effect=float(np.abs(altered-full).max())
        measured=dict(private_first_mapped_difference_rad=effect,full_WBC_reconstruction_error_rad=error,
                      above_same_state_WBC_error=bool(effect>error))
    return dict(**base,private_winner=winner,comparison='computed_from_actual_publication',original_objective=record['original_objective'],
                private_objective=val['cost'],private_first_force_difference_N=delta,missing_current_footprint_keys=missing,
                cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall,**measured)


def run(attempt):
    raw=ROOT/'outputs'/REL/attempt;r=ROOT/'results'/REL/'w57';out=r/'p/physical'
    receipt=json.loads((r/'attempts'/(attempt+'.json')).read_text())
    if not(receipt['arm']=='P' and receipt['case']=='A' and receipt['material']=='M1' and receipt['level']=='E30'
           and receipt['task_mode']=='original' and receipt['settings']['planner_history']=='full'):
        raise ValueError('Declared census is the full original M1 P/A E30 case')
    end=receipt['simulation_s']
    allrows=[json.loads(l) for l in (raw/'w55-f/material-candidates.jsonl').open()]
    selected=[{k:x[k] for k in ('call','time_s','data','selected','detached_history','original_objective')}
              for x in allrows if x.get('data')]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(one,((raw,v,end) for v in selected),chunksize=3))
    active=[x for x in rows if x['detached'] and x['following_physics']]
    eligible=[x for x in active if x['private_first_mapped_difference_rad'] is not None]
    switches=[x for x in active if x['recorded_winner']!=x['private_winner']]
    child=resource.getrusage(resource.RUSAGE_CHILDREN)
    res=dict(attempt=attempt,original_replans=len(allrows),material_candidate_records=len(rows),
        all_material_with_following_physics=sum(x['following_physics'] for x in rows),
        empty_interventions_with_following_physics=sum(not x['detached'] and x['following_physics'] for x in rows),
        nonempty_with_following_physics=len(active),rank_switches=len(switches),
        switch_directions=dict(Counter(x['recorded_winner']+' to '+x['private_winner'] for x in switches)),
        first_force_change_N=q([x['private_first_force_difference_N'] for x in active]),
        WBC_covered=len(eligible),WBC_coverage_missing=len(active)-len(eligible),
        mapped_above_full_numerical_reconstruction=sum(x['above_same_state_WBC_error'] for x in eligible),
        first_mapped_difference_rad=q([x['private_first_mapped_difference_rad'] for x in eligible]),
        WBC_reconstruction_error_rad=q([x['full_WBC_reconstruction_error_rad'] for x in eligible]),
        switches=switches,wall_s=time.perf_counter()-start,child_core_s=child.ru_utime+child.ru_stime,
        scope='All published material replans of this single real episode. Reset detached queried cells once; retain current complete footprints, WBC and actor. Empty interventions share the recorded result by identity. Pure physics-replay scratch contains regenerated WBC matrices only. Observed next-contact outcomes still belong to the full actual plant trajectory. No new robot transitions.')
    with gzip.open(out/(attempt+'-all-protected-plans.jsonl.gz'),'wt') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    (out/(attempt+'-all-protected.json')).write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in res.items() if k not in ('scope','switches')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt');run(p.parse_args().attempt)
