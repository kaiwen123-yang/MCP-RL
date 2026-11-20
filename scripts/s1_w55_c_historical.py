"""Replay prespecified original R2 conditions for all six loading families."""
import csv
import json
import resource
import sys
import time
from pathlib import Path

import jax
import numpy as np
from s1_w55_c_data import RAW,COMPACT,ROOT,WEIGHTS,write
from tm02.memory_data import _final_histories
from tm02.reference import Reference
from tm02.material import load_weights
from tm02.evaluate import integrate
from tm02.memory_evaluate import local_recovery_quantities

jax.config.update('jax_enable_x64',True)
SAVED=ROOT.parent/'MCP-RL-w54/results/terrain_memory/reproduction/material/tm02-memory-study-20261023/final/primary-history-metrics.csv'
LARGE=RAW/'historical'
SELECTED=COMPACT/'historical-selection.json'


def settings():
    return [json.loads((ROOT/'experiments/tm02'/f).read_text()) for f in
            ('r2-memory-study.json','r2-multihistory.json','r2-revisit.json')]


def prepare():
    t0=time.time();c0=time.process_time()
    with SAVED.open() as f:
        original=[r for r in csv.DictReader(f) if r['model_id']=='B_random1000--ideal']
    wanted={}
    for fam in sorted({r['family'] for r in original}):
        rr=[r for r in original if r['family']==fam]
        field='stress_RMSE_MPa' if rr[0]['mode']=='strain_control' else 'full_history_strain_RMSE'
        ordered=sorted(rr,key=lambda r:float(r[field]))
        selected={s['condition_id'] for s in (rr[0],ordered[len(rr)//2],ordered[-1])}
        for r in rr:
            if r['condition_id'] in selected:
                wanted[r['history_id']]=dict(family=fam,metric=field,archived_frozen_value=float(r[field]),condition=r['condition_id'])
    study,base,revisit=settings()
    law=Reference(base['reference_material']['parameters']);LARGE.mkdir(exist_ok=True)
    for meta,t,boundary,_ in _final_histories(study,base,revisit):
        hid=meta['history_id']
        if hid not in wanted:continue
        stress=meta['mode']=='stress_control'
        ref=law.path(t,boundary,base['numerics'],stress_control=stress)
        np.savez_compressed(LARGE/(hid+'.npz'),time=t,boundary=boundary,target=ref['strain' if stress else 'stress'])
        wanted[hid].update(mode=meta['mode'],steps=len(t)-1)
    write(SELECTED,dict(rule='Per original family: first archived condition, median baseline-error history condition, worst baseline-error history condition; both F6 candidates are retained. Frozen W54-independent generator, full histories, physical tensor shear convention. Other 700-history catalog members are outside this check.',
        original_table=str(SAVED),target_paths=wanted,start_unix=t0,end_unix=time.time(),cpu_s=time.process_time()-c0))
    print('Prepared',len(wanted),'original complete histories',sum(r['steps'] for r in wanted.values()),flush=True)


def evaluate():
    t0=time.time();c0=time.process_time()
    protocol=json.loads(SELECTED.read_text());study,base,_=settings()
    filename=RAW/'W55-short-load-ideal-potentials.npz'
    w,norm=load_weights(filename)
    results={}
    for hid,record in protocol['target_paths'].items():
        with np.load(LARGE/(hid+'.npz')) as f:
            tt=f['time'];cc=f['boundary']
            stress=record['mode']=='stress_control'
            result,failure=integrate(w,norm,tt,cc,base['numerics'],stress_control=stress)
            output=result['strain' if stress else 'stress']
            rmse=float(np.sqrt(np.mean((output[1:]-f['target'][1:])**2))) if failure is None else None
        np.savez_compressed(LARGE/(hid+'-candidate.npz'),**result)
        results[hid]=dict(**record,candidate_value=rmse,failure=failure,
                           candidate_over_frozen=rmse/record['archived_frozen_value'] if rmse is not None else None)
        print(hid,'rmse',rmse,'failure',failure,flush=True)
    ids=dict(arm='B',cell='ideal',condition='ideal',noise_realization=None,seed=0)
    entries=[dict(**ids,model_id=name,weights=str(p)) for name,p in [('frozen',WEIGHTS),('candidate',filename)]]
    recs=local_recovery_quantities(entries,base['reference_material']['parameters'],LARGE/'poles',LARGE/'poles')
    write(COMPACT/'historical-result.json',dict(evaluation_scope=protocol['rule'],target_paths=results,zero_state=recs,
          start_unix=t0,end_unix=time.time(),cpu_s=time.process_time()-c0))
    print('Historical assessment complete.',flush=True)


if __name__=='__main__':
    {'prepare':prepare,'evaluate':evaluate}[sys.argv[1]]()
