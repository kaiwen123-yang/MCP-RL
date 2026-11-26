"""Recover M3's sparse FP64 publication from completed measured packets before the legal action."""
from pathlib import Path
import json

import numpy as np
from s1_w59_model_material import W59MaterialObserver
from run_s1_w59_interventions import ROOT


def source_publications(states):
    pars=json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
    sources={}
    for s in states:
        source=Path(s['snapshot']).parents[1];sources.setdefault(source,[]).append(s)
    publications={}
    for source,group in sources.items():
        times={round(s['request_s']/.0002):s['snapshot_id'] for s in group}
        last=max(times);obs=W59MaterialObserver(pars,ROOT/'work/assets/B_random1000--ideal-potentials.npz')
        physics=iter(sorted(source.glob('physical-*.npz')))
        physical=np.empty((0,0));index=0
        for p in sorted(source.glob('material-[0-9]*.npz')):
            with np.load(p) as f:mat=f['foot']
            for row in mat:
                if index>=len(physical):
                    with np.load(next(physics)) as f:physical=f['physical']
                    index=0
                snap=physical[index];index+=1
                t0,t1=row[:2]
                if abs(snap[0]-t0)>1e-8 or abs(snap[1]-t1)>1e-8:
                    raise ValueError('Measured material and force clocks differ')
                bottom=row[6:18].reshape(4,3);velocity=row[18:30].reshape(4,3)
                obs.completed(float(t0),float(t1),bottom,velocity,snap[32:42:3])
                tick=round(t1/.0002)
                if tick%10==0:
                    obs.observe(float(t1),bottom,velocity)
                    if tick in times:publications[times[tick]]=obs.planning_publication()
                if tick>=last:break
            if tick>=last:break
    if len(publications)!=len(states):
        raise ValueError('Some registered legal instants lack their original measured packet')
    return publications
