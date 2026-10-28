#!/usr/bin/env python3
"""Causal observer replay on an existing complete recorded task, no physics."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from s1_go2_r10_observer import MaterialObserver


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    parameters=json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
    weights=os.environ['MPC_PANN_WEIGHTS']
    obs=MaterialObserver(parameters,weights)
    data=np.load(a.trajectory)
    records=[];query_time=-1.;begin=time.perf_counter()
    for i,t in enumerate(data['time']):
        bottom=data['feet'][i].copy();bottom[:,2]-=.022
        obs.observe(t,bottom,data['feet_velocity'][i])
        if int(t/.5)>query_time:
            # Replay's diagnostic query is the currently observed footprint;
            # policy integration uses the latest completed plan at t+.20 s.
            feature,mask=obs.recovery(bottom[:,:2])
            records.append(dict(time_s=float(t),query_xy_m=bottom[:,:2].tolist(),
                soft=mask.tolist(),recovery_m=feature.tolist()))
            query_time=int(t/.5)
            print(json.dumps(dict(time_s=float(t),elapsed_wall_s=time.perf_counter()-begin,
                cells=len(obs.ground.e))),flush=True)
    a.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.out/'observer-final.npz',**obs.states())
    (a.out/'recovery.json').write_text(json.dumps(records,indent=2)+'\n')
    result=dict(kind='causal_record_replay_not_closed_loop_or_training',
        trajectory=str(a.trajectory),observed_through_s=float(data['time'][-1]),
        new_physical_steps=0,training_transitions=0,**obs.timing())
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
