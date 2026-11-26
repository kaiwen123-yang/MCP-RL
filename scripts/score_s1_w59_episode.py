"""Score W59 walking phases, actual strides and every source load of an original route."""
import argparse
import json
from pathlib import Path

import numpy as np

from publish_s1_go2_w54 import write
from score_s1_w58_gait import score_episode
from s1_w59_contact import LEGS, score
from s1_w59_geometry import original_route
from s1_w59_lease import ROOT

REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
STATE=ROOT/'results'/REL/'w59'
WORK=ROOT/'work/w59/gait'


def run(name):
    raw=ROOT/'outputs'/REL/name
    r=STATE/'episodes'/(name+'.json')
    terminal=json.loads(r.read_text())
    saved=WORK/'phase-evidence'/(name+'.json')
    if not saved.is_file():
        saved.parent.mkdir(parents=True,exist_ok=True)
        detail=score_episode(raw,r,saved)
    else:
        detail=json.loads(saved.read_text())
    with np.load(raw/'trajectory.npz') as z:
        t=z['time'];legs=z['actual_grf'][:,:,2]>1.;phase=z['phase'];feet=z['feet']
    with (raw/'w55-f/contact-events.jsonl').open() as f:
        events=[json.loads(line) for line in f]
    clearance={leg:[] for leg in LEGS}
    for e in events:
        if e['kind']!='liftoff_committed':
            continue
        lo,hi=np.searchsorted(t,[e['time_s']-1.e-9,e['fixed_touchdown_s']-1.e-9])
        if hi<=lo or e['fixed_touchdown_s']>t[-1]+1.e-8:
            continue
        for leg in e['legs']:
            i=LEGS.index(leg)
            clearance[leg].append(float(np.max(feet[lo:hi,i,2])-.022))
    dt=np.diff(t);m=np.isin(phase[:-1],(1,3))
    duty=(dt[m,None]*legs[:-1][m]).sum(axis=0)/dt[m].sum() if m.any() else np.full(4,np.nan)
    obj=dict(attempt=name,source=str(saved.relative_to(ROOT)),phase_names=list(detail['groups']),
        phases={k:dict(swings=v['landings']['committed_leg_swings'], bearing_fraction=v['landings']['bearing_fraction'],
            mean_off_s_per_committed_swing=v['landings']['primary_observed_off_s_per_committed_swing'],
            first20ms_peak_N=v['landings']['first20ms_peak_N'], periods=v['landings']['per_leg'],
            motion_positive_CoT=v['positive_CoT'], fewer_than_two_supports_s=v['fewer_than_two_supports_s'])
            for k,v in detail['groups'].items()},
        loaded_duty_during_translation={leg:float(duty[i]) if m.any() else None for i,leg in enumerate(LEGS)},
        absolute_left_right_duty_difference=([float(abs(duty[0]-duty[1])),float(abs(duty[2]-duty[3]))] if m.any() else None),
        completed_swing_peak_clearance_m={leg:dict(n=len(v),median=float(np.median(v)) if v else None,min=float(min(v)) if v else None) for leg,v in clearance.items()})
    write(STATE/'gait/phases'/(name+'.json'),obj)
    contact=STATE/'gait/new'/(name+'.json')
    if not contact.exists():
        write(contact,score(raw,terminal))
    if 'original' in name:
        route=WORK/'new-routes'/(name+'.json')
        route.parent.mkdir(parents=True,exist_ok=True)
        if not route.is_file():
            original_route(raw,terminal,route)
        write(STATE/'geometry/qualification'/(name+'.json'),json.loads(route.read_text()))
    print(name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt',nargs='+')
    for n in p.parse_args().attempt:
        run(n)
