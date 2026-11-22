"""Measure true unloaded gaps and the reference-law free curve before task selection."""
import json
from pathlib import Path

import numpy as np

from s1_w55_c_data import make_reference
from s1_w60_cpu_observer import ROOT

R12 = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
RAW = ROOT.parent/'MCP-RL-w56-r60'/'outputs'/R12
OUT = ROOT/'results'/R12/'w57/m'


def norms(q):
    return np.linalg.norm(q*np.array([1.,np.sqrt(2.)]), axis=-1)


def main():
    result = {'criterion':'same world cell, exact preceding load event, release q tensor norm >1e-8',
              'state':'plant known only in this offline design calculation', 'cases':{}}
    for case in ('a','b'):
        d, = RAW.glob(f'job-*-f-{case}-r60-*-E60-attempt01')
        released, rows = {}, []
        for line in (d/'material-events.jsonl').open():
            e = json.loads(line)
            key = tuple(e['world_key'])
            if e['type']=='release':
                released[key] = e
            elif e['type']=='revisit' and key in released:
                r = released.pop(key)
                if r['previous_load_event_id']==e['previous_load_event_id'] and norms(r['q'])>1e-8:
                    gap = e['time_s']-r['time_s']
                    rows.append((gap,norms(e['q_before_step'])/norms(r['q']),e['cross_leg'],r))
        groups = {}
        for label, condition in [('cross_foot_under_3s',lambda r:r[0]<3 and r[2]),
                                 ('unloaded_at_least_6s',lambda r:r[0]>=6)]:
            selected = np.array([list(row[:2]) for row in rows if condition(row)])
            groups[label] = {'n':len(selected),'gap_s_and_remaining_q_norm_fraction_min_p50_p95_max':
                             np.quantile(selected,[0,.5,.95,1],axis=0).tolist() if len(selected) else None}
        # The full original free law is integrated from unique real release states.
        start = {r['event_id']:r for _,_,cross,r in rows if cross}
        e = np.array([v['e'] for v in start.values()])
        q = np.array([v['q'] for v in start.values()])
        initial = norms(q)
        law = make_reference()
        curve = {}
        for j in range(1,3001):
            e,q,_,_ = law.step(e,q,np.zeros(len(q)),.002,np.zeros(len(q),bool))
            if j in (100,250,500,600,1000,2000,3000):
                curve[f'{j*.002:.3f}'] = np.quantile(norms(q)/initial,[0,.5,.95,1]).tolist()
        result['cases'][case] = dict(episode=d.name,groups=groups,curve_source_release_count=len(start),
             actual_release_reference_free_q_norm_fraction_min_p50_p95_max=curve)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'recovery-design.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
