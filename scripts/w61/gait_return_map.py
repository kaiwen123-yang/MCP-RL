"""Estimate return changes between measured right-front unloading sections per chart."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

SCALE=np.array([.10,.10,.08,.08])


def section_rows(run,case):
    with np.load(run/'trajectory.npz') as tr:
        t,v,xyz,rpy,force,ref=(tr[k] for k in ('time','qvel','qpos','body_rpy','force','requested_velocity'))
    with (run/'w55-f/contact-events.jsonl').open() as stream:
        events=[json.loads(s) for s in stream]
    sections=[];missing=0
    for e in events:
        if e['kind']!='liftoff_committed' or 'FR' not in e['legs']:
            continue
        a=np.searchsorted(t,e['time_s']-1.e-9)
        b=np.searchsorted(t,e['time_s']+.044)
        gap=np.flatnonzero(force[a:b,1,2]<=1.)
        if not len(gap):
            missing+=1;continue
        k=a+gap[0]
        R=Rotation.from_matrix(Rotation.from_euler('z',rpy[k,2]).as_matrix())
        body=R.inv().apply(np.r_[v[k,:2]-ref[k,:2],0.])
        sections.append(dict(time=float(t[k]),chart=e.get('orbit_chart','trot'),
            state=np.r_[body[:2],rpy[k,:2]].tolist()))
    pairs=[]
    for s1,s2 in zip(sections,sections[1:]):
        if s1['chart']!=s2['chart']:continue
        pulse=case['push_start_s']
        forced=(s1['time']<=pulse+case['push_duration_s'] and s2['time']>=pulse)
        speed_change=s1['time']<=case['switch_s']<=s2['time']
        pairs.append(dict(chart=s1['chart'],forced=forced or speed_change,from_state=s1['state'],to_state=s2['state']))
    return dict(observed=len(sections),unconfirmed=missing,sections=sections,pairs=pairs)


def fit(training):
    x=np.asarray([p['from_state'] for p in training])/SCALE
    y=np.asarray([p['to_state'] for p in training])/SCALE
    X=np.column_stack((np.ones(len(x)),x));rank=np.linalg.matrix_rank(X)
    if rank<5:
        return None, int(rank)
    return np.linalg.solve(X.T@X+np.diag([0.,.05,.05,.05,.05]),X.T@y),int(rank)


def audit(paths,spec):
    groups={}
    for p in paths:
        result=json.loads((p/'result.json').read_text());case=result['case']
        method=p.name.removeprefix('decoder-v1-'+case+'-')
        groups.setdefault(method,{})[case]=section_rows(p,spec['cases'][case])
    report={}
    for method,runs in groups.items():
        report[method]={}
        for chart in ('trot','crawl'):
            data={case:[p for p in r['pairs'] if p['chart']==chart and not p['forced']] for case,r in runs.items()}
            pooled=[p for v in data.values() for p in v];matrix,rank=fit(pooled) if pooled else (None,0)
            blocks={}
            for case,held in data.items():
                train=[p for name,other in data.items() if name!=case for p in other]
                beta,r=fit(train) if train else (None,0)
                if beta is None or not held:
                    blocks[case]=dict(train=len(train),test=len(held),rank=r,identifiable=False);continue
                x=np.asarray([p['from_state'] for p in held])/SCALE;y=np.asarray([p['to_state'] for p in held])/SCALE
                pred=np.column_stack((np.ones(len(x)),x))@beta
                blocks[case]=dict(train=len(train),test=len(held),rank=r,identifiable=True,
                    normalized_RMSE=float(np.sqrt(np.mean((pred-y)**2))),
                    persistence_RMSE=float(np.sqrt(np.mean((x-y)**2))))
            report[method][chart]=dict(pairs=len(pooled),rank=rank,fit=matrix.tolist() if matrix is not None else None,
                fit_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(matrix[1:].T)))) if matrix is not None else None,
                held_out_scenario=blocks)
    return dict(definition='First observed FR Fz<=1 N within 44 ms after its committed departure. Same-chart returns only; exclude returns across the externally programmed pulse or speed switch.',
            units=['body-frame velocity error x m/s','body-frame velocity error y m/s','roll rad','pitch rad'],
            scale=SCALE.tolist(),interpretation='Descriptive ridge fit on sampled motions. Spectral radius concerns the fitted local affine map only.',
            comparison=report,scenarios=groups)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();spec=json.loads(a.contract.read_text())
    runs=[a.root/f'decoder-v1-{c}-{m}' for c in spec['fresh_query'] for m in ['fixed','library','cpg','bridge','6111','6112','6113']]
    for run in runs:
        while not (run/'result.json').exists():time.sleep(10)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(audit(runs,spec),indent=2,allow_nan=False)+'\n')
