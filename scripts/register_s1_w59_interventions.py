"""Lock disjoint legal-request IDs from unperturbed B3 before viewing any candidate outcome."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

LEGS=('FL','FR','RL','RR')
STRATA=('rigid_steady','early_touch_or_search','cross_foot_reload')
REQUESTS={
    'nominal1':[0.,.05,0.], 'nominal2':[0.,.05,0.],
    'due_minus':[-.05,.05,0.], 'due_plus':[.05,.05,0.],
    'ramp_minus':[0.,.04,0.], 'ramp_plus':[0.,.08,0.],
    'rho_minus':[0.,.05,-.10], 'rho_plus':[0.,.05,.10]}
SEEDS=(5901,5902,5903)


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def observed(raw,case):
    catalog=rows(raw/'w59-legal-requests/manifest.jsonl')
    events=rows(raw/'w55-f/contact-events.jsonl')
    material=rows(raw/'material-events.jsonl')
    cross=[v for v in material if v['type']=='revisit' and v['cross_leg'] and v['area_m2']>0 and v['previous_load_event_id']>=0 and v['unloaded_gap_s'] is not None]
    with np.load(raw/'trajectory.npz') as data:
        t=data['time']; phase=data['phase']
        feet=data['feet']; vel=data['qvel'][:,:3]
    times=[];normal=[]
    for part in sorted(raw.glob('physical-*.npz')):
        with np.load(part) as data:physical=data['physical']
        times.append(physical[:,0].copy());normal.append(physical[:,32:42:3].copy())
    ft=np.concatenate(times);fz=np.concatenate(normal);del physical
    out=[]
    for i,item in enumerate(catalog):
        tm=item['request_s']
        if i+1>=len(catalog) or tm+2.>t[-1]+1e-8:continue
        next_request=catalog[i+1]['request_s']
        if next_request-tm>1.:continue
        incoming=item['incoming']
        end=tm+.20
        # Classify by the ensuing undisturbed contact, never by perturbation outcomes.
        # Rigid cruise is considered steady when both feet make their landing without
        # a search or sustained (>=4 ms) loss of load after premature physical touch.
        bad_touch=False;premature=False
        for leg in incoming:
            lo,hi=np.searchsorted(ft,[tm+.1-1e-8,next_request+1e-8])
            loaded=fz[lo:hi,leg]>1.
            times=ft[lo:hi]
            starts=np.flatnonzero(loaded)
            if len(starts) and times[starts[0]]<end-.002:
                premature=True
                tail=loaded[starts[0]:]
                lost=np.r_[False,~tail,False]
                runs=np.diff(np.flatnonzero(lost[1:]!=lost[:-1]).reshape(-1,2),axis=1).ravel()
                bad_touch |= bool(len(runs) and runs.max()>=20)
        events_here=[v for v in events if tm+1e-8<v['time_s']<=next_request+1e-8]
        search=any(v['kind']=='touchdown_search' and any(LEGS[l] in v['legs'] for l in incoming)
                   for v in events_here)
        hits=[v for v in cross if tm+1e-8<v['time_s']<=next_request+1e-8 and v['leg'] in incoming]
        ti=np.flatnonzero((t>=tm-1e-8)&(t<=next_request+1e-8))
        rigid=not any((.8+.015811<=x<=2.2-.015811 and abs(y)<=.6-.015811)
                      for x,y in feet[ti][:,incoming,:2].reshape(-1,2))
        steady=bool(np.all(np.isin(phase[ti],[1,3])) and np.median(abs(vel[ti,0]))>=.10)
        if hits:label=STRATA[2]
        elif rigid and steady and not premature and not bad_touch:label=STRATA[0]
        elif search or bad_touch or premature:label=STRATA[1]
        else:label=None
        path=raw/'w59-legal-requests'/item['file']
        sid=f'{raw.name}/r{item["request_id"]:05d}'
        out.append(dict(snapshot_id=sid,case=case,request_id=item['request_id'],request_s=tm,
            next_request_s=next_request,incoming=incoming,snapshot=str(path.resolve()),
            sha256=item['sha256'],source_build=item['source_build'],stratum=label,
            source_undisturbed=dict(rigid=rigid,steady=steady,early=bool(premature),
                rebound=bool(bad_touch),search=bool(search),cross_events=[v['event_id'] for v in hits])))
    return out


def spaced(eligible, count, occupied, salt):
    # Greedy separation in baseline physical time after an ID-only seeded tie break.
    chosen=[]
    while len(chosen)<count:
        pool=[v for v in eligible if v['snapshot_id'] not in occupied]
        if not pool:break
        def key(v):
            own=[s['request_s'] for s in occupied.values() if s['case']==v['case']]
            tie=hashlib.sha256((salt+v['snapshot_id']).encode()).hexdigest()
            return (min(abs(v['request_s']-t) for t in own) if own else 0.,tie)
        one=max(pool,key=key)
        occupied[one['snapshot_id']]=one
        chosen.append(one)
    return chosen


def distinct_prefix_end(a,b):
    with np.load(a/'trajectory.npz') as fa, np.load(b/'trajectory.npz') as fb:
        n=min(len(fa['time']),len(fb['time']))
        ta,tb=fa['time'][:n],fb['time'][:n]
        if not np.allclose(ta,tb,rtol=0.,atol=1e-8):
            raise ValueError('Original B3 physical clocks have different grids')
        # Reject the physically indistinguishable startup prefix, regardless of case/filename.
        distinct=(np.max(abs(fa['qpos'][:n]-fb['qpos'][:n]),axis=1)>1e-6
                  )|(np.max(abs(fa['qvel'][:n]-fb['qvel'][:n]),axis=1)>1e-5)
        windows=np.convolve(distinct.astype(int),np.ones(250,dtype=int),'valid')
        full=np.flatnonzero(windows==250)
        return float(ta[full[0]+249]) if len(full) else float(ta[-1])


def register(a,b,output):
    start=distinct_prefix_end(a,b)
    sources={'A':observed(a,'A'),'B':observed(b,'B')}
    for case in sources:
        sources[case]=[r for r in sources[case] if r['request_s']>=start]
    # Choose the earliest separation with the full required baseline strata on both cases
    # and twelve later legal requests/case. Only unperturbed chronology participates.
    boundaries=sorted({round(z['request_s']+2.,6) for case in sources.values() for z in case})
    boundaries=[t for t in boundaries if all(
        all(sum(v['stratum']==k and v['request_s']+2.<=t+1e-8 for v in case)>=4 for k in STRATA)
        and sum(v['request_s']>=t-1e-8 for v in case)>=12 for case in sources.values())]
    cut=boundaries[0] if boundaries else 19.
    occupied={}; matrix={}; actors={}; missing=[]
    for role,case in (('development','A'),('sealed_admission','B')):
        matrix[role]=[]
        for stratum in STRATA:
            eligible=[v for v in sources[case] if v['stratum']==stratum and v['request_s']+2.<=cut+1e-8]
            selected=spaced(eligible,4,occupied,'W59/calibration/'+role+'/'+stratum)
            matrix[role]+=selected
            if len(selected)!=4:missing.append([role,stratum,len(selected),4])
    for seed in SEEDS:
        actors[seed]=[]
        for case in ('A','B'):
            selected=spaced([v for v in sources[case] if v['request_s']>=cut-1e-8],4,occupied,'W59/first-update/'+str(seed))
            actors[seed]+=selected
            if len(selected)!=4:missing.append(['first_update',seed,case,len(selected),4])
    reg=dict(schema='W59_locked_requests_v2_unperturbed_overlap',sources={k:v[0]['snapshot_id'].split('/')[0] if v else None for k,v in sources.items()},
        rule='Disjoint source cases A development, B sealed; fixed unperturbed class, 4 states/class/role. Farthest time on same trajectory with SHA256 ID ties. 8 other states per actor seed; 4 per case.',
        classification='Cross-foot positive-area reload first; rigid straight cruise with no measured early touch or rebound second; early touch or search otherwise. Search is retained on the rigid rows: every observed exchange contained at least one search event.',
        supersedes_unperturbed_selection='registry.json: no perturbed branch issued; giving any search event precedence filled even the flat no-early-touch bucket.',
        status='blocked_incomplete_strata' if missing else 'locked',missing=missing,
        same_clock_prefix_excluded_through_s=start,calibration_window_end_max_s=cut,
        first_update_window_start_min_s=cut,calibration_and_actor_windows_disjoint=True,
        eligible_counts={k:dict(Counter(v['stratum'] or 'unlabelled' for v in vs)) for k,vs in sources.items()},
        states=matrix,first_update=actors,signed=REQUESTS,duration_s=2.,candidate_outcomes_seen=0,
        sources_sha256={k:hashlib.sha256((v/'w59-legal-requests/manifest.jsonl').read_bytes()).hexdigest()
                        for k,v in [('A',a),('B',b)]})
    if output.exists():
        if json.loads(output.read_text())!=json.loads(json.dumps(reg)):
            raise ValueError('An existing registration cannot be revised after the fact')
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as f:json.dump(reg,f,indent=2);f.write('\n')
    return reg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a',required=True,type=Path);p.add_argument('--b',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    r=register(args.a,args.b,args.output)
    print(json.dumps({k:r[k] for k in ('status','missing','eligible_counts')}))


if __name__=='__main__':main()
