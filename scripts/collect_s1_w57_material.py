"""Replay complete measured A/B episodes. Save causal forecasts' starting states only."""
import argparse
from collections import defaultdict
import gzip
import json
import os
import resource
import time

import numpy as np

from s1_w57_material import W57MaterialObserver
from s1_w60_cpu_observer import ROOT, CPUWindowObserver
from s1_w55_c_data import R12, DT, WEIGHTS

RAW = ROOT.parent/'MCP-RL-w56-r60'/'outputs'/R12
OUT = ROOT/'results'/R12/'w57/m'
ART = ROOT/'outputs'/R12/'w57/m'
PARAMETERS = json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']


def geometry_keys(xy):
    half = np.sqrt(.001)/2
    if not (.8+half<=xy[0]<=2.2-half and abs(xy[1])<=.6-half):
        return []
    lo, hi = np.floor((xy-half)/.02).astype(int), np.floor((xy+half)/.02).astype(int)
    return [(ix,iy) for ix in range(lo[0],hi[0]+1) for iy in range(lo[1],hi[1]+1)
            if np.all(np.minimum(xy+half,(np.array([ix,iy])+1)*.02)>
                      np.maximum(xy-half,np.array([ix,iy])*.02))]


def load_episode(case):
    base, = RAW.glob(f'job-*-f-{case}-r60-*-E60-attempt01')
    blocks = sorted(base.glob('material-[0-9]*.npz'))
    foot = np.concatenate([np.load(block)['foot'] for block in blocks])
    return base, blocks, foot


def targets(base, foot):
    bottom = foot[:,6:18].reshape(-1,4,3)
    soft_fz = foot[:,32:42:3]
    onset = [np.flatnonzero(np.diff(np.r_[False,soft_fz[:,leg]>0].astype(int))==1) for leg in range(4)]
    prev_release = {}
    grouped = {}
    eligible = defaultdict(int)
    for line in (base/'material-events.jsonl').open():
        event = json.loads(line)
        tick, leg, key = round(event['time_s']/DT)-1, event['leg'], tuple(event['world_key'])
        release = prev_release.get(key)
        if release is not None and release['previous_load_event_id']!=event.get('previous_load_event_id'):
            release = None
        if event['type']=='release':
            family = 'release'
            prev_release[key] = event
        else:
            prev_release.pop(key,None)
            family = ('first_load' if event['type']=='load' else
                      'long_recovery' if release and event['time_s']-release['time_s']>=6 else
                      'cross_foot_revisit' if event['cross_leg'] else 'same_foot_reload')
        start = (tick//10)*10
        target = start+100
        if target>=len(foot):
            continue
        pos = onset[leg].searchsorted(tick,side='right')-1
        if pos<0:
            continue
        interval = int(onset[leg][pos])
        group = leg, interval, family
        area = event.get('area_m2',0.) or event.get('area_before_m2',0.)
        eligible[family]+=1
        prior = grouped.get(group)
        if prior and (prior['event_tick']<tick or prior['area']>=area):
            continue
        keys = sorted({key, *geometry_keys(bottom[target,leg,:2])})
        grouped[group] = dict(event_id=event['event_id'],event_tick=tick,family=family,leg=leg,
            foot_interval_onset=interval,area=area,key=key,start=start,target=target,keys=keys,
            event=event,release=release if family not in ('release','first_load') else None)
    return list(grouped.values()),dict(eligible)


def state_at(g, keys):
    rows = []
    for key in keys:
        i = g.ids.get(tuple(key))
        rows.append(np.r_[g.e[i],g.q[i],g.stress[i]].tolist() if i is not None else [0.]*6)
    return rows


def dump(path, data):
    with gzip.open(path,'wt',encoding='utf-8') as fp:
        json.dump(data,fp,allow_nan=False,separators=(',',':'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case',choices=('a','b'))
    p.add_argument('mode', choices=('M0','M1'))
    a = p.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    tag=f'{a.case}-{a.mode}'
    wall0, cpu0 = time.time(), time.process_time()
    base,blocks,foot = load_episode(a.case)
    rows,eligible = targets(base,foot)
    sources, release = defaultdict(list), defaultdict(list)
    for j,q in enumerate(rows):
        sources[q['start']].append(j)
        if q['release']:
            end_tick=round(q['release']['time_s']/DT)
            pub=((end_tick+9)//10)*10
            if pub<=q['event_tick']:
                q['release_publication_tick']=pub
                release[pub].append(j)
    m = W57MaterialObserver(PARAMETERS,WEIGHTS,a.mode)
    legacy = CPUWindowObserver(PARAMETERS,WEIGHTS) if a.mode=='M0' else None
    parity = dict(e=0.,q=0.,stress_MPa=0.,force_N=0.,area_m2=0.,max_time_s=0.,different_loaded=0,
                  different_world_keys=0)
    bins = defaultdict(lambda:np.zeros(4,dtype=np.int64))
    sums = np.zeros(4)
    peaks = np.zeros(2)
    for j in range(0,len(foot),10):
        for i in sources.get(j,[]):
            record=rows[i]
            record['observed_state']=state_at(m.ground,record['keys'])
            record['publication_id']=m.windows
            record['publication_s']=m.time
        for i in release.get(j,[]):
            record=rows[i]
            record['release_state']=state_at(m.ground,[record['key']])[0]
            record['source_observed_history']=m.history(record['key'])
        for row in foot[j:j+10]:
            pos,vel=row[6:18].reshape(4,3), row[18:30].reshape(4,3)
            m.completed(row[0],row[1],pos,vel,row[32:42:3])
            if legacy is not None:
                legacy.completed(row[0],row[1],pos,vel)
        t=foot[j+9,1]
        m.observe(t,None,None)
        if legacy is not None:
            legacy.observe(t,None,None)
            for k,v in [('e',m.ground.e-legacy.ground.e),('q',m.ground.q-legacy.ground.q),
                        ('stress_MPa',m.ground.stress-legacy.ground.stress),
                        ('force_N',m.last_forces-legacy.last_forces),('area_m2',m.last_areas-legacy.last_areas)]:
                parity[k]=max(parity[k],float(np.max(np.abs(v),initial=0.)))
            parity['max_time_s']=max(parity['max_time_s'],abs(m.time-legacy.time))
            parity['different_loaded']+=int(np.count_nonzero(m.ground.loaded!=legacy.ground.loaded))
            parity['different_world_keys']+=int(m.ground.ids!=legacy.ground.ids)
        valid=m.last_areas[-1]>0
        cor=m.last_correction
        observed=foot[j+9,32:42:3]
        bins['soft_foot_publications']+=valid
        bins['positive_measured']+=valid & (observed>0)
        bins['selected_nonzero']+=cor[:,4]!=0
        bins['qualified_common_columns']+=cor[:,6]>0
        bins['inadmissible_candidates']+=cor[:,7].astype(int)
        for k,value in enumerate((np.abs(cor[valid,0]-observed[valid]).sum(),
                                  np.abs(cor[valid,5]-observed[valid]).sum(),
                                  (cor[valid,3]-cor[valid,2]).sum(),valid.sum())):
            sums[k]+=value
        peaks=np.maximum(peaks,[np.max(np.abs(cor[:,4]),initial=0.),np.max(cor[:,3]-cor[:,2],initial=0.)])
        if (j+10)%25000==0:
            print(json.dumps(dict(mode=a.mode,case=a.case,physical_s=t,publication=m.windows,
                                 elapsed_wall_s=time.time()-wall0,source=base.name)),flush=True)
    out=ART/f'{tag}-starts.json.gz'
    dump(out,rows)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result=dict(episode=base.name,mode=a.mode,case=a.case,selected_events=len(rows),eligible_cell_events=eligible,
        publication_count=m.windows,original_material_rows=len(foot),original_physical_end_s=m.time,
        recorded_files=len(blocks),distinct_observed_world_cells=len(m.ground.ids),
        observer_events=m.ground.event_serial,ledger_mode='observer-only, no plant e/q/load labels',
        frozen_r60_nominal_parity=parity if legacy is not None else None,
        correction_counts={k:v.tolist() for k,v in bins.items()},sum_error_before_N=float(sums[0]),
        sum_error_after_N=float(sums[1]),sum_force_range_N=float(sums[2]),soft_publication_foot_count=int(sums[3]),
        max_duration_abs_s=float(peaks[0]),max_supported_force_range_N=float(peaks[1]),
        observer_timing=m.timing(),domain_exit=m.ground.domain_exit,artifact=str(out.relative_to(ROOT)),pid=os.getpid(),
        started_unix_s=wall0,ended_unix_s=time.time(),cpu_core_s=time.process_time()-cpu0,
        process_max_rss_MiB=usage.ru_maxrss/1024,deadline=None,wall_deadline_enabled=False,
        kill_on_conversation_end=False,new_robot_episodes=0,new_policy_transitions=0)
    (OUT/f'{tag}-replay.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
