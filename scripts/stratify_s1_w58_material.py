"""Complete the source-release denominator and describe within-cell contact geometry."""
from collections import Counter
import argparse
import json
import time

import numpy as np

from collect_s1_w58_material import BG, MODES, OUT, RAW, ROOT, SPLITS, WORK
from collect_s1_w58_material import dump_gz, measured_episode, read_gz
from s1_w55_c_data import DT, R12, write
from score_s1_w58_material import summarize_errors

SELECT=ROOT/'results'/R12/'w58/m/selection.json'
HALF=np.sqrt(.001)/2
FAMILIES=('first_load','release','continuous_free_recovery','same_foot_reload','cross_foot_revisit','long_recovery')


def geometry(foot, source, target):
    # The stored release is stamped at the end of the first unloaded substep.
    # Its previous real loaded boundary lies one additional substep back.
    ks=round(source['time_s']/DT)-2
    kt=round(target['time_s']/DT)-1
    xy=foot[:,6:18].reshape(-1,4,3)
    a=xy[ks,source['leg'],:2]
    b=xy[kt,target['leg'],:2]
    cell=np.asarray(target['world_key'])*.02
    lo=np.maximum.reduce([cell,a-HALF,b-HALF])
    hi=np.minimum.reduce([cell+.02,a+HALF,b+HALF])
    return dict(source_load_event_id=source['previous_load_event_id'], source_release_event_id=source['event_id'],
        source_leg=source['leg'],target_leg=target['leg'],free_s=target['time_s']-source['time_s'],
        source_cell_loaded_area_m2=source['area_before_m2'],target_cell_loaded_area_m2=target['area_m2'],
        intersection_of_two_soles_with_cell_m2=float(np.maximum(hi-lo,0.).prod()),
        q_norm_at_release=float(np.linalg.norm(np.asarray(source['q'])*[1.,np.sqrt(2.)])))


def summarize_stratum(events, modes, bands):
    sampled=[e for e in events if e['sampled_release_prediction']]
    rows={}
    for m in MODES:
        records=[modes[m][e['event_id']] for e in sampled]
        table=summarize_errors(records)
        for key,metric in table.items():
            radius=bands[m].get(key,{}).get('band_full_width')
            vals=np.abs([r['errors'][key] for r in records if key in r['errors']])
            metric['band_full_width']=radius
            metric['covered']=None if radius is None else int(np.count_nonzero(vals<=radius/2))
            metric['coverage']=None if radius is None else float(np.mean(vals<=radius/2))
        rows[m]=table
    return dict(all_source_matched_cell_events=len(events),sampled=len(sampled),models=rows)


def analyze(episode, calibration_table):
    raw=RAW/episode
    _,foot,_=measured_episode(raw)
    path=OUT/f'{episode}-forecasts.json.gz'
    result=read_gz(path)
    modes={m:{r['event_id']:r for r in result['release_to_reload'] if r['mode']==m} for m in MODES}
    m2={r['event_id']:r for r in read_gz(WORK/f'{episode}-M2-starts.json.gz') if 'release_state' in r}
    releases={}
    cross=[]
    for event in map(json.loads,(raw/'material-events.jsonl').open()):
        key=tuple(event['world_key'])
        if event['type']=='release':
            releases[key]=event
            continue
        old=releases.pop(key,None)
        if not event.get('cross_leg'):
            continue
        matched=old is not None and old['previous_load_event_id']==event['previous_load_event_id']
        eid=event['event_id']
        row=dict(event_id=eid,world_key=list(key),target_end_s=event['time_s'],
                 source_release_matched=matched, sampled_release_prediction=eid in modes['M2'])
        if matched:
            row.update(geometry(foot,old,event))
        cross.append(row)
    # The first archived consumer spelled this source-cell force with "foot".
    # Normalize the saved field and retain the two measured areas with the event.
    for mode,records in modes.items():
        for eid,r in records.items():
            old=r.pop('source_foot_normal_force_before_N',None)
            if old is not None:
                r['source_cell_normal_force_before_N']=old
            if mode=='M2':
                r['released_path_difference']=m2[eid]['released_path_difference']
    dump_gz(path,result)
    matched=[x for x in cross if x['source_release_matched']]
    strata=dict(all=matched,
        positive_last_sole_intersection=[r for r in matched if r['intersection_of_two_soles_with_cell_m2']>0],
        zero_last_sole_intersection=[r for r in matched if r['intersection_of_two_soles_with_cell_m2']==0],
        free_less_than_6s=[r for r in matched if r['free_s']<6],
        free_at_least_6s=[r for r in matched if r['free_s']>=6],
        release_q_norm_above_1e_8=[r for r in matched if r['q_norm_at_release']>1e-8],
        release_q_norm_at_most_1e_8=[r for r in matched if r['q_norm_at_release']<=1e-8])
    bands={m:calibration_table[m].get('source_release/cross_foot',{}) for m in MODES}
    rank=[r['aggregate_force_observability'] for r in result['forward'] if r['aggregate_force_observability']]
    future=[r for r in result['forward'] if r['mode']=='M2']
    counts=Counter(r['family'] for r in future)
    counts['continuous_free_recovery']=sum(r['continuously_unloaded_after_actual_release'] for r in future)
    return dict(complete_plant_cross_foot_cell_load_count=len(cross),
        actual_foot_and_cell_intersections=cross,
        prior_release_missing_or_nonmatching=len(cross)-len(matched),
        comparison_strata={s:summarize_stratum(v,modes,bands) for s,v in strata.items()},
        short_horizon_counts={f:counts[f] for f in FAMILIES},
        evaluated_single_resultant_rank=dict(n=len(rank),counts=dict(Counter(v['local_force_rank'] for v in rank)),
          local_nullity_min_median_max=np.quantile([v['local_nullity'] for v in rank],[0,.5,1]).tolist() if rank else None,
          per_foot_latent_q_dimensions_min_median_max=np.quantile([v['prior_q_components'] for v in rank],[0,.5,1]).tolist() if rank else None))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--follow',action='store_true')
    a=p.parse_args()
    if a.follow:
        while not SELECT.is_file():
            time.sleep(20)
    selection=json.loads(SELECT.read_text())
    calibration=SPLITS['calibration'][0]
    bands=json.loads((OUT/'errors-and-coverage.json').read_text())['by_episode'][calibration]
    by_episode={ep:analyze(ep,bands) for ep in SPLITS['calibration']+SPLITS['retained']}
    target=OUT/'source-footprint-strata.json.gz'
    dump_gz(target,dict(source_splits=SPLITS,by_episode=by_episode,
        gap_definition='End of true source release through end of the target loading step on the same world cell.',
        overlap_definition='Intersection of the source sole at its last actual loaded kinematic boundary, '
          'the target sole at reload, and the fixed world cell. A cell can be positively loaded by both feet '
          'with no actual shared continuous XY area. These strata retain every denominator and do not select M.',
        **BG))
    selection.update(geometry_and_identifiability=str(target.relative_to(ROOT)),
        per_event_force_fields='source_cell_normal_force_before_N is the source cell normal component on its last '
          'loaded step. target_cell_normal_force_N is its first next loaded value.')
    write(SELECT,selection)


if __name__=='__main__':
    main()
