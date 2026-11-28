"""Attribute every missing FR->RR opportunity in the frozen 24 W59 windows."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def reason(row,end,mode):
    if row['target_id'] is None:
        return 'no_subsequent_load'
    if row['target_leg']!=3:
        return 'FR_reloaded_first' if row['target_leg']==1 else 'other_non_RR_leg_first'
    if row['target_s']>end:
        return 'RR_after_traversal'
    if row['release_id'] is None:
        return 'missing_release'
    if row['gap_s']<=0.:
        return 'nonpositive_gap'
    if row['outcome']!='positive_cross_foot_overlap':
        return 'zero_footprint_intersection'
    if mode=='recovered' and row['gap_s']<6.:
        return 'free_gap_below_6s'
    if mode=='short' and row['gap_s']>=3.:
        return 'free_gap_at_least_3s'
    return 'matched'


def analyze(repo,reference):
    git=['git','-C',str(repo),'show']
    audit=json.loads(subprocess.check_output(git+[reference+':docs/research/20261128-w59-review/evidence_gait.json']))
    rid='tm-s1-go2-local-readiness-r12'; raw=repo/'outputs/terrain_memory/simulation'/rid
    episodes=repo/'results/terrain_memory/simulation'/rid/'w59/episodes'
    totals=Counter();attribution=Counter();per_source=Counter();runs=[]
    for run in audit['mechanism']['per_attempt']:
        attempt=run['attempt'];receipt=json.loads((episodes/(attempt+'.json')).read_text())
        mech=receipt['mechanism'];file=raw/attempt/mech['ledger']
        if not file.is_file():
            runs.append(dict(attempt=attempt,source=str(file),state='ledger_not_on_disk'));continue
        with gzip.open(file,'rt') as stream:
            rows=[r for r in map(json.loads,stream) if r['source_leg']==1 and r['key']==mech['targets'][0]['world_key']]
        records=[]
        for target in mech['targets']:
            i=target['traversal']; trans=receipt['transitions']
            begin=next(e['time_s'] for e in trans if e['cycle']==i and e['phase']=='outbound')
            end=next(e['time_s'] for e in trans if e['cycle']>i)
            source=[r for r in rows if begin<=r['source_s']<=end]
            pair=[r for r in source if r['target_leg']==3 and r['target_s']<=end]
            if len(source)!=target['source_loads'] or len(pair)!=target['source_target_pairs']:
                raise ValueError((attempt,i,len(source),len(pair),target))
            category='passed' if target['achieved'] else 'no_source' if not source else 'no_pair' if not pair else 'pair_no_match'
            totals[category]+=1
            events=pair if category=='pair_no_match' else source
            outcomes={str(r['source_id']):reason(r,end,mech['mode']) for r in events}
            if category in ('no_pair','pair_no_match'):
                attribution[(category,'+'.join(sorted(set(outcomes.values()))))]+=1
                per_source.update((category,v) for v in outcomes.values())
            records.append(dict(target=target['id'],begin_s=begin,end_s=end,category=category,by_source_event=outcomes))
        runs.append(dict(attempt=attempt,source=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),targets=records))
    return dict(review_source=reference,counts=dict(totals),target_reason_sets={'/'.join(k):n for k,n in attribution.items()},
        underlying_source_counts={'/'.join(k):n for k,n in per_source.items()},runs=runs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('frozen_repo',type=Path);p.add_argument('review_commit')
    p.add_argument('result',type=Path);a=p.parse_args();data=analyze(a.frozen_repo,a.review_commit)
    a.result.parent.mkdir(parents=True,exist_ok=True);a.result.write_text(json.dumps(data,indent=2)+'\n')
    print(data['counts'],data['target_reason_sets'])
