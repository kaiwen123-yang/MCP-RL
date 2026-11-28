"""Score fixed reachability targets against original 5 kHz material events."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from s1_w57_mechanism import add_footprint_overlap


def score(raw,receipt,protocol):
    outcome=json.loads(receipt.read_text());spec=json.loads(protocol.read_text())
    targets={tuple(t['world_key']):t for t in spec['targets']}
    requested=outcome['transitions']
    start=next((t['time_s'] for t in requested if t['phase']=='outbound'),None)
    stop=next((t['time_s'] for t in requested if start is not None and t['time_s']>start and t['phase']!='outbound'),
              outcome['simulation_s'])
    chains={};releases={};events={}
    for line in (raw/'material-events.jsonl').open():
        e=json.loads(line);key=tuple(e['world_key'])
        if key not in targets or start is None or e['time_s']<start or e['time_s']>stop:
            continue
        previous=e['previous_load_event_id']
        if e['type']=='release':
            releases[previous]=e
        else:
            events[e['event_id']]=e
            if previous in chains:
                chains[previous]['next']=e
                if previous in releases:
                    chains[previous]['release']=releases[previous]
            if e['leg']==1:
                chains[e['event_id']]={'source':e}
    pulses=[]
    f=raw/'w55-f/contact-events.jsonl'
    if f.is_file():
        for line in f.open():
            data=json.loads(line)
            if data['kind']=='liftoff_committed' and 'FR' in data['legs']:
                pulses.append(data['time_s'])
    phases=defaultdict(list);by_key=defaultdict(list)
    candidates=[]
    for c in chains.values():
        source=c['source'];nxt=c.get('next');rel=c.get('release');key=tuple(source['world_key'])
        j=np.searchsorted(pulses,source['time_s'])
        if j>0 and j<len(pulses):
            phases[key].append(float((source['time_s']-pulses[j-1])/(pulses[j]-pulses[j-1])))
        record=dict(key=list(key),source_id=source['event_id'],source_leg=1,source_s=source['time_s'],
              release_id=rel['event_id'] if rel else None,release_s=rel['time_s'] if rel else None,
              target_id=nxt['event_id'] if nxt else None,target_s=nxt['time_s'] if nxt else None,
              target_leg=nxt['leg'] if nxt else None,exact_overlap_m2=None,
              gap_s=nxt['time_s']-rel['time_s'] if rel and nxt else None,
              preload_free_e=nxt.get('preload_free_e') if nxt else None,
              preload_free_q=nxt.get('preload_free_q') if nxt else None)
        if nxt and nxt['leg']==3 and rel and record['gap_s']>0.:
            candidates.append(record)
        by_key[key].append(record)
    add_footprint_overlap(raw,candidates)
    rows=[]
    for key,target in targets.items():
        opportunities=by_key[key]
        pairs=[r for r in opportunities if r['target_leg']==3]
        hits=[r for r in pairs if (r['exact_overlap_m2'] or 0.)>0. and 0.<r['gap_s']<3.]
        rows.append(dict(id=target['id'],key=list(key),source_loads=len(opportunities),pairs=len(pairs),
            first_incidence_phase=phases[key][0] if phases[key] else None,matched=len(hits),events=opportunities))
    phase=[r['first_incidence_phase'] for r in rows if r['first_incidence_phase'] is not None]
    return dict(attempt=outcome['attempt'],protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
       sha256_events=hashlib.sha256((raw/'material-events.jsonl').read_bytes()).hexdigest(),
       observed_s=outcome['simulation_s'],success=outcome['success'],termination=outcome['reason'],
       first_outbound_start_s=start,first_outbound_end_s=stop,first_source_phase_histogram=np.histogram(phase,[0,.25,.5,.75,1.])[0].tolist(),
       total_targets=len(rows),target_source=sum(r['source_loads']>0 for r in rows),
       target_pair=sum(r['pairs']>0 for r in rows),target_match=sum(r['matched']>0 for r in rows),
       source_events=sum(r['source_loads'] for r in rows),paired_events=sum(r['pairs'] for r in rows),
       matched_events=sum(r['matched'] for r in rows),targets=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('protocol',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();record=score(a.raw,a.receipt,a.protocol);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('attempt','total_targets','target_source','target_pair','target_match','source_events','paired_events','matched_events')}))
