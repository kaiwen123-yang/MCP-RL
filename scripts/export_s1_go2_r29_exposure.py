"""Offline W40 query exposure and missed sub-2ms load labels; never actor input."""
import json,math
from pathlib import Path
import numpy as np
HALF=.015811388300841896
def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
def footprint(point):
    cells=[]
    for ix in range(math.floor((point[0]-HALF)/.02),math.floor((point[0]+HALF)/.02)+1):
        for iy in range(math.floor((point[1]-HALF)/.02),math.floor((point[1]+HALF)/.02)+1):
            dx=min(point[0]+HALF,(ix+1)*.02)-max(point[0]-HALF,ix*.02)
            dy=min(point[1]+HALF,(iy+1)*.02)-max(point[1]-HALF,iy*.02)
            if dx>0 and dy>0:cells.append((ix,iy))
    return cells
def export_episode(folder,destination,*,initialization_failure=None,resolved=None):
    folder=Path(folder);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    events=rows(folder/'material-events.jsonl');queries=rows(folder/'query-exposure.jsonl')
    # The recorded W44 failure occurs in the first planner RPC, before the
    # runtime starts the plant or creates trajectory.npz. This is unavailable
    # exposure evidence, not zero exposure or a reconstructed trajectory.
    if (not (folder/'trajectory.npz').exists() and initialization_failure is not None
            and initialization_failure['attempt']==folder.name
            and initialization_failure['physical_steps']==0
            and initialization_failure['error']
            and resolved['r29_recording']['terminal_s']==0):
        path=destination/'query-exposure-offline.json'
        path.write_text(json.dumps(dict(source=str(folder),scope='offline diagnostic; not fed to actor',
            status='unavailable_no_plant_start',reason='Planner initialization failed before any plant step; trajectory.npz was never created.',
            queries=None,missed_endpoint_pulses=None,query_count=None,all_outside_count=None,
            inside_no_history_count=None,raw_preplant_query_rows=queries,
            first_initialization_error=initialization_failure,
            startup_input=resolved['startup'],execution_fingerprint=resolved['execution_fingerprint'],
            source_model=str(folder/'model.mjb'),source_query_log=str(folder/'query-exposure.jsonl'),
            unavailable_source=str(folder/'trajectory.npz'),
            physical_evidence='No physical trajectory or exposure classification is inferred from the preplant observation.'),indent=2)+'\n')
        return path
    loads={};first={};pulses=[]
    with np.load(folder/'trajectory.npz') as a:
        times=a['diagnostic_time'];feet=a['feet'];valid=np.isclose(a['time'],times,rtol=0,atol=1e-9)
        # Geometry coverage at observer endpoints is sufficient to establish
        # some missed pulses; no claim about unrecorded observer e/q.
        for e in events:
            key=tuple(e['world_key'])
            if e['type']=='load':
                first.setdefault(key,float(e['time_s']));loads[key]=e
            elif e['type']=='release' and key in loads:
                begin=loads.pop(key);lo=float(begin['time_s'])-.0002;hi=float(e['time_s'])-.0002
                indices=np.flatnonzero(valid & (times>=lo-1e-9)&(times<hi-1e-9))
                covered=any(key in footprint(p[:2]) for i in indices for p in feet[i])
                if not covered:
                    pulses.append(dict(world_key=list(key),load_event_id=begin['event_id'],
                        release_event_id=e['event_id'],boundary_start_s=lo,boundary_end_s=hi,
                        duration_s=hi-lo,tag='no_observer_endpoint_covers_loaded_cell',
                        evidence='actual material load/release plus saved control-endpoint footprint; no material replay'))
    classified=[]
    for q in queries:
        t=q['time_s'];inside=np.array(q['inside']);feature=np.array(q['feature_nonzero'])
        keys=[footprint(p) for p in q['query_xy']]
        hit=[any(first.get(k,float('inf'))<=t for k in ks) for ks in keys]
        missed=[any(tuple(p['world_key']) in ks and p['boundary_end_s']<=t for p in pulses) for ks in keys]
        classified.append(dict(time_s=t,inside=q['inside'],previous_true_load=hit,
             overlaps_prior_unobserved_pulse=missed,feature_nonzero=q['feature_nonzero'],
             all_outside=not bool(inside.any()),any_inside_without_history=bool(np.any(inside & ~np.array(hit)))))
    path=destination/'query-exposure-offline.json'
    path.write_text(json.dumps(dict(source=str(folder),scope='offline diagnostic; not fed to actor',
        queries=classified,missed_endpoint_pulses=pulses,query_count=len(classified),
        all_outside_count=sum(x['all_outside'] for x in classified),
        inside_no_history_count=sum(x['any_inside_without_history'] for x in classified),
        pulse_limit='No endpoint spatial coverage proves sampling missed that pulse; endpoint coverage alone does not prove observer loading.'),indent=2)+'\n')
    return path
