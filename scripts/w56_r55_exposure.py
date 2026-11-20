"""Join original r55 actor queries, per-cell loading histories and subsequent measured touchdowns."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import ctypes as ct
import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np
from s1_w55_c_replay import pack
from tm02.s1_contact import Ground

ROOT = Path(__file__).resolve().parents[1]
REL = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
F = ROOT.parent / 'MCP-RL-w55-f'
SMALL = ROOT / 'results' / REL / 'w56' / 'exposure'
FROZEN = F / 'work/assets/B_random1000--ideal-potentials.npz'
LEGS = ('FL','FR','RL','RR')


def jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


def replay(bottom, times, cell_queries, half):
    model = pack(np.load(FROZEN))
    dll = ct.CDLL(str(ROOT/'work/w56-query-columns.so'))
    vector = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
    fn = dll.query_states
    fn.argtypes = [vector,ct.c_int,vector,vector,np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),ct.c_int,vector]
    interior = ((bottom[:,:,0] >= .8+half) & (bottom[:,:,0] <= 2.2-half)
                & (abs(bottom[:,:,1]) <= .6-half))
    states = {}
    for cell, request in cell_queries.items():
        ticks = np.array(sorted(request),dtype=np.int32)
        end = int(ticks[-1])+1
        lo = np.array(cell)*.02
        xy = bottom[:end,:,:2]
        intersects = ((xy+half>lo)&(xy-half<lo+.02)).all(axis=2)
        normal = np.min(np.where(intersects & interior[:end],bottom[:end,:,2]/.05,np.inf),axis=1)
        out = np.empty((len(ticks),7))
        error = fn(model,end,times,normal,ticks,len(ticks),out)
        if error != -1:
            raise ValueError((cell,'material replay residual',error))
        states[cell] = dict(zip(ticks,out))
    return states


def label_revisits(target, raw):
    """Attach completed-lap and geometric overlap labels to the reconstructed event table."""
    with np.load(raw/'trajectory.npz') as trace:
        t, phase = trace['time'], trace['phase']
    # A return across x=.8 precedes the home hold. Only the subsequent
    # turn_home starts the next lap; final complete retains the last lap.
    next_laps = t[1:][(phase[:-1]==7)&(phase[1:]==8)].tolist()
    half=np.sqrt(.001)/2
    area_by_kind=defaultdict(float)
    query_by_kind=defaultdict(set)
    target_tmp=target.with_suffix('.classifying')
    with gzip.open(target,'rt',newline='') as src, gzip.open(target_tmp,'wt',newline='') as dst:
        rows=csv.DictReader(src)
        fields=['mission_lap','last_true_load_lap','same_world_cell_history','exact_query_next_touch_overlap_m2']
        out=csv.DictWriter(dst,fieldnames=rows.fieldnames+fields)
        out.writeheader()
        for row in rows:
            xy=[float(row['query_x']),float(row['query_y'])]
            lap=1+bisect_right(next_laps,float(row['time_s']))
            prior=1+bisect_right(next_laps,float(row['last_load_s'])) if row['last_load_s'] else None
            kind='outside' if not row['cell_ix'] else 'never_loaded'
            if prior is not None:
                leg='same_foot' if LEGS[int(row['previous_leg'])]==row['leg'] else 'cross_foot'
                kind=('earlier_lap_' if prior<lap else 'same_lap_')+leg
            overlap=0.
            if float(row['next_touch_cell_area_m2'])>0:
                td=[float(row[k]) for k in ('next_touch_x','next_touch_y')]
                key=[int(row[k]) for k in ('cell_ix','cell_iy')]
                width=[max(0.,min(x+half,z+half,(k+1)*.02)-max(x-half,z-half,k*.02))
                       for x,z,k in zip(xy,td,key)]
                overlap=width[0]*width[1]
            out.writerow(dict(row,mission_lap=lap,last_true_load_lap=prior,
                              same_world_cell_history=kind,exact_query_next_touch_overlap_m2=overlap))
            area_by_kind[kind]+=overlap
            query_by_kind[kind].add((row['time_s'],row['leg']))
    target_tmp.replace(target)
    return dict(foot_queries_touching_each_history_type={k:len(v) for k,v in query_by_kind.items()},
                exact_next_touch_overlap_area_m2_by_history=dict(area_by_kind),
                exact_next_touch_overlap_area_m2=sum(area_by_kind.values()))


def collect(case,level):
    begin = time.time()
    compact = F/'results'/REL/'w55/f'
    stage = json.loads((compact/f'F-{case}-r55'/(level+'.json')).read_text())
    name = stage['attempt']
    raw = F/'outputs'/REL/name
    result = json.loads((compact/(name+'.json')).read_text())
    with np.load(raw/'trajectory.npz') as saved:
        times = saved['time']
    # Use the boundaries actually applied at the control endpoints. The extra
    # final trajectory state has no new actor decision and no following hold.
    columns = result['material']['foot_columns']
    ids = [columns.index(f'bottom_{i}_{a}') for i in range(4) for a in 'xyz']
    blocks = []
    for path in sorted(raw.glob('material-[0-9]*.npz')):
        with np.load(path) as a:
            foot = a['foot']
            keep = (np.rint(foot[:,0]/.0002).astype(np.int64)%10)==0
            blocks.append(foot[keep][:,[0]+ids])
    endpoints = np.concatenate(blocks)
    if not np.allclose(endpoints[:,0],times[:len(endpoints)],rtol=0,atol=1e-10):
        raise ValueError('Material boundary and control clocks disagree')
    bottom = endpoints[:,1:].reshape(-1,4,3)
    plans = jsonl(raw/'w55-f/mpc-solves.jsonl')
    plant = jsonl(raw/'material-events.jsonl')
    events = defaultdict(list)
    for e in plant:
        events[tuple(e['world_key'])].append(e)
    event_times = {key:[r['time_s'] for r in rows] for key,rows in events.items()}
    contacts = jsonl(raw/'w55-f/contact-events.jsonl')
    touchdowns = {leg:[x for x in contacts if x['kind']=='actual_touchdown' and x['legs']==[leg]
                       and x['prior_mode'] in ('swing','search')] for leg in LEGS}
    decisions = json.loads((raw/'pann-action-dependence.json').read_text())
    transitions = jsonl(raw.parent/'evaluation-policy'/name/'transitions.jsonl')
    if len(decisions)!=len(transitions):
        raise ValueError('Original decision/transition count disagrees')
    pred_times = [e['time_s'] for e in plans]
    ground = Ground({'id':'kv'}, {})
    keys = []
    def footprint(xy):
        indices,areas = ground.footprint(np.array(xy))
        if len(keys)!=len(ground.ids):
            keys[:] = list(ground.ids)
        return dict(zip((keys[i] for i in indices),areas))

    queries = []
    requests = defaultdict(set)
    passages = result['legal_route']['passages']
    for decision, transition in zip(decisions,transitions):
        t = decision['time_s']
        tick = int(round(t/.002))
        if abs(transition['time_s']-transition['dt_s']-t)>1e-8:
            raise ValueError('Transition does not describe the executed decision')
        p = bisect_left(pred_times,t)-1
        xy = np.zeros((4,2))
        if p>=0:
            with np.load(raw/'w55-f'/plans[p]['file']) as plan:
                xy = plan['foot_center_plan'][np.argmin(abs(plan['times']-(t+.2))),:,:2]
        lap = 1 + sum(x['direction']=='return' and x['exit_s'] is not None and x['exit_s']<=t
                      for x in passages)
        for i,point in enumerate(xy):
            cells = footprint(point)
            idx = bisect_left([e['time_s'] for e in touchdowns[LEGS[i]]],t)
            td = touchdowns[LEGS[i]][idx] if idx < len(touchdowns[LEGS[i]]) else None
            tdxy = td['anchor'][:2] if td else None
            hit = footprint(tdxy) if td else {}
            query = dict(tick=tick,time_s=t,leg=LEGS[i],phase=decision['phase'],lap_after_return_crossings=lap,
                plan_call=None if p<0 else plans[p]['call'],xy=point.tolist(),cells=cells,td=td,hit=hit,
                feature=np.reshape(transition['pann_recovery_m'],(4,3))[i],
                cap_delta=decision['with_pann_cap']-decision['zero_feature_cap'])
            queries.append(query)
            for key in cells:
                requests[key].add(tick)
    states = replay(bottom,times,requests,ground.half)
    SMALL.mkdir(parents=True,exist_ok=True)
    # One row per query footprint cell, including a sentinel for an outside foot.
    target = SMALL/f'{case}-{level}-query-cells.csv.gz'
    seen, reloaded, matched = set(),set(),set()
    errors, previous_area, overlap_area, nonzero = [],[],[],0
    with gzip.open(target,'wt',newline='') as handle:
        w = csv.writer(handle)
        w.writerow('time_s leg phase lap_after_return_crossings prior_plan query_x query_y cell_ix cell_iy area_m2 previous_type previous_event previous_leg previous_event_age_s next_touch_event next_touch_s next_touch_x next_touch_y next_touch_cell_area_m2 common_cell_area_bound_m2 last_load_s last_release_s next_real_load_id next_real_load_s next_real_load_leg next_real_load_N next_real_load_preload_e11 next_real_load_preload_q11 observer_e11 observer_e22 observer_q11 observer_q22 projected_free_depth_2ms_m projected_free_depth_200ms_m projected_free_depth_500ms_m recorded_foot_recovery_2ms_m recorded_foot_recovery_200ms_m recorded_foot_recovery_500ms_m same_state_cap_delta_m_s'.split())
        for q in queries:
            t,foot = q['time_s'],q['cells']
            touch = q['td']
            touch_fields = [touch['event'],touch['time_s'],*touch['anchor'][:2]] if touch else [None]*4
            rec = np.zeros(3)
            loaded_area, shared = 0.,0.
            for key,area in foot.items() or [((None,None),0.)]:
                ticks = event_times.get(key,[])
                j = bisect_right(ticks,t+1e-10)
                before = events[key][:j]
                past = before[-1] if before else None
                ld = next((e for e in reversed(before) if e['type']!='release'),None)
                release = next((e for e in reversed(before) if e['type']=='release'),None)
                upcoming = next((e for e in events[key][j:] if e['type']!='release'),None)
                z = states[key][q['tick']] if key in states else [None]*7
                if key in states: rec += np.array(z[4:])*(area/ground.area)
                if ld:
                    loaded_area += area
                    seen.add(key)
                    if upcoming:
                        reloaded.add(key)
                        if key in q['hit']: matched.add(key)
                td_area=q['hit'].get(key,0.)
                shared+=min(area,td_area)
                w.writerow([t,q['leg'],q['phase'],q['lap_after_return_crossings'],q['plan_call'],*q['xy'],*key,area,
                    *([past['type'],past['event_id'],past['leg'],t-past['time_s']] if past else [None]*4),
                    *touch_fields,td_area,min(area,td_area),ld and ld['time_s'],release and release['time_s'],
                    *([upcoming['event_id'],upcoming['time_s'],upcoming['leg'],upcoming['normal_N'],upcoming['preload_free_e'][0],upcoming['preload_free_q'][0]] if upcoming else [None]*6),
                    *z,*q['feature'],q['cap_delta']])
            nonzero += bool(np.any(q['feature']))
            errors.append(max(abs(rec-q['feature'])))
            previous_area.append(loaded_area);overlap_area.append(shared)
    loaded_keys = {tuple(e['world_key']) for e in plant if e['type']!='release'}
    summary = dict(attempt=name,case=case,level=level,control_decisions=len(decisions),
        foot_queries=len(queries),inside_foot_queries=sum(bool(q['cells']) for q in queries),
        prior_load_foot_queries=sum(a>0 for a in previous_area),nonzero_explicit_features=nonzero,
        all_loaded_world_cells=len(loaded_keys),prior_loaded_cells_queried=len(seen),
        prior_loaded_queried_then_loaded_again_cells=len(reloaded),
        prior_loaded_queried_then_reloaded_and_shared_with_next_touch_cells=len(matched),
        prior_loaded_query_area_m2=sum(previous_area),sum_min_query_and_next_touch_cell_areas_m2=sum(overlap_area),
        max_abs_recorded_foot_recovery_m=float(max(max(abs(q['feature'])) for q in queries)),
        max_abs_reconstruction_vs_recorded_recovery_m=float(max(errors)),
        max_abs_same_state_cap_delta_m_s=max(abs(d['with_pann_cap']-d['zero_feature_cap']) for d in decisions),
        table=target.name,source_root=str(raw),start_unix_s=begin,end_unix_s=time.time(),
        scope='Actor before same-tick replan. Frozen FP64 CPU material law replays observed endpoints; e/q reconstructed. Actual recovery values read from original transitions. Per-cell shared area is the smaller of the two covered areas in that cell. Reloads are identified by original event ID. Unavailable future touchdown or load is blank.')
    summary.update(label_revisits(target,raw))
    (SMALL/f'{case}-{level}.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',choices=['A','B'])
    p.add_argument('level',choices=['E60','E300','E600'])
    a=p.parse_args()
    collect(a.case,a.level)
