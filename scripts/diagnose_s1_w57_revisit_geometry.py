"""Resolve actual same-side plantar overlap at each W57 traversal, without re-scoring the task."""
import argparse
from collections import Counter, defaultdict
import json

import numpy as np
from scipy.spatial import cKDTree

from publish_s1_go2_w54 import write
from summarize_s1_w57 import RAW, R, read

DT=.0002
CELL=.02
HALF=np.sqrt(.001)/2
LEGS=('FL','FR','RL','RR')


class Boundaries:
    def __init__(self, folder):
        chunks=[]
        for p in sorted(folder.glob('material-[0-9]*.npz')):
            with np.load(p) as z:
                f=z['foot']
            chunks.append(f[:,[0,1,6,7,8,9,10,11,12,13,14,15,16,17]])
        rows=np.concatenate(chunks)
        self.clocks=rows[:,:2]
        self.xyz=rows[:,2:].reshape(-1,4,3)

    def centroid(self, end_tick, leg):
        return self.xyz[end_tick-1,leg,:2]

    def support(self, start, end, leg):
        return self.xyz[start-1:end,leg,:2]


def event_histories(folder, spec, boundary):
    xlo,xhi=spec['target_strip_x_m']
    ylo,yhi=spec['target_strip_y_m']
    objects={}; loads=[]; active={}; last={}
    audit=Counter(dict.fromkeys(('duplicate_event_id','previous_load_id_mismatch',
        'release_source_key_owner_id_mismatch','direct_owner_change_without_free_interval'),0))
    maxclock=0.
    for line in (folder/'material-events.jsonl').open():
        ev=json.loads(line)
        key=tuple(ev['world_key'])
        x,y=(np.array(key)+.5)*CELL
        if not(xlo<=x<xhi and ylo<=y<yhi):
            continue
        tick=round(ev['time_s']/DT)
        maxclock=max(maxclock,abs(boundary.clocks[tick-1,1]-ev['time_s']),
                     abs(boundary.clocks[tick-1,0]+DT-ev['time_s']))
        i=ev['event_id']
        if i in objects:
            audit['duplicate_event_id']+=1
        objects[i]=ev
        if ev['type']=='release':
            audit['actual_release_events']+=1
            src=active.pop(key,None)
            if src is None or src['id']!=ev['previous_load_event_id'] or src['leg']!=ev['leg']:
                audit['release_source_key_owner_id_mismatch']+=1
            else:
                src.update(end=tick-1,release=tick,release_id=i)
            continue
        audit['actual_load_events']+=1
        previous=last.get(key)
        if ev['previous_load_event_id']!=(previous['id'] if previous else -1):
            audit['previous_load_id_mismatch']+=1
        if key in active:
            active[key]['end']=tick-1
            audit['direct_owner_change_without_free_interval']+=1
        src=dict(id=i,key=key,leg=ev['leg'],start=tick,end=len(boundary.clocks),time_s=ev['time_s'],
                 previous_id=ev['previous_load_event_id'],release=None,release_id=None)
        last[key]=src;active[key]=src;loads.append(src)
    audit['max_event_vs_measured_left_plus_dt_or_state_clock_s']=maxclock
    return loads,dict(audit)


def overlap(a,b,lo,hi):
    lower=np.maximum.reduce([a-HALF,b-HALF,np.asarray(lo)])
    upper=np.minimum.reduce([a+HALF,b+HALF,np.asarray(hi)])
    return float(np.prod(np.maximum(upper-lower,0.)))


def groups(indices):
    if not len(indices):
        return []
    split=np.flatnonzero(np.diff(indices)>1)+1
    return [(int(v[0]),int(v[-1])) for v in np.split(indices,split)]


def positive_support(episodes, lo, hi):
    parts=[np.arange(max(lo,v['start']),min(hi,v['end'])+1,dtype=int) for v in episodes
           if v['end']>=lo and v['start']<=hi]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0,dtype=int)


def nearest_forward_geometry(boundary, src_ticks, dst_ticks, leg_a, leg_b, strip_lo, strip_hi):
    # A real source support visit must precede a separate next destination visit.
    best=np.inf;min_x=np.inf
    witness=None;ordered=0
    for pre in groups(src_ticks):
        xy=boundary.support(*pre,leg_a)
        sx=np.sort(xy[:,0])
        tree=cKDTree(xy)
        for post in groups(dst_ticks):
            if pre[1]>=post[0]:
                continue
            ordered+=1
            uv=boundary.support(*post,leg_b)
            j=np.searchsorted(sx,uv[:,0])
            nearest=np.minimum(abs(sx[np.minimum(j,len(sx)-1)]-uv[:,0]),
                               abs(sx[np.maximum(j-1,0)]-uv[:,0])).min()
            min_x=min(min_x,nearest)
            d,i=tree.query(uv,p=np.inf)
            k=int(np.argmin(d))
            if d[k]<best:
                best=float(d[k])
                v=xy[i[k]];w=uv[k]
                witness=dict(source_tick=pre[0]+int(i[k]),target_tick=post[0]+k,
                             source_xy_m=v.tolist(),target_xy_m=w.tolist(),dx_m=float(w[0]-v[0]),
                             dy_m=float(w[1]-v[1]),intersection_clipped_to_strip_m2=overlap(v,w,strip_lo,strip_hi))
    return dict(ordered_disjoint_support_visit_pairs=ordered,
        minimum_longitudinal_footprint_edge_clearance_m=float(max(0.,min_x-2*HALF)) if ordered else None,
        minimum_planar_infinity_distance_between_centers_m=best if ordered else None,
        witness=witness)


def analyze(attempt):
    folder=RAW/attempt
    result=read(R/'episodes'/(attempt+'.json'))
    receipt=read(R/'attempts'/(attempt+'.json'))
    spec=receipt['settings']['mechanism'];mode=receipt['task_mode']
    b=Boundaries(folder)
    loads,audit=event_histories(folder,spec,b)
    byid={v['id']:v for v in loads}
    bykey=defaultdict(list)
    for v in loads:bykey[v['key']].append(v)
    official=[json.loads(l) for l in (folder/'w57-revisit-pairs.jsonl').open()]
    published={(v['cycle'],v['source_id'],v['release_id'],v['target_id']):v for v in official}
    all_confirmed=set(); all_direct=set(); cycles=[]
    sx=spec['target_strip_x_m'];sy=spec['target_strip_y_m']
    strip_lo=[sx[0],sy[0]];strip_hi=[sx[1],sy[1]]
    with np.load(folder/'trajectory.npz') as trajectory:
        rt=np.rint(trajectory['time']/DT).astype(int)
        base=trajectory['qpos'][:,:2];feet=trajectory['feet'][:,:,:2]
        yaw=trajectory['rpy'][:,2]
    for w in result['mechanism']['traversals']:
        number=w['cycle'];direction=w['direction'];lo,hi=(round(w[k]/DT) for k in ('start_s','end_s'))
        in_window=[v for v in loads if v['start']<=hi and v['end']>=lo] # includes contacts beginning before the traversal
        onset=[v for v in loads if lo<=v['start']<=hi]
        byleg=[[v for v in in_window if v['leg']==leg] for leg in range(4)]
        positive_keys=[{v['key'] for v in vs} for vs in byleg]
        ticks=[positive_support(vs,lo,hi) for vs in byleg]
        support={name:dict(actual_loaded_cell_intervals=len(vs),actual_unique_loaded_cells=len(positive_keys[i]),
                  intersecting_contact_without_new_load_in_cycle=sum(v['start']<lo for v in vs),
                  load_event_count=sum(v['leg']==i for v in onset),release_event_count=sum(
                      v['leg']==i and v['release'] is not None and lo<=v['release']<=hi for v in loads),
                  loaded_foot_5kHz_unique_times=len(ticks[i]),longitudinal_cell_indices=sorted({x for x,y in positive_keys[i]}),
                  transverse_cell_indices=sorted(set(y for x,y in positive_keys[i])))
                 for i,(name,vs) in enumerate(zip(LEGS,byleg))}
        ti=(rt>=lo)&(rt<=hi)
        off=feet[ti]-base[ti,None]
        ca=np.cos(yaw[ti])[:,None];sa=np.sin(yaw[ti])[:,None]
        bodyxy=np.stack((ca*off[:,:,0]+sa*off[:,:,1],-sa*off[:,:,0]+ca*off[:,:,1]),axis=-1)
        sides={}
        for src,dst in ((0,2),(1,3)) if direction>0 else ((2,0),(3,1)):
            ordered_cell_pairs=[]
            current_ids=[];ac=[];outside=[]
            all_incoming=[]
            for target in onset:
                if target['leg']!=dst:continue
                key=target['key']
                for s in bykey[key]:
                    if (s['start']>=lo and s['start']<target['start'] and s['leg']==src and
                        s['release'] is not None and s['release']<target['start']):
                        ordered_cell_pairs.append([s['id'],s['release_id'],target['id']])
                prev=byid.get(target['previous_id'])
                if prev is None or prev['leg']!=src:continue
                all_incoming.append([prev['id'],target['id']])
                if prev['release'] is None or prev['release']>=target['start']:continue
                a=b.centroid(prev['start'],src);last=b.centroid(prev['end'],src);now=b.centroid(target['start'],dst)
                kl=np.array(key)*CELL;kh=kl+CELL
                areas=[overlap(a,now,kl,kh),overlap(last,now,kl,kh)]
                gap=target['time_s']-b.clocks[prev['release']-1,1]
                passed=gap<spec['short_gap_upper_s'] if mode=='short' else gap>=spec['free_target_hold_s']
                if prev['start']<lo:
                    outside.append(dict(source=prev['id'],release=prev['release_id'],target=target['id'],
                        gap_s=gap,exact_overlap_m2=max(areas),meets_gap=bool(passed)))
                    continue
                current_ids.append([prev['id'],prev['release_id'],target['id']])
                identity=(number,prev['id'],prev['release_id'],target['id'])
                all_direct.add(identity)
                if max(areas)>0. and passed:all_confirmed.add(identity)
                ac.append(dict(source=prev['id'],release=prev['release_id'],target=target['id'],key=list(key),
                               gap_s=gap,source_onset_area_m2=areas[0],last_loaded_area_m2=areas[1],
                               gap_condition=bool(passed),positive_area=bool(max(areas)>0.),
                               area_error_vs_published_m2=abs(max(areas)-published[identity]['exact_overlap_m2']) if identity in published else None))
            ka,kb=positive_keys[src],positive_keys[dst]
            sides[f'{LEGS[src]}_to_{LEGS[dst]}']=dict(
                shared_loaded_world_keys=sorted(ka&kb),overlapping_longitudinal_key_indices=sorted({x for x,y in ka}&{x for x,y in kb}),
                overlapping_transverse_key_indices=sorted({y for x,y in ka}&{y for x,y in kb}),
                ordered_same_key_true_release_then_later_load_count=len(ordered_cell_pairs),
                all_immediate_source_id_correct_direction_including_outside_cycle=len(all_incoming),
                excluded_immediate_predecessor_from_prior_traversals=len(outside),excluded_across_cycle_evidence=outside,
                valid_same_cycle_immediate_ID_and_positive_unloaded_gap=len(current_ids),
                positive_area_rows_before_gap_gate=sum(r['positive_area'] for r in ac),
                gap_condition_rows_before_area_gate=sum(r['gap_condition'] for r in ac),
                both_area_and_gap=sum(r['positive_area'] and r['gap_condition'] for r in ac),common_input_evidence=ac,
                footprint_distance=nearest_forward_geometry(b,ticks[src],ticks[dst],src,dst,strip_lo,strip_hi))
        cycles.append(dict(cycle=number,direction=direction,start_s=w['start_s'],end_s=w['end_s'],
                           support=support,actual_median_foot_body_xy_m=dict(zip(LEGS,np.median(bodyxy,axis=0).tolist())),sides=sides))
    official_ok={i for i,v in published.items() if v['condition_met'] and v['exact_overlap_m2']>0.}
    return dict(attempt=attempt,task_mode=mode,arm=receipt['arm'],material=receipt['material'],case=receipt['case'],
      numerical_units=dict(cell_edge_m=CELL,foot_side_m=2*HALF,actual_substep_s=DT,strip_x_m=sx,strip_y_m=sy),
      direction_mapping='Within a fixed-heading forward traversal FL→RL, FR→RR; reverse RL→FL, RR→FR. Checked against the body-frame position medians of all four measured foot centers in each traversal.',
      source_event_clock_audit=audit,official_matched_event_rows=len(official),independent_matched_event_rows=len(all_direct),
      symmetric_difference_full_ID_tuples=sorted(all_direct ^ set(published)),official_confirmed_event_rows=len(official_ok),
      independent_confirmed_event_rows=len(all_confirmed),symmetric_difference_confirmed_ID_tuples=sorted(all_confirmed ^ official_ok),
      max_abs_exact_overlap_reconstruction_error_m2=max((v['area_error_vs_published_m2'] for c in cycles for s in c['sides'].values()
         for v in s['common_input_evidence'] if v['area_error_vs_published_m2'] is not None),default=None),
      directional_summary={label:dict(cycles=[v['cycle'] for v in cycles if v['direction']==sign],
           confirmed_cycles=sorted({i[0] for i in all_confirmed if any(c['cycle']==i[0] and c['direction']==sign for c in cycles)}),
           loaded_common_key_total_over_sides=sum(len(s['shared_loaded_world_keys']) for c in cycles if c['direction']==sign for s in c['sides'].values()),
           actual_incoming_same_key_prior_traversal_rows=sum(s['excluded_immediate_predecessor_from_prior_traversals'] for c in cycles if c['direction']==sign for s in c['sides'].values()),
           cross_cycle_positive_area_and_gap_if_cycle_rule_were_ignored=sum(v['meets_gap'] and v['exact_overlap_m2']>0.
               for c in cycles if c['direction']==sign for s in c['sides'].values() for v in s['excluded_across_cycle_evidence']),
           cross_cycle_unloaded_gap_min_p50_max_s=(lambda a:np.quantile(a,[0,.5,1]).tolist() if a else None)(
               [v['gap_s'] for c in cycles if c['direction']==sign for s in c['sides'].values() for v in s['excluded_across_cycle_evidence']]),
           confirmed_event_rows=sum(s['both_area_and_gap'] for c in cycles if c['direction']==sign for s in c['sides'].values()),
           min_median_max_longitudinal_edge_gap_m=(lambda xs:np.quantile(xs,[0,.5,1]).tolist() if xs else None)(
                  [s['footprint_distance']['minimum_longitudinal_footprint_edge_clearance_m'] for c in cycles if c['direction']==sign
                   for s in c['sides'].values() if s['footprint_distance']['ordered_disjoint_support_visit_pairs']]))
           for label,sign in [('forward',1),('reverse',-1)]},cycles=cycles,
      execution='Reuses the original full 5 kHz left-boundary and material-event streams. No new physical, training, score or task identity.')


if __name__=='__main__':
    arg=argparse.ArgumentParser(description=__doc__);arg.add_argument('attempt');a=arg.parse_args()
    d=analyze(a.attempt)
    write(R/'m'/('revisit-geometry-'+a.attempt+'.json'),d)
    print(json.dumps(dict(attempt=a.attempt,directions=d['directional_summary'],official=d['official_confirmed_event_rows'],
                          independent=d['independent_confirmed_event_rows'],id_diff=d['symmetric_difference_confirmed_ID_tuples'],
                          area_error=d['max_abs_exact_overlap_reconstruction_error_m2'],clock=d['source_event_clock_audit'])),flush=True)
