"""A task-level fixed footprint and a separate all-opportunities ledger for the original route."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from s1_w57_mechanism import add_footprint_overlap
from s1_w58_revisit import ForwardRevisitMission
from publish_s1_go2_w54 import write

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'experiments/terrain_memory/s1-w59-footprints.json'


def mechanism_spec(split):
    protocol = json.loads((ROOT/PROTOCOL).read_text())
    key = protocol['world_keys'][split]
    xy = ((np.asarray(key) + .5)*.02).tolist()
    return dict(split=split, world_key=key, fixed_center_m=xy,
        target_strip_x_m=[key[0]*.02, (key[0]+1)*.02],
        target_strip_y_m=[key[1]*.02, (key[1]+1)*.02],
        center_park_x_m=xy[0], body_reversal_x_m=[xy[0]-.40, xy[0]+.40],
        right_stance_offset=protocol['right_stance_offset'], cycles=23,
        free_target_hold_s=6., short_gap_upper_s=3.,
        observation_s={'short':168., 'recovered':288.}, trials=12)


class FootprintMission(ForwardRevisitMission):
    """Aim the body path for successive front/hind passes over a fixed ground footprint."""
    def update(self,t,dt,pos,velocity,ori,omega,standing):
        old_y, old_vy = self.body_target[1], self.v[1]
        target, heading = super().update(t,dt,pos,velocity,ori,omega,standing)
        if self.returning_home or self.phase not in ('outbound','return','far_hold'):
            return target, heading
        # Estimate each foot's unloaded-to-loaded station from prior measured stance offsets.
        # This sets the shared task waypoint. The gait commits its own landing and all hard gates.
        x,y = self.spec['fixed_center_m']
        front,hind = self.spec['right_stance_offset']
        u = np.clip((pos[0]-(x-front[0]))/(front[0]-hind[0]),0.,1.)
        waypoint_y = y-((1-u)*front[1]+u*hind[1])
        self.active_goal[1] = waypoint_y
        desired_y = np.clip(.8*(waypoint_y-pos[1]),-.1,.1)
        self.v[1] = old_vy+np.clip(desired_y-old_vy,-.5*dt,.5*dt)
        self.body_target[1] = old_y + self.v[1]*dt
        return self.body_target.copy(), heading


def cell_chains(raw):
    """One row per actual source load, including never released or never reloaded before the horizon."""
    loads = {}
    for line in (raw/'material-events.jsonl').open():
        ev = json.loads(line)
        cell = tuple(ev['world_key'])
        prior = (cell,ev['previous_load_event_id'])
        if ev['type'] == 'release':
            if prior in loads:
                loads[prior].update(release_id=ev['event_id'], release_s=ev['time_s'])
            continue
        if prior in loads:
            v = loads[prior]
            if v['target_id'] is not None:
                raise ValueError(f'Duplicate immediately subsequent load for {prior}')
            v.update(target_id=ev['event_id'],target_s=ev['time_s'],target_leg=ev['leg'],
                     target_N=ev['normal_N'])
            if v['release_s'] is not None:
                v['gap_s'] = ev['time_s'] - v['release_s']
        identity = (cell,ev['event_id'])
        if identity in loads:
            raise ValueError(f'Duplicate material event {identity}')
        loads[identity] = dict(key=list(cell),source_id=ev['event_id'],source_leg=ev['leg'],
            source_s=ev['time_s'],source_N=ev['normal_N'],release_id=None,release_s=None,
            target_id=None,target_s=None,target_leg=None,target_N=None,gap_s=None,
            exact_overlap_m2=None)
    rows = list(loads.values())
    different = [r for r in rows if (r['gap_s'] or 0)>0 and r['target_leg']!=r['source_leg']]
    add_footprint_overlap(raw,different)
    for r in rows:
        if r['release_id'] is None:
            r['outcome'] = 'load_without_release' if r['target_id'] is None else 'owner_changed_without_release'
        elif r['target_id'] is None:
            r['outcome'] = 'no_reload_before_end'
        elif r['gap_s'] <= 0:
            r['outcome'] = 'nonpositive_free_interval'
        elif r['source_leg'] == r['target_leg']:
            r['outcome'] = 'same_leg_reload'
        else:
            r['outcome'] = 'positive_cross_foot_overlap' if r['exact_overlap_m2']>0 else 'zero_cross_foot_overlap'
    return rows


def save_complete(raw,rows):
    p = raw/'w59-source-load-opportunities.jsonl.gz'
    with gzip.open(p,'wt') as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False)+'\n')
    return p.name


def score_targets(result,c,mission,raw):
    spec = mission.spec
    chains = cell_chains(raw)
    ledger = save_complete(raw,chains)
    targets = []
    traversals = {x['cycle']:x for x in mission.completed}
    for i in range(12):
        cycle = 2*i+1
        lane = traversals.get(cycle)
        observed = lane is not None
        if lane is None and cycle==len(traversals)+1 and mission.cycle_start is not None:
            lane = dict(start_s=mission.cycle_start,end_s=result['simulation_s'])
        rows = [] if lane is None else [r for r in chains if r['key']==spec['world_key'] and
            r['source_leg']==1 and lane['start_s']<=r['source_s']<=lane['end_s']]
        pairs = [r for r in rows if r['target_leg']==3 and r['target_s']<=lane['end_s']]
        positive = [r for r in pairs if r['outcome']=='positive_cross_foot_overlap' and (
            0.<r['gap_s']<spec['short_gap_upper_s'] if mission.mode=='short' else r['gap_s']>=spec['free_target_hold_s'])]
        clear_s = mission.verified_clear_s.get(cycle,0.)
        achieved = observed and bool(positive) and (mission.mode=='short' or clear_s>=6.-1.e-8)
        targets.append(dict(id=f'F{i+1:02d}', traversal=cycle, world_key=spec['world_key'],
            body_completed=observed, continuous_clear_s=clear_s,source_loads=len(rows),
            source_target_pairs=len(pairs), matched=[[r['source_id'],r['release_id'],r['target_id']] for r in positive],
            achieved=achieved))
    n = sum(r['achieved'] for r in targets)
    full = result['simulation_s']>=c['r11_stage']['duration_s']-1.e-7
    home = mission.phase=='complete' and result['hold']['final_stable_s']>=2.-1.e-7
    passed = full and home and n==12 and result['reason']=='stage_horizon_reached'
    reason = result['reason']
    result.update(success=passed, sustained_execution_success=passed, original_mission_success=None,
        case_key=c['case_key'],case_identity=c['case_identity'],evaluation_stage=c['r11_stage'],
        qualification_usable=False,qualification=dict(usable=False,role='W59 fixed-footprint mechanism task'),
        stage_status='passed' if passed else 'technical_interruption' if reason=='execution_error'
          else 'mechanism_criteria_unmet' if reason=='stage_horizon_reached' else 'physical_failure',
        mechanism=dict(split=spec['split'],mode=mission.mode,requested=12,achieved=n,targets=targets,
          body_traversals=len(traversals),home_stable=home, source_loads=len(chains),
          outcomes=dict(Counter(r['outcome'] for r in chains)), ledger=ledger))
    return result


def original_route(raw,result,output):
    rows = cell_chains(raw)
    events = result['transitions']
    deadline = result['simulation_s']
    passages = {'outbound':[], 'return':[]}
    for j,r in enumerate(events):
        if r['phase'] in passages:
            passages[r['phase']].append((r['time_s'], events[j+1]['time_s'] if j+1<len(events) else deadline))
    trips = result['evaluation_stage'].get('target_circuits') or 1
    bins = []
    for j in range(trips):
        for phase,v in passages.items():
            interval=v[j] if j<len(v) else None
            sources = [] if interval is None else [r for r in rows if interval[0]<=r['source_s']<interval[1]]
            for band in range(7):
                for side in [-1,1]:
                    these = [r for r in sources if 40+10*band<=r['key'][0]<50+10*band and
                             (1 if r['key'][1]>=0 else -1)==side]
                    bins.append(dict(circuit=j+1,phase=phase,band=band,side=side,
                        phase_entered=interval is not None, source_loads=len(these),
                        cross_positive_in_passage=sum(r['outcome']=='positive_cross_foot_overlap' and
                        r['target_s']<=interval[1] for r in these)))
    with gzip.open(output.with_suffix('.jsonl.gz'),'wt') as f:
        for r in rows:
            f.write(json.dumps(r,allow_nan=False)+'\n')
    write(output,dict(attempt=raw.name,quantification='All plant loads plus all prespecified bands, sides and passages.',
        source_loads=len(rows),outcomes=dict(Counter(r['outcome'] for r in rows)),route_slots=len(bins),bins=bins))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    original_route(a.raw,json.loads(a.result.read_text()),a.output)
