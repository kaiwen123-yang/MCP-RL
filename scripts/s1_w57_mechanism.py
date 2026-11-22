"""World-fixed cross-foot revisit task and independent physical-event scoring."""
import json
from collections import defaultdict

import numpy as np

from s1_go2_r11_runtime import Mission


class RevisitMission(Mission):
    def __init__(self, task, spec, mode):
        super().__init__(dict(task))
        self.spec = spec
        self.mode = mode
        self.center = float(spec['center_park_x_m'])
        self.edges = np.asarray(spec['body_reversal_x_m'])
        self.bounds = np.asarray(spec['target_strip_x_m'])
        self.half_foot = np.sqrt(.001) / 2
        self.completed = []
        self.cycle_start = None
        self.clear_since = None
        self.parked_this_cycle = False
        self.target_clear = False
        self.returning_home = False
        self.active_goal = np.zeros(2)

    def measure_feet(self, xy, fz):
        covered = (xy[:, 0] + self.half_foot > self.bounds[0]) & (xy[:, 0] - self.half_foot < self.bounds[1])
        ylo, yhi = self.spec['target_strip_y_m']
        covered &= (xy[:, 1] + self.half_foot > ylo) & (xy[:, 1] - self.half_foot < yhi)
        self.target_clear = not np.any(covered & (fz > 0.))

    def observation_goal_xy(self):
        if self.returning_home:
            key = 'turnaround_xy_m' if self.phase == 'turn' else 'return_xy_m'
            return np.asarray(self.task[key])
        return self.active_goal

    def change_phase(self, phase, t, pos, velocity, ori, omega, standing):
        if phase == self.phase and self.transitions:
            return
        self.phase = phase
        self.transitions.append(dict(time_s=t, phase=phase, cycle=len(self.completed)+1,
                                     position_m=pos.copy(), velocity_m_s=velocity.copy(), rpy_rad=ori.copy(),
                                     omega_rad_s=omega.copy(), all_feet_contact=bool(standing)))

    def update(self, t, dt, pos, velocity, ori, omega, standing):
        if self.returning_home:
            return super().update(t, dt, pos, velocity, ori, omega, standing)
        state = (t, pos, velocity, ori, omega, standing)
        if t < self.task['initial_stand_s']:
            self.change_phase('initial_stand', *state)
            return self.body_target.copy(), self.heading
        if self.cycle_start is None:
            self.cycle_start = t
        sign = 1 if len(self.completed) % 2 == 0 else -1
        end_x = self.edges[int(sign > 0)]
        parking = self.mode == 'recovered' and not self.parked_this_cycle
        target = self.center if parking else end_x
        self.active_goal = np.array([target, 0.])
        moving = 'outbound' if sign > 0 else 'return'
        if parking and (self.phase == 'far_hold' or
                        (abs(pos[0]-target) < .045 and abs(velocity[0]) < .08)):
            self.change_phase('far_hold', *state)
            self.task['turnaround_xy_m'] = [self.center, 0.]
            if self.target_clear:
                if self.clear_since is None:
                    self.clear_since = t
            else:
                self.clear_since = None
            if self.clear_since is not None and t-self.clear_since >= self.spec['free_target_hold_s']:
                self.parked_this_cycle = True
                self.change_phase(moving, *state)
                target = end_x
                self.active_goal[0] = target
        else:
            self.change_phase(moving, *state)
        if not parking and sign*(pos[0]-end_x) >= 0:
            self.completed.append(dict(cycle=len(self.completed)+1, start_s=self.cycle_start,
                                       end_s=t, direction=int(sign), free_hold_start_s=self.clear_since))
            self.cycle_start = t
            self.clear_since = None
            self.parked_this_cycle = False
            if len(self.completed) == self.spec['cycles']:
                self.returning_home = True
                self.task['turnaround_xy_m'] = [float(end_x), 0.]
                self.change_phase('turn', *state)
                return super().update(t, dt, pos, velocity, ori, omega, standing)
            sign = -sign
            target = self.center if self.mode == 'recovered' else self.edges[int(sign > 0)]
            self.active_goal[0] = target
            self.change_phase('outbound' if sign > 0 else 'return', *state)
        if self.phase == 'far_hold':
            self.v[:] = np.r_[np.clip(.8*(self.active_goal-pos[:2]), -.1, .1), 0.]
            self.w[:] = 0.
            self.body_target[:2] = self.active_goal
        else:
            cap = self.task['command_speed_m_s']
            vx = float(np.clip(.8*(target-pos[0]), -cap, cap)) if (
                self.mode == 'recovered' and not self.parked_this_cycle) else sign*cap
            desired = np.array([vx, np.clip(-.8*pos[1], -.1, .1), 0.])
            self.v += np.clip(desired-self.v, -.5*dt, .5*dt)
            self.w[2] += np.clip(np.clip(-1.5*ori[2], -.5, .5)-self.w[2], -dt, dt)
            self.body_target[:2] += self.v[:2]*dt
        return self.body_target.copy(), self.heading


def matched_pairs(events_path, windows, spec, mode):
    loads, releases = {}, {}
    pairs = []
    lo, hi = spec['target_strip_x_m']
    ylo, yhi = spec['target_strip_y_m']
    for line in events_path.open():
        event = json.loads(line)
        key = tuple(event['world_key'])
        x, y = (np.asarray(key)+.5)*.02
        if not (lo <= x < hi and ylo <= y < yhi):
            continue
        prior_id = event['previous_load_event_id']
        if event['type'] == 'release':
            releases[(key,prior_id)] = event
            continue
        if prior_id in loads and (key,prior_id) in releases:
            source = loads[prior_id]
            release = releases[key,prior_id]
            leg, next_leg = source['leg'], event['leg']
            selected = next((w for w in windows if w['start_s'] <= source['time_s']
                             and event['time_s'] <= w['end_s']), None)
            gap = event['time_s']-release['time_s']
            if (selected and tuple(source['world_key']) == key and next_leg == (leg+2) % 4
                    and (leg < 2) == (selected['direction'] > 0) and gap > 0.):
                pairs.append(dict(cycle=selected['cycle'], key=list(key), source_id=prior_id,
                                  release_id=release['event_id'], target_id=event['event_id'],
                                  source_leg=leg, target_leg=next_leg, source_s=source['time_s'],
                                  release_s=release['time_s'], target_s=event['time_s'], gap_s=gap,
                                  condition_met=(gap < spec['short_gap_upper_s'] if mode == 'short'
                                                 else gap >= spec['free_target_hold_s']),
                                  release_q=release['q'], preload_q=event['q_before_step'],
                                  source_N=source['normal_N'], target_N=event['normal_N']))
        loads[event['event_id']] = event
    return pairs


def add_footprint_overlap(out, pairs):
    # The recorded material state at event time t used the left boundary t-dt.
    # The final still-loaded footprint of a release is one sample before that.
    needed = defaultdict(set)
    dt = .0002
    for p in pairs:
        for time_s, leg in [(p['source_s'],p['source_leg']), (p['release_s']-dt,p['source_leg']),
                            (p['target_s'],p['target_leg'])]:
            tick = round(time_s/dt)-1
            needed[tick//5000].add((tick,leg))
    xy = {}
    for chunk, queries in needed.items():
        with np.load(out/f'material-{chunk:04d}.npz') as z:
            foot = z['foot']
        for tick,leg in queries:
            row=foot[tick % 5000]
            if abs(row[0]-tick*dt)>1.e-6:
                raise ValueError('Material boundary clock differs from event tick')
            xy[tick,leg] = row[6+3*leg:8+3*leg]
    half = np.sqrt(.001)/2
    for p in pairs:
        target = xy[round(p['target_s']/dt)-1,p['target_leg']]
        lower = np.asarray(p['key'])*.02
        areas=[]
        for tm in (p['source_s'],p['release_s']-dt):
            prior = xy[round(tm/dt)-1,p['source_leg']]
            left = np.maximum(np.maximum(target-half,prior-half),lower)
            right = np.minimum(np.minimum(target+half,prior+half),lower+.02)
            areas.append(float(np.prod(np.maximum(right-left,0.))))
        p['exact_overlap_m2'] = max(areas)


def score_revisits(result, c, mission, out):
    spec = mission.spec
    windows = list(mission.completed)
    if mission.cycle_start is not None and len(windows)<spec['cycles']:
        windows.append(dict(cycle=len(windows)+1, start_s=mission.cycle_start, end_s=result['simulation_s'],
                            direction=(1 if len(windows)%2==0 else -1), incomplete_traversal=True))
    pairs = matched_pairs(out/'material-events.jsonl', windows, spec, mission.mode)
    add_footprint_overlap(out, pairs)
    accepted = [p for p in pairs if p['condition_met'] and p['exact_overlap_m2']>0.]
    successful = sorted({p['cycle'] for p in accepted} & {w['cycle'] for w in mission.completed})
    with (out/'w57-revisit-pairs.jsonl').open('w') as f:
        for p in pairs:
            f.write(json.dumps(p)+'\n')
    gaps=[p['gap_s'] for p in accepted]
    full = result['simulation_s'] >= c['r11_stage']['duration_s']-1.e-7
    passed = bool(result['reason']=='stage_horizon_reached' and full and len(successful)==spec['cycles']
                  and mission.phase=='complete' and result['hold']['final_stable_s']>=2.-1.e-7)
    r = result['reason']
    result.update(success=passed, sustained_execution_success=passed, original_mission_success=None,
                  stage_status='passed' if passed else ('physical_material_failure' if r.startswith((
                      'body_fall','invalid_contact','material_domain_exit')) else 'technical_interruption'
                      if r=='execution_error' else 'mechanism_criteria_unmet'),
                  case_key=c['case_key'], case_identity=c['case_identity'],
                  evaluation_stage=c['r11_stage'], qualification_usable=False,
                  qualification=dict(usable=False,role='independent mechanism task',predecessor_attempt=None),
                  mechanism=dict(mode=mission.mode,cycles_requested=spec['cycles'],traversals=mission.completed,
                                 achieved_cycles=successful,confirmed_cycle_count=len(successful),
                                 positive_overlap_event_rows=len(accepted), home_stable=mission.phase=='complete',
                                 gap_min_median_max_s=np.quantile(gaps,[0,.5,1]).tolist() if gaps else None,
                                 pair_evidence='w57-revisit-pairs.jsonl', raw_pairs=len(pairs)))
    return result
