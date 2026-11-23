"""The W58 route scores twelve predeclared forward load events and retains the eleven transit legs."""
import json
from pathlib import Path

from s1_w57_mechanism import RevisitMission, matched_pairs, add_footprint_overlap


class ForwardRevisitMission(RevisitMission):
    def __init__(self, task, spec, mode):
        super().__init__(task, spec, mode)
        self.preparation_s = {}
        self.verified_clear_s = {}

    def requires_parking(self, direction):
        return self.mode == 'recovered' and direction == 1

    def update(self, t, dt, pos, velocity, ori, omega, standing):
        previous = self.phase
        traversal = len(self.completed)+1
        state = super().update(t, dt, pos, velocity, ori, omega, standing)
        if previous == 'far_hold' and not self.returning_home:
            self.preparation_s[traversal] = self.preparation_s.get(traversal, 0.)+dt
            if self.parked_this_cycle and self.clear_since is not None:
                self.verified_clear_s[traversal] = t-self.clear_since
        return state


def score_targets(result, config, mission, out):
    spec = mission.spec
    completed = {r['cycle']:r for r in mission.completed}
    windows = list(mission.completed)
    nxt = len(completed)+1
    if mission.cycle_start is not None and nxt <= spec['cycles']:
        windows.append(dict(cycle=nxt, start_s=mission.cycle_start, end_s=result['simulation_s'],
                            direction=1 if nxt%2 else -1))
    pairs = matched_pairs(out/'material-events.jsonl', windows, spec, mission.mode)
    pairs = [p for p in pairs if p['cycle']%2 and p['source_leg']==1 and p['target_leg']==3]
    add_footprint_overlap(out, pairs)
    targets = []
    for target in spec['target_events']:
        number=target['traversal']
        matches=[p for p in pairs if p['cycle']==number]
        sufficient_hold=(mission.mode=='short' or mission.verified_clear_s.get(number,0.)>=spec['free_target_hold_s']-1.e-8)
        realized=number in completed and sufficient_hold and any(p['condition_met'] and p['exact_overlap_m2']>0 for p in matches)
        targets.append(dict(**target, completed_traversal=number in completed, realized=realized,
                            preparation_station_s=mission.preparation_s.get(number,0.),
                            continuous_clear_s=mission.verified_clear_s.get(number,0.), matches=matches))
    ledger = out/'w58-target-events.jsonl'
    with ledger.open('w') as stream:
        for target in targets:
            stream.write(json.dumps(target,allow_nan=False)+'\n')
    count=sum(t['realized'] for t in targets)
    complete=result['simulation_s'] >= config['r11_stage']['duration_s']-1.e-7
    reason=result['reason']
    passed=bool(reason=='stage_horizon_reached' and complete and count==12 and
                mission.phase=='complete' and result['hold']['final_stable_s']>=2.-1.e-7)
    status=('passed' if passed else 'physical_material_failure' if reason.startswith(('body_fall','invalid_contact','material_domain_exit'))
            else 'technical_interruption' if reason=='execution_error' else 'mechanism_criteria_unmet')
    result.update(stage_status=status,success=passed,sustained_execution_success=passed,
        original_mission_success=None, case_key=config['case_key'],case_identity=config['case_identity'],
        evaluation_stage=config['r11_stage'],qualification_usable=False,
        qualification=dict(usable=False,role='W58 v2 mechanism route',predecessor_attempt=None),
        mechanism=dict(split=config['w58']['split'],mode=mission.mode,cycles_requested=12,
            confirmed_cycle_count=count, all_target_rows=targets,
            traversals=mission.completed,body_traversals=len(completed),home_stable=mission.phase=='complete',
            preparation_station_s=sum(mission.preparation_s.values()),pair_evidence=ledger.name))
    return result

