"""W57 contact requests and material force planning on the r60 F controller."""
import json
from collections import Counter

import numpy as np

from s1_w54_f_r52_backend import STAND_PHASES
from s1_w56_f_r60_backend import FController as R60


class FController(R60):
    def __init__(self, model, data, feet, base, out, config):
        super().__init__(model, data, feet, base, out, config)
        self.timing_agent = None
        self.last_wbc_time_s = None
        self.swing_ids = np.zeros(4, int)
        self.touch_ids = np.zeros(4, int)
        self.pending_touch = [None]*4
        self.request_count = 0
        self.request_record = None
        self.plan_publication = None
        self.planner_mode = config['w57']['planner']
        self.planner_history = config['w57'].get('planner_history', 'full')
        self.manifest.update(
            w57_planner=self.planner_mode, w57_planner_history=self.planner_history,
            timing_requests='handover-requests.jsonl',
            timing_comparison='Same realized sensor/WBC sequence, with the original time request and ramp.',
            candidate_records='material-candidates.jsonl' if self.planner_mode == 'P1' else None)
        self._write_manifest()

    def _write_request(self, record):
        with (self.record_dir / 'handover-requests.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def timing_query_plan(self, t, mq, mission):
        """Target the first future touchdown for each leg on the just-published base schedule."""
        events = self._forecast_events(t, t + 0.85, mq, mission)
        publication = f'W57-base-{self.request_count}'
        plan = []
        for leg, choices in enumerate(events):
            next_event = next((event for event in choices if event[1] > t + 1.e-9), None)
            row = dict(leg=leg, event_id=None, target_time_s=None, xy=None,
                       publication_time_s=float(t), publication_id=publication,
                       valid=False, source='unavailable')
            if next_event is not None:
                start, end, event = next_event
                source = ('committed_swing' if event is self.swing[leg] else
                          'touchdown_search' if event.get('search') else 'forecast_pair')
                target_id = (self.pending_touch[leg] if source == 'committed_swing' else
                             self.pending_touch[leg] or self._recontact_id(leg)
                             if source == 'touchdown_search' else self._swing_id(leg, 1))
                row.update(event_id=target_id, target_time_s=float(end),
                           xy=event['landing'][:2].tolist(), valid=True, source=source)
            plan.append(row)
        self.plan_publication = plan
        return plan

    def _new_handover_request(self, t, mission, mq, material_info, grf):
        if self.timing_agent is not None and not self.timing_agent.has_next_interval(t):
            self._write_request(dict(stage='natural_horizon_no_interval', time_s=float(t),
                                     planned_pair=self.next_pair.tolist(), sampled=False))
            return
        self.request_count += 1
        nominal = float(self.next_liftoff)
        active = self.timing_agent is not None
        finite = self.last_wbc_time_s is not None and np.isfinite(
            [self.last_body_residual, self.last_angular_residual]).all()
        self.plan_publication = self.timing_query_plan(t, mq, mission)
        action = (float(self.timing_agent.request(self, t, mission, mq, material_info, grf, nominal))
                  if active and finite else 0.)
        if not -1. <= action <= 1.:
            raise ValueError(f'Unbounded timing head output {action}')
        requested = nominal + .050 * action
        self.next_liftoff = requested
        self.gate_blocks = Counter()
        self.zero_eligible = None
        self.zero_transfer_start = self.transfer_start
        self.transfer_changed = False
        self.request_record = dict(
            decision_id=self.request_count, sample_time_s=float(t), nominal_due_s=nominal,
            base_due_s=nominal, requested_due_s=requested, planned_pair=self.next_pair.tolist(),
            action=action, sampled=bool(active and finite), last_wbc_time_s=self.last_wbc_time_s,
            sample_reason=('timing_head' if active and finite else
                           'unpublished_wbc_residual' if active else 'zero_new_action'),
            material_publication_s=float(material_info['material_observer'].time),
            plan_publication=self.plan_publication)
        self._write_request(dict(stage='request', **self.request_record))

    def _record_handover_gate(self, t, all_ground, support, measured, solved, released):
        record = self.request_record
        if record is None:
            return
        cfg = self.cfg['contact_events']
        ramp_s = cfg['load_transfer_ramp_s']
        if all_ground and self.zero_transfer_start is None:
            self.zero_transfer_start = max(float(t), record['nominal_due_s'] - .05)
        z = (np.clip((t - self.zero_transfer_start) / ramp_s, 0., 1.)
             if self.zero_transfer_start is not None else 0.)
        zero_fraction = z**3 * (10. - 15.*z + 6.*z*z)
        self.transfer_changed |= abs(zero_fraction - self.transfer_fraction) > 1.e-10
        weight = self.mass * 9.81
        raw = dict(
            prior_swing=bool(np.all(self.event_mode == 'stance')),
            confirmed=bool(np.all(self.confirmed[support])),
            bearing=bool(np.all(self.load_bearing[support])),
            measured=measured >= cfg['handover_measured_bodyweight_fraction']*weight,
            solved=solved >= cfg['handover_solved_bodyweight_fraction']*weight,
            release=released <= cfg['release_solved_bodyweight_fraction']*weight,
            body=self.last_body_residual <= cfg['handover_body_accel_residual_m_s2'],
            angular=self.last_angular_residual <= cfg['handover_angular_accel_residual_rad_s2'])
        if (self.zero_eligible is None and t >= record['nominal_due_s'] - 1.e-10 and
                zero_fraction >= 1.-1.e-10 and all(raw.values())):
            self.zero_eligible = float(t)
        if t >= record['requested_due_s'] - 1.e-10:
            failed = [name for name, passed in raw.items() if not passed]
            if self.transfer_fraction < 1.-1.e-10:
                failed.append('transfer_ramp')
            self.gate_blocks.update(failed)

    def _swing_id(self, leg, offset=0):
        return f'{self.out.name}/leg-{leg}/swing-{self.swing_ids[leg]+offset}'

    def _recontact_id(self, leg):
        return f'{self.out.name}/leg-{leg}/contact-{self.touch_ids[leg]+1}'

    def _update_schedule(self, t, data, mq, mission, soft, material_info, observed, grf):
        if mission.phase in STAND_PHASES and self.request_record is not None:
            self.zero_transfer_start = None
        return super()._update_schedule(t, data, mq, mission, soft, material_info, observed, grf)

    def _event(self, t, kind, legs, **extra):
        if kind == 'actual_touchdown':
            leg = int(legs[0])
            extra['target_event_id'] = self.pending_touch[leg] or self._recontact_id(leg)
            self.touch_ids[leg] += 1
            self.pending_touch[leg] = None
        elif kind == 'liftoff_committed':
            for leg in legs:
                self.swing_ids[leg] += 1
                self.pending_touch[leg] = self._swing_id(leg)
            extra['target_event_ids'] = [self.pending_touch[leg] for leg in legs]
        super()._event(t, kind, legs, **extra)
        if kind != 'liftoff_committed' or self.request_record is None:
            return
        # A late accepted action has a witnessed earlier zero-gate instant.
        # An early accepted action can precede the first possible zero gate.
        earlier = self.zero_eligible is None
        later = self.zero_eligible is not None and t - self.zero_eligible > 1.e-9
        changed = bool(earlier or later)
        record = dict(**self.request_record, actual_liftoff_s=float(t),
                      accepted_s=float(t), first_zero_eligible_s=self.zero_eligible,
                      effective=changed, mask=not changed, transfer_fraction_changed=bool(self.transfer_changed),
                      gate_blocks=dict(self.gate_blocks), accepted_gate=dict(self.handover_diagnostics),
                      reason='earlier_than_zero_gate' if earlier else
                             'later_than_zero_gate' if later else 'same_gate_tick')
        self._write_request(dict(stage='executed', **record))
        if self.timing_agent is not None and record['sampled']:
            self.timing_agent.executed(t, record)
        self.request_record = None

    def wbc(self, t, data, mq, mission, scheduled, observed, grf, soft, material_info):
        action = super().wbc(t, data, mq, mission, scheduled, observed, grf, soft, material_info)
        self.last_wbc_time_s = float(t)
        if getattr(self, 'w57_new_plan_call', None) == self.plan_count:
            with (self.record_dir/'material-execution.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(call=self.plan_count, time_s=float(t),
                    event=self.event_number, mapped_input_rad=action.tolist(),
                    wbc_force_N=self.last_force.tolist()))+'\n')
            self.w57_new_plan_call=None
        return action

    def finish(self, final_time, reason):
        if self.request_record is not None:
            self._write_request(dict(stage='censored', actual_liftoff_s=None, accepted_s=None,
                                     physical_end_s=float(final_time), end_reason=str(reason),
                                     first_zero_eligible_s=self.zero_eligible,
                                     gate_blocks=dict(self.gate_blocks), **self.request_record))
        return super().finish(final_time, reason)

    def select_force_plan(self, *args):
        from s1_w57_planning import select_online
        return select_online(self, *args)
