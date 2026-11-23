"""Causal contact-event observations and original-task credit for the W57 timing head."""
import json
from pathlib import Path

import numpy as np
import mujoco

from s1_w57_learning import SPEC, EventPolicy, jsonline


PHASES = SPEC['normalization']['phase_order']
STATUSES = SPEC['normalization']['statuses']


def blind_observation(observation):
    selected = observation.copy()
    selected[24:36] = 0.
    status = selected[59:].reshape(4, 6)
    if SPEC.get('experiment') == 'W58':
        status[:, 2] = status[:, 2:5].sum(axis=1)
        status[:, 3:5] = 0.
    else:
        status[:, 2:] = 0.
    return selected


def material_query(observer, targets, t):
    ground = observer.ground
    # Only the private geometry map may create as-yet unvisited cells.
    geometry = ground.clone()
    xy = np.zeros((4, 2))
    rows = []
    for target in targets:
        leg = target['leg']
        valid = bool(target['valid'] and observer.windows and target['target_time_s'] > t + 1e-9)
        row = dict(target, query_time_s=float(t), target_event_id=target['event_id'],
                   target_publication_age_s=float(t - target['publication_time_s']),
                   observer_publication_id=int(observer.windows), observer_publication_s=float(observer.time),
                   observer_publication_age_s=float(t - observer.time), query_valid=valid,
                   soft_mask=None, status='missing', cells=[])
        if valid:
            xy[leg] = target['xy']
            row['soft_mask'] = bool(ground.interior(xy[leg]))
            if row['soft_mask']:
                indices, areas = geometry.footprint(xy[leg], create=True)
                previously_loaded = []
                currently_loaded = False
                for index, area in zip(indices, areas):
                    key = tuple(np.floor(np.asarray(geometry.xy[index])/geometry.cell).astype(int))
                    history = observer.history(key)
                    row['cells'].append(dict(key=[int(key[0]), int(key[1])], area_m2=float(area),
                                             **history))
                    if history['ever_loaded']:
                        previously_loaded.append(history)
                    currently_loaded |= history['loaded']
                if currently_loaded:
                    row['status'] = 'loaded'
                elif not previously_loaded:
                    row['status'] = 'virgin'
                elif all(h['release_time_s'] is not None and observer.time-h['release_time_s'] >= 6.-1e-9
                         for h in previously_loaded):
                    row['status'] = 'recovered'
                else:
                    row['status'] = 'residual'
            else:
                row['status'] = 'outside'
        rows.append(row)
    # Outside placeholders at (0,0) have zero footprint and are explicitly masked.
    heights, _ = observer.recovery(xy)
    flags = np.zeros((4, len(STATUSES)))
    for row in rows:
        leg = row['leg']
        flags[leg, STATUSES.index(row['status'])] = 1.
        row['free_recovery_m'] = heights[leg].tolist() if row['query_valid'] else None
        row['future_sample_time_s'] = [float(observer.time+h) for h in SPEC['query']['horizons_s']]
    return heights, flags, rows


def features(controller, t, mission, mq, grf, observer, targets, base_due, previous_action):
    height, flags, queries = material_query(observer, targets, t)
    pair = np.zeros(4)
    pair[controller.next_pair] = 1.
    phase = np.zeros(len(PHASES))
    phase[PHASES.index(mission.phase)] = 1.
    gates = controller.cfg['contact_events']
    inputs = np.concatenate((
        np.asarray(grf)[:, 2]/(controller.mass*9.81/4), mq['foot_velocity'][:, 2]/.15,
        controller.confirmed, controller.load_bearing, pair, controller.event_mode != 'swing',
        height.ravel()/.001, [(base_due-t)/.25, controller.transfer_fraction,
        controller.last_body_residual/gates['handover_body_accel_residual_m_s2'],
        controller.last_angular_residual/gates['handover_angular_accel_residual_rad_s2']],
        np.asarray(mission.v)/.25, mq['com_velocity']/.25, mq['omega'],
        [previous_action], phase, flags.ravel())).astype('float32')
    return inputs, queries


def original_reward(start, end, failed):
    dt, progress, indentation, slip, cap_change, laps, active_time = (end-start).tolist()
    terms = dict(legal_progress=5*progress, elapsed=-dt/60, indentation=-indentation/60,
                 loaded_slip=-.1*slip/60, old_cap_change=-.01*cap_change,
                 legal_laps=10*laps, failure=-10.*int(failed))
    quantities = dict(dt_s=dt, signed_legal_route=progress, indentation_sq_s=indentation,
                      loaded_slip_sq_s=slip, old_cap_action_change=cap_change,
                      legal_circuits=laps, active_old_cap_s=active_time)
    return sum(terms.values()), terms, quantities


class W57TimingHead:
    def __init__(self, controller, joint, config, out, attempt):
        self.controller = controller
        self.joint = joint
        self.out = Path(out)
        self.attempt = attempt
        self.policy = EventPolicy(config)
        self.previous_action = 0.
        self.pending = None
        self.last_targets = None
        self.count = 0
        self.closed = 0
        self.training = 0
        self.witnessed = 0
        self.ramp_only = 0
        self.censored = 0

    def has_next_interval(self, t):
        return t < self.joint.session_s - 1e-8

    def selected(self, inputs):
        return blind_observation(inputs) if self.policy.head == 'blind' else inputs

    def request(self, controller, t, mission, mq, material_info, grf, base_due):
        self.last_targets = controller.plan_publication
        inputs, queries = features(controller, t, mission, mq, grf, material_info['material_observer'],
                                    self.last_targets, base_due, self.previous_action)
        selected = self.selected(inputs)
        snapshot = self.joint.snapshot(t, mission)
        self.close_pending(t, selected, snapshot)
        identifier = f'{self.attempt}/decision-{controller.request_count}'
        sample = self.policy.action(selected)
        action = sample['action']
        delta = (self.policy.mean(inputs) - self.policy.mean(blind_observation(inputs))
                 if self.policy.head == 'full' else None)
        self.pending = dict(decision_id=identifier, decision_number=controller.request_count,
                            sample=sample, observation=selected, snapshot=snapshot, execution=None,
                            history_status=[r['status'] for r in queries])
        self.previous_action = action
        self.count += 1
        jsonline(self.out/'timing-decisions.jsonl', dict(
            decision_id=identifier, decision_number=controller.request_count, sample_time_s=float(t),
            phase=mission.phase, base_due_s=float(base_due), requested_due_s=float(base_due+.05*action),
            last_wbc_time_s=controller.last_wbc_time_s, queries=queries, selected_observation=selected.tolist(),
            head=self.policy.head, full_minus_blind_deterministic_request_s=(None if delta is None else .05*delta),
            **sample))
        return action

    def executed(self, t, record):
        if self.pending is None or self.pending['decision_number'] != record['decision_id']:
            raise ValueError('Handover execution has no matching sampled event.')
        keys = ('accepted_s','actual_liftoff_s','first_zero_eligible_s','effective','mask',
                'transfer_fraction_changed','gate_blocks','reason','requested_due_s','nominal_due_s')
        self.pending['execution'] = {name:record[name] for name in keys}

    def close_pending(self, t, next_obs, snapshot, terminated=False, truncated=False, end_reason=None):
        pending = self.pending
        if pending is None:
            return
        dt = float(t-pending['snapshot'][0])
        if dt <= 0:
            if terminated or truncated:
                jsonline(self.out/'timing-zero-duration.jsonl', dict(
                    decision_id=pending['decision_id'], physical_s=float(t), reason=end_reason))
                self.pending = None
                return
            raise ValueError('A second timing request cannot close a zero-duration transition.')
        reward, terms, metrics = original_reward(pending['snapshot'], snapshot, terminated)
        execution = pending['execution']
        if execution is None:
            self.censored += 1
        else:
            self.witnessed += int(execution['effective'])
            self.ramp_only += int(execution['transfer_fraction_changed'] and not execution['effective'])
        next_value = 0. if terminated else self.policy.value(next_obs)
        sample = pending['sample']
        row = dict(attempt=self.attempt, decision_id=pending['decision_id'], source_dir=str(self.out),
                   case=self.policy.case, head=self.policy.head, version=sample['version'],
                   sampling=sample['sampling'], sample_serial=sample['sample_serial'], rng_seed=sample['rng_seed'],
                   start_s=float(pending['snapshot'][0]), end_s=float(t), dt_s=dt,
                   obs=pending['observation'].tolist(), next_obs=next_obs.tolist(),
                   raw_action=sample['raw_action'], action=sample['action'],
                   logp_raw=sample['logp_raw'], logp_action=sample['logp_action'], value=sample['value'],
                   next_value=next_value, reward=reward, reward_terms=terms, metrics=metrics,
                   execution=execution, history_status=pending['history_status'],
                   terminated=terminated, truncated=truncated, end_reason=end_reason)
        jsonline(self.out/'timing-transitions.jsonl', row)
        self.pending = None
        self.closed += 1
        if sample['sampling'] == 'on_policy':
            self.training += 1
            self.policy.append(row)

    def finish(self, result):
        if self.pending is not None:
            t = float(self.joint.data.time)
            reason = result['reason']
            if reason == 'execution_error':
                # An exception is not a material terminal and has no certified successor.
                jsonline(self.out/'timing-interrupted.jsonl', dict(decision_id=self.pending['decision_id'],
                         simulation_s=t, reason=reason))
                self.pending = None
            else:
                failure = reason.startswith(('material_domain_exit','body_fall','invalid_contact'))
                mission = self.joint.mission
                controller = self.controller
                model,data = self.joint.model,self.joint.data
                mujoco.mj_forward(model,data)
                _, velocity, _, grf = self.joint.measurements()
                Jcom = np.zeros((3,model.nv))
                Jr = np.zeros_like(Jcom)
                mujoco.mj_jacSubtreeCom(model,data,Jcom,controller.base)
                mujoco.mj_jacBody(model,data,None,Jr,controller.base)
                mq = dict(com_velocity=Jcom@data.qvel,omega=Jr@data.qvel,foot_velocity=velocity)
                base_due = controller.next_liftoff if controller.moving else t
                targets = self.last_targets
                inputs, queries = features(controller, t, mission, mq, grf, self.joint.observer,
                                            targets, base_due, self.previous_action)
                # No new action at a terminal or natural time limit. Retain invalid/expired targets.
                jsonline(self.out/'timing-final-observation.jsonl', dict(time_s=t, reason=reason,
                    queries=queries, phase=mission.phase, observation=self.selected(inputs).tolist(),
                    last_wbc_time_s=controller.last_wbc_time_s))
                self.close_pending(t, self.selected(inputs), self.joint.snapshot(t,mission),
                                   terminated=failure, truncated=not failure, end_reason=reason)
        summary = dict(head=self.policy.head, samples_requested=self.count,
                       real_transitions=self.closed, new_training_transitions=self.training,
                       liftoff_changed_on_same_gate_sequence=self.witnessed,
                       only_transfer_ramp_changed=self.ramp_only, unexecuted_requests=self.censored,
                       old_W44_actor_updates=0, schema=SPEC['identity'])
        result['w57_timing'] = summary
        (self.out/'timing-summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def attach_timing(agent, joint, config, out, attempt):
    session = W57TimingHead(agent, joint, config, out, attempt)
    joint.w57_timing = session
    agent.timing_agent = session

