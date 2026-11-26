"""One event-clock request consumed by the original contact schedule, force QP and WBC."""
from collections import Counter
import json

import numpy as np
from s1_w57_f_backend import FController as W57Controller

NOMINAL = np.array([0., .050, 0.])
LOWER = np.array([-.050, .040, -.10])
UPPER = np.array([ .050, .080,  .10])
SCALE = np.array([.050, .030, .10])
FIXED = np.array([0., .060, .05])


def compose_request(base, residual):
    raw = np.asarray(base) + SCALE * np.tanh(residual)
    return np.clip(raw, LOWER, UPPER), raw


class FController(W57Controller):
    def __init__(self, model, data, feet, base, out, config):
        super().__init__(model, data, feet, base, out, config)
        self.settings = config['w59']
        from s1_w59_model_cells import CausalCellMap
        self.causal_cells = CausalCellMap()
        self.w59_data = data
        self.macro_request = NOMINAL.copy()
        self.preference_pair = None
        self.snapshot_buffer = None
        self.branch_override = None
        self.execution_wm = None
        self.w59_learner = None
        self.w59_action_reference = np.zeros((4, 3))
        self.w59_pair_applicable = False
        if self.settings['arm'].startswith('P3'):
            from s1_w59_planning import EventPlanner
            self.execution_wm = EventPlanner(self.settings)
        self.manifest.update(w59_action_fields=['due_offset_s','transfer_ramp_s','pair_load_preference_rho'],
                             w59_exchange='One request per original new-handover boundary, before force planning',
                             w59_force_log='w55-f/control-*.npz', legacy_timing_head=False)
        self._write_manifest()

    def update(self,t,data,mission,contact,grf,soft,material_info=None,replan=False):
        self.causal_cells.observe(t,data.geom_xpos[self.feet],grf)
        return super().update(t,data,mission,contact,grf,soft,material_info,replan)

    def _new_handover_request(self, t, mission, mq, material_info, grf):
        self.request_count += 1
        nominal_due = float(self.next_liftoff)
        self.cfg['contact_events']['load_transfer_ramp_s'] = float(NOMINAL[1])
        self.plan_publication = self.timing_query_plan(t, mq, mission)
        origin = self.settings['arm']
        state = None
        if self.execution_wm is None:
            base = FIXED.copy() if origin == 'Fixed-action' else NOMINAL.copy()
            planning = {}
        else:
            state, base, planning = self.execution_wm.choose(self,t,mission,mq,material_info,grf)
        request = base
        if self.w59_learner is not None:
            request = self.w59_learner.request(self,t,mission,mq,material_info,grf,base,state)
            origin = 'R3'
        if self.branch_override is not None:
            expected_id, expected_time, replacement = self.branch_override
            if expected_id != self.request_count or abs(expected_time-t)>1e-9:
                raise ValueError('Counterfactual did not reach its original legal request')
            self.branch_override = None
            if replacement is not None:
                request = np.asarray(replacement, dtype=np.float64)
                origin = 'registered_signed_or_nominal_intervention'
        request = np.asarray(request, dtype=np.float64)
        if request.shape != (3,) or not np.isfinite(request).all() or (request < LOWER).any() or (request > UPPER).any():
            raise ValueError(f'Final request outside W59 authority: {request}')
        self.macro_request = request.copy()
        self.preference_pair = np.setdiff1d(np.arange(4), self.next_pair)
        self.cfg['contact_events']['load_transfer_ramp_s'] = float(request[1])
        self.next_liftoff = nominal_due + float(request[0])
        self.gate_blocks = Counter()
        self.zero_eligible = None
        self.zero_transfer_start = self.transfer_start
        self.transfer_changed = False
        self.request_record = dict(decision_id=self.request_count, sample_time_s=float(t),
            nominal_due_s=nominal_due, base_due_s=nominal_due, requested_due_s=self.next_liftoff,
            planned_pair=self.next_pair.tolist(), incoming_pair=self.preference_pair.tolist(),
            action=float(request[0]/SCALE[0]), macro_action=request.tolist(), p3_base=base.tolist(),
            sampled=False, last_wbc_time_s=self.last_wbc_time_s, sample_reason=origin,
            material_publication_s=float(material_info['material_observer'].time),
            plan_publication=self.plan_publication, planner=planning)
        if self.snapshot_buffer is not None:
            from s1_w59_snapshots import save_request
            save_request(self,t)
        self._write_request(dict(stage='request', **self.request_record))

    def _forecast_events(self,t,end_time,mq,mission):
        events = super()._forecast_events(t,end_time,mq,mission)
        # The currently published request expires when the next outgoing pair lifts.
        self.forecast_handover = next((start for start,_,entry in events[int(self.next_pair[0])]
            if start >= t-1e-10 and not entry.get('search') and 'rise' in entry
            and entry is not self.swing[int(self.next_pair[0])]), np.inf)
        return events

    def _future_contact_feet(self,t,times,mq,mission):
        contact, feet = super()._future_contact_feet(t,times,mq,mission)
        self.w59_horizon_mask = np.zeros(len(times)-1, bool)
        if self._pair_can_bear(self.measured) and self.macro_request[2]:
            pair = self.preference_pair
            self.w59_horizon_mask = contact[:-1,pair].all(axis=1) & (times[:-1] < self.forecast_handover)
        return contact, feet

    def _pair_can_bear(self, active):
        pair = self.preference_pair
        return bool(self.moving and pair is not None and np.all(active[pair]) and
                    np.all(self.confirmed[pair]) and np.all(self.load_bearing[pair]))

    def _solve(self,P,q,A,lo,hi,kind,t,warm=None):
        if kind == 'mpc' and self.macro_request[2] and self.w59_horizon_mask.any():
            # Move the existing regularizer's reference by (+rho mg,-rho mg).
            # The original state objective, cones, equality dynamics and force bounds remain active.
            count = len(self.w59_horizon_mask)
            q = q.copy()
            force_gradient = q[12*(count+1):].reshape(count,4,3)
            offset = 2*self.cfg['mpc']['force_regularization']*self.macro_request[2]*self.mass*9.81
            first,second = self.preference_pair
            force_gradient[self.w59_horizon_mask,first,2] -= offset
            force_gradient[self.w59_horizon_mask,second,2] += offset
        return super()._solve(P,q,A,lo,hi,kind,t,warm)

    def _redistribute(self,forces,mq,active,required=None):
        target, wrench = super()._redistribute(forces,mq,active,required)
        self.w59_pair_applicable = self._pair_can_bear(active)
        self.w59_action_reference = np.zeros((4,3))
        if self.w59_pair_applicable and self.macro_request[2]:
            first,second = self.preference_pair
            offset = self.macro_request[2]*self.mass*9.81
            self.w59_action_reference[[first,second],2] = [offset,-offset]
            target = target + self.w59_action_reference
        self.w59_force_ref = target.copy()
        return target, wrench

    def record_macro_force(self,row,grf):
        row.update(w59_request=self.macro_request.copy(), w59_request_id=self.request_count,
            w59_preference_active=self.w59_pair_applicable,
            w59_reference_offset=self.w59_action_reference.copy(),
            w59_requested_wbc_force=self.w59_force_ref.copy(), w59_measured_grf=grf.copy())
