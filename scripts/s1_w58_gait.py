"""Causal terminal swing, contact capture and reference transfer for the W58 G1 identity."""
import json
from pathlib import Path

import mujoco
import numpy as np

from s1_w54_f_r52_backend import curve, quintic
from s1_w57_f_backend import FController as W57Controller


def clearance_correction(position, velocity, reference, landing_z, remaining, control_dt):
    """One held-input tracking deficit, only when a decelerating foot would reach ground early."""
    p, v, a = reference
    if (remaining <= control_dt or velocity[2] >= v[2] or v[2] >= 0. or
            a[2] <= 0. or position[2] >= p[2] or position[2] <= landing_z):
        return 0.
    # With uniform braking to zero over the remaining swing the foot travels v*T/2.
    if position[2] + .5 * velocity[2] * remaining >= landing_z:
        return 0.
    return float(p[2] - position[2] + control_dt * (v[2] - velocity[2]))


class ContactReference:
    """Add a finite C1 offset at each swing, flight-search and measured support transition."""

    def __init__(self, time, reference, duration):
        self.start = time
        self.incoming = reference
        self.duration = duration
        self.mode = None

    def _offset(self, time):
        # All six endpoint boundary values at the end of the transfer are zero.
        if time >= self.start + self.duration:
            return np.zeros(3), np.zeros(3), np.zeros(3)
        return curve(self.coefficients, time - self.start)

    def update(self, time, mode, position, velocity):
        if self.mode != mode:
            if self.mode is None:
                incoming_p, incoming_v, _ = self.incoming
            else:
                off_p, off_v, _ = self._offset(time)
                elapsed = time - self.last_time
                incoming_p = self.last_base_position + elapsed * self.last_base_velocity + off_p
                incoming_v = self.last_base_velocity + off_v
            self.start = time
            self.mode = mode
            self.coefficients = quintic(
                incoming_p - position, incoming_v - velocity, np.zeros(3),
                np.zeros(3), np.zeros(3), np.zeros(3), self.duration)
        self.last_time = time
        self.last_base_position = position.copy()
        self.last_base_velocity = velocity.copy()
        return self._offset(time)


class EarlyTouchGait:
    """Place before the common force controller in the method resolution order for G1."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_force = [None] * 4
        self.contact_bridge = [None] * 4
        self.approach_correction = [None] * 4
        self.manifest['w58_gait'] = json.loads((Path(__file__).resolve().parents[1] /
            'experiments/terrain_memory/s1-w58-gait.json').read_text())
        self._write_manifest()

    def _start_swing(self, t, leg, *args):
        self.first_force[leg] = self.contact_bridge[leg] = self.approach_correction[leg] = None
        return super()._start_swing(t, leg, *args)

    def sample_swing_touch(self, force_time, data, contacts, grf):
        """Latch the first force at its actual evaluation time. The 2 ms actuator hold is unchanged."""
        for leg, swing in enumerate(self.swing):
            if (self.event_mode[leg] == 'swing' and self.first_force[leg] is None and
                    swing['apex'] - 1.e-9 <= force_time < swing['end'] - 1.e-9 and contacts[leg] and
                    grf[leg, 2] > self.cfg['contact_events']['touch_off_N']):
                twist = np.empty(6)
                mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_GEOM,
                                         int(self.feet[leg]), twist, 0)
                self.first_force[leg] = dict(force_time_s=float(force_time),
                    contact_center=data.geom_xpos[self.feet[leg]].copy().tolist(),
                    foot_velocity_m_s=twist[3:].copy().tolist(), measured_Fz_N=float(grf[leg, 2]))

    def _offset_at(self, leg, t):
        start, duration, coeff = self.approach_correction[leg]
        return curve(coeff, np.clip(t - start, 0., duration))

    def _swing_reference(self, t, leg):
        ref = super()._swing_reference(t, leg)
        if self.approach_correction[leg] is None:
            return ref
        return tuple(x + y for x, y in zip(ref, self._offset_at(leg, t)))

    def _future_contact_feet(self, t, times, mq, mission):
        contact, feet = super()._future_contact_feet(t, times, mq, mission)
        # The inherited event forecast uses the same live search/contact state and measured
        # force at the boundary. Only this already-committed descent has a correction.
        for leg, correction in enumerate(self.approach_correction):
            if correction is not None and self.event_mode[leg] == 'swing':
                mask = (times > t + 1.e-9) & (times >= correction[0]) & (times < self.swing[leg]['end'])
                for k in np.flatnonzero(mask):
                    feet[k, leg] += self._offset_at(leg, times[k])[0]
        return contact, feet

    def _update_schedule(self, t, data, mq, mission, soft, material_info, observed, grf):
        dt = self.cfg['wbc']['dt_s']
        for leg, swing in enumerate(self.swing):
            if self.event_mode[leg] != 'swing' or t < swing['apex']:
                continue
            ref = self._swing_reference(t, leg)
            force = self.first_force[leg]
            if force is not None or t >= swing['end']:
                self.contact_bridge[leg] = ContactReference(
                    t, ref, self.cfg['contact_events']['load_transfer_ramp_s'])
                if force is not None:
                    swing['contact_point'] = np.array(force['contact_center'])
                    self.event_mode[leg] = 'search'
                    self._event(t, 'early_touch_capture', [leg],
                        target_event_id=self.pending_touch[leg], planned_end_s=swing['end'],
                        incoming_reference_m=ref[0].tolist(), incoming_reference_m_s=ref[1].tolist(),
                        control_position_m=mq['feet'][leg].tolist(),
                        control_velocity_m_s=mq['foot_velocity'][leg].tolist(),
                        control_measured_contact=bool(observed[leg]), **force)
                # With no early touch the original scheduled search changes the mode below.
            elif self.approach_correction[leg] is None and not observed[leg]:
                remaining = swing['end'] - t
                height = clearance_correction(mq['feet'][leg], mq['foot_velocity'][leg],
                                              ref, swing['landing'][2], remaining, dt)
                if height:
                    zero = np.zeros(3)
                    coeff = quintic(zero, zero, zero, np.array([0., 0., height]),
                                    zero, zero, remaining)
                    self.approach_correction[leg] = (t, remaining, coeff)
                    self._event(t, 'terminal_clearance_correction', [leg],
                        target_event_id=self.pending_touch[leg], reference_offset_m=height,
                        remaining_s=remaining, target_surface_center_m=float(swing['landing'][2]),
                        control_z_m=float(mq['feet'][leg, 2]),
                        control_vz_m_s=float(mq['foot_velocity'][leg, 2]),
                        reference_z_m=float(ref[0][2]), reference_vz_m_s=float(ref[1][2]))
        # Only the current observed force can activate WBC reaction. The actual-touch and
        # bearing counters, support-pair load and solved-release checks run in the base.
        return super()._update_schedule(t, data, mq, mission, soft, material_info, observed, grf)

    def contact_reference(self, t, leg, mode, pref, vref, target, mq):
        bridge = self.contact_bridge[leg]
        if bridge is None:
            return pref, vref, target
        if mode == 2 and 'contact_point' in self.swing[leg]:
            # Acquire the point that actually touched. Preserve the existing search speed.
            delta_xy = self.swing[leg]['contact_point'][:2] - pref[:2]
            pref = pref.copy()
            pref[:2] += delta_xy
            target[:2] += self.cfg['wbc']['stance_kp'] * delta_xy
        if mode == 2:
            # Give the velocity search an integrable normal reference between observations.
            pref = pref.copy()
            if bridge.mode == mode:
                pref[2] = bridge.last_base_position[2] + (t - bridge.last_time) * vref[2]
            else:
                pref[2] = mq['feet'][leg, 2]
        dp, dv, da = bridge.update(t, int(mode), pref, vref)
        gains = np.full(3, self.cfg['wbc']['stance_kp'])
        if mode == 3:
            gains[2] = self.cfg['wbc']['material_normal_kp']
        # Existing soft normal inversion, rolling Jacobians, measured support preload and
        # torque-constrained QP retain ownership. Add only the transient reference offset.
        target = target + gains * dp + self.cfg['wbc']['stance_kd'] * dv + da
        return pref + dp, vref + dv, target


class FController(EarlyTouchGait, W57Controller):
    pass
