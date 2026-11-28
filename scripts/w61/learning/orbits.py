"""A bounded four-foot hold and an original diagonal transfer with C2 Cartesian offsets."""
import json
import time

import numpy as np

from s1_w54_f_r52_backend import STAND_PHASES, assemble_centroidal, assemble_force_horizon, curve, quintic, skew
from s1_w55_f_r55_backend import ENDPOINT_HOLDS, standing_stages
from s1_w59_f_backend import FController as W59

from hybrid import SPEC, action_map


def offsets(rotation, leg, action):
    side = 1. if leg in (0, 2) else -1.
    landing = rotation @ [action[0], side * action[1], 0.]
    landing[2] = 0.
    apex = .5 * landing + [0., 0., action[2]]
    zero = np.zeros(3)
    return landing, quintic(zero, zero, zero, apex, landing/.2, zero, .1), quintic(
        apex, landing/.2, zero, landing, zero, zero, .1)


def apply_offset(swing, rotation, leg, action):
    landing, rise, fall = offsets(rotation, leg, action)
    swing["landing"] = swing["landing"] + landing
    swing["rise"] = swing["rise"] + rise
    swing["fall"] = swing["fall"] + fall
    return landing


def smoothstep(value):
    x = np.clip(value, 0., 1.)
    return x**3 * (10. - 15.*x + 6.*x*x)


def response_context(ctl, mq):
    feet = mq["contact_force_point"]
    G = np.vstack((np.tile(np.eye(3), (1,4)), np.concatenate([skew(f-mq["com"]) for f in feet], axis=1)))
    wrench = ctl.previous_required if hasattr(ctl, "previous_required") else np.r_[0., 0., ctl.mass*9.81, 0., 0., 0.]
    return dict(rotation=mq["R"].tolist(), pair=ctl.next_pair.tolist(), G=G.tolist(), wrench=wrench.tolist(), weight=ctl.mass*9.81)


def nominal_response(context, mode, raw):
    """Published release time, nominal support, C2 foot offsets and frozen wrench redistribution.

    Uses the original WBC force-reference least-squares at two horizon instants. This
    local response models the issued load target and excludes the post-QP actuator command.
    """
    action = action_map(mode, raw)
    pair = np.asarray(context["pair"])
    G, wrench = np.asarray(context["G"]), np.asarray(context["wrench"])
    duration = action[3]
    ramp = .05
    result = [duration/.05]
    for tau in (.035, .06):
        z = 0. if mode else smoothstep((tau-(duration-ramp))/ramp)
        weights = np.ones(4)*(1.+z)
        weights[pair] = 1.-z
        scale = np.repeat(weights, 3)
        f = scale*np.linalg.lstsq(G*scale, wrench, rcond=None)[0]
        result.extend(weights/2.)
        result.extend(f/(context["weight"]*.25*np.sqrt(12)))
    # Analytic C2 curve perturbations at three fixed phases of the requested swing.
    for leg in pair:
        _, rise, fall = offsets(np.asarray(context["rotation"]), leg, action)
        for phase in (.05,.10,.20):
            d = curve(rise if phase<=.1 else fall, phase if phase<=.1 else phase-.1)[0]
            result.extend(d / (np.array([.004,.002,.002])*np.sqrt(18)))
    return np.asarray(result, np.float64)


class FController(W59):
    def __init__(self, *args, **kwargs):
        self.policy_mode = None
        self.trajectory_action = np.zeros(4)
        self.wait_until = None
        self.previous_mode = -1
        self.transfer_issued = False
        self.applied_trajectories = []
        super().__init__(*args, **kwargs)

    def _write_request(self, record):
        # The original B3 request is fixed; the W61 decision and real events have their own records.
        return

    def _new_handover_request(self, t, mission, mq, material_info, grf):
        super()._new_handover_request(t,mission,mq,material_info,grf)
        self.transfer_issued = False
        self.policy_mode = None
        self.trajectory_action = np.zeros(4)

    def _update_schedule(self, t, data, mq, mission, soft, material_info, observed, grf):
        walking = mission.phase not in STAND_PHASES and self.walk_origin is not None
        all_down = np.all((self.event_mode == "stance") & self.confirmed & observed)
        hold_done = self.policy_mode == 1 and t+1.e-10 >= self.wait_until
        eligible = walking and all_down and (self.policy_mode is None or hold_done)
        if eligible:
            self.plan_publication = self.timing_query_plan(t, mq, mission)
            mode, action = self.session.decide(self,t,mission,mq,grf)
            if mode is not None:
                self.previous_mode = mode
                self.policy_mode = mode
                self.trajectory_action = action
                self.wait_until = t + action[3]
                self.transfer_start = None
                self.transfer_fraction = 0.
                self.next_liftoff = self.wait_until + (.052 if mode == 1 else 0.)
        if self.policy_mode == 1:
            # Holding all supports must not unload the next pair while waiting for a legal boundary.
            self.next_liftoff = max(self.next_liftoff, t+.052)
            self.transfer_start = None
            self.transfer_fraction = 0.
        return super()._update_schedule(t,data,mq,mission,soft,material_info,observed,grf)

    def _start_swing(self, t, leg, data, mq, mission, soft, material_info):
        super()._start_swing(t,leg,data,mq,mission,soft,material_info)
        swing = self.swing[leg]
        original = swing["landing"].copy()
        change = apply_offset(swing, mq["R"],leg,self.trajectory_action)
        self.applied_trajectories.append(dict(t=t,leg=int(leg),mode=self.policy_mode,
            nominal_landing=original.tolist(),issued_landing=swing["landing"].tolist(),
            offset=change.tolist(),apex=curve(swing["rise"],.1)[0].tolist(),
            apex_offset_m=float(self.trajectory_action[2]),duration_request_s=float(self.trajectory_action[3])))

    def _forecast_events(self, t, end_time, mq, mission):
        events = super()._forecast_events(t,end_time,mq,mission)
        if self.policy_mode == 0:
            for leg in self.next_pair:
                for start,end,entry in events[leg]:
                    if entry is not self.swing[leg] and start >= t-1.e-10 and "rise" in entry:
                        apply_offset(entry,mq["R"],leg,self.trajectory_action)
                        break
        return events

    def _redistribute(self, forces, mq, active, required=None):
        target, wrench = super()._redistribute(forces,mq,active,required)
        self.previous_required = wrench.copy()
        return target,wrench

    def replan(self,t,data,mq,mission):
        # Same two parent assemblers and solve path; bulk per-QP matrix serialization is replaced
        # by the scheduled/solved first forces and the original failure matrix capture.
        n, spacing = self.cfg["mpc"]["intervals"], self.cfg["mpc"]["dt_s"]
        x0,ref = self._body_reference(t,mq,mission)
        times = t + np.arange(n+1)*spacing
        contact, feet = self._future_contact_feet(t,times,mq,mission)
        points = feet.copy()
        points[:,:,2] -= self.radius[None,:]
        points[0] = mq["contact_force_point"]
        if mission.phase in ENDPOINT_HOLDS:
            stages,_,_ = standing_stages(self.mass,mq["inertia"],x0,points[:-1],contact[:-1],times,spacing,self.cfg["force_pyramid_slope"])
        else:
            stages = [assemble_centroidal(self.mass,mq["inertia"],ref[k,3:6],points[k],mq["rpy"],mq["omega"],spacing,self.cfg["force_pyramid_slope"]) for k in range(n)]
        inputs = dict(x0=x0,reference=ref,last_force=self.last_force.copy(),contact=contact[:-1],
            force_normal_upper=np.full((n,4),np.inf),stage_A=np.array([x["A"] for x in stages]),
            stage_B=np.array([x["B"] for x in stages]),stage_c=np.array([x["c"] for x in stages]))
        qp = assemble_force_horizon(inputs,self.cfg)
        z,status = self._solve(qp["P"],qp["q"],qp["A"],qp["lower"],qp["upper"],"mpc",t)
        nx = 12*(n+1)
        self.last_plan = dict(time=t,times=times,state=z[:nx].reshape(n+1,12),reference=ref,
            forces=z[nx:].reshape(n,4,3),feet=feet,contact=contact[:-1],blocks=stages[0],status=status)
        self.latest_plan = dict(times=times,query_feet=feet.copy(),source="W61 frozen B3 force horizon")
        self.plan_count += 1

    def flush(self):
        # Complete per-tick kinematics, command, GRFs and contacts are recorded by the W61 collector.
        self.rows.clear()
        self.drift_rows.clear()
