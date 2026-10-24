"""Shared finite contact intents with short or event-extended robot prediction.

Candidate loads come from a centroidal wrench tracking plan, with inherited
friction and the actual current Go2 joint torque bounds. The cloned material
is advanced from current state under the planned foot displacement. No
recorded future simulator states or forces enter this module.
"""
import copy
import itertools
import time
import numpy as np
from scipy.spatial.transform import Rotation
from quadruped_pympc.helpers.periodic_gait_generator import PeriodicGaitGenerator
from quadruped_pympc.helpers.quadruped_utils import GaitType
from .reference import IntegrationFailure

LEGS = ('FL', 'FR', 'RL', 'RR')


class NoFeasibleIntent(RuntimeError):
    pass


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class IntentGait(PeriodicGaitGenerator):
    def install(self):
        self.pending = None
        self.hold_remaining = 0.
        self.intent = (0., 2.4, 0.)
        self.intent_serial = 0

    def run(self, dt, new_step_freq, out=None):
        if dt>0 and self.gait_type != GaitType.FULL_STANCE.value:
            all_stance = np.all(self._phase_signal < self.duty_factor)
            if self.pending is not None and all_stance:
                self.intent = self.pending
                self.pending = None
                self.step_freq = self.intent[1]
                self.hold_remaining = self.intent[2]
                self.intent_serial += 1
            held = min(dt, self.hold_remaining) if all_stance else 0.
            self.hold_remaining -= held
            dt -= held
        return super().run(dt, self.step_freq, out)

    def compute_contact_sequence(self, contact_sequence_dts, contact_sequence_lenghts):
        saved = (self.pending, self.hold_remaining, self.intent, self.intent_serial, self.step_freq)
        result = super().compute_contact_sequence(contact_sequence_dts, contact_sequence_lenghts)
        self.pending, self.hold_remaining, self.intent, self.intent_serial, self.step_freq = saved
        return result


class IntentPlanner:
    def __init__(self, env, wrapper, contact_env, method, horizon, mass, inertia):
        self.env, self.wrapper, self.contact_env = env, wrapper, contact_env
        self.method, self.horizon, self.mass, self.inertia = method, horizon, mass, inertia
        self.candidates = list(itertools.product([-.04, 0., .04], [2., 2.4], [0., .1]))
        self.selected = (0., 2.4, 0.)
        self.choices = []
        self.offsets = np.zeros((4, 3))
        self.stance_targets = np.array([env.feet_pos()[l] for l in LEGS])
        self.target_indent = np.zeros(4)
        self.last_upper_s = 0.
        self.last_constitutive_s = 0.
        self.previous_serial = 0
        wb = wrapper.wb_interface
        wb.pgg.__class__ = IntentGait
        wb.pgg.install()
        raw = wb.update_state_and_reference
        def update(*args, **kwargs):
            result = raw(*args, **kwargs)
            state, ref, sequence, height, optimize = result
            yaw = state['orientation'][2]
            for leg, name in enumerate(LEGS):
                if wb.previous_contact[leg] and not wb.current_contact[leg]:
                    self.offsets[leg] = wb.pgg.intent[0]*np.array([np.cos(yaw), np.sin(yaw), 0.])
                ref['ref_foot_'+name] = np.array(ref['ref_foot_'+name], copy=True) + self.offsets[leg]
            if wb.pgg.intent_serial != self.previous_serial:
                self.previous_serial = wb.pgg.intent_serial
                wb.frg.stance_time = wb.pgg.duty_factor/wb.pgg.step_freq
                wb.stc.regenerate_swing_trajectory_generator(step_height=wb.step_height,
                    swing_period=(1-wb.pgg.duty_factor)/wb.pgg.step_freq)
                wrapper.best_sample_freq = wb.pgg.step_freq
            return state, ref, sequence, height, 0
        wb.update_state_and_reference = update

    def choose(self, t, mission):
        begin = time.perf_counter()
        wb = self.wrapper.wb_interface
        horizon = .24 if self.method == 'C2' else self.horizon
        values = []
        for intent in self.candidates:
            try:
                values.append(self.forecast(intent, horizon, mission, t))
            except IntegrationFailure as error:
                values.append(dict(cost=None,covered_revisits=[],predicted_duration_s=None,
                    max_indentation_mm=None,max_force_N=None,infeasible_steps=1,
                    integration_failure=str(error),trace=[]))
        costs = [x['cost'] if not x['infeasible_steps'] else np.inf for x in values]
        if not np.isfinite(costs).any():
            for value in values:
                del value['trace']
            self.last_upper_s = time.perf_counter()-begin
            self.choices.append(dict(time_s=float(t),candidate_id=None,intent=None,horizon_s=horizon,
                wall_s=self.last_upper_s,candidates=values,selected_plan=[],failure='no_feasible_intent'))
            raise NoFeasibleIntent('All 12 candidate robot/contact plans violate the shared physical constraints.')
        best = int(np.argmin(costs))
        selected_plan = values[best]['trace']
        for value in values:
            del value['trace']
        self.selected = self.candidates[best]
        wb.pgg.pending = self.selected
        self.last_upper_s = time.perf_counter()-begin
        self.choices.append(dict(time_s=float(t), candidate_id=best, intent=list(self.selected),
            horizon_s=horizon, wall_s=self.last_upper_s, candidates=values, selected_plan=selected_plan))

    def forecast(self, intent, horizon, mission, start_time):
        env, wb = self.env, self.wrapper.wb_interface
        ground = self.contact_env.predictor.clone()
        p, v = env.base_pos.copy(), env.base_lin_vel(frame='world').copy()
        ori, omega = env.base_ori_euler_xyz.copy(), env.base_ang_vel(frame='base').copy()
        foot, actual_vel = self.contact_env.kinematics()
        phase = wb.pgg.phase_signal.copy()
        current = wb.current_contact.copy()
        prior = current.copy()
        nominal = np.array([env.hip_positions()[l] for l in LEGS])-p
        yaw0 = ori[2]
        rot0 = Rotation.from_euler('z', yaw0).as_matrix()
        nominal = nominal@rot0
        nominal[:, 1] += np.array([.095, -.095, .095, -.095])
        jac = env.feet_jacobians(frame='world', return_rot_jac=False)
        jac = np.array([jac[l][:, env.legs_qvel_idx[l]] for l in LEGS])
        limits = np.array([env.mjModel.actuator_ctrlrange[env.legs_tau_idx[l], 1] for l in LEGS])
        sim_mission = copy.deepcopy(mission)
        sim_mission.transitions = []
        lift = foot.copy()
        touchdown = foot.copy()
        hold, applied = 0., False
        frequency = wb.pgg.step_freq
        height = .28+float(np.mean(self.contact_env.radius))
        cost = 0.
        covered, load_seen = [], set(np.flatnonzero(ground.ever_loaded).tolist())
        revisit_foot_events = 0
        elapsed = 0.
        max_indent, max_force = 0., 0.
        violations = 0
        trace = []
        while elapsed < horizon-1e-10:
            dt = min(.02 if elapsed<.24-1e-10 else .1, horizon-elapsed)
            refv, refw = sim_mission.update(start_time+elapsed, dt, p, v, ori, omega, False)
            if not applied and np.all(phase<.65):
                frequency, hold, applied = intent[1], intent[2], True
            frozen = min(dt, hold) if np.all(phase<.65) else 0.
            hold -= frozen
            phase = (phase+(dt-frozen)*frequency)%1.
            current = (phase<.65).astype(int)
            rot = Rotation.from_euler('z', ori[2]).as_matrix()
            landed = []
            for leg in range(4):
                if prior[leg] and not current[leg]:
                    lift[leg] = foot[leg]
                    touchdown[leg] = p+rot@nominal[leg]
                    touchdown[leg, :2] += refv[:2]*(.65/frequency)/2
                    touchdown[leg, :2] += intent[0]*rot[:2, 0]
                    touchdown[leg, 2] = 0.
                if not current[leg]:
                    fraction = (phase[leg]-.65)/.35
                    foot[leg] = (1-fraction)*lift[leg]+fraction*touchdown[leg]
                    foot[leg, 2] += .056*np.sin(np.pi*fraction)
                elif not prior[leg]:
                    foot[leg] = touchdown[leg]
                    landed.append(leg)
            stance = np.flatnonzero(current)
            # Newton-Euler wrench demand depends on the simulated candidate state.
            acceleration = 4.*(refv-v)
            acceleration[2] += 80.*(height-p[2])-8.*v[2]
            angular = 5.*(refw-omega)-np.r_[25.*ori[:2], 0.]
            body_rotation = Rotation.from_euler('xyz', ori).as_matrix()
            wrench = np.r_[self.mass*(acceleration+np.array([0., 0., 9.81])), body_rotation@self.inertia@angular]
            allocation = np.hstack([np.vstack((np.eye(3), skew(foot[l]-p))) for l in stance])
            f = np.linalg.lstsq(allocation, wrench, rcond=None)[0].reshape(-1, 3)
            planned = np.zeros((4, 3))
            for leg, force in zip(stance, f):
                force[2] = np.clip(force[2], 0., self.mass*9.81)
                xy = np.linalg.norm(force[:2])
                if xy > .5*force[2]:
                    force[:2] *= .5*force[2]/xy
                torque = jac[leg].T@force
                scale = max(1., float(np.max(np.abs(torque)/limits[leg])))
                force /= scale
                planned[leg] = force
            before = foot.copy()
            for leg in stance:
                foot[leg, 2] = -ground.displacement_for_force(foot[leg, :2], planned[leg, 2], dt)
            if landed:
                qpose = env.mjData.qpos.copy()
                qpose[:3] = p
                quat = Rotation.from_euler('xyz', ori).as_quat()
                qpose[3:7] = quat[[3, 0, 1, 2]]
                targets = foot.copy()
                targets[:, 2] += self.contact_env.radius
                joint_solution = wb.ik.compute_solution(qpose, *targets)
                wb.ik._update_kinematics()
                residual = np.array([wb.ik.env.feet_pos()[l] for l in LEGS])-targets
                limits_j = env.mjModel.jnt_range[1:]
                if (np.any(joint_solution < limits_j[:, 0]) or np.any(joint_solution > limits_j[:, 1]) or
                    np.max(np.linalg.norm(residual,axis=1))>.002):
                    violations += 1
                    cost += 1e4*(horizon-elapsed)
                    break
                future_jac = wb.ik.env.feet_jacobians(frame='world',return_rot_jac=False)
                jac = np.array([future_jac[l][:,env.legs_qvel_idx[l]] for l in LEGS])
            velocity = (foot-before)/dt
            # Tangential slip required by the common regularized Coulomb law.
            for leg in stance:
                if ground.soft(foot[leg, :2]) and planned[leg, 2]>0:
                    ratio = min(.99, np.linalg.norm(planned[leg, :2])/(.5*planned[leg, 2]))
                    if ratio>0:
                        velocity[leg, :2] = -.01*ratio/np.sqrt(1-ratio**2)*planned[leg, :2]/np.linalg.norm(planned[leg, :2])
                        foot[leg, :2] += velocity[leg, :2]*dt
            reaction, indent, _ = ground.step(foot, velocity, dt, elapsed+dt, log=False)
            for leg in stance:
                if not ground.soft(foot[leg, :2]):
                    reaction[leg] = planned[leg]
            reaction[~current.astype(bool)] = 0.
            for leg in landed:
                if reaction[leg, 2]>0:
                    ids, aa = ground.footprint(foot[leg, :2])
                    overlap = [idx for idx in ids if idx in load_seen]
                    if overlap and revisit_foot_events<2:
                        covered.append(dict(time_s=elapsed+dt, leg=leg, patches=overlap,
                            patch_xy_m=[ground.xy[idx] for idx in overlap],
                            overlap_m2=float(sum(a for idx, a in zip(ids, aa) if idx in load_seen))))
                        revisit_foot_events += 1
                    load_seen.update(ids)
            acceleration = reaction.sum(axis=0)/self.mass-np.array([0., 0., 9.81])
            moment = np.cross(foot-p, reaction).sum(axis=0)
            omega += np.linalg.solve(self.inertia, body_rotation.T@moment-np.cross(omega, self.inertia@omega))*dt
            ori = Rotation.from_matrix(body_rotation@Rotation.from_rotvec(omega*dt).as_matrix()).as_euler('xyz')
            p += v*dt+.5*acceleration*dt*dt
            v += acceleration*dt
            # All terms are rates integrated by elapsed physical time.
            stage = (5.*np.sum(((v-refv)/.25)**2)+25.*((p[2]-height)/.28)**2+
                     5.*np.sum((ori[:2]/.2)**2)+np.sum(((omega-refw)/.5)**2)+
                     .02*np.sum((planned/(self.mass*9.81))**2)+
                     2.*np.sum(((reaction-planned)/(self.mass*9.81))**2)+
                     .1*np.maximum(-np.sum(reaction*velocity, axis=1), 0).sum()/(self.mass*9.81*.25)+1.)
            cost += float(stage*dt)
            trace.append(np.r_[elapsed+dt, p, v, ori, omega, foot.ravel(), planned.ravel(), reaction.ravel()].tolist())
            max_indent = max(max_indent, float(indent.max()))
            max_force = max(max_force, float(np.max(np.linalg.norm(reaction, axis=1))))
            if ground.domain_exit is not None or p[2]<.12 or max(abs(ori[:2]))>1.2:
                violations += 1
                cost += 1e4*(horizon-elapsed)
                break
            prior = current.copy()
            elapsed += dt
        terminal_goal = np.array(mission.task['turnaround_xy_m'] if mission.phase in ('outbound', 'turn') else mission.task['return_xy_m'])
        cost += float(np.linalg.norm(terminal_goal-p[:2])/.25)
        return dict(cost=cost, covered_revisits=covered, predicted_duration_s=elapsed,
                    max_indentation_mm=max_indent*1000, max_force_N=max_force, infeasible_steps=violations, trace=trace)

    def stance_torque(self, tau, feet, feet_velocity, t):
        begin = time.perf_counter()
        env, wb = self.env, self.wrapper.wb_interface
        jac = env.feet_jacobians(frame='world', return_rot_jac=False)
        for leg, name in enumerate(LEGS):
            if wb.current_contact[leg]:
                if not wb.previous_contact[leg]:
                    self.stance_targets[leg] = feet[name]
                if self.contact_env.predictor.soft(feet[name][:2]):
                    if round(t/.002)%10 == 0:
                        self.target_indent[leg] = self.contact_env.predictor.displacement_for_force(
                            feet[name][:2], max(0., self.wrapper.nmpc_GRFs[name][2]), .02)
                    target = self.stance_targets[leg].copy()
                    target[2] = self.contact_env.radius[leg]-self.target_indent[leg]
                    correction = 500.*(target-feet[name])-10.*feet_velocity[name]
                    tau[name] += jac[name][:, env.legs_qvel_idx[name]].T@correction
                    wb.last_des_foot_pos[name] = target
        self.last_constitutive_s = time.perf_counter()-begin
        return tau
