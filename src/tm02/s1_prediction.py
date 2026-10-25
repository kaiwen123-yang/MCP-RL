"""Aligned, bounded S1 forecasts using the executed controller in the near prefix.

The first 0.24 s uses copied MuJoCo state, native gait/WB and the same MPC with
copied warm start and random keys.  The remaining tail is explicitly a 20 ms
centroidal approximation using that same MPC and native swing generator.  It
is not a replacement least-squares wrench controller or an event-compression
algorithm.  Forecasts never advance the live controller or material state.
"""
import copy
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from gym_quadruped.utils.quadruped_utils import LegsAttr
from quadruped_pympc import config as cfg
from quadruped_pympc.helpers.quadruped_utils import GaitType
from .reference import IntegrationFailure
from .s1_planner import NoFeasibleIntent
from .s1_contact import StanceReferenceInfeasible

LEGS = ('FL', 'FR', 'RL', 'RR')


def array(value):
    return np.array([value[name] for name in LEGS])


def attrs(value):
    return LegsAttr(**{name: value[i].copy() for i, name in enumerate(LEGS)})


def clone_env(env):
    other = copy.copy(env)
    other.mjModel = copy.copy(env.mjModel)
    other.mjData = copy.copy(env.mjData)
    other.viewer = None
    other.sensors = []
    return other


def clone_interface(interface):
    other = copy.copy(interface)
    other.__dict__.pop('compute_control', None)  # live timing closure
    other.previous_contact_mpc = interface.previous_contact_mpc.copy()
    other.controller = copy.copy(interface.controller)
    for name, value in vars(interface.controller).items():
        if isinstance(value, np.ndarray):
            setattr(other.controller, name, value.copy())
    # JAX keys/arrays are immutable; prepare_state replaces branch attributes.
    other.controller.robot = copy.copy(interface.controller.robot)
    other.controller.jitted_compute_control = interface.controller._s1_raw_jit
    return other


class FootstepCommitment:
    """One scalar for the next not-yet-lifted group, consumed at liftoff."""
    def __init__(self, wb):
        self.wb = wb
        self.pending = None
        self.offsets = np.zeros((4, 3))
        self.serial = 0
        self.applied = []
        self.time = 0.

    def install(self):
        self.wb.update_state_and_reference = self.update

    def clone(self, wb):
        other = FootstepCommitment(wb)
        other.pending = self.pending
        other.offsets = self.offsets.copy()
        other.serial, other.time = self.serial, self.time
        other.install()
        return other

    def update(self, *args, **kwargs):
        result = type(self.wb).update_state_and_reference(self.wb, *args, **kwargs)
        state, ref, sequence, height, optimize = result
        lifting = np.flatnonzero((self.wb.previous_contact == 1) & (self.wb.current_contact == 0))
        if len(lifting):
            self.offsets[lifting] = 0.
            if self.pending is not None:
                yaw = state['orientation'][2]
                direction = np.array([np.cos(yaw), np.sin(yaw), 0.])
                self.offsets[lifting] = float(self.pending)*direction
                self.serial += 1
                self.applied.append(dict(time_s=float(self.time), legs=lifting.tolist(),
                    offset_m=float(self.pending), serial=self.serial))
                self.pending = None
        for leg, name in enumerate(LEGS):
            # This is one absolute displacement of this step's nominal target,
            # never an accumulated shift and never a change to an older swing.
            if self.wb.current_contact[leg] == 0:
                ref['ref_foot_'+name] = np.array(ref['ref_foot_'+name], copy=True)+self.offsets[leg]
        return state, ref, sequence, height, optimize


class AlignedPlanner:
    def __init__(self, env, wrapper, contact_env, method, horizon, mass, inertia,
                 tracking, moving, stand):
        self.env, self.wrapper, self.contact_env = env, wrapper, contact_env
        self.method, self.horizon = method, float(horizon)
        self.mass, self.inertia = float(mass), np.asarray(inertia)
        self.tracking, self.moving, self.stand = tracking, moving, stand
        self.choices = []
        self.last_upper_s = 0.
        self.selected = (0., 2.4, 0.)
        self.commitment = FootstepCommitment(wrapper.wb_interface) if method in ('B2', 'B3') else None
        if self.commitment is not None:
            self.commitment.install()

    def _branch(self):
        env = clone_env(self.env)
        ground = self.contact_env.predictor.clone()
        contact = copy.copy(self.contact_env)
        contact.env, contact.ground, contact.predictor = env, ground, ground
        for name in ('force', 'indent', 'contact', 'native_type', 'native_affinity'):
            setattr(contact, name, getattr(self.contact_env, name).copy())
        contact.physics_hist, contact.material_hist, contact.predictor_hist = [], [], []
        wrapper = copy.copy(self.wrapper)
        for name, value in vars(self.wrapper).items():
            if isinstance(value, (np.ndarray, LegsAttr, dict)):
                setattr(wrapper, name, copy.deepcopy(value))
        wb0 = copy.copy(self.wrapper.wb_interface)
        wb0.__dict__.pop('update_state_and_reference', None)
        wb0.__dict__.pop('compute_stance_and_swing_torque', None)
        ik = copy.copy(wb0.ik)
        ik.env = clone_env(wb0.ik.env)
        wb0.ik = ik
        wrapper.wb_interface = copy.deepcopy(wb0, {id(ik): ik})
        moving, stand = clone_interface(self.moving), clone_interface(self.stand)
        wrapper._s1_moving, wrapper._s1_stand = moving, stand
        wrapper.srbd_controller_interface = stand if self.wrapper.srbd_controller_interface is self.stand else moving
        tracker = self.tracking.clone(env, wrapper.wb_interface, contact)
        commitment = self.commitment.clone(wrapper.wb_interface) if self.commitment is not None else None
        return env, wrapper, contact, moving, stand, tracker, commitment

    def observe(self, t, mission, step, tau):
        begin = time.perf_counter()
        value = self.forecast(None, self.horizon, mission, t, step, tau)
        self.last_upper_s = time.perf_counter()-begin
        self.choices.append(dict(time_s=float(t), start_time_s=float(t), kind='nominal_continuation',
            candidate_id=0, intent=[0., 2.4, 0.], horizon_s=self.horizon,
            wall_s=self.last_upper_s, selected_plan=value.pop('trace'),
            aligned_trace=value.pop('aligned_trace'), candidates=[value]))

    def choose(self, t, mission, step, tau):
        begin = time.perf_counter()
        horizon = .24 if self.method == 'B2' else self.horizon
        pending = self.commitment.pending
        # Candidate zero is the actual plan, including any unconsumed selection.
        intents = [None]+[x for x in (0., -.04, .04) if x != pending and not (pending is None and x == 0.)]
        values = [self.forecast(x, horizon, mission, t, step, tau) for x in intents]
        costs = np.array([x['cost'] if not x['infeasible_steps'] else np.inf for x in values])
        finite = np.flatnonzero(np.isfinite(costs))
        best = None
        if len(finite):
            minimum = np.min(costs[finite])
            # Exact ties and not-yet-effective actions keep continuation.
            best = 0 if costs[0] == minimum else int(finite[np.argmin(costs[finite])])
            if best and not values[best]['action_observed'] and np.isfinite(costs[0]):
                best = 0
        trace = [] if best is None else values[best]['trace']
        aligned = [] if best is None else values[best]['aligned_trace']
        for value in values:
            value.pop('trace')
            value.pop('aligned_trace')
        self.last_upper_s = time.perf_counter()-begin
        self.choices.append(dict(time_s=float(t), start_time_s=float(t), kind='decision', candidate_id=best,
            intent=None if best is None else [pending if best == 0 and pending is not None else (intents[best] or 0.), 2.4, 0.],
            horizon_s=horizon, wall_s=self.last_upper_s, candidates=values,
            selected_plan=trace, aligned_trace=aligned,
            failure='no_feasible_intent' if best is None else None))
        if best is None:
            raise NoFeasibleIntent('All explicitly evaluated continuation/next-group candidates violate physical constraints.')
        if best:
            self.commitment.pending = intents[best]
            self.selected = (intents[best], 2.4, 0.)

    def forecast(self, offset, horizon, mission, start_time, start_step, tau):
        env, wrapper, contact, moving, stand, tracker, commitment = self._branch()
        if commitment is not None and offset is not None:
            commitment.pending = offset
        sim_mission = copy.deepcopy(mission)
        sim_mission.transitions = []
        tau = copy.deepcopy(tau)
        wb = wrapper.wb_interface
        trace, aligned, covered = [], [], []
        cost, elapsed, maximum_indent, maximum_force = 0., 0., 0., 0.
        first = None
        physical_prefix = min(.24, horizon)
        initial_actual = contact.contact.copy()
        snapshot = dict(phase_signal=wb.pgg.phase_signal.tolist(), current_contact=wb.current_contact.tolist(),
            previous_contact=wb.previous_contact.tolist(), actual_contact=initial_actual.tolist(),
            lift_off=array(wb.frg.lift_off_positions).tolist(), touchdown=array(wrapper.nmpc_footholds).tolist(),
            swing_elapsed_s=list(wb.stc.swing_time), foot_velocity=array(env.feet_vel(frame='world')).tolist(),
            mission_phase=sim_mission.phase, pending_offset_m=None if commitment is None else commitment.pending,
            standing=bool(wb.pgg.gait_type == GaitType.FULL_STANCE.value),
            current_force_plan=array(wrapper.nmpc_GRFs).tolist(), last_mpc_step=wrapper.last_mpc_step_num)
        snapshot['mission_linear_reference'] = sim_mission.v.tolist()
        snapshot['mission_angular_reference'] = sim_mission.w.tolist()
        snapshot['mpc_warm_start'] = np.asarray(wrapper.srbd_controller_interface.controller.best_control_parameters).tolist()
        snapshot['mpc_random_key'] = np.asarray(wrapper.srbd_controller_interface.controller.sampling_key).tolist()
        def reject(condition, when, leg=None, **detail):
            return dict(condition=condition, absolute_time_s=float(when), relative_time_s=float(when-start_time),
                leg=leg, position_m=env.base_pos.tolist(), orientation_rad=env.base_ori_euler_xyz.tolist(), **detail)
        def record(dt, segment, grf_time):
            nonlocal cost, maximum_indent, maximum_force
            p, v = env.base_pos.copy(), env.base_lin_vel(frame='world').copy()
            ori, omega = env.base_ori_euler_xyz.copy(), env.base_ang_vel(frame='base').copy()
            feet = array(env.feet_pos(frame='world'))
            planned, reaction = array(wrapper.nmpc_GRFs), contact.force.copy()
            bottom = feet.copy()
            bottom[:, 2] -= contact.radius
            velocity = array(env.feet_vel(frame='world'))
            height = cfg.locomotion_params['ref_z']
            stage = (5.*np.sum(((v-sim_mission.v)/.25)**2)+25.*((p[2]-height)/.28)**2+
                5.*np.sum((ori[:2]/.2)**2)+np.sum(((omega-sim_mission.w)/.5)**2)+
                .02*np.sum((planned/(self.mass*9.81))**2)+2.*np.sum(((reaction-planned)/(self.mass*9.81))**2)+
                .1*np.maximum(-np.sum(reaction*velocity,axis=1),0).sum()/(self.mass*9.81*.25)+1.)
            cost += float(stage*dt)
            maximum_indent = max(maximum_indent,float(contact.indent.max()))
            maximum_force = max(maximum_force,float(np.linalg.norm(reaction,axis=1).max()))
            trace.append(np.r_[elapsed,p,v,ori,omega,bottom.ravel(),planned.ravel(),reaction.ravel()].tolist())
            aligned.append(dict(absolute_time_s=float(start_time+elapsed),grf_sample_time_s=float(grf_time),
                base_pos=p.tolist(),orientation=ori.tolist(),feet=feet.tolist(),
                planned_grf=planned.tolist(),actual_grf=reaction.tolist(),contact=contact.contact.tolist(),
                prediction_segment=segment))
        try:
            for n in range(round(physical_prefix/.002)):
                t = start_time+n*.002
                mujoco.mj_forward(env.mjModel, env.mjData)
                standing = wb.pgg.gait_type == GaitType.FULL_STANCE.value
                if n:
                    sim_mission.update(t,.002,env.base_pos,env.base_lin_vel(frame='world'),
                        env.base_ori_euler_xyz,env.base_ang_vel(frame='base'),standing and np.all(contact.contact))
                want_stand = sim_mission.phase in ('initial_stand','terminal_stand','complete')
                if want_stand and not standing and np.all(wb.current_contact):
                    wb.pgg.set_full_stance()
                    wrapper.srbd_controller_interface, wrapper.last_mpc_step_num = stand, None
                elif not want_stand and standing:
                    wb.pgg.restore_previous_gait()
                    wrapper.srbd_controller_interface, wrapper.last_mpc_step_num = moving, None
                if commitment is not None:
                    commitment.time = t
                feet, fv = env.feet_pos(frame='world'), env.feet_vel(frame='world')
                tau = wrapper.compute_actions(env.com.copy(),env.base_pos.copy(),env.base_lin_vel(frame='world').copy(),
                    env.base_ori_euler_xyz.copy(),env.base_ang_vel(frame='base').copy(),feet,env.hip_positions(frame='world'),
                    env.legs_qvel_idx,None,LEGS,.002,sim_mission.v.copy(),sim_mission.w.copy(),start_step+n,
                    env.mjData.qpos,env.mjData.qvel,env.feet_jacobians(frame='world',return_rot_jac=False),
                    env.feet_jacobians_dot(frame='world',return_rot_jac=False),fv,env.legs_qfrc_passive,
                    env.legs_qfrc_bias,env.legs_mass_matrix,env.legs_qpos_idx,env.legs_qvel_idx,tau,
                    self.inertia.flatten(),env.mjData.contact)
                if np.any((wb.previous_contact == 0)&(wb.current_contact == 1)):
                    # The compliant WB has replaced soft-stance references;
                    # solve those actual targets before assessing reachability.
                    wb.ik.compute_solution(env.mjData.qpos.copy(), *array(wb.last_des_foot_pos))
                    wb.ik._update_kinematics()
                    residual = np.linalg.norm(array(wb.ik.env.feet_pos())-array(wb.last_des_foot_pos),axis=1)
                    joints = wb.ik.env.mjData.qpos[7:]
                    limits = env.mjModel.jnt_range[1:]
                    if np.any(joints<limits[:,0]) or np.any(joints>limits[:,1]) or residual.max()>.002:
                        first = reject('touchdown_inverse_kinematics',t,leg=int(np.argmax(residual)),
                            residual_m=residual.tolist(),threshold_m=.002,joints_rad=joints.tolist())
                        break
                action = np.zeros(env.mjModel.nu)
                for name in LEGS:
                    action[env.legs_tau_idx[name]] = tau[name]
                action = np.clip(action,env.mjModel.actuator_ctrlrange[:,0],env.mjModel.actuator_ctrlrange[:,1])
                for sub in range(10):
                    gt = t+sub*.0002
                    terminated,info,nonfoot = contact.advance(action,.0002,gt,record_material=False)
                    if contact.ground.domain_exit:
                        first = reject('material_domain_exit',gt,material=contact.ground.domain_exit)
                    elif nonfoot:
                        first = reject('soft_region_nonfoot_contact',gt,geometries=nonfoot)
                    elif terminated:
                        first = reject('invalid_contact',gt,contacts=info['invalid_contacts'])
                    if first:
                        break
                elapsed = (n+1)*.002
                mujoco.mj_forward(env.mjModel,env.mjData)
                if not first and (max(abs(env.base_ori_euler_xyz[:2]))>1.2 or env.base_pos[2]<.12):
                    first = reject('body_fall',start_time+elapsed)
                if n%10 == 9 or first:
                    record(.002*(n%10+1),'exact_prefix',gt)
                if first:
                    break
            if first is None and elapsed < horizon-1e-10:
                elapsed,cost,maximum_indent,maximum_force,first,tail_trace,tail_aligned = self._tail(
                    env,wrapper,contact,sim_mission,commitment,start_time,elapsed,horizon,cost,maximum_indent,maximum_force)
                trace.extend(tail_trace)
                aligned.extend(tail_aligned)
        except IntegrationFailure as error:
            first = reject('material_integration_failure',start_time+elapsed,error=str(error))
        except StanceReferenceInfeasible as error:
            first = reject('constitutive_inverse_outside_material_domain',start_time+elapsed,detail=error.detail)
        for event in contact.ground.events:
            if event['type']=='revisit' and event.get('previous_leg',event['leg'])!=event['leg']:
                patch=event['patch']
                covered.append(dict(absolute_time_s=float(event['time_s']),time_s=float(event['time_s']-start_time),
                    leg=event['leg'],previous_leg=event['previous_leg'],patches=[patch],
                    patch_xy_m=[contact.ground.xy[patch]],overlap_m2=event.get('area_m2')))
        goal = mission.task['turnaround_xy_m'] if sim_mission.phase in ('outbound','turn') else mission.task['return_xy_m']
        cost += float(np.linalg.norm(np.asarray(goal)-env.base_pos[:2])/.25)
        applications = [] if commitment is None else commitment.applied
        effective = applications[0]['time_s'] if applications else None
        return dict(cost=cost,infeasible_steps=int(first is not None),first_rejection=first,
            candidate_offset_m=offset,continuation=offset is None,action_effective_time_s=effective,
            action_observed=bool(effective is not None and effective < start_time+elapsed),
            applied_groups=applications,predicted_duration_s=elapsed,max_indentation_mm=maximum_indent*1000,
            max_force_N=maximum_force,covered_revisits=covered,initial_state=snapshot,
            exact_prefix_s=physical_prefix,tail_model='20 ms native gait/swing + native MPC centroidal/material approximation',
            trace=trace,aligned_trace=aligned)

    def _tail(self, env, wrapper, contact, mission, commitment, start, elapsed, horizon,
              cost, maximum_indent, maximum_force):
        """Declared rigid-body/kinematic tail; the native MPC supplies forces.

        Joint dynamics and native hard-contact compliance are approximated in
        this tail.  The actual 2 ms control/0.2 ms physics prefix is retained in
        the same forecast for measuring this approximation separately.
        """
        wb, ground = wrapper.wb_interface, contact.ground
        p, v = env.base_pos.copy(), env.base_lin_vel(frame='world').copy()
        ori, omega = env.base_ori_euler_xyz.copy(), env.base_ang_vel(frame='base').copy()
        foot, foot_velocity = array(env.feet_pos(frame='world')), array(env.feet_vel(frame='world'))
        rotation0 = Rotation.from_euler('xyz',ori).as_matrix()
        hip_relative = (array(env.hip_positions(frame='world'))-p)@rotation0
        com_relative = (env.com-p)@rotation0
        trace, aligned, first = [], [], None
        while elapsed < horizon-1e-10:
            dt = min(.02,horizon-elapsed)
            t = start+elapsed
            standing = wb.pgg.gait_type == GaitType.FULL_STANCE.value
            mission.update(t,dt,p,v,ori,omega,standing and np.all(contact.contact))
            want_stand = mission.phase in ('initial_stand','terminal_stand','complete')
            if want_stand and not standing and np.all(wb.current_contact):
                wb.pgg.set_full_stance()
                wrapper.srbd_controller_interface = wrapper._s1_stand
            elif not want_stand and standing:
                wb.pgg.restore_previous_gait()
                wrapper.srbd_controller_interface = wrapper._s1_moving
            rotation = Rotation.from_euler('xyz',ori).as_matrix()
            if commitment is not None:
                commitment.time = t
            state,reference,sequence,_,optimize = wb.update_state_and_reference(
                p+rotation@com_relative,p,v,ori,omega,attrs(foot),attrs(p+hip_relative@rotation.T),
                env.legs_qvel_idx,None,LEGS,dt,mission.v.copy(),mission.w.copy(),None)
            interface = wrapper.srbd_controller_interface
            result = interface.compute_control(state,reference,sequence,self.inertia.flatten(),
                wb.pgg.phase_signal,wb.pgg.step_freq,optimize,time_since_last_solve=dt)
            (wrapper.nmpc_GRFs,wrapper.nmpc_footholds,wrapper.nmpc_joints_pos,
             wrapper.nmpc_joints_vel,wrapper.nmpc_joints_acc,wrapper.best_sample_freq,
             wrapper.nmpc_predicted_state) = result
            planned = array(wrapper.nmpc_GRFs)
            prior = wb.previous_contact.copy()
            current = wb.current_contact.copy()
            landed = np.flatnonzero((prior == 0)&(current == 1))
            wb.stc.update_swing_time(current,LEGS,dt)
            before = foot.copy()
            for leg,name in enumerate(LEGS):
                if current[leg] == 0:
                    position,velocity,_ = wb.stc.swing_generator.compute_trajectory_references(
                        wb.stc.swing_time[leg],wb.frg.lift_off_positions[name],wrapper.nmpc_footholds[name],
                        wb.esd.hitmoments[name],wb.esd.hitpoints[name])
                    foot[leg],foot_velocity[leg] = np.asarray(position).reshape(3),np.asarray(velocity).reshape(3)
                else:
                    if leg in landed:
                        foot[leg] = np.asarray(wrapper.nmpc_footholds[name]).reshape(3)
                    if ground.soft(foot[leg,:2]):
                        try:
                            inverse = ground.force_inverse(foot[leg,:2],planned[leg,2],dt)
                        except (IntegrationFailure, StanceReferenceInfeasible) as error:
                            first = dict(condition='material_inverse_failure',absolute_time_s=t,relative_time_s=elapsed,
                                leg=int(leg),error=str(error),position_m=p.tolist(),orientation_rad=ori.tolist())
                            break
                        foot[leg,2] = contact.radius[leg]-inverse['displacement_m']
                    else:
                        foot[leg,2] = contact.radius[leg]
                    foot_velocity[leg] = (foot[leg]-before[leg])/dt
            if first:
                break
            # The same dissipative tangential constitutive law determines the
            # slip required by the planned horizontal force in the tail.
            for leg in np.flatnonzero(current):
                if ground.soft(foot[leg,:2]) and planned[leg,2]>0:
                    horizontal = np.linalg.norm(planned[leg,:2])
                    ratio = horizontal/(.5*planned[leg,2])
                    if ratio >= 1.:
                        first = dict(condition='friction_cone',absolute_time_s=t,relative_time_s=elapsed,
                            leg=int(leg),force_N=planned[leg].tolist(),position_m=p.tolist(),orientation_rad=ori.tolist())
                        break
                    if horizontal>0:
                        foot_velocity[leg,:2] = -.01*ratio/np.sqrt(1-ratio**2)*planned[leg,:2]/horizontal
                        foot[leg,:2] += foot_velocity[leg,:2]*dt
            if first:
                break
            bottom = foot.copy()
            bottom[:,2] -= contact.radius
            try:
                reaction,indent,_ = ground.step(bottom,foot_velocity,dt,t+dt)
            except IntegrationFailure as error:
                first = dict(condition='material_integration_failure',absolute_time_s=t+dt,relative_time_s=elapsed+dt,
                    leg=None,error=str(error),position_m=p.tolist(),orientation_rad=ori.tolist())
                break
            for leg in range(4):
                if not ground.soft(foot[leg,:2]):
                    reaction[leg] = planned[leg] if current[leg] else 0.
            reaction[current == 0] = 0.
            jac = env.feet_jacobians(frame='world',return_rot_jac=False)
            for leg,name in enumerate(LEGS):
                torque = -jac[name][:,env.legs_qvel_idx[name]].T@planned[leg]
                lower,upper = env.mjModel.actuator_ctrlrange[env.legs_tau_idx[name]].T
                if np.any(torque<lower) or np.any(torque>upper):
                    first = dict(condition='planned_grf_torque_limit',absolute_time_s=t,relative_time_s=elapsed,
                        leg=int(leg),requested_torque_Nm=torque.tolist(),position_m=p.tolist(),orientation_rad=ori.tolist())
                    break
            if first:
                break
            acceleration = reaction.sum(axis=0)/self.mass-np.array([0.,0.,9.81])
            moment = np.cross(bottom-p,reaction).sum(axis=0)
            angular_acceleration = np.linalg.solve(self.inertia,rotation.T@moment-np.cross(omega,self.inertia@omega))
            p += v*dt+.5*acceleration*dt*dt
            v += acceleration*dt
            next_rotation = rotation@Rotation.from_rotvec(omega*dt+.5*angular_acceleration*dt*dt).as_matrix()
            omega += angular_acceleration*dt
            ori = Rotation.from_matrix(next_rotation).as_euler('xyz')
            qpose = env.mjData.qpos.copy()
            qpose[:3] = p
            quat = Rotation.from_matrix(next_rotation).as_quat()
            qpose[3:7] = quat[[3,0,1,2]]
            joints = wb.ik.compute_solution(qpose,*foot)
            wb.ik._update_kinematics()
            residual = np.linalg.norm(array(wb.ik.env.feet_pos())-foot,axis=1)
            joint_limits = env.mjModel.jnt_range[1:]
            if len(landed) and (np.any(joints<joint_limits[:,0]) or np.any(joints>joint_limits[:,1]) or residual.max()>.002):
                leg = int(np.argmax(residual))
                first = dict(condition='touchdown_inverse_kinematics',absolute_time_s=t+dt,relative_time_s=elapsed+dt,
                    leg=leg,residual_m=residual.tolist(),threshold_m=.002,joints_rad=joints.tolist(),
                    targets_m=foot.tolist(),position_m=p.tolist(),orientation_rad=ori.tolist())
            qpose[7:] = joints
            env.mjData.qpos[:] = qpose
            env.mjData.qvel[:3] = v
            env.mjData.qvel[3:6] = omega
            mujoco.mj_forward(env.mjModel,env.mjData)
            contact.force,contact.indent,contact.contact = reaction,indent,reaction[:,2]>0
            elapsed += dt
            height = cfg.locomotion_params['ref_z']
            stage = (5.*np.sum(((v-mission.v)/.25)**2)+25.*((p[2]-height)/.28)**2+
                5.*np.sum((ori[:2]/.2)**2)+np.sum(((omega-mission.w)/.5)**2)+
                .02*np.sum((planned/(self.mass*9.81))**2)+2.*np.sum(((reaction-planned)/(self.mass*9.81))**2)+
                .1*np.maximum(-np.sum(reaction*foot_velocity,axis=1),0).sum()/(self.mass*9.81*.25)+1.)
            cost += float(stage*dt)
            maximum_indent = max(maximum_indent,float(indent.max()))
            maximum_force = max(maximum_force,float(np.linalg.norm(reaction,axis=1).max()))
            trace.append(np.r_[elapsed,p,v,ori,omega,bottom.ravel(),planned.ravel(),reaction.ravel()].tolist())
            aligned.append(dict(absolute_time_s=float(start+elapsed),grf_sample_time_s=float(t+dt),
                base_pos=p.tolist(),orientation=ori.tolist(),feet=foot.tolist(),planned_grf=planned.tolist(),
                actual_grf=reaction.tolist(),contact=contact.contact.tolist(),prediction_segment='centroidal_tail'))
            if first is None and ground.domain_exit:
                first = dict(condition='material_domain_exit',absolute_time_s=t+dt,relative_time_s=elapsed,
                    leg=None,material=ground.domain_exit,position_m=p.tolist(),orientation_rad=ori.tolist())
            if first is None and (p[2]<.12 or max(abs(ori[:2]))>1.2):
                first = dict(condition='body_fall',absolute_time_s=t+dt,relative_time_s=elapsed,
                    leg=None,position_m=p.tolist(),orientation_rad=ori.tolist())
            if first:
                break
        return elapsed,cost,maximum_indent,maximum_force,first,trace,aligned
