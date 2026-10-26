"""Bounded contact transition on the retained Go2/WB and column constitutive law.

The common swing correction restores the omitted floating-base terms. T1/T2
add one normal approach/loading path; the original nominal clock keeps running.
"""
import time
import numpy as np
import mujoco
from .s1_stance import CompliantStanceTracking
from .s1_force import project_contact_force, loading_fraction


def quintic(p, v, a, target, end_v, duration):
    c = np.empty((6, 3))
    c[:3] = p, v*duration, .5*a*duration**2
    c[3:] = np.linalg.solve(np.array([[1.,1.,1.],[3.,4.,5.],[6.,12.,20.]]),
        np.array([target-c[:3].sum(axis=0), end_v*duration-c[1]-2*c[2], -2*c[2]]))
    return c


def evaluate(c, elapsed, duration):
    u = min(elapsed/duration, 1.)
    p = np.array([u**k for k in range(6)]) @ c
    v = np.array([k*u**(k-1) for k in range(1,6)]) @ c[1:]/duration
    a = np.array([k*(k-1)*u**(k-2) for k in range(2,6)]) @ c[2:]/duration**2
    if elapsed > duration:
        p = p+(elapsed-duration)*v
    return p, v, a


class FloatingBaseSwing:
    """Shared correction; retain the native Cartesian feedback term and gains."""
    def __init__(self, env, wb):
        self.env, self.wb = env, wb
        self.legs = tuple(wb.legs_order)
        self.begin()
        wb.stc.compute_swing_control_cartesian_space = self.compute

    def begin(self):
        self.last = {k:np.full((4,3), np.nan) for k in (
            'swing_reference_position','swing_reference_velocity','swing_reference_acceleration',
            'swing_cartesian_feedback_torque','swing_joint_inertia_torque',
            'swing_base_correction_torque','swing_bias_torque','swing_base_kinematic_acceleration')}

    def compute(self, leg_id, q_dot, J, J_dot, lift_off, touch_down, foot_pos,
                foot_vel, h, mass_matrix, early_stance_hitmoments, early_stance_hitpoints):
        env, stc = self.env, self.wb.stc
        name = self.legs[leg_id]
        p,v,a = stc.swing_generator.compute_trajectory_references(stc.swing_time[leg_id],
            lift_off, touch_down, early_stance_hitmoments, early_stance_hitpoints)
        feedback = stc.position_gain_fb*(p-foot_pos)+stc.velocity_gain_fb*(v-foot_vel)
        full_j = env.feet_jacobians(frame='world', return_rot_jac=False)[name]
        full_jdot = env.feet_jacobians_dot(frame='world', return_rot_jac=False)[name]
        mass = np.zeros((env.mjModel.nv,env.mjModel.nv))
        mujoco.mj_fullM(env.mjModel, env.mjData, mass)
        ids = env.legs_qvel_idx[name]
        base_acc = env.mjData.qacc[:6].copy()
        inv = np.linalg.pinv(J)
        missing_kinematic = full_j[:,:6]@base_acc+full_jdot[:,:6]@env.mjData.qvel[:6]
        cart = J.T@feedback
        inertia = mass_matrix@inv@(a+feedback-J_dot@q_dot)
        correction = mass[np.ix_(ids,np.arange(6))]@base_acc-mass_matrix@inv@missing_kinematic
        for key,value in zip(self.last, (p,v,a,cart,inertia,correction,h,missing_kinematic)):
            self.last[key][leg_id] = value
        return cart+inertia+correction+h, p, v


def query_precontact(ground, xy, remaining, force, dt, effective_mass, virgin=False):
    """A once-initialized query copy, free recovery, then common-sole inverse."""
    branch = ground.clone()
    ids,areas = branch.footprint(np.asarray(xy))
    ids = np.asarray(ids,int)
    initial_e,initial_q = branch.e[ids].copy(),branch.q[ids].copy()
    changed = ids[branch.ever_loaded[ids] & ~branch.loaded[ids]]
    if virgin:
        branch.e[changed] = 0.
        branch.q[changed] = 0.
        branch.stress[changed] = 0.
        branch.last_area[changed] = 0.
        branch.last_normal[changed] = 0.
    query_e,query_q = branch.e[ids].copy(),branch.q[ids].copy()
    if len(ids) and branch.kind=='tm02_anchor':
        e,q,s,_ = branch.law.step(branch.e[ids],branch.q[ids],np.zeros(len(ids)),remaining,np.zeros(len(ids),bool))
        branch.e[ids],branch.q[ids],branch.stress[ids] = e,q,s
        surface = float(np.max(e[:,0])*branch.h)
    elif len(ids) and branch.kind.startswith('kv'):
        branch.e[ids,0] *= np.exp(-branch.terrain['k_N_per_m']*remaining/branch.terrain['d_N_s_per_m'])
        branch.q[ids] = branch.e[ids]
        surface = float(np.max(branch.e[ids,0]))
    else:
        surface = 0.
    inverse = branch.force_inverse(xy,force,dt)
    compression = max(0.,inverse['displacement_m']+surface)
    if branch.soft(xy):
        # Advance no more than one queried nominal-load compression per control
        # interval. TM02 additionally retains one more interval of domain margin
        # and the elastic stopping-distance bound for the current leg inertia.
        speed = compression/dt
        if branch.kind=='tm02_anchor':
            margin = max(0.,branch.h*.02-inverse['displacement_m'])
            stiffness = force/max(compression,np.finfo(float).eps)
            speed = min(speed, margin/(2*dt), margin*np.sqrt(stiffness/effective_mass))
    else:
        # Same normal permission, derived from one stance load acting through
        # the native rigid-contact regularization interval (2ms).
        speed = force*dt/effective_mass
    return dict(surface_m=surface, target_displacement_m=inverse['displacement_m'],
        approach_speed_m_s=float(speed), nominal_load_N=float(force), effective_mass_kg=float(effective_mass),
        query_dt_s=remaining, inverse_dt_s=dt, footprint_xy_m=list(xy), patch_ids=ids.tolist(),
        areas_m2=list(areas), world_e=initial_e.tolist(),world_q=initial_q.tolist(),
        initial_e=query_e.tolist(),initial_q=query_q.tolist(),
        reset_patch_ids=changed.tolist() if virgin else [],
        presently_loaded_ids=ids[branch.loaded[ids]].tolist(),
        predicted_e=branch.e[ids].tolist(),predicted_q=branch.q[ids].tolist(),
        inverse_force_N=inverse['forward_force_N'], inverse_residual_N=inverse['force_residual_N'])


class ContactTransition(CompliantStanceTracking):
    """C2 normal approach, actual-contact loading, then common-state support."""
    def __init__(self, env, wb, contact_env, common, controller):
        super().__init__(env,wb,contact_env)
        self.common,self.controller = common,controller
        self.nominal = np.ones(4,bool)
        self.previous_nominal = self.nominal.copy()
        self.support = np.zeros(4,bool)
        self.curves = [None]*4
        self.has_swung = np.zeros(4,bool)
        self.ever_supported = np.zeros(4,bool)
        self.ever_lost_support = np.zeros(4,bool)
        self.alpha_horizon = np.ones((4,12))
        self.touch_time = np.full(4,np.nan)
        self.mode = np.zeros(4,int)  # 0 native, 1 approach, 2 loading, 3 support
        self.reference = np.zeros((4,3,3))  # leg, position/velocity/acceleration, xyz
        self.reference[:,0] = self.anchor
        self.load_duration = min((2*wb.pgg.duty_factor-1)/(2*wb.pgg.step_freq),
                                 1/np.sqrt(wb.stc.position_gain_fb))
        self.queries = []
        self.transitions = []
        self.query_seconds = 0.
        self.mpc_sequence = np.ones((4,12))
        self.applied_grf = np.zeros((4,3))
        self.raw_update = wb.update_state_and_reference
        wb.update_state_and_reference = self.update_state

    def update_state(self,*args,**kw):
        result = self.raw_update(*args,**kw)
        state,ref,sequence,height,optimize = result
        t = self.env.mjData.time
        self.query_seconds = 0.
        self.previous_nominal = self.nominal.copy()
        self.nominal = self.wb.current_contact.astype(bool).copy()
        liftoff = ~self.nominal & self.previous_nominal
        self.has_swung[liftoff] = True
        for leg in np.flatnonzero(liftoff):
            self.curves[leg] = None
            self.touch_time[leg] = np.nan
            self.mode[leg] = 0
        actual = self.contact_env.contact.copy()
        # A leg deliberately lifting off remains a motion-controlled leg even
        # during its last unloading sample; descending early contact is usable.
        descending = np.array(self.wb.stc.swing_time)>=self.wb.stc.swing_period/2
        self.support = actual & (self.nominal | (descending & self.has_swung))
        lost = self.previous_actual_contact & ~self.support
        self.ever_lost_support[lost] = True
        for leg in np.flatnonzero(lost):
            self.curves[leg] = None
            self.touch_time[leg] = np.nan
            self.transitions.append(dict(time_s=float(t),leg=int(leg),event='support_lost_force_invalidated'))
        for leg in np.flatnonzero(self.support & ~self.previous_actual_contact):
            initial_support = not self.ever_supported[leg] and not self.has_swung[leg]
            self.touch_time[leg] = t-self.load_duration if initial_support else t
            self.ever_supported[leg] = True
            self.curves[leg] = None
            self.anchor[leg] = np.asarray(args[5][self.legs[leg]])
            self.transitions.append(dict(time_s=float(t),leg=int(leg),event='first_control_after_actual_contact'))
        for leg in np.flatnonzero(self.nominal & ~self.support):
            if self.curves[leg] is None:
                self.begin_recontact(leg)
        physical = sequence.copy()
        arrival_times = np.full(4,np.nan)
        physical[:,0] = self.support
        for leg,curve in enumerate(self.curves):
            if curve is not None and not self.support[leg]:
                # Delay only this pending touchdown, not every future stance.
                p = np.asarray(args[5][self.legs[leg]])
                gap = max(0.,p[2]-self.contact_env.radius[leg]-curve['surface'])
                arrival = curve['end'] if t<curve['end'] else t+gap/max(curve['speed'],np.finfo(float).eps)
                arrival_times[leg] = arrival
                elapsed = np.arange(physical.shape[1])*.02
                next_liftoff = (self.wb.pgg.duty_factor-self.wb.pgg.phase_signal[leg])%1/self.wb.pgg.step_freq
                physical[leg,(elapsed<arrival-t)&(elapsed<next_liftoff)] = 0
        self.mpc_sequence = physical.copy()
        alpha = np.zeros_like(physical)
        period = 1/self.wb.pgg.step_freq
        for leg in range(4):
            entry = self.touch_time[leg]
            nominal_touch = t+(1-self.wb.pgg.phase_signal[leg])*period
            pending = arrival_times[leg]
            next_liftoff = t+(self.wb.pgg.duty_factor-self.wb.pgg.phase_signal[leg])%1*period
            for k in range(physical.shape[1]):
                sample = t+k*.02
                if not physical[leg,k]:
                    continue
                if k>0 and not physical[leg,k-1]:
                    if np.isfinite(pending) and sample>=pending and sample<next_liftoff:
                        entry = pending
                    else:
                        entry = nominal_touch+np.floor((sample-nominal_touch+1e-12)/period)*period
                alpha[leg,k] = loading_fraction(sample-entry,self.load_duration)
        self.alpha_horizon = alpha[:,:12].copy()
        return state,ref,np.vstack((physical,alpha)),height,optimize

    def begin_recontact(self, leg):
        env,wb,ce = self.env,self.wb,self.contact_env
        name = self.legs[leg]
        t = float(env.mjData.time)
        position = np.asarray(env.feet_pos(frame='world')[name])
        velocity = np.asarray(env.feet_vel(frame='world')[name])
        full_j = env.feet_jacobians(frame='world',return_rot_jac=False)[name]
        jdot = env.feet_jacobians_dot(frame='world',return_rot_jac=False)[name]
        if self.mode[leg] in (1,2,3):
            p,v,a = self.reference[leg].copy()
        else:
            p,v,a = position.copy(),velocity.copy(),full_j@env.mjData.qacc+jdot@env.mjData.qvel
        jac = full_j[:,env.legs_qvel_idx[name]]
        effective_mass = float(1/(jac[2]@np.linalg.solve(env.legs_mass_matrix[name],jac[2])))
        duration = wb.stc.swing_period
        target = p.copy()
        start = time.perf_counter()
        pair = [query_precontact(ce.ground,target[:2],duration,env.mjModel.body_mass.sum()*9.81/2,
            .002,effective_mass,virgin=virgin) for virgin in (False,True)]
        self.query_seconds += time.perf_counter()-start
        chosen = pair[0]
        target[2] = ce.radius[leg]+chosen['surface_m']
        end_v = np.array([0.,0.,-chosen['approach_speed_m_s']])
        coeff = quintic(p,v,a,target,end_v,duration)
        self.curves[leg] = dict(start=t,end=t+duration,duration=duration,coefficients=coeff,
            speed=chosen['approach_speed_m_s'],surface=chosen['surface_m'])
        self.queries.append(dict(query_id=len(self.queries)+1,time_s=t,leg=leg,controller=self.controller,
            T1=pair[0],T2=pair[1],selected_prior='T1',reason='unsupported_nominal_stance',
            executed_variable='normal_approach_endpoint_and_speed',executed_endpoint_m=target.tolist(),
            executed_velocity_m_s=end_v.tolist(),curve_coefficients=coeff.tolist(),remaining_s=duration,
            actuator_limits_Nm=env.mjModel.actuator_ctrlrange.tolist()))
        self.transitions.append(dict(time_s=t,leg=leg,event='unsupported_nominal_stance_approach',
            reference_source='continuous' if self.mode[leg] in (1,2,3) else 'actual'))

    def compute(self, simulation_dt, qpos, qvel, feet_jac, feet_jac_dot, feet_pos, feet_vel,
                legs_qfrc_passive, legs_qfrc_bias, legs_mass_matrix, nmpc_GRFs, nmpc_footholds,
                legs_qpos_idx, legs_qvel_idx, tau, optimize_swing, best_sample_freq,
                nmpc_joints_pos, nmpc_joints_vel, nmpc_joints_acc, nmpc_predicted_state, mujoco_contact=None):
        begin = time.perf_counter()
        wb,env,ce = self.wb,self.env,self.contact_env
        t = float(env.mjData.time)
        tau,joints,joint_vel = type(wb).compute_stance_and_swing_torque(wb,simulation_dt,qpos,qvel,
            feet_jac,feet_jac_dot,feet_pos,feet_vel,legs_qfrc_passive,legs_qfrc_bias,legs_mass_matrix,
            nmpc_GRFs,nmpc_footholds,legs_qpos_idx,legs_qvel_idx,tau,optimize_swing,best_sample_freq,
            nmpc_joints_pos,nmpc_joints_vel,nmpc_joints_acc,nmpc_predicted_state,mujoco_contact)
        native_seconds = time.perf_counter()-begin
        mass = np.zeros((env.mjModel.nv,env.mjModel.nv))
        mujoco.mj_fullM(env.mjModel,env.mjData,mass)
        base_acc = env.mjData.qacc[:6].copy()
        position = np.array([feet_pos[n] for n in self.legs])
        velocity = np.array([feet_vel[n] for n in self.legs])
        grfs = np.array([nmpc_GRFs[n] for n in self.legs])
        alpha = np.zeros(4)
        alpha[self.support] = loading_fraction(t-self.touch_time[self.support],self.load_duration)
        projection_start = time.perf_counter()
        # nmpc_GRFs is the last solved force vector, held by the original wrapper.
        # No parameter re-extraction, changed gravity basis, or redistribution.
        from quadruped_pympc import config as cfg
        applied = project_contact_force(grfs,self.support,alpha,cfg.mpc_params['grf_max'],cfg.mpc_params['mu'])
        projection_seconds = time.perf_counter()-projection_start
        self.applied_grf = applied
        self.inverse_seconds = 0.
        feedback_tau = np.zeros((4,3))
        inertia_tau = np.zeros((4,3))
        bias_tau = np.zeros((4,3))
        grf_tau = np.zeros((4,3))
        inverse_residual = np.full(4,np.nan)
        inverse_force = np.full(4,np.nan)
        active = np.zeros(4,bool)
        kp,kd = wb.stc.position_gain_fb,wb.stc.velocity_gain_fb
        for leg,name in enumerate(self.legs):
            ids = legs_qvel_idx[name]
            jac = feet_jac[name][:,ids]
            inv = np.linalg.pinv(jac)
            raw_p = self.common.last['swing_reference_position'][leg]
            descending = not self.nominal[leg] and wb.stc.swing_time[leg]>=wb.stc.swing_period/2
            if descending and self.curves[leg] is None and not self.support[leg]:
                p = raw_p.copy()
                v = self.common.last['swing_reference_velocity'][leg].copy()
                a = self.common.last['swing_reference_acceleration'][leg].copy()
                remaining = wb.stc.swing_period-wb.stc.swing_time[leg]
                target = np.asarray(nmpc_footholds[name]).reshape(3).copy()
                effective_mass = float(1/(jac[2]@np.linalg.solve(legs_mass_matrix[name],jac[2])))
                tq=time.perf_counter()
                pair = [query_precontact(ce.ground,target[:2],remaining,env.mjModel.body_mass.sum()*9.81/2,
                    simulation_dt,effective_mass,virgin=virgin) for virgin in (False,True)]
                self.query_seconds += time.perf_counter()-tq
                chosen = pair[int(self.controller=='T2')]
                target[2] = ce.radius[leg]+chosen['surface_m']
                end_v=np.array([0.,0.,-chosen['approach_speed_m_s']])
                c = quintic(p,v,a,target,end_v,remaining)
                self.curves[leg]=dict(start=t,end=t+remaining,duration=remaining,coefficients=c,
                    speed=chosen['approach_speed_m_s'],surface=chosen['surface_m'])
                self.queries.append(dict(query_id=len(self.queries)+1,time_s=t,leg=leg,controller=self.controller,
                    T1=pair[0],T2=pair[1],selected_prior=self.controller,
                    executed_variable='normal_approach_endpoint_and_speed',
                    executed_endpoint_m=target.tolist(),executed_velocity_m_s=end_v.tolist(),
                    curve_coefficients=c.tolist(),remaining_s=remaining,
                    actuator_limits_Nm=env.mjModel.actuator_ctrlrange.tolist()))
                self.transitions.append(dict(time_s=t,leg=leg,event='normal_approach'))
            curve=self.curves[leg]
            if self.support[leg]:
                grf_tau[leg] = -jac.T@applied[leg]
                tau[name] = grf_tau[leg].copy()
                if not ce.ground.soft(position[leg,:2]):
                    self.mode[leg] = 2 if alpha[leg]<1 else 3
                    if self.has_swung[leg] or self.ever_lost_support[leg]:
                        p,v,a=self.reference[leg]
                        w=np.sqrt(kp)
                        jerk=w**3*(self.anchor[leg]-p)-3*w*w*v-3*w*a
                        d=simulation_dt
                        self.reference[leg]=p+v*d+.5*a*d*d+jerk*d**3/6,v+a*d+.5*jerk*d*d,a+jerk*d
                    else:
                        self.reference[leg]=self.anchor[leg],np.zeros(3),np.zeros(3)
                    continue
                tq=time.perf_counter()
                inverse=ce.ground.force_inverse(position[leg,:2],applied[leg,2],simulation_dt)
                self.inverse_seconds += time.perf_counter()-tq
                target=self.anchor[leg].copy()
                target[2]=ce.radius[leg]-inverse['displacement_m']
                if self.mode[leg] not in (1,2,3):
                    # Initial support, before any swing, starts from actual p/v/a.
                    self.reference[leg] = position[leg],velocity[leg],feet_jac[name]@env.mjData.qacc+feet_jac_dot[name]@qvel
                p,v,a=self.reference[leg]
                omega=np.sqrt(kp)
                jerk=omega**3*(target-p)-3*omega**2*v-3*omega*a
                d=simulation_dt
                p,v,a=p+v*d+.5*a*d*d+jerk*d**3/6,v+a*d+.5*jerk*d*d,a+jerk*d
                self.mode[leg]=2 if alpha[leg]<1 else 3
                inverse_residual[leg]=inverse['force_residual_N']
                inverse_force[leg]=inverse['forward_force_N']
                active[leg]=True
            elif curve is not None:
                p,v,a=evaluate(curve['coefficients'],t-curve['start'],curve['duration'])
                self.mode[leg]=1
                tau[name]=np.zeros(3)
            else:
                self.mode[leg] = 0
                # Native swing is the sole owner here; expose its actual terms.
                feedback_tau[leg] = self.common.last['swing_cartesian_feedback_torque'][leg]
                inertia_tau[leg] = (self.common.last['swing_joint_inertia_torque'][leg]
                                   +self.common.last['swing_base_correction_torque'][leg])
                bias_tau[leg] = self.common.last['swing_bias_torque'][leg]
                if np.all(np.isfinite(raw_p)):
                    self.reference[leg]=raw_p,self.common.last['swing_reference_velocity'][leg],self.common.last['swing_reference_acceleration'][leg]
                continue
            self.reference[leg]=p,v,a
            kinematic_bias=feet_jac_dot[name]@qvel+feet_jac[name][:,:6]@base_acc
            inertia_tau[leg]=legs_mass_matrix[name]@inv@(a-kinematic_bias)+mass[np.ix_(ids,np.arange(6))]@base_acc
            feedback_tau[leg]=legs_mass_matrix[name]@inv@(kp*(p-position[leg])+kd*(v-velocity[leg]))
            # Preserve the same Cartesian feedback for free motion as common B1.
            if not self.support[leg]:
                feedback_tau[leg]+=jac.T@(kp*(p-position[leg])+kd*(v-velocity[leg]))
            bias_tau[leg]=legs_qfrc_bias[name]
            tau[name]+=inertia_tau[leg]+feedback_tau[leg]+bias_tau[leg]
            wb.last_des_foot_pos[name]=p.copy()
        self.previous_actual_contact=self.support.copy()
        self.last_seconds=time.perf_counter()-begin
        self.last=dict(stance_tracking_active=active,stance_reference_position=self.reference[:,0].copy(),
            stance_reference_velocity=self.reference[:,1].copy(),stance_reference_acceleration=self.reference[:,2].copy(),
            stance_inverse_force_residual_N=inverse_residual,stance_inverse_forward_force_N=inverse_force,
            stance_native_grf_torque=grf_tau,stance_inertia_torque=inertia_tau,
            stance_tracking_torque=feedback_tau,stance_bias_torque=bias_tau,
            actual_support_available=self.support.copy(),transition_mode=self.mode.copy(),
            transition_load_fraction=alpha,executed_grf=applied,mpc_contact_sequence=self.mpc_sequence[:,:12].copy(),
            held_grf=grfs.copy(),commanded_grf=applied.copy(),force_projection_s=projection_seconds,
            held_force_minus_command_N=(grfs-applied).sum(axis=0),
            command_minus_hold_force_N=(applied-grfs).sum(axis=0),
            command_minus_hold_moment_Nm=np.cross(position-env.base_pos,applied-grfs).sum(axis=0),
            wrench_reference_point_m=env.base_pos.copy(),mpc_load_fraction=self.alpha_horizon.copy(),
            actual_contact_at_control=ce.contact.copy(),actual_grf_at_control=ce.force.copy(),
            final_leg_torque_requested=np.array([tau[name] for name in self.legs]),
            feet_acceleration=np.array([feet_jac[name]@env.mjData.qacc+feet_jac_dot[name]@qvel for name in self.legs]),
            precontact_query_count=len(self.queries),precontact_query_s=self.query_seconds,
            floating_base_acceleration=base_acc,whole_body_leg_execution_s=self.last_seconds,
            native_whole_body_leg_execution_s=native_seconds,constitutive_inverse_s=self.inverse_seconds,
            stance_tracking_algebra_s=self.last_seconds-native_seconds-self.inverse_seconds-self.query_seconds)
        return tau,joints,joint_vel
