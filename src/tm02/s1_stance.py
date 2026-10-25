"""Constitutive stance references executed inside the retained PyMPC WB path.

This finite leg inverse-dynamics implementation is not STANCE's joint WBOpt.
The native GRF map and friction compensation are retained exactly once.
"""
import time
import numpy as np
import mujoco


class CompliantStanceTracking:
    def __init__(self, env, wb, contact_env):
        self.env, self.wb, self.contact_env = env, wb, contact_env
        self.legs = tuple(wb.legs_order)
        feet = env.feet_pos(frame='world')
        self.anchor = np.array([feet[leg] for leg in self.legs])
        self.previous_actual_contact = np.zeros(4, bool)
        self.previous_ref_velocity = np.zeros((4, 3))
        self.previous_ref_position = self.anchor.copy()
        self.previous_active = np.zeros(4, bool)
        self.last = {}
        self.last_seconds = 0.
        self.inverse_seconds = 0.

    def install(self):
        self.wb.compute_stance_and_swing_torque = self.compute
        return self

    def clone(self, env, wb, contact_env):
        branch = type(self)(env, wb, contact_env)
        for name in ('anchor', 'previous_actual_contact', 'previous_ref_position', 'previous_ref_velocity', 'previous_active'):
            setattr(branch, name, getattr(self, name).copy())
        return branch.install()

    def record(self):
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self.last.items()}

    def compute(self, simulation_dt, qpos, qvel, feet_jac, feet_jac_dot, feet_pos, feet_vel,
                legs_qfrc_passive, legs_qfrc_bias, legs_mass_matrix, nmpc_GRFs, nmpc_footholds,
                legs_qpos_idx, legs_qvel_idx, tau, optimize_swing, best_sample_freq,
                nmpc_joints_pos, nmpc_joints_vel, nmpc_joints_acc, nmpc_predicted_state, mujoco_contact=None):
        begin = time.perf_counter()
        wb, env, ce = self.wb, self.env, self.contact_env
        # Execute the unchanged native stand/gait/swing/friction path first.
        tau, desired_joints, desired_joint_vel = type(wb).compute_stance_and_swing_torque(
            wb, simulation_dt, qpos, qvel, feet_jac, feet_jac_dot, feet_pos, feet_vel,
            legs_qfrc_passive, legs_qfrc_bias, legs_mass_matrix, nmpc_GRFs, nmpc_footholds,
            legs_qpos_idx, legs_qvel_idx, tau, optimize_swing, best_sample_freq,
            nmpc_joints_pos, nmpc_joints_vel, nmpc_joints_acc, nmpc_predicted_state, mujoco_contact)
        native_seconds = time.perf_counter()-begin
        position = np.array([feet_pos[leg] for leg in self.legs])
        velocity = np.array([feet_vel[leg] for leg in self.legs])
        actual_contact = ce.contact.copy()
        touchdown = actual_contact & ~self.previous_actual_contact
        self.anchor[touchdown] = position[touchdown]
        active = np.array([bool(wb.current_contact[i]) and actual_contact[i] and ce.ground.soft(position[i,:2])
                           for i in range(4)])
        ref_position = np.full((4, 3), np.nan)
        ref_velocity = np.full((4, 3), np.nan)
        ref_acceleration = np.full((4, 3), np.nan)
        inverse_residual = np.full(4, np.nan)
        inverse_force = np.full(4, np.nan)
        inverse_strain = np.full(4, np.nan)
        inertia_torque = np.zeros((4, 3))
        feedback_torque = np.zeros((4, 3))
        bias_torque = np.zeros((4, 3))
        native_grf_torque = np.array([-feet_jac[name][:,legs_qvel_idx[name]].T @ nmpc_GRFs[name]
                                     for name in self.legs])
        self.inverse_seconds = 0.
        if np.any(active):
            mass = np.zeros((env.mjModel.nv, env.mjModel.nv))
            mujoco.mj_fullM(env.mjModel, env.mjData, mass)
            # The unactuated base acceleration is the latest physics estimate,
            # held over this 2 ms control interval; it is not jointly optimized.
            base_acceleration = env.mjData.qacc[:6].copy()
            kp, kd = wb.stc.position_gain_fb, wb.stc.velocity_gain_fb
            for leg in np.flatnonzero(active):
                name = self.legs[leg]
                ti = time.perf_counter()
                inverse = ce.ground.force_inverse(position[leg,:2], nmpc_GRFs[name][2], simulation_dt)
                self.inverse_seconds += time.perf_counter()-ti
                target = self.anchor[leg].copy() if actual_contact[leg] else np.asarray(nmpc_footholds[name]).reshape(3).copy()
                # The centre is radius above the sole; total displacement is
                # measured from virgin z=0, not from the recovering surface.
                target[2] = ce.radius[leg]-inverse['displacement_m']
                # Different grid histories share one moving sole. Differentiate
                # this leg's reference, never a spatial average of column states.
                previous_position = self.previous_ref_position[leg] if self.previous_active[leg] else position[leg]
                target_velocity = np.array([0., 0., (target[2]-previous_position[2])/simulation_dt])
                previous_velocity = self.previous_ref_velocity[leg] if self.previous_active[leg] else velocity[leg]
                target_acceleration = np.array([0., 0., (target_velocity[2]-previous_velocity[2])/simulation_dt])
                joint_ids = legs_qvel_idx[name]
                jac = feet_jac[name][:,joint_ids]
                inverse_jac = np.linalg.pinv(jac)
                kinematic_bias = feet_jac_dot[name] @ qvel + feet_jac[name][:,:6] @ base_acceleration
                qdd_ff = inverse_jac @ (target_acceleration-kinematic_bias)
                qdd_feedback = inverse_jac @ (kp*(target-position[leg])+kd*(target_velocity-velocity[leg]))
                inertia_torque[leg] = legs_mass_matrix[name] @ qdd_ff + mass[np.ix_(joint_ids,np.arange(6))] @ base_acceleration
                feedback_torque[leg] = legs_mass_matrix[name] @ qdd_feedback
                bias_torque[leg] = legs_qfrc_bias[name]
                # Native -J.T GRF remains in tau; no second GRF feedforward.
                tau[name] += inertia_torque[leg]+feedback_torque[leg]+bias_torque[leg]
                wb.last_des_foot_pos[name] = target
                ref_position[leg], ref_velocity[leg], ref_acceleration[leg] = target, target_velocity, target_acceleration
                inverse_residual[leg] = inverse['force_residual_N']
                inverse_force[leg] = inverse['forward_force_N']
                inverse_strain[leg] = inverse['max_abs_principal_strain'] if inverse['max_abs_principal_strain'] is not None else np.nan
                self.previous_ref_velocity[leg] = target_velocity
                self.previous_ref_position[leg] = target
        self.previous_actual_contact = actual_contact
        self.previous_active = active
        self.last_seconds = time.perf_counter()-begin
        self.last = dict(stance_tracking_active=active, stance_reference_position=ref_position,
            stance_reference_velocity=ref_velocity, stance_reference_acceleration=ref_acceleration,
            stance_reference_effective_delay_s=simulation_dt,
            stance_anchor=self.anchor.copy(), stance_inverse_force_residual_N=inverse_residual,
            stance_inverse_forward_force_N=inverse_force, stance_inverse_max_principal_strain=inverse_strain,
            stance_native_grf_torque=native_grf_torque, stance_inertia_torque=inertia_torque,
            stance_tracking_torque=feedback_torque, stance_bias_torque=bias_torque,
            stance_inverse_s=self.inverse_seconds, stance_wb_s=self.last_seconds,
            constitutive_inverse_s=self.inverse_seconds, whole_body_leg_execution_s=self.last_seconds,
            native_whole_body_leg_execution_s=native_seconds,
            stance_tracking_algebra_s=self.last_seconds-native_seconds-self.inverse_seconds)
        return tau, desired_joints, desired_joint_vel
