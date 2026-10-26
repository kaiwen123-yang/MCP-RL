"""Go2 task execution using the pinned PyMPC sampler and whole-body interface."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import importlib.metadata
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.robot_cfgs import get_robot_config
from gym_quadruped.utils.quadruped_utils import LegsAttr
from quadruped_pympc import config as cfg

LEGS = ('FL', 'FR', 'RL', 'RR')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x:x.tolist()) + '\n')


def array(attr):
    return np.array([attr[l] for l in LEGS])


def angle(x):
    return np.arctan2(np.sin(x), np.cos(x))


class Mission:
    def __init__(self, task):
        self.task = task
        self.phase = 'initial_stand'
        self.entered = 0.
        self.stable_since = None
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.transitions = [dict(time_s=0., phase=self.phase)]

    def update(self, t, dt, pos, velocity, ori, omega, standing):
        c = self.task
        goal = np.array(c['turnaround_xy_m'] if self.phase in ('outbound', 'turn') else c['return_xy_m'])
        distance = np.linalg.norm(goal-pos[:2])
        yaw_goal = np.pi if self.phase in ('turn', 'return', 'terminal_stand', 'complete') else 0.
        old = self.phase
        if self.phase == 'initial_stand' and t >= c['initial_stand_s']:
            self.phase = 'outbound'
        elif self.phase == 'outbound' and distance < c['goal_position_radius_m'] and np.linalg.norm(velocity[:2]) < .06:
            self.phase = 'turn'
        elif self.phase == 'turn' and abs(angle(np.pi-ori[2])) < c['turn_heading_tolerance_rad']:
            self.phase = 'return'
        elif self.phase == 'return' and distance < .06 and np.linalg.norm(velocity[:2]) < .05:
            self.phase = 'terminal_stand'
        elif self.phase == 'terminal_stand':
            stable = (standing and distance < c['goal_position_radius_m'] and
                      np.linalg.norm(velocity) < c['terminal_speed_max_m_s'] and
                      np.linalg.norm(omega) < .1 and max(abs(ori[:2])) < .15)
            self.stable_since = (t if self.stable_since is None else self.stable_since) if stable else None
            if self.stable_since is not None and t-self.stable_since >= c['terminal_stand_s']:
                self.phase = 'complete'
        if old != self.phase:
            self.entered = t
            self.transitions.append(dict(time_s=float(t), phase=self.phase, position_m=pos.tolist()))
        desired = np.zeros(3)
        yaw_rate = 0.
        if self.phase in ('outbound', 'return'):
            goal = np.array(c['turnaround_xy_m'] if self.phase == 'outbound' else c['return_xy_m'])
            delta = goal-pos[:2]
            desired[:2] = delta * min(.8, c['command_speed_m_s']/max(np.linalg.norm(delta), 1e-12))
            yaw_goal = 0. if self.phase == 'outbound' else np.pi
            yaw_rate = np.clip(1.5*angle(yaw_goal-ori[2]), -.5, .5)
        elif self.phase == 'turn':
            desired[:2] = np.clip(.8*(np.array(c['turnaround_xy_m'])-pos[:2]), -.1, .1)
            yaw_rate = min(c['turn_yaw_rate_rad_s'], 1.5*((np.pi-ori[2])%(2*np.pi)))
        self.v += np.clip(desired-self.v, -.5*dt, .5*dt)
        self.w[2] += np.clip(yaw_rate-self.w[2], -dt, dt)
        if self.phase in ('initial_stand', 'terminal_stand', 'complete'):
            # A full-stance controller still needs a position-holding reference;
            # zero velocity alone permits accumulated terminal drift.
            self.v[:] = np.r_[np.clip(-.8*pos[:2], -.1, .1), 0.]
            self.w[:] = 0.
        return self.v.copy(), self.w.copy()


def configure(c, method):
    nominal = method in ('C0', 'B0', 'B1', 'B2', 'B3', 'T1', 'T2', 'T1R')
    rc = get_robot_config('go2')
    cfg.robot, cfg.robot_cfg = 'go2', rc
    cfg.robot_leg_joints, cfg.robot_feet_geom_names = rc.leg_joints, rc.feet_geom_names
    cfg.qpos0_js, cfg.hip_height = rc.qpos0_js, rc.hip_height
    cfg.hip_offset_x, cfg.hip_offset_y = 0., .095
    cfg.mpc_params.update(type='sampling', sampling_method='random_sampling', device='gpu',
        horizon=12, dt=.02, num_parallel_computations=1024, num_sampling_iterations=1,
        sigma_random_sampling=[.2, 3., 10.], control_parametrization='cubic_spline', num_splines=1,
        optimize_step_freq=nominal or method == 'C1', step_freq_available=[2.4] if nominal else [1.3, 2., 2.4],
        use_inertia_recomputation=False, gradient_refinement_steps=0, use_riccati_feedback=False,
        use_foothold_optimization=False, use_foothold_constraints=False, use_integrators=False,
        external_wrenches_compensation=False, use_RTI=False, use_nonuniform_discretization=False,
        shift_solution=False, mu=.5, paper_input_cost=True,
        paper_Q=[0., 0., 111500., 5000., 5000., 200., 11200., 11200., 0., 20., 20., 600.])
    cfg.locomotion_params.update(liftoff_frame='world', phase_consistent_swing=True,
        swing_generator='scipy', swing_position_gain_fb=500., swing_velocity_gain_fb=10.,
        step_height=.2*rc.hip_height, ref_z=rc.hip_height, reflex_trigger_mode=False,
        reflex_next_steps_height_enhancement=False, velocity_modulator=False,
        start_and_stop_activated=False, use_friction_compensation=False,
        visual_foothold_adaptation='blind', gait='trot')
    cfg.locomotion_params['gait_params']['trot'].update(step_freq=2.4, duty_factor=.65)
    cfg.simulation_params.update(scene='flat', dt=.002, mpc_frequency=50)
    if method in ('T1','T2','T1R'):
        cfg.mpc_params['optimize_step_freq'] = False  # supplied actual/future contact sequence; fixed 2.4 Hz


def timing(values, deadline):
    x = np.array(values)
    x = x[np.isfinite(x)]
    return dict(count=len(x), p50_ms=float(np.median(x)*1000), p95_ms=float(np.percentile(x, 95)*1000),
                max_ms=float(x.max()*1000), deadline_misses=int(np.sum(x>deadline)), deadline_s=deadline) if len(x) else None


def run(root, args):
    c = json.loads((root/args.config).read_text())
    transition = c.get('stage') in ('S1_contact_transition_and_precontact_information','S1_contact_execution_repair')
    repair = transition or c.get('stage') == 'S1_control_execution_and_prediction_repair'
    if repair:
        c['resolved_numerics'] = c['numerics']
    configure(c, args.controller)
    from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper
    from quadruped_pympc.interfaces.srbd_controller_interface import SRBDControllerInterface
    from quadruped_pympc.helpers.quadruped_utils import GaitType
    from .s1_contact import ContactEnvironment, StanceReferenceInfeasible
    from .reference import IntegrationFailure
    base = json.loads((root/c['base']['material_config']).read_text())
    terrain = next(x for x in c['terrains'] if x['id'] == args.terrain)
    substeps = c.get('resolved_numerics', {}).get('physics_substeps', 1)
    env = QuadrupedEnv(robot='go2', scene='flat', sim_dt=.002, ref_base_lin_vel=0.,
        ref_base_ang_vel=0., ground_friction_coeff=.5, base_vel_command_type='forward', state_obs_names=())
    # The default native contact permits ~6 mm of apparent rigid-floor
    # penetration. At a material boundary that becomes a spurious imposed
    # material strain. Resolve the intended rigid z=0 plane with stiff native
    # constraint regularization, without changing any robot inertial parameter.
    hard_geoms = np.r_[array(env._feet_geom_id), np.flatnonzero(env.mjModel.geom_bodyid==0)]
    env.mjModel.geom_solref[hard_geoms] = [.002, 1.]
    env.mjModel.geom_solimp[hard_geoms] = [.99, .999, .001, .5, 2.]
    env.mjModel.geom_margin[hard_geoms] = 0.
    env.mjModel.opt.timestep = .002/substeps
    env.reset(seed=0, random=False)
    cfg.mass = float(env.mjModel.body_mass.sum())
    cfg.inertia = env.get_base_inertia()
    cfg.mpc_params['grf_max'] = cfg.mass*cfg.gravity_constant
    env.terrain_limits = (1000., -1000., 1000., -1000.)
    env.mjModel.opt.timestep = .002/substeps
    feet_ids = array(env._feet_geom_id)
    native_type = env.mjModel.geom_contype[feet_ids].copy()
    native_affinity = env.mjModel.geom_conaffinity[feet_ids].copy()
    out = root/c['outputs']['large_root']/args.cohort
    small = root/c['outputs']['compact_root']/args.cohort
    out.mkdir(parents=True, exist_ok=True)
    small.mkdir(parents=True, exist_ok=True)
    upstream = root/'work/terrain-memory/upstreams/Quadruped-PyMPC'
    upstream_patch = subprocess.check_output(['git','diff','--binary','HEAD'],cwd=upstream)
    (out/'pinned-upstream-executed.patch').write_bytes(upstream_patch)
    snapshot = out/'source'
    snapshot.mkdir(exist_ok=True)
    for path in [root/'scripts/run_terrain_memory_s1.py',root/args.config,*sorted((root/'src/tm02').glob('s1*.py'))]:
        (snapshot/path.name).write_bytes(path.read_bytes())
    nominal_feet = array(env.feet_pos())
    fore_hind = float(np.mean(nominal_feet[:2, 0])-np.mean(nominal_feet[2:, 0]))
    resolved = dict(config=c, controller_parameters=cfg.mpc_params, locomotion_parameters=cfg.locomotion_params,
        actual_mass_kg=cfg.mass, inertia_kg_m2=cfg.inertia.tolist(), nominal_feet_m=nominal_feet.tolist(),
        fore_hind_spacing_m=fore_hind, event_horizon_s=max(2.5, (fore_hind+.08)/.25+.6),
        joint_names=[mujoco.mj_id2name(env.mjModel, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(env.mjModel.njnt)],
        joint_ranges_rad=env.mjModel.jnt_range.tolist(), actuator_limits_Nm=env.mjModel.actuator_ctrlrange.tolist(),
        initial_qpos=env.mjData.qpos.tolist(), foot_geom_ids=array(env._feet_geom_id).tolist(),
        foot_geom_size_m=env.mjModel.geom_size[array(env._feet_geom_id)].tolist(),
        native_contact_solref=[.002, 1.], native_contact_solimp=[.99, .999, .001, .5, 2.], native_contact_margin_m=0.,
        acceleration_m_s2=.5, heading_gain_s_inv=1.5, position_gain_s_inv=.8, terminal_stance_entry_radius_m=.06,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_sha256={str(p.relative_to(root)):digest(p) for p in [root/'scripts/run_terrain_memory_s1.py', *sorted((root/'src/tm02').glob('s1*.py'))]},
        host=subprocess.check_output(['hostname'], text=True).strip(), gpu=str(jax.devices()),
        cpu=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip(),
        upstream_patch_sha256=hashlib.sha256(upstream_patch).hexdigest(),
        package_versions={p:importlib.metadata.version(p) for p in ('mujoco','gym-quadruped','jax','jaxlib','numpy','scipy')},
        python=sys.version,
        material_state_units={'tm02_anchor':'e,q are strain; stress is original MPa before gamma scaling',
                              'kv':'e[:,0]=q[:,0]=surface height m; remaining columns unused; no 3D strain interpretation'},
        new_training_runs=0)
    resolved['execution_fingerprint'] = hashlib.sha256(json.dumps([c, resolved['source_sha256'], resolved['upstream_patch_sha256']],sort_keys=True).encode()).hexdigest()
    save_json(small/f'{args.terrain}-{args.controller}-resolved.json', resolved)
    mujoco.mj_saveModel(env.mjModel, str(out/'go2.mjb'), None)
    for seed in args.seeds:
        name = f'{args.terrain}-{args.controller}-seed-{seed}'
        if (small/f'{name}.json').exists():
            existing = json.loads((small/f'{name}.json').read_text())
            if existing['execution_fingerprint'] != resolved['execution_fingerprint']:
                raise ValueError('Saved episode belongs to another execution version; retain it under its original cohort.')
            print(json.dumps(dict(reused=name)), flush=True)
            continue
        if repair:
            args.register_start(seed, resolved['execution_fingerprint'])
        env.mjModel.geom_contype[feet_ids] = native_type
        env.mjModel.geom_conaffinity[feet_ids] = native_affinity
        env.mjData.qfrc_applied[:] = 0.
        env.reset(seed=0, random=False)
        weights = Path(args.source_output_root)/c['base']['frozen_predictor']['weights_relative_to_source_run'] if args.controller=='C4' else None
        if weights and digest(weights)!=c['base']['frozen_predictor']['weights_sha256']:
            raise ValueError('The frozen B ideal weights differ from the specified published model.')
        contact_env = ContactEnvironment(env, c, terrain, base['reference_material']['parameters'], weights)
        wrapper = QuadrupedPyMPC_Wrapper(env.feet_pos, legs_order=LEGS, feet_geom_id=env._feet_geom_id)
        moving = wrapper.srbd_controller_interface
        opt = cfg.mpc_params['optimize_step_freq']
        cfg.mpc_params['optimize_step_freq'] = False
        stand = SRBDControllerInterface()
        cfg.mpc_params['optimize_step_freq'] = opt
        solve_times, compilation = [], []
        last = dict(solve=np.nan, first=False)
        for interface in (moving, stand):
            ctl = interface.controller
            ctl.master_key = jax.random.PRNGKey(100000+seed)
            ctl.sampling_key = ctl.master_key
            raw = interface.compute_control
            raw_jit = ctl.jitted_compute_control
            ctl._s1_raw_jit = raw_jit
            signatures = set()
            def optimizer(*a, _jit=raw_jit, _signatures=signatures, **kw):
                signature = tuple((np.shape(x),str(getattr(x,'dtype',type(x).__name__)),getattr(x,'weak_type',False))
                                  for x in jax.tree_util.tree_leaves((a,kw)))
                first = signature not in _signatures
                _signatures.add(signature)
                result = _jit(*a, **kw)
                jax.block_until_ready(result)
                last['first'] = first
                return result
            ctl.jitted_compute_control = optimizer
            def timed(*a, _raw=raw, **kw):
                start = time.perf_counter()
                result = _raw(*a, **kw)
                elapsed = time.perf_counter()-start
                last['solve'] = elapsed
                (compilation if last['first'] else solve_times).append(elapsed)
                return result
            interface.compute_control = timed
        wb = wrapper.wb_interface
        planner = None
        tracker = None
        common_swing = None
        if args.controller in ('C2', 'C3', 'C4'):
            from .s1_planner import IntentPlanner, NoFeasibleIntent
            planner = IntentPlanner(env, wrapper, contact_env, args.controller, resolved['event_horizon_s'], cfg.mass, cfg.inertia)
        wb.pgg.set_full_stance()
        wrapper.srbd_controller_interface = stand
        if args.controller in ('B1', 'B2', 'B3', 'T1', 'T2', 'T1R'):
            from .s1_stance import CompliantStanceTracking
            from .s1_prediction import AlignedPlanner, NoFeasibleIntent
            tracker = CompliantStanceTracking(env, wb, contact_env).install()
            if not transition:
                planner = AlignedPlanner(env, wrapper, contact_env, args.controller,
                    resolved['event_horizon_s'], cfg.mass, cfg.inertia, tracker, moving, stand)
            else:
                from .s1_transition import FloatingBaseSwing, ContactTransition
                common_swing = FloatingBaseSwing(env, wb)
                if args.controller in ('T1','T2','T1R'):
                    tracker = ContactTransition(env, wb, contact_env, common_swing, args.controller).install()
        mission = Mission(c['task'])
        tau = LegsAttr(*[np.zeros(3) for _ in LEGS])
        hist = {}
        reason = ''
        integration_error = None
        reference_failure = None
        duration_s = 0.
        begin = time.perf_counter()
        for step in range(round(c['task']['maximum_episode_s']/.002)):
            start = time.perf_counter()
            t = step*.002
            mujoco.mj_forward(env.mjModel, env.mjData)
            pos, v = env.base_pos.copy(), env.base_lin_vel(frame='world').copy()
            ori, omega = env.base_ori_euler_xyz.copy(), env.base_ang_vel(frame='base').copy()
            standing = wb.pgg.gait_type == GaitType.FULL_STANCE.value
            refv, refw = mission.update(t, .002, pos, v, ori, omega, standing and np.all(contact_env.contact))
            want_stand = mission.phase in ('initial_stand', 'terminal_stand', 'complete')
            if want_stand and not standing and np.all(wb.current_contact):
                wb.pgg.set_full_stance()
                wrapper.srbd_controller_interface = stand
                wrapper.last_mpc_step_num = None
            elif not want_stand and standing:
                wb.pgg.restore_previous_gait()
                wrapper.srbd_controller_interface = moving
                wrapper.last_mpc_step_num = None
            feet = env.feet_pos(frame='world')
            fv = env.feet_vel(frame='world')
            upper_s = 0.
            if args.controller == 'B1' and not transition and step > 0 and step%500 == 0:
                planner.observe(t, mission, step, tau)
                upper_s = planner.last_upper_s
            elif args.controller in ('B2', 'B3') and not want_stand and step%100 == 0:
                try:
                    planner.choose(t, mission, step, tau)
                except NoFeasibleIntent:
                    reason = 'controller_'+planner.choices[-1]['failure']
                    duration_s = t
                    break
                upper_s = planner.last_upper_s
            elif planner and not repair and not want_stand and step%100 == 0:
                try:
                    planner.choose(t, mission)
                except NoFeasibleIntent:
                    reason = 'controller_no_feasible_intent'
                    duration_s = t
                    break
                upper_s = planner.last_upper_s
            last.update(solve=np.nan, first=False)
            tc = time.perf_counter()
            if repair and planner and planner.commitment:
                planner.commitment.time = t
            if common_swing:
                common_swing.begin()
            try:
                tau = wrapper.compute_actions(env.com.copy(), pos, v, ori, omega, feet,
                    env.hip_positions(frame='world'), env.legs_qvel_idx, None, LEGS, .002,
                    refv, refw, step, env.mjData.qpos, env.mjData.qvel,
                    env.feet_jacobians(frame='world', return_rot_jac=False),
                    env.feet_jacobians_dot(frame='world', return_rot_jac=False), fv,
                    env.legs_qfrc_passive, env.legs_qfrc_bias, env.legs_mass_matrix,
                    env.legs_qpos_idx, env.legs_qvel_idx, tau, cfg.inertia.flatten(), env.mjData.contact)
            except StanceReferenceInfeasible as error:
                reference_failure = dict(time_s=t, **error.detail)
                reason = 'stance_reference_infeasible'
                duration_s = t
                break
            except IntegrationFailure as error:
                reference_failure = dict(time_s=t, reason='constitutive_inverse_integration_failure', error=str(error))
                reason = 'stance_reference_integration_failure'
                duration_s = t
                break
            if planner and not repair:
                tau = planner.stance_torque(tau, feet, fv, t)
            requested = np.zeros(env.mjModel.nu)
            for leg in LEGS:
                requested[env.legs_tau_idx[leg]] = tau[leg]
            action = np.clip(requested, env.mjModel.actuator_ctrlrange[:, 0], env.mjModel.actuator_ctrlrange[:, 1])
            control_s = time.perf_counter()-tc
            ac, _, grf = env.feet_contact_state(ground_reaction_forces=True)
            record = dict(time=t, wall_start_s=start-begin, qpos=env.mjData.qpos.copy(), qvel=env.mjData.qvel.copy(),
                base_pos=pos, velocity=v, orientation=ori, angular_velocity=omega,
                command=refv, angular_command=refw, phase=list(('initial_stand','outbound','turn','return','terminal_stand','complete')).index(mission.phase),
                torque=action, requested_torque=requested, feet=array(feet), feet_velocity=array(fv),
                planned_grf=array(wrapper.nmpc_GRFs), actual_grf=array(grf), contact=array(ac),
                planned_contact=wb.current_contact.copy(), frequency=float(wb.pgg.step_freq),
                footholds=array(wrapper.nmpc_footholds), swing_target=array(wb.last_des_foot_pos),
                solve_s=last['solve'], compile_included=last['first'], control_s=control_s, upper_s=upper_s,
                executed_intent=np.array(wb.pgg.intent) if planner and not repair else np.array([0., wb.pgg.step_freq, 0.]),
                intent_serial=wb.pgg.intent_serial if planner and not repair else -1)
            if tracker:
                record.update(tracker.record())
            if common_swing:
                record.update(common_swing.last)
            if repair and planner and planner.commitment:
                record['executed_intent'] = np.array(planner.selected)
                record['intent_serial'] = planner.commitment.serial
            tp = time.perf_counter()
            material_s, predictor_s = 0., 0.
            nonfoot = []
            for sub in range(substeps):
                try:
                    terminated, info, nonfoot = contact_env.advance(action, .002/substeps, t+sub*.002/substeps,
                        update_predictor=sub==0, record_material=sub==substeps-1)
                except IntegrationFailure as error:
                    integration_error = dict(time_s=t+sub*.002/substeps, error=str(error))
                    reason = 'material_integration_failure'
                    terminated = False
                    break
                material_s += contact_env.material_seconds
                predictor_s += contact_env.predictor_seconds
                if terminated or nonfoot or contact_env.ground.domain_exit:
                    break
            record['physics_s'] = time.perf_counter()-tp
            duration_s = t+(sub+1)*.002/substeps
            record['actual_grf_sample_time_s'] = t+sub*.002/substeps
            record['actual_grf_post_step_time_s'] = float(env.mjData.time)
            record['actual_grf'] = contact_env.force.copy()
            record['contact'] = contact_env.contact.copy()
            record['indentation_m'] = contact_env.indent.copy()
            record['material_s'], record['predictor_s'] = material_s, predictor_s
            record['total_s'] = time.perf_counter()-start
            for key, value in record.items():
                hist.setdefault(key, []).append(value)
            if contact_env.ground.domain_exit:
                reason = 'material_domain_exit'
            elif nonfoot:
                reason = 'soft_region_nonfoot_contact:'+','.join(nonfoot)
            elif terminated:
                reason = 'invalid_contact:'+','.join(info['invalid_contacts'])
            elif max(abs(env.base_ori_euler_xyz[:2])) > 1.2 or env.base_pos[2] < .12:
                reason = 'body_fall'
            if reason or mission.phase == 'complete':
                break
            if step % 2500 == 0:
                print(json.dumps(dict(episode=name, time_s=t, phase=mission.phase, x_m=float(pos[0]), wall_s=time.perf_counter()-begin)), flush=True)
        hist = {k:np.asarray(vv) for k, vv in hist.items()}
        success = mission.phase == 'complete' and not reason
        reason = reason or ('completed' if success else 'task_timeout')
        trace = out/f'{name}.npz'
        np.savez_compressed(trace, **hist, final_qpos=env.mjData.qpos, final_qvel=env.mjData.qvel)
        material_trace = out/f'{name}-material.npz'
        contact_env.save(material_trace)
        save_json(out/f'{name}-events.json', dict(reference=contact_env.ground.events,
            predictor=contact_env.predictor.events if contact_env.predictor is not contact_env.ground else None))
        if planner:
            save_json(out/f'{name}-plans.json', planner.choices)
            if repair and planner.commitment:
                save_json(out/f'{name}-footstep-commitments.json', planner.commitment.applied)
        if transition and args.controller in ('T1','T2','T1R'):
            save_json(out/f'{name}-queries.json', tracker.queries)
            save_json(out/f'{name}-transitions.json', tracker.transitions)
        error = hist['velocity']-hist['command']
        qvel_idx = env.mjModel.jnt_dofadr[env.mjModel.actuator_trnid[:, 0]]
        power = hist['torque']*hist['qvel'][:, qvel_idx]
        physical = np.array(contact_env.physics_hist)
        contact_power = np.sum(physical[:, 13:25].reshape(-1, 4, 3)*physical[:, 25:37].reshape(-1, 4, 3), axis=2)
        row = dict(terrain=args.terrain, controller=args.controller, seed=seed, success=success,
            termination=reason, duration_s=duration_s, final_position_m=env.base_pos.tolist(),
            velocity_rmse_m_s=float(np.sqrt(np.mean(error**2))),
            attitude_rms_rad=float(np.sqrt(np.mean(hist['orientation'][:, :2]**2))),
            joint_positive_mechanical_work_J=float(np.maximum(power, 0).sum()*.002),
            signed_robot_contact_work_J=float(contact_power.sum()*.002/substeps),
            positive_ground_contact_work_J=float(np.maximum(-contact_power, 0).sum()*.002/substeps),
            grf_rmse_N=float(np.sqrt(np.mean((hist['actual_grf']-hist['planned_grf'])**2))),
            angular_velocity_rmse_rad_s=float(np.sqrt(np.mean((hist['angular_velocity']-hist['angular_command'])**2))),
            max_roll_pitch_rad=float(np.max(np.abs(hist['orientation'][:,:2]))),
            torque_saturation_fraction=float(np.mean(np.abs(hist['requested_torque']-hist['torque'])>1e-8)),
            max_joint_limit_excess_rad=float(np.maximum(np.maximum(env.mjModel.jnt_range[1:,0]-hist['qpos'][:,7:],
                hist['qpos'][:,7:]-env.mjModel.jnt_range[1:,1]),0).max()),
            contact_slip_distance_m=float((np.linalg.norm(physical[:,13:25].reshape(-1,4,3)[:,:,:2],axis=2)*
                (physical[:,25:37].reshape(-1,4,3)[:,:,2]>0)).sum()*.002/substeps),
            wall_s=time.perf_counter()-begin, compilation_s=compilation,
            timing=dict(mpc=timing(solve_times, .02), control=timing(hist['control_s'][~hist['compile_included']], .002),
                        physics=timing(hist['physics_s'], .002), total=timing(hist['total_s'][~hist['compile_included']], .002)),
            transitions=mission.transitions, trajectory=str(trace.relative_to(root)), sha256=digest(trace),
            source_sha256=resolved['source_sha256'])
        row['execution_fingerprint'] = resolved['execution_fingerprint']
        row['first_material_call_including_compile_s'] = contact_env.predictor.law.first_call_including_compile_s if args.controller=='C4' else None
        row['first_material_call_since_episode_start_wall_s'] = (contact_env.predictor.law.first_call_started_perf_s-begin
            if args.controller=='C4' and contact_env.predictor.law.first_call_started_perf_s is not None else None)
        row.update(material_domain_exit=contact_env.ground.domain_exit, integration_error=integration_error,
            stance_reference_failure=reference_failure,
            material_rate_extrema=contact_env.ground.rate_extrema,
            predictor_domain_exit=contact_env.predictor.domain_exit,
            max_indentation_mm=float(np.max(hist['indentation_m'])*1000),
            max_abs_principal_strain=contact_env.ground.max_strain if args.terrain=='tm02_anchor' else None,
            max_original_stress_MPa=contact_env.ground.max_stress if args.terrain=='tm02_anchor' else None,
            max_strain_rate_s_inv=contact_env.ground.max_rate if args.terrain=='tm02_anchor' else None,
            max_KV_deformation_rate_m_s=contact_env.ground.max_rate if args.terrain.startswith('kv') else None,
            max_newton_residual_MPa=contact_env.ground.max_residual,
            patches=len(contact_env.ground.xy), contact_events=len(contact_env.ground.events),
            revisits=sum(x['type']=='revisit' for x in contact_env.ground.events),
            material_trajectory=str(material_trace.relative_to(root)), material_sha256=digest(material_trace))
        row['timing'].update(material=timing(hist['material_s'], .002), predictor=timing(hist['predictor_s'], .002))
        row['timing']['upper'] = timing([x['wall_s'] for x in planner.choices], .2) if planner else None
        if tracker:
            for key in ('whole_body_leg_execution_s', 'native_whole_body_leg_execution_s',
                        'constitutive_inverse_s', 'stance_tracking_algebra_s'):
                row['timing'][key.removesuffix('_s')] = timing(hist[key], .002)
        if transition:
            row['common_swing_correction'] = c['resolved_transition']['common_swing_correction']
            row['events_path'] = str(out/f'{name}-events.json')
            row['model_path'] = str(out/'go2.mjb')
            revisits = [e for e in contact_env.ground.events if e['type']=='revisit' and
                e.get('previous_leg') in (0,1) and e['leg'] in (2,3) and
                e.get('unloaded_gap_s') is not None and e['unloaded_gap_s']>0 and e['area_m2']>0 and e['normal_N']>0]
            row['front_to_hind_revisit'] = bool(revisits)
            row['front_to_hind_column_events'] = len(revisits)
            row['precontact_query_used'] = args.controller in ('T1','T2','T1R') and len(tracker.queries)>0
            if args.controller in ('T1','T2','T1R'):
                row['queries_path'] = str(out/f'{name}-queries.json')
                row['transitions_path'] = str(out/f'{name}-transitions.json')
                row['timing']['precontact_query'] = timing(hist['precontact_query_s'],.002)
                if args.controller=='T1R':
                    row['timing']['force_projection'] = timing(hist['force_projection_s'],.002)
        save_json(small/f'{name}.json', row)
        if repair:
            args.register_result(seed, small/f'{name}.json')
        print(json.dumps(row), flush=True)
    env.close()
