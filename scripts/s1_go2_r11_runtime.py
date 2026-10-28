#!/usr/bin/env python3
"""Complete S1 missions through the pinned author's headless Agent.

No old locomotion controller is imported. Physics is advanced only in run();
render() replays recorded states without integrating another trajectory.
"""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'experiments/terrain_memory/s1-go2-wb-baseline.json'
PHASES = ['initial_stand', 'outbound', 'turn', 'return', 'terminal_stand', 'complete', 'far_hold', 'home_hold', 'turn_home']
LEGS = ['FL', 'FR', 'RL', 'RR']


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))+'\n')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def git(path):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()


def angle(x):
    return np.arctan2(np.sin(x), np.cos(x))


class Mission:
    """Original s1_robot.Mission decisions and bounded velocity references."""
    def __init__(self, task):
        self.task, self.phase, self.stable_since = task, 'initial_stand', None
        self.v, self.w = np.zeros(3), np.zeros(3)
        self.transitions = []
        self.body_target = np.array([0., 0., .27])
        self.heading = 0.

    def update(self, t, dt, pos, velocity, ori, omega, standing):
        c = self.task
        goal = np.array(c['turnaround_xy_m'] if self.phase in ('outbound', 'turn') else c['return_xy_m'])
        distance = np.linalg.norm(goal-pos[:2])
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
        if old != self.phase or not self.transitions:
            self.transitions.append(dict(time_s=float(t), phase=self.phase, position_m=pos.copy(),
                velocity_m_s=velocity.copy(), rpy_rad=ori.copy(), omega_rad_s=omega.copy(), all_feet_contact=bool(standing)))
        desired, yaw_rate = np.zeros(3), 0.
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
            self.v[:] = np.r_[np.clip(-.8*pos[:2], -.1, .1), 0.]
            self.w[:] = 0.
        # Integrate the original bounded world velocity along the fixed route;
        # the reference stops at its endpoint instead of passing through it.
        if self.phase in ('outbound', 'return'):
            self.body_target[0] = np.clip(self.body_target[0]+self.v[0]*dt, 0., 3.)
        if self.phase == 'turn':
            self.heading = min(np.pi, self.heading+max(0., self.w[2])*dt)
        elif self.phase in ('return', 'terminal_stand', 'complete'):
            self.heading = min(np.pi, self.heading+c['turn_yaw_rate_rad_s']*dt)
        return self.body_target.copy(), self.heading


def contact_state(model, data, feet, allowed_bodies, full=False):
    contact, force = np.zeros(4, dtype=bool), np.zeros((4, 3))
    invalid = []
    total_force, nonfoot_force, nonfoot_contacts = np.zeros(3), np.zeros(3), 0
    for k in range(data.ncon):
        con = data.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if b1 == 0 and b2 not in allowed_bodies:
            invalid.append([g1, g2, b1, b2])
        if b2 == 0 and b1 not in allowed_bodies:
            invalid.append([g1, g2, b1, b2])
        world_contact = (b1==0 and b2!=0) or (b2==0 and b1!=0)
        if world_contact:
            wrench=np.zeros(6)
            mujoco.mj_contactForce(model,data,k,wrench)
            world_force=con.frame.reshape(3,3).T@wrench[:3]*(1 if b1==0 else -1)
            total_force+=world_force
            if g1 not in feet and g2 not in feet:
                nonfoot_force+=world_force; nonfoot_contacts+=1
        for i, foot in enumerate(feet):
            if (foot == g1 and b2 == 0) or (foot == g2 and b1 == 0):
                contact[i] = True
                force[i] += world_force
    return (contact,force,invalid,total_force,nonfoot_force,nonfoot_contacts) if full else (contact,force,invalid)


def timings(values, deadline):
    a = np.asarray(values)
    return dict(count=len(a), total_s=float(a.sum()), p50_ms=float(np.median(a)*1e3),
        p95_ms=float(np.percentile(a, 95)*1e3), max_ms=float(a.max()*1e3),
        logical_deadline_s=deadline, deadline_misses=int((a > deadline).sum())) if len(a) else None


@contextlib.contextmanager
def index_file(root, config):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root/'index.json'
        idx = json.loads(path.read_text()) if path.exists() else dict(
            run_id=config['run_id'], status='executing', attempts=[], stages={},
            budget=config.get('campaign_budget', config.get('budget')), historical_reference=config.get('frozen_reference', config.get('frozen_previous')),
            training_runs=0, hardware_runs=0, nx_realtime_measured=False)
        yield idx
        dump(path, idx)


def run(args, c, joint=None):
    from mujoco_mpc.agent import Agent
    small = ROOT/c['outputs']['root']
    out = ROOT/'outputs/terrain_memory/simulation'/c['run_id']/args.attempt
    out.mkdir(parents=True, exist_ok=False)
    campaign = 'campaign_budget' in c
    support = True
    training = bool(c.get('r11_training'))
    policy_hook = bool(c.get('r11_policy'))
    compact = training and not joint.full_recording
    if campaign:
        with index_file(small, c) as idx:
            idx['attempts'].append(dict(attempt=args.attempt, stage=args.stage, purpose=args.purpose,
                condition=args.condition, arm=args.arm, mission=args.mission, cohort=args.cohort,
                status='initializing', physical_started=False, command=sys.argv,
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                case_key=c.get('case_key'), case=c.get('case_identity'), evaluation_stage=c['r11_stage']))
    upstream = Path(args.author_source).resolve()
    if args.stage == 'G0':
        model_path = upstream/'mjpc/tasks/quadruped/task_flat.xml'
        model = mujoco.MjModel.from_xml_path(str(model_path))
        model_meta = {'identity': 'author_native', 'xml_sha256': sha(model_path),
            'robot_xml_sha256': sha(upstream/'mjpc/tasks/quadruped/go2.xml')}
        model_meta['source_actuator_biastype'] = model.actuator_biastype.copy()
        # The pinned task XML supplies PD coefficients but omits affine bias.
        # Explicitly activate that approved input representation on BOTH models.
        model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
        model_meta['declared_input_repair'] = 'Enable affine bias for the existing gainprm=60 and biasprm=[0,-60,-5]; native physical and unlimited-force settings are retained.'
        # The 50 MB native mesh-bearing MJB exceeds agent_server's 20 MiB RPC
        # limit. Official XML transport sends only source, with on-host assets.
        prepared_xml = out/'model.xml'
        mujoco.mj_saveLastXML(str(prepared_xml), model)
        xml = ET.parse(prepared_xml)
        xml.getroot().find('compiler').set('meshdir', str(upstream/'mjpc/tasks/quadruped/assets'))
        # mj_saveLastXML does not copy back actuator_biastype mutations; the
        # explicit source attribute must survive the subsequent XML Init.
        for actuator in xml.getroot().find('actuator'):
            actuator.set('biastype', 'affine')
        xml.write(prepared_xml, encoding='unicode')
        model = mujoco.MjModel.from_xml_path(str(prepared_xml))
        model_meta['transport'] = 'Official XML Init with absolute on-host mesh directory'
        model_meta['prepared_xml_sha256'] = sha(prepared_xml)
        model_path = prepared_xml
        base_name, foot_names = 'trunk', LEGS
    else:
        from s1_go2_r11_model import build_project_model
        model_path, model_meta = build_project_model(upstream, Path(args.original_robot_dir),
            Path(args.original_scene), Path(args.original_resolved), out/'model')
        model = mujoco.MjModel.from_xml_path(str(model_path))
        base_name = model_meta['base_body_name']
        foot_names = model_meta['foot_geom_names']
    if 'startup' in c:
        from s1_go2_r11_model import write_flat_ground_height
        model_meta['flat_ground'] = write_flat_ground_height(model, model_path)
        if args.stage == 'G0' and not np.isclose(model_meta['flat_ground']['world_height_m'], c['startup']['G0_ground_height_m']):
            raise ValueError('G0 actual plane height differs from the fixed protocol')
        model = mujoco.MjModel.from_xml_path(str(model_path))
        model_meta['prepared_xml_sha256'] = sha(model_path)
    if campaign:
        from s1_go2_r11_model import write_equilibrium_settings
        model_meta['controller_settings'] = write_equilibrium_settings(model_path, c, args.arm)
        model = mujoco.MjModel.from_xml_path(str(model_path))
        model_meta['prepared_xml_sha256'] = sha(model_path)
    prediction_model = mujoco.MjModel.from_xml_path(str(model_path)) if joint is not None else model
    if campaign and model_meta['controller_settings']['prediction_smoothing_width_m'] is not None:
        from s1_go2_r11_model import build_prediction_model
        model_meta['prediction_only_model'] = build_prediction_model(model_path, out/'prediction-model.xml',
            model_meta['controller_settings']['prediction_smoothing_width_m'])
        prediction_model = mujoco.MjModel.from_xml_path(str(out/'prediction-model.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    initial = next(x for x in c['single_mission_protocol']['formal_conditions'] if x['id']==args.condition) if campaign else None
    yaw = (initial['initial_yaw_offset_deg'] if campaign else {'nominal': 0., 'yaw_plus5': 5., 'yaw_minus5': -5.}[args.condition])*np.pi/180
    if campaign:
        data.qvel[:3] = initial['initial_base_linear_velocity_m_s']
        data.qvel[3:6] = initial['initial_base_angular_velocity_rad_s']
    data.qpos[3:7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
    mujoco.mj_forward(model, data)
    feet = np.array([model.geom(n).id for n in foot_names])
    base = model.body(base_name).id
    allowed = {0} | {i for i in range(1, model.nbody) if i != base and not model.body_mocapid[i] >= 0}
    # The original environment allows leg-body/world contact; trunk/world fails.
    if args.stage == 'G1':
        allowed = {0} | {model.body(n).id for n in model_meta['allowed_ground_body_names']}
    joint_ids = model.actuator_trnid[:, 0]
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    offset = model.site('head').pos.copy()
    goal = model.body_mocapid[model.body('goal').id]
    dt, physical_dt = .002, float(model.opt.timestep)
    substeps = round(dt/physical_dt)
    if not np.isclose(substeps*physical_dt, dt):
        raise ValueError('Native physics timestep does not divide the specified 500Hz feedback interval')
    source = {p: sha(ROOT/p) for p in ['scripts/s1_go2_r11_runtime.py', 'scripts/s1_go2_r11_model.py', 'scripts/s1_go2_wb_equilibrium.py', *(['scripts/s1_go2_support_reference.py'] if args.arm=='S' else []), args.config, *[v for k,v in c['upstream'].items() if 'patch' in k and isinstance(v,str) and v.endswith('.patch')]] if (ROOT/p).exists()}
    resolved = dict(stage=args.stage, condition=args.condition, source_commit=git(ROOT), source_sha256=source,
        author_commit=(upstream/'SOURCE_COMMIT').read_text().strip(), author_source=str(upstream), model=model_meta,
        model_path=str(model_path), mass_kg=float(model.body_subtreemass[base]),
        compiled_mass_including_mocap_kg=float(model.body_mass.sum()), physics_dt_s=physical_dt,
        logical_feedback_dt_s=dt, logical_planning_dt_s=.02, workers=args.workers,
        synchronous_slow_simulation=True, initial_qpos=data.qpos.copy(), initial_qvel=data.qvel.copy(),
        execution_cpus=sorted(os.sched_getaffinity(0)), process_nice=os.getpriority(os.PRIO_PROCESS,0),
        joint_names=[model.joint(int(j)).name for j in joint_ids], joint_ranges=model.jnt_range[joint_ids],
        actuator_gear=model.actuator_gear, actuator_gainprm=model.actuator_gainprm,
        actuator_biasprm=model.actuator_biasprm, actuator_gaintype=model.actuator_gaintype,
        actuator_biastype=model.actuator_biastype, actuator_ctrlrange=model.actuator_ctrlrange,
        actuator_ctrllimited=model.actuator_ctrllimited, actuator_forcerange=model.actuator_forcerange,
        actuator_forcelimited=model.actuator_forcelimited, head_offset=offset,
        model_numerics={model.numeric(i).name: model.numeric(i).data.copy() for i in range(model.nnumeric)},
        author_planner_defaults=dict(ilqg_num_rollouts=10, derivative_skip=0, fd_tolerance=1e-6,
            fd_mode=0, ilqg_representation=1, action_limits=1, nominal_feedback_scaling=1,
            max_regularization_iterations=5, planner_step_iterations=1),
        author_prediction_approximation='Agent MakeDifferentiable temporarily sets solimp[0]=0 for gradient-planner rollouts; original plant contact settings stay unchanged. Prediction dt=.01s.',
        mujoco_version=mujoco.__version__, python=sys.version, command=sys.argv,
        agent_server=dict(path=str(Path(args.agent_server).resolve()),sha256=sha(args.agent_server),size=Path(args.agent_server).stat().st_size),
        host=subprocess.check_output(['hostname'], text=True).strip())
    if joint is not None:
        joint.attach(model,data,feet,allowed,out,prediction_model)
        resolved['r9_material_contract']=joint.contract
        resolved['r9_material_source_sha256']=sha(ROOT/'scripts/s1_go2_joint_material.py')
    if c.get('stage') == 'S1R11':
        resolved['r11_configuration'] = c
        resolved['full_configuration_sha256']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        resolved['r11_source_sha256'] = {name:sha(ROOT/'scripts'/name) for name in
            ('run_s1_go2_r11.py', 's1_go2_r11_protocol.py')}
    def actual_contact(full=False):
        native=contact_state(model,data,feet,allowed,full=True)
        return joint.contacts(native,full) if joint is not None else (native if full else native[:3])
    mujoco.mj_saveModel(model, str(out/'model.mjb'), None)
    resolved['model_mjb_sha256'] = sha(out/'model.mjb')
    fingerprint = hashlib.sha256(json.dumps([source, resolved['model_mjb_sha256'],resolved['agent_server']['sha256'],args.workers], sort_keys=True).encode()).hexdigest()
    resolved['execution_fingerprint'] = fingerprint
    if campaign:
        resolved.update(arm=args.arm, mission=args.mission, cohort=args.cohort,
            prediction_settings=model_meta['controller_settings'],
            author_prediction_approximation=model_meta['controller_settings'],
            author_source_sha256={name:sha(upstream/name) for name in ['mjpc/tasks/quadruped/quadruped.cc',
                'mjpc/tasks/quadruped/task_flat.xml','mjpc/agent.cc','mjpc/trajectory.cc',
                'mjpc/planners/model_derivatives.cc','mjpc/planners/ilqg/planner.cc','mjpc/planners/ilqg/policy.cc','mjpc/grpc/agent_service.cc'] if (upstream/name).exists()})
    small.mkdir(parents=True, exist_ok=True)
    dump(small/f'{args.attempt}-resolved.json', resolved)
    if support and args.arm=='AUTHOR_N':
        resolved['binary_source_identity']='Frozen R7 executable, SHA256 recorded above; current upstream C files are not compiled into this retained executable.'
        resolved['frozen_binary_source_sha256']={name:sha(ROOT/'work/r8-source-before'/name) for name in
            ['mjpc/agent.cc','mjpc/trajectory.cc','mjpc/planners/model_derivatives.cc']}
    mission, hist, physics, plans = Mission(c['task']), [], [], []
    chunks=[];physical_count=0;plan_count=0;peak_torque=np.zeros(12);prediction_grid=None
    def save_streams():
        nonlocal physical_count,plan_count,peak_torque,prediction_grid
        if physics:
            values=np.asarray(physics);physical_count+=len(values)
            peak_torque=np.maximum(peak_torque,np.max(np.abs(values[:,2:14]),axis=0))
            path=out/('physical-%03d.npz'%sum(x['stream']=='physical' for x in chunks))
            np.savez_compressed(path,physical=values)
            chunks.append(dict(stream='physical',path=str(path),start_s=float(values[0,0]),
                end_s=float(values[-1,1]),samples=len(values),size=path.stat().st_size,sha256=sha(path)))
            physics.clear()
        if plans:
            times=np.asarray([x[1]['times'] for x in plans]);plan_count+=len(plans)
            path=out/('prediction-%03d.npz'%sum(x['stream']=='prediction' for x in chunks))
            np.savez_compressed(path,plan_origin=np.asarray([x[0] for x in plans]),
                plan_states=np.asarray([x[1]['states'] for x in plans]),
                plan_actions=np.asarray([x[1]['actions'] for x in plans]),plan_times=times)
            chunks.append(dict(stream='prediction',path=str(path),start_s=float(plans[0][0]),
                end_s=float(plans[-1][0]),samples=len(plans),size=path.stat().st_size,sha256=sha(path)))
            prediction_grid=dict(nodes=int(times.shape[1]),transitions=int(times.shape[1]-1),
                span_min_s=float(np.min(times[:,-1]-times[:,0])),span_max_s=float(np.max(times[:,-1]-times[:,0])),
                timestep_s=float(np.median(np.diff(times,axis=1))))
            plans.clear()
    if campaign and args.mission == 'long':
        from s1_go2_wb_equilibrium import ContinuousMission
        mission = ContinuousMission(c['task'], c['long_mission_protocol'])
    duration = c['r11_stage']['duration_s']
    first_completion = None
    planning_times, feedback_times, sync_times, physics_times = [], [], [], []
    startup_plans, startup_actions, startup_times = [], [], []
    startup = dict(home_ctrl=data.ctrl.copy(), state_time_s=float(data.time),
        initial_qpos=data.qpos.copy(), initial_qvel=data.qvel.copy(), completed_iterations=0,
        plant_steps_during_initialization=0, reset_calls_during_initialization=0)
    resolved['startup'] = startup
    reason, error, started = '', None, False
    begin = time.perf_counter()
    phase_settings = None
    previous_phase = None
    with (out/'agent-server.log').open('w') as server_log, (contextlib.nullcontext(joint.agent) if training else Agent(task_id='Quadruped Flat', model=prediction_model,
            server_binary_path=args.agent_server, extra_flags=[f'--mjpc_workers={args.workers}'],
            subprocess_kwargs=dict(stdout=server_log, stderr=subprocess.STDOUT), run_init=False)) as agent:
        try:
            agent.init(task_id='Quadruped Flat', model=prediction_model, send_as='xml')
            agent.set_mode('Quadruped')
            agent.set_task_parameters({'Gait switch': 'Manual'})
            for step in range(round(duration/dt)+1):
                t = float(data.time)
                mujoco.mj_forward(model, data)
                ori = Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler('xyz')
                velocity, omega = data.qvel[:3].copy(), data.qvel[3:6].copy()
                if joint is not None and not policy_hook:
                    joint.control(t,mission)
                contact, grf, _ = actual_contact()
                body_goal, heading = mission.update(t, dt, data.qpos[:3], velocity, ori, omega, bool(contact.all()))
                if c.get('r11_body_reference') and np.any(joint.soft) and mission.phase in ('outbound','return'):
                    direction=1. if mission.phase=='outbound' else -1.
                    lead=direction*(mission.body_target[0]-data.qpos[0])
                    limit=c['r11_body_reference']['maximum_longitudinal_lead_m']
                    if lead>limit:
                        mission.body_target[0]=data.qpos[0]+direction*limit
                        body_goal[0]=mission.body_target[0]
                if policy_hook:
                    joint.control(t,mission)
                    if joint.stop_requested:
                        reason=joint.stop_requested
                        break
                if joint is not None and started and c.get('stage') == 'S1R11' and time.time() >= joint.deadline:
                    reason = 'development_budget_truncated' if args.purpose=='development' else 'campaign_budget_truncated'
                    break
                mission_changed = mission.phase != previous_phase
                previous_phase = mission.phase
                if mission.phase == 'complete' and first_completion is None:
                    first_completion = t
                standing = mission.phase in ('initial_stand', 'terminal_stand', 'complete', 'far_hold', 'home_hold')
                settings = 'stand_settings' if standing else 'move_and_turn_settings'
                ts = time.perf_counter()
                settings_changed = settings != phase_settings
                if settings_changed:
                    table = c['task'][settings]
                    # Native gait selection applies its own numeric defaults on Transition.
                    # Select first; apply the declared numeric values after that transition.
                    agent.set_task_parameter('Gait',table['Gait'])
                    phase_settings = settings
                if support and args.arm=='S' and standing and settings_changed:
                    from s1_go2_support_reference import stand_reference,reference_userdata
                    tr=time.perf_counter()
                    ref_model,ref_data=joint.native_reference_state() if joint is not None else (model,data)
                    reference=stand_reference(ref_model,ref_data,heading,c['support_reference']['H_realized']['absolute_base_height_m'])
                    reference['wall_s']=time.perf_counter()-tr
                    resolved.setdefault('stand_references',[]).append(reference)
                    dump(small/f'{args.attempt}-resolved.json',resolved)
                    if not reference['feasible']:
                        raise ValueError('Native static support reference infeasible; see retained reference and entry state')
                    data.userdata[:62]=reference_userdata(reference)
                if c.get('local_contact_model',{}).get('causal_force_anchor'):
                    bottom=joint.force_boundary_bottom
                    patches=data.userdata[62:78].reshape(4,4)
                    patches[:,0]=(joint.force[:,2]>1)&joint.soft
                    patches[:,1:3]=bottom[:,:2]
                    patches[:,3]=bottom[:,2]+joint.force[:,2]/c['local_contact_model']['instantaneous_stiffness_N_m']
                agent.set_task_parameter('Heading', float(heading))
                rotation = Rotation.from_euler('z', heading).as_matrix()
                data.mocap_pos[goal] = body_goal+rotation@offset
                data.mocap_quat[goal] = [np.cos(heading/2), 0, 0, np.sin(heading/2)]
                agent.set_state(time=t, qpos=data.qpos, qvel=data.qvel, act=data.act,
                    mocap_pos=data.mocap_pos, mocap_quat=data.mocap_quat, userdata=data.userdata)
                if settings_changed:
                    agent.set_task_parameters({k:v for k,v in table.items()
                        if k not in ('Gait','Balance','Upright','Height')})
                    agent.set_cost_weights({k:table[k] for k in ('Balance','Upright','Height')} |
                        {'Orientation':c['task']['orientation_cost_weight']})
                    if campaign:
                        agent.set_cost_weights(c['stationary_objective'].get('cost_weights',{}) |
                            {'Angmom':0.} | model_meta['controller_settings']['cost_weights'])
                    # The same-time transition activates the requested cadence without
                    # advancing the plant or re-entering the gait-default branch.
                    agent.set_state(time=t,qpos=data.qpos,qvel=data.qvel,act=data.act,
                        mocap_pos=data.mocap_pos,mocap_quat=data.mocap_quat,userdata=data.userdata)
                    actual_parameters=agent.get_task_parameters()
                    resolved.setdefault('applied_settings', []).append(dict(time_s=t,
                        declared_parameters={k:table[k] for k in ('Gait','Cadence','Duty ratio','Amplitude')},
                        task_parameters=actual_parameters,cost_weights=agent.get_cost_weights()))
                    dump(small/f'{args.attempt}-resolved.json',resolved)
                    if any(actual_parameters[k]!=table[k] for k in ('Gait','Cadence','Duty ratio','Amplitude')):
                        raise RuntimeError('Declared gait settings were overwritten during native transition; see actual readback')
                sync_s = time.perf_counter()-ts
                sync_times.append(sync_s)
                plan_s = 0.
                if step == 0 and 'startup' in c:
                    startup['pre_optimization_nominal_action'] = agent.get_action(time=t, nominal_action=True, averaging_duration=0)
                    startup['body_target'] = body_goal.copy()
                    startup['heading_target'] = float(heading)
                    startup['mocap_pos'] = data.mocap_pos.copy()
                    startup['mocap_quat'] = data.mocap_quat.copy()
                    dump(small/f'{args.attempt}-resolved.json', resolved)
                    for iteration in range(1, c['startup']['planner_steps']+1):
                        tp = time.perf_counter()
                        agent.planner_step()
                        elapsed = time.perf_counter()-tp
                        startup_times.append(elapsed)
                        plan = agent.best_trajectory()
                        startup_plans.append(plan)
                        target = agent.get_action(time=t, nominal_action=False, averaging_duration=0)
                        startup_actions.append(target)
                        startup['completed_iterations'] = iteration
                        np.savez_compressed(out/'startup.npz',
                            iteration=np.arange(1, iteration+1), time_s=np.zeros(iteration),
                            home_ctrl=startup['home_ctrl'],
                            seed_action=startup['pre_optimization_nominal_action'],
                            states=np.asarray([p['states'] for p in startup_plans]),
                            actions=np.asarray([p['actions'] for p in startup_plans]),
                            times=np.asarray([p['times'] for p in startup_plans]),
                            feedback_target=np.asarray(startup_actions), planner_s=np.asarray(startup_times))
                        print(json.dumps(dict(attempt=args.attempt, startup_iteration=iteration,
                            plant_time_s=t, planner_s=elapsed)), flush=True)
                    plan_s = sum(startup_times)
                    startup['planning_total_s'] = plan_s
                    if not compact:plans.append([t, plan])
                    if policy_hook:joint.latest_plan=plan
                elif step % 10 == 0 or (support and args.stage=='G1' and (settings_changed or mission_changed)):
                    tp = time.perf_counter()
                    agent.planner_step()
                    plan_s = time.perf_counter()-tp
                    planning_times.append(plan_s)
                    plan = agent.best_trajectory()
                    if not compact:plans.append([t, plan])
                    if policy_hook:joint.latest_plan=plan
                if compact and plan_s:
                    plan_count+=1
                    prediction_grid=dict(nodes=len(plan['times']),transitions=len(plan['times'])-1,
                        span_min_s=float(plan['times'][-1]-plan['times'][0]),
                        span_max_s=float(plan['times'][-1]-plan['times'][0]),timestep_s=.002)
                tf = time.perf_counter()
                action = agent.get_action(time=t, nominal_action=False, averaging_duration=0)
                nominal_action = agent.get_action(time=t, nominal_action=True, averaging_duration=0) if support else None
                feedback_s = time.perf_counter()-tf
                feedback_times.append(feedback_s)
                if step == 0:
                    startup['actual_first_feedback_action'] = action.copy()
                    dump(small/f'{args.attempt}-resolved.json', resolved)
                data.ctrl[:] = action
                mujoco.mj_forward(model, data)
                contact, grf, _, total_grf, nonfoot_grf, nonfoot_count = actual_contact(full=True)
                if compact and step>0 and (step-1)%25:hist.pop()
                hist.append(dict(time=t, diagnostic_time=t, qpos=data.qpos.copy(), qvel=data.qvel.copy(), rpy=ori,
                    q_target=action.copy(), torque=data.qfrc_actuator[vadr].copy(), actuator_force=data.actuator_force.copy(),
                    contact=contact, actual_grf=grf, total_ground_force=total_grf, nonfoot_ground_force=nonfoot_grf, nonfoot_ground_contacts=nonfoot_count, feet=data.geom_xpos[feet].copy(), body_target=body_goal,
                    heading_target=heading, mocap_pos=data.mocap_pos.copy(), mocap_quat=data.mocap_quat.copy(),
                    command_velocity=mission.v.copy(), command_omega=mission.w.copy(), phase=PHASES.index(mission.phase),
                    planner_s=plan_s, feedback_s=feedback_s, sync_s=sync_s))
                if joint is not None:
                    hist[-1].update(requested_speed_cap=float(joint.cap),cap_effective=mission.phase in ('outbound','return'))
                if support:
                    feet_velocity=[]
                    for foot in feet:
                        spatial=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_GEOM,int(foot),spatial,0)
                        feet_velocity.append(spatial[3:])
                    hist[-1].update(nominal_q_target=nominal_action,
                        unclipped_pd_request=60*(action-data.qpos[qadr])-5*data.qvel[vadr],
                        base_absolute_height=float(data.xpos[base,2]),trunk_com_absolute_height=float(data.xipos[base,2]),
                        relative_height_residual=float(data.xipos[base,2]-data.geom_xpos[feet,2].mean()-.27),
                        feet_velocity=np.asarray(feet_velocity),phase_replan=bool(mission_changed and step>0 and args.stage=='G1'))
                    if args.arm=='S':
                        hist[-1].update(support_reference_userdata=data.userdata.copy(),support_active=standing,
                            active_relative_height_target=float(data.userdata[1]) if standing else .27)
                    if step>0 and step%15000==0:save_streams()
                if ((not campaign or (training and args.mission=='single')) and mission.phase == 'complete') or step >= round(duration/dt):
                    reason = 'stage_horizon_reached'
                    break
                tp = time.perf_counter()
                for _ in range(substeps):
                    sample_t = float(data.time)
                    raw_pd=60*(data.ctrl-data.qpos[qadr])-5*data.qvel[vadr]
                    if joint is not None and not joint.before_physics(physical_dt,mission.phase):
                        reason=joint.reason
                        break
                    if not started:first_integration_unix_s=time.time()
                    mujoco.mj_step(model, data)
                    if not started:
                        with index_file(small, c) as idx:
                            if 'campaign_started_unix_s' not in idx:
                                idx.update(campaign_started_unix_s=first_integration_unix_s,
                                    campaign_deadline_unix_s=first_integration_unix_s+48*3600,
                                    campaign_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(first_integration_unix_s)))
                            if campaign:
                                row=next(x for x in idx['attempts'] if x['attempt']==args.attempt)
                                row.update(status='running', physical_started=True,
                                    execution_fingerprint=fingerprint, source_commit=resolved['source_commit'])
                            else:
                                if len(idx['attempts']) >= 8:
                                    raise ValueError('Historical eight-start allowance exhausted')
                                idx['attempts'].append(dict(attempt=args.attempt, stage=args.stage, purpose=args.purpose,
                                    condition=args.condition, status='running', execution_fingerprint=fingerprint,
                                    command=sys.argv, source_commit=resolved['source_commit']))
                        started = True
                        if joint is not None:
                            joint.started(small,c,index_file)
                            if args.purpose=='development':
                                reserve=idx.get('evaluation_reserve',{}).get('wall_s',6*3600)
                                joint.deadline=min(joint.deadline,idx['campaign_started_unix_s']+24*3600,
                                    idx['campaign_deadline_unix_s']-reserve)
                    actual_contacts, actual_grf, invalid, total_grf, nonfoot_grf, nonfoot_count = actual_contact(full=True)
                    if joint is not None:
                        joint.after_physics(physical_dt,actual_grf)
                    if compact:
                        physical_count+=1
                        peak_torque=np.maximum(peak_torque,np.abs(data.qfrc_actuator[vadr]))
                    else:
                        physics.append(np.r_[sample_t, data.time, data.qfrc_actuator[vadr], data.actuator_force,
                            actual_contacts, actual_grf.ravel(), total_grf, nonfoot_grf, nonfoot_count])
                        if support:physics[-1]=np.r_[physics[-1],raw_pd]
                    if invalid:
                        reason = 'invalid_contact:'+str(invalid)
                        break
                physics_times.append(time.perf_counter()-tp)
                now_ori = Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler('xyz')
                if not reason and (max(abs(now_ori[:2])) > 1.2 or data.qpos[2] < .12):
                    reason = 'body_fall'
                if reason:
                    # Include the actual terminal pose for end-to-end replay.
                    row = dict(hist[-1]); row.update(time=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(), rpy=now_ori)
                    hist.append(row)
                    break
                if step % 2500 == 0:
                    print(json.dumps(dict(attempt=args.attempt, time_s=t, phase=mission.phase,
                        xyz=data.qpos[:3].tolist(), wall_s=time.perf_counter()-begin)), flush=True)
        except Exception:
            error = traceback.format_exc()
            reason = 'execution_error'
            print(error, flush=True)
        finally:
            dump(small/f'{args.attempt}-resolved.json', resolved)
    wall = time.perf_counter()-begin
    if not started:
        with index_file(small, c) as idx:
            idx.setdefault('initialization_failures', []).append(dict(attempt=args.attempt,
                stage=args.stage, condition=args.condition, physical_steps=0, error=error,
                completed_startup_iterations=len(startup_plans), command=sys.argv,
                source_commit=resolved['source_commit'], execution_fingerprint=fingerprint))
        raise RuntimeError('No plant start: '+str(error))
    arrays = {k: np.asarray([r[k] for r in hist]) for k in hist[0]}
    if support:
        save_streams()
        np.savez_compressed(out/'trajectory.npz',**arrays)
    else:
        np.savez_compressed(out/'trajectory.npz', **arrays, physical=np.asarray(physics),
            plan_origin=np.asarray([x[0] for x in plans]), plan_states=np.asarray([x[1]['states'] for x in plans]),
            plan_actions=np.asarray([x[1]['actions'] for x in plans]), plan_times=np.asarray([x[1]['times'] for x in plans]))
    result = dict(attempt=args.attempt, stage=args.stage, purpose=args.purpose, condition=args.condition,
        status='completed', success=(first_completion is not None if args.mission=='single' else
            first_completion is not None and first_completion <= c['long_mission_protocol']['route_completion_deadline_s'] and
            reason=='stage_horizon_reached' and mission.final_stable_s >= c['task']['terminal_stand_s']), reason=reason, error=error,
        original_mission_success=(first_completion is not None) if args.mission=='single' else None,
        first_completion_s=first_completion, mission=args.mission, arm=args.arm, cohort=args.cohort,
        arrivals=getattr(mission,'arrivals',[]), completed_round_trips=getattr(mission,'laps',int(first_completion is not None)),
        final_stable_s=getattr(mission,'final_stable_s',None),
        simulation_s=float(data.time), wall_s=wall, final_qpos=data.qpos.copy(), final_qvel=data.qvel.copy(),
        final_origin_distance_m=float(np.linalg.norm(data.qpos[:2])),
        final_heading_error_rad=float(abs(angle(np.pi-Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler('xyz')[2]))),
        max_x_m=float(arrays['qpos'][:,0].max()), min_body_height_m=float(arrays['qpos'][:,2].min()),
        max_abs_torque_Nm=peak_torque if support else np.max(np.abs(np.asarray(physics)[:,2:14]),axis=0),
        max_joint_limit_excess_rad=float(np.maximum(np.maximum(model.jnt_range[joint_ids,0]-arrays['qpos'][:,qadr],
            arrays['qpos'][:,qadr]-model.jnt_range[joint_ids,1]),0).max()),
        transitions=mission.transitions, execution_fingerprint=fingerprint, source_commit=resolved['source_commit'],
        startup=dict(iterations=len(startup_plans), planning_total_s=sum(startup_times),
            record=f'{args.attempt}-resolved.json', archive=str(out/'startup.npz')),
        timing=dict(startup_planning=timings(startup_times,.02), planning=timings(planning_times,.02), feedback=timings(feedback_times,.002),
            state_parameter_rpc=timings(sync_times,.002), physics=timings(physics_times,.002)),
        samples_500Hz=len(hist), physical_samples=physical_count if support else len(physics),
        trajectory=dict(path=str(out/'trajectory.npz'), size=(out/'trajectory.npz').stat().st_size, sha256=sha(out/'trajectory.npz')),
        physical_columns=['force_evaluation_time_s','post_integration_time_s']+[f'torque_{i}' for i in range(12)]+[f'actuator_force_{i}' for i in range(12)]+[f'contact_{l}' for l in LEGS]+[f'grf_{l}_{a}' for l in LEGS for a in 'xyz']+[f'total_grf_{a}' for a in 'xyz']+[f'nonfoot_grf_{a}' for a in 'xyz']+['nonfoot_ground_contacts'],
        physical_time_note='MuJoCo mj_step forces/contact correspond to the integration evaluation, preceding its returned qpos. Force evaluation and post-integration clocks are both stored.',
        terminal_row_note='Failure terminal qpos/qvel are appended; held command and last diagnostic values retain their preceding control timestamp.')
    if campaign:
        from s1_go2_wb_equilibrium import hold_metrics
        result['hold'] = hold_metrics(arrays, first_completion, duration, args.mission)
        if args.mission == 'long':
            result['success'] = bool(result['success'] and result['hold']['endpoint_position_violation_samples']==0)
            if reason=='stage_horizon_reached' and first_completion is not None and not result['success']:
                result['reason'] = ('route_deadline_missed' if first_completion > c['long_mission_protocol']['route_completion_deadline_s'] else 'left_terminal_endpoint' if result['hold']['endpoint_position_violation_samples'] else 'terminal_stability_failed')
        if support:
            result['realized_prediction']=dict(**prediction_grid,planning_origins=plan_count)
            result['chunks']=chunks
            result['physical_columns'] += [f'unclipped_pd_request_{i}' for i in range(12)]
            result['sustained_execution_success']=bool(result['success'] and reason=='stage_horizon_reached' and result['hold']['full_observation'])
            result['recording']='500Hz control trajectory; physical and prediction streams split every30s on the same uninterrupted clock.'
            if args.mission=='long':
                result['last_30s_hold']=hold_metrics(arrays,duration-30,duration,args.mission)
        else:
            actual_plan_times = np.asarray([x[1]['times'] for x in plans])
            result['realized_prediction'] = dict(nodes=int(actual_plan_times.shape[1]),
                span_min_s=float(np.diff(actual_plan_times[:,[0,-1]],axis=1).min()),
                span_max_s=float(np.diff(actual_plan_times[:,[0,-1]],axis=1).max()),
                timestep_s=float(np.median(np.diff(actual_plan_times,axis=1))),planning_origins=len(plans))
    if joint is not None:
        result=joint.finish(result)
    if compact:
        result.update(samples_500Hz=None,recorded_control_samples=len(hist),control_record_rate_Hz=20,
            recording='Training: 20Hz robot states, 2Hz transitions, per-episode actual physical counters; no full prediction/physics dumps.')
    from s1_go2_r11_protocol import evaluate_stage
    result=evaluate_stage(result,c,joint,arrays)
    result['full_configuration_sha256']=resolved['full_configuration_sha256']
    dump(small/f'{args.attempt}.json',result)
    with index_file(small,c) as idx:
        row=next(x for x in idx['attempts'] if x['attempt']==args.attempt)
        row.update({k:result[k] for k in ('status','success','reason','simulation_s','wall_s','evaluation_stage','stage_status','case_key','case_identity','qualification')})
        row['result']=f'{args.attempt}.json'
    print(json.dumps({k:result[k] for k in ('attempt','stage_status','success','reason','simulation_s','wall_s','case_key')},default=str),flush=True)
    return result


def render(args,c):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    out=ROOT/'outputs/terrain_memory/simulation'/c['run_id']/args.attempt
    with np.load(out/'trajectory.npz') as archive:
        a={key:archive[key] for key in ('time','qpos','qvel','mocap_pos','mocap_quat','phase')}
    result=json.loads((ROOT/c['outputs']['root']/f'{args.attempt}.json').read_text())
    model=mujoco.MjModel.from_binary_path(str(out/'model.mjb'))
    data=mujoco.MjData(model)
    model.vis.global_.offwidth,model.vis.global_.offheight=1280,720
    renderer=mujoco.Renderer(model,height=720,width=1280)
    camera=mujoco.MjvCamera(); camera.azimuth=125.; camera.elevation=-23.; camera.distance=4.8
    fps=30
    sample=np.unique(np.r_[np.searchsorted(a['time'],np.arange(0,a['time'][-1],1/fps)),len(a['time'])-1])
    with imageio.get_writer(out/'complete-mission.mp4',fps=fps,codec='libx264',quality=8) as writer:
        for i in sample:
            data.time=float(a['time'][i]); data.qpos[:]=a['qpos'][i]; data.qvel[:]=a['qvel'][i]
            data.mocap_pos[:]=a['mocap_pos'][i]; data.mocap_quat[:]=a['mocap_quat'][i]
            mujoco.mj_forward(model,data)
            camera.lookat[:]=[data.qpos[0],data.qpos[1],.25]
            renderer.update_scene(data,camera=camera)
            frame=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,1280,74),fill=(15,20,30))
            draw.text((18,10),f"{args.attempt} | {result['stage']} | {result['condition']} | t={data.time:.3f}s | {PHASES[int(a['phase'][i])]}",fill='white')
            draw.text((18,34),f"Body x={data.qpos[0]:.3f} m, y={data.qpos[1]:.3f} m | full recorded mission | synchronous simulation, not NX realtime",fill='white')
            if i==sample[-1]: draw.text((18,55),result['reason'],fill='yellow')
            writer.append_data(np.asarray(frame))
        if result['simulation_s'] > float(a['time'][-1])+1e-10:
            # A planner exception can occur between the last logged command
            # and its following state. The saved result contains that endpoint.
            data.time=result['simulation_s']; data.qpos[:]=result['final_qpos']; data.qvel[:]=result['final_qvel']
            mujoco.mj_forward(model,data)
            camera.lookat[:]=[data.qpos[0],data.qpos[1],.25]
            renderer.update_scene(data,camera=camera)
            frame=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,1280,74),fill=(15,20,30))
            draw.text((18,10),f"{args.attempt} | actual terminal state | t={data.time:.3f}s",fill='white')
            draw.text((18,34),result['reason'],fill='yellow')
            writer.append_data(np.asarray(frame))
    renderer.close()
    print(json.dumps(dict(video=str(out/'complete-mission.mp4'),size=(out/'complete-mission.mp4').stat().st_size,sha256=sha(out/'complete-mission.mp4'))))

