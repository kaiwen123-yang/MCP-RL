#!/usr/bin/env python3
"""Complete S1 missions through the pinned author's headless Agent.

No old locomotion controller is imported. Physics is advanced only in run();
render() replays recorded states without integrating another trajectory.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'experiments/terrain_memory/s1-go2-wb-baseline.json'
PHASES = ['initial_stand', 'outbound', 'turn', 'return', 'terminal_stand', 'complete']
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


def contact_state(model, data, feet, allowed_bodies):
    contact, force = np.zeros(4, dtype=bool), np.zeros((4, 3))
    invalid = []
    for k in range(data.ncon):
        con = data.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if b1 == 0 and b2 not in allowed_bodies:
            invalid.append([g1, g2, b1, b2])
        if b2 == 0 and b1 not in allowed_bodies:
            invalid.append([g1, g2, b1, b2])
        for i, foot in enumerate(feet):
            if (foot == g1 and b2 == 0) or (foot == g2 and b1 == 0):
                contact[i] = True
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, k, wrench)
                force[i] += con.frame.reshape(3, 3).T @ wrench[:3] * (1 if foot == g2 else -1)
    return contact, force, invalid


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
            budget=config['budget'], historical_reference=config['protocol']['historical_reference'],
            training_runs=0, hardware_runs=0, nx_realtime_measured=False)
        yield idx
        dump(path, idx)


def run(args, c):
    from mujoco_mpc.agent import Agent
    small = ROOT/c['outputs']['root']
    out = ROOT/'outputs/terrain_memory/simulation'/c['run_id']/args.attempt
    out.mkdir(parents=True, exist_ok=False)
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
        base_name, foot_names = 'trunk', LEGS
    else:
        from s1_go2_wb_model import build_project_model
        model_path, model_meta = build_project_model(upstream, Path(args.original_robot_dir),
            Path(args.original_scene), Path(args.original_resolved), out/'model')
        model = mujoco.MjModel.from_xml_path(str(model_path))
        base_name = model_meta['base_body_name']
        foot_names = model_meta['foot_geom_names']
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    yaw = {'nominal': 0., 'yaw_plus5': 5., 'yaw_minus5': -5.}[args.condition]*np.pi/180
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
    source = {p: sha(ROOT/p) for p in ['scripts/run_s1_go2_wb_baseline.py', 'scripts/s1_go2_wb_model.py', CONFIG] if (ROOT/p).exists()}
    resolved = dict(stage=args.stage, condition=args.condition, source_commit=git(ROOT), source_sha256=source,
        author_commit=(upstream/'SOURCE_COMMIT').read_text().strip(), author_source=str(upstream), model=model_meta,
        model_path=str(model_path), mass_kg=float(model.body_mass.sum()), physics_dt_s=physical_dt,
        logical_feedback_dt_s=dt, logical_planning_dt_s=.02, workers=args.workers,
        synchronous_slow_simulation=True, initial_qpos=data.qpos.copy(), initial_qvel=data.qvel.copy(),
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
        host=subprocess.check_output(['hostname'], text=True).strip())
    mujoco.mj_saveModel(model, str(out/'model.mjb'), None)
    resolved['model_mjb_sha256'] = sha(out/'model.mjb')
    fingerprint = hashlib.sha256(json.dumps([source, resolved['model_mjb_sha256']], sort_keys=True).encode()).hexdigest()
    resolved['execution_fingerprint'] = fingerprint
    small.mkdir(parents=True, exist_ok=True)
    dump(small/f'{args.attempt}-resolved.json', resolved)
    mission, hist, physics, plans = Mission(c['task']), [], [], []
    planning_times, feedback_times, sync_times, physics_times = [], [], [], []
    reason, error, started = '', None, False
    begin = time.perf_counter()
    phase_settings = None
    with (out/'agent-server.log').open('w') as server_log, Agent(task_id='Quadruped Flat', model=model,
            server_binary_path=args.agent_server, extra_flags=[f'--mjpc_workers={args.workers}'],
            subprocess_kwargs=dict(stdout=server_log, stderr=subprocess.STDOUT)) as agent:
        agent.set_mode('Quadruped')
        agent.set_task_parameters({'Gait switch': 'Manual'})
        try:
            for step in range(round(c['task']['maximum_episode_s']/dt)+1):
                t = float(data.time)
                mujoco.mj_forward(model, data)
                ori = Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler('xyz')
                velocity, omega = data.qvel[:3].copy(), data.qvel[3:6].copy()
                contact, grf, _ = contact_state(model, data, feet, allowed)
                body_goal, heading = mission.update(t, dt, data.qpos[:3], velocity, ori, omega, bool(contact.all()))
                standing = mission.phase in ('initial_stand', 'terminal_stand', 'complete')
                settings = 'stand_settings' if standing else 'move_and_turn_settings'
                ts = time.perf_counter()
                if settings != phase_settings:
                    table = c['task'][settings]
                    agent.set_task_parameters({k: v for k, v in table.items() if k not in ('Balance','Upright','Height')})
                    agent.set_cost_weights({k: table[k] for k in ('Balance','Upright','Height')} | {'Orientation': c['task']['orientation_cost_weight']})
                    resolved.setdefault('applied_settings', []).append(dict(time_s=t,
                        task_parameters=agent.get_task_parameters(), cost_weights=agent.get_cost_weights()))
                    phase_settings = settings
                agent.set_task_parameter('Heading', float(heading))
                rotation = Rotation.from_euler('z', heading).as_matrix()
                data.mocap_pos[goal] = body_goal+rotation@offset
                data.mocap_quat[goal] = [np.cos(heading/2), 0, 0, np.sin(heading/2)]
                agent.set_state(time=t, qpos=data.qpos, qvel=data.qvel, act=data.act,
                    mocap_pos=data.mocap_pos, mocap_quat=data.mocap_quat, userdata=data.userdata)
                sync_s = time.perf_counter()-ts
                sync_times.append(sync_s)
                plan_s = 0.
                if step % 10 == 0:
                    tp = time.perf_counter()
                    agent.planner_step()
                    plan_s = time.perf_counter()-tp
                    planning_times.append(plan_s)
                    plan = agent.best_trajectory()
                    plans.append([t, plan])
                tf = time.perf_counter()
                action = agent.get_action(time=t, nominal_action=False)
                feedback_s = time.perf_counter()-tf
                feedback_times.append(feedback_s)
                data.ctrl[:] = action
                mujoco.mj_forward(model, data)
                contact, grf, _ = contact_state(model, data, feet, allowed)
                hist.append(dict(time=t, diagnostic_time=t, qpos=data.qpos.copy(), qvel=data.qvel.copy(), rpy=ori,
                    q_target=action.copy(), torque=data.qfrc_actuator[vadr].copy(), actuator_force=data.actuator_force.copy(),
                    contact=contact, actual_grf=grf, feet=data.geom_xpos[feet].copy(), body_target=body_goal,
                    heading_target=heading, mocap_pos=data.mocap_pos.copy(), mocap_quat=data.mocap_quat.copy(),
                    command_velocity=mission.v.copy(), command_omega=mission.w.copy(), phase=PHASES.index(mission.phase),
                    planner_s=plan_s, feedback_s=feedback_s, sync_s=sync_s))
                if mission.phase == 'complete' or step >= round(c['task']['maximum_episode_s']/dt):
                    reason = 'completed' if mission.phase == 'complete' else 'task_timeout'
                    break
                if not started:
                    with index_file(small, c) as idx:
                        if any(x['attempt'] == args.attempt for x in idx['attempts']):
                            raise ValueError('Attempt already exists; retain and reuse its actual record')
                        if len(idx['attempts']) >= 8:
                            raise ValueError('Eight-start hard budget exhausted')
                        stage_starts = sum(x['stage'] == args.stage and x['purpose'] == args.purpose for x in idx['attempts'])
                        if stage_starts >= (2 if args.stage == 'G0' else 3):
                            raise ValueError('Stage allowance exhausted')
                        if args.stage == 'G1' and not any(x.get('success') and x['stage']=='G0' for x in idx['attempts']):
                            raise ValueError('G0 complete-mission gate has not passed')
                        idx['attempts'].append(dict(attempt=args.attempt, stage=args.stage, purpose=args.purpose,
                            condition=args.condition, status='running', execution_fingerprint=fingerprint,
                            command=sys.argv, source_commit=resolved['source_commit'], started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
                    started = True
                tp = time.perf_counter()
                for _ in range(substeps):
                    sample_t = float(data.time)
                    mujoco.mj_step(model, data)
                    actual_contact, actual_grf, invalid = contact_state(model, data, feet, allowed)
                    physics.append(np.r_[sample_t, data.time, data.qfrc_actuator[vadr], data.actuator_force,
                        actual_contact, actual_grf.ravel()])
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
            if error is None:
                resolved['final_task_parameters'] = agent.get_task_parameters()
                resolved['final_cost_weights'] = agent.get_cost_weights()
            dump(small/f'{args.attempt}-resolved.json', resolved)
    wall = time.perf_counter()-begin
    if not started:
        raise RuntimeError('No plant start: '+str(error))
    arrays = {k: np.asarray([r[k] for r in hist]) for k in hist[0]}
    np.savez_compressed(out/'trajectory.npz', **arrays, physical=np.asarray(physics),
        plan_origin=np.asarray([x[0] for x in plans]), plan_states=np.asarray([x[1]['states'] for x in plans]),
        plan_actions=np.asarray([x[1]['actions'] for x in plans]), plan_times=np.asarray([x[1]['times'] for x in plans]))
    result = dict(attempt=args.attempt, stage=args.stage, purpose=args.purpose, condition=args.condition,
        status='completed', success=reason=='completed', reason=reason, error=error,
        simulation_s=float(data.time), wall_s=wall, final_qpos=data.qpos.copy(), final_qvel=data.qvel.copy(),
        final_origin_distance_m=float(np.linalg.norm(data.qpos[:2])),
        final_heading_error_rad=float(abs(angle(np.pi-Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler('xyz')[2]))),
        max_x_m=float(arrays['qpos'][:,0].max()), min_body_height_m=float(arrays['qpos'][:,2].min()),
        max_abs_torque_Nm=np.max(np.abs(np.asarray(physics)[:,2:14]),axis=0),
        max_joint_limit_excess_rad=float(np.maximum(np.maximum(model.jnt_range[joint_ids,0]-arrays['qpos'][:,qadr],
            arrays['qpos'][:,qadr]-model.jnt_range[joint_ids,1]),0).max()),
        transitions=mission.transitions, execution_fingerprint=fingerprint, source_commit=git(ROOT),
        timing=dict(planning=timings(planning_times,.02), feedback=timings(feedback_times,.002),
            state_parameter_rpc=timings(sync_times,.002), physics=timings(physics_times,.002)),
        samples_500Hz=len(hist), physical_samples=len(physics),
        trajectory=dict(path=str(out/'trajectory.npz'), size=(out/'trajectory.npz').stat().st_size, sha256=sha(out/'trajectory.npz')),
        physical_columns=['force_evaluation_time_s','post_integration_time_s']+[f'torque_{i}' for i in range(12)]+[f'actuator_force_{i}' for i in range(12)]+[f'contact_{l}' for l in LEGS]+[f'grf_{l}_{a}' for l in LEGS for a in 'xyz'],
        physical_time_note='MuJoCo mj_step forces/contact correspond to the integration evaluation, preceding its returned qpos. Force evaluation and post-integration clocks are both stored.',
        terminal_row_note='Failure terminal qpos/qvel are appended; held command and last diagnostic values retain their preceding control timestamp.')
    dump(small/f'{args.attempt}.json',result)
    with index_file(small,c) as idx:
        row=next(x for x in idx['attempts'] if x['attempt']==args.attempt)
        row.update({k:result[k] for k in ('status','success','reason','simulation_s','wall_s')})
        row['result']=f'{args.attempt}.json'
    print(json.dumps(result,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)


def render(args,c):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    out=ROOT/'outputs/terrain_memory/simulation'/c['run_id']/args.attempt
    a=np.load(out/'trajectory.npz')
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
    renderer.close()
    print(json.dumps(dict(video=str(out/'complete-mission.mp4'),size=(out/'complete-mission.mp4').stat().st_size,sha256=sha(out/'complete-mission.mp4'))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=CONFIG)
    p.add_argument('--stage',choices=['G0','G1'],default='G0')
    p.add_argument('--purpose',choices=['development','formal'],default='development')
    p.add_argument('--condition',choices=['nominal','yaw_plus5','yaw_minus5'],default='nominal')
    p.add_argument('--attempt',required=True)
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--author-source',default=str(ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'))
    p.add_argument('--agent-server',required=True)
    p.add_argument('--original-robot-dir')
    p.add_argument('--original-scene')
    p.add_argument('--original-resolved')
    p.add_argument('--render',action='store_true')
    args=p.parse_args(); c=json.loads((ROOT/args.config).read_text())
    (render if args.render else run)(args,c)


if __name__=='__main__':
    main()
