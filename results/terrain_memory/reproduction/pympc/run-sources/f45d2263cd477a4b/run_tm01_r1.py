#!/usr/bin/env python3
"""TM01 full MuJoCo experiments; all outcomes and 500 Hz states are retained."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--experiment", choices=["fig3", "fig4", "table1"], required=True)
PARSER.add_argument("--method", choices=["acados", "naive", "mppi", "adaptive_fixed", "adaptive"], required=True)
PARSER.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
PARSER.add_argument("--run", default="r1-foothold-restored")
PARSER.add_argument("--config", default=str(ROOT / "experiments/tm01/r1-foothold-restored.json"))
PARSER.add_argument("--seed-start", type=int)
PARSER.add_argument("--seed-stop", type=int)
PARSER.add_argument("--timing-context", choices=["isolated", "shared-throughput"], default="isolated",
    help="Shared-device trajectories are valid physical experiments, but not latency comparisons.")
ARGS = PARSER.parse_args()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cpu" if ARGS.device == "cpu" or ARGS.method == "acados" else "cuda"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.robot_cfgs import get_robot_config
from gym_quadruped.utils.quadruped_utils import LegsAttr
from quadruped_pympc import config as cfg

C = json.loads(Path(ARGS.config).read_text())
E = C[ARGS.experiment]
if "model_mjcf" in C:
    os.environ["TM01_ROBOT_XML"] = str(ROOT/C["model_mjcf"])
LEGS = ("FL", "FR", "RL", "RR")
cfg.robot = C["robot"]
rc = get_robot_config(robot_name=cfg.robot)
cfg.robot_cfg = rc
cfg.robot_leg_joints, cfg.robot_feet_geom_names = rc.leg_joints, rc.feet_geom_names
cfg.qpos0_js, cfg.hip_height = rc.qpos0_js, rc.hip_height
# Old configurations retain their actual .083 m / body-frame execution.
cfg.hip_offset_x, cfg.hip_offset_y = 0.0, C["locomotion"].get("hip_offset_y", 0.083)
adaptive_family = ARGS.method in ("adaptive_fixed", "adaptive")
candidate_hz = ([C["locomotion"]["frequency_hz"]] if ARGS.method == "adaptive_fixed"
                else C["locomotion"]["candidate_hz"])
cfg.mpc_params.update(
    type="nominal" if ARGS.method == "acados" else "sampling",
    sampling_method="mppi" if ARGS.method == "mppi" else "random_sampling",
    device=ARGS.device, horizon=C["mpc"]["horizon"], dt=C["mpc"]["dt"],
    num_parallel_computations=C["mpc"]["rollouts"],
    num_sampling_iterations=C["mpc"]["iterations"],
    temperature_mppi=C["mpc"]["lambda_mppi"], sigma_mppi=C["mpc"]["sigma_mppi"],
    sigma_random_sampling=C["mpc"]["sigma_naive"], paper_Q=C["mpc"]["Q_sampling"], paper_input_cost=C["mpc"]["include_input_cost"],
    control_parametrization="cubic_spline", num_splines=C["mpc"]["num_cubic_splines"],
    optimize_step_freq=adaptive_family, step_freq_available=candidate_hz,
    use_inertia_recomputation=False, gradient_refinement_steps=0, use_riccati_feedback=False,
    use_foothold_optimization=False, use_foothold_constraints=False, use_integrators=False,
    external_wrenches_compensation=False, use_RTI=False, num_qp_iterations=1,
    use_nonuniform_discretization=False, shift_solution=False,
    mu=C["locomotion"]["friction"],
)
cfg.locomotion_params.update(
    liftoff_frame=C["locomotion"].get("liftoff_frame", "body"),
    phase_consistent_swing=C["locomotion"]["phase_consistent_swing"],
    swing_generator=C["locomotion"]["swing_generator"], swing_position_gain_fb=C["locomotion"]["swing_kp"],
    swing_velocity_gain_fb=C["locomotion"]["swing_kd"],
    step_height=C["locomotion"]["step_height_fraction"] * cfg.hip_height,
    ref_z=cfg.hip_height, reflex_trigger_mode=False,
    reflex_next_steps_height_enhancement=False, velocity_modulator=False,
    start_and_stop_activated=False, use_friction_compensation=False,
    visual_foothold_adaptation="blind", gait="trot",
)
cfg.locomotion_params["gait_params"]["trot"].update(
    step_freq=C["locomotion"]["frequency_hz"], duty_factor=C["locomotion"]["duty_factor"])
cfg.simulation_params.update(scene="flat", dt=C["locomotion"]["dt"], mpc_frequency=C["mpc"]["frequency_hz"])
dt = cfg.simulation_params["dt"]
env = QuadrupedEnv(robot=cfg.robot, scene="flat", sim_dt=dt,
    ref_base_lin_vel=0.0, ref_base_ang_vel=0.0,
    ground_friction_coeff=C["locomotion"]["friction"], base_vel_command_type="forward",
    state_obs_names=())
env.reset(seed=0, random=False)
cfg.mass = float(np.sum(env.mjModel.body_mass))
cfg.inertia = env.get_base_inertia()
cfg.mpc_params["grf_max"] = cfg.mass * cfg.gravity_constant
# The task is flat locomotion, with no artificial finite-map termination.
env.terrain_limits = (1000.0, -1000.0, 1000.0, -1000.0)

from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper
from quadruped_pympc.interfaces.wb_interface import WBInterface
wrapper = QuadrupedPyMPC_Wrapper(env.feet_pos, legs_order=LEGS, feet_geom_id=env._feet_geom_id)
interface = wrapper.srbd_controller_interface
controller = interface.controller
if ARGS.method != "acados":
    import jax
    import jax.numpy as jnp
    # The chosen runtime is explicit: a failed GPU allocation is not a CPU experiment.
    device_name = str(jax.devices()[0])
else:
    device_name = "CPU"
runroot = ROOT / "outputs/terrain_memory/reproduction/pympc" / ARGS.run / ARGS.experiment / ARGS.method
smallroot = ROOT / "results/terrain_memory/reproduction/pympc" / ARGS.run / ARGS.experiment / ARGS.method
runroot.mkdir(parents=True, exist_ok=True)
smallroot.mkdir(parents=True, exist_ok=True)
if ARGS.seed_start is not None:
    batch=f"batch-{ARGS.seed_start:03d}-{ARGS.seed_stop:03d}"
    runroot=runroot/batch
    smallroot=smallroot/batch
    runroot.mkdir(parents=True,exist_ok=True)
    smallroot.mkdir(parents=True,exist_ok=True)
    E["seeds"]=[seed for seed in E["seeds"] if ARGS.seed_start<=seed<ARGS.seed_stop]
raw_compute = interface.compute_control
last = {}
first_solve = True

if ARGS.method != "acados":
    raw_optimizer = controller.jitted_compute_control

    def timed_optimizer(*args, **kwargs):
        start = time.perf_counter()
        result = raw_optimizer(*args, **kwargs)
        jax.block_until_ready(result)
        last["optimizer_seconds"] = time.perf_counter() - start
        return result

    controller.jitted_compute_control = timed_optimizer

def timed_compute(*args, **kwargs):
    global first_solve
    t0 = time.perf_counter()
    result = raw_compute(*args, **kwargs)
    # SRBD interface already transfers the chosen GRFs to numpy, synchronizing JAX.
    last["solve_seconds"] = time.perf_counter() - t0
    last["compile_included"] = first_solve and ARGS.method != "acados"
    first_solve = False
    state, ref = args[:2]
    last["state"] = np.concatenate([state[k] for k in ("position","linear_velocity","orientation","angular_velocity")])
    last["reference"] = np.concatenate([ref[k] for k in ("ref_position","ref_linear_velocity","ref_orientation","ref_angular_velocity")])
    last["status"] = controller.previous_status if ARGS.method == "acados" else 0
    if ARGS.method == "acados":
        last["optimizer_seconds"] = float(controller.acados_ocp_solver.get_stats("time_tot"))
    return result

interface.compute_control = timed_compute
source_paths = [ROOT/"scripts/run_tm01_r1.py", Path(ARGS.config).resolve(),
    ROOT/"experiments/tm01/author.patch", ROOT/"experiments/tm01/foothold-2024.patch",
    ROOT/"scripts/patch_tm01_gym.py", ROOT/"scripts/prepare_tm01_2024_model.py"]
for source_path in source_paths:
    (runroot/source_path.name).write_bytes(source_path.read_bytes())
upstream = ROOT/"work/terrain-memory/upstreams/Quadruped-PyMPC"
source_files = {p.name: p.read_bytes() for p in source_paths}
# Runs made between commits must retain the actual code in the small Git results.
source_files["upstream.patch"] = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=upstream)
source_id = hashlib.sha256(b"".join(name.encode()+b"\0"+data for name,data in sorted(source_files.items()))).hexdigest()
snapshot = ROOT/"results/terrain_memory/reproduction/pympc/run-sources"/source_id[:16]
snapshot.mkdir(parents=True, exist_ok=True)
for name,data in source_files.items():
    (snapshot/name).write_bytes(data)
meta = dict(command=sys.argv, paper=C["paper"], experiment=ARGS.experiment, method=ARGS.method,
    seeds=E["seeds"], configuration=str(Path(ARGS.config).resolve().relative_to(ROOT)), configuration_sha256=hashlib.sha256(Path(ARGS.config).read_bytes()).hexdigest(), source_commit=subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT,text=True).strip(),
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    platform=platform.platform(), cpu=next(x.split(":",1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")), device=device_name,
    actual_model_mass_kg=cfg.mass, actual_model_inertia=cfg.inertia.tolist(),
    hip_height=cfg.hip_height,
    actual_joint_damping=env.mjModel.dof_damping.tolist(),
    actual_joint_friction=env.mjModel.dof_frictionloss.tolist(),
    actual_joint_armature=env.mjModel.dof_armature.tolist(),
    actual_torque_limits=env.mjModel.actuator_ctrlrange.tolist(),
    output_directory=str(runroot.relative_to(ROOT)), completed_episodes=0)
meta["source_snapshot"] = str(snapshot.relative_to(ROOT))
meta["source_snapshot_sha256"] = {name:hashlib.sha256(data).hexdigest() for name,data in source_files.items()}
meta["upstream_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
meta["controller_class"] = type(controller).__module__ + "." + type(controller).__name__
meta["candidate_hz"] = candidate_hz if adaptive_family else [C["locomotion"]["frequency_hz"]]
meta["hip_offset_y"] = cfg.hip_offset_y
meta["liftoff_frame"] = cfg.locomotion_params["liftoff_frame"]
meta["active_foothold_offset_y"] = wrapper.wb_interface.frg.hip_offset_y
meta["active_controller_parameters"] = dict(horizon=controller.horizon, dt=controller.dt,
    swing_kp=cfg.locomotion_params["swing_position_gain_fb"],
    swing_kd=cfg.locomotion_params["swing_velocity_gain_fb"],
    grf_max=cfg.mpc_params["grf_max"], mu=cfg.mpc_params["mu"],
    phase_consistent_swing=cfg.locomotion_params["phase_consistent_swing"])
if ARGS.method != "acados":
    meta["active_controller_parameters"].update(rollouts=controller.num_parallel_computations,
        Q_diagonal=np.diag(controller.Q).tolist(), R_diagonal=np.diag(controller.R).tolist(),
        control_parameters=controller.num_control_parameters, sampling_method=controller.sampling_method,
        controller_device=str(controller.device), model_device=str(controller.robot.device))
meta["dependency_versions"] = "experiments/tm01/packages.json"
meta["installed_versions"] = {p.metadata["Name"]: p.version for p in importlib.metadata.distributions()}
meta["timing_context"] = ARGS.timing_context
meta["timing_definitions"] = {
    "optimizer_seconds": "acados native time_tot, or synchronized JAX optimizer call; excludes interface conversion",
    "solve_seconds": "SRBD interface including state conversion, optimizer, and output transfer",
    "control_seconds": "state acquisition through clipped joint torque, including MPC and whole-body control",
    "physics_seconds": "MuJoCo environment step",
    "closed_loop_seconds": "control_seconds + physics_seconds; excludes trajectory serialization and diagnostic extraction",
    "solver_status": "acados native status; sampling zero is a placeholder, not an optimality certificate",
}
meta["actuator_names"] = [mujoco.mj_id2name(env.mjModel, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(env.mjModel.nu)]
meta["upstream_submodules"] = subprocess.check_output(["git", "submodule", "status"], cwd=upstream, text=True).strip()
meta["compute_processes_at_start"] = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
if ARGS.device=="gpu" and ARGS.method!="acados":
    meta["gpu"] = subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip()
(smallroot/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
mujoco.mj_saveLastXML(str(runroot/"model.xml"),env.mjModel)
mujoco.mj_saveModel(env.mjModel, str(runroot/"model.mjb"), None)
meta["model_binary_sha256"] = hashlib.sha256((runroot/"model.mjb").read_bytes()).hexdigest()
base_id = int(env.mjModel.jnt_bodyid[0])
rows = []
for seed in E["seeds"]:
    env.reset(seed=seed,random=False)
    # Independent episodes require resetting warm starts and all low-level history.
    wrapper.wb_interface = WBInterface(env.feet_pos(frame="world"), LEGS, env._feet_geom_id)
    wrapper.last_mpc_step_num = None
    wrapper.best_sample_freq = C["locomotion"]["frequency_hz"]
    interface.previous_contact_mpc = np.ones(4)
    wrapper.nmpc_GRFs = LegsAttr(*[np.zeros(3) for _ in LEGS])
    if ARGS.method == "acados":
        controller.reset()
        controller.previous_status = -1
        controller.previous_contact_sequence[:] = 0
        controller.integral_errors[:] = 0
        controller.previous_optimal_GRF[:] = 0
    else:
        controller.best_control_parameters = jnp.zeros(controller.num_control_parameters,dtype=jnp.float32)
        controller.master_key = jax.random.PRNGKey(seed+100000)
        controller.sampling_key = controller.master_key
    last.clear()
    rng = np.random.default_rng(seed)
    nw = int(np.ceil(E["seconds"] / E.get("wrench_period", 2)))+1
    schedule = rng.uniform(-E.get("wrench_limit",0), E.get("wrench_limit",0), (nw,6))
    nsteps = round(E["seconds"]/dt)
    ref_v = np.array(E["velocity"],dtype=float)
    ref_w = np.zeros(3)
    tau = LegsAttr(*[np.zeros(3) for _ in LEGS])
    hist = {k:[] for k in ("time","qpos","qvel","com","velocity","orientation","angular_velocity",
        "command","wrench","torque","feet","planned_grf","actual_grf","contact","planned_contact","frequency",
        "phase","swing_time","solve_seconds","optimizer_seconds","control_seconds","physics_seconds","closed_loop_seconds",
        "compile_included","solver_status","reference_state","quadratic_cost","requested_torque",
        "liftoff","foothold_target","swing_target","feet_velocity")}
    t_start = time.perf_counter()
    reason = ""
    for step in range(nsteps):
        t = step*dt
        if ARGS.experiment == "fig4":
            wrench = np.array(E["wrench"],float) if E["wrench_start"]<=t<E["wrench_end"] else np.zeros(6)
        else:
            elapsed = t-E["wrench_start"]
            active = elapsed>=0 and elapsed%E["wrench_period"]<E["wrench_duration"]
            wrench = schedule[int(elapsed//E["wrench_period"])] if active else np.zeros(6)
        env.mjData.qfrc_applied[:] = 0
        mujoco.mj_applyFT(env.mjModel, env.mjData, wrench[:3], wrench[3:], env.com, base_id, env.mjData.qfrc_applied)
        control_start = time.perf_counter()
        feet = env.feet_pos(frame="world")
        v = env.base_lin_vel(frame="world").copy()
        ori = env.base_ori_euler_xyz.copy()
        omega = env.base_ang_vel(frame="base").copy()
        com = env.com.copy()
        last["solve_seconds"] = np.nan
        last["optimizer_seconds"] = np.nan
        last["compile_included"] = False
        tau = wrapper.compute_actions(
            com, env.base_pos.copy(), v, ori, omega, feet, env.hip_positions(frame="world"),
            env.legs_qvel_idx, None, LEGS, dt, ref_v.copy(), ref_w.copy(), env.step_num,
            env.mjData.qpos, env.mjData.qvel,
            env.feet_jacobians(frame="world",return_rot_jac=False),
            env.feet_jacobians_dot(frame="world",return_rot_jac=False),
            env.feet_vel(frame="world"), env.legs_qfrc_passive, env.legs_qfrc_bias,
            env.legs_mass_matrix, env.legs_qpos_idx, env.legs_qvel_idx, tau,
            cfg.inertia.flatten(), env.mjData.contact)
        action=np.zeros(env.mjModel.nu)
        requested_torque=np.zeros(env.mjModel.nu)
        for leg in LEGS:
            idx=env.legs_tau_idx[leg]
            requested_torque[idx]=tau[leg]
            bounds=0.9*env.mjModel.actuator_ctrlrange[idx]
            action[idx]=np.clip(tau[leg],bounds[:,0],bounds[:,1])
        control_seconds = time.perf_counter() - control_start
        actual_contact, _, actual_grf=env.feet_contact_state(ground_reaction_forces=True)
        grf=np.array([wrapper.nmpc_GRFs[l] for l in LEGS])
        contact=wrapper.wb_interface.current_contact.copy()
        ref_force=np.zeros((4,3))
        ref_force[:,2]=cfg.mass*9.81*contact/max(contact.sum(),1)
        state=np.concatenate([com,v,ori,omega])
        err=state-last["reference"]
        quadratic=float(np.sum(np.array(C["mpc"]["Q"])*err**2) +
            np.sum((grf-ref_force)**2*np.array(C["mpc"]["R_force"])))
        recorded_feet_velocity = env.feet_vel(frame="world")
        data=dict(time=t,qpos=env.mjData.qpos.copy(),qvel=env.mjData.qvel.copy(),com=com,
            velocity=v,orientation=ori,angular_velocity=omega,command=ref_v.copy(),wrench=wrench,
            torque=action,requested_torque=requested_torque,feet=np.array([feet[l] for l in LEGS]),planned_grf=grf,
            liftoff=np.array([wrapper.wb_interface.frg.lift_off_positions[l] for l in LEGS]),
            foothold_target=np.array([wrapper.nmpc_footholds[l] for l in LEGS]),
            swing_target=np.array([wrapper.wb_interface.last_des_foot_pos[l] for l in LEGS]),
            feet_velocity=np.array([recorded_feet_velocity[l] for l in LEGS]),
            actual_grf=np.array([actual_grf[l] for l in LEGS]),
            contact=np.array([actual_contact[l] for l in LEGS]),planned_contact=contact.copy(),
            frequency=float(wrapper.wb_interface.pgg.step_freq),phase=wrapper.wb_interface.pgg.phase_signal.copy(),swing_time=np.array(wrapper.wb_interface.stc.swing_time),
            solve_seconds=last["solve_seconds"],compile_included=last["compile_included"],
            optimizer_seconds=last["optimizer_seconds"],control_seconds=control_seconds,
            solver_status=last["status"],reference_state=last["reference"].copy(),quadratic_cost=quadratic)
        physics_start = time.perf_counter()
        _,_,terminated,truncated,info=env.step(action)
        data["physics_seconds"] = time.perf_counter() - physics_start
        data["closed_loop_seconds"] = control_seconds + data["physics_seconds"]
        for key,value in data.items(): hist[key].append(value)
        if terminated: reason="environment:"+",".join(info["invalid_contacts"]) if info["invalid_contacts"] else "environment_bounds"
        if reason: break
    hist={k:np.asarray(v) for k,v in hist.items()}
    hist["wrench_schedule"]=schedule
    hist["final_qpos"]=env.mjData.qpos.copy()
    hist["final_qvel"]=env.mjData.qvel.copy()
    trace=runroot/f"seed-{seed:03d}.npz"
    np.savez_compressed(trace,**hist)
    error=np.linalg.norm(hist["velocity"]-hist["command"],axis=1)
    solve=hist["solve_seconds"]
    steady=solve[np.isfinite(solve)&~hist["compile_included"]]
    row=dict(experiment=ARGS.experiment,method=ARGS.method,seed=seed,success=int(not reason),
        termination=reason or "completed",duration_s=(step+1)*dt,requested_duration_s=E["seconds"],
        velocity_error_mean_m_s=float(error.mean()),velocity_error_rmse_m_s=float(np.sqrt(np.mean(error**2))),
        quadratic_cost_mean=float(hist["quadratic_cost"].mean()),
        quadratic_cost_integral=float(hist["quadratic_cost"].sum()*dt),
        solve_mean_ms=float(steady.mean()*1000),solve_median_ms=float(np.median(steady)*1000),
        max_frequency_hz=float(hist["frequency"].max()),wall_seconds=time.perf_counter()-t_start,
        trajectory=str(trace.relative_to(ROOT)),sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    rows.append(row)
    with (smallroot/"episodes.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
    meta["completed_episodes"]=len(rows)
    meta["successes"]=sum(r["success"] for r in rows)
    (smallroot/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(row),flush=True)
env.close()
