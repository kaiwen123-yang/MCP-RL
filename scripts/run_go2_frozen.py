"""Run full Go2 contact simulation with frozen public policy and ordinary MPC.

No optimisation of policy/model parameters, no hardware or network interfaces.
The MPC optimises commands only. All unsuccessful runs are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from go2_frozen.planner import CommandMPC, MPCConfig, ReferenceFollower, observation_state
from go2_frozen.simulator import Go2Sim
from go2_frozen.task import Task, make_cases, wrap_angle


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def flatten(obs, metrics):
    row = {"time": obs["time"], "stage": metrics["stage"],
           "progress": metrics["progress"], "yaw": obs["yaw"], "yaw_rate": obs["yaw_rate"],
           "goal_distance": metrics["goal_distance"], "planar_speed": metrics["planar_speed"],
           "obstacle_contact": int(obs["obstacle_contact"]),
           "obstacle_contact_ever": int(obs["obstacle_contact_ever"]),
           "fallen": int(obs["fallen"]), "fallen_ever": int(obs["fallen_ever"])}
    for key in ("position", "rpy", "velocity_world", "velocity_body", "omega_body", "command",
                "joint_pos", "joint_vel", "torque", "torque_command", "action", "foot_contact"):
        for i, v in enumerate(obs[key]):
            row[f"{key}_{i}"] = float(v)
    return row


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_identity():
    files = [ROOT / "scripts/run_go2_frozen.py", *sorted((ROOT / "src/go2_frozen").glob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in files}


def run(args):
    spec = next(s for s in make_cases() if s.name == args.case)
    run_dir = args.output / f"{args.controller}_{args.case}"
    if run_dir.exists():
        raise FileExistsError(f"Retain prior runs; choose a fresh --output directory: {run_dir}")
    run_dir.mkdir(parents=True)
    raw_dir = args.raw_output / f"{args.controller}_{args.case}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cfg = MPCConfig()
    controller = CommandMPC(cfg) if args.controller == "mpc" else ReferenceFollower(cfg)
    nominal_model = CommandMPC(cfg)
    task = Task(spec)
    sim = Go2Sim(args.assets / "rl_sar/policy/go2",
                 args.assets / "rl_sar_zoo/go2_description/mjcf",
                 raw_dir, obstacles=task.obstacles())
    wall_start = time.perf_counter()
    obs = sim.reset(y=spec.start_y, warmup_seconds=2.0)
    write_json(raw_dir / "warmup.json", sim.warmup_trace)
    initial = serial(obs)
    metrics = task.update(obs)
    rows = [flatten(obs, metrics)]
    decisions, errors, boundaries = [], [], []
    renderer = writer = camera = None
    if args.video:
        import imageio.v2 as imageio
        import mujoco
        renderer = mujoco.Renderer(sim.model, height=480, width=640)
        writer = imageio.get_writer(str(raw_dir / "rollout.mp4"), fps=10, codec="libx264", quality=7)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 4.5, 135.0, -38.0
    command = np.zeros(3)
    steps = 0
    plan_every = round(cfg.dt / sim.policy_interval)
    planned_next = None
    replay_state = observation_state(obs)
    replay_records = []
    input_integral = abs_work = path_integral = distance_travelled = 0.0
    max_tilt = 0.0
    reason = "timeout"
    try:
        while obs["time"] < spec.timeout - 1e-8:
            if obs["fallen_ever"]:
                reason = "fall"
                break
            if steps % plan_every == 0:
                actual_state = observation_state(obs)
                if planned_next is not None:
                    error = actual_state - planned_next
                    error[2] = wrap_angle(error[2])
                    errors.append({"time": obs["time"], **{f"error_{i}": float(v) for i, v in enumerate(error)}})
                replay_records.append({"time": obs["time"],
                                       **{f"actual_{i}": float(v) for i, v in enumerate(actual_state)},
                                       **{f"replay_{i}": float(v) for i, v in enumerate(replay_state)}})
                command, info = controller.plan(obs, task)
                planned_next = nominal_model.step_model(actual_state, command)
                replay_state = nominal_model.step_model(replay_state, command)
                decisions.append({"time": obs["time"], "stage": task.stage,
                                  "actual_state": actual_state, "command": command, **info})
            previous = obs
            previous_stage = metrics["stage"]
            obs = sim.step(command)
            metrics = task.update(obs)
            if metrics["stage"] != previous_stage:
                boundaries.append({"from_stage": previous_stage, "to_stage": metrics["stage"],
                                   "observation": serial(obs), "task": serial(metrics)})
            dt = sim.policy_interval
            input_integral += float(np.sum(obs["torque"] ** 2)) * dt
            # Mechanical absolute power proxy, not battery electrical energy.
            abs_work += float(np.sum(np.abs(obs["torque"] * obs["joint_vel"]))) * dt
            reference = task.reference(task.project(obs["position"][:2]))
            path_integral += float(np.sum((obs["position"][:2] - reference[:2]) ** 2)) * dt
            distance_travelled += float(np.linalg.norm(obs["position"][:2] - previous["position"][:2]))
            max_tilt = max(max_tilt, float(np.max(np.abs(obs["rpy"][:2]))))
            rows.append(flatten(obs, metrics))
            steps += 1
            if renderer is not None and steps % 5 == 0:
                camera.lookat[:] = [obs["position"][0], obs["position"][1], 0.35]
                renderer.update_scene(sim.data, camera=camera)
                writer.append_data(renderer.render())
            if metrics["success"]:
                reason = "success"
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    provenance = sim.provenance()
    policy_unchanged = (provenance["weight_sha256_before"] == provenance["weight_sha256_after"]
                        and provenance["tensor_sha256_before"] == provenance["tensor_sha256_after"])
    latencies = np.array([d["solve_seconds"] for d in decisions])
    error_matrix = np.array([[e[f"error_{i}"] for i in range(6)] for e in errors])
    one_step_rmse = np.sqrt(np.mean(error_matrix ** 2, axis=0)).tolist() if len(errors) else None
    replay_error = np.array([[r[f"actual_{i}"] - r[f"replay_{i}"] for i in range(6)] for r in replay_records])
    replay_error[:, 2] = wrap_angle(replay_error[:, 2])
    summary = {
        "status": "COMPLETED_EVALUATION", "termination": reason,
        "controller": args.controller, "case": spec.name, "task": spec.to_dict(),
        "task_metrics": metrics, "initial_observation": initial, "final_observation": serial(obs),
        "success": bool(metrics["success"] and not obs["fallen_ever"] and not obs["obstacle_contact_ever"]),
        "elapsed_sim_seconds": obs["time"], "evaluation_wall_seconds": time.perf_counter() - wall_start,
        "policy_steps": steps, "physics_steps_including_warmup": sim.physics_steps,
        "policy_unchanged": policy_unchanged, "policy_training_updates": 0, "model_training_updates": 0,
        "external_disturbances": False, "hardware_connected": False,
        "state_information": "simulator pose and velocity for high-level control; unchanged public policy observations",
        "torque_squared_integral_Nm2_s": input_integral,
        "absolute_mechanical_work_proxy_J": abs_work,
        "path_squared_error_integral_m2_s": path_integral,
        "distance_travelled_m": distance_travelled, "max_abs_roll_pitch_rad": max_tilt,
        "planner_calls": len(decisions),
        "planner_candidate_rollouts": sum(d["candidate_rollouts"] for d in decisions),
        "planner_model_steps": sum(d["model_steps"] for d in decisions),
        "planner_latency_p50_seconds": float(np.quantile(latencies, .5)),
        "planner_latency_p95_seconds": float(np.quantile(latencies, .95)),
        "planner_latency_max_seconds": float(latencies.max()),
        "planner_latency_exceeds_period_count": int(np.sum(latencies > cfg.dt)),
        "one_interval_prediction_rmse_xy_yaw_vxy_yawrate": one_step_rmse,
        "recorded_command_open_loop_rmse_xy_yaw_vxy_yawrate": np.sqrt(np.mean(replay_error ** 2, axis=0)),
        "prediction_evaluation": "same actual commands; one-interval reinitialised prediction versus full open-loop replay; no future replanning derivative",
        "controller_metadata": controller.metadata(), "provenance": provenance,
        "source_sha256": source_identity(), "python_version": platform.python_version(),
        "notes": ["Single deterministic simulation per declared controller/case, not independent statistical trials.",
                  "Time is simulation time; the physics pauses during command optimisation. Latency is separately reported.",
                  "Warmup is separately retained and its fall/contact flags remain active.",
                  "MPC uses an engineering response model, not a fitted/certified model or an optimal oracle."],
    }
    csv_write(raw_dir / "trajectory_50hz.csv", rows)
    csv_write(run_dir / "trajectory_10hz.csv", rows[::5] + ([rows[-1]] if (len(rows)-1) % 5 else []))
    csv_write(run_dir / "prediction_errors.csv", errors)
    csv_write(run_dir / "recorded_command_replay.csv", replay_records)
    write_json(run_dir / "decisions.json", decisions)
    write_json(run_dir / "stage_transitions.json", boundaries)
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(serial({k: summary[k] for k in ("controller", "case", "success", "termination",
          "elapsed_sim_seconds", "evaluation_wall_seconds", "planner_latency_p95_seconds", "policy_unchanged")}), ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "work/go2-public-assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, default=ROOT / "work/go2-runs")
    parser.add_argument("--controller", choices=("mpc", "follower"), default="mpc")
    parser.add_argument("--case", choices=("nominal", "narrow", "wide"), default="nominal")
    parser.add_argument("--video", action="store_true")
    run(parser.parse_args())
