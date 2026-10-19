"""One predeclared fair frozen-Go2 evaluation or privileged model diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from go2_frozen.counterfactual import CounterfactualRollouts
from go2_frozen.fair_bank import CandidateBankController
from go2_frozen.fair_control import FairController
from go2_frozen.planner import observation_state
from go2_frozen.simulator import Go2Sim
from go2_frozen.task import Task, make_cases, wrap_angle
from run_go2_frozen import csv_write, flatten, serial


def compact_pretty(value, depth=0):
    value = serial(value)
    pad = "  " * depth
    if isinstance(value, dict):
        return "{\n" + ",\n".join("  " * (depth+1) + json.dumps(k) + ": " + compact_pretty(v, depth+1)
                                    for k, v in value.items()) + "\n" + pad + "}"
    if isinstance(value, list) and value and any(isinstance(v, (dict, list)) for v in value):
        return "[\n" + ",\n".join("  " * (depth+1) + compact_pretty(v, depth+1) for v in value) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    path.write_text(compact_pretty(value) + "\n", encoding="utf-8")


def identity():
    files = [ROOT / "scripts/run_go2_fair.py", ROOT / "scripts/run_go2_frozen.py",
             ROOT / "configs/go2_fair_comparison.json", *sorted((ROOT / "src/go2_frozen").glob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in files}


def run(args):
    spec = next(s for s in make_cases() if s.name == args.case)
    label = f"{args.suite}_{args.speed}_{args.mode}"
    if args.suite == "bank":
        label += f"_{args.predictor}"
    label += "_" + args.case
    directory, raw_directory = args.output / label, args.raw_output / label
    if directory.exists() or raw_directory.exists():
        raise FileExistsError(f"Preserve every attempt: output already exists for {label}")
    directory.mkdir(parents=True)
    raw_directory.mkdir(parents=True)
    task = Task(spec)
    sim = Go2Sim(args.assets / "rl_sar/policy/go2", args.assets / "rl_sar_zoo/go2_description/mjcf",
                 raw_directory, obstacles=task.obstacles())
    forks = CounterfactualRollouts(sim)
    controller = (FairController(args.mode, args.speed) if args.suite == "primary" else
                  CandidateBankController(args.mode, args.predictor, forks=forks, speed_band=args.speed))
    cfg = controller.config
    baseline = json.loads((ROOT / f"notes/go2-frozen-2026-10-19/runs/mpc_{spec.name}/summary.json").read_text())
    assert baseline["task"] == spec.to_dict(), "Scene/task contract changed"
    old_scene = ROOT / f"work/go2-final-runs/mpc_{spec.name}/scene.xml"
    scene_hash = hashlib.sha256(sim.scene_path.read_bytes()).hexdigest()
    old_scene_hash = hashlib.sha256(old_scene.read_bytes()).hexdigest() if old_scene.exists() else None
    if old_scene_hash is not None:
        assert scene_hash == old_scene_hash, "Generated physics scene changed"
    start = time.perf_counter()
    obs = sim.reset(y=spec.start_y, warmup_seconds=2.0)
    initial = serial(obs)
    write_json(raw_directory / "warmup.json", sim.warmup_trace)
    metrics = task.update(obs)
    rows = [flatten(obs, metrics)]
    decisions, errors, transitions = [], [], []
    steps = 0
    planned_next = None
    command = np.zeros(3)
    integral_torque2 = integral_work = integral_path2 = travelled = 0.0
    command_speed_violation = 0.0
    max_tilt = 0.0
    reason, runtime_error = "timeout", None
    renderer = writer = camera = None
    if args.video:
        import imageio.v2 as imageio
        import mujoco
        renderer = mujoco.Renderer(sim.model, height=480, width=640)
        writer = imageio.get_writer(str(raw_directory / "rollout.mp4"), fps=10, codec="libx264", quality=7)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 4.5, 135., -38.
    print(f"START {label}", flush=True)
    try:
        while obs["time"] < spec.timeout - 1e-8:
            if obs["fallen_ever"]:
                reason = "fall"
                break
            if steps % round(cfg.dt / sim.policy_interval) == 0:
                state = observation_state(obs)
                if planned_next is not None:
                    error = state - planned_next
                    error[2] = wrap_angle(error[2])
                    errors.append({"time": obs["time"], **{f"error_{i}": v for i, v in enumerate(error)}})
                previous_command = command.copy()
                command, info = controller.plan(obs, task)
                cap = float(controller.translation_cap(state, task))
                violation = max(0., float(np.linalg.norm(command[:2])) - cap)
                command_speed_violation = max(command_speed_violation, violation)
                assert violation < 1e-9, "Issued command exceeds common speed envelope"
                predicted, _ = controller.nominal_rollout(state, command[None], task)
                planned_next = predicted[0]
                decisions.append({"time": obs["time"], "stage": task.stage, "actual_state": state,
                                  "last_command": previous_command, "command": command,
                                  "translation_speed_ceiling": cap, **info})
                if args.suite == "bank" and len(decisions) % 10 == 0:
                    print(f"PROGRESS {label} t={obs['time']:.1f}s forks={forks.diagnostics['rollouts_completed']}", flush=True)
            previous_obs, previous_stage = obs, metrics["stage"]
            obs = sim.step(command)
            metrics = task.update(obs)
            if metrics["stage"] != previous_stage:
                transitions.append({"from_stage": previous_stage, "to_stage": metrics["stage"],
                                    "observation": serial(obs), "task": serial(metrics)})
            dt = sim.policy_interval
            integral_torque2 += float(np.sum(obs["torque"] ** 2)) * dt
            integral_work += float(np.sum(np.abs(obs["torque"] * obs["joint_vel"]))) * dt
            ref = task.reference(task.project(obs["position"][:2]))
            integral_path2 += float(np.sum((obs["position"][:2] - ref[:2]) ** 2)) * dt
            travelled += float(np.linalg.norm(obs["position"][:2] - previous_obs["position"][:2]))
            max_tilt = max(max_tilt, float(np.max(np.abs(obs["rpy"][:2]))))
            rows.append(flatten(obs, metrics))
            steps += 1
            if renderer is not None and steps % 5 == 0:
                camera.lookat[:] = [*obs["position"][:2], 0.35]
                renderer.update_scene(sim.data, camera=camera)
                writer.append_data(renderer.render())
            if metrics["success"]:
                reason = "success"
                break
    except Exception as exc:
        reason, runtime_error = "execution_error", f"{type(exc).__name__}: {exc}"
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    provenance = sim.provenance()
    unchanged = (provenance["weight_sha256_before"] == provenance["weight_sha256_after"] and
                 provenance["tensor_sha256_before"] == provenance["tensor_sha256_after"])
    latencies = np.array([d["solve_seconds"] for d in decisions])
    error_array = np.array([[r[f"error_{j}"] for j in range(6)] for r in errors])
    metadata = controller.metadata()
    summary = {
        "status": "COMPLETED_EVALUATION" if runtime_error is None else "EXECUTION_ERROR",
        "termination": reason, "runtime_error": runtime_error,
        "suite": args.suite, "mode": args.mode, "speed_band": args.speed,
        "predictor": args.predictor if args.suite == "bank" else "nominal",
        "case": spec.name, "task": spec.to_dict(), "task_metrics": metrics,
        "initial_observation": initial, "final_observation": serial(obs),
        "success": bool(metrics["success"] and not obs["fallen_ever"] and not obs["obstacle_contact_ever"]),
        "elapsed_sim_seconds": obs["time"], "evaluation_wall_seconds": time.perf_counter() - start,
        "policy_steps": steps, "actual_physics_steps_including_warmup": sim.physics_steps,
        "actual_policy_inferences_including_warmup": provenance["inference_calls"],
        "counterfactual_budget": dict(forks.diagnostics), "policy_unchanged": unchanged,
        "policy_training_updates": 0, "model_training_updates": 0,
        "maximum_command_speed_envelope_violation": command_speed_violation,
        "external_disturbances": False, "hardware_connected": False,
        "state_information": "same high-level simulator pose/velocity; unchanged public low-level policy observations",
        "scene_sha256": scene_hash, "previous_scene_sha256": old_scene_hash,
        "scene_matches_previous_evaluation": scene_hash == old_scene_hash if old_scene_hash else None,
        "torque_squared_integral_Nm2_s": integral_torque2,
        "absolute_mechanical_work_proxy_J": integral_work,
        "path_squared_error_integral_m2_s": integral_path2,
        "path_rms_error_m": float(np.sqrt(integral_path2 / max(obs["time"], 1e-12))),
        "distance_travelled_m": travelled, "max_abs_roll_pitch_rad": max_tilt,
        "planner_calls": len(decisions),
        "cem_candidate_rollouts": sum(d["candidate_rollouts"] for d in decisions),
        "cem_scored_model_steps": sum(d["model_steps"] for d in decisions),
        "bank_nominal_model_steps": sum(d.get("bank_nominal_model_steps", 0) for d in decisions),
        "planner_latency_p50_seconds": float(np.median(latencies)) if len(latencies) else None,
        "planner_latency_p95_seconds": float(np.quantile(latencies, .95)) if len(latencies) else None,
        "planner_latency_max_seconds": float(latencies.max()) if len(latencies) else None,
        "planner_latency_exceeds_period_count": int(np.sum(latencies > cfg.dt)),
        "one_interval_prediction_rmse_xy_yaw_vxy_yawrate": np.sqrt(np.mean(error_array ** 2, axis=0)) if len(errors) else None,
        "controller_metadata": metadata, "provenance": provenance,
        "source_sha256": identity(), "python_version": platform.python_version(),
        "notes": ["No policy/model training or fitting. Objectives and speed bands frozen before evaluation.",
                  "Physics pauses during planning. Oracle model access is privileged and its full budget is separate.",
                  "Task evaluation is success first then elapsed time; path and work are tradeoffs, not shared scalar training reward.",
                  "Finite-bank predictor intervention is conditional on these six candidates and nominal proposal construction, not a globally exact controller."]}
    csv_write(raw_directory / "trajectory_50hz.csv", rows)
    csv_write(directory / "trajectory_10hz.csv", rows[::5] + ([rows[-1]] if (len(rows)-1) % 5 else []))
    csv_write(directory / "prediction_errors.csv", errors)
    write_json(directory / "decisions.json", decisions)
    write_json(directory / "stage_transitions.json", transitions)
    write_json(directory / "summary.json", summary)
    result = {"label": label, "success": summary["success"], "time_s": obs["time"],
              "path_rms_m": summary["path_rms_error_m"], "work_proxy_J": integral_work,
              "oracle_steps": forks.diagnostics["physics_steps"], "frozen": unchanged, "termination": reason}
    print("DONE " + json.dumps(serial(result), ensure_ascii=False), flush=True)
    if runtime_error:
        raise RuntimeError(runtime_error)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "work/go2-public-assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--suite", choices=("primary", "bank"), default="primary")
    parser.add_argument("--mode", choices=("follower", "tracking", "mission"), required=True)
    parser.add_argument("--speed", choices=("standard", "fast"), default="fast")
    parser.add_argument("--predictor", choices=("nominal", "physics"), default="nominal")
    parser.add_argument("--case", choices=("nominal", "narrow", "wide"), default="nominal")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.suite == "bank" and (args.mode == "follower" or args.speed != "fast" or args.case != "nominal"):
        parser.error("The declared bank diagnostic uses nominal case, fast speed, and tracking/mission only")
    run(args)
