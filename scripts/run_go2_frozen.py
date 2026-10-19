"""Bounded CPU-only Go2 frozen-policy simulation. No optimizer or robot SDK."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time


def plain(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def write_json(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--config", default="configs/go2_frozen_tasks.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["locomotion", "tasks"], default="tasks")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--conditions", nargs="+")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    os.nice(config["resource_budget"]["nice"])
    os.sched_setaffinity(0, set(config["resource_budget"]["cpu_affinity"]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from go2_frozen.simulation import Go2Simulation
    from go2_frozen.task import (feature, advance_stage, nominal_command, running_cost,
                                 terminal_ok, terminal_surrogate, is_fallen, kinematic_step, wrap)
    from go2_frozen.planner import CommandMPC
    from go2_frozen.realized_learning import (ExecutionIdentity, ExecutedMacroBatch, PredictedMacroBatch,
                                              factual_response_loss, realized_macro_costs, realized_return_to_go)
    policy_path = args.asset_root/"himloco/himloco.pt"
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    started, cpu_started = time.perf_counter(), time.process_time()
    record = {"status": "RUNNING", "mode": args.mode, "pid": os.getpid(),
              "versions": {x: importlib.metadata.version(x) for x in ("mujoco", "torch", "numpy", "PyYAML")},
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "nice": os.getpriority(os.PRIO_PROCESS, 0),
              "gpu_used": False, "automatic_monitoring": False, "training_updates": 0,
              "policy_sha256_before": policy_sha,
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "cases": []}
    write_json(args.output/"execution.json", record)
    write_json(args.output/"protocol.json", config)
    sim = Go2Simulation(args.asset_root/"go2/scene_go2.xml", policy_path,
                        args.asset_root/"himloco/config.yaml", physics_dt=config["physics_dt"],
                        obstacles=config["obstacles"] if args.mode == "tasks" else [])
    record["simulation"] = sim.provenance()
    if args.mode == "locomotion":
        sim.reset(settle_seconds=config["settle_seconds"])
        rows = []
        for command, duration in [([0.3, 0, 0], 3), ([0.15, 0, 0.5], 2), ([0, 0, 0], 2)]:
            for _ in range(round(duration/config["macro_dt"])):
                step = sim.step_command(command, config["macro_dt"])
                rows.append({"command": command, **step})
        write_json(args.output/"locomotion.json", rows)
        record["locomotion"] = {"last_state": sim.state(), "minimum_height": min(v["state"]["position"][2] for v in rows),
                                "maximum_tilt": max(v["state"]["tilt"] for v in rows)}
    else:
        for method in args.methods or config["methods"]:
            for condition in config["conditions"]:
                if args.conditions and condition["name"] not in args.conditions:
                    continue
                sim.reset(settle_seconds=config["settle_seconds"])
                initial = sim.state()
                task_start_time = float(initial["time"])
                planner = None if method == "waypoint" else CommandMPC(sim, config, method)
                stage, hold, gate_crossed, all_clear, pushed = 0, 0.0, False, False, False
                collided, invalid_ground, fallen, success = False, False, False, False
                rows, traces, planned, actual, costs, resources, latencies = [], [], [], [], [], [], []
                case_start = time.perf_counter()
                reason = "deadline"
                push_event = None
                prior_y = float(initial["position"][1])
                for macro in range(round(config["deadline_seconds"]/config["macro_dt"])):
                    begin = time.perf_counter()
                    state = sim.state()
                    stage = advance_stage(state, stage, config)
                    if planner is None:
                        command = nominal_command(state, stage, config)
                        plan = {"command": command, "predicted_feature": feature(kinematic_step(state, command, config["macro_dt"])),
                                "wall_seconds": 0.0, "candidates": 0}
                    else:
                        plan = planner.plan(sim, stage)
                        command = plan["command"]
                    force = None
                    if not pushed and condition["push_direction"] and state["position"][0] >= config["push"]["trigger_x"]:
                        force = [0.0, condition["push_direction"]*sim.total_mass*config["push"]["impulse_over_mass"]/config["push"]["duration"], 0.0]
                        push_event = {"task_time": state["time"]-task_start_time, "stage": stage,
                                      "position": state["position"], "velocity": state["linear_velocity_world"],
                                      "force_world": force, "duration": config["macro_dt"], "robot_mass": sim.total_mass}
                        pushed = True
                    step = sim.step_command(command, config["macro_dt"], external_force=force)
                    elapsed = time.perf_counter()-begin
                    latencies.append(elapsed)
                    state = step["state"]
                    collided |= bool(step["obstacle_contacts"])
                    invalid_ground |= bool(step["nonfoot_ground_contacts"])
                    fallen |= is_fallen(state, config)
                    frame_states = step["trace"]
                    for frame in frame_states:
                        frame["task_time"] = frame["time"]-task_start_time
                        frame["stage"] = stage
                        frame["macro"] = macro
                        x, y = frame["position"][:2]
                        if prior_y <= config["gate"]["y"] < y and config["gate"]["x_min"] < x < config["gate"]["x_max"]:
                            gate_crossed = True
                        prior_y = y
                        all_clear = gate_crossed and frame["collision_aabb_min"][1] > config["gate"]["y"]+config["gate"]["half_thickness"]
                        hold = hold + sim.policy_dt if stage == len(config["route"])-1 and all_clear and terminal_ok(frame, config) else 0.0
                        fallen |= is_fallen(frame, config)
                    traces.extend(frame_states)
                    measured_cost = config["macro_dt"]*running_cost(state, stage, config)
                    planned.append(plan["predicted_feature"])
                    actual.append(feature(state))
                    costs.append(measured_cost)
                    resources.append(step["torque_squared_integral"])
                    rows.append({"macro": macro, "stage": stage, "task_time": state["time"]-task_start_time,
                                 "state": state, "command": command, "force_world": force,
                                 "predicted_feature": plan["predicted_feature"], "task_cost": measured_cost,
                                 "torque_squared_integral": step["torque_squared_integral"],
                                 "absolute_work": step["absolute_work"],
                                 "obstacle_contacts": step["obstacle_contacts"], "nonfoot_ground_contacts": step["nonfoot_ground_contacts"],
                                 "planner_seconds": plan["wall_seconds"], "end_to_end_seconds": elapsed})
                    success = hold >= config["success"]["hold_seconds"]-1e-8 and not (collided or invalid_ground or fallen)
                    if success:
                        reason = "success"
                        break
                    if fallen:
                        reason = "fall"
                        break
                    if time.perf_counter()-case_start > config["resource_budget"]["maximum_case_wall_seconds"]:
                        reason = "compute_budget_exceeded"
                        break
                # Every failure is retained; failure penalties are explicit and
                # applied only to the measured complete-task report, not hidden.
                costs[-1] += terminal_surrogate(state, stage, config) + (0.0 if success else 100.0)
                rows[-1]["task_cost"] = costs[-1]
                identity = ExecutionIdentity(policy_sha, config["protocol_id"], method+"/"+condition["name"])
                executed = ExecutedMacroBatch(torch.tensor(np.asarray(actual))[:, None, :], torch.tensor(resources)[:, None],
                                               identity, torch.tensor(costs)[:, None], complete_task=reason != "compute_budget_exceeded")
                predicted = PredictedMacroBatch(torch.tensor(np.asarray(planned))[:, None, :], torch.zeros(len(resources), 1),
                                                 identity, method)
                factual = factual_response_loss(predicted, executed, resource_weight=0.0)
                realized = realized_macro_costs(executed)
                returns = realized_return_to_go(realized.total, True) if executed.complete_task else None
                error = np.asarray(planned)-np.asarray(actual)
                error[:, 2] = wrap(error[:, 2])
                case = {"method": method, "condition": condition["name"], "success": success, "reason": reason,
                        "stage_reached": stage, "gate_crossed": gate_crossed, "whole_robot_clear": all_clear,
                        "obstacle_collision": collided, "nonfoot_ground_contact": invalid_ground, "fallen": fallen,
                        "task_seconds": state["time"]-task_start_time, "terminal_hold_seconds": hold,
                        "final_position_error": float(np.linalg.norm(np.asarray(state["position"][:2])-config["route"][-1][:2])),
                        "final_yaw_error": abs(float(wrap(state["yaw"]-config["route"][-1][2]))),
                        "final_speed": float(np.linalg.norm(state["linear_velocity_world"][:2])),
                        "task_cost": sum(costs), "torque_squared_integral": sum(resources),
                        "absolute_mechanical_work": sum(x["absolute_work"] for x in rows),
                        "prediction_feature_rmse": np.sqrt((error**2).mean(0)),
                        "factual_loss_real_batch": float(factual.loss),
                        "realized_return_start": float(returns[0, 0]) if returns is not None else None,
                        "push_event": push_event, "wall_seconds": time.perf_counter()-case_start,
                        "end_to_end_p50_seconds": float(np.median(latencies)),
                        "end_to_end_p95_seconds": float(np.quantile(latencies, .95)),
                        "end_to_end_max_seconds": max(latencies),
                        "deadline_miss_fraction": float(np.mean(np.asarray(latencies)>config["macro_dt"])),
                        "executed_physics_steps": len(rows)*round(config["macro_dt"]/config["physics_dt"]),
                        "executed_policy_calls": len(rows)*round(config["macro_dt"]/sim.policy_dt),
                        "planning_candidates": planner.total_candidates if planner else 0,
                        "prediction_physics_steps": planner.predictor.counters["physics_steps"] if planner and planner.predictor else 0,
                        "prediction_policy_calls": planner.predictor.counters["policy_inferences"] if planner and planner.predictor else 0,
                        "training_updates": 0}
                folder = args.output/(method+"--"+condition["name"])
                folder.mkdir()
                write_json(folder/"summary.json", case)
                write_json(folder/"macro.json", rows)
                write_json(folder/"trace.json", traces)
                record["cases"].append(case)
                write_json(args.output/"execution.json", record)
                print(json.dumps(plain(case), allow_nan=False), flush=True)
    status = "COMPLETED_WITH_INCOMPLETE_CASES" if any(c["reason"] == "compute_budget_exceeded" for c in record["cases"]) else "COMPLETED"
    record.update(status=status, wall_seconds=time.perf_counter()-started,
                  process_cpu_seconds=time.process_time()-cpu_started,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  policy_sha256_after=hashlib.sha256(policy_path.read_bytes()).hexdigest())
    write_json(args.output/"execution.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "cases" and k != "locomotion"}), flush=True)


if __name__ == "__main__":
    main()
