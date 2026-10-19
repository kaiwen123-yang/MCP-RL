"""Pure offline analysis of the nine saved Go2 frozen-policy cases.

Uses JSON/numpy only. No MuJoCo, Torch, policy inference, model rollout, server
access, or parameter update occurs. Process completion and saved case evidence
are reported separately because the supplied run has two recorded exceptions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


FEATURES = ["world_x_m", "world_y_m", "world_yaw_rad", "world_vx_m_s", "world_vy_m_s", "world_yaw_rate_rad_s"]
METHODS = ["waypoint", "kinematic_mpc", "policy_rollout_mpc"]
CONDITIONS = ["nominal", "push_positive_y", "push_negative_y"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def feature(state):
    return np.array([*state["position"][:2], state["yaw"], *state["linear_velocity_world"][:2], state["angular_velocity_world"][2]])


def prediction_statistics(error, mask):
    selected = error[mask]
    if not len(selected):
        return {"macro_count": 0, "rmse": None, "max_abs": None}
    return {"macro_count": len(selected), "rmse": np.sqrt(np.mean(selected**2, axis=0)),
            "max_abs": np.max(np.abs(selected), axis=0),
            "position_vector_rmse_m": np.sqrt(np.mean(np.sum(selected[:, :2]**2, axis=1))),
            "planar_velocity_vector_rmse_m_s": np.sqrt(np.mean(np.sum(selected[:, 3:5]**2, axis=1)))}


def terminal_pass(frame, config):
    goal, limits = config["route"][-1], config["success"]
    return (np.linalg.norm(np.asarray(frame["position"][:2])-goal[:2]) <= limits["position_tolerance"]
            and abs(wrap(frame["yaw"]-goal[2])) <= limits["yaw_tolerance"]
            and np.linalg.norm(frame["linear_velocity_world"][:2]) <= limits["speed_tolerance"]
            and abs(frame["angular_velocity_world"][2]) <= limits["yaw_rate_tolerance"])


def verify_success(traces, macros, case, config, policy_dt):
    gate, crossing, hold, max_hold = config["gate"], False, 0.0, 0.0
    hold_start = None
    previous_y = 0.0  # The frozen protocol resets base position to world origin.
    clear = False
    trace_fall, terminal_frames, held_frames = False, 0, 0
    for frame in traces:
        x, y = frame["position"][:2]
        if previous_y <= gate["y"] < y and gate["x_min"] < x < gate["x_max"]:
            crossing = True
        previous_y = y
        clear = crossing and frame["collision_aabb_min"][1] > gate["y"]+gate["half_thickness"]
        terminal = terminal_pass(frame, config)
        terminal_frames += int(terminal)
        passes = frame["stage"] == len(config["route"])-1 and clear and terminal
        if passes:
            if hold_start is None:
                hold_start = frame["time"]
            hold = frame["time"] - hold_start
        else:
            hold_start, hold = None, 0.0
        held_frames = held_frames + 1 if passes else 0
        max_hold = max(max_hold, hold)
        trace_fall |= (frame["position"][2] < config["fall"]["minimum_height"]
                       or frame["tilt"] > config["fall"]["maximum_tilt"])
    obstacle_steps = sum(row["obstacle_contacts"] for row in macros)
    nonfoot_steps = sum(row["nonfoot_ground_contacts"] for row in macros)
    reconstructed = hold >= config["success"]["hold_seconds"]-1e-8 and not (obstacle_steps or nonfoot_steps or trace_fall)
    times = np.array([frame["time"] for frame in traces])
    return {"summary_success": case["success"], "recomputed_success": bool(reconstructed),
            "success_agrees": bool(reconstructed) == case["success"],
            "frame_count": len(traces), "terminal_condition_frames": terminal_frames,
            "final_consecutive_hold_frames": held_frames, "final_hold_seconds": hold,
            "maximum_hold_seconds": max_hold, "hold_seconds_difference": hold-case["terminal_hold_seconds"],
            "hold_definition": "Last timestamp minus first timestamp of the final uninterrupted qualifying frame sequence; first qualifying frame contributes zero elapsed duration",
            "gate_crossed": crossing, "whole_robot_clear_at_end": bool(clear),
            "obstacle_contact_physics_steps": obstacle_steps,
            "nonfoot_ground_contact_physics_steps": nonfoot_steps, "fallen_in_saved_frames": bool(trace_fall),
            "minimum_base_height_m": min(f["position"][2] for f in traces),
            "maximum_tilt_rad": max(f["tilt"] for f in traces),
            "trace_policy_interval_max_error_seconds": float(np.max(np.abs(np.diff(times)-policy_dt))) if len(times)>1 else None,
            "contact_scope": "macro counts aggregate all 5ms physics steps; terminal/full-body clearance checked at saved 20ms policy frames",
            "not_claimed": "continuous-time or real-hardware safety"}


def inspect_case(folder, execution, config):
    case, macros, traces = (read(folder/name) for name in ("summary.json", "macro.json", "trace.json"))
    policy_dt = execution["simulation"]["policy"]["policy_period_seconds"]
    actual = np.stack([feature(row["state"]) for row in macros])
    predicted = np.array([row["predicted_feature"] for row in macros])
    error = predicted-actual
    error[:, 2] = wrap(error[:, 2])
    pushed = np.array([row["force_world"] is not None and np.linalg.norm(row["force_world"])>0 for row in macros])
    push_indices = np.flatnonzero(pushed)
    groups = {"all": np.ones(len(macros), dtype=bool), "push_macro": pushed, "nonpush_macros": ~pushed}
    if len(push_indices):
        groups["prepush_macros"] = np.arange(len(macros)) < push_indices[0]
        groups["postpush_macros"] = np.arange(len(macros)) > push_indices[-1]
    stats = {name: prediction_statistics(error, mask) for name, mask in groups.items()}
    end_to_end = np.array([row["end_to_end_seconds"] for row in macros])
    planner = np.array([row["planner_seconds"] for row in macros])
    success = verify_success(traces, macros, case, config, policy_dt)
    physics = len(macros)*round(config["macro_dt"]/config["physics_dt"])
    policy_calls = len(macros)*round(config["macro_dt"]/policy_dt)
    consistency = {
        "macro_task_cost_sum_abs_error": abs(sum(row["task_cost"] for row in macros)-case["task_cost"]),
        "macro_torque_integral_sum_abs_error": abs(sum(row["torque_squared_integral"] for row in macros)-case["torque_squared_integral"]),
        "macro_work_sum_abs_error": abs(sum(row["absolute_work"] for row in macros)-case["absolute_mechanical_work"]),
        "reported_prediction_rmse_max_abs_error": float(np.max(np.abs(stats["all"]["rmse"]-case["prediction_feature_rmse"]))),
        "executed_physics_steps_agree": physics == case["executed_physics_steps"],
        "executed_policy_calls_agree": policy_calls == case["executed_policy_calls"],
        "trace_count_agrees": len(traces) == policy_calls,
        "macro_count_time_abs_error": abs(len(macros)*config["macro_dt"]-case["task_seconds"]),
        "push_macro_count": int(pushed.sum()),
        "expected_push_macro_count": 0 if case["condition"] == "nominal" else 1,
    }
    if case["method"] == "policy_rollout_mpc":
        consistency["prediction_physics_steps_agree"] = case["prediction_physics_steps"] == case["planning_candidates"]*config["planner"]["horizon"]*round(config["macro_dt"]/config["physics_dt"])
        consistency["prediction_policy_calls_agree"] = case["prediction_policy_calls"] == case["planning_candidates"]*config["planner"]["horizon"]*round(config["macro_dt"]/policy_dt)
    row = {key: case[key] for key in ("method", "condition", "success", "reason", "task_seconds", "task_cost",
           "torque_squared_integral", "absolute_mechanical_work", "wall_seconds", "final_position_error",
           "final_yaw_error", "final_speed", "terminal_hold_seconds", "executed_physics_steps",
           "executed_policy_calls", "planning_candidates", "prediction_physics_steps", "prediction_policy_calls", "training_updates")}
    row.update(macro_count=len(macros), position_prediction_rmse_m=stats["all"]["position_vector_rmse_m"],
               timestamp_terminal_hold_seconds=success["final_hold_seconds"],
               kinematic_candidate_macro_steps=case["planning_candidates"]*config["planner"]["horizon"] if case["method"] == "kinematic_mpc" else 0,
               waypoint_reporting_prediction_macro_steps=len(macros) if case["method"] == "waypoint" else 0,
               velocity_prediction_rmse_m_s=stats["all"]["planar_velocity_vector_rmse_m_s"],
               yaw_prediction_rmse_rad=stats["all"]["rmse"][2],
               planner_mean_seconds=float(planner.mean()), planner_p95_seconds=float(np.quantile(planner,.95)),
               end_to_end_mean_seconds=float(end_to_end.mean()), end_to_end_p95_seconds=float(np.quantile(end_to_end,.95)),
               deadline_miss_fraction=float(np.mean(end_to_end > config["macro_dt"])),
               deadline_miss_macro_count=int(np.sum(end_to_end > config["macro_dt"])),
               success_recomputed=success["recomputed_success"])
    return row, {"method": case["method"], "condition": case["condition"], "prediction": stats,
                 "success_evidence": success, "consistency": consistency, "push_event": case["push_event"],
                 "feature_order": FEATURES,
                 "prediction_scope": "One selected-command macro, 0.2s; same exact nominal simulator does not test model generalization"}


def compare(left, right):
    keys = ("task_seconds", "task_cost", "torque_squared_integral", "absolute_mechanical_work", "wall_seconds",
            "final_position_error", "final_yaw_error", "final_speed", "position_prediction_rmse_m",
            "velocity_prediction_rmse_m_s", "end_to_end_mean_seconds")
    return {"left": left["method"], "right": right["method"], "condition": left["condition"],
            "left_minus_right": {k: left[k]-right[k] for k in keys},
            "change_percent": {k: 100*(left[k]/right[k]-1) if right[k] else None for k in keys},
            "successes": [left["success"], right["success"]],
            "scope": "Descriptive same-condition comparison, not a randomized statistical estimate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("work/go2-frozen-20261019/downloaded"))
    parser.add_argument("--output", type=Path, default=Path("notes/go2-frozen-2026-10-19"))
    parser.add_argument("--replacement-root", type=Path, action="append", default=[],
                        help="Additional completed run root with execution/protocol and replacement case folders; repeatable")
    args = parser.parse_args()
    rows, mechanisms, runs, manifest, configs = [], [], [], [], []
    receipt = read(args.input/"execution-receipt.json")
    for path in (args.input/"execution-receipt.json", args.input/"locomotion-v1"/"execution.json", args.input/"source-manifest.json"):
        manifest.append({"file": str(path.relative_to(args.input)).replace("\\", "/"), "sha256": sha(path)})
    original_rows, original_mechanisms = [], []
    original_keys, replacement_keys = set(), set()
    all_attempt_rows, all_attempt_mechanisms = [], []
    source_roots = [(args.input/name, name, False) for name in ("tasks-v1", "tasks-v1-policy-rollout")]
    source_roots += [(root, "replacement/"+root.name, True) for root in args.replacement_root]
    for root, run_name, is_replacement in source_roots:
        execution, config = read(root/"execution.json"), read(root/"protocol.json")
        configs.append(config)
        runs.append({"run": run_name, "saved_process_status": execution["status"],
                     "saved_case_count": len(execution["cases"]),
                     "wall_seconds": execution.get("wall_seconds"), "process_cpu_seconds": execution.get("process_cpu_seconds"),
                     "policy_sha256_before": execution["policy_sha256_before"],
                     "policy_sha256_after": execution.get("policy_sha256_after"),
                     "training_updates": execution["training_updates"], "gpu_used": execution["gpu_used"],
                     "automatic_monitoring": execution["automatic_monitoring"],
                     "policy_provenance": execution["simulation"]["policy"],
                     "model_xml_sha256": execution["simulation"]["model_xml_sha256"]})
        for path in (root/"execution.json", root/"protocol.json"):
            manifest.append({"file": run_name+"/"+path.name, "sha256": sha(path)})
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or "--" not in folder.name:
                continue
            row, mechanism = inspect_case(folder, execution, config)
            row["source_run"] = run_name
            row["is_timestamp_hold_replacement"] = is_replacement
            mechanism["source_run"] = run_name
            key = (row["method"], row["condition"])
            if is_replacement:
                if key in replacement_keys:
                    raise ValueError(f"Duplicate replacement for {key}; provide one selected correction run")
                if key not in original_keys:
                    raise ValueError(f"Replacement {key} has no original frozen case")
                replacement_keys.add(key)
                rows = [r for r in rows if (r["method"],r["condition"]) != key]
                mechanisms = [m for m in mechanisms if (m["method"],m["condition"]) != key]
            else:
                original_keys.add(key)
                original_rows.append(row); original_mechanisms.append(mechanism)
            rows.append(row); mechanisms.append(mechanism)
            all_attempt_rows.append(row); all_attempt_mechanisms.append(mechanism)
            for name in ("summary.json", "macro.json", "trace.json"):
                path = folder/name
                manifest.append({"file": run_name+"/"+str(path.relative_to(root)).replace("\\", "/"), "sha256": sha(path)})
    index = {(r["method"],r["condition"]): r for r in rows}
    expected = {(m,c) for m in METHODS for c in CONDITIONS}
    missing = sorted(expected-set(index))
    issues = []
    if len(index) != len(rows): issues.append("duplicate_method_condition")
    if any(config != configs[0] for config in configs[1:]): issues.append("protocols_differ")
    for m in mechanisms:
        check = m["consistency"]
        if not m["success_evidence"]["success_agrees"]: issues.append(f"success_mismatch:{m['method']}:{m['condition']}")
        if any(not value for key,value in check.items() if key.endswith("_agree") or key.endswith("_agrees")):
            issues.append(f"count_mismatch:{m['method']}:{m['condition']}")
        if any(value > 1e-7 for key,value in check.items() if key.endswith("abs_error")):
            issues.append(f"arithmetic_mismatch:{m['method']}:{m['condition']}")
        if check["push_macro_count"] != check["expected_push_macro_count"]:
            issues.append(f"push_count_mismatch:{m['method']}:{m['condition']}")
    pairs = [compare(index[("policy_rollout_mpc", c)], index[(baseline,c)])
             for c in CONDITIONS for baseline in ("waypoint", "kinematic_mpc")
             if ("policy_rollout_mpc",c) in index and (baseline,c) in index]
    conditions = []
    for method in METHODS:
        for condition in CONDITIONS[1:]:
            if (method,condition) in index and (method,"nominal") in index:
                delta = compare(index[(method,condition)], index[(method,"nominal")])
                delta.update(perturbed_condition=condition, reference_condition="nominal")
                conditions.append(delta)
    counter_keys = ("executed_physics_steps", "executed_policy_calls", "planning_candidates", "prediction_physics_steps", "prediction_policy_calls", "kinematic_candidate_macro_steps", "waypoint_reporting_prediction_macro_steps", "training_updates")
    totals = {key: sum(r[key] for r in rows) for key in counter_keys}
    all_attempt_totals = {key: sum(r[key] for r in all_attempt_rows) for key in counter_keys}
    totals.update(task_reset_settling_physics_steps=len(rows)*round(configs[0]["settle_seconds"]/configs[0]["physics_dt"]),
                  settling_scope="Derived for the nine completed task cases only; locomotion validation and startup/debug overhead are separate")
    all_attempt_totals.update(task_reset_settling_physics_steps=len(all_attempt_rows)*round(configs[0]["settle_seconds"]/configs[0]["physics_dt"]),
                              completed_case_attempts=len(all_attempt_rows),
                              scope="All original and replacement completed task attempts, including superseded cases; other startup/debug and locomotion work separate")
    physics_dt = configs[0]["physics_dt"]
    policy_dt = runs[0]["policy_provenance"]["policy_period_seconds"]
    locomotion = read(args.input/"locomotion-v1"/"execution.json")
    locomotion_seconds = locomotion["locomotion"]["last_state"]["time"]
    # The saved first-run exception occurred while constructing the predictor,
    # after reset(settle_seconds) for the previously unexecuted next case.
    clone_failure_settle = configs[0]["settle_seconds"]
    setup = {
        "completed_task_attempts": len(all_attempt_rows),
        "settling_seconds_per_task_attempt": configs[0]["settle_seconds"],
        "completed_attempt_settling_physics_steps": all_attempt_totals["task_reset_settling_physics_steps"],
        "clone_failure_preplan_settling_seconds": clone_failure_settle,
        "clone_failure_settling_physics_steps": round(clone_failure_settle/physics_dt),
        "clone_failure_count_source": "Saved clone exception and original runner reset-before-planner order; not a finalized process counter",
        "locomotion_total_seconds_including_its_settle": locomotion_seconds,
        "locomotion_physics_steps_including_settle": round(locomotion_seconds/physics_dt),
        "locomotion_policy_calls_including_settle": round(locomotion_seconds/policy_dt),
    }
    physical_all = (all_attempt_totals["executed_physics_steps"] + setup["completed_attempt_settling_physics_steps"]
                    + setup["clone_failure_settling_physics_steps"] + setup["locomotion_physics_steps_including_settle"])
    setup["all_execution_physics_steps_including_original_repair_settle_failure_locomotion"] = physical_all
    setup["all_execution_policy_calls_including_original_repair_settle_failure_locomotion"] = (
        all_attempt_totals["executed_policy_calls"]
        + len(all_attempt_rows)*round(configs[0]["settle_seconds"]/policy_dt)
        + round(clone_failure_settle/policy_dt) + setup["locomotion_policy_calls_including_settle"])
    archives = [{"file": path.name, "sha256": sha(path)} for path in
                (args.input.parent/"results.tar.gz", args.input.parent/"holdfix-results.tar.gz") if path.exists()]
    source_archives = [{"file": path.name, "sha256": sha(path)} for path in
                       (args.input/"source-tasks.tar", args.input/"source-policy-rollout.tar", args.input/"source-holdfix.tar") if path.exists()]
    methods = []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        if selected:
            methods.append({"method": method, "cases": len(selected), "successes": sum(r["success_recomputed"] for r in selected),
                            "mean_task_cost": np.mean([r["task_cost"] for r in selected]),
                            "mean_task_seconds": np.mean([r["task_seconds"] for r in selected]),
                            "mean_torque_squared_integral": np.mean([r["torque_squared_integral"] for r in selected]),
                            "mean_absolute_mechanical_work": np.mean([r["absolute_mechanical_work"] for r in selected]),
                            "mean_case_wall_seconds": np.mean([r["wall_seconds"] for r in selected]),
                            "deadline_miss_macros": sum(r["deadline_miss_macro_count"] for r in selected),
                            "macro_count": sum(r["macro_count"] for r in selected),
                            "descriptive_three_conditions_not_independent_seeds": True})
    summary = {"status": "COMPLETE_NINE_CASE_EVIDENCE" if not missing and not issues else "INCOMPLETE_OR_INCONSISTENT",
               "cases": rows, "methods": methods, "missing_cases": missing, "consistency_issues": issues,
               "protocol": configs[0], "runs": runs, "original_execution_receipt": receipt,
               "timestamp_hold_correction": {
                   "original_recorded_successes": sum(r["success"] for r in original_rows),
                   "original_timestamp_supported_successes": sum(r["success_recomputed"] for r in original_rows),
                   "original_unsupported_case_keys": [[m["method"],m["condition"]] for m in original_mechanisms if not m["success_evidence"]["recomputed_success"]],
                   "replacement_case_keys": sorted(replacement_keys),
                   "original_evidence_preserved": True,
                   "replacement_selection": "Only explicitly supplied method/condition replacements; all other original cases retained"},
               "totals": totals, "all_attempt_totals_including_superseded": all_attempt_totals,
               "setup_and_failure_accounting": setup, "downloaded_archives": archives, "source_archives": source_archives, "source_manifest": manifest,
               "weight_identity_evidence": {
                   "all_saved_before_hashes_equal": len({run["policy_sha256_before"] for run in runs}) == 1,
                   "all_available_after_hashes_equal_before": all(run["policy_sha256_after"] == run["policy_sha256_before"] for run in runs if run["policy_sha256_after"] is not None),
                   "runs_without_saved_after_hash": [run["run"] for run in runs if run["policy_sha256_after"] is None],
                   "scope": "File hashes and frozen execution provenance; no additional policy inference or weight mutation performed by analysis"},
               "new_simulation_steps": 0, "new_model_steps": 0, "new_policy_calls": 0, "optimizer_steps": 0,
               "process_status_note": "Original six cases saved before clone API exception; three original unexecuted cases continued separately; final stdout serialization exception occurred after saving continuation results. Timestamp-hold replacements, if supplied, are separate additional executions. Do not claim original processes had clean exit 0.",
               "interpretation_limits": [
                   "All feedback weights are frozen; differences are command planning and prediction differences, not new learned capability.",
                   "Policy rollout uses privileged exact nominal simulator dynamics and full policy history. Zero nonpush error is same-simulator consistency, not generalization or real-robot accuracy.",
                   "Kinematic collision scoring uses an approximate heading-aware rectangle; policy-rollout scoring uses full simulated contact. Outcome differences are not isolated solely to dynamics prediction.",
                   "Future force is absent from planning. A push macro is a deliberate unpredicted intervention; later forecasts restart from the actual perturbed state/history.",
                   "Physical simulation is paused while planning. Missed 0.2s computation periods are not inserted as actuation delay; successful simulated tasks are not real-time deployment validation.",
                   "Torque-squared integral and absolute mechanical work are distinct physical proxies, not measured electrical energy.",
                   "Three deterministic conditions are development evidence, not success-rate probability, robustness proof, or hardware transfer.",
                   "First-run process CPU/total wall counters were not finalized after its exception; completed per-case wall times remain available. Do not invent a total process CPU time."
               ]}
    mechanism = {"status": summary["status"], "case_evidence": mechanisms, "method_pairs": pairs,
                 "within_method_condition_changes": conditions, "new_simulation_steps": 0,
                 "new_model_steps": 0, "new_policy_calls": 0, "optimizer_steps": 0}
    write(args.output/"summary.json", summary); write(args.output/"mechanism.json", mechanism)
    write(args.output/"original-summary.json", {"status": "ORIGINAL_NINE_CASES_WITH_HOLD_DEFECT_RETAINED", "cases": original_rows,
                                               "recorded_successes": sum(r["success"] for r in original_rows),
                                               "timestamp_supported_successes": sum(r["success_recomputed"] for r in original_rows),
                                               "source": "Original saved task attempts, not overwritten by corrected reruns"})
    write(args.output/"original-mechanism.json", {"case_evidence": original_mechanisms})
    with (args.output/"original-cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(original_rows[0]))
        writer.writeheader(); writer.writerows(plain(original_rows))
    with (args.output/"cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(plain(rows))
    lines = ["# Go2冻结策略九案例离线汇总", "", "本报告仅重算已保存summary、macro与trace；没有新仿真、策略推理或训练。", "",
             f"状态：{summary['status']}；完整案例{len(rows)}/9；按连续合格帧首末timestamp跨度核实成功{sum(r['success_recomputed'] for r in rows)}/9。三条件不是独立随机种子。", "",
             f"原始9案例全部留在original-*文件：原报告成功9/9，严格timestamp只支持{sum(r['success_recomputed'] for r in original_rows)}/9；本报告采用{len(replacement_keys)}条明确补跑替换，其余原案例保留。所有原尝试与补跑均计入all_attempt_totals_including_superseded。", "",
             "| 方法 | 条件 | 成功 | 任务秒 | 任务cost | 力矩平方积分 | 绝对机械功 | 位置预测RMSE m | 宏步总计算均值 s | 超0.2s比例 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['condition']} | {int(r['success_recomputed'])} | {r['task_seconds']:.2f} | {r['task_cost']:.5f} | {r['torque_squared_integral']:.2f} | {r['absolute_mechanical_work']:.2f} | {r['position_prediction_rmse_m']:.6f} | {r['end_to_end_mean_seconds']:.6f} | {r['deadline_miss_fraction']:.1%} |")
    lines += ["", "同条件policy-rollout MPC相对kinematic MPC：", ""]
    for p in pairs:
        if p["right"] != "kinematic_mpc": continue
        d,pc=p["left_minus_right"],p["change_percent"]
        lines.append(f"- {p['condition']}：任务cost {pc['task_cost']:+.2f}%，时间 {d['task_seconds']:+.2f}s，力矩平方积分 {pc['torque_squared_integral']:+.2f}%，绝对机械功 {pc['absolute_mechanical_work']:+.2f}%，案例墙钟 {pc['wall_seconds']:+.1f}%。")
    lines += ["", "终端成功已按保存的20ms帧重新检查连续保持、全体碰撞几何越门及终态条件；接触计数覆盖每个5ms物理步。细节见mechanism.json。", "",
              "精确策略展开的无扰零误差只表示相同仿真与history的一致性；push宏步单独报告，不能把无扰误差迁移成泛化结论。所有方法均冻结同一策略，没有新反馈学习收益。", "",
              "0.2s预算按实际宏步计算时间核对；仿真在规划期间暂停，因此成功不等于可按真实控制周期部署。资源指标是力矩平方积分与绝对机械功，不能直接称机器人电耗。", "",
              "原始两次进程的异常边界保留：第一进程六案例完成后在clone API处中止，第二进程三案例完整落盘后最终stdout序列化报错。随后timestamp保持修正的补跑单独留存，不能写成原全流程exit0；第一进程未完成的CPU总计保留未知。", "",
              f"成本账同时保留原始9次与补跑：完成任务尝试{len(all_attempt_rows)}次，任务内物理步{all_attempt_totals['executed_physics_steps']}、预测物理步{all_attempt_totals['prediction_physics_steps']}；另计每尝试2秒settle、clone异常前额外2秒settle及含自身settle的9秒locomotion，共计实际执行物理步{physical_all}。不是只报告最终选用9案例的开销。", "",
              "主证据文件：summary.json、cases.csv、mechanism.json。既有来源hash和执行receipt已收入结构化汇总。", ""]
    (args.output/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(plain({"status": summary["status"], "cases": len(rows), "issues": issues,
                            "methods": methods, "totals": totals}), indent=2))


if __name__ == "__main__":
    main()
