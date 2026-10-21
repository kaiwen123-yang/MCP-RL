"""Offline factual-model fit and Go2 continuous-task comparison.

Reads downloaded JSON only (plus hashes of existing files). No Torch, MuJoCo,
policy loading, prediction rollout, training, sampling, or network access.
The three already-known task conditions are development comparisons, not new
task generalization. Never writes over the previous Go2 frozen study.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath

import numpy as np

from analyze_go2_frozen import (FEATURES, feature, plain, prediction_statistics,
                                read, sha, verify_success, wrap, write)


METHODS = ("id_one_step", "compact_multistep", "kinematic_mpc", "policy_rollout_mpc")
CONDITIONS = ("nominal", "push_positive_y", "push_negative_y")
LIMITS = [
    "同一已知scene、route及三种已知条件的开发比较，不是未见任务、平台或地形泛化。",
    "本轮允许拟合事实响应模型；冻结策略参数更新必须单独计为零，不能把模型拟合叫策略学习。",
    "单步和多步模型同更新次数不等于同前反向计算或同墙钟；分别报告完整账本。",
    "在线预测误差对应被选命令的下一宏步；不是全部候选的误差，也不是独立五步开环精度。",
    "不同控制器的在线RMSE来自各自访问的状态/命令，不能直接当作同查询模型精度比较；共同验证窗口才提供匹配输入的精度对照。",
    "未来外力不交给预测器，推扰宏步与普通宏步分别报告；预测更准不自动等于完整任务更好。",
    "200ms判断使用每次完整控制周期的p95、最大值与超期比例；平均时间不能代替它们。",
    "本机/服务器观察到的耗时不构成NX时延、实机控制频率或硬实时保证。",
    "最终成功按真实MuJoCo轨迹、全身几何、接触及时间戳保持判断，不借用紧凑预测的未来接触或关节状态。",
    "两个学习模型与kinematic使用同一航向矩形proxy；exact policy rollout仍拥有完整名义物理/几何，属于额外信息的特权对照。",
    "学习模型共同零输出先验为当前yaw的Euler位移，旧kinematic使用midpoint yaw，两者不被混称为完全相同先验。",
    "失败或算力截断全部保留；早停失败的小累计成本不用于宣称胜过成功任务。",
    "只有少量确定性开发轨迹，不做统计显著性或总体鲁棒性结论。",
]


def basename(value):
    return PurePosixPath(str(value).replace("\\", "/")).name


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def latency_stats(values, deadline):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean_seconds": None, "p95_seconds": None,
                "max_seconds": None, "deadline_miss_count": 0,
                "deadline_miss_fraction": None, "all_observed_within_deadline": None}
    misses = values > deadline
    return {"count": len(values), "mean_seconds": float(values.mean()),
            "p50_seconds": float(np.median(values)), "p95_seconds": float(np.quantile(values, .95)),
            "max_seconds": float(values.max()), "deadline_seconds": deadline,
            "deadline_miss_count": int(misses.sum()), "deadline_miss_fraction": float(misses.mean()),
            "all_observed_within_deadline": bool(not misses.any()),
            "samples_seconds": values.tolist()}


def public_fit(fit):
    """Publish scientific metrics/counts; omit remote directories and commands."""
    dataset = fit.get("dataset", {})
    metadata = dataset.get("metadata", {})
    episodes = {}
    for split in ("training", "validation"):
        episodes[split] = [{key: (basename(item[key]) if key == "file" else item[key])
                            for key in ("file", "seed", "split", "transitions", "sha256", "truncated", "fallen",
                                        "seconds", "obstacle_contact_steps", "nonfoot_ground_contact_steps",
                                        "physics_steps_including_settle", "policy_calls_including_settle") if key in item}
                           for item in dataset.get(split + "_episodes", [])]
    train_ids = {row.get("sha256") for row in episodes["training"] if row.get("sha256")}
    validation_ids = {row.get("sha256") for row in episodes["validation"] if row.get("sha256")}
    arms = {}
    for name, arm in fit.get("arms", {}).items():
        arms[name] = {key: arm[key] for key in (
            "checkpoint_sha256", "model_optimizer_steps", "policy_optimizer_steps", "fit_wall_seconds",
            "validation", "work", "last_train_loss", "training_windows", "validation_windows",
            "short_windows", "masked_windows", "loss_definition", "model_parameters", "parameter_count",
            "checkpoint_bytes", "initial_state_sha256", "final_state_sha256", "validation_wall_seconds", "arm_wall_seconds",
        ) if key in arm}
        if "checkpoint" in arm:
            arms[name]["checkpoint"] = basename(arm["checkpoint"])
    return {
        "status": fit.get("status", "UNKNOWN"), "config": fit.get("config", {}),
        "dataset": {
            "manifest_sha256": dataset.get("manifest_sha256"),
            "training_episodes": episodes["training"], "validation_episodes": episodes["validation"],
            "training_transitions": dataset.get("training_transitions"),
            "validation_transitions": dataset.get("validation_transitions"),
            "policy_sha256": dataset.get("policy_sha256"),
            "collection": {key: metadata.get(key) for key in (
                "status", "actual_transitions", "wall_seconds", "cpu_seconds", "policy_updates",
                "response_model_updates", "policy_sha256_before", "policy_sha256_after",
                "future_force_available", "scene_changed")},
            "episode_hash_overlap": sorted(train_ids & validation_ids),
            "episode_hash_disjointness_check_available": bool(train_ids and validation_ids),
            "validation_scope": "new nominal episodes in the same declared scene/route distribution",
        },
        "arms": arms,
        **{key: fit.get(key) for key in (
            "shared_initial_state_sha256", "parameter_count", "zero_head_prior", "shared_batch_indices_sha256",
            "normalization_source", "resource_settings", "protocol_sha256", "source_identity", "protocol")},
        **{key: fit.get(key) for key in (
            "total_model_optimizer_steps", "total_policy_optimizer_steps", "new_environment_steps",
            "total_fit_wall_seconds", "policy_sha256_before", "policy_sha256_after", "policy_inferences")},
    }


def inspect_case(folder, config, policy_dt):
    case, macros, traces = (read(folder / name) for name in ("summary.json", "macro.json", "trace.json"))
    if not macros or not traces:
        return {"method": case["method"], "condition": case["condition"], "success": False,
                "reason": case.get("reason", "NO_TRAJECTORY"), "evidence_complete": False}, {
                    "issue": "NO_MACRO_OR_TRACE", "case_status": case.get("reason")}
    actual = np.stack([feature(row["state"]) for row in macros])
    predicted = np.asarray([row["predicted_feature"] for row in macros], dtype=float)
    error = predicted - actual
    error[:, 2] = wrap(error[:, 2])
    pushed = np.array([row.get("force_world") is not None and np.linalg.norm(row["force_world"]) > 0
                       for row in macros])
    scopes = {"all": np.ones(len(macros), dtype=bool), "push_macro": pushed, "nonpush_macros": ~pushed}
    pulse = np.flatnonzero(pushed)
    if len(pulse):
        scopes.update(prepush_macros=np.arange(len(macros)) < pulse[0],
                      postpush_macros=np.arange(len(macros)) > pulse[-1])
    prediction = {name: prediction_statistics(error, mask) for name, mask in scopes.items()}
    success = verify_success(traces, macros, case, config, policy_dt)
    latency = {name: latency_stats([row[key] for row in macros], config["macro_dt"])
               for name, key in (("planner", "planner_seconds"), ("end_to_end", "end_to_end_seconds"))}
    expected_physics = len(macros) * round(config["macro_dt"] / config["physics_dt"])
    expected_policy = len(macros) * round(config["macro_dt"] / policy_dt)
    checks = {
        "success_agrees": success["success_agrees"],
        "physical_steps_agree": case.get("executed_physics_steps") == expected_physics,
        "policy_calls_agree": case.get("executed_policy_calls") == expected_policy,
        "trace_frames_agree": len(traces) == expected_policy,
        "task_time_error_seconds": abs(len(macros)*config["macro_dt"] - case["task_seconds"]),
        "task_cost_sum_error": abs(sum(row["task_cost"] for row in macros) - case["task_cost"]),
        "torque_integral_sum_error": abs(sum(row["torque_squared_integral"] for row in macros) - case["torque_squared_integral"]),
        "mechanical_work_sum_error": abs(sum(row["absolute_work"] for row in macros) - case["absolute_mechanical_work"]),
        "push_macro_count": int(pushed.sum()),
    }
    if "prediction_feature_rmse" in case:
        checks["reported_prediction_rmse_max_error"] = float(np.max(np.abs(
            prediction["all"]["rmse"] - np.asarray(case["prediction_feature_rmse"]))))
    names = (
        "method", "condition", "success", "reason", "task_seconds", "task_cost", "torque_squared_integral",
        "absolute_mechanical_work", "wall_seconds", "final_position_error", "final_yaw_error", "final_speed",
        "terminal_hold_seconds", "executed_physics_steps", "executed_policy_calls", "planning_candidates",
        "prediction_physics_steps", "prediction_policy_calls", "training_updates", "policy_optimizer_steps",
        "model_optimizer_steps", "prediction_model_steps", "prediction_model_calls", "predictor_calls",
        "compact_response_steps", "compact_encoder_calls", "response_checkpoint_sha256",
    )
    row = {key: case.get(key) for key in names}
    row.update(
        evidence_complete=True, success_recomputed=success["recomputed_success"],
        macro_count=len(macros), timestamp_terminal_hold_seconds=success["final_hold_seconds"],
        compute_budget_truncated=case.get("reason") == "compute_budget_exceeded",
        position_prediction_rmse_m=prediction["all"]["position_vector_rmse_m"],
        velocity_prediction_rmse_m_s=prediction["all"]["planar_velocity_vector_rmse_m_s"],
        yaw_prediction_rmse_rad=prediction["all"]["rmse"][2],
        yaw_rate_prediction_rmse_rad_s=prediction["all"]["rmse"][5],
        planner_p95_seconds=latency["planner"]["p95_seconds"],
        planner_max_seconds=latency["planner"]["max_seconds"],
        end_to_end_mean_seconds=latency["end_to_end"]["mean_seconds"],
        end_to_end_p95_seconds=latency["end_to_end"]["p95_seconds"],
        end_to_end_max_seconds=latency["end_to_end"]["max_seconds"],
        deadline_miss_fraction=latency["end_to_end"]["deadline_miss_fraction"],
        deadline_miss_macro_count=latency["end_to_end"]["deadline_miss_count"],
        observed_all_macros_within_deadline=latency["end_to_end"]["all_observed_within_deadline"],
    )
    mechanism = {"method": case["method"], "condition": case["condition"], "feature_order": FEATURES,
                 "prediction": prediction, "success_evidence": success, "latency": latency,
                 "consistency": checks, "push_event": case.get("push_event"),
                 "prediction_scope": "selected-command next 0.2s, not all candidates or five-step validation",
                 "macro_count": len(macros)}
    return row, mechanism


def compare(left, right):
    names = ("task_seconds", "task_cost", "torque_squared_integral", "absolute_mechanical_work",
             "position_prediction_rmse_m", "velocity_prediction_rmse_m_s", "yaw_prediction_rmse_rad",
             "end_to_end_p95_seconds", "end_to_end_max_seconds", "deadline_miss_fraction")
    both_success = left.get("success_recomputed") and right.get("success_recomputed")
    comparable_cost = bool(both_success and not left.get("compute_budget_truncated") and not right.get("compute_budget_truncated"))
    delta = {key: left[key]-right[key] for key in names if finite_number(left.get(key)) and finite_number(right.get(key))}
    percent = {key: 100*(left[key]/right[key]-1) if right[key] else None for key in delta}
    return {"left": left["method"], "right": right["method"], "condition": left["condition"],
            "success": [left.get("success_recomputed"), right.get("success_recomputed")],
            "both_successful_cost_comparison": comparable_cost,
            "left_minus_right": delta, "change_percent": percent,
            "interpretation": "descriptive known-condition development comparison; no statistical generalization claim"}


def execution_ledger(rows, dataset, config, policy_dt, fit, evaluation):
    episodes = dataset.get("episodes", [])
    collection_physics = sum(e["physics_steps_including_settle"] for e in episodes)
    collection_policy = sum(e["policy_calls_including_settle"] for e in episodes)
    task_physics = sum(row.get("executed_physics_steps") or 0 for row in rows)
    task_policy = sum(row.get("executed_policy_calls") or 0 for row in rows)
    predicted_physics = sum(row.get("prediction_physics_steps") or 0 for row in rows)
    predicted_policy = sum(row.get("prediction_policy_calls") or 0 for row in rows)
    settling_physics = len(rows) * round(config["settle_seconds"] / config["physics_dt"])
    settling_policy = len(rows) * round(config["settle_seconds"] / policy_dt)
    walls = [dataset.get("wall_seconds"), fit.get("total_fit_wall_seconds"), evaluation.get("wall_seconds")]
    return {
        "collection_episodes": len(episodes), "collection_macro_transitions": dataset.get("actual_transitions"),
        "collection_physics_steps_including_settle": collection_physics,
        "collection_policy_calls_including_settle": collection_policy,
        "evaluation_completed_case_count": len(rows),
        "evaluation_task_physics_steps": task_physics, "evaluation_task_policy_calls": task_policy,
        "evaluation_reset_settling_physics_steps": settling_physics,
        "evaluation_reset_settling_policy_calls": settling_policy,
        "evaluation_prediction_physics_steps": predicted_physics,
        "evaluation_prediction_policy_calls": predicted_policy,
        "real_execution_physics_steps_including_all_settle": collection_physics + task_physics + settling_physics,
        "all_physics_steps_including_prediction": collection_physics + task_physics + settling_physics + predicted_physics,
        "all_policy_calls_including_prediction": collection_policy + task_policy + settling_policy + predicted_policy,
        "compact_response_steps_during_evaluation": sum(row.get("compact_response_steps") or 0 for row in rows),
        "compact_encoder_calls_during_evaluation": sum(row.get("compact_encoder_calls") or 0 for row in rows),
        "model_optimizer_steps": fit.get("total_model_optimizer_steps"),
        "policy_optimizer_steps": fit.get("total_policy_optimizer_steps"),
        "collection_wall_seconds": dataset.get("wall_seconds"),
        "fit_wall_seconds": fit.get("total_fit_wall_seconds"),
        "evaluation_wall_seconds": evaluation.get("wall_seconds"),
        "sum_recorded_stage_wall_seconds": sum(walls) if all(finite_number(v) for v in walls) else None,
        "scope": "New collection, model fitting and twelve completed task cases only; prior frozen-study work is not rerun or charged twice. Installation/transfer/setup wall time is outside the recorded stage sum.",
    }


def compare_old_baselines(rows, old_summary):
    old = {(row["method"], row["condition"]): row for row in old_summary.get("cases", [])}
    keys = ("task_seconds", "task_cost", "torque_squared_integral", "absolute_mechanical_work",
            "final_position_error", "final_yaw_error", "final_speed", "executed_physics_steps", "executed_policy_calls")
    checks = []
    for row in rows:
        key = row["method"], row["condition"]
        if key[0] not in ("kinematic_mpc", "policy_rollout_mpc") or key not in old:
            continue
        previous = old[key]
        differences = {field: row[field] - previous[field] for field in keys
                       if finite_number(row.get(field)) and finite_number(previous.get(field))}
        hold_difference = row.get("timestamp_terminal_hold_seconds", 0.) - previous.get("timestamp_terminal_hold_seconds", 0.)
        checks.append({"method": key[0], "condition": key[1], "differences": differences,
                       "timestamp_hold_difference_seconds": hold_difference,
                       "success_agrees": row.get("success_recomputed") == previous.get("success_recomputed", previous.get("success")),
                       "within_1e_8": all(abs(value) <= 1e-8 for value in [*differences.values(), hold_difference]),
                       "latency_excluded": True})
    return {"matched_cases": len(checks), "checks": checks,
            "all_matched_within_1e_8": bool(checks) and all(c["within_1e_8"] and c["success_agrees"] for c in checks),
            "scope": "Task/physical outputs only; new runtime timings are independently retained"}


def fmt(value):
    return "—" if value is None else (f"{value:.6g}" if finite_number(value) else str(value))


def report(data):
    fit, rows = data["model_fit"], data["cases"]
    lines = ["# Go2事实紧凑预测：模型拟合与连续任务结果", "",
             f"证据状态：`{data['status']}`。已保存case数：{len(rows)}；预定12个；缺失：{data['missing_cases']}。",
             "",
             "同一已知scene和三种已知条件的开发比较。模型拟合与冻结策略更新分别计账；本报告不声称新任务泛化。",
             "", "## 模型拟合与策略冻结", "",
             f"拟合状态：`{fit['status']}`；模型优化步：{fit['total_model_optimizer_steps']}；"
             f"策略优化步：{fit['total_policy_optimizer_steps']}；拟合墙钟：{fmt(fit['total_fit_wall_seconds'])} s。",
             f"训练转移：{fit['dataset']['training_transitions']}；验证转移：{fit['dataset']['validation_transitions']}；"
             f"训练/验证episode哈希交集：{fit['dataset']['episode_hash_overlap']}。",
             "", "| 模型 | 模型更新 | 策略更新 | 拟合wall s | 验证窗口 |", "|---|---:|---:|---:|---:|"]
    for name, arm in fit["arms"].items():
        lines.append(f"| {name} | {arm.get('model_optimizer_steps')} | {arm.get('policy_optimizer_steps')} | "
                     f"{fmt(arm.get('fit_wall_seconds'))} | {arm.get('validation',{}).get('windows')} |")
    lines += ["", "| 模型 | 参数量 | checkpoint字节 | 初值与共同hash一致 | 末参数相对初值改变 |",
              "|---|---:|---:|---|---|"]
    for name, arm in fit["arms"].items():
        initial = arm.get("initial_state_sha256")
        lines.append(f"| {name} | {arm.get('parameter_count')} | {arm.get('checkpoint_bytes')} | "
                     f"{initial == fit.get('shared_initial_state_sha256') if initial else None} | "
                     f"{initial != arm.get('final_state_sha256') if initial and arm.get('final_state_sha256') else None} |")
    lines += ["", "上述参数身份来自拟合记录的逐tensor哈希；评价加载的checkpoint哈希另外核对。",
              "", "| 模型 | 验证口径 | 有效标签 | 位置RMSE m | yaw RMSE rad | body速度RMSE m/s | yaw-rate RMSE rad/s |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name, arm in fit["arms"].items():
        for scope in ("one_step", "autoregressive_h5"):
            item = arm.get("validation", {}).get(scope, {})
            lines.append(f"| {name} | {scope} | {item.get('labels')} | {fmt(item.get('position_rmse_m'))} | "
                         f"{fmt(item.get('yaw_rmse_rad'))} | {fmt(item.get('body_velocity_rmse_m_s'))} | "
                         f"{fmt(item.get('yaw_rate_rmse_rad_s'))} |")
    lines += ["", "验证集单步与五步递推误差、逐时域误差、模型hash和实际前反向计数保存在summary.json。"
              "单步和多步模型更新次数相同不代表计算量相同。", "",
              "验证速度口径是body平面速度；下面在线selected-command误差是world平面速度，不能直接相减。", "",
              "## 完整任务与200ms周期", "",
              "| 方法 | 条件 | 实际成功 | 原因 | 任务秒 | 成本 | 位置RMSE m | 速度RMSE m/s | 总周期p95 s | 总周期max s | 超期数/宏步 |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['condition']} | {row.get('success_recomputed')} | {row.get('reason')} | "
                     f"{fmt(row.get('task_seconds'))} | {fmt(row.get('task_cost'))} | "
                     f"{fmt(row.get('position_prediction_rmse_m'))} | {fmt(row.get('velocity_prediction_rmse_m_s'))} | "
                     f"{fmt(row.get('end_to_end_p95_seconds'))} | {fmt(row.get('end_to_end_max_seconds'))} | "
                     f"{row.get('deadline_miss_macro_count')}/{row.get('macro_count')} |")
    lines += ["", "只有所有已观察周期都≤0.2 s，才能说这次观测未超期；p95较低但最大值超期仍需报告。"
              "这不是硬实时或NX测量。", "",
              "## 真实资源与终点", "",
              r"| 方法 | 条件 | ∫τ²dt | ∫\|τdq\|dt | 终点位置误差m | 终点yaw误差rad | 终点速度m/s | 严格保持秒 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['condition']} | {fmt(row.get('torque_squared_integral'))} | "
                     f"{fmt(row.get('absolute_mechanical_work'))} | {fmt(row.get('final_position_error'))} | "
                     f"{fmt(row.get('final_yaw_error'))} | {fmt(row.get('final_speed'))} | "
                     f"{fmt(row.get('timestamp_terminal_hold_seconds'))} |")
    lines += ["", "## compact_multistep相对对照", "",
              "| 对照 | 条件 | 双方成功 | 成本变化% | 时间变化% | p95周期变化% | 最大周期变化% |",
              "|---|---|---|---:|---:|---:|---:|"]
    for pair in data["comparisons"]:
        change = pair["change_percent"]
        lines.append(f"| {pair['right']} | {pair['condition']} | {pair['both_successful_cost_comparison']} | "
                     f"{fmt(change.get('task_cost'))} | {fmt(change.get('task_seconds'))} | "
                     f"{fmt(change.get('end_to_end_p95_seconds'))} | {fmt(change.get('end_to_end_max_seconds'))} |")
    lines += ["", "早停失败或算力截断的累计成本仅描述已发生代价，不与成功完整任务排序。",
              "", "## 评价调用计数", "",
              "| 方法 | 条件 | 实际物理步 | 实际policy前向 | 预测物理步 | 预测policy前向 | compact encoder调用 | compact response步 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['condition']} | {fmt(row.get('executed_physics_steps'))} | "
                     f"{fmt(row.get('executed_policy_calls'))} | {fmt(row.get('prediction_physics_steps'))} | "
                     f"{fmt(row.get('prediction_policy_calls'))} | {fmt(row.get('compact_encoder_calls'))} | "
                     f"{fmt(row.get('compact_response_steps'))} |")
    lines += ["", "拟合产生的模型更新与这里的冻结预测调用分开；计数不是等价FLOPs。",
              "", "## 本轮完整计算账", "", "| 项目 | 实际记录/按固定启动规则换算 |", "|---|---:|"]
    for key, value in data["complete_execution_ledger"].items():
        if key != "scope":
            lines.append(f"| {key} | {fmt(value)} |")
    old = data.get("old_baseline_reproduction")
    if old:
        lines += ["", f"与上一轮最终基线核对：{old['matched_cases']}条case；任务和物理输出在1e-8内一致："
                  f"{old['all_matched_within_1e_8']}。旧耗时不复用；本轮耗时独立报告。"]
    lines += ["", "数据采集的每episode计数已包含其启动稳定；任务评价另计12次启动，避免漏算或重复相加。",
              "", "## 证据边界", ""]
    lines += ["- " + item for item in LIMITS]
    lines += ["", f"数据一致性待核项：{len(data['issues'])}。"]
    lines += ["- " + item for item in data["issues"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="New downloaded root containing fit/ and evaluation/")
    parser.add_argument("--output", type=Path, required=True, help="New public result directory")
    parser.add_argument("--baseline-summary", type=Path,
                        default=Path(__file__).resolve().parents[1] / "notes/go2-frozen-2026-10-19/summary.json")
    args = parser.parse_args()
    old = Path(__file__).resolve().parents[1] / "notes" / "go2-frozen-2026-10-19"
    if args.output.resolve() == old.resolve() or args.output.resolve().is_relative_to(old.resolve()):
        parser.error("Do not overwrite the previous frozen-policy study")
    root, evaluation = args.input.resolve(), args.input.resolve() / "evaluation"
    fit_path = root / "fit" / "fit-summary.json"
    raw_fit, config, execution = read(fit_path), read(evaluation / "protocol.json"), read(evaluation / "execution.json")
    dataset_path = root / "data" / "dataset.json"
    dataset = read(dataset_path)
    fit = public_fit(raw_fit)
    policy_dt = execution["simulation"]["policy"]["policy_period_seconds"]
    rows, mechanisms, manifest, issues = [], [], [], []
    for path in (fit_path, dataset_path, evaluation / "protocol.json", evaluation / "execution.json"):
        manifest.append({"file": path.relative_to(root).as_posix(), "sha256": sha(path)})
    for method in METHODS:
        for condition in CONDITIONS:
            folder = evaluation / (method + "--" + condition)
            if not (folder / "summary.json").exists():
                continue
            row, mechanism = inspect_case(folder, config, policy_dt)
            rows.append(row); mechanisms.append(mechanism)
            for name in ("summary.json", "macro.json", "trace.json"):
                manifest.append({"file": (folder / name).relative_to(root).as_posix(), "sha256": sha(folder / name)})
            if row.get("evidence_complete"):
                checks = mechanism["consistency"]
                if any(value is False for key, value in checks.items() if key.endswith("agrees") or key.endswith("agree")):
                    issues.append(f"count_or_success_mismatch:{method}:{condition}")
                if any(value > 1e-7 for key, value in checks.items() if key.endswith("error") or key.endswith("error_seconds")):
                    issues.append(f"arithmetic_mismatch:{method}:{condition}")
                for key in ("training_updates", "policy_optimizer_steps"):
                    if row.get(key) not in (None, 0):
                        issues.append(f"policy_update_in_evaluation:{method}:{condition}:{key}")
    index = {(row["method"], row["condition"]): row for row in rows}
    missing = [f"{m}--{c}" for m in METHODS for c in CONDITIONS if (m,c) not in index]
    if fit["dataset"]["episode_hash_overlap"]:
        issues.append("train_validation_episode_overlap")
    if sha(dataset_path) != fit["dataset"].get("manifest_sha256"):
        issues.append("downloaded_dataset_manifest_differs_from_fit_source")
    if dataset.get("policy_sha256_before") != dataset.get("policy_sha256_after"):
        issues.append("policy_hash_changed_during_collection")
    if dataset.get("policy_updates") != 0:
        issues.append("collection_policy_updates_not_verified_zero")
    if fit["total_policy_optimizer_steps"] != 0:
        issues.append("model_fit_policy_updates_not_verified_zero")
    recorded_model_updates = sum(arm.get("model_optimizer_steps", 0) for arm in fit["arms"].values())
    if recorded_model_updates != fit["total_model_optimizer_steps"]:
        issues.append("fit_total_model_updates_mismatch")
    for name, arm in fit["arms"].items():
        if arm.get("policy_optimizer_steps") != 0:
            issues.append(f"fit_arm_policy_updates_not_verified_zero:{name}")
        if arm.get("initial_state_sha256") != fit.get("shared_initial_state_sha256"):
            issues.append(f"fit_initial_parameter_identity_mismatch:{name}")
        checkpoint = root / "fit" / arm.get("checkpoint", "")
        if checkpoint.is_file():
            checkpoint_hash = sha(checkpoint)
            manifest.append({"file": checkpoint.relative_to(root).as_posix(), "sha256": checkpoint_hash})
            if checkpoint_hash != arm.get("checkpoint_sha256"):
                issues.append(f"downloaded_response_checkpoint_hash_mismatch:{name}")
    for row in rows:
        if row["method"] in fit["arms"]:
            if row.get("response_checkpoint_sha256") != fit["arms"][row["method"]].get("checkpoint_sha256"):
                issues.append(f"evaluation_response_checkpoint_hash_mismatch:{row['method']}:{row['condition']}")
    for key in ("policy_updates", "response_model_updates_during_evaluation"):
        if execution.get(key) != 0:
            issues.append(f"evaluation_{key}_not_verified_zero")
    before, after = execution.get("policy_sha256_before"), execution.get("policy_sha256_after")
    policy_unchanged = before == after if before is not None and after is not None else None
    if policy_unchanged is not True:
        issues.append("evaluation_policy_hash_unchanged_not_verified")
    if fit["dataset"].get("policy_sha256") != before:
        issues.append("fit_and_evaluation_frozen_policy_hash_mismatch")
    comparisons = [compare(index[("compact_multistep", condition)], index[(reference, condition)])
                   for condition in CONDITIONS for reference in ("id_one_step", "kinematic_mpc", "policy_rollout_mpc")
                   if ("compact_multistep", condition) in index and (reference, condition) in index]
    counts = {}
    for key in ("executed_physics_steps", "executed_policy_calls", "planning_candidates", "prediction_physics_steps",
                "prediction_policy_calls", "training_updates", "policy_optimizer_steps", "model_optimizer_steps",
                "prediction_model_steps", "prediction_model_calls", "predictor_calls",
                "compact_response_steps", "compact_encoder_calls"):
        present = [row[key] for row in rows if finite_number(row.get(key))]
        counts[key] = {"sum_recorded": sum(present) if present else None, "cases_reporting": len(present)}
    ledger = execution_ledger(rows, dataset, config, policy_dt, raw_fit, execution)
    old_check = compare_old_baselines(rows, read(args.baseline_summary)) if args.baseline_summary.exists() else None
    if old_check is not None:
        old_check["reference_summary_sha256"] = sha(args.baseline_summary)
    if old_check and not old_check["all_matched_within_1e_8"]:
        issues.append("old_fixed_baseline_task_outputs_differ")
    data = {"status": "COMPLETE_TWELVE_CASE_EVIDENCE" if not missing and not issues else "PARTIAL_OR_REVIEW_REQUIRED",
            "model_fit": fit, "evaluation_process_status": execution.get("status"), "cases": rows,
            "mechanism": mechanisms, "comparisons": comparisons, "missing_cases": missing, "issues": issues,
            "policy_sha256_before": before, "policy_sha256_after": after, "policy_hash_unchanged": policy_unchanged,
            "evaluation_work": counts, "evaluation_wall_seconds": execution.get("wall_seconds"),
            "evaluation_process_cpu_seconds": execution.get("process_cpu_seconds"),
            "policy_updates_during_evaluation": execution.get("policy_updates"),
            "response_model_updates_during_evaluation": execution.get("response_model_updates_during_evaluation"),
            "complete_execution_ledger": ledger, "old_baseline_reproduction": old_check,
            "protocol": config, "source_manifest": manifest, "limitations": LIMITS,
            "fit_updates_are_not_policy_updates": True, "old_research_results_overwritten": False}
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "summary.json", data)
    write(args.output / "mechanism.json", mechanisms)
    with (args.output / "cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(plain(rows))
    (args.output / "report.md").write_text(report(data), encoding="utf-8")
    print(json.dumps({"status": data["status"], "cases": len(rows), "issues": issues,
                      "policy_hash_unchanged": policy_unchanged, "missing_cases": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
