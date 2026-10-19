"""Read downloaded task-response logs and publish compact, attributable evidence.

This module uses only the Python standard library.  It never imports the learner,
loads checkpoints, connects to a server, reruns an environment, or trains a model.
Run completion is separate from task success.  Completed failed trajectories
remain in task-cost means; truncated trajectories are never silently imputed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


COMPLETE = "COMPLETED_CONFIGURED_BUDGET"
VECTOR_FIELDS = (
    "success", "completed_horizon", "evaluation_truncated", "endpoint_conditions_met",
    "first_reach_time", "terminal_position_error", "terminal_velocity_error",
    "corridor_violation_count", "maximum_corridor_excess", "max_applied_input_norm",
    "max_requested_input_norm", "normalized_input_integral", "task_cost",
)
REQUIRED_NUMBERS = (
    "task_cost", "normalized_input_integral", "terminal_position_error",
    "terminal_velocity_error", "corridor_violation_count", "maximum_corridor_excess",
    "max_applied_input_norm", "max_requested_input_norm",
)
SCALARS = (
    "success_rate", "task_cost_mean", "normalized_input_integral_mean",
    "corridor_violation_case_rate", "corridor_violation_sample_count",
    "terminal_position_error_mean", "terminal_velocity_error_mean",
    "maximum_corridor_excess", "maximum_positive_corridor_excess",
)
INTERPRETATION = [
    "运行完成状态不是任务成功；完整失败轨迹保留在任务成本、输入和误差统计中。",
    "主比较仅用全部预定arm共同完成且交互/更新预算核对通过的训练seed；截断与缺失另列。",
    "匹配物理交互和更新次数不等于同总计算或同墙钟；敏感度、规划与拟合开销单列。",
    "三个训练seed只描述本次小试，不进行显著性或总体优势推断。",
    "四组差异属于匹配设置下的观测性消融，不是完整因果归因或独立算法创新证明。",
    "日志没有同条件重复响应分布；不将跨状态、任务或训练seed的离散程度称为条件响应方差。",
    "固定评价case在训练seed之间复用；不能把重复case数当成独立场景数量。",
    "round0输出一致性检查不等于初始权重、Adam状态或全部轨迹相同的独立证明。",
    "模型拟合损失与训练shaping只作过程诊断，不能替代真实完整任务指标。",
]


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mean(values):
    return statistics.mean(values) if values else None


def stats(values):
    available = [value for value in values if numeric(value)]
    return {"values": values, "n": len(available), "unavailable": len(values) - len(available),
            "mean": mean(available),
            "sample_sd": statistics.stdev(available) if len(available) > 1 else None,
            "minimum": min(available) if available else None,
            "maximum": max(available) if available else None}


def sum_work(records):
    result = defaultdict(float)
    for record in records:
        for key, value in record.get("work", {}).items():
            if numeric(value):
                result[key] += value
    return {key: int(value) if float(value).is_integer() else value
            for key, value in sorted(result.items())}


def work_difference(left, right):
    return {key: {"record_sum": left.get(key, 0), "reported": right.get(key, 0)}
            for key in sorted(set(left) | set(right))
            if not math.isclose(left.get(key, 0), right.get(key, 0), rel_tol=1e-9, abs_tol=1e-8)}


def read_stages(path):
    if not path.exists():
        return []
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Incomplete or invalid JSONL: {path.name}, line {line_no}; "
                             "download a stable log before summarizing") from error
    return records


def aggregate_cases(cases):
    full = [case for case in cases if case["full_valid"]]
    complete_set = bool(cases) and len(full) == len(cases)
    n = len(full)
    reached = [case["first_reach_time"] for case in full if numeric(case["first_reach_time"])]
    result = {
        "cases": len(cases), "full_valid_cases": n,
        "truncated_cases": sum(case["evaluation_truncated"] for case in cases),
        "invalid_metric_cases": sum(bool(case["invalid_numeric_fields"]) for case in cases),
        "complete_case_set": complete_set,
        "success_count": sum(case["success"] for case in full),
        "success_rate": sum(case["success"] for case in full) / n if complete_set else None,
        "success_rate_complete_subset": sum(case["success"] for case in full) / n if n else None,
        "task_cost_mean": mean([case["task_cost"] for case in full]),
        "task_cost_includes_complete_failed_cases": True,
        "normalized_input_integral_mean": mean([case["normalized_input_integral"] for case in full]),
        "terminal_position_error_mean": mean([case["terminal_position_error"] for case in full]),
        "terminal_velocity_error_mean": mean([case["terminal_velocity_error"] for case in full]),
        "corridor_violation_case_count": sum(case["corridor_violation_count"] > 0 for case in full),
        "corridor_violation_case_rate": (sum(case["corridor_violation_count"] > 0 for case in full) / n) if n else None,
        "corridor_violation_sample_count": sum(case["corridor_violation_count"] for case in full),
        "maximum_corridor_excess": max((case["maximum_corridor_excess"] for case in full), default=None),
        "maximum_positive_corridor_excess": max((max(0., case["maximum_corridor_excess"]) for case in full), default=None),
        "maximum_applied_input_norm": max((case["max_applied_input_norm"] for case in full), default=None),
        "first_reach_cases": len(reached), "first_reach_time_mean_reached_only": mean(reached),
    }
    return result


def normalize_evaluations(evaluations, expected_n):
    checkpoints = {}
    for round_text, items in evaluations.items():
        round_number = int(round_text)
        cases, batches, seen = [], [], set()
        for details in items:
            eval_seed, planner = int(details["evaluation_seed"]), details["planner"]
            identity = (eval_seed, planner)
            if identity in seen:
                raise ValueError(f"Duplicate evaluation batch at round {round_number}: {identity}")
            seen.add(identity)
            if details.get("deterministic") is not True or details.get("training_shaping_disabled") is not True:
                raise ValueError("Primary evaluation must be deterministic with training shaping disabled")
            metrics = details["task_metrics"]
            for key in VECTOR_FIELDS:
                if not isinstance(metrics.get(key), list) or len(metrics[key]) != expected_n:
                    raise ValueError(f"Evaluation {round_number}/{eval_seed}: {key} must have {expected_n} cases")
            for index in range(expected_n):
                case = {key: metrics[key][index] for key in VECTOR_FIELDS}
                invalid = [key for key in REQUIRED_NUMBERS if not numeric(case[key])]
                for key in ("success", "completed_horizon", "evaluation_truncated", "endpoint_conditions_met"):
                    if not isinstance(case[key], bool):
                        raise ValueError(f"Evaluation flag {key} is not Boolean")
                if case["completed_horizon"] == case["evaluation_truncated"]:
                    raise ValueError("Completed-horizon and truncated flags disagree")
                if case["success"] and (not case["completed_horizon"] or not case["endpoint_conditions_met"]):
                    raise ValueError("Task success is inconsistent with endpoint/horizon flags")
                case.update(evaluation_seed=eval_seed, environment_index=index, planner=planner,
                            task="go" if index < expected_n // 2 else "stop",
                            invalid_numeric_fields=invalid,
                            full_valid=case["completed_horizon"] and not invalid)
                cases.append(case)
            batches.append({"evaluation_seed": eval_seed, "planner": planner,
                            "deterministic": True, "training_shaping_disabled": True,
                            "planner_batch_latency_seconds": {
                                key: metrics.get(key) for key in
                                ("planner_batch_latency_p50_seconds", "planner_batch_latency_p95_seconds",
                                 "planner_batch_latency_max_seconds")}})
        checkpoints[str(round_number)] = {
            "round": round_number, "batches": batches, "cases": cases,
            "scopes": {scope: aggregate_cases([case for case in cases if scope == "all" or case["task"] == scope])
                       for scope in ("all", "go", "stop")},
        }
    return checkpoints


def case_key(case):
    return case["evaluation_seed"], case["environment_index"], case["planner"]


def compare_checkpoints(left, right):
    """left minus right, preserving exact matching of evaluation conditions."""
    left_cases = {case_key(case): case for case in left["cases"]}
    right_cases = {case_key(case): case for case in right["cases"]}
    if left_cases.keys() != right_cases.keys():
        return {"status": "UNMATCHED_EVALUATION_CASES"}
    if not all(case["full_valid"] for case in [*left_cases.values(), *right_cases.values()]):
        return {"status": "INCOMPLETE_OR_INVALID_EVALUATION"}
    differences = []
    for key in sorted(left_cases):
        a, b = left_cases[key], right_cases[key]
        differences.append({"evaluation_seed": key[0], "environment_index": key[1], "planner": key[2],
                            "task": a["task"], "task_cost_delta": a["task_cost"] - b["task_cost"],
                            "success_delta": int(a["success"]) - int(b["success"]),
                            "normalized_input_integral_delta": a["normalized_input_integral"] - b["normalized_input_integral"],
                            "terminal_position_error_delta": a["terminal_position_error"] - b["terminal_position_error"],
                            "terminal_velocity_error_delta": a["terminal_velocity_error"] - b["terminal_velocity_error"],
                            "corridor_violation_sample_delta": a["corridor_violation_count"] - b["corridor_violation_count"]})
    scopes = {}
    for scope in ("all", "go", "stop"):
        rows = [row for row in differences if scope == "all" or row["task"] == scope]
        base_cost = right["scopes"][scope]["task_cost_mean"]
        cost_delta = mean([row["task_cost_delta"] for row in rows])
        scopes[scope] = {
            "cases": len(rows), "task_cost_delta": cost_delta,
            "task_cost_change_percent": 100 * cost_delta / base_cost if base_cost else None,
            "success_change_percentage_points": 100 * mean([row["success_delta"] for row in rows]),
            "normalized_input_integral_delta": mean([row["normalized_input_integral_delta"] for row in rows]),
            "terminal_position_error_delta": mean([row["terminal_position_error_delta"] for row in rows]),
            "terminal_velocity_error_delta": mean([row["terminal_velocity_error_delta"] for row in rows]),
            "corridor_violation_sample_delta": sum(row["corridor_violation_sample_delta"] for row in rows),
            "lower_cost_cases": sum(row["task_cost_delta"] < 0 for row in rows),
            "higher_cost_cases": sum(row["task_cost_delta"] > 0 for row in rows),
        }
    return {"status": "MATCHED_FULL_CASES", "direction": "left_minus_right",
            "scopes": scopes, "per_case_differences": differences}


def round_zero_check(runs, seed, arms, atol, rtol):
    available = {arm: runs[(seed, arm)]["checkpoints"].get("0") for arm in arms}
    missing = [arm for arm, checkpoint in available.items() if checkpoint is None]
    if missing:
        return {"status": "NOT_VERIFIABLE", "missing_arms": missing,
                "atol": atol, "rtol": rtol, "parameter_identity_verified": False}
    reference_arm = "ordinary" if "ordinary" in arms else arms[0]
    reference = {case_key(case): case for case in available[reference_arm]["cases"]}
    comparisons = {}
    for arm in arms:
        candidate = {case_key(case): case for case in available[arm]["cases"]}
        if candidate.keys() != reference.keys():
            comparisons[arm] = {"within_tolerance": False, "issue": "case identities differ"}
            continue
        maximum, exact, within, mismatches = 0., True, True, []
        for key in reference:
            for field in VECTOR_FIELDS:
                a, b = candidate[key][field], reference[key][field]
                same = a == b
                exact = exact and same
                if numeric(a) and numeric(b):
                    difference = abs(a - b)
                    maximum = max(maximum, difference)
                    close = math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
                else:
                    close = same
                if not close:
                    within = False
                    mismatches.append({"evaluation_seed": key[0], "environment_index": key[1],
                                       "field": field, "difference": a - b if numeric(a) and numeric(b) else None})
        comparisons[arm] = {"exact_match": exact, "within_tolerance": within,
                            "maximum_absolute_numeric_difference": maximum, "mismatches": mismatches}
    return {"status": "MATCHED" if all(row["within_tolerance"] for row in comparisons.values()) else "MISMATCH",
            "reference_arm": reference_arm, "atol": atol, "rtol": rtol,
            "arms": comparisons, "parameter_identity_verified": False,
            "evidence": "round0 per-environment task outputs, excluding latency"}


def stage_ledger(records):
    by_stage = {}
    for stage in sorted({record["stage"] for record in records}):
        selected = [record for record in records if record["stage"] == stage]
        by_stage[stage] = {"records": len(selected),
                           "wall_seconds": sum(record["wall_seconds"] for record in selected),
                           "work": sum_work(selected)}
    training = [record for record in records if record["stage"] != "evaluation"]
    evaluation = [record for record in records if record["stage"] == "evaluation"]
    diagnostics = [{"round": record["round"], "stage": record["stage"],
                    "values": {key: value for key, value in record.get("details", {}).items() if numeric(value)}}
                   for record in training if record["stage"] in {"model_proposal", "model_calibration", "ppo_update"}]
    return {"record_count": len(records), "work": sum_work(records),
            "training_work": sum_work(training), "evaluation_work": sum_work(evaluation),
            "stage_wall_seconds": sum(record["wall_seconds"] for record in records),
            "stages": by_stage, "optimization_diagnostics": diagnostics}


def summarize(source, initial_atol=1e-6, initial_rtol=1e-6):
    source = source.resolve()
    configuration = read_json(source / "config.json")
    overall = read_json(source / "summary.json")
    if configuration.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1 task-response run")
    experiment, environment = configuration["experiment"], configuration["environment"]
    seeds, arms = experiment["training_seeds"], experiment["arms"]
    if len(set(seeds)) != len(seeds) or len(set(arms)) != len(arms):
        raise ValueError("Duplicate configured seed or arm")
    wanted_rounds = sorted({0, experiment["comparison_rounds"],
                           *range(experiment["evaluation_every_rounds"], experiment["comparison_rounds"] + 1,
                                  experiment["evaluation_every_rounds"])})
    expected_n = experiment["evaluation_environments"]
    if expected_n % 2:
        raise ValueError("Paired go/stop layout requires an even evaluation count")
    final_round = str(experiment["comparison_rounds"])
    batch = experiment["environments"] * environment["horizon"]
    per_round = {"physical_steps": 3 * batch,
                 "policy_optimizer_steps": configuration["ppo"]["epochs"] * max(1, batch // configuration["ppo"]["minibatch_size"]),
                 "model_optimizer_steps": configuration["learning"]["model_steps"] + experiment["model_calibration_steps"]}
    expected_work = {key: value * experiment["comparison_rounds"] for key, value in per_round.items()}
    reported_seeds = {int(row["seed"]): row for row in overall.get("seeds", [])}
    runs, warmups, manifest, issues = {}, {}, [], []

    def register(path):
        if path.exists():
            manifest.append({"file": path.relative_to(source).as_posix(),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    for name in ("config.json", "summary.json", "source-identity.json", "plan.json"):
        register(source / name)
    for seed in seeds:
        seed_root = source / f"seed-{seed}"
        global_seed = reported_seeds.get(seed, {})
        warm_path = seed_root / "common-warmup" / "summary.json"
        warm = read_json(warm_path) if warm_path.exists() else global_seed.get("warmup", {})
        register(warm_path)
        warm_stages_path = seed_root / "common-warmup" / "stages.jsonl"
        warm_records = read_stages(warm_stages_path)
        register(warm_stages_path)
        warm_ledger = stage_ledger(warm_records)
        warm_differences = work_difference(warm_ledger["work"], warm.get("work", {}))
        expected_warm = {key: value * experiment["common_warmup_rounds"] for key, value in per_round.items()}
        warm_complete = (warm.get("status") == COMPLETE
                         and warm.get("completed_rounds") == experiment["common_warmup_rounds"]
                         and warm.get("pair_calibrated") is True
                         and all(warm_ledger["training_work"].get(key, 0) == value for key, value in expected_warm.items())
                         and not warm_differences)
        warmups[str(seed)] = {key: warm.get(key) for key in
                              ("status", "completed_rounds", "pair_calibrated", "wall_seconds", "overrun_seconds")}
        warmups[str(seed)].update(verified_complete=warm_complete, ledger=warm_ledger,
                                  ledger_report_differences=warm_differences)
        for arm in arms:
            arm_root = seed_root / arm
            summary_path, stages_path = arm_root / "summary.json", arm_root / "stages.jsonl"
            global_arm = global_seed.get("arms", {}).get(arm, {})
            report = read_json(summary_path) if summary_path.exists() else global_arm
            register(summary_path); register(stages_path)
            if summary_path.exists() and global_arm and report != global_arm:
                issues.append({"seed": seed, "arm": arm, "issue": "root_and_arm_summary_differ", "uses": "arm_summary"})
            records = read_stages(stages_path)
            ledger = stage_ledger(records)
            evaluations = defaultdict(list)
            for record in records:
                if record["stage"] == "evaluation":
                    evaluations[str(record["round"])].append(record["details"])
            evaluation_difference = dict(evaluations) != report.get("evaluations", {})
            if evaluation_difference:
                issues.append({"seed": seed, "arm": arm, "issue": "stage_and_summary_evaluations_differ", "uses": "stage_records"})
            checkpoints = normalize_evaluations(evaluations, expected_n)
            differences = work_difference(ledger["work"], report.get("work", {}))
            complete_checkpoints = all(
                str(number) in checkpoints
                and checkpoints[str(number)]["scopes"]["all"]["complete_case_set"]
                and {(b["evaluation_seed"], b["planner"]) for b in checkpoints[str(number)]["batches"]}
                    == {(int(s), "cem") for s in experiment["evaluation_seeds"]}
                for number in wanted_rounds)
            matched_work = all(ledger["training_work"].get(key, 0) == value for key, value in expected_work.items())
            completion_verified = (report.get("status") == COMPLETE
                                   and report.get("completed_rounds") == experiment["comparison_rounds"]
                                   and report.get("pair_calibrated") is True
                                   and complete_checkpoints and matched_work
                                   and not differences and not evaluation_difference and warm_complete)
            if report.get("status") == COMPLETE and not completion_verified:
                issues.append({"seed": seed, "arm": arm, "issue": "reported_completion_not_verified"})
            changes = {number: compare_checkpoints(checkpoint, checkpoints["0"])
                       for number, checkpoint in checkpoints.items() if number != "0" and "0" in checkpoints}
            runs[(seed, arm)] = {
                "training_seed": seed, "arm": arm, "reported_status": report.get("status", "NOT_RECORDED"),
                "completed_rounds": report.get("completed_rounds", 0),
                "pair_calibrated": report.get("pair_calibrated"), "lifecycle_phase": report.get("lifecycle_phase"),
                "wall_seconds": report.get("wall_seconds"), "overrun_seconds": report.get("overrun_seconds"),
                "completion_verified": completion_verified, "matched_training_work": matched_work,
                "expected_training_work": expected_work, "ledger": ledger,
                "ledger_report_differences": differences,
                "evaluation_summary_matches_stages": not evaluation_difference,
                "checkpoints": checkpoints, "changes_from_round0": changes,
            }

    initial_checks = {str(seed): round_zero_check(runs, seed, arms, initial_atol, initial_rtol) for seed in seeds}
    for (seed, arm), run in runs.items():
        run["primary_eligible"] = run["completion_verified"] and initial_checks[str(seed)]["status"] == "MATCHED"
        if run["primary_eligible"]:
            run["cohort"] = "COMPLETE_MATCHED_TRAINING_BUDGET"
        elif run["reported_status"] == "BUDGET_EXHAUSTED":
            run["cohort"] = "BUDGET_TRUNCATED"
        elif run["reported_status"].startswith("NOT_"):
            run["cohort"] = "NOT_STARTED_OR_NOT_RECORDED"
        else:
            run["cohort"] = "INCOMPLETE_OR_IDENTITY_UNVERIFIED"
    common_seeds = [seed for seed in seeds if all(runs[(seed, arm)]["primary_eligible"] for arm in arms)]
    aggregates = {}
    for arm in arms:
        aggregates[arm] = {scope: {key: stats([runs[(seed, arm)]["checkpoints"][final_round]["scopes"][scope][key]
                                              for seed in common_seeds]) for key in SCALARS}
                           for scope in ("all", "go", "stop")}
    comparisons = {}
    if "codesign" in arms:
        for reference in ("uniform", "weighted_id", "ordinary"):
            if reference not in arms:
                continue
            per_seed = {}
            for seed in seeds:
                candidate, baseline = runs[(seed, "codesign")], runs[(seed, reference)]
                points = {}
                for number in wanted_rounds:
                    key = str(number)
                    if key in candidate["checkpoints"] and key in baseline["checkpoints"]:
                        points[key] = compare_checkpoints(candidate["checkpoints"][key], baseline["checkpoints"][key])
                per_seed[str(seed)] = {"primary_common_seed": seed in common_seeds, "checkpoints": points}
            keys = ("task_cost_delta", "task_cost_change_percent", "success_change_percentage_points",
                    "normalized_input_integral_delta", "terminal_position_error_delta",
                    "terminal_velocity_error_delta", "corridor_violation_sample_delta")
            aggregate = {scope: {key: stats([per_seed[str(seed)]["checkpoints"][final_round]["scopes"][scope][key]
                                            for seed in common_seeds]) for key in keys}
                         for scope in ("all", "go", "stop")}
            comparisons[reference] = {"direction": "codesign_minus_reference", "paired_training_seeds": common_seeds,
                                      "primary_final_aggregate": aggregate, "per_seed": per_seed,
                                      "interpretation": "observational ablation, not a full causal attribution"}
    total_work = sum_work([{"work": warm["ledger"]["work"]} for warm in warmups.values()]
                          + [{"work": run["ledger"]["work"]} for run in runs.values()])
    root_work_differences = work_difference(total_work, overall.get("work", {}))
    if root_work_differences:
        issues.append({"issue": "root_work_differs_from_all_stage_records"})
    return {
        "schema_version": 1, "experiment_name": configuration["name"],
        "source_run_status": overall.get("status", "UNKNOWN"),
        "summary_status": "COMPLETE_PRIMARY_COHORT" if len(common_seeds) == len(seeds) else "PARTIAL_PRIMARY_COHORT",
        "configured_training_seeds": seeds, "configured_arms": arms, "configured_evaluation_rounds": wanted_rounds,
        "configured_evaluation_seeds": experiment["evaluation_seeds"],
        "primary_complete_common_training_seeds": common_seeds,
        "missing_from_primary_cohort": [seed for seed in seeds if seed not in common_seeds],
        "task_layout": "first half go, second half stop; paired physical conditions within each evaluation batch",
        "public_configuration": {key: configuration[key] for key in
                                 ("environment", "model", "planner", "learning", "ppo", "experiment")},
        "source_manifest": manifest,
        "implementation_source_identity": read_json(source / "source-identity.json") if (source / "source-identity.json").exists() else None,
        "common_warmup": warmups, "round0_equivalence": initial_checks,
        "runs": [runs[(seed, arm)] for seed in seeds for arm in arms],
        "primary_final_aggregate": aggregates, "codesign_comparisons": comparisons,
        "actual_work_from_stages": total_work, "root_work_report_differences": root_work_differences,
        "data_consistency_issues": issues, "interpretation_limits": INTERPRETATION,
        "conditional_response_variance": {"status": "NOT_MEASURED", "reason": "No repeated conditional response distributions are saved"},
    }


def fmt(value, digits=6):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    return f"{value:.{digits}g}" if numeric(value) else str(value)


def markdown(data):
    lines = ["# 任务响应共同学习：固定协议小试结果", "",
             f"原始运行状态：`{data['source_run_status']}`；聚合状态：`{data['summary_status']}`。",
             "",
             "以下是下载日志的只读汇总。运行完成不代表任务成功；完整失败轨迹的成本也进入均值。",
             f"预定训练seed：{data['configured_training_seeds']}；全部组共同完成且身份/交互预算核对通过："
             f"**{data['primary_complete_common_training_seeds']}**。",
             "",
             "## 执行与预算", "",
             "| seed | arm | 运行状态 | 完成轮 | 组别 | wall s | 训练物理步 | 评价物理步 | PPO更新 | 模型更新 |",
             "|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for run in data["runs"]:
        training, evaluation = run["ledger"]["training_work"], run["ledger"]["evaluation_work"]
        lines.append(f"| {run['training_seed']} | {run['arm']} | {run['reported_status']} | {run['completed_rounds']} | "
                     f"{run['cohort']} | {fmt(run['wall_seconds'])} | {training.get('physical_steps', 0)} | "
                     f"{evaluation.get('physical_steps', 0)} | {training.get('policy_optimizer_steps', 0)} | "
                     f"{training.get('model_optimizer_steps', 0)} |")
    lines += ["", "公共warmup在实际总成本中每个seed只计一次：", "",
              "| seed | 状态 | 完成轮 | wall s | 物理步 | 计数核验 |", "|---|---|---:|---:|---:|---|"]
    for seed, warmup in data["common_warmup"].items():
        lines.append(f"| {seed} | {warmup['status']} | {fmt(warmup['completed_rounds'])} | {fmt(warmup['wall_seconds'])} | "
                     f"{warmup['ledger']['work'].get('physical_steps', 0)} | {fmt(warmup['verified_complete'])} |")
    lines += ["", "主比较匹配的是训练交互与更新预算；额外模型求导/规划工作量不同，不能据此称同总算力或同墙钟。",
              "", "## 每个检查点的完整任务结果", "",
              "若某行有截断/无效轨迹，成功率留空，成本仅描述完整有效子集；该行不能进入完整主比较。",
              "", "| seed | arm | round | 完整/全部case | 成功数 | 成功率 | go成本 | stop成本 | 输入积分均值 | 越界case/采样数 | 终点位置误差 | 终点速度误差 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in data["runs"]:
        for number, checkpoint in sorted(run["checkpoints"].items(), key=lambda item: int(item[0])):
            scopes, all_cases = checkpoint["scopes"], checkpoint["scopes"]["all"]
            lines.append(f"| {run['training_seed']} | {run['arm']} | {number} | "
                         f"{all_cases['full_valid_cases']}/{all_cases['cases']} | {all_cases['success_count']} | "
                         f"{fmt(all_cases['success_rate'])} | {fmt(scopes['go']['task_cost_mean'])} | "
                         f"{fmt(scopes['stop']['task_cost_mean'])} | {fmt(all_cases['normalized_input_integral_mean'])} | "
                         f"{all_cases['corridor_violation_case_count']}/{all_cases['corridor_violation_sample_count']} | "
                         f"{fmt(all_cases['terminal_position_error_mean'])} | {fmt(all_cases['terminal_velocity_error_mean'])} |")
    lines += ["", "## 候选相对对照的配对变化", "",
              "仅使用全部四组共同完成的seed和最终检查点；正成本差表示codesign成本更高，负差表示更低。成功差单位为百分点。",
              "每seed、每case的差值，以及各组相对round0的变化均保留在JSON，未按结果筛选种子。",
              "", "| 对照 | 任务 | seed数 | 成本差均值 | 成本差逐seed | 成功差pp均值 | 输入积分差均值 |",
              "|---|---|---:|---:|---|---:|---:|"]
    for reference, comparison in data["codesign_comparisons"].items():
        for scope, row in comparison["primary_final_aggregate"].items():
            lines.append(f"| {reference} | {scope} | {len(comparison['paired_training_seeds'])} | "
                         f"{fmt(row['task_cost_delta']['mean'])} | "
                         f"{', '.join(fmt(v) for v in row['task_cost_delta']['values']) or '—'} | "
                         f"{fmt(row['success_change_percentage_points']['mean'])} | "
                         f"{fmt(row['normalized_input_integral_delta']['mean'])} |")
    lines += ["", "`codesign-uniform`检验任务敏感度选择；`codesign-weighted_id`检验模型任务目标项；"
              "`codesign-ordinary`比较整个候选与普通任务PPO/监督辨识。三种对比是观测性消融，不单独证明全因果链或原创性。",
              "", "## 共同warmup输出核对", "",
              "| seed | round0状态 | 最大绝对数值差 |", "|---|---|---:|"]
    for seed, check in data["round0_equivalence"].items():
        maximum = max((row.get("maximum_absolute_numeric_difference", 0.) for row in check.get("arms", {}).values()), default=None)
        lines.append(f"| {seed} | {check['status']} | {fmt(maximum)} |")
    lines += ["", "该检查对比各组round0的同case输出，排除计时；精确一致、数值差和容差均保留在JSON。"
              "它没有独立加载/验证初始参数或优化器状态。", "", "## 实际总计数", "",
              "| 计数/时间项 | 全部实际阶段合计 |", "|---|---:|"]
    for key, value in data["actual_work_from_stages"].items():
        lines.append(f"| {key} | {fmt(value, 12)} |")
    lines += ["", "各arm阶段耗时和计数见JSON；batch规划延迟不除以环境数冒充单机部署时延。",
              "", "## 证据边界", ""]
    lines += [f"- {item}" for item in data["interpretation_limits"]]
    lines += ["", f"数据一致性待核项：**{len(data['data_consistency_issues'])}**。"]
    for issue in data["data_consistency_issues"]:
        lines.append(f"- {json.dumps(issue, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Downloaded raw run directory containing config.json and summary.json")
    parser.add_argument("destination", type=Path, help="Output directory for public summary.json and report.md")
    parser.add_argument("--initial-atol", type=float, default=1e-6)
    parser.add_argument("--initial-rtol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if args.initial_atol < 0 or args.initial_rtol < 0:
        parser.error("Initial-output tolerances must be nonnegative")
    source, destination = args.source.resolve(), args.destination.resolve()
    if destination == source or destination.is_relative_to(source):
        parser.error("Keep public outputs outside the original run directory")
    data = summarize(source, args.initial_atol, args.initial_rtol)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (destination / "report.md").write_text(markdown(data), encoding="utf-8")
    print(json.dumps({"summary_status": data["summary_status"],
                      "complete_common_seeds": data["primary_complete_common_training_seeds"],
                      "data_consistency_issues": data["data_consistency_issues"],
                      "outputs": [str(destination / "summary.json"), str(destination / "report.md")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
