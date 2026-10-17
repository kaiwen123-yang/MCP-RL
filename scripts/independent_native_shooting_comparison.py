"""Reintegrate every saved candidate using only historical ResponseServo methods.

No environment constructor/step, planner, new prediction step, training, backward,
optimizer or parameter write is used. This is an independent fixed-pool shooting
check, not a continuous trajectory optimizer or a performance experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from response_ppo.env import EnvironmentConfig, ResponseServo
from response_ppo.realization import map_normalized_action

OUT = ROOT / "notes/closed-loop-interface-2026-10-17"
REPLAY = ROOT / "work/neural-demand-audit-2026-10-17/replays"
PROTOCOL = OUT / "independent-shooting-protocol.md"
SOURCE_RESULTS = OUT / "pool-and-gradient-results.json"
VALIDATION_RESULTS = OUT / "validation-results.json"
KEYS = ("mean.0.weight", "mean.0.bias", "mean.2.weight", "mean.2.bias",
        "mean.4.weight", "mean.4.bias")
TOLERANCE = 1e-10
NUMERIC_FIELDS = ("actual_domain_margin_max", "reference_domain_margin_max",
                  "input_margin_max", "tube_margin_max",
                  "actual_one_step_excess_max", "support_worst_case_excess_max")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    # Identical trained-network functional helper used in the earlier audit.
    hidden = torch.tanh(F.linear(obs, p[:1088].reshape(64, 17), p[1088:1152]))
    hidden = torch.tanh(F.linear(hidden, p[1152:5248].reshape(64, 64), p[5248:5312]))
    return F.linear(hidden, p[5312:5376].reshape(1, 64), p[5376:5377]).squeeze(-1)


def known_goals(seed: int, cfg: EnvironmentConfig) -> torch.Tensor:
    # Original initialization draw order, stopping before hidden phase is drawn.
    generator = torch.Generator(device="cpu").manual_seed(seed * 10000 + 5000)
    torch.randn(16, 3, generator=generator)
    count = (256 + cfg.horizon * cfg.command_steps) // cfg.target_steps + 3
    magnitude = .15 + .15 * torch.rand(16, count, generator=generator)
    sign = torch.where(torch.rand(16, count, generator=generator) < .5, -1., 1.)
    return (magnitude * sign).double()


def quadratic(value: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,ij,bj->b", value, metric, value)


def original_shooting(p, x, z, design, commands, goals, cfg, P, B, E, support):
    batch, count, horizon, _ = commands.shape
    instances = batch * count
    env = object.__new__(ResponseServo)
    env.config, env.P, env.B, env.E = cfg, P, B, E
    env.batch, env.support = instances, support
    env.x = x[:, None, :].expand(-1, count, -1).reshape(instances, 3).clone()
    env.z = z[:, None, :].expand(-1, count, -1).reshape(instances, 3).clone()
    env.raw_design = design[:, None, :].expand(-1, count, -1).reshape(instances, 4)
    actual_cost = torch.zeros(instances, dtype=torch.float64)
    reference_cost = torch.zeros_like(actual_cost)
    margins = {key: torch.full_like(actual_cost, -torch.inf) for key in NUMERIC_FIELDS}
    support_empty = torch.zeros(instances, dtype=torch.bool)
    actual_violation = torch.zeros_like(support_empty)
    tube_violation = torch.zeros_like(support_empty)
    actual_bounds = torch.tensor(cfg.actual_bounds, dtype=torch.float64)
    reference_bounds = torch.tensor(cfg.reference_bounds, dtype=torch.float64)
    low_steps = horizon * cfg.command_steps
    for low in range(low_steps):
        command = commands[:, :, low // cfg.command_steps].reshape(instances, 2)
        env.command = torch.stack((command[:, 0], 1. + command[:, 1] * (.2 / .45)), -1)
        # observation -> interval -> response is the original method chain.
        obs = env.observation()
        normalized = torch.tanh(mean(p, obs))
        znext, uref, interval = env.interval()
        emergency = env.fallback(znext)  # Original step also computes this in every arm.
        mapped = map_normalized_action(normalized, interval)
        action = torch.where(interval.feasible, mapped, emergency) if support else cfg.input_bound * normalized
        previous_error = env.x - env.z
        previous_value = quadratic(previous_error, P)
        xnext = env.f0(env.x) + B * action[:, None]  # Nominal w=0; support W unchanged.
        error = xnext - znext
        value = quadratic(error, P)
        excess = value - cfg.rho * previous_value - cfg.delta
        target = goals[:, low, None].expand(-1, count).reshape(instances)
        actual_cost += (xnext[:, 0] - target).square() + .005 * action.square() + .01 * xnext[:, 1].square()
        reference_cost += (znext[:, 0] - target).square() + .005 * uref.square() + .01 * znext[:, 1].square()
        actual_margin = xnext.abs() - actual_bounds
        reference_margin = znext.abs() - reference_bounds
        worst_minus = error - E * cfg.disturbance_bound
        worst_plus = error + E * cfg.disturbance_bound
        worst_excess = torch.maximum(quadratic(worst_minus, P), quadratic(worst_plus, P)) - cfg.rho * previous_value - cfg.delta
        values = {
            "actual_domain_margin_max": actual_margin.amax(-1),
            "reference_domain_margin_max": reference_margin.amax(-1),
            "input_margin_max": action.abs() - cfg.input_bound,
            "tube_margin_max": value - cfg.tube_budget,
            "actual_one_step_excess_max": excess,
            "support_worst_case_excess_max": worst_excess,
        }
        for key, result in values.items():
            margins[key] = torch.maximum(margins[key], result)
        support_empty |= ~interval.feasible
        actual_violation |= (xnext.abs() > actual_bounds + 1e-7).any(-1)
        tube_violation |= value > cfg.tube_budget + 1e-8
        env.x, env.z = xnext, znext
    result = {key: value.reshape(batch, count).numpy() for key, value in margins.items()}
    result.update(
        actual_cost=(actual_cost / low_steps).reshape(batch, count).numpy(),
        reference_cost=(reference_cost / low_steps).reshape(batch, count).numpy(),
        support_empty_any=support_empty.reshape(batch, count).numpy(),
        fallback_used_any=(support_empty & support).reshape(batch, count).numpy(),
        actual_domain_violation_any=actual_violation.reshape(batch, count).numpy(),
        tube_violation_any=tube_violation.reshape(batch, count).numpy(),
    )
    return result


def choice(cost, mask):
    eligible = mask & np.isfinite(cost)
    return np.argmin(np.where(eligible, cost, np.inf), axis=-1), eligible.any(-1)


def compare_query(result, saved, commands, reference_rows):
    errors = {}
    for key in ("actual_cost", "reference_cost", *NUMERIC_FIELDS):
        difference = np.abs(result[key] - saved[key])
        errors[key] = float(difference.max()) if np.isfinite(difference).all() else None
    numeric_ok = all(value is not None and value <= TOLERANCE for value in errors.values())
    mask = result["reference_domain_margin_max"] <= 0
    saved_mask = saved["reference_domain_margin_max"] <= 0
    full = (mask & ~result["actual_domain_violation_any"] & ~result["tube_violation_any"]
            & (result["input_margin_max"] <= 0))
    saved_full = (saved_mask & (saved["actual_domain_margin_max"] <= 1e-7)
                  & (saved["tube_margin_max"] <= 1e-8) & (saved["input_margin_max"] <= 0))
    selected_actual, actual_exists = choice(result["actual_cost"], mask)
    selected_reference, reference_exists = choice(result["reference_cost"], mask)
    full_actual, full_exists = choice(result["actual_cost"], full)
    actual_same, reference_same, exists_same = [], [], []
    for environment, row in enumerate(reference_rows):
        actual_same.append(bool(np.array_equal(commands[environment, selected_actual[environment]],
                                              commands[environment, row["actual_winner_index"]])))
        reference_same.append(bool(np.array_equal(commands[environment, selected_reference[environment]],
                                                 commands[environment, row["reference_winner_index"]])))
        exists_same.append(bool(full_exists[environment]) == row["common_full_physical_screen"]["exists"])
    flags = {key: bool(np.array_equal(result[key], saved[key]))
             for key in ("support_empty_any", "fallback_used_any")}
    masks_ok = bool(np.array_equal(mask, saved_mask) and np.array_equal(full, saved_full))
    return {
        "numeric_max_absolute_errors": errors,
        "numeric_within_frozen_tolerance": numeric_ok,
        "reference_gate_mask_equal": bool(np.array_equal(mask, saved_mask)),
        "secondary_common_mask_equal": bool(np.array_equal(full, saved_full)),
        "flags_equal": flags,
        "actual_distinct_winner_same_count": sum(actual_same),
        "reference_distinct_winner_same_count": sum(reference_same),
        "query_count": len(reference_rows),
        "reference_gate_feasible_count": int(actual_exists.sum()),
        "secondary_common_no_feasible_count": int((~full_exists).sum()),
        "secondary_exists_matches_report": all(exists_same),
        "actual_winner_indices": selected_actual.tolist(),
        "reference_winner_indices": selected_reference.tolist(),
        "secondary_actual_winner_indices": full_actual.tolist(),
        "failed_actual_winner_environments": [i for i, ok in enumerate(actual_same) if not ok],
        "failed_reference_winner_environments": [i for i, ok in enumerate(reference_same) if not ok],
        "passed": bool(numeric_ok and masks_ok and all(flags.values()) and all(actual_same)
                       and all(reference_same) and all(exists_same) and actual_exists.all() and reference_exists.all()),
    }


def case_audit(case):
    start = time.perf_counter()
    seed, arm = case["seed"], case["arm"]
    directory = ROOT / f"work/ppo-mechanism-2026-10-16/formal/seedrun-{seed}/seed-{seed}"
    checkpoint = directory / f"{arm}.pt"
    checkpoint_before = sha(checkpoint)
    if checkpoint_before != case["checkpoint_byte_sha256"]:
        raise ValueError("Checkpoint hash differs from saved source results")
    weight = torch.load(checkpoint, map_location="cpu", weights_only=True)
    p = torch.cat([weight["actor"][key].reshape(-1) for key in KEYS]).double()
    config = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    cfg = EnvironmentConfig(**config["environment"])
    replay = REPLAY / f"{seed}-{arm}.npz"
    if sha(replay) != case["replay_byte_sha256"]:
        raise ValueError("Replay hash differs from saved source results")
    with np.load(replay, allow_pickle=False) as record:
        data = {key: torch.tensor(record[key]).double()
                for key in ("x", "z", "raw_design", "P", "B", "E")}
    targets = known_goals(seed, cfg)
    queries = []
    completed_steps = 0
    candidate_count = 0
    for query in case["queries"]:
        tick = query["tick"]
        path = ROOT / query["pool_relative_path"]
        item = {"tick": tick, "source_pool": query["pool_relative_path"], "source_pool_sha256": sha(path)}
        try:
            if item["source_pool_sha256"] != query["pool_byte_sha256"]:
                raise ValueError("Pool hash differs from saved source results")
            with np.load(path, allow_pickle=False) as pool_file:
                saved = {key: pool_file[key].copy() for key in pool_file.files}
            pool = saved["commands_planner"]
            if pool.shape != (16, 34, 8, 2):
                raise ValueError(f"Unexpected frozen pool shape: {pool.shape}")
            goals = targets[:, torch.arange(tick, tick + 80) // cfg.target_steps]
            result = original_shooting(p, data["x"][tick], data["z"][tick],
                                       data["raw_design"][tick], torch.tensor(pool), goals,
                                       cfg, data["P"], data["B"], data["E"], arm.endswith("support"))
            completed_steps += 16 * 34 * 80
            candidate_count += 16 * 34
            item.update(compare_query(result, saved, pool, query["selections"]))
            item["status"] = "completed"
        except Exception as error:
            item.update(status="failed", passed=False,
                        error=f"{type(error).__name__}: {error}")
        queries.append(item)
    unchanged = sha(checkpoint) == checkpoint_before
    return {
        "seed": seed, "arm": arm, "queries": queries,
        "checkpoint_sha256": checkpoint_before, "checkpoint_unchanged": unchanged,
        "all_queries_passed": bool(len(queries) == 4 and all(row["passed"] for row in queries) and unchanged),
        "wall_seconds": time.perf_counter() - start,
        "budget": {"complete_candidate_trajectories": candidate_count,
                   "offline_actual_model_transitions": completed_steps,
                   "actor_observations": completed_steps,
                   "original_observation_instances": completed_steps,
                   "original_interval_instances_including_observation": 2 * completed_steps,
                   "original_reference_response_instances": 2 * completed_steps,
                   "original_fallback_instances": completed_steps,
                   "environment_step_calls": 0, "mpc_calls": 0,
                   "training_updates": 0, "optimizer_steps": 0, "checkpoint_writes": 0},
    }


def make_report(cases, before, source_hash, protocol_hash, wall):
    rows = [row for case in cases for row in case.get("queries", [])]
    computed = [row for row in rows if row.get("status") == "completed"]
    numeric = {key: max((row["numeric_max_absolute_errors"][key]
                        for row in computed if row["numeric_max_absolute_errors"][key] is not None), default=None)
               for key in ("actual_cost", "reference_cost", *NUMERIC_FIELDS)}
    budget = {}
    for case in cases:
        for key, value in case.get("budget", {}).items():
            budget[key] = budget.get(key, 0) + value
    after = {name: sha(ROOT / name) for name in before}
    passed = len(cases) == 12 and len(rows) == 48 and all(case.get("all_queries_passed", False) for case in cases) and before == after
    return {
        "date": "2026-10-17", "status": "completed" if len(cases) == 12 else "running",
        "protocol_sha256": protocol_hash, "source_results_sha256": source_hash,
        "script_sha256": sha(Path(__file__)), "tolerance": TOLERANCE,
        "dtype": "float64", "device": "cpu", "torch_threads": torch.get_num_threads(),
        "production_before": before, "production_after": after,
        "all_passed": passed, "cases": cases,
        "summary": {"completed_pool_files": len(computed),
                    "query_count": sum(row["query_count"] for row in computed),
                    "max_absolute_errors": numeric,
                    "actual_distinct_winner_same_count": sum(row["actual_distinct_winner_same_count"] for row in computed),
                    "reference_distinct_winner_same_count": sum(row["reference_distinct_winner_same_count"] for row in computed),
                    "secondary_common_no_feasible_count": sum(row["secondary_common_no_feasible_count"] for row in computed),
                    "all_reference_masks_equal": all(row["reference_gate_mask_equal"] for row in computed),
                    "all_secondary_masks_equal": all(row["secondary_common_mask_equal"] for row in computed)},
        "budget": budget, "wall_seconds": wall,
        "scope": "Independent original-method reintegration of every saved candidate. No new prediction step, env.step, planner, optimizer or training. Equality verifies fixed-pool shooting, not continuous TO superiority.",
        "extra_geometry_cost": "Original observation calls interval once; manual execution calls it again. Interval/reference counts include both, and fallback is evaluated in every arm. Same candidate count is not equal runtime.",
    }


def write_report(report):
    (OUT / "independent-shooting-results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for case in report["cases"]:
        values = [q["numeric_max_absolute_errors"]["actual_cost"] for q in case.get("queries", [])
                  if q.get("status") == "completed"]
        rows.append(f"| {case['seed']} | {case['arm']} | {max(values) if values else 'NA'} | {case.get('all_queries_passed', False)} | {case.get('wall_seconds', 0):.3f} |")
    summary = report["summary"]
    text = f"""# 全候选独立原生射击法复算结果

日期：2026-10-17。状态：`{report['status']}`；全部通过：`{report['all_passed']}`。
[冻结补充协议](independent-shooting-protocol.md)、[逐项JSON](independent-shooting-results.json)、[独立脚本](../../scripts/independent_native_shooting_comparison.py)。

只用原ResponseServo基础方法重新积分保存的全部候选，不调用新closed_loop_prediction步进或预测函数；不调用env.step、MPC或训练器。目标、命令、原生支持、冻结mean actor、float64与w=0/supportW=.04均保持。原checkpoint和生产代码前后hash保留在JSON。

| seed | arm | actual成本最大绝对差 | 四个池均通过 | 墙钟秒 |
|---:|---|---:|---|---:|
{chr(10).join(rows)}

已完成池文件：`{summary['completed_pool_files']}`；query：`{summary['query_count']}`；actual不同命令赢家一致：`{summary['actual_distinct_winner_same_count']}`；reference赢家一致：`{summary['reference_distinct_winner_same_count']}`。原reference掩码全同：`{summary['all_reference_masks_equal']}`；次级共同actual/tube/input掩码全同：`{summary['all_secondary_masks_equal']}`。次级池无可行候选仍为`{summary['secondary_common_no_feasible_count']}`，不证明连续问题不可行。

逐项数值最大差：

```json
{json.dumps(summary['max_absolute_errors'], ensure_ascii=False, indent=2)}
```

预算：

```json
{json.dumps(report['budget'], ensure_ascii=False, indent=2)}
```

总墙钟`{report['wall_seconds']:.3f}`秒。旧observation内部interval加手动interval重复计算几何，已计入调用数；不据此声称与新接口同墙钟预算或速度优劣。

这是在先前同数组argmin检查之后**追加**的完整独立候选积分证据。通过只支持保存的同目标成本、约束和固定池选支一致；不是新连续TO优化器、全局最优证明、RL学习或性能优势。
"""
    (OUT / "independent-shooting-results.md").write_text(text, encoding="utf-8")


def main():
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    protocol_hash = sha(PROTOCOL)
    source_hash = sha(SOURCE_RESULTS)
    source = json.loads(SOURCE_RESULTS.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_RESULTS.read_text(encoding="utf-8"))
    before = {name: sha(ROOT / name) for name in validation["production_before"]}
    if before != validation["production_before"]:
        raise ValueError("Historical production source hash changed before independent check")
    cases = []
    started = time.perf_counter()
    with torch.no_grad():
        for case in source["cases"]:
            try:
                result = case_audit(case)
            except Exception as error:
                result = {"seed": case["seed"], "arm": case["arm"], "queries": [],
                          "all_queries_passed": False, "status": "failed",
                          "error": f"{type(error).__name__}: {error}"}
            cases.append(result)
            report = make_report(cases, before, source_hash, protocol_hash, time.perf_counter() - started)
            write_report(report)
            print(json.dumps({"seed": result["seed"], "arm": result["arm"],
                              "passed": result["all_queries_passed"],
                              "wall_seconds": result.get("wall_seconds"),
                              "error": result.get("error")}, ensure_ascii=False), flush=True)
    if not report["all_passed"]:
        raise SystemExit("Independent shooting has recorded failures; inspect all retained cases")


if __name__ == "__main__":
    main()
