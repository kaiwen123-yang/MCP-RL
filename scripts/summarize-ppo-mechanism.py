"""Summarize frozen PPO pilot logs without importing or running the learner.

Evaluation means in the raw runner average every numeric field, including maxima
and counts. This script deliberately uses the underlying evaluation.batches:
counts/work are summed, maxima are maximized, and sample means/rates are weighted
by their denominators. Model-forward counts cover planner candidates only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ARMS = ("joint_support", "fixed_support", "joint_penalty", "fixed_penalty")
SEEDS = (17, 29, 43)
LABELS = ("short", "long")
MEANS = (
    "task_cost", "total_cost", "realized_violation_rate", "mean_positive_excess",
    "empty_support_rate", "mean_support_width", "geometric_bound_violation_rate",
    "actual_domain_violation_rate", "reference_domain_violation_rate", "tube_violation_rate",
)
MAXIMA = ("max_excess", "max_V", "max_bound_excess")
COUNTS = (
    "actual_domain_violation_count", "reference_domain_violation_count",
    "tube_violation_count", "geometric_bound_violation_count",
)
PLANNER_COUNTS = ("objective_calls", "candidate_sequences", "candidate_macro_steps")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_batches(batches):
    n = sum(int(b["physical_steps"]) for b in batches)
    instances = sum(int(b["planner_instances"]) for b in batches)
    out = {key: sum(float(b[key]) * b["physical_steps"] for b in batches) / n for key in MEANS}
    out.update({key: max(float(b[key]) for b in batches) for key in MAXIMA})
    out.update({key: sum(int(b[key]) for b in batches) for key in COUNTS})
    out.update({key: sum(int(b["planner_counts"][key]) for b in batches) for key in PLANNER_COUNTS})
    out.update(
        batches=len(batches), physical_steps=n, planner_instances=instances,
        planner_low_level_model_steps=sum(int(b["planner_low_level_model_steps"]) for b in batches),
        planner_domain_failure_rate=sum(b["planner_domain_failure_rate"] * b["planner_instances"] for b in batches) / instances,
        wall_seconds=sum(float(b["wall_seconds"]) for b in batches),
    )
    # These two event counts were saved only as rates; recover their integer
    # counts for transparent denominators, retaining the original weighted rate.
    for name in ("realized_violation", "empty_support"):
        out[name + "_count_from_rate"] = sum(round(b[name + "_rate"] * b["physical_steps"]) for b in batches)
    out["planner_domain_failure_count_from_rate"] = sum(round(b["planner_domain_failure_rate"] * b["planner_instances"]) for b in batches)
    return out


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "minimum": min(values), "maximum": max(values), "n_seeds": len(values)}


def physical_design(raw):
    return [lo + span / (1 + math.exp(-x)) for x, lo, span in zip(raw, (1.2, .7, 3., 0.), (1.8, .7, 11., .5))]


def ledger_row(seed, arm, phase, batches, optimizer_steps=0, ppo_wall_seconds=0.):
    a = aggregate_batches(batches)
    row = {"seed": seed, "arm": arm, "phase": phase,
           **{key: a[key] for key in ("batches", "physical_steps", "planner_instances", *PLANNER_COUNTS,
                                      "planner_low_level_model_steps", "wall_seconds")},
           "ppo_optimizer_steps": optimizer_steps, "ppo_update_wall_seconds": ppo_wall_seconds}
    return row


def total_ledger(rows):
    fields = ("physical_steps", "planner_instances", *PLANNER_COUNTS, "planner_low_level_model_steps", "ppo_optimizer_steps")
    return {key: sum(int(row[key]) for row in rows) for key in fields}


def summarize(root, raw_root, output):
    output.mkdir(parents=True, exist_ok=True)
    per_seed, curves, ledger, checks, sources, results = [], [], [], [], [], {}
    all_batches = {(arm, label): [] for arm in ARMS for label in LABELS}
    frozen_config = None
    frozen_source_hashes = None
    for seed in SEEDS:
        run_path = raw_root / "formal" / f"seedrun-{seed}"
        manifest_path, result_path = run_path / "manifest.json", run_path / f"seed-{seed}" / "result.json"
        manifest, result = read_json(manifest_path), read_json(result_path)
        if manifest["status"] != "completed":
            raise ValueError(f"Formal seed {seed} is not complete")
        if set(result["arms"]) != set(ARMS):
            raise ValueError(f"Formal seed {seed} does not contain all four arms")
        before, after = manifest["source_hashes_before_run"], result["source_hashes"]
        if before != after:
            raise ValueError(f"Formal seed {seed} changed source hashes while running")
        initial = {arm: result["arms"][arm]["initial_actor_sha256"] for arm in ARMS}
        if len(set(initial.values())) != 1:
            raise ValueError(f"Formal seed {seed} initial actors differ")
        cfg = {"environment": result["environment"], "ppo": result["ppo"], "design_update": result["design_update"]}
        if frozen_config is None:
            frozen_config, frozen_source_hashes = cfg, before
        elif cfg != frozen_config or before != frozen_source_hashes:
            raise ValueError(f"Formal seed {seed} uses a different configuration or source")
        current_match = all(digest(root / name) == sha for name, sha in after.items())
        checks.append({"seed": seed, "manifest_status": manifest["status"],
                       "source_before_equals_after": True, "current_source_matches_run": current_match,
                       "same_initial_actor_across_arms": True, "initial_actor_sha256": next(iter(initial.values()))})
        sources.append({"kind": "formal", "seed": seed, "result_sha256": digest(result_path), "manifest_sha256": digest(manifest_path)})
        results[seed] = result
        for arm in ARMS:
            item = result["arms"][arm]
            history = item["history"]
            if len(history) != manifest["updates"]:
                raise ValueError(f"Missing training batches: {seed}/{arm}")
            opt_steps = sum(int(row["ppo"]["optimizer_steps"]) for row in history)
            train_batches = [row["rollout"] for row in history]
            train_row = ledger_row(seed, arm, "formal_training", train_batches, opt_steps,
                                   sum(row["ppo"]["wall_seconds"] for row in history))
            if train_row["physical_steps"] != item["budget"]["training_physical_steps"] or opt_steps != item["budget"]["optimizer_steps"]:
                raise ValueError(f"Budget and actual training logs disagree: {seed}/{arm}")
            if item["provenance"]["actor_parameters"] != 5378 or item["provenance"]["critic_parameters"] != 5377:
                raise ValueError(f"Unexpected capacity: {seed}/{arm}")
            ledger.append(train_row)
            checks.append({"seed": seed, "arm": arm, "training_batches": len(history),
                           "actor_parameters": item["provenance"]["actor_parameters"],
                           "critic_parameters": item["provenance"]["critic_parameters"],
                           "optimizer_steps": opt_steps,
                           "optimizer_steps_per_batch": sorted(set(row["ppo"]["optimizer_steps"] for row in history)),
                           "max_behavior_logprob_error": max(row["ppo"]["behavior_logprob_max_error"] for row in history),
                           "applied_design_total_path_length": sum(row["applied_design_step_l2"] for row in history)})
            cumulative_physics = cumulative_model = 0
            for row in history:
                rollout, ppo = row["rollout"], row["ppo"]
                cumulative_physics += rollout["physical_steps"]
                cumulative_model += rollout["planner_low_level_model_steps"]
                flat = {"seed": seed, "arm": arm, "update": row["update"],
                        "cumulative_physical_steps": cumulative_physics,
                        "cumulative_planner_model_steps": cumulative_model,
                        **{key: rollout[key] for key in (*MEANS, *MAXIMA, *COUNTS)},
                        "planner_domain_failure_rate": rollout["planner_domain_failure_rate"],
                        "rollout_wall_seconds": rollout["wall_seconds"],
                        "applied_design_step_l2": row["applied_design_step_l2"]}
                flat.update({"ppo_" + key: value for key, value in ppo.items() if isinstance(value, (int, float))})
                flat.update({f"design_raw_{i}_after": value for i, value in enumerate(row["design_after"])})
                curves.append(flat)
            eval_physical = 0
            for label in LABELS:
                batches = item["evaluation"][label]["batches"]
                if len(batches) != manifest["test_batches"]:
                    raise ValueError(f"Missing test batches: {seed}/{arm}/{label}")
                agg = aggregate_batches(batches)
                all_batches[arm, label].extend(batches)
                eval_physical += agg["physical_steps"]
                row = {"seed": seed, "arm": arm, "horizon": label, **agg}
                row.update({f"design_raw_{i}": x for i, x in enumerate(item["design_final"])})
                row.update({f"design_physical_{name}": x for name, x in zip(("omega", "zeta", "kappa", "factor"), physical_design(item["design_final"]))})
                per_seed.append(row)
                ledger.append(ledger_row(seed, arm, "formal_evaluation_" + label, batches))
            if eval_physical != item["budget"]["evaluation_physical_steps"]:
                raise ValueError(f"Budget and actual evaluation logs disagree: {seed}/{arm}")

    paired = []
    for seed in SEEDS:
        for label in LABELS:
            cost = {row["arm"]: row["task_cost"] for row in per_seed if row["seed"] == seed and row["horizon"] == label}
            js, fs, jp, fp = (cost[arm] for arm in ARMS)
            paired.append({"seed": seed, "horizon": label,
                           "joint_support_minus_fixed_support": js - fs,
                           "joint_support_improvement_percent": 100 * (fs - js) / fs,
                           "joint_penalty_minus_fixed_penalty": jp - fp,
                           "interaction_difference_of_differences": (js - fs) - (jp - fp),
                           "joint_support_better_than_fixed": js < fs})
    arms_summary = {}
    for arm in ARMS:
        arms_summary[arm] = {}
        for label in LABELS:
            a = aggregate_batches(all_batches[arm, label])
            a["task_cost_seed_statistics"] = stats([row["task_cost"] for row in per_seed if row["arm"] == arm and row["horizon"] == label])
            arms_summary[arm][label] = a

    midpoint_rows, midpoint_ledger = [], []
    for seed in SEEDS:
        for arm in ("joint_support", "fixed_support"):
            path = raw_root / "midpoint" / f"seed-{seed}-{arm}.json"
            midpoint = read_json(path)
            if midpoint["source_hashes"] != frozen_source_hashes:
                raise ValueError(f"Midpoint source mismatch: {seed}/{arm}")
            if midpoint["design_mean"] != results[seed]["arms"][arm]["design_final"]:
                raise ValueError(f"Midpoint response differs: {seed}/{arm}")
            sources.append({"kind": "midpoint", "seed": seed, "arm": arm, "result_sha256": digest(path)})
            for label in LABELS:
                batches = midpoint["evaluation"][label]["batches"]
                a = aggregate_batches(batches)
                learned = next(row for row in per_seed if row["seed"] == seed and row["arm"] == arm and row["horizon"] == label)
                midpoint_rows.append({"seed": seed, "arm_design": arm, "horizon": label,
                                      "midpoint_task_cost": a["task_cost"], "learned_task_cost": learned["task_cost"],
                                      "learned_improvement_over_midpoint_percent": 100 * (a["task_cost"] - learned["task_cost"]) / a["task_cost"],
                                      "learned_minus_midpoint": learned["task_cost"] - a["task_cost"],
                                      **{"midpoint_" + key: a[key] for key in (*MAXIMA, *COUNTS, "physical_steps", "empty_support_rate", "realized_violation_rate")}})
                midpoint_ledger.append(ledger_row(seed, arm, "midpoint_diagnostic_" + label, batches))
    main_totals, diagnostic_totals = total_ledger(ledger), total_ledger(midpoint_ledger)
    if main_totals["physical_steps"] != 2949120 or main_totals["ppo_optimizer_steps"] != 30720:
        raise ValueError("Formal physical/PPO totals do not match the frozen protocol")
    if diagnostic_totals["physical_steps"] != 491520:
        raise ValueError("Midpoint totals do not match the frozen diagnostic protocol")

    config = {"seeds": list(SEEDS), "arms": list(ARMS), **frozen_config,
              "updates": 40, "environments": 16, "train_steps": 256, "test_batches": 4,
              "short_steps": 256, "long_steps": 1024,
              "source_hashes": frozen_source_hashes,
              "upstream_ppo": {key: results[SEEDS[0]]["arms"][ARMS[0]]["provenance"][key] for key in (
                  "tianshou", "torch", "learner_class", "learn_method", "upstream_file", "upstream_file_sha256")},
              "fixed_metric": results[SEEDS[0]]["arms"][ARMS[0]]["history"][0]["rollout"]["metric"],
              "action_mapping": {"support": "midpoint + halfwidth * tanh(raw); empty support uses logged minimax fallback",
                                 "penalty": "input_bound * tanh(raw)"},
              "notes": ["PPO is common infrastructure, not the claimed innovation.",
                        "Support is analytic and independent of policy weights.",
                        "Reference model is analytic known-model-compatible dynamics with four design parameters.",
                        "Short and long experiments are independently sampled at their fixed lengths, not paired prefixes.",
                        "High-level planning uses persistent nominal z; true x enters low-level feedback."]}
    summary = {
        "status": "completed_and_aggregated_from_batch_logs", "date": "2026-10-16",
        "aggregation": {"cost_and_rates": "weighted by physical steps (all compared batches have equal lengths)",
                        "planner_failure_rate": "weighted by planner instances", "max_fields": "maximum over underlying batches",
                        "counts_and_work": "sum over underlying batches", "seed_uncertainty": "sample SD across three seeds; no significance claim",
                        "raw_runner_mean": "not used: it incorrectly averages max/count/work fields"},
        "arms": arms_summary, "paired_seed_comparisons": paired,
        "interaction_definition": "(J_support-F_support)-(J_penalty-F_penalty); negative means the joint design effect is more favorable under support, not proof of novelty",
        "joint_support_improvement_seed_statistics": {label: stats([row["joint_support_improvement_percent"] for row in paired if row["horizon"] == label]) for label in LABELS},
        "checks": checks, "source_evidence": sources,
        "formal_budget": main_totals,
        "midpoint_diagnostic_budget": diagnostic_totals,
        "formal_plus_midpoint_budget": total_ledger(ledger + midpoint_ledger),
        "midpoint_table": "midpoint-comparison.csv; learned metrics are reused from formal evaluation and not counted twice",
        "existing_midpoint_record": "midpoint-summary.json is retained; this script reaggregates raw midpoint batches to correct maxima/count semantics",
        "timing_limit": "Three CPU seed processes ran concurrently; logged wall times are descriptive, not a controlled speed ranking or GPU/NX benchmark.",
        "excluded_budget": "development seed 901, backend smoke and algebra checks are separate and not included in formal or midpoint totals",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "config.json", config)
    write_csv(output / "per-seed.csv", per_seed)
    write_csv(output / "ledger.csv", ledger + midpoint_ledger)
    write_csv(output / "learning-curves.csv", curves)
    write_csv(output / "paired-comparison.csv", paired)
    write_csv(output / "midpoint-comparison.csv", midpoint_rows)
    print(json.dumps({"formal_budget": main_totals, "midpoint_budget": diagnostic_totals,
                      "paired": paired,
                      "task_means": {a: {h: arms_summary[a][h]["task_cost"] for h in LABELS} for a in ARMS},
                      "midpoint": [{k: row[k] for k in ("seed", "arm_design", "horizon", "learned_improvement_over_midpoint_percent")} for row in midpoint_rows]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("work/ppo-mechanism-2026-10-16"))
    parser.add_argument("--output", type=Path, default=Path("notes/ppo-mechanism-2026-10-16/results"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    summarize(repo, args.raw_root.resolve(), args.output.resolve())
