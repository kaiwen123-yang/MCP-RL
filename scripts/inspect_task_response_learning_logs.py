"""Summarize already saved A/B/C logs without importing torch or project models.

This script performs JSON/statistical arithmetic only.  It does not load neural
weights, propagate a model, step an environment, or execute an optimizer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def distribution(values):
    values = [float(x) for x in values]
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def grouped(rows, keys):
    output = defaultdict(list)
    for row in rows:
        output[tuple(row[k] for k in keys)].append(row)
    return output


def analyze(run, mechanism):
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    experiment = config["experiment"]
    long_weight = config["learning"]["long_response_weight"]
    collection_rows, model_rows, ppo_rows, provenance = [], [], [], []
    for seed in experiment["training_seeds"]:
        for arm in ["common-warmup", *experiment["arms"]]:
            source = run / f"seed-{seed}" / arm / "stages.jsonl"
            raw = source.read_bytes()
            provenance.append({"file": str(source.relative_to(run)).replace("\\", "/"),
                               "sha256": hashlib.sha256(raw).hexdigest()})
            records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
            by_round = defaultdict(dict)
            for line, record in enumerate(records, 1):
                by_round[record["round"]][record["stage"]] = record
                base = {"seed": seed, "arm": arm, "round": record["round"],
                        "stage": record["stage"], "source_line": line}
                details = record["details"]
                if record["stage"] in {"A_collect", "B_collect", "C_collect", "evaluation"}:
                    actual = [-x for x in details["real_task_return_per_environment"]]
                    penalties = details["response_penalty_per_environment"]
                    shaped_cost = [-x for x in details["shaped_training_return_per_environment"]]
                    collection_rows.append({
                        **base, "policy_mode": details.get("collection_policy_mode", "deterministic_evaluation"),
                        "episodes": len(actual), "actual_cost_mean": statistics.mean(actual),
                        "response_penalty_mean": statistics.mean(penalties),
                        "penalty_percent_of_actual_total": 100 * sum(penalties) / sum(actual),
                        "per_episode_penalty_percent": [100*p/a for p, a in zip(penalties, actual)],
                        "reward_decomposition_max_abs_error": max(abs(s-a-p) for s, a, p in zip(shaped_cost, actual, penalties)),
                        "actual_input_integral_mean": statistics.mean(details["task_metrics"]["normalized_input_integral"]),
                        "actual_successes": sum(details["task_metrics"]["success"]),
                    })
                elif record["stage"] in {"model_proposal", "model_calibration"}:
                    paired_stage = "A_collect" if record["stage"] == "model_proposal" else "C_collect"
                    # The paired collection precedes this model stage in the same round.
                    paired_actual = [-x for x in by_round[record["round"]][paired_stage]["details"]["real_task_return_per_environment"]]
                    identification = details["local_realization"] + long_weight * details["long_realization"]
                    task_term = details["task_coefficient"] * details["predicted_task"]
                    model_rows.append({
                        **base, **{k: details[k] for k in ("local_realization", "long_realization", "predicted_task", "task_coefficient", "model_loss")},
                        "identification_term": identification, "task_term": task_term,
                        "task_over_identification": task_term / identification if identification else None,
                        "task_percent_of_total_loss": 100*task_term/details["model_loss"] if details["model_loss"] else None,
                        "loss_decomposition_abs_error": abs(details["model_loss"]-identification-task_term),
                        "paired_actual_collection": paired_stage,
                        "paired_actual_cost_mean": statistics.mean(paired_actual),
                        "paired_actual_minus_logged_prediction": statistics.mean(paired_actual)-details["predicted_task"],
                        "evaluation_time": "Loss is evaluated before the fourth optimizer step, after three updates; not final post-stage loss",
                    })
                elif record["stage"] == "ppo_update":
                    ppo_rows.append({**base, **details})
    primary_collections = [r for r in collection_rows if r["arm"] != "common-warmup" and r["stage"] != "evaluation"]
    primary_models = [r for r in model_rows if r["arm"] != "common-warmup"]
    reward_summary = []
    for (arm, stage), rows in grouped(primary_collections, ("arm", "stage")).items():
        actual_sum = sum(r["actual_cost_mean"]*r["episodes"] for r in rows)
        penalty_sum = sum(r["response_penalty_mean"]*r["episodes"] for r in rows)
        reward_summary.append({"arm": arm, "stage": stage, "batch_count": len(rows),
                               "actual_cost": distribution(r["actual_cost_mean"] for r in rows),
                               "response_penalty": distribution(r["response_penalty_mean"] for r in rows),
                               "aggregate_penalty_percent_of_actual_cost": 100*penalty_sum/actual_sum,
                               "episode_penalty_percent": distribution(x for r in rows for x in r["per_episode_penalty_percent"]),
                               "actual_input_integral": distribution(r["actual_input_integral_mean"] for r in rows),
                               "max_reward_decomposition_error": max(r["reward_decomposition_max_abs_error"] for r in rows)})
    model_summary = []
    for (arm, stage), rows in grouped(primary_models, ("arm", "stage")).items():
        keys = ("local_realization", "long_realization", "identification_term", "predicted_task",
                "task_term", "task_over_identification", "task_percent_of_total_loss",
                "paired_actual_cost_mean", "paired_actual_minus_logged_prediction")
        model_summary.append({"arm": arm, "stage": stage, "batch_count": len(rows),
                              **{k: distribution(r[k] for r in rows if r[k] is not None) for k in keys},
                              "optimistic_batches_on_own_recorded_commands": sum(r["paired_actual_minus_logged_prediction"] > 0 for r in rows),
                              "max_loss_decomposition_error": max(r["loss_decomposition_abs_error"] for r in rows)})
    trajectory_summary = []
    for (seed, arm, stage), rows in grouped(primary_models, ("seed", "arm", "stage")).items():
        rows.sort(key=lambda r: r["round"])
        trajectory_summary.append({"seed": seed, "arm": arm, "stage": stage,
                                   "first_round": rows[0], "last_round": rows[-1],
                                   "interpretation": "Different rounds use different commands, states, actors and samples; not an update effect on a fixed batch"})
    ppo_summary = []
    for (arm,), rows in grouped([r for r in ppo_rows if r["arm"] != "common-warmup"], ("arm",)).items():
        keys = ("loss/vf", "loss/clip", "loss/ent", "approx_kl", "clip_fraction", "behavior_logprob_max_error")
        ppo_summary.append({"arm": arm, "updates": len(rows),
                            **{k: distribution(r[k] for r in rows) for k in keys},
                            "optimizer_steps": sum(r["optimizer_steps"] for r in rows)})
    mechanism_data = json.loads(mechanism.read_text(encoding="utf-8"))
    final_models = [r for r in mechanism_data["across_training_seeds"] if r["round"] == experiment["comparison_rounds"]]
    evaluation_optimism = []
    for (arm,), rows in grouped(final_models, ("arm",)).items():
        keys = ("actual_task_cost_mean", "recorded_commands_model_cost_mean", "optimism_gap_actual_minus_model_mean",
                "local_position_rmse_l2_mean", "local_velocity_rmse_l2_mean", "local_resource_rmse_mean",
                "long_position_rmse_l2_mean", "long_velocity_rmse_l2_mean", "long_resource_rmse_mean")
        evaluation_optimism.append({"arm": arm, **{k: statistics.mean(r[k] for r in rows) for k in keys},
                                    "source": "Existing saved-evidence analysis; no new prediction executed by this script"})
    return {
        "status": "COMPLETE_EXISTING_LOG_STATISTICS", "config": config,
        "source_logs": provenance,
        "existing_mechanism_sha256_lf": hashlib.sha256(mechanism.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "existing_mechanism_hash_policy": "Normalize CRLF to LF for cross-checkout source identity",
        "operations": {"new_physical_steps": 0, "new_model_steps": 0, "new_actor_forwards": 0,
                       "optimizer_steps": 0, "checkpoint_loads": 0, "weight_writes": 0},
        "reward_summary": reward_summary, "model_summary": model_summary,
        "ppo_summary": ppo_summary, "cross_round_descriptions": trajectory_summary,
        "post_calibration_evaluation_from_existing_analysis": evaluation_optimism,
        "collection_rows": collection_rows, "model_rows": model_rows,
        "evidence_boundaries": [
            "Training logs retain only the loss computed before the fourth model optimizer step. They contain neither pre-update same-batch losses nor the final post-step loss.",
            "A and C have different sampled conditions, commands and actor versions. Proposal-minus-calibration is not a same-data estimate of calibration effect.",
            "A/C fit deterministic deployment execution; B PPO uses stochastic trajectories. Their penalty differences confound policy randomness with states, commands and versions.",
            "Objective value dominance and reward magnitude are not gradient-norm or causal attribution measurements.",
            "Existing final evaluation demonstrates residual model optimism after the complete calibration stage, but does not identify the unique cause.",
            "Critic and actor observe current physical state, current command and task context; cached macro-reference targets and W are not explicit observations.",
            "CEM uses a one-second endpoint surrogate, model-task fitting uses the four-second recorded-command rollout, W uses a held-command suffix, and PPO uses actual four-second receding-horizon execution.",
            "No new model evaluation, environment transition, parameter update, or server action occurs in this inspection."
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("work/server-task-response-20261019/downloaded/run"))
    parser.add_argument("--mechanism", type=Path, default=Path("notes/server-pilot-2026-10-19/mechanism/analysis.json"))
    parser.add_argument("--output", type=Path, default=Path("notes/method-reassessment-2026-10-19/log-analysis.json"))
    args = parser.parse_args()
    report = analyze(args.run, args.mechanism)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "source_log_count": len(report["source_logs"]),
                      "operations": report["operations"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
