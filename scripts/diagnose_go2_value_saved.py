"""Diagnose recorded critic values against saved reward suffixes.

Reads transitions.pt with weights_only=True, but never loads a network or runs
a model, simulator, optimizer, gradient calculation, or new data collection.
Scope: all 27 final evaluation episodes; nine round-0 shared-start references
(frozen arm); nine seed-11 round-3 evaluations. Matching final/seed-11 round-3
stochastic behavior collections are retained separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


STAGES = {0: "straight", 1: "turn", 2: "door", 3: "stop", 4: "completed"}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def metrics(values, suffix, remaining_seconds, exact):
    error = values - suffix
    output = {
        "steps": len(values),
        "recorded_V_mean": float(np.mean(values)),
        "recorded_V_min": float(np.min(values)),
        "recorded_V_max": float(np.max(values)),
        "positive_V_fraction": float(np.mean(values > 0)),
        "predicted_remaining_cost_seconds_mean": float(np.mean(-values)),
        "observed_seconds_to_episode_boundary_mean": float(np.mean(remaining_seconds)),
        "observed_reward_suffix_mean": float(np.mean(suffix)),
        "suffix_target_kind": "complete_actual_return" if exact else "observed_partial_suffix_only",
        "V_minus_actual_return_bias": float(np.mean(error)) if exact else None,
        "V_vs_actual_return_rmse": float(np.sqrt(np.mean(error**2))) if exact else None,
        "V_vs_actual_return_mae": float(np.mean(np.abs(error))) if exact else None,
        "optimistic_V_fraction": float(np.mean(error > 0)) if exact else None,
        "pessimistic_V_fraction": float(np.mean(error < 0)) if exact else None,
        "V_return_sign_mismatch_fraction": (
            float(np.mean(np.sign(values) != np.sign(suffix))) if exact else None),
        "predicted_cost_minus_realized_cost_bias_seconds": (
            float(np.mean(-error)) if exact else None),
        "predicted_cost_median_seconds": float(np.median(-values)),
        "observed_seconds_to_boundary_median": float(np.median(remaining_seconds)),
    }
    return output


def selected_collections(config):
    final = int(config["rounds"])
    for seed in config["seeds"]:
        yield seed, "frozen", 0, "evaluation", "initial_reference", f"seed{seed}/frozen/evaluation_round0"
        for arm in config["arms"]:
            for phase in ("evaluation", "behavior"):
                yield seed, arm, final, phase, "final", f"seed{seed}/{arm}/round{final}/{phase}"
    for arm in config["arms"]:
        for phase in ("evaluation", "behavior"):
            yield 11, arm, 3, phase, "seed11_round3", f"seed11/{arm}/round3/{phase}"


def diagnose(runs, output):
    config_path = runs / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["ppo"]["gamma"] != 1.0:
        raise ValueError("This saved-return diagnosis is for the declared undiscounted protocol.")
    dt = float(config["policy_dt"])
    episode_rows, region_rows, entry_rows, collection_rows = [], [], [], []
    groups = defaultdict(list)
    fingerprints = {"config.json": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    for seed, arm, round_number, phase, cohort, relative in selected_collections(config):
        directory = runs / relative
        tensor_path, summary_path = directory / "transitions.pt", directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved = torch.load(tensor_path, map_location="cpu", weights_only=True)
        batch = {key: value.detach().cpu().numpy() for key, value in saved["batch"].items()
                 if key in ("state", "memory", "reward", "old_value", "terminated",
                            "truncated", "case_index", "horizon")}
        for path in (tensor_path, summary_path):
            fingerprints[path.relative_to(runs).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        boundaries = np.flatnonzero(batch["terminated"] | batch["truncated"])
        if len(boundaries) != len(summary["episodes"]) or boundaries[-1] != len(batch["reward"]) - 1:
            raise ValueError(f"Saved episode boundary/summary mismatch: {relative}")
        timing = ("post_common_fit_value" if round_number == 0 else
                  "post_current_B_value_update" if phase == "evaluation" else
                  "pre_current_B_value_update")
        collection_rows.append({
            "seed": seed, "arm": arm, "round": round_number, "phase": phase,
            "cohort": cohort, "behavior_mode": summary["behavior"],
            "episode_count": len(boundaries), "recorded_steps": len(batch["reward"]),
            "critic_timing": timing, "source": relative,
        })
        start = 0
        for end, episode in zip(boundaries, summary["episodes"]):
            stop = int(end) + 1
            segment = slice(start, stop)
            state = batch["state"][segment].astype(np.float64)
            memory = batch["memory"][segment].astype(np.float64)
            values = batch["old_value"][segment].astype(np.float64)
            rewards = batch["reward"][segment].astype(np.float64)
            suffix = np.cumsum(rewards[::-1], dtype=np.float64)[::-1]
            remaining = np.arange(len(rewards), 0, -1, dtype=np.float64) * dt
            exact = bool(batch["terminated"][end])
            outcome = ("complete_success" if exact and episode["success"] else
                       "complete_failure" if exact else "sampling_cut")
            case = episode["case"]
            case_index = config["cases"].index(case)
            if not np.all(batch["case_index"][segment] == case_index):
                raise ValueError(f"Saved case identity disagrees: {relative}/{case}")
            spec = episode["task"]
            goal = np.array([spec["straight_length"] + spec["turn_radius"], spec["goal_y"]])
            distance = np.linalg.norm(state[:, :2] - goal, axis=1)
            speed = np.linalg.norm(state[:, 6:8], axis=1)
            yaw_error = np.abs(np.arctan2(np.sin(state[:, 5] - np.pi / 2),
                                         np.cos(state[:, 5] - np.pi / 2)))
            stage = memory[:, 0].astype(int)
            near = (stage == 3) & (distance <= 0.35)
            in_position = (stage == 3) & (distance <= spec["goal_position_tolerance"])
            settled = (in_position & (speed <= spec["goal_speed_tolerance"])
                       & (yaw_error <= spec["goal_yaw_tolerance"]))
            masks = {"all": np.ones(len(values), dtype=bool)}
            masks.update({f"stage_{index}_{name}": stage == index for index, name in STAGES.items()})
            masks.update({
                "stop_farther_than_0p35m": (stage == 3) & ~near,
                "stop_within_0p35m": near,
                "stop_within_position_tolerance": in_position,
                "stop_near_unsettled": near & ~settled,
                "stop_settled_now": settled,
            })
            base = {
                "cohort": cohort, "phase": phase, "seed": seed, "arm": arm,
                "round": round_number, "case": case, "episode": episode["episode"],
                "outcome": outcome, "reason": episode["reason"], "critic_timing": timing,
                "source": relative + "/transitions.pt",
            }
            all_metrics = metrics(values, suffix, remaining, exact)
            episode_rows.append({
                **base, "recorded_steps": len(rewards),
                "observed_episode_seconds": float(episode["elapsed_seconds"]),
                "true_completion_seconds": (
                    float(episode["completion_seconds"]) if outcome == "complete_success" else None),
                "saved_reward_sum": float(suffix[0]), "saved_summary_return": episode["real_return"],
                "initial_recorded_V": float(values[0]),
                "initial_predicted_remaining_cost_seconds": float(-values[0]),
                "exact_full_suffix_available": exact, **all_metrics,
            })
            for region, mask in masks.items():
                if not np.any(mask):
                    continue
                result = metrics(values[mask], suffix[mask], remaining[mask], exact)
                row = {**base, "region": region, "observed_region_seconds": float(mask.sum() * dt),
                       **result}
                region_rows.append(row)
                key = (cohort, phase, arm, outcome, region)
                groups[key].append({
                    "seed": seed, "case": case, "episode": episode["episode"],
                    "values": values[mask], "suffix": suffix[mask],
                    "remaining": remaining[mask], "exact": exact,
                    "bias": result["V_minus_actual_return_bias"],
                })
                if region in ("stage_3_stop", "stop_within_0p35m", "stop_within_position_tolerance",
                              "stop_settled_now"):
                    index = int(np.flatnonzero(mask)[0])
                    entry_rows.append({
                        **base, "region": region, "first_pre_action_time_seconds": float(memory[index, 1]),
                        "recorded_V": float(values[index]),
                        "predicted_remaining_cost_seconds": float(-values[index]),
                        "actual_seconds_to_episode_boundary": float(remaining[index]),
                        "observed_reward_suffix": float(suffix[index]),
                        "V_minus_actual_return": float(values[index] - suffix[index]) if exact else None,
                        "distance_m": float(distance[index]), "planar_speed_m_s": float(speed[index]),
                        "yaw_error_rad": float(yaw_error[index]), "hold_elapsed_seconds": float(memory[index, 2]),
                        "macro_phase": float(batch["horizon"][start + index] / 10.0),
                    })
            start = stop
    grouped_rows = []
    for key, entries in sorted(groups.items()):
        cohort, phase, arm, outcome, region = key
        v = np.concatenate([entry["values"] for entry in entries])
        g = np.concatenate([entry["suffix"] for entry in entries])
        remaining = np.concatenate([entry["remaining"] for entry in entries])
        biases = [entry["bias"] for entry in entries if entry["bias"] is not None]
        grouped_rows.append({
            "cohort": cohort, "phase": phase, "arm": arm, "outcome": outcome, "region": region,
            "episodes_with_region": len(entries),
            "seeds_present": json.dumps(sorted({entry["seed"] for entry in entries})),
            "equal_episode_mean_V_bias": float(np.mean(biases)) if biases else None,
            **metrics(v, g, remaining, all(entry["exact"] for entry in entries)),
        })
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "collections.csv", collection_rows)
    write_csv(output / "episodes.csv", episode_rows)
    write_csv(output / "episode_region_metrics.csv", region_rows)
    write_csv(output / "group_metrics.csv", grouped_rows)
    write_csv(output / "region_entry_values.csv", entry_rows)
    evaluation = [row for row in episode_rows if row["phase"] == "evaluation"]
    behavior = [row for row in episode_rows if row["phase"] == "behavior"]
    summary = {
        "schema": 1,
        "status": "SAVED_VALUE_DIAGNOSIS_COMPLETE",
        "evaluation_episode_count": len(evaluation),
        "evaluation_by_cohort": {
            name: sum(row["cohort"] == name for row in evaluation)
            for name in ("initial_reference", "final", "seed11_round3")},
        "behavior_episode_count": len(behavior),
        "behavior_outcomes": {
            outcome: sum(row["outcome"] == outcome for row in behavior)
            for outcome in ("complete_success", "complete_failure", "sampling_cut")},
        "final_evaluation_stop_metrics": [
            row for row in grouped_rows
            if row["cohort"] == "final" and row["phase"] == "evaluation"
            and row["region"] in ("stage_3_stop", "stop_within_0p35m", "stop_settled_now")],
        "interpretation": [
            "No network/checkpoint inference: only recorded old_value and actual reward/state tensors are read.",
            "V is trained on stochastic B; deterministic evaluation suffix errors diagnose deployment calibration, not a miscomputed training target.",
            "Behavior old_value is from before the current B update; evaluation old_value is after that value update. Their difference is not a same-critic distribution experiment.",
            "Complete success suffixes contain time cost only, so -V can be compared with actual remaining seconds. For failures, -V and -return include the 70-second-equivalent failure penalty.",
            "Sampling cuts have only observed partial suffix sums. Full-return bias/RMSE are null; no bootstrap is declared ground truth.",
            "Positive V means negative predicted remaining cost despite the nonpositive recorded reward objective; optimistic bias is V minus realized suffix return.",
            "Group bias/RMSE are transition-weighted descriptive diagnostics; equal_episode_mean_V_bias is reported separately. No independent-sample inference or confidence intervals.",
            "The 0.35-m region is Euclidean distance to the goal, not the planner's projected remaining-route coordinate; the position-tolerance and settled regions use original TaskSpec thresholds.",
            "Round zero uses one frozen-arm shared-initialization reference per seed; duplicated arm starts are not counted as additional episodes.",
            "Recorded-state V calibration does not show that V affected a planner decision. Predicted-complete candidate terminal values can be masked to zero.",
        ],
        "model_forward_calls": 0, "gradient_calls": 0, "optimizer_updates": 0,
        "new_physics_steps": 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {
        "script": "scripts/diagnose_go2_value_saved.py",
        "script_sha256_lf": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "input_sha256": fingerprints,
        "tensor_loading": "torch.load(map_location='cpu', weights_only=True), saved transition tensors only",
        "checkpoint_files_read": 0, "model_forward_calls": 0,
        "optimizer_updates": 0, "new_physics_steps": 0,
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = diagnose(args.runs, args.output)
    print(json.dumps({key: summary[key] for key in
                      ("status", "evaluation_episode_count", "evaluation_by_cohort",
                       "behavior_episode_count", "behavior_outcomes", "new_physics_steps")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
