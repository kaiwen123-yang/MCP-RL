"""Frozen-design interval-midpoint comparator: zero learning, original tasks.

Waits only for completed support-arm artifacts. Does not modify the frozen
training files or repeat the learned-policy evaluation diagonals.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch

from .env import EnvironmentConfig
from .pilot import collect, average_metrics, source_hashes
from .ppo_backend import ActionSample


class MidpointController:
    """The same support map with raw action fixed to zero, no network call."""
    def __init__(self, device="cpu"):
        self.config = SimpleNamespace(device=device)

    def act(self, obs, deterministic=True, generator=None):
        raw = obs.new_zeros(obs.shape[0], 1)
        scalar = obs.new_zeros(obs.shape[0])
        return ActionSample(action=raw, raw_action=raw, log_prob=scalar, value=scalar)


def evaluate(formal_root: Path, output_root: Path, seed: int, arm: str):
    source_dir = formal_root/f"seedrun-{seed}"/f"seed-{seed}"
    checkpoint_path, arm_path = source_dir/f"{arm}.pt", source_dir/f"{arm}.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    learned = json.loads(arm_path.read_text(encoding="utf-8"))
    formal_manifest = json.loads((source_dir.parent/"manifest.json").read_text(encoding="utf-8"))
    actual_hashes = source_hashes()
    if actual_hashes != formal_manifest["source_hashes_before_run"]:
        raise RuntimeError("Frozen simulator/learner source differs from the formal run.")
    design = checkpoint["design_mean"]
    backend = MidpointController()
    env_config = EnvironmentConfig()
    evaluation = {}
    started = time.perf_counter()
    for label, steps in (("short", 256), ("long", 1024)):
        batches = []
        for batch in range(4):
            _, metrics, _, _ = collect(backend, design, arm, seed*10000+5000+batch,
                                       steps, 16, False, env_config)
            batches.append(metrics)
        average = average_metrics(batches)
        old = learned["evaluation"][label]["mean"]
        evaluation[label] = {
            "midpoint": average,
            "learned": old,
            "midpoint_minus_learned_task_cost": average["task_cost"]-old["task_cost"],
            "learned_improvement_relative_midpoint_percent": 100*(average["task_cost"]-old["task_cost"])/average["task_cost"],
            "paired_batch_task_differences": [m["task_cost"]-l["task_cost"] for m,l in zip(batches, learned["evaluation"][label]["batches"])],
            "batches": batches,
        }
    result = {
        "seed": seed, "arm_design": arm, "controller": "raw action zero -> exact feasible-interval midpoint",
        "design_mean": design.tolist(), "environment": asdict(env_config),
        "source_hashes": actual_hashes,
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "evaluation": evaluation,
        "budget": {"physical_steps": 4*16*(256+1024), "learning_steps": 0, "neural_policy_forward_calls": 0,
                   "wall_seconds": time.perf_counter()-started,
                   "candidate_sequences": sum(m["planner_counts"]["candidate_sequences"] for e in evaluation.values() for m in e["batches"]),
                   "candidate_low_level_model_steps": sum(m["planner_low_level_model_steps"] for e in evaluation.values() for m in e["batches"])},
        "interpretation": "Conditions on each learned design. Tests added value of its learned actor versus analytic interval midpoint; no new training or parameter search.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root/f"seed-{seed}-{arm}.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def write_summary(results, public_path):
    summary = {
        "status": "completed" if len(results)==6 else "partial",
        "scope": "Pre-registered frozen-design midpoint comparator; zero learning; learned diagonals reused",
        "combinations": [],
        "budget": {"physical_steps": sum(r["budget"]["physical_steps"] for r in results), "learning_steps": 0,
                   "candidate_sequences": sum(r["budget"]["candidate_sequences"] for r in results),
                   "candidate_low_level_model_steps": sum(r["budget"]["candidate_low_level_model_steps"] for r in results),
                   "wall_seconds_sum": sum(r["budget"]["wall_seconds"] for r in results)},
        "timing_note": "Concurrent with formal CPU runs; not an isolated speed benchmark",
    }
    for result in sorted(results, key=lambda r:(r["seed"], r["arm_design"])):
        row = {"seed": result["seed"], "arm_design": result["arm_design"], "design_mean": result["design_mean"], "budget": result["budget"]}
        for label, evaluation in result["evaluation"].items():
            row[label] = {key: value for key,value in evaluation.items() if key != "batches"}
        summary["combinations"].append(row)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=Path("work/ppo-mechanism-2026-10-16/formal"))
    parser.add_argument("--output", type=Path, default=Path("work/ppo-mechanism-2026-10-16/midpoint"))
    parser.add_argument("--public-summary", type=Path, default=Path("notes/ppo-mechanism-2026-10-16/results/midpoint-summary.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    pending = {(seed,arm) for seed in (17,29,43) for arm in ("joint_support","fixed_support")}
    results = []
    while pending:
        completed_now = []
        for seed,arm in sorted(pending):
            cached = args.output/f"seed-{seed}-{arm}.json"
            directory = args.formal_root/f"seedrun-{seed}"/f"seed-{seed}"
            if cached.exists():
                result = json.loads(cached.read_text(encoding="utf-8"))
            elif (directory/f"{arm}.pt").exists() and (directory/f"{arm}.json").exists():
                result = evaluate(args.formal_root, args.output, seed, arm)
            else:
                continue
            results.append(result)
            completed_now.append((seed,arm))
            write_summary(results, args.public_summary)
            print(json.dumps({"completed": [seed,arm], "short_improvement_percent": result["evaluation"]["short"]["learned_improvement_relative_midpoint_percent"],
                              "long_improvement_percent": result["evaluation"]["long"]["learned_improvement_relative_midpoint_percent"]}), flush=True)
        pending.difference_update(completed_now)
        if pending:
            time.sleep(60)  # one artifact-availability check per minute


if __name__ == "__main__":
    main()
