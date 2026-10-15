"""Post-hoc frozen-checkpoint actor/response swap; no learning or selection.

Only the two off-diagonal combinations are evaluated. Diagonal evaluations are
read from the completed pilot results. This diagnostic is additional, charged
interaction and cannot retroactively become a preregistered success criterion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch

from .dynamics import PhysicalConfig, response_design
from .learning import Actor, JointLearner, UpdateConfig, execution_features, parameter_vector
from .pilot import average_metrics, rollout


ROOT = Path(__file__).resolve().parents[2]
GRID = {
    "position": [-0.3, 0.0, 0.3],
    "velocity": [-0.5, 0.0, 0.5],
    "actuator_output": [-0.8, 0.0, 0.8],
    "command": [-0.35, 0.0, 0.35],
}
SUM_KEYS = ("physical_steps", "planner_batch_calls", "planner_instance_calls",
            "candidate_sequences", "objective_calls", "response_transitions", "wall_seconds")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frozen_seed(formal, seed):
    run = formal / f"seedrun-{seed}"
    specification = json.loads((run / "specification.json").read_text(encoding="utf-8"))
    if specification["test_batches"] != 8 or specification["batch"] != 8 or specification["episode_steps"] != 100:
        raise ValueError("Cross diagnostic requires the frozen 8x8x100 evaluation protocol")
    for name, expected in specification["source_sha256"].items():
        if file_hash(Path(__file__).parent / name) != expected:
            raise ValueError(f"Frozen pilot source changed: {name}")
    checkpoints, results, provenance = {}, {}, {}
    for label, arm in (("J", "joint"), ("T", "tuning_only")):
        folder = run / f"seed-{seed}" / arm
        checkpoint = folder / "checkpoint.pt"
        result_file = folder / "result.json"
        checkpoints[label] = torch.load(checkpoint, map_location="cpu", weights_only=True)
        results[label] = json.loads(result_file.read_text(encoding="utf-8"))
        provenance[label] = {
            "checkpoint_relative": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": file_hash(checkpoint),
            "result_relative": result_file.relative_to(ROOT).as_posix(),
            "result_sha256": file_hash(result_file),
        }
        if not torch.allclose(checkpoints[label]["mean"],
                              torch.tensor(results[label]["mean"]), atol=0, rtol=0):
            raise ValueError("Checkpoint response does not match the recorded evaluated response")
    if checkpoints["J"]["config"] != checkpoints["T"]["config"]:
        raise ValueError("Actor/response swaps require identical update/constraint configuration")
    return checkpoints, results, provenance


def instantiate(actor_checkpoint, response_checkpoint):
    actor = Actor()
    actor.load_state_dict(actor_checkpoint["actor"])
    actor.eval()
    return JointLearner(actor, response_checkpoint["mean"].clone(), "diagnostic",
                        UpdateConfig(**actor_checkpoint["config"]))


def action_map_diagnostic(checkpoints):
    grid = torch.tensor(list(product(*GRID.values())), dtype=torch.float32)
    assert grid.shape == (81, 4)
    features = execution_features(grid[:, :3], grid[:, 3:4])
    actions = {}
    parameter_values = {}
    with torch.no_grad():
        for label in ("J", "T"):
            learner = instantiate(checkpoints[label], checkpoints[label])
            actions[label] = PhysicalConfig().U * torch.tanh(learner.actor(features))
            parameter_values[label] = parameter_vector(learner.actor)
    delta = actions["J"] - actions["T"]
    stats = {
        "points": 81,
        "physical_interactions": 0,
        "actor_forward_point_evaluations": 162,
        "action_rms_difference": float(delta.square().mean().sqrt()),
        "action_max_absolute_difference": float(delta.abs().max()),
        "action_mean_absolute_difference": float(delta.abs().mean()),
        "points_differing_above_1e_6": int((delta.abs() > 1e-6).sum()),
        "actor_parameter_l2_difference": float(torch.linalg.vector_norm(parameter_values["J"]-parameter_values["T"])),
        "input_limit": PhysicalConfig().U,
        "interpretation": "Different physical actions on common observations refute identical-function parameter relabeling on these probes; they do not prove useful coadaptation.",
    }
    raw = {"grid": grid.tolist(), "J_actions": actions["J"].tolist(),
           "T_actions": actions["T"].tolist(), "difference": delta.tolist()}
    return stats, raw


def run_seed(formal, seed, output):
    checkpoints, diagonals, provenance = read_frozen_seed(formal, seed)
    result = {
        "seed": seed,
        "checkpoint_provenance": provenance,
        "diagonal_status": "read_existing_evaluation_not_rerun",
        "diagonals": {"JJ": diagonals["J"]["evaluation"],
                      "TT": diagonals["T"]["evaluation"]},
        "T_actor_change_norm_recorded": diagonals["T"]["actor_change_norm"],
        "raw_response_logits": {key: checkpoints[key]["mean"].tolist() for key in checkpoints},
        "bounded_response_parameters": {key: response_design(checkpoints[key]["mean"]).tolist()
                                        for key in checkpoints},
        "off_diagonals": {},
    }
    for label, actor_label, response_label in (("JT", "J", "T"), ("TJ", "T", "J")):
        learner = instantiate(checkpoints[actor_label], checkpoints[response_label])
        actor_before = parameter_vector(learner.actor)
        response_before = learner.mean.clone()
        evaluations = []
        per_batch = []
        for index in range(8):
            test_seed = 500000 + seed*1000 + index
            evaluation = rollout(learner, test_seed, deterministic=True, batch=8, steps=100,
                                 update=False, trace=False)
            evaluations.append(evaluation)
            per_batch.append({"batch_index": index, "rollout_seed": test_seed,
                              "metrics": evaluation["metrics"]})
        metrics = average_metrics(evaluations)
        assert metrics["physical_steps"] == 6400
        assert torch.equal(actor_before, parameter_vector(learner.actor))
        assert torch.equal(response_before, learner.mean)
        result["off_diagonals"][label] = {
            "actor_source": actor_label, "response_source": response_label,
            "metrics": metrics, "per_batch": per_batch,
            "parameter_updates": 0,
        }
        print(json.dumps({"seed": seed, "combination": label, **metrics}), flush=True)
    grid_stats, grid_raw = action_map_diagnostic(checkpoints)
    result["action_map"] = grid_stats
    result["action_map_raw"] = grid_raw
    matrix = {**result["diagonals"],
              **{key: value["metrics"] for key, value in result["off_diagonals"].items()}}
    result["contrasts"] = {}
    for metric in ("task_cost", "input_response_mse"):
        value = {key: matrix[key][metric] for key in ("JJ", "JT", "TJ", "TT")}
        result["contrasts"][metric] = {
            "cells": value,
            "interaction_JJ_minus_JT_minus_TJ_plus_TT": value["JJ"]-value["JT"]-value["TJ"]+value["TT"],
            "J_actor_effect_with_T_response": value["JT"]-value["TT"],
            "J_actor_effect_with_J_response": value["JJ"]-value["TJ"],
            "J_response_effect_with_T_actor": value["TJ"]-value["TT"],
            "J_response_effect_with_J_actor": value["JJ"]-value["JT"],
            "combined_JJ_minus_TT": value["JJ"]-value["TT"],
        }
    result["additional_budget"] = {
        key: sum(value["metrics"][key] for value in result["off_diagonals"].values())
        for key in SUM_KEYS
    }
    result["additional_budget"].update(actor_grid_point_evaluations=162, training_updates=0,
                                       reused_diagonal_physical_steps_charged_here=0)
    (output / f"seed-{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, default=ROOT/"work/learning-pilot-2026-10-15/formal")
    parser.add_argument("--output", type=Path, default=ROOT/"work/learning-pilot-2026-10-15/cross-diagnostic")
    parser.add_argument("--public", type=Path, default=ROOT/"notes/learning-pilot-2026-10-15/results/cross-diagnostic.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"summary.json").exists():
        raise FileExistsError("Completed diagnostic exists; use a different output path for any explicitly charged rerun")
    started = time.perf_counter()
    results = [run_seed(args.formal, seed, args.output) for seed in args.seeds]
    extra_budget = {key: sum(result["additional_budget"][key] for result in results) for key in SUM_KEYS}
    extra_budget.update(actor_grid_point_evaluations=162*len(results), training_updates=0,
                        physical_integrator_substeps=extra_budget["physical_steps"]*PhysicalConfig().substeps,
                        reused_diagonal_physical_steps_charged_here=0)
    assert extra_budget["physical_steps"] == len(args.seeds)*12800
    interactions = {}
    for metric in ("task_cost", "input_response_mse"):
        values = [result["contrasts"][metric]["interaction_JJ_minus_JT_minus_TJ_plus_TT"] for result in results]
        interactions[metric] = {"per_seed": values, "mean": float(np.mean(values)),
                                "sample_std": float(np.std(values, ddof=1)) if len(values)>1 else None,
                                "negative_count": sum(value<0 for value in values)}
    summary = {
        "status": "post_hoc_mechanism_diagnostic_no_method_change",
        "interpretation": "Costs are lower-is-better. Negative JJ-JT-TJ+TT is favorable interaction for these frozen factors; neither sign proves a universal coupling theorem or satisfies an unmet feasibility constraint.",
        "seeds": args.seeds,
        "actor_rows": {"J": "joint trained actor", "T": "tuning_only frozen common actor"},
        "response_columns": {"J": "joint selected response", "T": "tuning_only selected response"},
        "grid_axes": GRID,
        "physical_configuration": asdict(PhysicalConfig()),
        "source_sha256": file_hash(Path(__file__)),
        "budget": extra_budget,
        "total_wall_seconds": time.perf_counter()-started,
        "interaction_summary": interactions,
        "per_seed": results,
    }
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    public = {key: value for key, value in summary.items() if key != "per_seed"}
    public["per_seed"] = [{key: value for key, value in result.items()
                           if key not in ("action_map_raw", "checkpoint_provenance")}
                          for result in results]
    for result in public["per_seed"]:
        result["off_diagonals"] = {key: {name: val for name, val in cell.items() if name != "per_batch"}
                                   for key, cell in result["off_diagonals"].items()}
    args.public.parent.mkdir(parents=True, exist_ok=True)
    args.public.write_text(json.dumps(public, indent=2), encoding="utf-8")
    print(json.dumps({"budget": extra_budget, "interaction_summary": interactions,
                      "action_grid_rms": [r["action_map"]["action_rms_difference"] for r in results]}), flush=True)


if __name__ == "__main__":
    main()
