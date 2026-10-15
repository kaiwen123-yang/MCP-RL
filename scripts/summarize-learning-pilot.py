"""Publish compact pilot evidence without weights, PDFs or private local paths."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    roots = sorted(args.source.glob("seedrun-*/summary.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in roots]
    if len(runs) != 3:
        raise ValueError("This frozen comparison requires all three completed seed runs")
    hashes = runs[0]["specification"]["source_sha256"]
    if any(run["specification"]["source_sha256"] != hashes for run in runs):
        raise ValueError("Source code changed between formal seed launches")
    args.destination.mkdir(parents=True, exist_ok=True)
    rows, ledger, curves, selections, shared = [], [], [], [], []
    for run in runs:
        seed = run["seeds"][0]
        sid = seed["seed"]
        ledger.append({"seed": sid, "mode": "shared", "phase": "teacher",
                       "physical_steps": seed["warmstart"]["teacher_physical_steps"],
                       "supervised_updates": seed["warmstart"]["supervised_updates"]})
        for phase, metrics in [("initial_selection", seed["initial_selection"]["metrics"]),
                               ("common_training", seed["common_training"])]:
            ledger.append({"seed": sid, "mode": "shared", "phase": phase, **metrics})
        shared.append({"seed": sid, "warmstart": seed["warmstart"],
                       "initial_selection": seed["initial_selection"], "common_training": seed["common_training"]})
        for mode, arm in seed["arms"].items():
            rows.append({"seed": sid, "mode": mode, **arm["evaluation"],
                         "validation_mean_feasible": arm["selection"]["validation_mean_feasible"],
                         "actor_change_norm": arm["actor_change_norm"]})
            selections.append({"seed": sid, "mode": mode, "raw_response": arm["mean"],
                               "selection": arm["selection"]})
            for phase in ["training", "calibration", "selection", "evaluation"]:
                metrics = arm[phase]["metrics"] if phase == "selection" else arm[phase]
                ledger.append({"seed": sid, "mode": mode, "phase": phase, **metrics})
            histories = arm["history"] if mode == "fixed_search" else [arm["history"]]
            for candidate, history in enumerate(histories):
                for item in history:
                    curves.append({"seed": sid, "mode": mode, "candidate": candidate,
                                   "update": item["update"], "task_cost": item["task_cost"],
                                   "tracking_constraint": item["constraints"][0],
                                   "input_constraint": item["constraints"][1],
                                   "alpha": item["alpha"], "actor_step_norm": item["actor_step_norm"],
                                   "response_step_norm": item["response_step_norm"],
                                   "linear_tracking_after": item["linear_constraint_after"][0],
                                   "linear_input_after": item["linear_constraint_after"][1]})
    aggregate = {}
    for mode in runs[0]["seeds"][0]["arms"]:
        values = [row for row in rows if row["mode"] == mode]
        aggregate[mode] = {}
        for key in ["task_cost", "realization_mse", "input_response_mse",
                    "input_episode_violation_fraction", "nominal_infeasible_fraction", "actor_change_norm"]:
            data = [row[key] for row in values]
            aggregate[mode][key] = {"mean": statistics.mean(data), "sample_sd": statistics.stdev(data), "values": data}
        aggregate[mode]["validation_mean_feasible_seeds"] = sum(row["validation_mean_feasible"] for row in values)
    comparisons = {}
    for mode in aggregate:
        if mode == "joint":
            continue
        j = aggregate["joint"]["task_cost"]["values"]
        b = aggregate[mode]["task_cost"]["values"]
        comparisons[mode] = {"joint_relative_cost_change_percent": [100*(x-y)/y for x, y in zip(j, b)],
                             "joint_lower_cost_seeds": sum(x < y for x, y in zip(j, b))}
    total_steps = sum(row["physical_steps"] for row in ledger)
    if total_steps != 1169088:
        raise ValueError(f"Unexpected formal physical interaction ledger: {total_steps}")
    summary = {"scope": "Frozen small nonlinear servo comparison, empirical constraints only",
               "seeds": [run["seeds"][0]["seed"] for run in runs],
               "unique_physical_steps": total_steps,
               "total_candidate_sequences": sum(row.get("candidate_sequences", 0) for row in ledger),
               "total_planner_response_transitions": sum(row.get("response_transitions", 0) for row in ledger),
               "id_model_transitions_additional": 3*6*100*64*10,
               "run_wall_seconds": [run["wall_seconds"] for run in runs],
               "time_note": "Three seed processes ran concurrently, CPU only; not matched wall-time or NX benchmark",
               "aggregate": aggregate, "paired_comparisons": comparisons,
               "same_source_hashes_all_seeds": True}
    for name, content in [("summary.json", summary), ("selections.json", selections),
                          ("shared-initialization.json", shared),
                          ("frozen-specifications.json", [run["specification"] for run in runs])]:
        (args.destination/name).write_text(json.dumps(content, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_csv(args.destination/"per-seed.csv", rows)
    write_csv(args.destination/"resource-ledger.csv", ledger)
    write_csv(args.destination/"learning-curves.csv", curves)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
