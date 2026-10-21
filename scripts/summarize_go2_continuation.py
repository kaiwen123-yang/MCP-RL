"""Export full-task outcomes and actual costs without local asset paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def episodes(rows):
    keys = ("case", "success", "reason", "complete", "completion_seconds", "elapsed_seconds",
            "real_return", "mechanical_work_proxy_J", "torque_squared_integral", "path_rms_m",
            "policy_steps", "stage_exits", "task")
    return [{key: row[key] for key in keys} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.comparison.read_text())
    if raw["status"] != "COMPLETED":
        raise ValueError("Wait for the complete fixed comparison before publishing outcomes")
    result = {"date": "2026-10-21", "status": raw["status"], "seed": raw["seed"],
              "protocol": raw["config"], "source_sha256": raw["source_sha256"],
              "wall_seconds": raw["wall_seconds"],
              "common_calibration": raw["common_calibration"], "arms": {}}
    total_physics = raw["common_calibration"]["warmup_and_execution_physics_steps"]
    actor_samples = 0
    for arm, record in raw["arms"].items():
        result["arms"][arm] = {key: record[key] for key in
                              ("ppo", "facts", "derivatives", "behavior", "calibration", "evaluation",
                               "parameter_hashes", "initial_parameter_hashes", "wall_seconds")}
        result["arms"][arm]["episodes"] = episodes(record["evaluation_episodes"])
        if arm != "frozen":
            actor_samples += record["ppo"]["samples"]
        total_physics += sum(record[phase]["warmup_and_execution_physics_steps"]
                             for phase in ("behavior", "calibration", "evaluation"))
    result["follower"] = {"stats": raw["follower"]["stats"],
                           "episodes": episodes(raw["follower"]["episodes"])}
    total_physics += raw["follower"]["stats"]["warmup_and_execution_physics_steps"]
    result["all_physics_steps"] = total_physics
    result["actor_training_samples_total"] = actor_samples
    result["all_stochastic_samples"] = sum(r["ppo"]["samples"] for r in raw["arms"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("arm\tcase\toutcome\tseconds\twork_J")
    for arm, record in [*result["arms"].items(), ("follower", result["follower"])]:
        for episode in record["episodes"]:
            print(f"{arm}\t{episode['case']}\t{episode['reason']}\t{episode['elapsed_seconds']:.2f}\t{episode['mechanical_work_proxy_J']:.2f}")
    print(json.dumps({key: result[key] for key in
                      ("wall_seconds", "all_physics_steps", "actor_training_samples_total", "all_stochastic_samples")}))


if __name__ == "__main__":
    main()
