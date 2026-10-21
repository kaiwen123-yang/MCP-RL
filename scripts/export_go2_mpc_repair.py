"""Export the actual full-mission attribution and repair outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def episode(row):
    result = {key: row[key] for key in ("case", "success", "reason", "elapsed_seconds",
              "completion_seconds", "mechanical_work_proxy_J", "path_rms_m", "task")}
    result["crossings"] = [{"from": event["from"], "to": event["to"], "time": event["time"],
                            "position": event["state"][:3], "rpy": event["state"][3:6],
                            "world_velocity": event["state"][6:9]} for event in row["stage_exits"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"date": "2026-10-21", "status": "COMPLETED", "optimization_updates": 0,
              "task_and_success_criteria_changed": False, "follower_fallback": False,
              "default_response_mode": "prior", "optional_response_mode": "local_response",
              "baseline_source": "go2-continuation-corrected-evaluation-2026-10-21.json",
              "saved_prediction_errors": json.loads((args.run_root / "saved_prediction_errors.json").read_text())}
    total_physics, total_seconds = 0, 0.
    for name, relative in (("attribution", "source/work/attribution"),
                           ("local_learned_response", "local-source/work/validation"),
                           ("default_mpc", "production-source/work/validation")):
        root = args.run_root / relative
        data = json.loads((root / "comparison.json").read_text())
        if data["status"] != "COMPLETED":
            raise ValueError(f"Unfinished full-mission batch: {name}")
        total_seconds += data["wall_seconds"]
        group = {"protocol": json.loads((root / "protocol.json").read_text()),
                 "wall_seconds": data["wall_seconds"], "records": []}
        for record in data["records"]:
            item = {key: record[key] for key in ("name", "options", "parameter_hashes",
                    "parameters_unchanged", "stats", "wall_seconds")}
            item["episodes"] = [episode(e) for e in record["episodes"]]
            saved = json.loads((root / record["name"] / "evaluation/summary.json").read_text())
            times = [step["solve_seconds"] for episode_ in saved["decisions"] for step in episode_["decisions"]]
            item["planning_seconds_median"] = float(np.median(times))
            item["planning_seconds_p95"] = float(np.percentile(times, 95))
            total_physics += record["stats"]["warmup_and_execution_physics_steps"]
            group["records"].append(item)
        result[name] = group
    result["all_physics_steps"] = total_physics
    result["complete_mission_execution_seconds"] = total_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("all_physics_steps", "complete_mission_execution_seconds")}))
    for group in ("local_learned_response", "default_mpc"):
        for record in result[group]["records"]:
            print(group, record["name"], [round(e["elapsed_seconds"], 2) for e in record["episodes"]],
                  'planning_p95_ms', round(1000 * record['planning_seconds_p95'], 3))


if __name__ == "__main__":
    main()
