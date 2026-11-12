#!/usr/bin/env python3
"""Read-only comparison of retained R8/S3 and W45/r34 resolved records.

No simulator, controller, network, or server experiments are run. The output
distinguishes identical recorded robot fields from different task/controller
semantics; equal MuJoCo version strings do not establish binary equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


R8_REL = Path("results/terrain_memory/simulation/tm-s1-go2-wb-support-r8")
W45_REL = Path("results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
R8_NAME = "project-s3-single-05-resolved.json"
W45_NAME = "job-20261112T103825Z-w45-a-eval-PANN_PPO_S5-s1-E30-attempt01-resolved.json"

ROBOT_FIELDS = (
    "mass_kg", "compiled_mass_including_mocap_kg", "initial_qpos", "initial_qvel",
    "joint_names", "joint_ranges", "actuator_gear", "actuator_gainprm",
    "actuator_biasprm", "actuator_gaintype", "actuator_biastype",
    "actuator_ctrlrange", "actuator_ctrllimited", "actuator_forcerange",
    "actuator_forcelimited", "head_offset", "physics_dt_s",
    "logical_feedback_dt_s", "logical_planning_dt_s",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(a: dict, b: dict) -> list[dict]:
    rows = []
    for key in sorted(a.keys() | b.keys()):
        row = {"field": key, "r8_present": key in a, "w45_present": key in b}
        row["equal"] = key in a and key in b and a[key] == b[key]
        if key in a:
            row["r8"] = a[key]
        if key in b:
            row["w45"] = b[key]
        rows.append(row)
    return rows


def first_gait(record: dict, gait: str) -> dict:
    return next(row for row in record["applied_settings"]
                if row["task_parameters"]["Gait"] == gait)


def reference_summary(record: dict) -> dict:
    ref = record["stand_references"][0]
    fields = ("body_height_m", "relative_height_m", "requested_height_m",
              "reference_kind", "feasible", "entry_time_s",
              "max_equation_residual", "qacc_norm",
              "pd_reference_identity_max_error_rad", "foot_force")
    out = {key: ref[key] for key in fields}
    out["moving_relative_height_goal_m"] = 0.27
    out["moving_goal_minus_stand_relative_height_m"] = 0.27 - ref["relative_height_m"]
    out["moving_goal_source"] = "Quadruped kHeightQuadruped; source reviewed separately"
    return out


def results_summary(repo: Path) -> dict:
    records = []
    fields = ("condition", "success", "original_mission_success",
              "sustained_execution_success", "simulation_s", "first_completion_s",
              "completed_round_trips", "min_body_height_m", "max_x_m", "reason")
    for path in sorted((repo / R8_REL).glob("project-s3*.json")):
        if path.name.endswith(("-resolved.json", "-evidence.json")):
            continue
        data = load(path)
        records.append({"file": str(path.relative_to(repo)), "sha256": digest(path),
                        "mission": data["mission"],
                        **{key: data.get(key) for key in fields}})
    counts = {}
    for mission in ("single", "long"):
        subset = [row for row in records if row["mission"] == mission]
        counts[mission] = {
            "total": len(subset),
            "success": sum(row["success"] is True for row in subset),
            "sustained_execution_success": sum(
                row["sustained_execution_success"] is True for row in subset),
        }
    return {"counts": counts, "records": records,
            "scope": "Original G1 rigid task; these are not TM02 qualifications."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    r8_path, w45_path = repo / R8_REL / R8_NAME, repo / W45_REL / W45_NAME
    a, b = load(r8_path), load(w45_path)
    robot = [{"field": field, "equal": a[field] == b[field],
              "r8": a[field], "w45": b[field]} for field in ROBOT_FIELDS]
    trot_a, trot_b = first_gait(a, "Trot"), first_gait(b, "Trot")
    source_keys = sorted(a["author_source_sha256"].keys() | b["author_source_sha256"].keys())
    out = {
        "schema": "r8_w45_resolved_comparison_v1",
        "method": "JSON exact equality for named fields; no simulation or causal inference.",
        "inputs": [{"file": str(path.relative_to(repo)), "sha256": digest(path),
                    "source_commit": data["source_commit"],
                    "author_commit": data["author_commit"],
                    "execution_fingerprint": data["execution_fingerprint"]}
                   for path, data in ((r8_path, a), (w45_path, b))],
        "robot_and_clock_fields": robot,
        "robot_and_clock_all_equal": all(row["equal"] for row in robot),
        "recorded_runtime": paired(
            {key: a[key] for key in ("mujoco_version", "workers", "synchronous_slow_simulation")},
            {key: b[key] for key in ("mujoco_version", "workers", "synchronous_slow_simulation")}),
        "prediction_numerics": paired(a["prediction_settings"]["numerics"],
                                      b["prediction_settings"]["numerics"]),
        "first_trot_time_s": {"r8": trot_a["time_s"], "w45": trot_b["time_s"]},
        "first_trot_task_parameters": paired(trot_a["task_parameters"], trot_b["task_parameters"]),
        "first_trot_cost_weights": paired(trot_a["cost_weights"], trot_b["cost_weights"]),
        "author_planner_defaults": paired(a["author_planner_defaults"], b["author_planner_defaults"]),
        "first_stand_reference": {"r8": reference_summary(a), "w45": reference_summary(b)},
        "retained_source_hashes": paired(
            {key: a["author_source_sha256"][key] for key in source_keys if key in a["author_source_sha256"]},
            {key: b["author_source_sha256"][key] for key in source_keys if key in b["author_source_sha256"]}),
        "r8_s3_results": results_summary(repo),
        "limitations": [
            "Matching robot fields do not mean equal terrain or full controller semantics.",
            "The MuJoCo version label is not a shared-library or ABI hash comparison.",
            "Equal cost weights do not imply equal residual definitions or activation.",
            "W45 prediction_settings agent_differentiable=1 is inherited metadata; r33/r34 model roles override the actual forward/derivative path.",
            "R8 rigid-task success is not success on the original TM02 strain .02 task.",
            "The Stand reference is not a complete dynamically feasible moving-gait reference.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "robot_and_clock_fields": len(robot),
                      "robot_and_clock_all_equal": out["robot_and_clock_all_equal"],
                      "r8_s3_results": out["r8_s3_results"]["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
