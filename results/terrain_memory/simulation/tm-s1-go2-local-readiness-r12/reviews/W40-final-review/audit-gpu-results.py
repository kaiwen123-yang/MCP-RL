#!/usr/bin/env python3
"""Audit the ten published W40 GPU arrays; no simulation or GPU execution.

Usage: python audit-gpu-results.py --raw-dir /path/to/ten/npz/files
The saved per-substep A/B matrices were NOT published: local_AB_max is
reported metadata, not independently reconstructed from possibly singular P.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def scalar(value):
    return float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    report = json.loads((root / "gpu/w40-gpu/result.json").read_text())
    delivery = json.loads((root / "gpu/w40-gpu/delivery.json").read_text())
    checks = []
    for asset in delivery["arrays"]:
        path = args.raw_dir / asset["name"]
        data = path.read_bytes()
        if len(data) != asset["size"]:
            raise ValueError(f"Size mismatch: {asset['name']}")
        with np.load(path, allow_pickle=False) as arrays:
            inventory = {
                key: {"shape": list(arrays[key].shape),
                      "dtype": str(arrays[key].dtype),
                      "finite": bool(np.isfinite(arrays[key]).all())}
                for key in arrays.files
            }
        checks.append({"name": asset["name"], "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest(),
                       "source_url": asset["url"], "arrays": inventory})
    if not all(v["finite"] for x in checks for v in x["arrays"].values()):
        raise ValueError("Nonfinite array")

    rows = []
    with np.load(args.raw_dir / "one-ten-substeps.npz", allow_pickle=False) as arrays:
        err = arrays["state_error"]
        risk_diff = arrays["gpu_risk"] - arrays["cpu_risk"]
        cpu = arrays["cpu"]
        gpu = arrays["gpu"]
        # Independently verify Euclidean state entries against the raw endpoints.
        # Quaternion tangent errors remain as recorded, without another simulator.
        euclidean_diffs = [
            np.max(np.abs(err[:, :, :3] - (gpu[:, :, :3] - cpu[:, :, :3]))),
            np.max(np.abs(err[:, :, 6:18] - (gpu[:, :, 7:19] - cpu[:, :, 7:19]))),
            np.max(np.abs(err[:, :, 18:] - (gpu[:, :, 19:] - cpu[:, :, 19:]))),
        ]
        for i, fixture in enumerate(report["fixtures"]):
            name = fixture["name"]
            original = next(x for x in report["results"] if x["fixture"] == name)
            original_fd = next(x for x in report["FD_results"] if x["fixture"] == name)
            pos = np.abs(err[i, :, :18]).max(axis=1)
            vel = np.abs(err[i, :, 18:]).max(axis=1)
            risk = np.abs(risk_diff[i]).max(axis=1)
            failures = np.flatnonzero((pos > 5e-7) | (vel > 1e-5) | (risk > 1e-5))
            row = {
                "fixture": name,
                "position_max": scalar(pos.max()),
                "velocity_max": scalar(vel.max()),
                "risk_max": scalar(risk.max()),
                "substep_position_max": pos.tolist(),
                "substep_velocity_max": vel.tolist(),
                "substep_risk_max": risk.tolist(),
                "first_state_or_risk_tolerance_failure_substep_1_based":
                    int(failures[0] + 1) if len(failures) else None,
                "state_and_risk_pass": bool(len(failures) == 0),
                "native_substep_local_AB_max_reported_only": original_fd["local_AB_max"],
            }
            for key in ("position_max", "velocity_max", "risk_max"):
                if row[key] != original[key]:
                    raise ValueError(f"Mismatch with published result: {name}/{key}")
            with np.load(args.raw_dir / (name + "-derivatives.npz"), allow_pickle=False) as deriv:
                for matrix, key in (("A", "composed_A_max"), ("B", "composed_B_max"),
                                    ("C", "active_branch_C_max"), ("D", "active_branch_D_max")):
                    diff = deriv[matrix + "_gpu"] - deriv[matrix]
                    row[key] = scalar(np.abs(diff).max())
                    if row[key] != original_fd[key]:
                        raise ValueError(f"Mismatch with published result: {name}/{key}")
                    index = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
                    row[matrix + "_error_max_index_zero_based"] = list(map(int, index))
                    row[matrix + "_relative_frobenius_error"] = scalar(
                        np.linalg.norm(diff) / max(np.linalg.norm(deriv[matrix]), 1e-30))
                row["prefix_P_max_error_before_each_substep"] = np.abs(
                    deriv["prefix_P_gpu"] - deriv["prefix_P"]).max(axis=(1, 2)).tolist()
                row["prefix_Q_max_error_before_each_substep"] = np.abs(
                    deriv["prefix_Q_gpu"] - deriv["prefix_Q"]).max(axis=(1, 2)).tolist()
            row["published_state_and_local_AB_criteria_pass"] = bool(
                row["state_and_risk_pass"] and original_fd["local_AB_max"] <= 5e-3)
            rows.append(row)
    result = {
        "review_source_commit": "c438f5463659ab7c8d3a7f5eced251639e14b2ef",
        "gpu_computation_source_commit": report["source_commit"],
        "scope": "All ten delivered NPZ read; finite/shape/size checks; recompute saved state and composed derivative maxima; no simulation rerun.",
        "delivery_status": delivery["status"],
        "files": checks,
        "total_bytes": sum(x["bytes"] for x in checks),
        "all_finite": True,
        "euclidean_state_error_vs_raw_endpoints_max_difference": max(map(float, euclidean_diffs)),
        "independent_scope_limits": [
            "Quaternion tangent error array was read, not recalculated with MuJoCo.",
            "Per-substep A/B matrices, perturbed next states and native contact-solver traces are not in these ten assets; local_AB_max cannot be independently recomputed.",
            "C/D compare the CPU-selected earliest risk branch on both prefix chains, not two independently selected nonsmooth peak derivatives.",
            "The original 8-fixture GPU one-substep timer and CPU 8-fixture ten-substep timer cover different workloads and cannot support an end-to-end MPC speedup ratio.",
        ],
        "fixtures": rows,
        "state_and_risk_passing_fixtures": sum(x["state_and_risk_pass"] for x in rows),
        "published_full_criteria_passing_fixtures": sum(x["published_state_and_local_AB_criteria_pass"] for x in rows),
        "failed_fixture_A_max_error_column_all_base_z": all(
            x["A_error_max_index_zero_based"][1] == 2 for x in rows
            if not x["published_state_and_local_AB_criteria_pass"]),
        "decision": "W40 remains unaligned and cannot replace the main controller. User authorizes expanded isolated G1-G5 work in docs/S1_W41_GPU_PLAN.md, including full planner and independent closed-loop experiments after identifying numerical differences; no fixed wall deadline or main hot swap.",
    }
    output = args.output or here / "gpu-array-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "files": len(checks), "bytes": result["total_bytes"],
                      "passing_fixtures": result["published_full_criteria_passing_fixtures"],
                      "all_finite": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
