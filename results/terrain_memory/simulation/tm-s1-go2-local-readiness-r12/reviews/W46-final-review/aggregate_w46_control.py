#!/usr/bin/env python3
"""Aggregate tracked W46 compact evidence, without downloads or simulation.

This recomputes selection/count/gap statistics from recorded JSON summaries. It
does NOT independently recompute MuJoCo rollout costs or verify original NPZs.
Usage: python aggregate_w46_control.py --repo /path/to/MCP-RL --output report.json
"""
import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path


BASE = Path("results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46")
LABELS = ("A148", "A153", "B155", "B158")


def aggregate(repo):
    inputs = []

    def read(name):
        path = repo / BASE / name
        raw = path.read_bytes()
        inputs.append({"path": str(BASE / name), "bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)

    rows, contexts = [], []
    for label in LABELS:
        case = read(label + ".json")
        policies = {x["branch"]: x["installed_policy_identity"] for x in case["results"]}
        contexts.append({"label": label, "complete_policy_hashes_reported": policies,
                         "constant_load_same_policy_hash": policies["constant"] == policies["load"],
                         "all_three_same_policy_hash": len(set(policies.values())) == 1,
                         "shared_reference_error_reported": case["shared_ubar_force_max_error"],
                         "trace_match_error_reported": case["trace_match_max_error"],
                         "disabled_equivalence_reported": case["disabled_equivalence"]})
        for result in case["results"]:
            call = result["call"]
            selected = result["selected"]
            attempts = call["search_attempts"]
            decisions = call["r34_decisions"]
            directions = [d for d in decisions if d["kind"] == 1]
            pairs = [(a, c) for a in attempts for c in a["native_candidates"]]
            positive = [(a, c) for a, c in pairs if c["alpha"] > 0]
            no_step = [(a, c) for a, c in pairs if c["alpha"] == 0]
            best_attempt, best_candidate = min(
                positive, key=lambda pair: pair[1]["cost"] - pair[0]["nominal_current_cost"])
            gap = best_candidate["cost"] - best_attempt["nominal_current_cost"]
            tolerances = [d["comparison_tolerance"] for d in decisions
                          if d["comparison_tolerance"] is not None]
            metrics = ("J0", "Jref", "dx_m", "dy_m", "z_min_m", "terminal_z_vz",
                       "max_abs_roll_pitch_rad", "target_jump_max_rad",
                       "torque_adjacent_substep_max_Nm", "torque_abs_max_Nm",
                       "actuator_saturation_fraction", "native_substeps", "nofoot_nodes")
            row = {
                "label": label, "branch": result["branch"],
                "origin_s": call["origin"], "input_policy_id": selected["input_policy_id"],
                "candidate_source": call["candidate_source"],
                "selected_feedback_scale": selected["feedback_scale"],
                "selected_policy_hash_reported": result["installed_policy_identity"],
                "stop_reason": call["stop_reason"],
                "input_incumbent_J": call["incumbent_current_cost"],
                "selected_J": selected["native_J"], "selected_J0": selected["J0"],
                "selected_Jref": selected["Jref"],
                "selected_prefix_substeps": selected["native_substeps"],
                "selected_prefix_torque_jump_Nm": selected["torque_adjacent_substep_max_Nm"],
                "rollouts": call["native_new_rollouts"],
                "candidate_records": call["candidate_records"],
                "transition_fd_calls": call["transition_fd_calls"],
                "risk_geometry_fd_calls": call["risk_geometry_fd_calls"],
                "derivative_passes": call["derivative_passes"],
                "backward_attempts": len(attempts),
                "backward_completed": sum(a["backward_complete"] for a in attempts),
                "lambda_used": [a["lambda_used"] for a in attempts],
                "line_search_records": len(directions),
                "positive_alpha_records": len(positive),
                "positive_alpha_eligible": sum(d["eligible"] for d in directions if d["alpha"] > 0),
                "positive_alpha_replaced_archive": sum(d["replaced"] for d in directions if d["alpha"] > 0),
                "decision_reasons": dict(collections.Counter(d["reason"] for d in directions)),
                "positive_rollout_failures": sum(c["failed"] for _, c in positive),
                "positive_trunk_violations": sum(c["trunk_violation"] > 0 for _, c in positive),
                "positive_projection_failures": sum(c["projection_failed_count"] for _, c in positive),
                "positive_restorations": sum(c["actuator_restoration_count"] for _, c in positive),
                "positive_projection_count_range": [min(c["projection_count"] for _, c in positive),
                                                     max(c["projection_count"] for _, c in positive)],
                "positive_all_cost_worse_than_nominal": all(c["cost"] > a["nominal_current_cost"]
                                                            for a, c in positive),
                "best_positive_cost_minus_nominal": gap,
                "best_positive_attempt": best_attempt["attempt"],
                "best_positive_alpha": best_candidate["alpha"],
                "max_comparison_tolerance": max(tolerances),
                "best_positive_gap_divided_by_max_tolerance": gap / max(tolerances),
                "zero_step_cost_minus_nominal": [c["cost"] - a["nominal_current_cost"]
                                                  for a, c in no_step],
                "nominal_bound_violation_max": max(a["nominal_bound_violation"] for a in attempts),
                "projected_gradient_inf_range": [min(a["projected_gradient_inf"] for a in attempts),
                                                  max(a["projected_gradient_inf"] for a in attempts)],
                "diagnostics_reported": [dict(sign=d["sign"],
                                                **{k: d["summary"][k] for k in metrics})
                                          for d in result["diagnostics"]],
            }
            rows.append(row)

    mechanism = read("mechanism-evidence.json")
    numerical = read("numerical-consistency.json")
    phase = read("phase.json")
    jobs = {label: read(label + ".json") for label in ("w46-a", "w46-b")}
    sumkeys = ("rollouts", "candidate_records", "transition_fd_calls", "risk_geometry_fd_calls",
               "derivative_passes", "backward_attempts", "backward_completed", "line_search_records",
               "positive_alpha_records", "positive_alpha_eligible", "positive_alpha_replaced_archive",
               "positive_rollout_failures", "positive_trunk_violations", "positive_projection_failures",
               "positive_restorations")
    totals = {k: sum(x[k] for x in rows) for k in sumkeys}
    totals.update(finite_solves=len(rows),
                  diagnostics=sum(len(x["diagnostics_reported"]) for x in rows),
                  stop_reasons=dict(collections.Counter(x["stop_reason"] for x in rows)),
                  all_positive_costs_worse=all(x["positive_all_cost_worse_than_nominal"] for x in rows))
    assert totals["finite_solves"] == 12 and totals["diagnostics"] == 36
    assert totals["positive_alpha_records"] == 540
    assert totals["rollouts"] == mechanism["native_solver_rollouts"]
    assert totals["transition_fd_calls"] == mechanism["transition_fd_calls"]
    assert totals["risk_geometry_fd_calls"] == mechanism["risk_geometry_fd_calls"]
    actual = {}
    for label, job in jobs.items():
        stage = job["stages"][0]
        actual[label] = {k: stage[k] for k in ("attempt", "status", "success", "reason",
                                              "simulation_s", "wall_s", "exit_code", "legal_route",
                                              "material_progress")}
        actual[label].update(new_PPO_samples=job["new_PPO_samples"],
                             new_PPO_updates=job["new_PPO_updates"],
                             deadline_unix_s=job["deadline_unix_s"])
    return {
        "schema": "w47_control_compact_aggregation_v1",
        "reviewed_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "method": "Pure local arithmetic over tracked compact JSON. No network, simulator, training or raw archive reads.",
        "raw_array_recomputation_performed": False,
        "limits": ["Costs, policies and diagnostic motions were recorded by W46; this script independently aggregates their published compact entries only.",
                   "The 12 finite solves start from saved W45 output policies and reconstructed context, not exact original pre-call optimizer state.",
                   "Finite stagnation is not a count of physical episode failures, and cannot be extrapolated to all 1944 actual planner calls.",
                   "Selected solver torque prefixes contain only 100 substeps; diagnostic torque metrics contain 1750 substeps."],
        "inputs": inputs, "totals": totals, "contexts": contexts, "finite": rows,
        "numerical_consistency_reported": numerical,
        "actual_qualification_reported": actual,
        "phase_status": phase["status"], "actual_build_sha256": phase["binary_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = aggregate(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(data["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
