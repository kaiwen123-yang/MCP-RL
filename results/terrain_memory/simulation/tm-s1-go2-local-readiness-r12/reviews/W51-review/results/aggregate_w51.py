#!/usr/bin/env python3
"""Read-only independent W51 audit of retained JSON/JSONL/NPZ and C FP64 output.

No imports of project execution code, MuJoCo, CUDA, torch, native agents, FD,
solvers, PPO or networking. Writes only the selected review JSON. It never
changes source evidence, hashes the repository, or replays physics. Run from
any directory with NumPy available; paths are explicit arguments.
"""
import argparse
import collections
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

BASE = "354b14bf5313332ae167f02c84ce6b68ef6bae90"
REVIEWED = "a621d7d35e6016634b388a3aef0d39d8d008aab0"
REL = Path("results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
ATTEMPTS = {
    "D0": "job-20261115T112019Z-w51-d0-eval-PANN_PPO_S5-s1-D8-attempt01",
    "D1": "job-20261115T113036Z-w51-d1-command-repair-eval-PANN_PPO_S5-s1-D8-attempt01",
    "D1_technical": "job-20261115T112019Z-w51-d1-eval-PANN_PPO_S5-s1-D8-attempt01",
}
WORK = ("native_new_rollouts", "candidate_records", "derivative_passes",
        "transition_fd_calls", "risk_geometry_fd_calls", "backward_attempts")
WINDOWS = [(0.0, 2.0), (2.002, 4.3), (4.3, 8.0)]
LEGS = ("FL", "FR", "RL", "RR")


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def reviewed_json(repo, relative):
    """Pin Git evidence even after the next plan changes the working tree."""
    raw = subprocess.check_output(["git", "show", REVIEWED + ":" + str(relative)], cwd=repo)
    return json.loads(raw)


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def counts(records, key):
    return dict(sorted(collections.Counter(str(row.get(key)) for row in records).items()))


def quantiles(values):
    return np.quantile(values, [0, .1, .5, .9, 1], axis=0).tolist()


def first_time(times, condition):
    ids = np.flatnonzero(condition)
    return float(times[ids[0]]) if len(ids) else None


def last_call_record(row):
    fields = ("call_id", "origin", "candidate_source", "installed_plan_id",
              "chosen_alpha", "policy_generation_time", "policy_generation_call",
              "policy_age_s", "reference_start", "reference_end",
              "reference_expired_by_s", "feedback_provenance")
    return {key: row.get(key) for key in fields}


def inspect_attempt(repo, originals, arm, name):
    path = originals / name
    result = reviewed_json(repo, REL / (name + ".json"))
    calls = rows(path / "planner-calls.jsonl")
    boundaries = rows(path / "w50-planner-boundaries.jsonl")
    actions = rows(path / "action-execution.jsonl")
    feedback = {row["query_id"]: row for row in rows(path / "feedback-actions.jsonl")}
    held_clock = rows(path / "w50-held-clock.jsonl")
    held = [row for row in actions if row["query_kind"] == "applied_hold"]
    pairs = [(row, feedback[row["query_id"]]) for row in held]
    events = read_json(path / "w51-event-manifest.json")
    with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
        tr = {key: archive[key] for key in archive.files}
    with np.load(path / "physical-000.npz", allow_pickle=False) as archive:
        physical = archive["physical"]
    model = ET.parse(path / "model/G1-project-go2.xml")
    radii = np.array([float(model.find(".//geom[@name='%s']" % leg).get("size").split()[0])
                      for leg in LEGS])
    t, q, velocity = tr["time"], tr["qpos"], tr["qvel"]
    fz, nf = tr["actual_grf"][:, :, 2], tr["nonfoot_ground_force"][:, 2]
    loads = fz > 1
    bottom = tr["feet"][:, :, 2] - radii
    contact = tr["contact"].astype(bool)
    updated = [row for row in calls if row["policy_updated"]]
    positive = [row for row in updated if row["candidate_source"].startswith(
        ("alpha_", "k_guided_", "h_guided_", "homotopy_", "w50_")) and row["chosen_alpha"] > 0]
    matched = [row for row in boundaries if row["previous_prediction"]["status"] == "available"]
    # Use the recorded exact plant clock to avoid calling a float32 RPC
    # timestamp a meaningful underflow/expiry at a reference boundary.
    tail = [(a, f) for a, f in pairs if a["time_s"] > f["reference_end"] + 1e-9]
    integrated_pairs = [(a, f) for a, f in pairs if a["time_s"] < physical[-1, 1] - 1e-9]
    integrated_tail = [(a, f) for a, f in integrated_pairs
                       if a["time_s"] > f["reference_end"] + 1e-9]
    cols = {name: i for i, name in enumerate(result["physical_columns"])}
    pd = physical[:, [cols["unclipped_pd_request_%d" % i] for i in range(12)]]
    limits = np.array([23.7, 23.7, 45.43] * 4)
    physical_nf = physical[:, cols["nonfoot_grf_z"]]
    actual_sat = np.abs(pd) > limits + 1e-10
    pd_max_excess = np.max(np.abs(pd) - limits, axis=1)
    all_native_flags = sum(not f["reference_covers_query"] for _, f in pairs)
    held_t = np.array([a["time_s"] for a, _ in pairs])
    raw_feedback = np.array([max(abs(v) for v in f["raw_K_delta_x"]) for _, f in pairs])
    raw_clip = np.array([any(x != y for x, y in zip(f["action_before_clip"], f["center_before_projection"])) for _, f in pairs])
    total_rewrite = np.array([any(x != y for x, y in zip(f["action_before_clip"], f["action_after_clip"])) for _, f in pairs])
    projection = np.array([bool(f["projection_changed"]) for _, f in pairs])
    out = {
        "attempt": name, "arm": arm,
        "control_evidence_usable": arm != "D1_technical",
        "record_root": str(path),
        "stage": {k: result.get(k) for k in (
            "stage_status", "reason", "error", "simulation_s", "wall_s", "physical_samples",
            "success", "qualification", "legal_route", "stage_progress", "max_x_m",
            "min_body_height_m", "completed_round_trips", "max_joint_limit_excess_rad")},
        "wall_seconds_per_simulated_second": result["wall_s"] / result["simulation_s"],
        "physical_substeps_from_npz": int(len(physical)),
        "physical_start_end_clock_s": physical[[0, -1], :2].tolist(),
        "trajectory_samples": len(t), "applied_hold_queries": len(held),
        "held_clock_rows": len(held_clock),
        "held_queries_followed_by_observed_physics": len(integrated_pairs),
        "planner_calls": len(calls), "boundary_rows": len(boundaries),
        "positive_installs": len(positive),
        "positive_same_problem_J_descent": sum(row["chosen_actual"] > 0 for row in positive),
        "positive_risk_priority_J_increase": sum(row["chosen_actual"] < 0 for row in positive),
        "all_policy_updated_count": len(updated),
        "installed_sources": counts(updated, "candidate_source"),
        "all_candidate_sources": counts(calls, "candidate_source"),
        "return_reasons": counts(calls, "return_reason"),
        "stop_reasons": counts(calls, "stop_reason"),
        "seed_statuses": counts(calls, "w51_seed_status"),
        "seed_rollouts": sum(row.get("w51_seed_rollouts", 0) for row in calls),
        "Q_is_S_calls": sum(bool(row.get("w51_Q_is_S")) for row in calls),
        "direct_S_installs": [last_call_record(row) for row in updated
                              if row["candidate_source"] == "motion_seed_S"],
        "work_completed_calls": {key: sum(row[key] for row in calls) for key in WORK},
        "work_scope": "Completed native call records only; interrupted unrecorded work remains unknown.",
        "last_policy_update": last_call_record(updated[-1]),
        "last_call": last_call_record(calls[-1]),
        "planner_wall_sum_s": sum(row["wall_s"] for row in calls),
        "planner_wall_quantiles_s": quantiles([row["wall_s"] for row in calls]),
        "prediction_available": len(matched),
        "prediction_unavailable": len(boundaries) - len(matched),
        "prediction_errors_max_component": {
            key: float(max(np.max(np.abs(row["previous_prediction"][key])) for row in matched))
            for key in ("base_position_m", "base_orientation_rad", "joint_position_rad",
                        "base_linear_velocity_m_s", "base_angular_velocity_rad_s", "joint_velocity_rad_s")},
        "feedback_lifecycle": {
            "canonical_clock_tail_query_count": len(tail),
            "canonical_clock_tail_query_fraction": len(tail) / len(held),
            "first_canonical_tail_time_s": tail[0][0]["time_s"] if tail else None,
            "max_canonical_tail_age_s": max(0, max(a["time_s"] - f["reference_end"] for a, f in pairs)),
            "integrated_hold_tail_count": len(integrated_tail),
            "integrated_hold_total": len(integrated_pairs),
            "native_covers_query_false_count": all_native_flags,
            "native_expired_gt_1e_minus9_s_count": sum(f["reference_expired_by_s"] > 1e-9 for _, f in pairs),
            "rounding_note": "Native RPC time is float32; covers=false also includes tiny before-start roundoff. Canonical plant time and 1e-9 s tolerance define meaningful tail counts here, not a changed control rule.",
            "last_provenance": pairs[-1][1]["feedback_provenance"],
            "max_abs_raw_K_delta_x_rad": float(raw_feedback.max()),
            "max_abs_raw_K_delta_x_time_s": float(held_t[np.argmax(raw_feedback)]),
            "raw_actuator_target_clip_queries": int(raw_clip.sum()),
            "first_raw_actuator_target_clip_s": first_time(held_t, raw_clip),
            "action_total_rewrite_queries_including_QP": int(total_rewrite.sum()),
            "projection_changed_queries": int(projection.sum()),
            "first_projection_changed_s": first_time(held_t, projection),
            "clip_semantics": "Original actuator target clipping compares action_before_clip to center_before_projection; action_after_clip includes subsequent metric QP projection and is not a pure clip counter.",
        },
        "trajectory": {
            "physical_duration_s": float(t[-1]), "xyz_initial_m": q[0, :3].tolist(),
            "xyz_final_m": q[-1, :3].tolist(), "max_x_m": float(q[:, 0].max()),
            "min_body_z_m": float(q[:, 2].min()), "vertical_velocity_range_m_s": quantiles(velocity[:, 2])[::4],
            "first_height_crossings_s": {str(height): first_time(t, q[:, 2] < height)
                                         for height in (.28, .27, .25, .20, .12)},
            "height_crossing_scope": "Descriptive fixed-height timestamps only, not newly introduced pass/fail or stop criteria.",
            "roll_pitch_yaw_max_abs_rad": np.max(np.abs(tr["rpy"]), axis=0).tolist(),
            "foot_order": list(LEGS), "foot_radii_from_original_XML_m": radii.tolist(),
            "contact_transitions_per_foot": np.count_nonzero(np.diff(contact.astype(int), axis=0), axis=0).tolist(),
            "contact_fraction_per_foot": contact.mean(axis=0).tolist(),
            "foot_load_over_1N_fraction": loads.mean(axis=0).tolist(),
            "per_foot_Fz_quantiles_N": quantiles(fz), "total_foot_Fz_quantiles_N": quantiles(fz.sum(axis=1)),
            "nonfoot_Fz_quantiles_N": quantiles(nf), "nonfoot_over_1N_fraction": float((nf > 1).mean()),
            "first_trajectory_nonfoot_over_1N_s": first_time(t, nf > 1),
            "first_physical_force_evaluation_nonfoot_over_1N_s": first_time(physical[:, 0], physical_nf > 1),
            "physical_force_clock_note": "physical first column is force-evaluation clock; second is post-integration clock; trajectory is 500 Hz same-time force.",
            "physical_PD_saturated_substeps": int(actual_sat.any(axis=1).sum()),
            "physical_PD_saturated_joint_substeps": actual_sat.sum(axis=0).tolist(),
            "first_physical_PD_saturation_s": first_time(physical[:, 0], actual_sat.any(axis=1)),
            "PD_excess_magnitude_diagnostics": [
                {"above_original_force_limit_by_Nm": float(excess),
                 "substeps": int((pd_max_excess > excess).sum()),
                 "first_force_evaluation_s": first_time(physical[:, 0], pd_max_excess > excess)}
                for excess in (1e-10, 1e-6, 1e-4, .01, .1, 1.0)],
            "PD_excess_scope": "Original force limits unchanged. Magnitude bins distinguish float32 boundary roundoff from substantial held-action saturation; these are descriptive and do not introduce pass/fail gates.",
            "max_abs_physical_PD_request_Nm": np.max(np.abs(pd), axis=0).tolist(),
            "recorded_command_x_integral_m": float(np.sum(tr["command_velocity"][:-1, 0] * np.diff(t))),
            "command_integral_note": "Descriptive left-rectangle integral of recorded command, not an original qualification target.",
            "moving_command_userdata_max_abs_difference": float(np.max(np.abs(
                tr["support_reference_userdata"][t >= 2.002 - 1e-9, 78:81] - tr["command_velocity"][t >= 2.002 - 1e-9]))),
            "moving_command_userdata_max_abs": float(np.max(np.abs(tr["support_reference_userdata"][t >= 2.002 - 1e-9, 78:81]))),
        },
        "events": events["events"],
        "retained_event_object_bytes_from_manifest": events["actual_bytes"],
        "windows": [],
        "feedback_diagnostic_slices": [],
        "new_PPO_samples": 0, "new_PPO_updates": 0,
    }
    # Reconstruct intended phase from every native call's recorded private gait
    # clock, not from contacts or a fabricated reference. Trot's FR/RL shifts
    # coincide, so native and plant foot order both map to [0,.5,.5,0].
    clock_origins = np.array([row["origin"] for row in calls])
    clock_id = np.searchsorted(clock_origins, t + 1e-9, side="right") - 1
    clock = np.array([calls[i]["gait_context"] for i in clock_id])
    phase = clock[:, 8] + (t - clock[:, 9]) * clock[:, 10]
    wrapped = (phase[:, None] / (2 * np.pi) - np.array([0, .5, .5, 0]) + .5) % 1 - .5
    swing = np.abs(wrapped) < .275
    for lo, hi in WINDOWS:
        mask = (t >= lo - 1e-9) & (t <= hi + 1e-9)
        ids = np.flatnonzero(mask)
        if not len(ids):
            out["windows"].append({"requested_s": [lo, hi], "status": "not_reached"})
            continue
        lc = loads[mask]
        pattern = lc.astype(np.int64) @ np.array([1, 2, 4, 8])
        wc = [row for row in calls if lo - 1e-9 <= row["origin"] <= hi + 1e-9]
        wb = [row for row in boundaries if lo - 1e-9 <= row["origin_s"] <= hi + 1e-9]
        wp = [(a, f) for a, f in pairs if lo - 1e-9 <= a["time_s"] <= hi + 1e-9]
        term_mean = {key: float(np.mean([row["current_solve"]["cost_components"][key] for row in wb]))
                     for key in wb[0]["current_solve"]["cost_components"]} if wb else {}
        speed = np.linalg.norm(tr["feet_velocity"][mask, :, :2], axis=2)
        w = {
            "requested_s": [lo, hi], "observed_s": [float(t[ids[0]]), float(t[ids[-1]])],
            "status": "complete_window" if t[ids[-1]] >= hi - 1e-9 else "partial_physical_window",
            "samples": len(ids), "body_dx_m": float(q[ids[-1], 0] - q[ids[0], 0]),
            "body_x_last_m": float(q[ids[-1], 0]),
            "z_min_m": float(q[mask, 2].min()), "z_median_m": float(np.median(q[mask, 2])),
            "z_last_m": float(q[ids[-1], 2]),
            "vz_range_m_s": [float(velocity[mask, 2].min()), float(velocity[mask, 2].max())],
            "roll_pitch_max_abs_rad": np.max(np.abs(tr["rpy"][mask, :2]), axis=0).tolist(),
            "nonfoot_mean_Fz_N": float(nf[mask].mean()),
            "nonfoot_over_1N_fraction": float((nf[mask] > 1).mean()),
            "four_feet_over_1N_fraction": float(lc.all(axis=1).mean()),
            "support_pattern_fractions": {str(v): float((pattern == v).mean()) for v in np.unique(pattern)},
            "support_pattern_bits": "FL=1, FR=2, RL=4, RR=8; diagonal masks=9,6",
            "diagonal_support_fraction": float(np.isin(pattern, [6, 9]).mean()),
            "load_transitions_per_foot": np.count_nonzero(np.diff(lc.astype(int), axis=0), axis=0).tolist(),
            "feet_Fz_mean_N": fz[mask].mean(axis=0).tolist(),
            "feet_world_tangent_speed_mean_m_s": speed.mean(axis=0).tolist(),
            "feet_bottom_z_range_m": [bottom[mask].min(axis=0).tolist(), bottom[mask].max(axis=0).tolist()],
            "feet_bottom_scope": "Original foot geom centers minus original XML radius, relative to z=0 plane; no assumption about unobserved deformed terrain.",
            "held_query_count": len(wp),
            "held_query_reference_tail_count": sum(a["time_s"] > f["reference_end"] + 1e-9 for a, f in wp),
            "planner_calls": len(wc), "policy_updates": sum(row["policy_updated"] for row in wc),
            "installed_sources": counts([row for row in wc if row["policy_updated"]], "candidate_source"),
            "native_cost_mean_terms": term_mean,
            "native_cost_mean_sum": sum(term_mean.values()),
            "native_cost_mean_fraction": {key: value / sum(term_mean.values()) for key, value in term_mean.items()},
            "cost_scope": "Arithmetic mean of installed same-call cost components, not a trajectory integral.",
            "diagnostic_load_threshold_N": 1, "qualification_threshold_changed": False,
        }
        if lo > 2:
            sw = swing[mask]
            w["target_swing_loaded_fraction"] = [float(lc[:, i][sw[:, i]].mean()) if sw[:, i].any() else None for i in range(4)]
            w["loaded_stance_world_tangent_speed_mean_m_s"] = [float(speed[:, i][(~sw[:, i]) & lc[:, i]].mean())
                if ((~sw[:, i]) & lc[:, i]).any() else None for i in range(4)]
            w["target_swing_bottom_z_range_m"] = [[float(bottom[mask, i][sw[:, i]].min()), float(bottom[mask, i][sw[:, i]].max())]
                if sw[:, i].any() else None for i in range(4)]
        out["windows"].append(w)
    for lo, hi in ((0, 2), (2.002, 2.976), (2.976, 3.016), (3.016, 3.6),
                   (3.6, 4.3), (4.3, 5.0), (5.0, 5.2304), (5.2304, 5.814), (5.814, 8.0)):
        mask = (held_t >= lo - 1e-9) & (held_t < hi - 1e-9)
        ids = np.flatnonzero(mask)
        if not len(ids):
            continue
        largest = ids[np.argmax(raw_feedback[ids])]
        physical_mask = (physical[:, 0] >= lo - 1e-9) & (physical[:, 0] < hi - 1e-9)
        state_ids = np.flatnonzero((t >= lo - 1e-9) & (t <= hi + 1e-9))
        out["feedback_diagnostic_slices"].append({
            "requested_start_end_exclusive_s": [lo, hi], "held_queries": len(ids),
            "max_abs_K_delta_x_rad": float(raw_feedback[largest]),
            "max_at_s": float(held_t[largest]), "median_abs_K_delta_x_max_component_rad": float(np.median(raw_feedback[ids])),
            "raw_actuator_target_clip_queries": int(raw_clip[ids].sum()),
            "projection_changed_queries": int(projection[ids].sum()),
            "total_action_rewrite_queries_including_QP": int(total_rewrite[ids].sum()),
            "physical_PD_saturated_substeps": int(actual_sat[physical_mask].any(axis=1).sum()),
            "body_z_start_end_min_m": [float(q[state_ids[0], 2]), float(q[state_ids[-1], 2]), float(q[state_ids, 2].min())],
            "interpretation": "Temporal association only; neither late maximum feedback nor height crossings by themselves establish the initial causal defect.",
        })
    return out, tr


def read_candidate_residuals(path):
    """Parse original schema-33 binary without importing runtime/solver modules."""
    raw = np.fromfile(path, dtype="<f8")
    cursor, values = 0, []
    while cursor < len(raw):
        h = raw[cursor:cursor + 19]
        cursor += 19
        if len(h) != 19 or h[0] != 33:
            raise ValueError("unexpected schema-33 record")
        T, ns, nu, nx, nr = map(int, h[5:10])
        residual = None
        for prefix in ("", "reference_"):
            for field, shape in (("states", (T, ns)), ("actions", (T - 1, nu)),
                                 ("times", (T,)), ("residual", (T, nr))):
                count = int(np.prod(shape))
                if prefix == "" and field == "residual":
                    residual = raw[cursor:cursor + count].reshape(shape).copy()
                cursor += count
        cursor += T * nu * nx + T * nu * nu + T * nu + T * nx * nu + T * nu
        for _ in ("prefix", "feet"):
            count = int(raw[cursor])
            cursor += 1 + count
        values.append((residual, float(h[11]), float(h[12]), bool(h[13])))
    if cursor != len(raw):
        raise ValueError("candidate file cursor mismatch")
    return values


def verify_c(repo, originals, w50):
    report = reviewed_json(repo, REL / "w51/task-c.json")
    with np.load(originals / "w51-C/fp64-costs.npz", allow_pickle=False) as z:
        gpu = {key: z[key] for key in z.files}
    record_keys = ("status", "started_unix_s", "ended_unix_s", "deadline", "wall_deadline_enabled",
                   "candidates", "saved_problem_candidates", "wrong_bridge_algebra_only_candidates",
                   "residual_shape", "cost_terms", "dtype", "device", "physics_steps", "native_rollouts",
                   "PPO_samples", "PPO_updates", "batches", "peak_allocated_bytes", "reduction_wall_s",
                   "timings_scope", "complete_native_cost_port", "formal_installation_allowed",
                   "coverage_annotation", "residual_reconstruction_max_errors")
    out = {key: report.get(key) for key in record_keys}
    out["wall_s"] = report["ended_unix_s"] - report["started_unix_s"]
    out["peak_allocated_MiB"] = report["peak_allocated_bytes"] / 2 ** 20
    out["peak_allocated_fraction_of_96_GiB"] = report["peak_allocated_bytes"] / (96 * 2 ** 30)
    out["independent_native_J_error"] = float(np.max(np.abs(gpu["total_J"] - gpu["native_J"])))
    out["independent_sum_term_J_error"] = float(np.max(np.abs(gpu["total_J"] - gpu["term_mean"].sum(axis=-1))))
    out["independent_mean_node_term_error"] = float(np.max(np.abs(gpu["term_mean"] - gpu["node_terms"].mean(axis=1))))
    out["cost_terms_coverage"] = report["coverage"]
    if w50 is None:
        out["CPU_independent_residual_verification"] = "not_requested"
        return out
    residuals, native, risks, failures = [], [], [], []
    definitions = []
    group_ranges = []
    for family, labels in (("task-b", ("A237", "B236", "A328", "B625")),
                          ("deployment", ("A237", "B236"))):
        for label in labels:
            fixture = read_json(w50 / "inputs" / label / "input.json")
            definitions.append(fixture["cost_definitions"])
            start = len(residuals)
            for residual, J, risk, failed in read_candidate_residuals(w50 / family / label / "snapshot.bin.r33-candidates"):
                residuals.append(residual)
                native.append(J)
                risks.append(risk)
                failures.append(failed)
            group_ranges.append((family + "/" + label, start, len(residuals)))
    if not all(d == definitions[0] for d in definitions):
        raise ValueError("C source fixture cost definitions differ")
    R, native, risks, failures = np.stack(residuals), np.array(native), np.array(risks), np.array(failures)
    nodes = []
    for d in definitions[0]:
        r = R[:, :, d["start"]:d["end"]]
        p = d["norm_parameter"]
        if d["norm"] == 0:
            n = .5 * np.square(r).sum(axis=-1)
        elif d["norm"] == 2:
            n = np.sqrt(np.square(r).sum(axis=-1) + p * p) - p
        elif d["norm"] == 6:
            n = (np.sqrt(np.square(r) + p * p) - p).sum(axis=-1)
        else:
            raise ValueError("unsupported original norm")
        nodes.append(d["weight"] * n)
    nodes = np.stack(nodes, axis=-1)
    terms = nodes.mean(axis=1)
    cost = terms.sum(axis=-1)
    n = R.shape[1] * (8 * R.shape[2] + 4 * len(definitions[0]) + 2) + 1
    unit = np.finfo(np.float64).eps / 2
    gamma = n * unit / (1 - n * unit)
    offset = sum(2 * abs(d["weight"] * d["norm_parameter"]) *
                 (d["end"] - d["start"] if d["norm"] == 6 else 1)
                 for d in definitions[0] if d["norm"] in (2, 6))
    budget = gamma * (np.abs(nodes).mean(axis=1).sum(axis=-1) + offset)
    def better(i, j, J):
        if failures[i] or not np.isfinite([J[i], risks[i]]).all():
            return False
        if failures[j] or not np.isfinite([J[j], risks[j]]).all():
            return True
        return bool(risks[i] < risks[j] or (risks[i] == risks[j] and J[j] - J[i] > budget[i] + budget[j]))
    groups = []
    for name, start, end in group_ranges:
        idx = range(start, end)
        mismatches = sum(better(i, j, gpu["total_J"]) != better(i, j, native) for i in idx for j in idx)
        groups.append({"group": name, "candidates": end - start,
                       "pairwise_comparisons_including_self": (end - start) ** 2,
                       "GPU_native_decision_mismatches": mismatches})
    out["CPU_independent_residual_verification"] = {
        "source_root": str(w50), "candidate_count": len(residuals), "residual_shape": list(R.shape),
        "all_six_fixture_cost_definitions_equal": True,
        "native_J_input_arrays_max_abs_difference": float(np.max(np.abs(native - gpu["native_J"]))),
        "CPU_native_max_abs_J_error": float(np.max(np.abs(cost - native))),
        "CPU_GPU_max_abs_J_error": float(np.max(np.abs(cost - gpu["total_J"]))),
        "CPU_GPU_max_abs_node_term_error": float(np.max(np.abs(nodes - gpu["node_terms"]))),
        "CPU_GPU_max_abs_term_mean_error": float(np.max(np.abs(terms - gpu["term_mean"]))),
        "roundoff_budget_max_abs_difference": float(np.max(np.abs(budget - gpu["original_roundoff_budget"]))),
        "groups": groups, "new_GPU_work": 0, "new_physics": 0,
        "scope": "Recomputed scalar norms on already saved native residuals; does not reconstruct contact solver residuals or establish control validity for 38 wrong-bridge candidates.",
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--w50-originals", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    phase = reviewed_json(repo, REL / "w51/phase.json")
    paired = reviewed_json(repo, REL / "w51/paired-results.json")
    changed = subprocess.check_output(["git", "diff", "--name-only", BASE, REVIEWED], cwd=repo, text=True).splitlines()
    inventory = []
    for relative in changed:
        if relative.endswith(".json"):
            raw = subprocess.check_output(["git", "show", REVIEWED + ":" + relative], cwd=repo)
            data = json.loads(raw)
            inventory.append({"path": relative, "bytes": len(raw),
                              "JSON_type": type(data).__name__,
                              "top_level_count": len(data) if isinstance(data, (dict, list)) else None})
    attempts, trajectories = {}, {}
    for arm, name in ATTEMPTS.items():
        attempts[arm], trajectories[arm] = inspect_attempt(repo, args.originals, arm, name)
    initial = {}
    for field in trajectories["D0"]:
        a = trajectories["D0"][field][trajectories["D0"]["time"] <= 2 + 1e-9]
        b = trajectories["D1"][field][trajectories["D1"]["time"] <= 2 + 1e-9]
        if a.dtype.kind in "biuf" and a.shape == b.shape:
            initial[field] = {"array_equal": bool(np.array_equal(a, b)),
                              "max_abs_difference": float(np.max(np.abs(a.astype(float) - b.astype(float))))}
    totals = {}
    for label, arms in (("valid_pair", ("D0", "D1")), ("all_three_including_technical", tuple(ATTEMPTS))):
        totals[label] = {key: sum(attempts[a]["work_completed_calls"][key] for a in arms) for key in WORK}
        totals[label].update({"planner_calls": sum(attempts[a]["planner_calls"] for a in arms),
                              "plant_physical_substeps": sum(attempts[a]["physical_substeps_from_npz"] for a in arms),
                              "simulation_s": sum(attempts[a]["stage"]["simulation_s"] for a in arms),
                              "PPO_samples": 0, "PPO_updates": 0})
    launchers = {a: reviewed_json(repo, REL / "w51" / filename) for a, filename in
                 (("D0", "D0.json"), ("D1", "D1.json"), ("D1_technical", "D1-technical-attempt01.json"))}
    intervals = {a: [d["stages"][0]["started_unix_s"], d["stages"][0]["ended_unix_s"]]
                 for a, d in launchers.items()}
    def overlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    c = verify_c(repo, args.originals, args.w50_originals)
    c_interval = [c["started_unix_s"], c["ended_unix_s"]]
    release = {}
    for family in ("paired-D8", "C-costs"):
        folder = REL / "w51/delivery" / family
        manifest = reviewed_json(repo, folder / "manifest.json")
        delivery = reviewed_json(repo, folder / "delivery.json")
        sources = reviewed_json(repo, folder / "sources.json")
        entries = sources if isinstance(sources, list) else sources.get("sources", [])
        release[family] = {"status": delivery["status"], "asset_count": len(manifest),
                           "asset_size_bytes": sum(row["size"] for row in manifest),
                           "ZIP_count": sum(row["name"].endswith(".zip") for row in manifest),
                           "direct_MP4_count": sum(row["name"].endswith(".mp4") for row in manifest),
                           "source_entry_count": len(entries),
                           "source_bytes": sum(row.get("bytes", row.get("size", 0)) for row in entries),
                           "server_originals_retained": delivery.get("server_originals_retained"),
                           "verified_receipt_count": len(delivery["verified_assets"]),
                           "receipt_manifest_name_sets_equal": {row["name"] for row in manifest} == {row["name"] for row in delivery["verified_assets"]},
                           "scope": "Committed publication records; independent live remote check is in the parent audit."}
    output = {
        "schema": "independent_W51_record_audit_v1", "reviewed_commit": REVIEWED, "plan_base_commit": BASE,
        "read_only": True, "new_physics": 0, "new_native_solves": 0, "new_FD": 0, "new_GPU_work": 0,
        "changed_JSON_full_parse_inventory": inventory,
        "changed_JSON_full_parse_count": len(inventory), "changed_JSON_bytes": sum(row["bytes"] for row in inventory),
        "current_phase_status": phase["status"], "current_phase_selection": phase.get("selection"),
        "original_evaluations_started": phase.get("original_evaluations_started"),
        "original_stages": paired.get("original_stages"),
        "original_stage_row_count": 0,
        "original_blocked_stage_row_count": 0,
        "original_stages_scope": "W51 declares the original ladder unrun; no W51 original E30/E60/E300/E600 execution or blocked row exists. Do not reuse W50's eight blocked rows as W51 failures.",
        "failure_accounting": {"D8_launches": 3, "valid_diagnostic_arms": 2,
                               "physical_failure": 1, "diagnostic_window_completed_without_normal_locomotion": 1,
                               "technical_invalid_command_attempt": 1, "formal_E30_launches": 0,
                               "qualified_original_task_runs": 0,
                               "causal_scientific_claim": "The corrected D1 stayed upright but did not produce support-phase progression; D8 completion is not original task success."},
        "technical_attempt_record": phase.get("technical_attempts"),
        "attempts": attempts, "totals": totals, "initial_0_2_s_pair_array_comparison": initial,
        "resources": {"reserved_affinity_from_phase": phase["resources"],
                      "stage_launcher_intervals_unix_s": intervals,
                      "valid_D0_D1_launcher_overlap_s": overlap(intervals["D0"], intervals["D1"]),
                      "D0_technical_D1_launcher_overlap_s": overlap(intervals["D0"], intervals["D1_technical"]),
                      "C_D0_overlap_s": overlap(c_interval, intervals["D0"]),
                      "C_technical_D1_overlap_s": overlap(c_interval, intervals["D1_technical"]),
                      "C_corrected_D1_overlap_s": overlap(c_interval, intervals["D1"]),
                      "first_D0_start_to_final_D1_end_s": intervals["D1"][1] - intervals["D0"][0],
                      "CPU_process_user_system_core_hours": None,
                      "CPU_utilization_time_series": None,
                      "whole_device_GPU_peak_memory_bytes": None,
                      "whole_device_GPU_utilization_time_series": None,
                      "C_torch_tensor_peak_allocated_bytes": c["peak_allocated_bytes"],
                      "resource_measurement_scope": "W51 has native process affinity snapshots, call wall timings, and C torch.cuda.max_memory_allocated. Historical CPU/RSS observations nested in index.json belong to earlier campaigns and are not W51 measurements.",
                      "interpretation": "Launcher overlap establishes simultaneous jobs, not CPU utilization. No continuous CPU or GPU utilization time series was found in these W51 records. The tiny C run is not sustained GPU parallelism."},
        "GPU_C": c, "publication": release,
        "threshold_scope": "The 1 N support threshold and windows are descriptive, not changes to original qualification or new stop gates.",
        "missing_evidence": [
            "No W51 original E30/E60/E300/E600 run exists; no full task or material-memory gain can be inferred.",
            "No continuous CPU utilization/GPU utilization trace is retained; affinity cannot establish saturation.",
            "One deterministic seed/paired comparison cannot establish population stability or a failure probability.",
            "Only selected release ZIPs were downloaded by the parent, although full physical/call/held time series for all three attempts are available here. Not every candidate/event object is audited by this script.",
            "C uses native residuals as inputs. Most residual generation, contact solver state, and active Stand branches remain outside its demonstrated scope.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "phase": output["current_phase_status"],
                      "attempts": {a: {"status": v["stage"]["stage_status"], "simulation_s": v["stage"]["simulation_s"],
                                      "calls": v["planner_calls"]} for a, v in attempts.items()},
                      "totals": totals, "C_independent_check": c.get("CPU_independent_residual_verification")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
