#!/usr/bin/env python3
"""Audit already recorded W51 arrays. No model loading or dynamics execution.

Only Python, NumPy and XML parsing are used. Missing published objects remain
explicitly missing; this script never runs a simulator or fills absent data.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ATTEMPTS = {
    "D0": "job-20261115T112019Z-w51-d0-eval-PANN_PPO_S5-s1-D8-attempt01",
    "D1": "job-20261115T113036Z-w51-d1-command-repair-eval-PANN_PPO_S5-s1-D8-attempt01",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def definitions(path, weights):
    terms, offset = [], 0
    for i, sensor in enumerate(ET.parse(path).findall(".//sensor/user")):
        user = list(map(float, sensor.attrib["user"].split()))
        end = offset + int(sensor.attrib["dim"])
        terms.append(dict(name=sensor.attrib["name"], start=offset, end=end,
                          norm=int(user[0]), p=user[4] if len(user) > 4 else None,
                          weight=weights[i]))
        offset = end
    return terms


def cost_parts(residual, terms):
    result = {}
    for term in terms:
        x = residual[:, term["start"]:term["end"]]
        p = term["p"]
        if term["norm"] == 0:
            values = .5 * np.sum(x*x, axis=1)
        elif term["norm"] == 2:
            values = np.sqrt(np.sum(x*x, axis=1) + p*p) - p
        elif term["norm"] == 6:
            values = np.sum(np.sqrt(x*x + p*p) - p, axis=1)
        else:
            raise ValueError(term)
        result[term["name"]] = float(np.mean(values * term["weight"]))
    return result


def interpolate(t, grid, values):
    if t <= grid[0]:
        return values[0]
    if t >= grid[-1]:
        return values[-1]
    hi = int(np.searchsorted(grid, t, side="right")); lo = hi-1
    w = (t-grid[lo])/(grid[hi]-grid[lo])
    return (1-w)*values[lo]+w*values[hi]


def quaternion_product(a, b):
    return np.r_[a[0]*b[0]-np.dot(a[1:], b[1:]),
                 a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])]


def state_difference(reference, current):
    a, b = reference[3:7], current[3:7]
    if np.array_equal(a, b) or np.array_equal(a, -b):
        rotation = np.zeros(3)
    else:
        q = quaternion_product(a*np.array([1., -1., -1., -1.]), b)
        if q[0] < 0:
            q = -q
        length = np.linalg.norm(q[1:])
        rotation = np.zeros(3) if length == 0 else q[1:]*(2*np.arctan2(length, q[0])/length)
    return np.r_[current[:3]-reference[:3], rotation,
                 current[7:19]-reference[7:19], current[19:37]-reference[19:37]]


def raw_center(policy, x, t):
    """Read-only evaluation of the recorded full pre-clamp policy, no FD."""
    ts = policy["reference_times"]
    ref = interpolate(t, ts, policy["reference_states"]).copy()
    ref[3:7] /= np.linalg.norm(ref[3:7])
    value = interpolate(t, ts[:-1], policy["reference_actions"]).copy()
    K = interpolate(t, ts[:-1], policy["K"][:-1])
    value += policy["semantics"][5] * (K @ state_difference(ref, x))
    if len(policy["delta_times"]):
        value += interpolate(t, policy["delta_times"], policy["delta_values"])
    return value


def candidate_rows(path, call, terms):
    rows = []
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if not key.endswith("_header"):
                continue
            prefix = key[:-7]
            h = data[key]
            x, t = data[prefix+"_accepted_states"], data[prefix+"_accepted_times"]
            r = data[prefix+"_accepted_residual"]
            f = data[prefix+"_foot_rows"].reshape(-1, 29)
            feet = f[:, 1:].reshape(-1, 4, 7)
            force = feet[:, :, 4] + feet[:, :, 5]
            loads = force > 1.
            g = call["gait_context"]
            phase = g[8] + (f[:, 0] - g[9]) * g[10]
            wrapped = (phase[:, None]/(2*np.pi) - np.array([0., .5, .5, 0.]) + .5) % 1 - .5
            swing = np.abs(wrapped) < .275
            parts = cost_parts(r, terms)
            rows.append(dict(index=int(prefix[-3:]), kind=int(h[2]), slot=int(h[3]),
                alpha=float(h[4]), feedback_scale=float(h[10]), native_J=float(h[11]),
                risk=float(h[12]), failure=bool(h[13]), metric_valid=bool(h[14]),
                origin_s=float(h[15]), cost_parts=parts,
                cost_recompute_error=sum(parts.values())-float(h[11]),
                dx_m=float(x[-1, 0]-x[0, 0]),
                dy_m=float(x[-1, 1]-x[0, 1]),
                min_z_m=float(x[:, 2].min()), final_z_m=float(x[-1, 2]),
                endpoint_vx_m_s=float(x[-1, 19]),
                four_feet_over_1N_fraction=float(loads.all(axis=1).mean()),
                zero_feet_over_1N_fraction=float((~loads.any(axis=1)).mean()),
                foot_order=["FL", "FR", "RL", "RR"],
                foot_bottom_min_m=feet[:, :, 3].min(axis=0).tolist(),
                foot_bottom_max_m=feet[:, :, 3].max(axis=0).tolist(),
                foot_Fz_min_N=force.min(axis=0).tolist(),
                foot_Fz_max_N=force.max(axis=0).tolist(),
                foot_dx_m=(feet[-1, :, 0]-feet[0, :, 0]).tolist(),
                target_swing_loaded_fraction=[float(loads[:, j][swing[:, j]].mean())
                    if swing[:, j].any() else None for j in range(4)]))
    return rows


def kinematics(path):
    with np.load(path, allow_pickle=False) as z:
        raw = z["kinematics_payload"]
    n = int(raw[3]); i = 5
    ts = raw[i:i+n]; i += n
    targets = raw[i:i+12*n].reshape(n, 4, 3); i += 12*n
    qik = raw[i:i+12*n].reshape(n, 12); i += 12*n
    delta = raw[i:i+12*n].reshape(n, 12)
    return ts, targets, qik, delta


def step_height(phase):
    """The unchanged r20 C2 2 Hz / duty .45 / 5 mm profile, native order."""
    phase = np.asarray(phase)
    w = (phase[..., None]/(2*np.pi) - np.array([0., .5, .5, 0.]) + .5) % 1 - .5
    swing = np.abs(w) < .275
    elapsed = (w+.275)/2
    rising = elapsed < .075
    duration = np.where(rising, .075, .20)
    u = np.where(rising, elapsed/.075, (elapsed-.075)/.20)
    change = np.where(rising, .005, -.005)
    smooth = u**3*(10+u*(-15+6*u))
    h = np.where(swing, np.where(rising, 0., .005)+change*smooth, 0.)
    v = np.where(swing, change/duration*30*u*u*(1-u)*(1-u), 0.)
    return h, v


def all_kinematics(folder, moving):
    rows, missing = [], []
    xyz_min = np.full(3, np.inf); xyz_max = -xyz_min
    exact_reconstruction = []
    with np.load(folder/"trajectory.npz", allow_pickle=False) as z:
        plant_t = z["time"].copy()
        plant_feet = z["feet"].copy()[:, [0, 2, 1, 3], :]
        plant_velocity = z["feet_velocity"].copy()[:, [0, 2, 1, 3], :]
    for seed in moving:
        file = folder/seed["objects"]["IK"]["file"]
        if not file.exists():
            missing.append(seed["call_id"]); continue
        ts, targets, _, _ = kinematics(file)
        xyz_min = np.minimum(xyz_min, targets.min(axis=(0, 1)))
        xyz_max = np.maximum(xyz_max, targets.max(axis=(0, 1)))
        meta = seed["motion"]
        h0, v0 = step_height(meta["phase"])
        plant_i = int(np.argmin(np.abs(plant_t-seed["time_s"])))
        if abs(plant_t[plant_i]-seed["time_s"]) > 1e-9:
            raise ValueError("No actual same-time kinematic row")
        for j, foot in enumerate(meta["foot_context"]):
            mask = ts <= foot["end"] + 1e-10
            if not foot["swing"]:
                continue
            b = targets[mask, j, 2] - .022
            duration = foot["end"]-seed["time_s"]
            u = np.clip((ts[mask]-seed["time_s"])/duration, 0, 1)
            H0 = 1-u**3*(10+u*(-15+6*u))
            H1 = u-6*u**3+8*u**4-3*u**5
            height, _ = step_height(meta["phase"]+(ts[mask]-seed["time_s"])*4*np.pi)
            endpoint = np.asarray(foot["touchdown"])
            reconstructed = endpoint + H0[:, None]*(plant_feet[plant_i, j]-endpoint) + duration*H1[:, None]*plant_velocity[plant_i, j]
            reconstructed[:, 2] += height[:, j]-H0*h0[j]-duration*H1*v0[j]
            err = float(np.max(np.abs(reconstructed-targets[mask, j])))
            exact_reconstruction.append(err)
            rows.append(dict(call_id=seed["call_id"], time_s=seed["time_s"],
                foot=["FL", "RL", "FR", "RR"][j], segment=foot["segment"],
                current_phase_rad=meta["phase"], remaining_duration_s=duration,
                original_step_height_m=float(h0[j]), original_step_velocity_m_s=float(v0[j]),
                actual_foot_height_m=float(plant_feet[plant_i, j, 2]-.022),
                actual_foot_velocity_m_s=plant_velocity[plant_i, j].tolist(),
                target_bottom_min_m=float(b.min()), target_bottom_max_m=float(b.max()),
                target_dx_m=float(targets[mask, j, 0][-1]-targets[0, j, 0]),
                saved_target_formula_reconstruction_error_m=err))
    return dict(read_calls=len(moving)-len(missing), missing_calls=missing,
        scope="Current-swing leg snapshots per replan, not unique physical swings or failures. All minima/maxima and never-above counts below refer to saved control knots, not continuous polynomial extrema.",
        extrema_scope="saved_control_knots_only; no continuous derivative-root extrema were computed in this W51 review",
        flat_plane_z_m=0., foot_radius_m=.022,
        target_xyz_min_m=xyz_min.tolist(), target_xyz_max_m=xyz_max.tolist(),
        current_swing_snapshots=len(rows),
        never_above_ground_snapshots=sum(r["target_bottom_max_m"] <= 0 for r in rows),
        target_min_below_initial_by_over_1um_snapshots=sum(r["target_bottom_min_m"] < min(0., r["actual_foot_height_m"])-1e-6 for r in rows),
        target_min_below_minus_1mm_snapshots=sum(r["target_bottom_min_m"] < -.001 for r in rows),
        minimum_target_bottom_m=min(r["target_bottom_min_m"] for r in rows),
        maximum_formula_reconstruction_error_m=max(exact_reconstruction),
        rows=rows)


def inspect_ik(folder, seed, candidate_file=None, b_index=None, s_index=None):
    entry = seed["objects"]["IK"]
    file = folder/entry["file"]
    if not file.exists():
        return dict(status="published_object_not_in_review_download", file=entry["file"])
    ts, targets, qik, delta = kinematics(file)
    meta = seed["motion"]
    # All inspected W51 D1 horizons are on the original flat z=0 ground.
    bottom = targets[:, :, 2] - .022
    result = dict(status="read", file=entry["file"],
        extrema_scope="saved_control_knots_only; not continuous polynomial extrema",
        native_foot_order=["FL", "RL", "FR", "RR"],
        target_bottom_min_m=bottom.min(axis=0).tolist(),
        target_bottom_max_m=bottom.max(axis=0).tolist(),
        target_dx_m=(targets[-1, :, 0]-targets[0, :, 0]).tolist(),
        delta_max_abs_rad=float(np.max(np.abs(delta))),
        current_segments=[])
    for j, foot in enumerate(meta["foot_context"]):
        mask = ts <= foot["end"] + 1e-10
        result["current_segments"].append(dict(foot=result["native_foot_order"][j],
            swing=foot["swing"], begin_s=float(ts[0]), end_s=foot["end"],
            minimum_target_bottom_m=float(bottom[mask, j].min()),
            maximum_target_bottom_m=float(bottom[mask, j].max()),
            target_dx_m=float(targets[mask, j, 0][-1]-targets[0, j, 0])))
    if candidate_file and candidate_file.exists():
        with np.load(candidate_file, allow_pickle=False) as z:
            if b_index is not None:
                bx = z[f"candidate{b_index:03d}_accepted_states"]
                result["delta_vs_IK_minus_B_joint_max_error_rad"] = float(np.max(np.abs(delta-(qik-bx[:-1, 7:19]))))
            if s_index is not None:
                prefix = f"candidate{s_index:03d}"
                sx = z[prefix+"_accepted_states"]
                su = z[prefix+"_accepted_actions"]
                sf = z[prefix+"_foot_rows"].reshape(-1, 29)[:, 1:].reshape(-1, 4, 7)
                # IK feet are native FL/RL/FR/RR; saved rollout feet are FL/FR/RL/RR.
                target_record_order = targets[:, [0, 2, 1, 3], :]
                result["S_actual_vs_IK_foot_position_max_error_m"] = np.max(np.abs(sf[:-1, :, :3]-target_record_order), axis=0).tolist()
                result["S_actual_vs_IK_joint_max_error_rad"] = float(np.max(np.abs(sx[:-1, 7:19]-qik)))
                result["S_held_action_minus_IK_joint_range_rad"] = np.stack([(su-qik).min(axis=0), (su-qik).max(axis=0)]).tolist()
                result["S_calf_action_minus_IK_mean_rad"] = (su-qik)[:, [2, 5, 8, 11]].mean(axis=0).tolist()
    objects = seed["objects"]
    if all((folder/objects[role]["file"]).exists() for role in ("B", "S")):
        with np.load(folder/objects["B"]["file"], allow_pickle=False) as b, np.load(folder/objects["S"]["file"], allow_pickle=False) as s:
            names = ["reference_states", "reference_actions", "reference_times", "K", "H", "base_Qu", "base_Qxu", "base_active", "semantics", "provenance"]
            result["B_to_S_preserved_arrays"] = {name:bool(np.array_equal(b[name], s[name])) for name in names}
            interp = lambda obj: np.column_stack([np.interp(ts, obj["delta_times"], obj["delta_values"][:, j]) for j in range(12)]) if len(obj["delta_times"]) else np.zeros((len(ts), 12))
            result["S_minus_B_delta_vs_IK_delta_max_error_rad"] = float(np.max(np.abs(interp(s)-interp(b)-delta)))
            if candidate_file and candidate_file.exists() and b_index is not None and s_index is not None:
                b_arrays = {key:b[key] for key in ["reference_times", "reference_states", "reference_actions", "K", "semantics", "delta_times", "delta_values"]}
                raw_B = np.array([raw_center(b_arrays, x, t) for x, t in zip(bx[:-1], ts)])
                raw_B_on_S = np.array([raw_center(b_arrays, x, t) for x, t in zip(sx[:-1], ts)])
                bias = raw_B-bx[:-1, 7:19]
                feedback_change = raw_B_on_S-raw_B
                projection = su-(raw_B_on_S+delta)
                result["raw_map_decomposition"] = dict(
                    identity="u_S-q_IK = (c_B(x_B)-q_B) + (c_B(x_S)-c_B(x_B)) + (u_S-c_S(x_S)) at saved knots",
                    retained_nominal_calf_bias_mean_rad=bias[:, [2, 5, 8, 11]].mean(axis=0).tolist(),
                    feedback_change_max_abs_rad=float(np.max(np.abs(feedback_change))),
                    clamp_QP_change_max_abs_rad=float(np.max(np.abs(projection))),
                    decomposition_error_max_abs_rad=float(np.max(np.abs((su-qik)-(bias+feedback_change+projection)))),
                    interpretation="This algebra identifies the retained bias and feedback/projection contributions; it is not a causal ablation of support force.")
    return result


def audit(folder, arm):
    calls = read_jsonl(folder/"planner-calls.jsonl")
    boundaries = {r["call_id"]:r for r in read_jsonl(folder/"w50-planner-boundaries.jsonl")}
    seeds = read_jsonl(folder/"w51-seeds.jsonl")
    seed_by_id = {r["call_id"]:r for r in seeds}
    by_id = {r["call_id"]:r for r in calls}
    summary = dict(arm=arm, attempt=folder.name, calls=len(calls),
        source_counts=dict(Counter(r["candidate_source"] for r in calls)),
        seed_status_counts=dict(Counter(r["S_status"] for r in seeds)),
        windows=[], moving_updates=[], inspected_candidate_calls=[], candidates=[])
    if arm == "D1":
        manifest = json.loads((folder/"w51-event-manifest.json").read_text())
        context_count, command_error, missing_context = 0, 0., []
        for event in manifest["event_calls"]:
            context_file = folder/event["objects"]["context"]["file"]
            if not context_file.exists():
                missing_context.append(event["call_record"]["call_id"]); continue
            mission = event["boundary"]["mission_actor"]
            command = np.r_[mission["command_velocity"], mission["command_omega"]]
            with np.load(context_file, allow_pickle=False) as context:
                command_error = max(command_error, float(np.max(np.abs(context["native_userdata"][78:84]-command))))
            context_count += 1
        summary["native_command_repair_readback"] = dict(event_contexts_read=context_count,
            missing_event_contexts=missing_context,
            max_userdata78_84_minus_mission_world_command=command_error)
    for lo, hi in [(0, 2.001), (2.001, 4.3), (4.3, 8.01)]:
        rows = [r for r in calls if lo <= r["origin"] < hi]
        summary["windows"].append(dict(lo_s=lo, hi_s=hi, calls=len(rows),
            source_counts=dict(Counter(r["candidate_source"] for r in rows)),
            positive_installs=sum(r["policy_updated"] and r["candidate_source"].startswith("homotopy_") for r in rows)))
    for r in calls:
        if r["origin"] > 2 and r["policy_updated"]:
            summary["moving_updates"].append({k:r[k] for k in ["call_id", "origin", "candidate_source", "chosen_alpha", "policy_generation_time", "reference_start", "reference_end"]})
    moving = [r for r in seeds if "motion" in r]
    if moving:
        summary["seed_global"] = dict(count=len(moving),
            valid=sum(r["motion"]["S_valid"] for r in moving),
            lower_J_than_B=sum(r["motion"]["Q_J"] < r["motion"]["B_J"] for r in moving),
            higher_J_than_B=sum(r["motion"]["Q_J"] > r["motion"]["B_J"] for r in moving),
            max_IK_error_m=max(r["motion"]["IK_max_error_m"] for r in moving),
            failed_nodes=dict(Counter(r["motion"]["IK_failed_node"] for r in moving)),
            min_S_minus_B_J=min(r["motion"]["Q_J"]-r["motion"]["B_J"] for r in moving),
            max_S_minus_B_J=max(r["motion"]["Q_J"]-r["motion"]["B_J"] for r in moving))
        previous = {}; changes = []; max_drift = 0.
        for r in moving:
            for j, foot in enumerate(r["motion"]["foot_context"]):
                key = (j, foot["segment"])
                current = np.r_[foot["anchor"], foot["touchdown"], foot["end"]]
                if key in previous:
                    max_drift = max(max_drift, float(np.max(np.abs(current-previous[key]))))
                else:
                    changes.append(dict(call_id=r["call_id"], foot=j, segment=foot["segment"], swing=foot["swing"]))
                previous[key] = current
        summary["seed_global"]["max_anchor_touchdown_end_drift_within_same_segment"] = max_drift
        summary["seed_global"]["observed_leg_segments"] = len(changes)
        summary["all_IK"] = all_kinematics(folder, moving)
        grouped = {}
        for row in summary["all_IK"]["rows"]:
            grouped.setdefault(row["call_id"], []).append(row)
        summary["all_IK"]["old_trajectory_phase_opportunity"] = dict(
            meaning="Array-only description of the old W51 trajectory, not a prediction of a repaired controller or an additional admission gate.",
            all_current_swing_before_fixed_apex_calls=sum(all(r["remaining_duration_s"] > .2 for r in rows) for rows in grouped.values()),
            some_current_swing_after_apex_and_at_or_below_ground_calls=sum(any(r["remaining_duration_s"] <= .2 and r["actual_foot_height_m"] <= 0 for r in rows) for rows in grouped.values()),
            caution="Do not gate on exact sign of measured foot vz: roundoff-sized negative velocities are present.")
        summary["recorded_install_identities"] = dict(
            Q_content_identity_equals_S_all=all(r["objects"]["Q"]["content_sha256"] == r["objects"]["S"]["content_sha256"] for r in moving),
            B_is_installed=sum(r["objects"]["B"]["content_sha256"] == r["installed_policy_sha256"] for r in moving),
            S_is_installed=sum(r["objects"]["S"]["content_sha256"] == r["installed_policy_sha256"] for r in moving),
            scope="Existing recorder content identities; no new whole-repository hashing.")
    for file in sorted(folder.glob("r33-candidates-call*.npz")):
        cid = int(file.stem[-6:]); call = by_id[cid]
        if call["origin"] < 2:
            continue
        terms = definitions(folder/"model/G1-project-go2.xml", call["task_weights"])
        rows = candidate_rows(file, call, terms)
        nominals = [r for r in rows if r["kind"] == 0]
        base = min(nominals, key=lambda r:(r["risk"], r["native_J"]))
        seed = next((r for r in rows if r["kind"] == 4), None)
        positives = [r for r in rows if r["kind"] == 1]
        best = min(positives, key=lambda r:(r["risk"], r["native_J"])) if positives else None
        for row in rows:
            row["J_minus_B"] = row["native_J"]-base["native_J"]
            row["cost_parts_minus_B"] = {name:value-base["cost_parts"][name] for name, value in row["cost_parts"].items()}
            row["J_without_InputReference_minus_B_without_InputReference"] = row["J_minus_B"]-row["cost_parts_minus_B"]["InputReference"]
        entry = dict(call_id=cid, origin_s=call["origin"], source_file=file.name,
            command_velocity=boundaries[cid]["mission_actor"]["command_velocity"],
            selected_source=call["candidate_source"], baseline_index=base["index"],
            seed_index=seed["index"] if seed else None,
            best_positive_index=best["index"] if best else None,
            baseline=base, seed=seed, best_positive=best, all_candidates=rows)
        if seed and "objects" in seed_by_id[cid]:
            entry["IK"] = inspect_ik(folder, seed_by_id[cid], file, base["index"], seed["index"])
        summary["inspected_candidate_calls"].append(cid)
        summary["candidates"].append(entry)
    summary["last_call_reference"] = {k:calls[-1][k] for k in ["origin", "policy_generation_time", "policy_age_s", "reference_start", "reference_end", "reference_expired_by_s", "candidate_source"]}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("analysis.json"))
    args = parser.parse_args()
    result = dict(scope="Previously published W51 native arrays; no new physics, solve, FD, model loading or PPO.",
        source_commit="a621d7d35e6016634b388a3aef0d39d8d008aab0",
        diagnostic_load_threshold_N=1, threshold_is_not_qualification=True,
        candidate_foot_order=["FL", "FR", "RL", "RR"],
        IK_foot_order=["FL", "RL", "FR", "RR"], cases={})
    for arm, name in ATTEMPTS.items():
        folder = args.originals/name
        if (folder/"w51-seeds.jsonl").exists():
            result["cases"][arm] = audit(folder, arm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({arm:{"calls":r["calls"], "candidate_calls":r["inspected_candidate_calls"],
        "seed_global":r.get("seed_global")} for arm, r in result["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
