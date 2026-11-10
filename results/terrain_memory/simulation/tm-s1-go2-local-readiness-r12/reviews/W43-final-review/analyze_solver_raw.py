"""Recompute W43 final E30 solver evidence from published original records.

No dynamics rollout, training, solver call, checkpoint edit, or physics experiment.
The optional MuJoCo operation below is kinematics only. Visual mesh elements are
removed from the supplied XML; joint/body/foot geometry is retained and its FK is
cross-checked against recorded physical foot positions.

Norm definitions follow the pinned author implementation e00c47a5:
https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/norm.cc
Costs use actual resolved weight readback and the recorded 176-node mean.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def rows(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(values):
    return dict(zip(["min", "median", "p95", "max"],
                    np.percentile(values, [0, 50, 95, 100]).tolist()))


def kinematics_model(xml_path):
    root = ET.parse(xml_path).getroot()
    removed = 0
    for parent in root.iter():
        for element in list(parent):
            if element.tag == "mesh" or (element.tag == "geom" and
                    ("mesh" in element.attrib or element.attrib.get("class") == "visual")):
                parent.remove(element)
                removed += 1
            elif element.tag == "geom" and element.attrib.get("type") == "mesh":
                element.set("type", "sphere")
                element.set("size", ".001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)
            for leg in ["FL", "FR", "RL", "RR"]]
    assert all(i >= 0 for i in feet)
    return model, mujoco.MjData(model), feet, removed


def cost_terms(residual, definitions):
    result = {}
    for term in definitions:
        q = residual[:, term["start"]:term["end"]]
        kind, scale = term["norm"], term["norm_parameter"]
        if kind == 0:
            costs = .5 * np.sum(q * q, axis=1)
        elif kind == 2:
            costs = np.sqrt(np.sum(q * q, axis=1) + scale * scale) - scale
        elif kind == 6:
            costs = np.sum(np.sqrt(q * q + scale * scale) - scale, axis=1)
        else:
            raise ValueError(f"Unimplemented norm in actual XML: {kind}")
        result[term["name"]] = float(np.mean(term["weight"] * costs))
    return result


def audit_stream(repo, raw, side, job, seed):
    base = repo / "results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12"
    resolved = json.loads((base / f"{job}-eval-PANN_PPO_S5-s{seed}-E30-resolved.json").read_text())
    xml_path = raw / f"{side}-e30-G1-project-go2.xml"
    definitions, offset = [], 0
    weights = resolved["applied_settings"][-1]["cost_weights"]
    for sensor in ET.parse(xml_path).findall(".//sensor/user"):
        name, dim = sensor.attrib["name"], int(sensor.attrib["dim"])
        user = [float(x) for x in sensor.attrib["user"].split()]
        definitions.append(dict(name=name, start=offset, end=offset + dim,
                                norm=int(user[0]), norm_parameter=user[4] if len(user) > 4 else None,
                                weight=weights[name]))
        offset += dim
    calls = rows(raw / f"{side}-e30-planner-calls.jsonl.gz")
    records = rows(raw / f"{side}-e30-planner-records.jsonl.gz")
    record_map = {r["plan_id"]: r for r in records}
    outbound_calls = [r for r in calls if r["origin"] >= 2.002 - 1e-8]
    outbound_records = [r for r in records if r["origin"] >= 2.002 - 1e-8]
    # cost_expected/cost_surprise use C++ std::to_string (six decimals).
    # The adjacent proxy_expected_decrease has full 17-digit precision and is
    # the chosen-alpha expectation, unlike calls.proxy_full_step_decrease.
    expected_errors, actual_ratios = [], []
    for record in outbound_records:
        alpha = record["action_step_alpha"]
        d0, d1 = record["dV_unscaled"]
        expected = -alpha * (d0 + alpha * d1) + 1e-16
        expected_errors.append(abs(expected - record["proxy_expected_decrease"]))
        assert record["cost_ratio_valid"] and expected > 0
        actual_ratios.append(record["cost_improvement"] / expected)
    assert max(expected_errors) < 1e-14
    gaps = []
    for row in outbound_calls:
        if not row["deployment_status"].startswith("reused") or not row["search_attempts"]:
            continue
        attempt = row["search_attempts"][0]
        current_v = row["incumbent_current_trunk_violation"]
        eligible = [c for c in attempt["native_candidates"] if not c["failed"] and
                    c["cost"] is not None and c["trunk_violation"] == current_v]
        best = min(c["cost"] for c in eligible)
        gaps.append(dict(call_id=row["call_id"], origin=row["origin"],
                         incumbent_cost=row["incumbent_current_cost"],
                         nominal_cost=attempt["nominal_current_cost"], best_candidate_cost=best,
                         missed_cost_improvement=row["incumbent_current_cost"] - best))
    tr = np.load(raw / f"{side}-e30-trajectory.npz")
    model, data, feet, removed = kinematics_model(xml_path)
    fk_errors = []
    for i in [0, 1001, 1990, 2000, 2060, 14950, 15000]:
        data.qpos[:] = tr["qpos"][i]
        mujoco.mj_kinematics(model, data)
        fk_errors.append(float(np.max(np.abs(data.geom_xpos[feet] - tr["feet"][i]))))
    feedback = rows(raw / f"{side}-e30-feedback-actions.jsonl.gz")
    by_query = {r["query_id"]: r for r in feedback}
    applied = [r for r in rows(raw / f"{side}-e30-action-execution.jsonl.gz")
               if r["query_kind"] == "applied_hold"]
    actual = [by_query[r["query_id"]] for r in applied]
    plan_reports = []
    for path in sorted(raw.glob(f"{side}-e30-accepted-reference-gain-H-plan*.npz")):
        x = np.load(path)
        pid = int(x["header"][1])
        record = record_map[pid]
        ac = cost_terms(x["accepted_residual"], definitions)
        ref = cost_terms(x["reference_residual"], definitions)
        accepted_sum, reference_sum = sum(ac.values()), sum(ref.values())
        assert abs(accepted_sum - x["header"][15]) < 1e-12
        assert abs(reference_sum - record["feedback_reference_cost"]) < 1e-12
        positions = []
        for state in x["accepted_states"]:
            data.qpos[:] = state[:model.nq]
            mujoco.mj_kinematics(model, data)
            positions.append(data.geom_xpos[feet].copy())
        positions = np.asarray(positions)
        clearances = positions[:, :, 2] - model.geom_size[feet, 0]
        report = dict(file=path.name, plan_id=pid, call_id=record["source_call_id"],
                      origin=float(x["header"][2]), nodes=len(x["accepted_states"]),
                      regularization_used=float(x["header"][8]),
                      accepted_cost_header=float(x["header"][15]),
                      accepted_cost_recomputed=accepted_sum,
                      reference_cost_recomputed=reference_sum,
                      native_local_cost_decrease=reference_sum - accepted_sum,
                      accepted_components=ac, reference_components=ref,
                      component_decrease={k: ref[k] - ac[k] for k in ac},
                      predicted_dx_m=float(x["accepted_states"][-1, 0] - x["accepted_states"][0, 0]),
                      predicted_dz_m=float(x["accepted_states"][-1, 2] - x["accepted_states"][0, 2]),
                      predicted_vx_range_m_s=[float(x["accepted_states"][:, 19].min()),
                                               float(x["accepted_states"][:, 19].max())],
                      foot_clearance_range_m=[float(clearances.min()), float(clearances.max())],
                      all_recorded_nodes_feet_below_ground_radius=bool(np.all(clearances < 0)),
                      H_eigenvalues=quantiles(np.linalg.eigvalsh(x["H"][:-1])),
                      lambda_over_H_diagonal=quantiles(x["header"][8] /
                          np.diagonal(x["H"][:-1], axis1=1, axis2=2)))
        times = x["accepted_times"][:11]
        if times[-1] <= tr["time"][-1] + 1e-8:
            ix = np.clip(np.searchsorted(tr["time"], times), 0, len(tr["time"]) - 1)
            before = np.maximum(ix - 1, 0)
            ix = np.where(abs(tr["time"][before] - times) < abs(tr["time"][ix] - times), before, ix)
            assert np.max(abs(tr["time"][ix] - times)) < 1e-8
            dq = tr["qpos"][ix] - x["accepted_states"][:11, :19]
            dv = tr["qvel"][ix] - x["accepted_states"][:11, 19:]
            report["first_20ms_actual_error"] = dict(
                base_xyz_max_abs_m=float(abs(dq[:, :3]).max()),
                joint_angle_max_abs_rad=float(abs(dq[:, 7:]).max()),
                qvel_max_abs=float(abs(dv).max()))
        action = next((a for a in applied if a["plan_id"] == pid and
                       abs(a["time_s"] - report["origin"]) < 1e-7), None)
        if action:
            report["installed_first_target_error_rad"] = float(np.max(np.abs(
                np.asarray(action["held_target"]) - x["accepted_actions"][0])))
        plan_reports.append(report)
    return dict(
        job=job, final_policy_update=55, trajectory_rows=len(tr["time"]),
        cost_definitions=definitions, cost_formula="weighted norm; arithmetic mean over all176 states",
        planner_calls=len(calls), accepted_plans=len(records), outbound_calls=len(outbound_calls),
        outbound_accepted_plans=len(outbound_records),
        accepted_alpha_counts=dict(collections.Counter(str(r["action_step_alpha"]) for r in outbound_records)),
        accepted_lambda_used_counts=dict(collections.Counter(str(r["regularization_used"]) for r in outbound_records)),
        actual_chosen_alpha_cost_surprise=quantiles(actual_ratios),
        recorded_six_decimal_cost_surprise=quantiles([r["cost_surprise"] for r in outbound_records]),
        selected_expected_formula_max_error=max(expected_errors),
        all_accepted_ratio_valid=all(r["cost_ratio_valid"] for r in outbound_records),
        accepted_lambda_after_counts=dict(collections.Counter(str(r["lambda_after"]) for r in outbound_calls
            if r["deployment_status"] == "accepted_native_descent")),
        accepted_attempt_counts=dict(collections.Counter(str(len(r["search_attempts"])) for r in outbound_calls
            if r["deployment_status"] == "accepted_native_descent")),
        reused_outbound_calls=len(gaps), reuse_candidate_gaps=gaps,
        positive_reuse_gaps=sum(g["missed_cost_improvement"] > 0 for g in gaps),
        maximum_reuse_gap=max(g["missed_cost_improvement"] for g in gaps),
        applied_hold_records=len(actual),
        action_QP_failures=sum(not r["qp_converged"] for r in actual),
        invalid_action_metrics=sum(not r["metric_valid"] for r in actual),
        projection_changes=sum(r["projection_changed"] for r in actual),
        max_PD_violation_Nm=max(max(map(abs, r["PD_violation_Nm"])) for r in actual),
        max_projection_delta_rad=max(max(map(abs, r["projection_delta"])) for r in actual),
        kinematics=dict(mujoco_version=mujoco.__version__, operation="mj_kinematics only; no integration",
                        visual_mesh_elements_removed=removed, nq=model.nq, nv=model.nv,
                        foot_order=["FL", "FR", "RL", "RR"], crosscheck_rows=7,
                        max_recorded_foot_position_difference_m=max(fk_errors)),
        saved_full_plan_count=len(plan_reports), plans=plan_reports,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--release-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    releases = json.loads(args.release_metadata.read_text())
    result = dict(schema="w43_final_solver_originals_audit_v1",
                  reviewed_commit="6a94c94c8e21e3b50fcca40ab4363165dee2987e",
                  scope="Both final update55 E30 original calls/actions and all34 published complete event-window plans; no new physics",
                  limitations=["Full accepted/reference matrices exist only for selected event windows.",
                               "Other candidate costs are recorded, but complete nonwinner trajectories/residuals are not published.",
                               "No counterfactual native solver, new rollout, training or checkpoint-content audit is performed here."],
                  streams={}, inputs=[])
    for side, job, seed in [("a", "job-20261110T035658Z-w43-a", 1),
                             ("b", "job-20261110T035700Z-w43-b", 2)]:
        result["streams"][side] = audit_stream(args.repo, args.raw, side, job, seed)
        assets = {a["name"]: a for a in releases[job]["assets"]}
        manifest = json.loads((args.repo / "results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs" /
                               job / "diagnostics/manifest.json").read_text())
        wanted = {"trajectory.npz", "planner-calls.jsonl.gz", "planner-records.jsonl.gz",
                  "feedback-actions.jsonl.gz", "action-execution.jsonl.gz", "G1-project-go2.xml"}
        for entry in manifest:
            basename = entry["name"].split("-g1__", 1)[-1]
            if basename not in wanted and "accepted-reference-gain-H-plan" not in basename:
                continue
            sequence = int(entry["name"].split("__f")[1][:6])
            if basename in wanted and sequence not in {1, 2, 4, 8, 9, 10}:
                continue
            if "accepted-reference-gain-H-plan" in basename and "-eval-" not in entry["path"]:
                continue
            local = args.raw / f"{side}-e30-{basename}"
            if not local.exists():
                continue
            asset = assets[entry["name"]]
            sha = digest(local)
            assert asset["digest"] == "sha256:" + sha
            assert asset["size"] == local.stat().st_size
            result["inputs"].append(dict(local_name=local.name, source_path=entry["path"],
                                          url=asset["url"], bytes=local.stat().st_size,
                                          sha256=sha, remote_digest_matched=True))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for side, s in result["streams"].items():
        print(side, "calls", s["planner_calls"], "full_plans", s["saved_full_plan_count"],
              "all_selected_predict_backward", all(p["predicted_dx_m"] < 0 for p in s["plans"]),
              "all_selected_no_foot_clearance", all(p["all_recorded_nodes_feet_below_ground_radius"] for p in s["plans"]))
    print("Verified input assets", len(result["inputs"]))


if __name__ == "__main__":
    main()
