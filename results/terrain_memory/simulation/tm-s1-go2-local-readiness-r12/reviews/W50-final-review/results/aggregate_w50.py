#!/usr/bin/env python3
"""Recompute W50 report arithmetic from committed records; no simulation imports."""
import argparse
import ast
import collections
import json
from pathlib import Path

REL = Path("results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
WORK = ("native_new_rollouts", "transition_fd_calls", "risk_geometry_fd_calls",
        "derivative_passes", "backward_attempts", "candidate_records")
CONSTRAINT_INPUTS = ("J", "pos", "margin", "D", "vel", "aref", "frictionloss")


def read(path):
    return json.loads(path.read_text())


def compact_motion(m):
    out = {k: m[k] for k in ("J", "risk", "minimum_z_m", "final_z_m",
                            "forward_displacement_m", "vertical_velocity_range_m_s")}
    out["J_parts"] = m["J_parts"]
    out["J_parts_sum"] = sum(m["J_parts"].values())
    out["J_parts_sum_minus_J"] = out["J_parts_sum"] - m["J"]
    out["support_height_cost_fraction"] = m["J_parts"]["MaterialSupportHeight"] / m["J"]
    out["gait_cost_fraction"] = m["J_parts"]["Gait"] / m["J"]
    out["position_cost_fraction"] = m["J_parts"]["Position"] / m["J"]
    return out


def analyze_originals(root, originals):
    """Decode archived arrays and recompute arithmetic; never import a simulator."""
    import numpy as np
    parser_path = root / 'scripts/compare_s1_go2_w44_saved.py'
    tree = ast.parse(parser_path.read_text())
    scope = {'np': np}
    # Importing the full file would import mujoco and its Agent. Compile only
    # these pure array-parsing / cost-arithmetic function bodies instead.
    for name in ('take_policy_arrays', 'parse_candidates', 'cost_terms'):
        node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(parser_path), 'exec'), scope)
    index = read(originals.parent / 'extracted-index.json')
    candidates = []
    for folder, labels in [('task-b', ('A237', 'B236', 'A328', 'B625')),
                           ('deployment', ('A237', 'B236'))]:
        for label in labels:
            path = originals / folder / label / 'snapshot.bin.r33-candidates'
            definitions = read(originals / 'inputs' / label / 'input.json')['cost_definitions']
            rows = []
            for candidate in scope['parse_candidates'](path):
                metadata, arrays = candidate['metadata'], candidate['arrays']
                parts = scope['cost_terms'](arrays['residual'], definitions)
                total = sum(parts.values())
                states = arrays['states']
                rows.append(dict(**metadata, recomputed_J=total,
                    recomputed_J_minus_native=total-metadata['native_J'], J_parts=parts,
                    dominant_J_part=max(parts, key=parts.get),
                    minimum_z_m=float(states[:, 2].min()), final_z_m=float(states[-1, 2]),
                    forward_displacement_m=float(states[-1, 0]-states[0, 0]),
                    reference_start_s=float(arrays['reference_times'][0]),
                    reference_end_s=float(arrays['reference_times'][-1]),
                    all_state_action_values_finite=bool(np.isfinite(states).all() and
                                                       np.isfinite(arrays['actions']).all())))
            candidates.append(dict(folder=folder, label=label,
                path=str(path.relative_to(originals)), records=len(rows),
                physical_input_validity=('reconstructed saved historical boundary; not exact historical replay'
                    if folder == 'task-b' else 'midpoint state evolved by ABI-invalid bridge; not valid physical equivalence'),
                max_abs_J_recomputation_error=max(abs(x['recomputed_J_minus_native']) for x in rows),
                all_candidate_failure_flags_false=all(not x['failure'] for x in rows),
                positive_dx_candidates=[dict(slot=x['slot'], alpha=x['alpha'], native_J=x['native_J'],
                    minimum_z_m=x['minimum_z_m'], final_z_m=x['final_z_m'],
                    forward_displacement_m=x['forward_displacement_m'])
                    for x in rows if x['kind'] == 1 and x['forward_displacement_m'] > 0],
                candidates=rows))
    deployment = []
    for label in ('A237', 'B236'):
        path = originals / 'deployment' / label / 'real-interface-40ms.npz'
        with np.load(path, allow_pickle=False) as data:
            substeps = data['substeps']
            forces = substeps[:, :, :12]
            denormal = (np.abs(forces) > 0) & (np.abs(forces) < np.finfo('float64').tiny)
            queries = read(path.with_name('queries.json'))
            deployment.append(dict(label=label, path=str(path.relative_to(originals)),
                array_shapes={k: list(data[k].shape) for k in data.files},
                cast_equal=bool(np.array_equal(data['held'], data['native_double'].astype('float32').astype('float64'))),
                max_single_FP32_cast_roundoff_rad=float(np.max(np.abs(data['held']-data['native_double']))),
                query_count=len(queries), query_times=data['query_times'].tolist(),
                physics_substeps=int(substeps.shape[0]*substeps.shape[1]),
                substeps_with_denormal_actuator_field=int(np.any(denormal, axis=2).sum()),
                denormal_actuator_field_values=int(denormal.sum()),
                recorded_actuator_first_substep=forces[0, 0].tolist(),
                physical_equivalence_valid=False,
                support_or_actuator_measurements_usable=False))
    return dict(scope='All 114 raw candidate records decoded and residual J recomputed; both deployment NPZs and query JSONs directly read. Uses the downloaded Release payloads and source/member mapping already established by the main review, without another all-file hash pass.',
        no_new_simulation=True, no_simulator_import=True, parser_source=str(parser_path.relative_to(root)),
        array_library='numpy', extracted_member_count=len(index),
        extracted_bytes=sum(x['bytes'] for x in index), additional_all_payload_hash_pass=False,
        raw_candidate_records=sum(x['records'] for x in candidates),
        saved_problem_raw_candidate_records=sum(x['records'] for x in candidates if x['folder']=='task-b'),
        max_abs_J_recomputation_error=max(x['max_abs_J_recomputation_error'] for x in candidates),
        candidate_groups=candidates, deployment_arrays=deployment)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--originals", type=Path,
                    help='Optional extracted Release directory, with extracted-index.json in its parent')
    args = ap.parse_args()
    root = args.repo.resolve()
    base = root / REL / "w50"
    phase, a, b, dep, c = [read(base / (x + ".json")) for x in
                           ("phase", "task-a", "task-b", "deployment", "task-c")]
    sources = read(base / "delivery/finite/sources.json")
    manifest = read(base / "delivery/finite/manifest.json")
    delivery = read(base / "delivery/finite/delivery.json")
    publication = read(base / "publication.json")
    events = b["events"]  # streams contain copies; do not count those twice.
    replans = [check["midpoint"]["call"] for check in dep["checks"]]
    total_work = {k: sum(e["work"][k] for e in events) + sum(r[k] for r in replans)
                  for k in WORK}
    scalar_work = {
        "saved_problem_solves": sum(e["native_solves"] for e in events),
        "saved_problem_full_rollouts": sum(e["work"]["native_new_rollouts"] for e in events),
        "short_deployment_replans": len(replans),
        "short_deployment_full_rollouts": sum(r["native_new_rollouts"] for r in replans),
        "short_deployment_substeps": sum(check["physics_substeps"] for check in dep["checks"]),
        "raw_center_zero_physics_evaluations": sum(e["raw_center_evaluations"] for e in events)
            + sum(r["raw_center_evaluations"] for r in replans),
    }
    comparisons = []
    qualified = []
    for e in events:
        baseline = compact_motion(e["baseline"])
        selected = e["selected"]
        candidates = e["homotopy"]["candidates"]
        row = dict(label=e["label"], origin_s=e["origin_s"],
            source=e["selection_source"], call_wall_s=e["call"]["wall_s"],
            candidates=candidates, positive_count=len(candidates),
            strict_positive_count=sum(x["strict_better_than_B"] and not x["failure"] for x in candidates),
            candidate_failure_count=sum(x["failure"] for x in candidates),
            baseline=baseline, selected=None,
            work=e["work"], same_epoch_r35_reference=e["same_epoch_r35_reference"],
            historical_exact_replay=e["historical_exact_replay"])
        if selected:
            m = compact_motion(selected["motion"])
            row["selected"] = dict(slot=selected["slot"], alpha=selected["alpha"],
                motion=m, action_change_vs_baseline_rad=selected["action_change_vs_baseline_rad"],
                reference_start_s=selected["reference_start_s"], reference_end_s=selected["reference_end_s"],
                cost_decrease=baseline["J"]-m["J"],
                relative_cost_decrease=(baseline["J"]-m["J"])/baseline["J"],
                minimum_z_change_mm=(m["minimum_z_m"]-baseline["minimum_z_m"])*1000,
                final_z_change_mm=(m["final_z_m"]-baseline["final_z_m"])*1000,
                forward_displacement_change_mm=(m["forward_displacement_m"]-baseline["forward_displacement_m"])*1000,
                cost_decrease_over_roundoff_budget=(baseline["J"]-m["J"])/(
                    e["homotopy"]["baseline_budget"]+selected["budget"]))
        for q in e["qualified_proposals"]:
            qualified.append(dict(label=e["label"], slot=q["slot"], alpha=q["alpha"],
                selected=q["slot"] == (selected["slot"] if selected else None),
                motion=compact_motion(q["motion"]),
                action_change_vs_baseline_rad=q["action_change_vs_baseline_rad"]))
        comparisons.append(row)
    streams = []
    for stream in a["streams"]:
        windows = []
        for w in stream["support_windows"]:
            fields = ("start_s", "end_exclusive_s", "physical_samples", "state_samples",
                      "displacement_x_m", "min_z_m", "max_z_m", "min_vz_m_s", "max_vz_m_s",
                      "mean_foot_fz_N", "mean_nonfoot_fz_N", "nonfoot_gt1N_fraction",
                      "actual_loaded_during_target_swing_fraction",
                      "actual_loaded_during_target_stance_fraction",
                      "actual_held_pd_saturation_any_fraction", "max_continuous_loaded_s",
                      "max_continuous_unloaded_s", "target_height_rmse_m")
            z = {k:w[k] for k in fields}
            z["all_four_loaded_fraction"] = next(x["fraction"] for x in w["load_patterns"]
                if x["loaded_FL_FR_RL_RR"] == [1, 1, 1, 1])
            z["mean_nonfoot_vertical_support_over_weight"] = w["mean_nonfoot_fz_N"]/(15.206408*9.81)
            windows.append(z)
        streams.append(dict(stream=stream["stream"], source_attempt=stream["source_attempt"],
            observed_counts=stream["observed_counts"], clocks=stream["clocks"],
            gait=stream["private_gait"], support_windows=windows,
            actual_held_reference_query=stream["actual_held_reference_query"],
            recorded_same_policy_prediction_errors=stream["recorded_same_policy_prediction_errors"],
            samples=[dict(time_s=x["time_s"], requested_time_s=x["requested_time_s"],
                xyz=x["qpos_xyz"], vxyz=x["base_vxyz"], foot_fz_N=x["true_fz_N"],
                nonfoot_fz_N=x["nonfoot_fz_N"], saturated_joints=x["saturated_joints"])
                for x in stream["state_support_samples"]]))
    gpu_first = {}
    gpu_later = {}
    for condition, v in c["first_step_canonical_comparisons"].items():
        gpu_first[condition] = dict(
            canonical_constraint_keys_equal=v["canonical_constraint_keys_equal"],
            all_constraint_inputs_equal=all(v["constraints"][k]["equal"] for k in CONSTRAINT_INPUTS),
            all_smooth_inputs_equal=all(x["equal"] for x in v["smooth_inputs"].values()),
            contact_values_equal=v["contact_values_equal"], force=v["constraints"]["force"],
            outputs=v["outputs"], active_rows=v["active_rows"], active_contacts=v["active_contacts"])
        rows = [x for x in c["ten_step_summary"] if x["condition"] == condition]
        gpu_later[condition] = dict(steps=len(rows),
            contact_equal_steps=[x["substep"] for x in rows if x["contact_values_equal"]],
            contact_different_steps=[x["substep"] for x in rows if not x["contact_values_equal"]],
            first_different_constraint_inputs=next((x for x in rows if x["constraint_input_differences"]), None),
            max_qpos=max(x["qpos_max_abs"] for x in rows), max_qvel=max(x["qvel_max_abs"] for x in rows),
            max_qacc=max(x["qacc_max_abs"] for x in rows))
    cpu_a, cpu_b = b["streams"]
    overlap = max(0., min(cpu_a["ended_unix_s"],cpu_b["ended_unix_s"])
                  -max(cpu_a["started_unix_s"],cpu_b["started_unix_s"]))
    remote_record = {x["name"]:x for x in delivery["verified_assets"]}
    published = []
    for x in manifest:
        r = remote_record.get(x["name"], {})
        published.append(dict(name=x["name"], bytes=x["size"],
            recorded_uploaded=r.get("status")=="uploaded", size_matches_receipt=x["size"]==r.get("size"),
            digest_matches_receipt="sha256:"+x["sha256"]==r.get("remote_digest"), url=r.get("url")))
    result = dict(schema="w50_review_arithmetic_v2", result_commit="09bd55e0cb70244938ad82db7dd202a31ac70f77",
        inputs=[str((base/(x+'.json')).relative_to(root)) for x in
                ("phase","task-a","task-b","deployment","task-c","publication")],
        scope="Read-only arithmetic over committed per-event records and detailed server-derived summaries. External original payload scope is recorded separately when --originals is provided. No new simulation.",
        work=total_work|scalar_work, reported_phase_work=phase["work"],
        work_matches_phase=all(phase["work"][k]==v for k,v in (total_work|scalar_work).items()),
        original_evaluations_started=phase["original_evaluations_started"],
        original_E30_attempted=phase["original_E30_attempted"], original_E30_passed=phase["original_E30_passed"],
        blocked_stages=sum(not x["executed"] for stages in phase["original_stages"].values() for x in stages),
        gate=phase["original_gate"], original_stages=phase["original_stages"],
        finite_candidate_total=sum(x["positive_count"] for x in comparisons),
        strict_positive_total=sum(x["strict_positive_count"] for x in comparisons),
        finite_candidate_failure_total=sum(x["candidate_failure_count"] for x in comparisons),
        comparisons=comparisons, qualified_proposals=qualified,
        old_original_support=streams,
        deployment=dict(status=dep["status"], valid_native_equivalence_substeps=0,
            technical_issue=dep["technical_issue"],
            preserved_claims=["actual RPC response single-FP32 cast", "complete policy initial import/export",
                              "new-epoch time and policy provenance in independent native planner"],
            excluded_claims=["physical equivalence to native predictor", "measured actuator force or support", "historical plant replay"],
            replans=[dict(label=x["label"], time_s=x["midpoint"]["time_s"],
                source=x["midpoint"]["call"]["candidate_source"],
                reference_start_s=x["midpoint"]["reference_start_s"],
                reference_end_s=x["midpoint"]["reference_end_s"],
                raw_center_evaluations=x["midpoint"]["call"]["raw_center_evaluations"],
                work={k:x["midpoint"]["call"][k] for k in WORK}) for x in dep["checks"]]),
        gpu=dict(work=c["work"], first_step=gpu_first, later_steps=gpu_later,
            first_step_same_world_F1_F2=c["F1_vs_F2_first_step_same_world"],
            conclusions=c["conclusions"], qualification=c["qualification"], probe=c["probe"]),
        resources=dict(cpu_quota=phase["resources"]["cpu_quota"], allocation=phase["resources"],
            cpu_finite_stream_overlap_s=overlap, cpu_A_wall_s=cpu_a["wall_s"], cpu_B_wall_s=cpu_b["wall_s"],
            C_wall_s=c["wall_s"], C_overlaps_finite_streams=c["ended_unix_s"]>min(cpu_a["started_unix_s"],cpu_b["started_unix_s"]),
            measured_cpu_or_gpu_utilization=None,
            finite_elapsed_s=phase["finished_finite_unix_s"]-phase["started_unix_s"],
            until_publication_elapsed_s=publication["ended_unix_s"]-phase["started_unix_s"]),
        delivery=dict(status=delivery["status"], archive_count=len(manifest), archive_bytes=sum(x["size"] for x in manifest),
            source_count=len(sources), source_bytes=sum(x["size"] for x in sources),
            source_suffix_counts=dict(collections.Counter(Path(x["source_path"]).suffix for x in sources)),
            video_sources=[x for x in sources if Path(x["source_path"]).suffix.lower() in (".mp4",".mov",".avi",".webm")],
            assets=published, payloads_downloaded_or_rehashed_in_this_review=False,
            computer_archive=phase["computer_archive"], server_originals_retained=phase["server_originals_retained"]))
    if args.originals is not None:
        result['original_payload_review'] = analyze_originals(root, args.originals.resolve())
        result['delivery']['payloads_downloaded_or_rehashed_in_this_review'] = True
    out = Path(__file__).with_name("analysis.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(out), work_matches_phase=result["work_matches_phase"],
        work=result["work"], qualified=f'{result["strict_positive_total"]}/{result["finite_candidate_total"]}',
        original_E30_attempted=result["original_E30_attempted"],
        cpu_finite_overlap_s=overlap, archive_bytes=result["delivery"]["archive_bytes"]),ensure_ascii=False))


if __name__ == "__main__":
    main()
