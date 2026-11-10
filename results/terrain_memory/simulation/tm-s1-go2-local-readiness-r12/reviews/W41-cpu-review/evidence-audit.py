#!/usr/bin/env python3
"""Recompute delivered W41 CPU evidence. No network, training, or GPU access.

Run from any directory; writes cpu-progress.json beside this file. The unit of
training accounting is (job/stream, transition_serial), never an inherited
session_records entry counted again. Historical controller comparisons are
descriptive, not matched causal experiments. Compact evidence is read in full;
the manifest is not treated as proof that all large ZIP members were read.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import statistics as st
import subprocess
import zlib


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
BASE = REPO / "results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12"
BASELINE = "9cc42d5a9189a2bfd7e1e6cd99b9300045a16724"
READ: dict[str, dict] = {}
JOBS = {"a": "job-20261107T100149Z-w41-a", "b": "job-20261107T093906Z-w41-b"}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    data = ([json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            if path.suffix == ".jsonl" else json.loads(raw))
    READ[str(path.relative_to(REPO))] = {
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "jsonl_rows": len(data) if path.suffix == ".jsonl" else None,
    }
    return data


def percentile(values, q):
    a = sorted(values)
    if not a:
        return None
    x = (len(a) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return a[lo] + (a[hi] - a[lo]) * (x - lo)


def summary(values):
    values = [v for v in values if v is not None]
    return (dict(n=len(values), min=min(values), q25=percentile(values, .25),
                 median=st.median(values), mean=st.mean(values), q75=percentile(values, .75),
                 p90=percentile(values, .9), max=max(values)) if values else dict(n=0))


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def family(reason):
    return str(reason).split(":")[0]


def episode(result, events=None, manifest=None):
    events, manifest = events or {}, manifest or {}
    progress = result.get("stage_progress", {})
    traversal = progress.get("material_traversal", {})
    route = result.get("legal_route") or {}
    first = events.get("first_material_events_by_type", {})
    load = first.get("load", {}).get("event", {}).get("time_s")
    event_rows = manifest.get("events", [])
    first_times = {}
    for item in event_rows:
        first_times.setdefault(item["kind"], item["time_s"])
    if load is None:
        load = first_times.get("first_material_load")
    natural = family(result.get("reason")) in {"body_fall", "invalid_contact", "material_domain_exit"}
    sim = result["simulation_s"]
    material = result.get("material", {})
    return {
        "attempt": result["attempt"],
        "result_source": str((BASE / (result["attempt"] + ".json")).relative_to(REPO)),
        "purpose": result.get("purpose"), "reason": result.get("reason"),
        "reason_family": family(result.get("reason")), "stage_status": result.get("stage_status"),
        "result_error": result.get("error"), "success": result.get("success"),
        "natural_physical_terminal": natural,
        "terminated": result.get("policy", {}).get("terminated"),
        "truncated": result.get("policy", {}).get("truncated"),
        "simulation_s": sim, "wall_s": result.get("wall_s"),
        "max_x_m": result.get("max_x_m"), "final_xyz_m": result["final_qpos"][:3],
        "actual_route_fraction": progress.get("actual_route_fraction"),
        "material_loaded": progress.get("actual_material_loaded"),
        "first_load_s": load,
        "time_first_load_to_end_s": sim - load if load is not None else None,
        "base_entry_s": traversal.get("west_entry_s"),
        "time_base_entry_to_end_s": sim - traversal["west_entry_s"] if traversal.get("west_entry_s") is not None else None,
        "east_exit_s": traversal.get("east_exit_s"), "lateral_exit_s": traversal.get("lateral_exit_s"),
        "first_outbound_material_traversal_completed": traversal.get("completed", False),
        "base_entry_and_foot_loading": bool(traversal.get("west_entry_s") is not None and traversal.get("material_loaded_during_crossing")),
        "far_arrival_s": progress.get("far_arrival_s"), "return_start_s": progress.get("return_start_s"),
        "raw_roundtrips": result.get("completed_round_trips", 0),
        "legal_route_available": bool(route),
        "legal_outbound": route.get("legal_outbound_traversals"),
        "legal_return": route.get("legal_return_traversals"),
        "legal_roundtrips": route.get("completed_roundtrips"),
        "final_legal_progress": route.get("progress"), "side_exits": route.get("side_exits"),
        "passages": route.get("passages"),
        "terminal_reward_settlement": result.get("terminal_reward_settlement"),
        "max_abs_principal_strain": material.get("max_abs_principal_strain"),
        "material_domain_exit": material.get("domain_exit"),
        "material_event_counts": events.get("material_event_counts"),
        "first_event_times_s": first_times,
        "event_counts": dict(Counter(x["kind"] for x in event_rows)),
        "saved_event_plans": len(manifest.get("plans", [])),
        "before_load": {
            kind: (first_times.get(kind) is not None and load is not None and first_times[kind] < load)
            for kind in ("first_nominal_PD_saturation", "feedback_added_saturation", "large_raw_feedback", "command_nonprogress_2s")
        },
        "source_commit": result.get("source_commit"),
        "execution_fingerprint": result.get("execution_fingerprint"),
        "configuration_fingerprint": result.get("full_configuration_sha256"),
        "timing": result.get("timing", {}),
        "r29_failure": result.get("r29_failure"),
    }


def aggregate(rows):
    natural = [x for x in rows if x["natural_physical_terminal"]]
    loaded = [x for x in rows if x["material_loaded"]]
    timing = {}
    for k in sorted({k for x in rows for k in x["timing"]}):
        entries = [x["timing"][k] for x in rows if k in x["timing"]]
        n = sum(x.get("count", 0) for x in entries)
        total = sum(x.get("total_s", 0) for x in entries)
        timing[k] = dict(count=n, total_s=total, mean_s=total / n if n else None,
                         deadline_misses=sum(x.get("deadline_misses", 0) for x in entries))
    event_kinds = sorted({k for x in rows for k in x["event_counts"]})
    return {
        "episodes": len(rows), "natural_physical_terminals": len(natural),
        "reason_counts": dict(Counter(x["reason_family"] for x in rows)),
        "full_invalid_contact_reasons": dict(Counter(x["reason"] for x in rows if x["reason_family"] == "invalid_contact")),
        "stage_status_counts": dict(Counter(x["stage_status"] for x in rows)),
        "success_true_episodes": sum(x["success"] is True for x in rows),
        "all_result_errors_null": all(x["result_error"] is None for x in rows),
        "r29_failure_packets": sum(x["r29_failure"] is not None for x in rows),
        "natural_lifetime_s": summary([x["simulation_s"] for x in natural]),
        "all_lifetime_s": summary([x["simulation_s"] for x in rows]),
        "natural_max_x_m": summary([x["max_x_m"] for x in natural]),
        "all_max_x_m": summary([x["max_x_m"] for x in rows]),
        "actual_route_fraction": summary([x["actual_route_fraction"] for x in rows]),
        "material_loaded_episodes": len(loaded),
        "base_entry_episodes": sum(x["base_entry_s"] is not None for x in rows),
        "base_entry_and_foot_loading_episodes": sum(x["base_entry_and_foot_loading"] for x in rows),
        "loaded_without_base_entry": sum(x["material_loaded"] and x["base_entry_s"] is None for x in rows),
        "east_exit_episodes": sum(x["east_exit_s"] is not None for x in rows),
        "lateral_exit_episodes": sum(x["lateral_exit_s"] is not None for x in rows),
        "first_outbound_material_traversal_completed": sum(x["first_outbound_material_traversal_completed"] for x in rows),
        "far_arrival_episodes": sum(x["far_arrival_s"] is not None for x in rows),
        "return_started_episodes": sum(x["return_start_s"] is not None for x in rows),
        "all_legal_route_available": all(x["legal_route_available"] for x in rows),
        "legal_outbound": sum(x["legal_outbound"] or 0 for x in rows) if all(x["legal_route_available"] for x in rows) else None,
        "legal_return": sum(x["legal_return"] or 0 for x in rows) if all(x["legal_route_available"] for x in rows) else None,
        "legal_roundtrips": sum(x["legal_roundtrips"] or 0 for x in rows) if all(x["legal_route_available"] for x in rows) else None,
        "raw_roundtrips": sum(x["raw_roundtrips"] for x in rows),
        "natural_first_load_to_end_s": summary([x["time_first_load_to_end_s"] for x in natural]),
        "natural_loaded_ends_within_half_second": sum(x["time_first_load_to_end_s"] is not None and x["time_first_load_to_end_s"] <= .5 for x in natural),
        "natural_base_entry_to_end_s": summary([x["time_base_entry_to_end_s"] for x in natural]),
        "natural_positive_final_legal_progress": sum((x["final_legal_progress"] or 0) > 0 for x in natural),
        "retracted_positive_progress": sum((x["terminal_reward_settlement"] or {}).get("retracted_positive_credit", 0) for x in rows),
        "loaded_event_precedence_counts": {k: sum(x["before_load"][k] for x in loaded) for k in (rows[0]["before_load"] if rows else {})},
        "episodes_with_event": {k: sum(x["event_counts"].get(k, 0) > 0 for x in rows) for k in event_kinds},
        "event_occurrences": {k: sum(x["event_counts"].get(k, 0) for x in rows) for k in event_kinds},
        "physics_s": sum(x["simulation_s"] for x in rows),
        "episode_wall_sum_s": sum(x["wall_s"] or 0 for x in rows), "timing": timing,
        "max_abs_principal_strain": summary([x["max_abs_principal_strain"] for x in rows]),
    }


# Reuse the already published reward/GAE arithmetic, after inspecting its source.
# This imports definitions only; it does not call the prior review's main().
spec = importlib.util.spec_from_file_location("w39_math", HERE.parent / "W39-review/analyze_results.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def numeric(transitions, values):
    out = old.numeric(transitions, values)
    out["all_rows_in_saved_training_state"] = all(x.get("in_saved_training_state") for x in transitions)
    out["all_pairs_unique"] = len({(x["episode"], x["transition_serial"]) for x in transitions}) == len(transitions)
    out["phase_start_summed_dt_s"] = sum(x["dt_s"] for x in transitions)
    out["missing_or_nonpositive_dt"] = sum(x.get("dt_s", 0) <= 0 for x in transitions)
    out["pann_nonzero_serials"] = out["pann_nonzero_serials"][:16]
    out["pann_nonzero_serials_note"] = "First 16 only; total count is exact. Episode and input JSONL provide all serials."
    return out


def event_windows(root):
    result = {}
    for folder in sorted((root / "numeric-evidence/event-windows").glob("*")):
        if not folder.is_dir():
            continue
        meta = read(folder / "event-summary.json")
        plans, actions, physical = {}, {}, {}
        columns = None
        for path in sorted(folder.glob("window-*.json")):
            d = read(path)
            if path.name.endswith("-plans.json"):
                for row in d["records"]:
                    plans[row["plan_id"]] = row
            elif path.name.endswith("-control.json"):
                for row in d["actions"]:
                    actions[row["control_time_s"]] = row
            elif path.name.endswith("-physical.json"):
                columns = d["columns"]
                for row in d["rows"]:
                    physical[row[0]] = row
        record = [x["record"] for x in actions.values()]
        derived = [x["derived"] for x in actions.values()]
        force_max = [max(abs(a), abs(b)) for a, b in meta["forcerange"]]
        actual_excess, raw_excess, substep0, remaining = [], [], [], []
        nonfoot_positive = 0
        if columns:
            ai = [columns.index(f"actuator_force_{i}") for i in range(12)]
            ri = [columns.index(f"unclipped_pd_request_{i}") for i in range(12)]
            ni = columns.index("nonfoot_grf_z")
            for t, row in physical.items():
                actual_excess.append(max(max(0, abs(row[j]) - lim) for j, lim in zip(ai, force_max)))
                excess = max(max(0, abs(row[j]) - lim) for j, lim in zip(ri, force_max))
                raw_excess.append(excess)
                # Integer 0.2 ms steps, grouped into the unchanged 2 ms hold.
                (substep0 if round(t / .0002) % 10 == 0 else remaining).append(excess)
                nonfoot_positive += row[ni] > 0
        result[folder.name] = {
            "scope": "Selected merged event windows only, not all episode planner calls or all physics steps.",
            "source": str((folder / "event-summary.json").relative_to(REPO)),
            "selected_duration_s": meta["selected_duration_s"], "windows_s": meta["windows_s"],
            "event_times_s": {k: x.get("time_s") for k, x in meta["events"].items()},
            "selected_unique_plans": len(plans), "selected_unique_controls": len(actions),
            "selected_unique_physics_rows": len(physical),
            "all_selected_metric_valid": all(x.get("metric_valid") for x in record),
            "all_selected_projection_converged": all(x.get("qp_converged") for x in record),
            "projection_status_counts": dict(Counter(x.get("projection_status") for x in record)),
            "selected_refinement_attempts": sum(bool(x.get("qp_refinement_attempted")) for x in record),
            "selected_max_scaled_KKT": max((x.get("kkt_scaled_residual", 0) for x in record), default=None),
            "selected_max_primal_rad": max((x.get("primal_residual_rad", 0) for x in record), default=None),
            "max_raw_K_delta_x_rad": max((x["max_abs_raw_K_delta_x_rad"] for x in derived), default=None),
            "max_target_clip_rad": max((x["max_target_clip_rad"] for x in derived), default=None),
            "accepted_trunk_violation_max_m": max((x["accepted_trunk_violation_m"] for x in plans.values()), default=None),
            "acceptance_mode_counts": dict(Counter(x["acceptance_mode"] for x in plans.values())),
            "raw_PD_over_original_force_limit_Nm": summary(raw_excess),
            "hold_first_substep_max_raw_PD_excess_Nm": max(substep0, default=None),
            "hold_remaining_nine_max_raw_PD_excess_Nm": max(remaining, default=None),
            "actual_actuator_force_excess_max_Nm": max(actual_excess, default=None),
            "physics_rows_nonfoot_Fz_positive": nonfoot_positive,
            "limitations": meta["missing"] + [
                "The manifest feedback_added_saturation event compares pre-projection center with nominal PD, not applied force.",
                "Raw PD violations near 1e-6 Nm may be target serialization rounding; actual actuator clamp is checked separately.",
                "These event windows do not contain complete planner-call or stderr soft-warning logs.",
            ],
        }
    return result


def build_stream(suffix):
    jobid = JOBS[suffix]
    root = BASE / "jobs" / jobid
    job = read(root / "job.json")
    progress = read(root / "training/progress.json")
    stopped = read(root / "training/stopped.json")
    collection = read(root / "training/collection.json")
    root_collection = read(root / "collection.json")
    identity = read(root / "training/stream-identity.json")
    learning = read(root / "training/learning.json")
    evaluation = read(root / "evaluation.json")
    ledger = read(root / "attempt-ledger.json")
    contract = read(root / "execution-contract.json")
    controller = read(root / "controller-identity.json")
    index = read(root / "numeric-evidence/index.json")
    rows, transitions, values, checks, configs, runtime_identities = [], [], [], [], [], []
    previous = job["initial_transitions"]
    for path in sorted(BASE.glob(f"train-{jobid}-*.json")):
        if path.name.endswith("-resolved.json"):
            continue
        result = read(path)
        ep = int(path.stem.rsplit("ep", 1)[-1])
        small = read(root / "training" / f"episode-{ep:05d}.json")
        resolved = read(path.with_name(path.stem + "-resolved.json"))
        configs.append(resolved["r12_configuration"]["r12_contract"]["joint_learning"]["ppo"])
        runtime_identities.append(dict(r11_source_sha256=resolved["r11_source_sha256"],
                                       author_source_sha256=resolved["author_source_sha256"],
                                       model_mjb_sha256=resolved["model_mjb_sha256"],
                                       agent_server_sha256=resolved["agent_server"]["sha256"],
                                       workers=resolved["workers"]))
        caps = read(root / "episodes" / result["attempt"] / "cap-decisions.json")
        events = read(root / "numeric-evidence/episodes" / (result["attempt"] + "-events.json"))
        manifest = read(root / "episodes" / result["attempt"] / "r29-event-manifest.json")
        ts = read(root / "numeric-evidence/training" / f"episode-{ep:05d}-transitions.jsonl")
        vs = read(root / "numeric-evidence/training" / f"episode-{ep:05d}-pann-values.jsonl")
        row = episode(result, events, manifest)
        nonzero = [v for v in vs if any(x != 0 for x in (v.get("pann_recovery_m") or []))]
        row.update(episode=ep, transitions=len(ts), reward_sum=sum(x["reward"] for x in ts),
                   current_pann_nonzero_rows=len(nonzero),
                   cap_m_s=summary([x["cap_m_s"] for x in ts]),
                   behavior_update_counts=dict(Counter(x["behavior_updates"] for x in ts)),
                   first_transition=ts[0]["transition_serial"], last_transition=ts[-1]["transition_serial"],
                   terminal_transition={k: ts[-1][k] for k in ("transition_serial", "reward", "terminated", "truncated", "dt_s", "signed_route_fraction", "failure", "intrinsic_timeout")},
                   startup_regression_status=small.get("startup_regression_check", {}).get("status"),
                   controller_build=small["controller_build"], controller_artifact=small["controller_artifact"],
                   nominal_workers=resolved["workers"], execution_cpus=resolved.get("execution_cpus"))
        current = result["policy"]["total_transitions"]
        checks.append(dict(episode=ep, result_counter_delta=current - previous,
                           small_count=small["transitions"], cap_rows=len(caps), transition_rows=len(ts), value_rows=len(vs),
                           all_equal=current - previous == small["transitions"] == len(caps) == len(ts) == len(vs),
                           same_terminal=(small["reason"] == result["reason"] and small["stage_status"] == result["stage_status"]),
                           small_route_matches=small["legal_route"] == result["legal_route"]))
        previous = current
        rows.append(row)
        transitions.extend(ts)
        values.extend(vs)
    batches = []
    for update in learning["updates"]:
        ts = [t for t in transitions if t["behavior_updates"] == update["update_count"] - 1]
        vs = [v for v in values if v["behavior_updates"] == update["update_count"] - 1]
        scalar_update = {k: v for k, v in update.items() if not isinstance(v, (dict, list))}
        batches.append(dict(update=scalar_update, numeric=numeric(ts, vs),
                            gae_arithmetic=old.gae_arithmetic(ts, vs, update, configs[0]["gamma"], configs[0]["gae_lambda"])))
    eval_rows = []
    for attempt in evaluation["attempts"]:
        result = read(BASE / (attempt + ".json"))
        resolved = read(BASE / (attempt + "-resolved.json"))
        events = read(root / "numeric-evidence/episodes" / (attempt + "-events.json"))
        manifest = read(root / "episodes" / attempt / "r29-event-manifest.json")
        row = episode(result, events, manifest)
        edir = root / "numeric-evidence/evaluation" / attempt
        ts = [x for f in sorted(edir.glob("*-transitions.jsonl")) for x in read(f)]
        vs = [x for f in sorted(edir.glob("*-pann-values.jsonl")) for x in read(f)]
        row["numeric"] = numeric(ts, vs) if ts and vs else None
        row["qualification"] = result.get("qualification")
        row["case_identity"] = result.get("case_identity")
        eval_rows.append(row)
    nat = [x for x in rows if x["natural_physical_terminal"]]
    half = len(nat) // 2
    m = identity.get("controller_version_boundary", {})
    lineage = identity["phase_lineage"]
    metrics = numeric(transitions, values)
    ids = sorted(t["transition_serial"] for t in transitions)
    saved_current_sessions = [s for s in progress["session_records"] if s.get("attempt", "").startswith("train-" + jobid)]
    return {
        "job": jobid, "group": job["group"], "seed": job["seed"],
        "controller_build": job["controller_build"], "controller_artifact": progress["controller_artifact"],
        "workers": job["mpc_workers"], "evaluation_workers": job["evaluation_solver_workers"],
        "phase_lineage": lineage, "ancestor_fork_identity": identity["fork_identity"],
        "parent_checkpoint": job["resume_checkpoint"], "current_resume_checkpoint": progress["checkpoint"],
        "last_completed_update_checkpoint": progress["last_completed_update"]["path"],
        "training_counts": {k: stopped.get(k) for k in ("actual_transitions", "updates", "unupdated_fragment", "excluded_old_version_transitions", "phase_initial_transitions", "phase_new_transitions", "phase_initial_updates", "phase_new_updates")},
        "training_end": {k: stopped.get(k) for k in ("reason", "last_stage_status", "last_episode_reason", "last_error", "last_r29_failure", "invocation_started_unix_s", "invocation_wall_s", "stopped_unix_s", "deadline_unix_s")},
        "no_new_fork_or_reseed": lineage["new_fork"] is False and lineage["reseeded_after_checkpoint_restore"] is False,
        "initial_tail": identity["initial_tail_buffer"], "continuous_environment_restored": identity["continuous_environment_restored"],
        "controller_artifact_history": progress["controller_artifact_history"],
        "stale_nested_controller_version_boundary": {
            "active_controller_build": m.get("active_controller_build"),
            "parent_controller_build": m.get("parent_controller_build"),
            "counter_semantics": "This legacy nested field is stale; actual phase_lineage, artifact, results and 4096 transitions identify S5-r30."
        },
        "summary": aggregate(rows), "numeric": metrics, "episodes": rows, "updates": batches,
        "temporal_comparison": {
            "basis": "First/last equal-size halves of natural episodes only; technical and all truncations excluded.",
            "first_half": aggregate(nat[:half]), "last_half": aggregate(nat[-half:]),
            "omitted_middle_episode": nat[half]["episode"] if len(nat) % 2 else None,
        },
        "checks": {
            "phase_serials_contiguous_exactly_once": ids == list(range(job["initial_transitions"] + 1, stopped["actual_transitions"] + 1)),
            "count_identity_256_updates_plus_tail_plus_excluded": progress["samples"] == progress["updates"] * 256 + progress["unupdated_fragment"] + progress["excluded_old_version_transitions"],
            "all_episode_counts_match": all(x["all_equal"] for x in checks),
            "all_episode_terminal_statuses_match": all(x["same_terminal"] for x in checks),
            "all_small_route_ledgers_match": all(x["small_route_matches"] for x in checks),
            "saved_current_phase_sessions_match": [x["attempt"] for x in saved_current_sessions] == [x["attempt"] for x in rows],
            "current_phase_session_record_count": len(saved_current_sessions),
            "all_update_batches_256": all(x["numeric"]["rows"] == 256 for x in batches),
            "updates_38_through_45": [x["update_count"] for x in learning["updates"]] == list(range(38, 46)),
            "all_resolved_ppo_configs_equal": all(x == configs[0] for x in configs),
            "all_runtime_sources_model_server_workers_equal": all(x == runtime_identities[0] for x in runtime_identities),
            "malformed_learning_lines": learning["malformed_lines"],
            "root_and_training_collections_agree": all(root_collection.get(k) == collection.get(k) for k in ("collected_transitions", "updates", "unupdated_fragment", "phase_new_transitions", "phase_new_updates")),
            "all_recorded_artifacts_r30": all(x["controller_artifact"] == "S5-r30" for x in rows) and all(x["controller_artifact"] == "S5-r30" for x in transitions),
        },
        "episode_count_checks": checks, "resolved_ppo_config": configs[0],
        "runtime_identity": runtime_identities[0],
        "source_commit_counts": dict(Counter(x["source_commit"] for x in rows)),
        "execution_fingerprint_counts": dict(Counter(x["execution_fingerprint"] for x in rows)),
        "evaluation": {
            "checkpoint": evaluation["checkpoint"], "checkpoint_transitions": evaluation["checkpoint_transitions"],
            "checkpoint_updates": evaluation["checkpoint_updates"], "status": evaluation["status"],
            "deterministic_policy_mean": evaluation["deterministic_policy_mean"],
            "ppo_updates": evaluation["ppo_updates"], "stages": evaluation["stages"], "episodes": eval_rows,
        },
        "event_windows": event_windows(root),
        "throughput": {
            "phase_transitions": stopped["phase_new_transitions"], "training_wall_s": stopped["invocation_wall_s"],
            "transitions_per_training_wall_hour": stopped["phase_new_transitions"] * 3600 / stopped["invocation_wall_s"],
            "physics_s_per_training_wall_second": sum(x["simulation_s"] for x in rows) / stopped["invocation_wall_s"],
            "basis": "One invocation including training startup; excludes final E30 and publication. No inference about current GPU resource use.",
        },
    }


def historical_results():
    cohorts = {}
    # W40 current files include A's four qp2 continuation episodes 315..318.
    for suffix in ("a", "b"):
        job = f"job-20261106T103028Z-w40-{suffix}"
        root = BASE / "jobs" / job
        rows = []
        for f in sorted(BASE.glob(f"train-{job}-*.json")):
            if not f.name.endswith("-resolved.json"):
                row = episode(read(f))
                row["episode"] = int(f.stem.rsplit("ep", 1)[-1])
                rows.append(row)
        stop = read(root / "training/stopped.json")
        ev = read(root / "evaluation.json")
        cohorts["W40_completed_" + suffix] = {
            "summary": aggregate(rows),
            "training_end": {k: stop.get(k) for k in ("actual_transitions", "updates", "phase_new_transitions", "phase_new_updates", "unupdated_fragment", "controller_artifact", "reason")},
            "evaluation": [{k: x.get(k) for k in ("level", "simulation_s", "reason", "stage_status", "success")} for x in ev["stages"]],
            "qp2_additional_episodes": [dict(episode=x["episode"], reason=x["reason"], simulation_s=x["simulation_s"], max_x_m=x["max_x_m"]) for x in rows if suffix == "a" and x["episode"] > 314],
        }
    for label, start, end in (("W34_r24", 15, 44), ("W35_r24", 45, 117)):
        rows = []
        for ep in range(start, end + 1):
            path = BASE / f"train-PANN_PPO_S5-s0-ep{ep:05d}.json"
            row = episode(read(path))
            row["episode"] = ep
            rows.append(row)
        cohorts[label] = {
            "summary": aggregate(rows),
            "confirmed_first_outbound_attempts": [x["attempt"] for x in rows if x["first_outbound_material_traversal_completed"]],
            "raw_roundtrip_attempts": [{k: x[k] for k in ("attempt", "raw_roundtrips", "first_outbound_material_traversal_completed", "lateral_exit_s")} for x in rows if x["raw_roundtrips"]],
            "metric_boundary": "R24 predates LegalRoute ledger. First outbound passage uses original stored 500 Hz material_traversal; missing bidirectional counts are not imputed. Raw roundtrips here have an invalid first outbound passage.",
        }
    original = read(HERE.parent / "W40-final-review/progress-audit.json")
    cohorts["W40_original_64_episode_snapshot"] = {
        "baseline": "c438f5463659ab7c8d3a7f5eced251639e14b2ef",
        "summary": original["combined"], "numeric_combined": original["numeric_combined"],
        "note": "Original review snapshot before A qp2 continuation; do not mistake it for the final W40 parent of W41.",
    }
    return cohorts


def selected_raw(raw_root, manifest):
    """Optional selected CPU original log/trajectory audit; never reads GPU files."""
    import numpy as np

    raw_root = Path(raw_root)
    catalog = {f["member"]: f for archive in manifest["archives"] for f in archive["files"]}
    coverage, episodes = {}, {}

    def register(path, size, sha, crc, rows=None):
        key = str(path.relative_to(raw_root))
        declared = catalog.get(key, {})
        coverage[key] = dict(bytes=size, sha256=sha,
                             crc32=f"{crc:08x}", jsonl_rows=rows,
                             manifest_source_size=declared.get("source_size"),
                             manifest_crc32=declared.get("zip_crc32"),
                             matches_manifest_size_and_crc=(size == declared.get("source_size") and f"{crc:08x}" == declared.get("zip_crc32")))

    def jsonl(path):
        sha, crc, size, n = hashlib.sha256(), 0, 0, 0
        with path.open("rb") as src:
            for line in src:
                sha.update(line)
                crc = zlib.crc32(line, crc)
                size += len(line)
                if line.strip():
                    n += 1
                    yield json.loads(line)
        register(path, size, sha.hexdigest(), crc, n)

    for subdir in ("w41-a/ep00340", "w41-a/evaluation-E30", "w41-b/ep00344", "w41-b/evaluation-E30"):
        folder = raw_root / subdir
        required = [folder / x for x in ("planner-calls.jsonl", "planner-records.jsonl", "trajectory.npz")]
        if not all(p.exists() for p in required):
            episodes[subdir] = dict(status="required_selected_files_not_all_available")
            continue
        calls = list(jsonl(required[0]))
        plans = list(jsonl(required[1]))
        candidates = [x for p in plans for x in p["candidates"]]
        raw = required[2].read_bytes()
        register(required[2], len(raw), hashlib.sha256(raw).hexdigest(), zlib.crc32(raw))
        del raw
        intervals = []
        with np.load(required[2], allow_pickle=False) as data:
            t, q, v, cmd = [data[k] for k in ("time", "qpos", "qvel", "command_velocity")]
            valid = abs(t - data["diagnostic_time"]) < 1e-8
            for lo, hi in ((2., 10.), (10., 15.), (15., 20.), (20., 30.), (30., 40.), (40., 60.)):
                ix = np.flatnonzero((t >= lo - 1e-8) & (t <= hi + 1e-8) & valid)
                if len(ix) < 2:
                    continue
                intervals.append(dict(requested_window_s=[lo, hi], observed_window_s=[float(t[ix[0]]), float(t[ix[-1]])],
                                      rows=len(ix), dx_m=float(q[ix[-1], 0] - q[ix[0], 0]),
                                      x_min_m=float(q[ix, 0].min()), x_max_m=float(q[ix, 0].max()),
                                      command_x_mean_m_s=float(cmd[ix, 0].mean()),
                                      actual_vx_mean_m_s=float(v[ix, 0].mean()),
                                      actual_vx_rms_m_s=float(np.sqrt((v[ix, 0] ** 2).mean())),
                                      nonfoot_contact_fraction=float((data["nonfoot_ground_contacts"][ix] > 0).mean()),
                                      all_feet_contact_fraction=float(np.all(data["contact"][ix] > 0, axis=1).mean()),
                                      requested_cap_mean_m_s=float(data["requested_speed_cap"][ix].mean())))
            trajectory = dict(rows=len(t), discarded_held_diagnostic_terminal_rows=int((~valid).sum()),
                              max_x_m=float(q[:, 0].max()), final_xyz_m=q[-1, :3].tolist(),
                              finite_state_and_command=all(np.isfinite(x).all() for x in (t, q, v, cmd)),
                              intervals=intervals)
        records_finite = finite(calls) and finite(plans)
        episodes[subdir] = {
            "scope": "Complete planner calls/accepted records and 500 Hz trajectory for this selected episode, not all 130 training episodes.",
            "calls": len(calls), "installed_plans": len(plans),
            "planner_return_reasons": dict(Counter(x["return_reason"] for x in calls)),
            "backward_attempts": dict(Counter(x["backward_attempts"] for x in calls)),
            "all_calls_policy_updated": all(x["policy_updated"] for x in calls),
            "all_calls_metric_valid": all(x["metric_valid"] for x in calls),
            "max_reused_policy_age_s": max(x["reused_policy_age_s"] for x in calls),
            "planner_wall_s": summary([x["wall_s"] for x in calls]),
            "all_exported_planner_numbers_finite": records_finite,
            "candidate_count": len(candidates),
            "candidate_failed_count": sum(bool(x["failed"]) for x in candidates),
            "candidate_projection_failed_count_sum": sum(x["projection_failed_count"] for x in candidates),
            "candidate_projection_count_sum": sum(x["projection_count"] for x in candidates),
            "candidate_actuator_restoration_count_sum": sum(x["actuator_restoration_count"] for x in candidates),
            "prediction_infeasible_count": sum(bool(x["prediction_infeasible"]) for x in plans),
            "accepted_trunk_violation_max_m": max(x["accepted_trunk_violation_m"] for x in plans),
            "nominal_trunk_violation_max_m": max(x["nominal_trunk_violation_m"] for x in plans),
            "acceptance_mode_counts": dict(Counter(x["acceptance_mode"] for x in plans)),
            "cost_ratio_reason_counts": dict(Counter(x["cost_ratio_reason"] for x in plans)),
            "regularization_used": summary([x["regularization_used"] for x in plans]),
            "plans_at_regularization_1e6": sum(x["regularization_used"] == 1e6 for x in plans),
            "installed_feedback_scaling_values": sorted({x["installed_feedback_scaling"] for x in plans}),
            "trajectory": trajectory,
        }
    allfiles = [f for a in manifest["archives"] for f in a["files"]]
    stderr = [f["member"] for f in allfiles if f["source_path"].endswith((".log", ".txt"))]
    return {
        "method": "Selected downloaded original CPU planner and trajectory files, verified against published ZIP-member size and CRC32; no additional physics.",
        "episodes": episodes,
        "stderr_log_assets_in_delivery_manifest": stderr,
        "soft_warning_boundary": "No delivered .log/.txt agent-server stderr assets are listed. Complete selected planner records have explicit failure counts; their success cannot establish zero unrecorded MuJoCo stderr/soft warnings.",
        "coverage": dict(files=len(coverage), bytes=sum(x["bytes"] for x in coverage.values()),
                         all_match_manifest=all(x["matches_manifest_size_and_crc"] for x in coverage.values()),
                         files_read=coverage),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, help="Optional downloaded w41-a/w41-b original member tree, read only.")
    args = parser.parse_args()
    streams = {suffix: build_stream(suffix) for suffix in JOBS}
    rows = [x for stream in streams.values() for x in stream["episodes"]]
    history = historical_results()
    combined = aggregate(rows)
    initial = min(s["training_end"]["invocation_started_unix_s"] for s in streams.values())
    end = max(s["training_end"]["stopped_unix_s"] for s in streams.values())
    manifest = read(BASE / "raw-bundles/w41-20261108/manifest.json")
    candidates = {
        "a": {"maximum_x": 380, "early_material_domain_exit": 340, "longest_survival": 365},
        "b": {"maximum_x_and_domain_exit": 335, "long_stagnation": 344, "deepest_fall": 337},
    }
    selected = []
    for suffix, roles in candidates.items():
        for role, ep in roles.items():
            row = next(x for x in streams[suffix]["episodes"] if x["episode"] == ep)
            raw = []
            for archive in manifest["archives"]:
                files = [f for f in archive["files"] if f.get("attempt") == row["attempt"]]
                if files:
                    raw.append(dict(archive=archive["name"], url=archive.get("url"), member_count=len(files),
                                    source_bytes=sum(f["source_size"] for f in files)))
            selected.append(dict(stream=suffix, role=role, episode=ep, attempt=row["attempt"],
                                 reason=row["reason"], simulation_s=row["simulation_s"],
                                 max_x_m=row["max_x_m"], first_load_s=row["first_load_s"],
                                 base_entry_s=row["base_entry_s"], archives=raw))
    out = {
        "schema": "W41_CPU_progress_audit_v1", "review_baseline": BASELINE,
        "git_head_when_recomputed": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "method": "All 130 W41 training episode results/resolved/episode/cap/event-manifest/compact-event records, all 4096 transition/value pairs, 16 update records, two final E30 results and all existing final-E30 event windows. No physics, actor forward pass, real checkpoint restoration, or GPU operation.",
        "streams": streams, "combined": combined,
        "cross_stream_runtime_identity_equal": streams["a"]["runtime_identity"] == streams["b"]["runtime_identity"],
        "cross_stream_fingerprint_note": "Different execution fingerprints include stream-specific execution-contract source path/hash. Resolved runtime/author sources, model.mjb, agent-server binary and workers are compared separately.",
        "failure_families": {k: aggregate([x for x in rows if x["reason_family"] == k]) for k in sorted({x["reason_family"] for x in rows})},
        "combined_numeric": {
            "phase_rows": sum(s["numeric"]["rows"] for s in streams.values()),
            "trained_this_phase": sum(len(s["updates"]) * 256 for s in streams.values()),
            "unupdated_tail": sum(s["training_counts"]["unupdated_fragment"] for s in streams.values()),
            "pann_current_nonzero_rows": sum(s["numeric"]["pann_current_nonzero_rows"] for s in streams.values()),
            "pann_current_nonzero_fraction": sum(s["numeric"]["pann_current_nonzero_rows"] for s in streams.values()) / 4096,
            "loaded_all_current_actor_features_zero_episodes": sum(x["material_loaded"] and x["current_pann_nonzero_rows"] == 0 for x in rows),
            "reward_sum": sum(s["numeric"]["reward_sum"] for s in streams.values()),
            "reward_formula_max_abs_error": max(s["numeric"]["reward_formula_check"]["max_absolute_error"] for s in streams.values()),
            "gae_mean_max_abs_difference": max(b["gae_arithmetic"]["absolute_mean_difference"] for s in streams.values() for b in s["updates"]),
        },
        "unique_ancestry": {
            "common_PANN_ancestor_samples": 6545, "W38_two_stream_new": 2048,
            "W39_two_stream_new": 2048, "W40_completed_two_stream_new": 2048,
            "W41_two_stream_new": 4096, "unique_PANN_tree_samples": 16785,
            "unique_trained_PANN_samples": 16640, "unique_excluded_historical_samples": 145,
            "separate_RAW_initial_samples": 256, "all_R12_learning_groups_unique_samples": 17041,
            "not_one_combined_policy": True,
        },
        "combined_training_throughput": {
            "wall_interval_s": end - initial, "phase_samples": 4096,
            "samples_per_shared_wall_hour": 4096 * 3600 / (end - initial),
            "basis": "Earliest CPU training invocation to latest CPU training stop. Includes overlapping CPU jobs and staggered starts; excludes final evaluations and publication."
        },
        "history": history, "targeted_raw_candidates": selected,
        "raw_delivery_metadata_only": {
            k: manifest.get(k) for k in ("status", "archive_count", "source_files", "source_bytes", "archive_bytes", "verified_archives", "verification")
        },
        "selected_raw_audit": selected_raw(args.raw_root, manifest) if args.raw_root else None,
        "limitations": [
            "The current r30 controller and continued PPO policies differ from earlier phases; no causal attribution from cross-phase proportions alone.",
            "E30 has one final deterministic attempt per branch, not repeated-seed qualification. E60 did not execute.",
            "No full legal first outbound crossing occurred. Longer survival or a full 60-second session is not near mission completion.",
            "Zero terminal execution_error does not certify absence of internal solver warnings. Full planner-call/soft-warning logs are outside this compact audit.",
            "Event first feedback saturation is before projection. Physical force clamp, serialization errors, and later hold substeps are separate quantities.",
            "Reward and GAE were recomputed from exported scalars; no network re-forward or optimizer replay was done.",
            "PANN feature nonzero counts concern actor-current observations, not terminal successors or all physical contact events. Query exposure causes require raw query records.",
            "Actual r24 first-outbound successes are retained; r24 lacked the new bidirectional ledger and did not establish a valid full legal roundtrip.",
            "Raw ZIP publication metadata is inspected, but this script does not download or decode every ZIP member. GPU files and processes are untouched.",
        ],
    }
    out["coverage"] = dict(files=len(READ), bytes=sum(v["bytes"] for v in READ.values()), read_files=dict(sorted(READ.items())))
    assert finite(out), "Nonfinite derived report value"
    path = HERE / "cpu-progress.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(path.relative_to(REPO)), "episodes": combined["episodes"],
                      "reasons": combined["reason_counts"], "rows": out["combined_numeric"]["phase_rows"],
                      "coverage_files": len(READ), "coverage_bytes": out["coverage"]["bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
