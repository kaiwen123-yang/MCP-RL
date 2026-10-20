"""Analyze saved Go2 adjoint-pilot JSON only; never import or run a simulator.

Usage:
  python scripts/analyze_go2_adjoint_pilot.py --runs work/downloaded-run --output notes/pilot-results
  # Add --execution-dir <launcher-folder> to reconcile events and actual elapsed time.
  # Use --invalid-comparison-reason "..." or runs/comparison_validity.json to invalidate inference.

Missing evaluation cells remain NOT_RUN with null success. Common initialization
cost is counted once per seed in the project ledger and separately allocated to
each method only in the explicitly labelled standalone-method view. Pickled
checkpoints and transitions are neither loaded nor executed.
Invalid comparisons retain descriptive outcomes and all recorded costs, but are
explicitly unsuitable for method rankings or paired scientific inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
COLLECTION_FIELDS = (
    "policy_steps", "physics_steps", "warmup_physics_steps",
    "warmup_and_execution_physics_steps", "planner_calls",
    "planner_seconds", "planner_model_steps", "planner_value_calls",
    "actor_calls", "collection_seconds",
)
COUNT_FIELDS = (
    "policy_steps", "physics_steps", "warmup_physics_steps",
    "warmup_and_execution_physics_steps", "actor_updates", "value_updates",
    "action_model_updates", "command_model_updates",
    "adjoint_forward_steps", "adjoint_reverse_steps",
)
EPISODE_FIELDS = (
    "seed", "arm", "round", "case", "episode", "status", "success", "failure_type",
    "elapsed_seconds", "completion_seconds", "real_return",
    "mechanical_work_proxy_J", "torque_squared_integral", "path_rms_m",
    "policy_steps", "source",
)
COLLECTION_EPISODE_FIELDS = (
    "seed", "arm", "round", "phase", "behavior_mode", "case", "episode",
    "complete", "success", "reason", "elapsed_seconds", "completion_seconds",
    "real_return", "mechanical_work_proxy_J", "torque_squared_integral",
    "path_rms_m", "policy_steps", "source",
)


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def total(values):
    known = [number(x) for x in values if number(x) is not None]
    return sum(known) if known else None


def mean(values):
    known = [number(x) for x in values if number(x) is not None]
    return statistics.mean(known) if known else None


def sd(values):
    known = [number(x) for x in values if number(x) is not None]
    return statistics.stdev(known) if len(known) >= 2 else None


def subtract(left, right):
    return left - right if number(left) is not None and number(right) is not None else None


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_csv(path, rows, fields=()):
    columns = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def comparison_validity(evidence, explicit_reason=None):
    """Read only validity fields, preserving invalidation independently of execution."""
    path = evidence.root / "comparison_validity.json"
    marker = evidence.read("comparison_validity.json")
    if path.is_file() and marker is None:
        raise ValueError("comparison_validity.json exists but is unreadable; do not assume a valid comparison.")
    reasons, sources = [], []
    if explicit_reason is not None:
        if not explicit_reason.strip():
            raise ValueError("--invalid-comparison-reason must not be empty.")
        reasons.append(explicit_reason.strip())
        sources.append("command-line invalidation")
    marker_invalid = marker is not None and (
        marker.get("comparison_valid") is False or marker.get("status") == "INVALID_COMPARISON")
    if marker_invalid:
        sources.append("runs/comparison_validity.json")
        for key in ("reason", "invalid_comparison_reason"):
            if isinstance(marker.get(key), str) and marker[key].strip():
                reasons.append(marker[key].strip())
        reasons.extend(value.strip() for value in marker.get("reasons", [])
                       if isinstance(value, str) and value.strip())
        if not reasons:
            reasons.append("The saved run is explicitly marked INVALID_COMPARISON.")
    valid = not explicit_reason and not marker_invalid
    return {
        "comparison_valid": bool(valid),
        "status": "NOT_INVALIDATED" if valid else "INVALID_COMPARISON",
        "reasons": list(dict.fromkeys(reasons)),
        "sources": sources,
        "interpretation": (
            "No invalidation is recorded; this flag alone does not certify optimizer independence "
            "or replace the protocol/evidence audit."
            if valid else
            "Descriptive execution facts and costs only. No fair method ranking, "
            "learning-advantage claim, or paired causal/statistical inference is supported."),
    }


def all_known_matches(values):
    values = list(values)
    if any(value is False for value in values):
        return False
    return True if values and all(value is True for value in values) else None


def network_hash_match(left, right):
    names = ("actor", "dynamics", "response", "value")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    if any(not isinstance(item.get(name), str) or not item[name]
           for item in (left, right) for name in names):
        return None
    return all(left[name] == right[name] for name in names)


def optimizer_steps_match(item, expected):
    if not isinstance(item, dict):
        return None
    states = number(item.get("initialized_parameter_states"))
    if states is None:
        return None
    if expected == 0:
        return (states == 0 and item.get("state_tensor_count") == 0
                and item.get("step_min") is None and item.get("step_max") is None
                and item.get("step_histogram") == {})
    minimum, maximum = number(item.get("step_min")), number(item.get("step_max"))
    histogram = item.get("step_histogram")
    if minimum is None or maximum is None or not isinstance(histogram, dict):
        return None
    return (states > 0 and minimum == maximum == expected
            and histogram == {str(expected): states})


def audit_initializations(evidence, config):
    """Compare saved startup JSON; no tensor/checkpoint loading."""
    proof = evidence.read("initialization-checkpoint-proof.json")
    proof_networks = (proof or {}).get("common_network_hashes", {})
    expected_steps = {
        "actor_optimizer": 0,
        "value_optimizer": (
            int(config["ppo"]["epochs"])
            * math.ceil(len(config["cases"]) * int(config["common_warmup_steps_per_case"])
                        / int(config["ppo"]["minibatch_size"]))),
        "f_optimizer": int(config["model_fit_steps"]),
        "g_optimizer": int(config["model_fit_steps"]),
    }
    seed_rows, branch_rows = [], []
    for seed in config["seeds"]:
        base = f"seed{seed}/common"
        common_optimizer = evidence.read(f"{base}/initialization_optimizer_state.json")
        common_network = evidence.read(f"{base}/initialization_hashes.json")
        network_source = f"{base}/initialization_hashes.json" if common_network else None
        if common_network is None and isinstance(proof_networks, dict):
            common_network = proof_networks.get(str(seed))
            if isinstance(common_network, dict):
                network_source = "initialization-checkpoint-proof.json"
        networks, optimizers = {}, {}
        for arm in config["arms"]:
            networks[arm] = evidence.read(f"seed{seed}/{arm}/initialization_hashes.json")
            optimizers[arm] = evidence.read(f"seed{seed}/{arm}/initialization_optimizer_state.json")
        reference_arm = next((arm for arm in config["arms"] if networks[arm] is not None), None)
        reference_network = networks.get(reference_arm)
        common_step_checks = {
            name: optimizer_steps_match((common_optimizer or {}).get(name), step)
            for name, step in expected_steps.items()}
        seed_rows.append({
            "seed": seed,
            "common_optimizer_file_present": common_optimizer is not None,
            "common_network_hashes_present": common_network is not None,
            "common_network_hash_source": network_source,
            "common_optimizer_summaries": common_optimizer,
            "common_expected_step_checks": common_step_checks,
            "common_expected_steps_match": all_known_matches(common_step_checks.values()),
            "first_available_network_reference_arm": reference_arm,
        })
        for arm in config["arms"]:
            payload = optimizers[arm]
            actual = payload.get("optimizers") if isinstance(payload, dict) else None
            loaded = actual if isinstance(actual, dict) else {}
            row = {
                "seed": seed, "arm": arm,
                "network_hash_file_present": networks[arm] is not None,
                "optimizer_state_file_present": payload is not None,
                "common_optimizer_file_present": common_optimizer is not None,
                "common_network_hashes_present": common_network is not None,
                "common_network_hash_source": network_source,
                "network_hashes_match_common": network_hash_match(networks[arm], common_network),
                "network_hashes_match_first_available_arm": network_hash_match(
                    networks[arm], reference_network),
                "network_reference_arm": reference_arm,
                "reported_optimizer_matches_common": (
                    payload.get("matches_common") if isinstance(payload, dict) else None),
            }
            hash_checks, summary_checks, step_checks = [], [], []
            for name, expected in expected_steps.items():
                left, right = loaded.get(name), (common_optimizer or {}).get(name)
                actual_hash = left.get("sha256") if isinstance(left, dict) else None
                common_hash = right.get("sha256") if isinstance(right, dict) else None
                hash_match = (actual_hash == common_hash if actual_hash and common_hash else None)
                summary_match = (left == right if isinstance(left, dict)
                                 and isinstance(right, dict) else None)
                step_match = optimizer_steps_match(left, expected)
                hash_checks.append(hash_match)
                summary_checks.append(summary_match)
                step_checks.append(step_match)
                row.update({
                    f"{name}_sha256": actual_hash,
                    f"{name}_matches_common_hash": hash_match,
                    f"{name}_matches_common_summary": summary_match,
                    f"{name}_expected_step": expected,
                    f"{name}_initialized_parameter_states": (
                        left.get("initialized_parameter_states") if isinstance(left, dict) else None),
                    f"{name}_step_min": left.get("step_min") if isinstance(left, dict) else None,
                    f"{name}_step_max": left.get("step_max") if isinstance(left, dict) else None,
                    f"{name}_expected_step_match": step_match,
                })
            row["actor_optimizer_empty"] = optimizer_steps_match(
                loaded.get("actor_optimizer"), 0)
            row["optimizer_hashes_match_common"] = all_known_matches(hash_checks)
            row["optimizer_summaries_match_common"] = all_known_matches(summary_checks)
            row["all_expected_warmup_steps_match"] = all_known_matches(step_checks)
            reported = row["reported_optimizer_matches_common"]
            calculated = row["optimizer_summaries_match_common"]
            row["reported_matches_flag_agrees"] = (
                reported == calculated if isinstance(reported, bool)
                and isinstance(calculated, bool) else None)
            row["network_and_optimizer_startup_matches_common"] = all_known_matches((
                row["network_hashes_match_common"], row["optimizer_hashes_match_common"],
                row["optimizer_summaries_match_common"], row["all_expected_warmup_steps_match"],
                row["reported_optimizer_matches_common"],
                row["reported_matches_flag_agrees"]))
            branch_rows.append(row)
    expected_count = len(config["seeds"]) * len(config["arms"])
    full_checks = [row["network_and_optimizer_startup_matches_common"] for row in branch_rows]
    any_mismatch = any(
        row[key] is False for row in branch_rows for key in
        ("network_hashes_match_common", "network_hashes_match_first_available_arm",
         "optimizer_hashes_match_common", "optimizer_summaries_match_common",
         "all_expected_warmup_steps_match", "reported_matches_flag_agrees"))
    count_present = sum(row["network_hash_file_present"] and row["optimizer_state_file_present"]
                        for row in branch_rows)
    status = ("STARTUP_MISMATCH" if any_mismatch else
              "ALL_DECLARED_STARTUP_RECORDS_MATCH" if all_known_matches(full_checks) is True
              and count_present == expected_count else
              "PARTIAL_STARTUP_EVIDENCE" if count_present else "MISSING_OPTIMIZER_STARTUP_EVIDENCE")
    return {
        "status": status,
        "expected_branch_count": expected_count,
        "branch_network_files_present": sum(row["network_hash_file_present"] for row in branch_rows),
        "branch_optimizer_files_present": sum(row["optimizer_state_file_present"] for row in branch_rows),
        "branches_with_both_files": count_present,
        "all_declared_branches_have_both_files": count_present == expected_count,
        "branches_optimizer_hash_match_common": sum(
            row["optimizer_hashes_match_common"] is True for row in branch_rows),
        "branches_network_hash_match_common": sum(
            row["network_hashes_match_common"] is True for row in branch_rows),
        "branches_with_all_expected_warmup_steps": sum(
            row["all_expected_warmup_steps_match"] is True for row in branch_rows),
        "all_declared_startup_records_match_common": all_known_matches(full_checks),
        "expected_optimizer_steps": expected_steps,
        "expected_steps_basis": "Fixed common B sample count, PPO epochs/minibatches and common model_fit_steps.",
        "common_by_seed": seed_rows,
        "branches": branch_rows,
        "interpretation": [
            "Missing files/comparisons remain null and cannot pass the nine-branch startup check.",
            "Network hashes may be compared with a separate common checkpoint proof JSON; this analyzer never opens checkpoints.",
            "First-available-branch network equality is reported separately and never substitutes for a missing common network fingerprint.",
            "Startup hashes compare saved values; alone they do not prove absence of later tensor aliasing.",
            "Old invalidated runs remain invalid even when some available network hashes match.",
        ],
        "checkpoint_or_pickle_loading": False,
    }, branch_rows


class Evidence:
    def __init__(self, root):
        self.root = root
        self.files = {}
        self.issues = []

    def read(self, relative):
        path = self.root / relative
        if not path.is_file():
            return None
        raw = path.read_bytes()
        self.files[Path(relative).as_posix()] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        }
        try:
            value = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError):
            self.issues.append({"source": Path(relative).as_posix(), "issue": "UNREADABLE_JSON"})
            return None
        if not isinstance(value, dict):
            self.issues.append({"source": Path(relative).as_posix(), "issue": "EXPECTED_JSON_OBJECT"})
            return None
        return value


class ExecutionSegment:
    """Whitelist launcher metadata and recover completed phases from its events."""

    def __init__(self, directory):
        self.evidence = Evidence(directory) if directory is not None else None
        self.ends = {}
        self.starts = Counter()
        self.start_elapsed = {}
        self.comparisons = []
        self.issues = []
        self.public = {}
        self.anchor_timestamp = None
        if directory is None:
            return
        runtime = self.evidence.read("runtime.json")
        exited = self.evidence.read("exit.json")
        failed = self.evidence.read("execution_failure.json")
        if runtime:
            self.public["runtime_identity"] = {
                key: runtime[key] for key in
                ("algorithm_commit", "configuration_sha256", "launcher_sha256")
                if key in runtime}
            self.public["package_versions"] = {
                key: runtime.get("runtime", {}).get(key)
                for key in ("torch", "numpy", "scipy", "mujoco", "PyYAML")}
        if exited:
            self.public["experiment_exit"] = {
                key: exited.get(key) for key in
                ("status", "wall_seconds", "cpu_user_seconds", "cpu_system_seconds", "peak_rss_kib",
                 "exit_code", "signal", "signal_number")}
        if failed:
            self.public["execution_failure"] = {
                key: failed.get(key) for key in ("status", "exception_type")}
        path = directory / "execution-events.jsonl"
        if not path.is_file():
            return
        raw = path.read_bytes()
        self.evidence.files["execution-events.jsonl"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        }
        allowed_path = re.compile(
            r"seed\d+/(?:common/(?:behavior|calibration)|"
            r"[a-z_]+/(?:evaluation_round0|round\d+/(?:behavior|calibration|evaluation)))")
        for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.issues.append({"source": f"execution-events.jsonl:L{line_number}",
                                    "issue": "UNREADABLE_EVENT_LINE"})
                continue
            if isinstance(event, dict) and self.anchor_timestamp is None:
                elapsed = number(event.get("elapsed_seconds"))
                try:
                    utc = datetime.datetime.fromisoformat(event.get("utc", "").replace("Z", "+00:00"))
                    if elapsed is not None and utc.tzinfo is not None:
                        self.anchor_timestamp = utc.timestamp() - elapsed
                except (TypeError, ValueError):
                    pass
            if not isinstance(event, dict) or event.get("event") not in ("phase_start", "phase_end"):
                continue
            phase_path, operation = event.get("path", ""), event.get("operation")
            if not isinstance(phase_path, str) or not allowed_path.fullmatch(phase_path):
                self.issues.append({"source": f"execution-events.jsonl:L{line_number}",
                                    "issue": "UNRECOGNIZED_PHASE_PATH_OMITTED"})
                continue
            if operation not in ("collect", "ppo", "fit_facts", "adjoint"):
                continue
            key = phase_path, operation
            if event["event"] == "phase_start":
                self.starts[key] += 1
                self.start_elapsed.setdefault(key, number(event.get("elapsed_seconds")))
            else:
                if key in self.ends:
                    self.issues.append({"source": f"execution-events.jsonl:L{line_number}",
                                        "issue": "DUPLICATE_PHASE_END", "path": phase_path,
                                        "operation": operation})
                    continue
                self.ends[key] = {
                    "stats": event.get("stats") if isinstance(event.get("stats"), dict) else {},
                    "wall_seconds": number(event.get("wall_seconds")),
                    "elapsed_seconds": number(event.get("elapsed_seconds")),
                    "line": line_number,
                }
        self.public["completed_instrumented_phases"] = len(self.ends)
        self.public["started_without_phase_end"] = [
            {"path": path, "operation": operation}
            for path, operation in self.starts if (path, operation) not in self.ends]
        self.public["sum_completed_event_phase_wall_seconds"] = total(
            event["wall_seconds"] for event in self.ends.values())

    def has_branch(self, base):
        prefix = base + "/"
        return any(path.startswith(prefix) for path, _ in (*self.starts, *self.ends))

    def interval(self, start_path, start_operation, end_path, end_operation):
        beginning = self.start_elapsed.get((start_path, start_operation))
        ending = self.ends.get((end_path, end_operation), {}).get("elapsed_seconds")
        return subtract(ending, beginning)

    def row(self, seed, arm, round_number, phase, kind, payload, source,
            event_path, operation):
        event = self.ends.get((event_path, operation))
        event_source = (f"{event.get('source_prefix', 'execution/')}execution-events.jsonl:L{event['line']}"
                        if event else None)
        if event is not None and payload is not None:
            fields = set(COUNT_FIELDS) | set(COLLECTION_FIELDS) | {
                "samples", "micro_samples", "macro_samples", "ppo_seconds",
                "fitting_seconds", "action_model_fitting_seconds",
                "command_model_fitting_seconds", "adjoint_seconds",
            }
            for field in sorted(fields & set(payload) & set(event["stats"])):
                runner_value, event_value = number(payload[field]), number(event["stats"][field])
                if runner_value is None or event_value is None:
                    continue
                matched = math.isclose(runner_value, event_value, rel_tol=1e-10, abs_tol=1e-10)
                self.comparisons.append({
                    "path": event_path, "operation": operation, "field": field,
                    "runner_value": runner_value, "event_value": event_value,
                    "matched": matched, "runner_source": source, "event_source": event_source,
                })
                if not matched:
                    self.issues.append({"issue": "RUNNER_EVENT_STATS_MISMATCH",
                                        "path": event_path, "operation": operation, "field": field})
        effective = payload if payload is not None else (event["stats"] if event else None)
        effective_source = source if payload is not None else event_source
        row = cost_row(seed, arm, round_number, phase, kind, effective, effective_source)
        row.update({
            "evidence_origin": ("runner_and_event" if payload is not None and event else
                                "runner" if payload is not None else
                                "event_only" if event else "not_recorded"),
            "instrumented_phase_wall_seconds": event["wall_seconds"] if event else None,
            "instrumented_phase_end_elapsed_seconds": event["elapsed_seconds"] if event else None,
            "instrumented_event_source": event_source,
        })
        row["effective_recorded_phase_wall_seconds"] = (
            row["instrumented_phase_wall_seconds"]
            if row["instrumented_phase_wall_seconds"] is not None else row["accounted_wall_seconds"])
        return row


class ExecutionEvidence(ExecutionSegment):
    """Merge chronological process segments without charging downtime as work."""

    def __init__(self, directories):
        super().__init__(None)
        self.segment_count = 0
        self.start_segment = {}
        if directories is None:
            return
        directories = list(directories) if isinstance(directories, (list, tuple)) else [directories]
        if not directories:
            return
        parts = [ExecutionSegment(Path(directory)) for directory in directories]
        self.segment_count = len(parts)
        self.evidence = Evidence(Path(directories[0]))
        offset = 0.0
        segment_rows = []
        downtime = []
        previous_end = None
        for index, part in enumerate(parts):
            source_prefix = f"execution/segment{index + 1}/" if len(parts) > 1 else "execution/"
            self.evidence.files.update({
                (f"segment{index + 1}/{name}" if len(parts) > 1 else name): hashes
                for name, hashes in part.evidence.files.items()})
            self.evidence.issues.extend(
                {**issue, "segment": index + 1} for issue in part.evidence.issues)
            self.issues.extend({**issue, "segment": index + 1} for issue in part.issues)
            for key, count in part.starts.items():
                self.starts[key] += count
                if key not in self.start_elapsed:
                    local = part.start_elapsed.get(key)
                    self.start_elapsed[key] = local + offset if local is not None and offset is not None else None
                    self.start_segment[key] = index
            for key, event in part.ends.items():
                if key in self.ends:
                    self.issues.append({"issue": "DUPLICATE_PHASE_END_ACROSS_SEGMENTS",
                                        "path": key[0], "operation": key[1], "segment": index + 1})
                    continue
                local = event["elapsed_seconds"]
                self.ends[key] = {
                    **event, "source_prefix": source_prefix, "segment_index": index,
                    "elapsed_seconds": local + offset if local is not None and offset is not None else None,
                }
            exited = part.public.get("experiment_exit", {})
            duration = number(exited.get("wall_seconds"))
            gap = subtract(part.anchor_timestamp, previous_end) if index else None
            if index:
                downtime.append({"before_segment": index + 1, "seconds": gap,
                                 "basis": "UTC clock anchors from event utc minus segment elapsed time"})
                if gap is not None and gap < 0:
                    self.issues.append({"issue": "OVERLAPPING_SEGMENT_CLOCK_INTERVALS",
                                        "segment": index + 1, "gap_seconds": gap})
            previous_end = (
                part.anchor_timestamp + duration
                if part.anchor_timestamp is not None and duration is not None else None)
            segment_rows.append({
                "segment": index + 1,
                "runtime_identity": part.public.get("runtime_identity"),
                "package_versions": part.public.get("package_versions"),
                "exit": exited or None,
                "execution_failure": part.public.get("execution_failure"),
                "completed_instrumented_phases": len(part.ends),
                "started_without_phase_end": part.public.get("started_without_phase_end", []),
                "active_elapsed_offset_seconds": offset,
            })
            offset = offset + duration if offset is not None and duration is not None else None
        identities = [p.public.get("runtime_identity", {}) for p in parts]
        identity_fields = ("algorithm_commit", "configuration_sha256")
        consistent_identity = all(
            len({identity[field] for identity in identities if field in identity}) <= 1
            for field in identity_fields)
        if not consistent_identity:
            self.issues.append({"issue": "EXECUTION_SEGMENT_ALGORITHM_OR_CONFIG_MISMATCH"})
        final_exit = parts[-1].public.get("experiment_exit", {})
        all_exits = [p.public.get("experiment_exit", {}) for p in parts]
        all_exit_times = all(number(e.get("wall_seconds")) is not None for e in all_exits)
        cumulative_exit = None
        if all_exit_times:
            cumulative_exit = {
                "status": final_exit.get("status"),
                "wall_seconds": total(e["wall_seconds"] for e in all_exits),
                "wall_time_basis": "sum of active process segments; downtime excluded",
                "cpu_user_seconds": total(e.get("cpu_user_seconds") for e in all_exits),
                "cpu_system_seconds": total(e.get("cpu_system_seconds") for e in all_exits),
                "peak_rss_kib": max(
                    (number(e.get("peak_rss_kib")) for e in all_exits if number(e.get("peak_rss_kib")) is not None),
                    default=None),
            }
        incidents = []
        for row in segment_rows:
            failed = row["execution_failure"]
            exit_status = (row["exit"] or {}).get("status")
            if (exit_status is not None and exit_status != "COMPLETED") or failed:
                kind = (failed or {}).get("exception_type")
                incidents.append({
                    "segment": row["segment"], "status": exit_status or "FAILED",
                    "exception_type": kind,
                    "classification": ("transport_output_incident" if kind == "BrokenPipeError"
                                       else "execution_interruption" if kind == "KeyboardInterrupt"
                                       or exit_status in ("INTERRUPTED", "STOPPED", "CANCELLED")
                                       else "execution_incident"),
                })
        unresolved_phases = [
            {"segment": row["segment"], **phase}
            for row in segment_rows for phase in row["started_without_phase_end"]]
        self.public = {
            "segment_count": len(parts), "segments": segment_rows,
            "runtime_identity": parts[0].public.get("runtime_identity"),
            "package_versions": parts[0].public.get("package_versions"),
            "algorithm_and_config_consistent_across_segments": consistent_identity,
            "completed_instrumented_phases": len(self.ends),
            "started_without_phase_end": [
                {"path": path, "operation": operation}
                for path, operation in self.starts if (path, operation) not in self.ends],
            "segment_phases_without_end": unresolved_phases,
            "unrecorded_interrupted_physics_steps": (
                None if any(p["operation"] == "collect" for p in unresolved_phases) else 0),
            "incomplete_work_interpretation": (
                "An unmatched phase start is not assigned an invented sample/update count. "
                "A later completed phase does not recover unrecorded work from an earlier process segment."),
            "sum_completed_event_phase_wall_seconds": total(e["wall_seconds"] for e in self.ends.values()),
            "total_active_wall_seconds": cumulative_exit["wall_seconds"] if cumulative_exit else None,
            "known_completed_segment_active_wall_seconds": total(e.get("wall_seconds") for e in all_exits),
            "all_segment_exit_times_available": all_exit_times,
            "downtime_intervals": downtime,
            "total_downtime_seconds": (sum(d["seconds"] for d in downtime)
                                       if all(d["seconds"] is not None and d["seconds"] >= 0 for d in downtime)
                                       else None),
            "incidents": incidents,
        }
        if cumulative_exit is not None:
            self.public["experiment_exit"] = cumulative_exit
        final_failure = parts[-1].public.get("execution_failure")
        if final_failure:
            self.public["execution_failure"] = final_failure

    def spans_segments(self, start_path, start_operation, end_path, end_operation):
        start = self.start_segment.get((start_path, start_operation))
        end = self.ends.get((end_path, end_operation), {}).get("segment_index")
        return start is not None and end is not None and start != end


def evaluation_rows(seed, arm, round_number, config, collection, source):
    episodes = collection.get("episodes", []) if collection else []
    rows = []
    repetitions = int(config["evaluation_complete_episodes_per_case"])
    for case in config["cases"]:
        observed = [ep for ep in episodes if ep.get("case") == case]
        # Preserve unexpected extra episodes rather than silently selecting one.
        for slot in range(max(repetitions, len(observed))):
            base = dict(seed=seed, arm=arm, round=round_number, case=case, episode=slot)
            if slot >= len(observed):
                rows.append({**base, "status": "NOT_RUN", "success": None,
                             "failure_type": None,
                             **{k: None for k in EPISODE_FIELDS if k not in base
                                and k not in ("status", "success", "failure_type", "source")},
                             "source": source if collection else None})
                continue
            ep = observed[slot]
            complete = ep.get("complete") is True
            success = bool(ep.get("success")) if complete else None
            rows.append({
                **base,
                "status": "COMPLETED" if complete else "INCOMPLETE_EPISODE",
                "success": success,
                "failure_type": ep.get("reason") if success is not True else None,
                "elapsed_seconds": number(ep.get("elapsed_seconds")),
                "completion_seconds": number(ep.get("completion_seconds")) if success else None,
                "real_return": number(ep.get("real_return")),
                "mechanical_work_proxy_J": number(ep.get("mechanical_work_proxy_J")),
                "torque_squared_integral": number(ep.get("torque_squared_integral")),
                "path_rms_m": number(ep.get("path_rms_m")),
                "policy_steps": number(ep.get("policy_steps")),
                "source": source,
            })
    return rows


def saved_collection_episode_rows(seed, arm, round_number, phase, collection, source):
    """Include actual saved episodes only, never slots inferred from the protocol."""
    if collection is None or source is None:
        return []
    rows = []
    for episode in collection.get("episodes", []):
        complete = episode.get("complete") is True
        reason = episode.get("reason")
        success = (bool(episode.get("success"))
                   if complete and reason != "sampling_cut" else None)
        rows.append({
            "seed": seed, "arm": arm, "round": round_number, "phase": phase,
            "behavior_mode": collection.get("behavior"),
            "case": episode.get("case"), "episode": episode.get("episode"),
            "complete": complete, "success": success, "reason": reason,
            "elapsed_seconds": number(episode.get("elapsed_seconds")),
            "completion_seconds": (
                number(episode.get("completion_seconds")) if success is True else None),
            "real_return": number(episode.get("real_return")),
            "mechanical_work_proxy_J": number(episode.get("mechanical_work_proxy_J")),
            "torque_squared_integral": number(episode.get("torque_squared_integral")),
            "path_rms_m": number(episode.get("path_rms_m")),
            "policy_steps": number(episode.get("policy_steps")),
            "source": source,
        })
    return rows


def summarize_collection_episodes(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["phase"], row["arm"], row["behavior_mode"]), []).append(row)
    result = []
    for (phase, arm, mode), group in sorted(groups.items(), key=lambda item: str(item[0])):
        successes = [row for row in group if row["success"] is True]
        failures = [row for row in group if row["success"] is False]
        completed_outcomes = len(successes) + len(failures)
        reasons = Counter(row["reason"] or "unspecified" for row in group)
        result.append({
            "phase": phase, "arm": arm, "behavior_mode": mode,
            "phase_identity": (
                "stochastic behavior B, not an independent test set" if phase == "behavior"
                else "deterministic mean calibration C, not the held-out evaluation"
                if phase == "calibration" else
                "declared deterministic complete-task evaluation"),
            "recorded_episode_rows": len(group),
            "saved_collection_summary_count": len({row["source"] for row in group}),
            "seeds_present": json.dumps(sorted({row["seed"] for row in group})),
            "complete_episode_rows": sum(row["complete"] for row in group),
            "successful_complete_episodes": len(successes),
            "failed_complete_episodes": len(failures),
            "sampling_cut_episodes": reasons.get("sampling_cut", 0),
            "other_incomplete_or_unknown_outcomes": sum(
                row["success"] is None and row["reason"] != "sampling_cut" for row in group),
            "success_fraction_observed_completed_only": (
                len(successes) / completed_outcomes if completed_outcomes else None),
            "successful_completion_mean_seconds": mean(
                row["completion_seconds"] for row in successes),
            "observed_elapsed_sum_seconds_including_cuts": total(
                row["elapsed_seconds"] for row in group),
            "observed_return_sum_including_cuts": total(row["real_return"] for row in group),
            "completed_return_mean": mean(
                row["real_return"] for row in (*successes, *failures)),
            "observed_work_sum_J_including_cuts": total(
                row["mechanical_work_proxy_J"] for row in group),
            "observed_torque_squared_sum_including_cuts": total(
                row["torque_squared_integral"] for row in group),
            "observed_policy_steps_including_cuts": total(row["policy_steps"] for row in group),
            **{f"reason_{reason}": reasons.get(reason, 0)
               for reason in ("success", "collision", "fall", "deadline", "sampling_cut", "unspecified")},
            "reason_counts": json.dumps(dict(reasons), ensure_ascii=False, sort_keys=True),
            "independent_test_sample_count": None,
            "interpretation": "Descriptive episode inventory; episodes across training rounds/cases are not independent test replicates.",
        })
    return result


def aggregate_evaluation(rows):
    completed = [r for r in rows if r["status"] == "COMPLETED"]
    successes = [r for r in completed if r["success"]]
    recorded = [r for r in rows if r["status"] != "NOT_RUN"]
    failure_counts = Counter(r["failure_type"] or "unspecified" for r in completed if not r["success"])
    return {
        "status": ("NOT_RUN" if not recorded else
                   "COMPLETE_EVALUATION" if len(completed) == len(rows) else "PARTIAL_EVALUATION"),
        "expected_evaluation_cells": len(rows),
        "recorded_evaluation_cells": len(recorded),
        "completed_evaluation_cells": len(completed),
        "not_run_cells": sum(r["status"] == "NOT_RUN" for r in rows),
        "incomplete_episode_cells": sum(r["status"] == "INCOMPLETE_EPISODE" for r in rows),
        "successes_observed": len(successes) if completed else None,
        "failures_observed": len(completed) - len(successes) if completed else None,
        "success_fraction_observed_complete_only": len(successes) / len(completed) if completed else None,
        "success_time_mean_seconds": mean(r["completion_seconds"] for r in successes),
        "success_work_mean_J": mean(r["mechanical_work_proxy_J"] for r in successes),
        "all_completed_work_mean_J": mean(r["mechanical_work_proxy_J"] for r in completed),
        "all_completed_torque2_mean": mean(r["torque_squared_integral"] for r in completed),
        "all_completed_path_rms_mean_m": mean(r["path_rms_m"] for r in completed),
        "all_completed_real_return_mean": mean(r["real_return"] for r in completed),
        "observed_elapsed_sum_seconds": total(r["elapsed_seconds"] for r in recorded),
        "observed_work_sum_J_including_incomplete": total(r["mechanical_work_proxy_J"] for r in recorded),
        **{f"failure_{reason}": failure_counts.get(reason, 0) if completed else None
           for reason in ("collision", "fall", "deadline", "unspecified")},
    }


def cost_row(seed, arm, round_number, phase, kind, payload, source):
    row = {"seed": seed, "arm": arm, "round": round_number, "phase": phase,
           "kind": kind, "shared_once_per_seed": arm == "common",
           "status": "RECORDED" if payload is not None else "NOT_RECORDED",
           "source": source if payload is not None else None}
    row.update({key: None if payload is None else 0 for key in COUNT_FIELDS})
    row.update({"accounted_wall_seconds": None,
                "planner_seconds_subset": None,
                "action_model_fitting_seconds_subset": None,
                "command_model_fitting_seconds_subset": None})
    if payload is None:
        return row
    if kind == "collection":
        row.update({key: number(payload.get(key)) for key in COLLECTION_FIELDS})
        # Older logs can still identify the total from their two explicit parts.
        if row["warmup_and_execution_physics_steps"] is None:
            if row["physics_steps"] is not None and row["warmup_physics_steps"] is not None:
                row["warmup_and_execution_physics_steps"] = row["physics_steps"] + row["warmup_physics_steps"]
        row["accounted_wall_seconds"] = number(payload.get("collection_seconds"))
        row["planner_seconds_subset"] = number(payload.get("planner_seconds"))
    elif kind == "ppo":
        row.update({key: number(payload.get(key)) for key in ("actor_updates", "value_updates")})
        row["accounted_wall_seconds"] = number(payload.get("ppo_seconds"))
        row["ppo_samples"] = number(payload.get("samples"))
        row["kl_early_stop"] = payload.get("kl_early_stop")
    elif kind == "model_fit":
        row.update({key: number(payload.get(key)) for key in ("action_model_updates", "command_model_updates")})
        row["accounted_wall_seconds"] = number(payload.get("fitting_seconds"))
        row["action_model_fitting_seconds_subset"] = number(payload.get("action_model_fitting_seconds"))
        row["command_model_fitting_seconds_subset"] = number(payload.get("command_model_fitting_seconds"))
        row["micro_samples_in_fit_pool"] = number(payload.get("micro_samples"))
        row["macro_samples_in_fit_pool"] = number(payload.get("macro_samples"))
    elif kind == "adjoint":
        row.update({key: number(payload.get(key)) for key in ("adjoint_forward_steps", "adjoint_reverse_steps")})
        row["accounted_wall_seconds"] = number(payload.get("adjoint_seconds"))
    return row


def aggregate_cost(rows):
    observed = [r for r in rows if r["status"] == "RECORDED"]
    kinds = {
        "policy_steps": "collection", "physics_steps": "collection",
        "warmup_physics_steps": "collection",
        "warmup_and_execution_physics_steps": "collection",
        "actor_updates": "ppo", "value_updates": "ppo",
        "action_model_updates": "model_fit", "command_model_updates": "model_fit",
        "adjoint_forward_steps": "adjoint", "adjoint_reverse_steps": "adjoint",
    }
    return {
        "recorded_phase_count": len(observed),
        "unrecorded_phase_count": len(rows) - len(observed),
        "accounting_complete": len(observed) == len(rows),
        **{key: total(r[key] for r in observed if r["kind"] == kinds[key])
           for key in COUNT_FIELDS},
        "accounted_phase_wall_seconds": total(r["accounted_wall_seconds"] for r in observed),
        "instrumented_phase_wall_seconds": total(r.get("instrumented_phase_wall_seconds") for r in observed),
        "instrumented_completed_phase_count": sum(r.get("instrumented_phase_wall_seconds") is not None for r in observed),
        "event_only_recovered_phase_count": sum(r.get("evidence_origin") == "event_only" for r in observed),
        "effective_recorded_phase_wall_seconds": total(
            r.get("effective_recorded_phase_wall_seconds", r["accounted_wall_seconds"]) for r in observed),
        "collection_wall_seconds": total(r["accounted_wall_seconds"] for r in observed if r["kind"] == "collection"),
        "ppo_wall_seconds": total(r["accounted_wall_seconds"] for r in observed if r["kind"] == "ppo"),
        "model_fitting_wall_seconds": total(r["accounted_wall_seconds"] for r in observed if r["kind"] == "model_fit"),
        "adjoint_wall_seconds": total(r["accounted_wall_seconds"] for r in observed if r["kind"] == "adjoint"),
        "planner_wall_seconds_subset_of_collection": total(r["planner_seconds_subset"] for r in observed),
        "action_model_fitting_seconds_subset": total(r["action_model_fitting_seconds_subset"] for r in observed),
        "command_model_fitting_seconds_subset": total(r["command_model_fitting_seconds_subset"] for r in observed),
        "cost_is_observed_accounted_work_not_unrecorded_running_work": True,
    }


def load_record(evidence, base, round_number, result):
    relative = f"{base}/round{round_number}/record.json"
    record = evidence.read(relative)
    if record is not None:
        return record, relative
    for item in (result or {}).get("records", []):
        if item.get("round") == round_number:
            return item, f"{base}/result.json"
    return None, None


def load_collection(evidence, relative, record=None, phase=None):
    collection = evidence.read(relative)
    if collection is not None:
        return collection, relative
    if record is not None and isinstance(record.get(phase), dict):
        return {"stats": record[phase],
                "episodes": record.get("evaluation_episodes", []) if phase == "evaluation" else []}, None
    return None, None


def analyze(runs, output, config_override=None, execution_dir=None,
            invalid_comparison_reason=None):
    evidence = Evidence(runs)
    execution = ExecutionEvidence(execution_dir)
    validity = comparison_validity(evidence, invalid_comparison_reason)
    comparison_valid = validity["comparison_valid"]
    config = evidence.read("config.json")
    config_source = "runs/config.json"
    if config is None:
        config_path = config_override or ROOT / "configs/go2_adjoint_learning.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_source = "declared_config_fallback; run config unavailable"
    optimizer_initialization, initialization_rows = audit_initializations(evidence, config)
    failure = evidence.read("execution_failure.json")
    run_source = evidence.read("source.json")
    expected_final = int(config["rounds"])
    cases, seeds, arms = config["cases"], config["seeds"], config["arms"]
    eval_rows, round_rows, ledger, branches, common_rows, initializations = [], [], [], [], [], []
    all_collection_episodes = []
    for seed in seeds:
        common_base = f"seed{seed}/common"
        common_ledger = []
        for phase in ("behavior", "calibration"):
            relative = f"{common_base}/{phase}/summary.json"
            collection = evidence.read(relative)
            all_collection_episodes.extend(saved_collection_episode_rows(
                seed, "common", 0, phase, collection, relative))
            common_ledger.append(execution.row(
                seed, "common", 0, phase, "collection",
                collection.get("stats") if collection else None, relative,
                f"{common_base}/{phase}", "collect"))
        common_fit = evidence.read(f"{common_base}/fit.json")
        common_ledger.append(execution.row(
            seed, "common", 0, "value_fit", "ppo",
            common_fit.get("value") if common_fit else None, f"{common_base}/fit.json",
            f"{common_base}/behavior", "ppo"))
        common_ledger.append(execution.row(
            seed, "common", 0, "fact_fit", "model_fit",
            common_fit.get("facts") if common_fit else None, f"{common_base}/fit.json",
            f"{common_base}/calibration", "fit_facts"))
        ledger.extend(common_ledger)
        common_cost = aggregate_cost(common_ledger)
        common_wall = execution.interval(f"{common_base}/behavior", "collect",
                                         f"{common_base}/calibration", "fit_facts")
        common_wall_basis = (
            "launcher elapsed: first common collection start to fact-fit end; "
            "excludes model loading and subsequent initialization checkpoint saving"
            if common_wall is not None else "unavailable; phase sums are reported separately")
        common_rows.append({"seed": seed, **common_cost,
                            "measured_common_wall_seconds": common_wall,
                            "common_wall_basis": common_wall_basis})
        seed_hashes = {}
        for arm in arms:
            base = f"seed{seed}/{arm}"
            result = evidence.read(f"{base}/result.json")
            seed_hashes[arm] = evidence.read(f"{base}/initialization_hashes.json")
            branch_ledger = []
            observed_round_records = []
            for round_number in range(expected_final + 1):
                record, record_source = (None, None) if round_number == 0 else load_record(
                    evidence, base, round_number, result)
                if record is not None:
                    observed_round_records.append(round_number)
                if round_number == 0:
                    collection_path = f"{base}/evaluation_round0/summary.json"
                else:
                    collection_path = f"{base}/round{round_number}/evaluation/summary.json"
                collection, collection_source = load_collection(evidence, collection_path, record, "evaluation")
                all_collection_episodes.extend(saved_collection_episode_rows(
                    seed, arm, round_number, "evaluation", collection, collection_source))
                source = collection_source or (record_source if collection else None)
                rows = evaluation_rows(seed, arm, round_number, config, collection, source)
                eval_rows.extend(rows)
                row = {"seed": seed, "arm": arm, "round": round_number,
                       "is_fixed_final_round": round_number == expected_final,
                       "training_round_recorded": record is not None if round_number else None,
                       **aggregate_evaluation(rows)}
                phase_ledger = []
                if round_number > 0:
                    for phase in ("behavior", "calibration"):
                        relative = f"{base}/round{round_number}/{phase}/summary.json"
                        other, other_source = load_collection(evidence, relative, record, phase)
                        all_collection_episodes.extend(saved_collection_episode_rows(
                            seed, arm, round_number, phase, other, other_source))
                        phase_ledger.append(execution.row(
                            seed, arm, round_number, phase, "collection",
                            other.get("stats") if other else None, other_source or record_source,
                            f"{base}/round{round_number}/{phase}", "collect"))
                    for phase, kind in (("ppo", "ppo"), ("facts", "model_fit"), ("derivatives", "adjoint")):
                        event_phase = "calibration" if phase == "facts" else "behavior"
                        operation = {"ppo": "ppo", "facts": "fit_facts", "derivatives": "adjoint"}[phase]
                        phase_ledger.append(execution.row(
                            seed, arm, round_number, phase, kind,
                            record.get(phase) if record else None, record_source,
                            f"{base}/round{round_number}/{event_phase}", operation))
                event_path = (f"{base}/evaluation_round0" if round_number == 0 else
                              f"{base}/round{round_number}/evaluation")
                phase_ledger.append(execution.row(
                    seed, arm, round_number, "evaluation", "collection",
                    collection.get("stats") if collection else None, source, event_path, "collect"))
                cost = aggregate_cost(phase_ledger)
                row.update({f"round_{key}": value for key, value in cost.items()})
                crosses_segments = execution.spans_segments(
                    f"{base}/evaluation_round0", "collect", event_path, "collect")
                recorded_basis = (record or {}).get("wall_time_basis")
                measured_branch_wall = (
                    number((record or {}).get("wall_seconds_including_initial_evaluation"))
                    if not crosses_segments or recorded_basis == "active_segments_excluding_downtime"
                    else None)
                wall_basis = "runner elapsed including initial evaluation and completed-round checkpoint saving"
                if recorded_basis == "active_segments_excluding_downtime":
                    wall_basis = "runner active-segment elapsed including round0 and checkpoint saving; downtime excluded"
                if measured_branch_wall is None:
                    measured_branch_wall = execution.interval(
                        f"{base}/evaluation_round0", "collect", event_path, "collect")
                    wall_basis = ("launcher elapsed: first round0 collection start to this evaluation end; "
                                  "sum active segments, downtime excluded; excludes any subsequent checkpoint saving")
                if measured_branch_wall is None:
                    wall_basis = "unavailable; phase sums are reported separately"
                cumulative_cost = aggregate_cost([*branch_ledger, *phase_ledger])
                cumulative_complete = cumulative_cost["accounting_complete"] and common_cost["accounting_complete"]
                row.update({
                    "reported_cumulative_wall_seconds_including_round0": measured_branch_wall,
                    "reported_cumulative_wall_basis": wall_basis,
                    "measured_common_wall_seconds": common_wall,
                    "common_wall_basis": common_wall_basis,
                    "reported_cumulative_wall_seconds_including_common": (
                        measured_branch_wall + common_wall
                        if measured_branch_wall is not None and common_wall is not None else None),
                    "cumulative_cost_accounting_complete": cumulative_complete,
                    "cumulative_standalone_physics_steps_including_common_and_stands": (
                        total([cumulative_cost["warmup_and_execution_physics_steps"],
                               common_cost["warmup_and_execution_physics_steps"]])
                        if cumulative_complete else None),
                    "cumulative_observed_physics_steps_lower_bound_including_common_and_stands": total(
                        [cumulative_cost["warmup_and_execution_physics_steps"],
                         common_cost["warmup_and_execution_physics_steps"]]),
                })
                round_rows.append(row)
                branch_ledger.extend(phase_ledger)
            ledger.extend(branch_ledger)
            cost = aggregate_cost(branch_ledger)
            present = (runs / base).is_dir() or execution.has_branch(base)
            final_rows = [r for r in eval_rows if r["seed"] == seed and r["arm"] == arm
                          and r["round"] == expected_final]
            final = aggregate_evaluation(final_rows)
            result_status = (result or {}).get("status")
            status = result_status or ("INCOMPLETE_NO_RESULT" if present else "NOT_RUN")
            reported_wall = number((result or {}).get("wall_seconds"))
            raw_runner_wall = reported_wall
            branch_wall_basis = "runner branch elapsed including initial evaluation"
            final_event_path = f"{base}/round{expected_final}/evaluation"
            if execution.spans_segments(f"{base}/evaluation_round0", "collect", final_event_path, "collect"):
                if (result or {}).get("wall_time_basis") == "active_segments_excluding_downtime":
                    branch_wall_basis = "runner active-segment branch elapsed; downtime excluded"
                else:
                    reported_wall = execution.interval(
                        f"{base}/evaluation_round0", "collect", final_event_path, "collect")
                    branch_wall_basis = (
                        "launcher active interval through final evaluation; downtime excluded; "
                        "excludes subsequent checkpoint saving")
            complete_branch = (status == "COMPLETED"
                               and len(observed_round_records) == expected_final
                               and final["status"] == "COMPLETE_EVALUATION")
            branch = {
                "seed": seed, "arm": arm, "runner_status": status,
                "protocol_complete": complete_branch,
                "training_rounds_recorded": len(observed_round_records),
                "fixed_final_round_status": final["status"],
                **cost,
                "reported_branch_wall_seconds_including_round0": reported_wall,
                "reported_branch_wall_basis": branch_wall_basis,
                "raw_runner_branch_wall_seconds": raw_runner_wall,
                "reported_cumulative_actor_updates": number((result or {}).get("actor_updates")),
                "reported_cumulative_value_updates_including_common": number((result or {}).get("value_updates")),
                "standalone_policy_steps_including_one_common": (
                    total([cost["policy_steps"], common_cost["policy_steps"]])
                    if present else None),
                "standalone_physics_steps_including_one_common_and_all_stands": (
                    total([cost["warmup_and_execution_physics_steps"],
                           common_cost["warmup_and_execution_physics_steps"]])
                    if present else None),
                "standalone_accounted_phase_wall_seconds_including_one_common": (
                    total([cost["accounted_phase_wall_seconds"],
                           common_cost["accounted_phase_wall_seconds"]])
                    if present else None),
                "standalone_effective_phase_wall_seconds_including_one_common": (
                    total([cost["effective_recorded_phase_wall_seconds"],
                           common_cost["effective_recorded_phase_wall_seconds"]])
                    if present else None),
                "standalone_cost_accounting_complete": cost["accounting_complete"] and common_cost["accounting_complete"],
            }
            branch["ordinary_unused_f_branch_seconds"] = cost["action_model_fitting_seconds_subset"] if arm == "ordinary" else None
            branch["ordinary_unused_f_common_seconds"] = common_cost["action_model_fitting_seconds_subset"] if arm == "ordinary" and present else None
            branch["ordinary_optimistic_branch_wall_minus_f_seconds"] = (
                subtract(reported_wall, cost["action_model_fitting_seconds_subset"])
                if arm == "ordinary" and cost["accounting_complete"] else None)
            branch["ordinary_optimistic_standalone_phase_seconds_minus_f"] = (
                subtract(branch["standalone_accounted_phase_wall_seconds_including_one_common"],
                         total([cost["action_model_fitting_seconds_subset"],
                                common_cost["action_model_fitting_seconds_subset"]]))
                if arm == "ordinary" and branch["standalone_cost_accounting_complete"]
                and cost["action_model_fitting_seconds_subset"] is not None
                and common_cost["action_model_fitting_seconds_subset"] is not None else None)
            branches.append(branch)
        available_hashes = [v for v in seed_hashes.values() if v is not None]
        initializations.append({
            "seed": seed, "arms_with_hashes": [a for a, h in seed_hashes.items() if h is not None],
            "all_declared_arms_present": len(available_hashes) == len(arms),
            "available_parameter_hashes_equal": (
                all(h == available_hashes[0] for h in available_hashes)
                if len(available_hashes) >= 2 else None),
        })
    final_rows = [r for r in eval_rows if r["round"] == expected_final]
    by_arm = []
    for arm in arms:
        rows = [r for r in final_rows if r["arm"] == arm]
        seed_summaries = [aggregate_evaluation([r for r in rows if r["seed"] == seed]) for seed in seeds]
        completed_seeds = [s for s in seed_summaries if s["status"] == "COMPLETE_EVALUATION"]
        by_arm.append({
            "arm": arm, **aggregate_evaluation(rows),
            "complete_seed_count": len(completed_seeds),
            "success_time_across_complete_seed_means_sd_seconds": sd(
                s["success_time_mean_seconds"] for s in completed_seeds),
            "return_across_complete_seed_means_sd": sd(
                s["all_completed_real_return_mean"] for s in completed_seeds),
        })
    pairs = []
    lookup = {(r["seed"], r["arm"], r["case"], r["episode"]): r for r in final_rows}
    if "ordinary" in arms and "adjoint" in arms:
        for seed in seeds:
            for case in cases:
                for episode in range(int(config["evaluation_complete_episodes_per_case"])):
                    a = lookup[seed, "ordinary", case, episode]
                    b = lookup[seed, "adjoint", case, episode]
                    both_complete = a["status"] == b["status"] == "COMPLETED"
                    both_success = both_complete and a["success"] and b["success"]
                    pairs.append({
                        "seed": seed, "case": case, "episode": episode,
                        "comparison_valid": comparison_valid,
                        "not_inferential": not comparison_valid,
                        "comparison_interpretation": (
                            "DESCRIPTIVE_PAIRED_DIFFERENCE_NO_SIGNIFICANCE_TEST"
                            if comparison_valid else "INVALID_COMPARISON_DESCRIPTIVE_ONLY"),
                        "ordinary_status": a["status"], "adjoint_status": b["status"],
                        "ordinary_success": a["success"], "adjoint_success": b["success"],
                        "both_success": bool(both_success) if both_complete else None,
                        "adjoint_minus_ordinary_completion_seconds": subtract(
                            b["completion_seconds"], a["completion_seconds"]) if both_success else None,
                        "adjoint_minus_ordinary_real_return": subtract(
                            b["real_return"], a["real_return"]) if both_complete else None,
                        "adjoint_minus_ordinary_work_J_all_completed": subtract(
                            b["mechanical_work_proxy_J"], a["mechanical_work_proxy_J"]) if both_complete else None,
                    })
    project_cost = aggregate_cost(ledger)
    unresolved_collects = [
        phase for phase in execution.public.get("segment_phases_without_end", [])
        if phase["operation"] == "collect"]
    project_cost.update({
        "recorded_physics_steps_lower_bound_including_stands": (
            project_cost["warmup_and_execution_physics_steps"]),
        "unrecorded_interrupted_physics_steps": None if unresolved_collects else 0,
        "physics_cost_status": (
            "LOWER_BOUND_UNRECORDED_INTERRUPTED_WORK" if unresolved_collects
            else "RECORDED_PHASES_ONLY"),
        "unmatched_collection_starts": unresolved_collects,
    })
    any_evidence = project_cost["recorded_phase_count"] > 0
    records_complete = all(b["protocol_complete"] for b in branches)
    execution_conflict = any(
        issue["issue"] in ("DUPLICATE_PHASE_END", "DUPLICATE_PHASE_END_ACROSS_SEGMENTS",
                           "EXECUTION_SEGMENT_ALGORITHM_OR_CONFIG_MISMATCH",
                           "RUNNER_EVENT_STATS_MISMATCH")
        for issue in execution.issues)
    all_complete = records_complete and not execution_conflict
    launcher_failed = (execution.public.get("experiment_exit", {}).get("status") == "FAILED"
                       or bool(execution.public.get("execution_failure")))
    current_failure = launcher_failed or (bool(failure) and execution.segment_count <= 1)
    execution_status = ("COMPLETED" if all_complete else "EVIDENCE_CONFLICT" if execution_conflict else
                        "FAILED_OR_INTERRUPTED" if current_failure else
                        "PARTIAL_SNAPSHOT" if any_evidence else "NOT_RUN")
    status = execution_status if comparison_valid else "INVALID_COMPARISON"
    collection_by_phase_arm = summarize_collection_episodes(all_collection_episodes)
    for rows in (eval_rows, round_rows, by_arm, branches,
                 all_collection_episodes, collection_by_phase_arm):
        for row in rows:
            row["comparison_valid"] = comparison_valid
            row["not_inferential"] = not comparison_valid
    summary = {
        "schema": 1, "status": status, "protocol_complete": all_complete,
        "execution_status": execution_status,
        "comparison_valid": comparison_valid,
        "comparison_validity": validity,
        "scientific_comparison_complete": all_complete and comparison_valid,
        "all_declared_branch_records_complete": records_complete,
        "completed_with_execution_incidents": all_complete and bool(
            failure or execution.public.get("incidents")),
        "config_source": config_source,
        "protocol": {"seeds": seeds, "arms": arms, "known_cases": cases,
                     "rounds": expected_final,
                     "evaluation_episodes_per_case": config["evaluation_complete_episodes_per_case"],
                     "selected_checkpoint": "fixed final round, never substituted with an earlier winner"},
        "interpretation": [
            "The three cases are known development conditions, not independent generalization repetitions.",
            "Training seeds are separate fits; pooled seed-by-case cells are not independent test tasks.",
            "If comparison_valid is false, all method outcomes and paired differences are descriptive only; no fair ranking or scientific treatment comparison is supported.",
            "Failure elapsed time is not completion time; completion means use successes only and require success counts alongside them.",
            "NOT_RUN cells have null success and are excluded from observed success fractions; missing outcomes are never filled as failures or wins.",
            "All recorded failure work and returns are retained; absolute mechanical work is a proxy, not battery energy.",
            "Round zero is retained and included in costs. Common data and fitting are counted once per seed in the project ledger.",
            "Standalone method costs allocate that method one full common initialization; do not sum those standalone columns for project totals.",
            "Planner time is a subset of collection time; f/g fit times are subsets of model fit time and are not added twice.",
            "Phase wall sums exclude unlogged setup, serialization and incomplete phases; an interrupted run's recorded cost is a lower bound.",
            "Instrumented phase times include launcher-wrapped work such as collection saving; effective phase time prefers that event duration and otherwise uses the runner timer.",
            "Multiple execution directories are chronological process segments. Total active wall time sums their exit durations; downtime from UTC anchors is listed separately and never charged as training.",
            "An earlier failed segment remains an execution incident. Final COMPLETED requires every declared branch/round and final evaluation, without duplicate completed phases or conflicting source/stat records.",
            "Completed event stats recover updates or fitting even if a later failure prevented the round record from being written. A start without an end is not assigned a made-up update count.",
            "Unrecorded physics in interrupted collection phases remains unknown; the summed recorded physics is a lower bound, including both process segments' recorded work.",
            "A missing evaluation row means no per-case outcome is available; an event's aggregate success count is never used to invent per-case results.",
            "The ordinary minus-f figures remove recorded unused f fitting only, assuming all other work unchanged. They are optimistic accounting estimates, not measured reruns or guaranteed speedups.",
            "The continuation critic fits stochastic behavior returns and is a proxy for deterministic mean-policy continuation.",
        ],
        "final_by_arm": by_arm,
        "branch_lifecycle_and_costs": branches,
        "shared_common_once_per_seed": common_rows,
        "sum_measured_common_intervals_seconds": total(
            row["measured_common_wall_seconds"] for row in common_rows),
        "project_observed_costs_common_counted_once": project_cost,
        "execution_measurements": execution.public,
        "sum_recorded_branch_wall_seconds_excluding_shared_common": total(
            b["reported_branch_wall_seconds_including_round0"] for b in branches),
        "initialization_parameter_identity": initializations,
        "initialization_network_and_optimizer_identity": optimizer_initialization,
        "all_saved_collection_episodes": {
            "recorded_episode_rows": len(all_collection_episodes),
            "saved_collection_summary_count": len({row["source"] for row in all_collection_episodes}),
            "successful_complete_episodes": sum(row["success"] is True for row in all_collection_episodes),
            "failed_complete_episodes": sum(row["success"] is False for row in all_collection_episodes),
            "sampling_cut_episodes": sum(row["reason"] == "sampling_cut" for row in all_collection_episodes),
            "by_phase_and_arm": collection_by_phase_arm,
            "interpretation": [
                "Only episodes in saved collection summary.json files are included; no missing slots are invented.",
                "Common and branch behavior B use stochastic Gaussian actions; calibration C and evaluation use the deterministic mean.",
                "Sampling cuts retain elapsed time, return and resource usage but have null success and completion time.",
                "Unfinished or unsaved episodes in interrupted phases remain unknown, not zero or failed episodes.",
                "Training/calibration episodes and repeated known cases are not independent held-out test samples.",
            ],
        },
        "root_failure": ({"status": failure.get("status"),
                          "exception_type": failure.get("exception_type")} if failure else None),
        "evidence_issues": [*evidence.issues, *execution.issues,
                            *(execution.evidence.issues if execution.evidence else [])],
        "run_source_hashes": run_source,
        "additional_training_updates": 0, "additional_physics_steps": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_csv(output / "evaluation_rounds.csv", eval_rows, EPISODE_FIELDS)
    write_csv(output / "final_evaluation.csv", final_rows, EPISODE_FIELDS)
    write_csv(output / "round_summary.csv", round_rows)
    write_csv(output / "final_by_arm.csv", by_arm)
    write_csv(output / "paired_final_adjoint_vs_ordinary.csv", pairs,
              ("seed", "case", "episode", "ordinary_status", "adjoint_status"))
    write_csv(output / "cost_ledger.csv", ledger)
    write_csv(output / "branch_costs.csv", branches)
    write_csv(output / "shared_common_costs.csv", common_rows)
    write_csv(output / "initialization_startups.csv", initialization_rows)
    write_csv(output / "all_collection_episodes.csv", all_collection_episodes, COLLECTION_EPISODE_FIELDS)
    write_csv(output / "all_collection_episodes_by_phase_arm.csv", collection_by_phase_arm,
              ("phase", "arm", "behavior_mode", "recorded_episode_rows"))
    write_csv(output / "execution_event_reconciliation.csv", execution.comparisons,
              ("path", "operation", "field", "runner_value", "event_value", "matched",
               "runner_source", "event_source"))
    script = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    write_json(output / "analysis_manifest.json", {
        "analysis_script": "scripts/analyze_go2_adjoint_pilot.py",
        "analysis_script_sha256_lf": hashlib.sha256(script).hexdigest(),
        "input_files": evidence.files,
        "execution_metadata_files": execution.evidence.files if execution.evidence else {},
        "comparison_valid": comparison_valid,
        "comparison_validity": validity,
        "config_sha256_canonical_json": hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "additional_training_updates": 0, "additional_physics_steps": 0,
        "checkpoint_or_pickle_loading": False,
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True,
                        help="Saved runner root containing config.json and seed*/")
    parser.add_argument("--output", type=Path, required=True,
                        help="Public analytical output folder; never the raw run root")
    parser.add_argument("--config", type=Path,
                        help="Declared protocol fallback only if runs/config.json is absent")
    parser.add_argument("--execution-dir", type=Path, action="append",
                        help="Optional launcher folder; repeat in chronological order for exact-resume segments")
    parser.add_argument("--invalid-comparison-reason",
                        help="Preserve raw facts/costs but mark INVALID_COMPARISON; also reads runs/comparison_validity.json")
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error("--runs must be an existing directory")
    if args.output.resolve() == args.runs.resolve():
        parser.error("--output must differ from the raw run root")
    if args.execution_dir is not None and any(not directory.is_dir() for directory in args.execution_dir):
        parser.error("Every --execution-dir must be an existing directory")
    summary = analyze(args.runs, args.output, args.config, args.execution_dir,
                      args.invalid_comparison_reason)
    print(json.dumps({"status": summary["status"],
                      "execution_status": summary["execution_status"],
                      "comparison_valid": summary["comparison_valid"],
                      "protocol_complete": summary["protocol_complete"],
                      "recorded_phases": summary["project_observed_costs_common_counted_once"]["recorded_phase_count"],
                      "additional_physics_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
