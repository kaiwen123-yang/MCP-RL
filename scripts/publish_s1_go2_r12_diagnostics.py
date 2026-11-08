#!/usr/bin/env python3
"""Deliver selected, losslessly packaged R12 diagnostics without running physics."""
import argparse
import contextlib
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from s1_go2_job_limits import expired, goal_based, remove_wall_deadline, request_timeout

ROOT = Path(__file__).resolve().parents[1]
RUN = "tm-s1-go2-local-readiness-r12"
SMALL = ROOT / "results/terrain_memory/simulation" / RUN
LARGE = ROOT / "outputs/terrain_memory/simulation" / RUN
PART_BYTES = 256 * 1024**2


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else ({} if default is None else default)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def local(path):
    result = (ROOT / path).resolve()
    result.relative_to(ROOT)
    return result


def scrub(message):
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "https_proxy", "HTTPS_PROXY"):
        if os.environ.get(name):
            message = message.replace(os.environ[name], "[redacted]")
    return message


def pending(ctx, error):
    directory, _, _, destination, clock = ctx
    value = read(destination / "evidence.json")
    value.update(job=directory.name, status="publication_pending", deadline=clock,
                 last_error=scrub(f"{type(error).__name__}: {error}"))
    write(destination / "evidence.json", value)
    return value


@contextlib.contextmanager
def locked(path, nonblocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def remaining(clock):
    deadline = clock.get("deadline_unix_s")
    if deadline is None:
        return None
    value = deadline - time.time()
    if value <= 0:
        raise TimeoutError("diagnostic_deadline_expired; ordinary resume does not extend it")
    return value


def context(args):
    directory = local(args.job_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"source_job_directory_unavailable: {directory}; no upload was performed")
    job = read(directory / "job.json")
    if not job:
        raise FileNotFoundError(f"source job.json unavailable: {directory}")
    transfer = directory / "diagnostics-transfer"
    transfer.mkdir(exist_ok=True)
    # A newly authorized delivery has its own immutable clock. The original
    # upload, training and GPU clocks are never overwritten or extended.
    clock_path = (transfer / "delivery-windows" / args.delivery_window / "deadline.json"
                  if args.delivery_window else transfer / "deadline.json")
    with locked(transfer / "clock.lock"):
        clock = read(clock_path)
        no_deadline = args.no_wall_deadline or goal_based(read(directory / "execution-contract.json"))
        if not clock:
            clock = dict(job=directory.name, started_unix_s=time.time(), hours=args.hours)
            clock["deadline_unix_s"] = None if no_deadline else clock["started_unix_s"] + args.hours * 3600
            if args.delivery_window:
                clock.update(delivery_window=args.delivery_window,
                    prior_clock=relative(transfer / "deadline.json"),
                    authorization="Explicit user-selected additional diagnostic delivery window; original clocks, published assets and server originals retained.")
        if no_deadline:
            remove_wall_deadline(clock)
            clock["hours"] = None
        write(clock_path, clock)
    destination = SMALL / "jobs" / directory.name / "diagnostics"
    return directory, job, transfer, destination, clock


def select(directory, job, old, extra):
    summary = read(SMALL / "jobs" / (directory.name + ".json"))
    evaluation = summary.get("evaluation") or read(directory / "evaluation.json")
    if "attempts" in summary:
        attempts = summary["attempts"]
    else:
        initial = set(job.get("initial_attempts", []))
        attempts = [x.name for x in LARGE.glob(f'train-{job["group"]}-s{job["seed"]}-ep*')
                    if x.is_dir() and x.name not in initial
                    and ("initial_attempts" in job or x.stat().st_mtime >= job["started_unix_s"])]
        attempts += evaluation.get("attempts", [])
    attempts = sorted(set(attempts))
    dense = set(old.get("expanded_attempts", [])) | set(extra)
    if dense - set(attempts):
        raise ValueError("--attempt must belong to this job: " + str(sorted(dense - set(attempts))))
    evals = set(evaluation.get("attempts", []))
    dense.update(evals)
    records = {name: read(SMALL / (name + ".json")) for name in attempts}
    first_failures = {}
    for name in attempts:
        if name in evals:
            continue
        reason = records[name].get("reason", "")
        if reason.startswith(("body_fall", "invalid_contact", "material_domain_exit")):
            first_failures.setdefault(reason.split(":", 1)[0], name)
    dense.update(first_failures.values())
    order = sorted(attempts, key=lambda n: (0 if n in evals else 1 if n in dense else 2, n))
    stream_path = (job.get("training_directory") or
                   summary.get("bulk_retention", {}).get("shared_training_directory") or
                   str((LARGE / "training" / f'{job["group"]}-seed{job["seed"]}').relative_to(ROOT)))
    stream = local(stream_path)
    collection = read(directory / "stream-records/collection.json") or read(stream / "collection.json")
    final = summary.get("training", {}).get("actual_transitions", collection.get("collected_transitions"))
    return summary, order, dense, records, stream, final


def transitions(stream, transfer, initial, final, old, clock):
    source, target = stream / "transitions.jsonl", transfer / "transitions-selected.jsonl"
    info = dict(source=relative(source), initial_exclusive=initial, final_inclusive=final)
    if not source.is_file() or final is None:
        return dict(info, status="missing_source_or_final_count"), None, None
    stamp = source.stat()
    info.update(source_size=stamp.st_size, source_mtime_ns=stamp.st_mtime_ns)
    if all(old.get(k) == v for k, v in info.items()) and target.is_file() and old.get("status") == "saved":
        bad = transfer / "transitions-unparsed.jsonl"
        return old, target, bad if bad.is_file() and bad.stat().st_size else None
    invalid = transfer / "transitions-unparsed.jsonl"
    counts, identities, bad_lines = 0, set(), []
    with source.open("rb") as src, target.open("wb") as out, invalid.open("wb") as bad:
        for number, line in enumerate(src, 1):
            remaining(clock)
            if not line.strip():
                continue
            try:
                serial = int(json.loads(line)["transition"])
            except (ValueError, KeyError, TypeError):
                bad.write(line); bad_lines.append(number)
                continue
            if initial < serial <= final:
                out.write(line); counts += 1; identities.add(serial)
    current = source.stat()
    stable = (stamp.st_size, stamp.st_mtime_ns) == (current.st_size, current.st_mtime_ns)
    info.update(status="saved" if stable else "source_changed", selected_rows=counts,
                distinct_transitions=len(identities), expected_rows=final-initial,
                malformed_source_lines=bad_lines, complete=stable and not bad_lines and
                len(identities) == counts == final-initial, columns="Original selected lines, unchanged")
    return info, target, invalid if bad_lines else None


class Parts:
    def __init__(self, prefix, clock):
        self.prefix, self.clock, self.handle, self.rows = prefix, clock, None, []

    def write(self, data):
        total = len(data)
        while data:
            remaining(self.clock)
            if self.handle is None or self.rows[-1]["size"] == PART_BYTES:
                if self.handle:
                    self.handle.close()
                path = self.prefix.with_name(self.prefix.name + f".part{len(self.rows)+1:05d}")
                self.handle = path.open("wb")
                self.rows.append(dict(path=relative(path), part=len(self.rows)+1,
                                      offset=sum(x["size"] for x in self.rows), size=0))
            chunk = data[:PART_BYTES-self.rows[-1]["size"]]
            self.handle.write(chunk); self.rows[-1]["size"] += len(chunk); data = data[len(chunk):]
        return total

    def close(self):
        if self.handle:
            self.handle.close()
        else:
            self.prefix.touch()
            self.rows = [dict(path=relative(self.prefix), part=1, offset=0, size=0)]
        if len(self.rows) == 1 and local(self.rows[0]["path"]) != self.prefix:
            local(self.rows[0]["path"]).replace(self.prefix)
            self.rows[0]["path"] = relative(self.prefix)


def package(source, item, previous, transfer, clock):
    stamp = source.stat()
    if (previous.get("source_size"), previous.get("source_mtime_ns")) == (stamp.st_size, stamp.st_mtime_ns):
        parts = previous.get("parts", [])
        if parts and all(local(p["path"]).is_file() and local(p["path"]).stat().st_size == p["size"] for p in parts):
            return dict(previous, **item)
    generation = previous.get("generation", 0) + 1
    codec = "gzip" if source.suffix in (".json", ".jsonl") else "identity"
    if source.suffix == ".npz" and stamp.st_size < PART_BYTES:
        return dict(item, status="prepared", generation=generation, codec="identity",
                    source_size=stamp.st_size, source_mtime_ns=stamp.st_mtime_ns, empty_source=stamp.st_size == 0,
                    parts=[dict(path=relative(source), part=1, offset=0, size=stamp.st_size, direct_source=True,
                                transfer_name=f'{item["file_id"]}-g{generation}__{source.name}')],
                    reconstruction="Original closed-episode NPZ bytes, without repacking or column changes")
    prefix = transfer / f'{item["file_id"]}-g{generation}__{source.name}{".gz" if codec == "gzip" else ""}'
    writer = Parts(prefix, clock)
    try:
        with source.open("rb") as src:
            destination = gzip.GzipFile(filename="", mode="wb", fileobj=writer, compresslevel=1, mtime=0) if codec == "gzip" else writer
            try:
                while block := src.read(2 * 1024**2):
                    destination.write(block)
            finally:
                if codec == "gzip":
                    destination.close()
    finally:
        writer.close()
    after = source.stat()
    if (stamp.st_size, stamp.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("source_changed_during_packaging: " + relative(source))
    return dict(item, status="prepared", generation=generation, codec=codec,
                source_size=stamp.st_size, source_mtime_ns=stamp.st_mtime_ns, empty_source=stamp.st_size == 0,
                parts=writer.rows, reconstruction="Concatenate parts in part order, then gunzip only when codec=gzip; original bytes/columns retained")


def published_bundle_sources(job, clock):
    """Reuse W40 files already delivered in the retained six-ZIP stream set."""
    path = SMALL / "raw-bundles/w40-20261107/manifest.json"
    bundle_manifest = read(path)
    bundles = [b for b in bundle_manifest.get("archives", []) if b["job"] == job]
    if not bundles:
        return {}, None
    remotes = {}
    for tag in sorted({b["release_tag"] for b in bundles}):
        release = json.loads(subprocess.check_output(
            ["gh", "api", f"repos/kaiwen123-yang/MCP-RL/releases/tags/{tag}"],
            cwd=ROOT, text=True, timeout=request_timeout(clock.get("deadline_unix_s"), 120)))
        remotes[tag] = {a["name"]: a for a in release["assets"]}
    sources, archives = {}, []
    for bundle in bundles:
        asset = remotes[bundle["release_tag"]].get(bundle["name"])
        if not (asset and asset["state"] == "uploaded" and asset["size"] == bundle["size"]
                and (not bundle.get("remote_digest") or asset.get("digest") == bundle["remote_digest"])):
            raise RuntimeError("Previously published ZIP metadata no longer matches: " + bundle["name"])
        archives.append(dict(name=bundle["name"], release_tag=bundle["release_tag"],
            url=asset["browser_download_url"], size=asset["size"], digest=asset.get("digest"), verified=True))
        for item in bundle["files"]:
            sources[item["source_path"]] = dict(item, archive=bundle["name"],
                archive_url=asset["browser_download_url"], release_tag=bundle["release_tag"])
    return sources, dict(status="published", manifest=relative(path), archives=archives,
        source_files=len(sources), current_files_reused=0, checked_unix_s=time.time(),
        verification="Existing remote asset uploaded-state/size/digest and saved member source size/mtime; no ZIP download or reupload.",
        original_individual_assets_retained=True)


def prepare(ctx, args):
    directory, job, transfer, destination, clock = ctx
    old = read(destination / "evidence.json")
    if old.get("status") == "published" and set(args.attempt) <= set(old.get("expanded_attempts", [])):
        return old
    try:
        remaining(clock)
    except TimeoutError as error:
        return pending(ctx, error)
    allocations = {x["name"]: x for x in read(destination / "manifest.json", [])}
    previous = {x["source_path"]: x for x in old.get("files", [])}
    bundled, bundle_delivery = published_bundle_sources(directory.name, clock)
    summary, attempts, dense, records, stream, final = select(directory, job, old, args.attempt)
    evidence = dict(schema_version="R12_selected_diagnostics_v1", job=directory.name,
                    job_directory=relative(directory), stream_id=job.get("stream_id"), phase_id=job.get("phase_id"),
                    job_source_commit=job.get("source_commit"), controller_build=job.get("controller_build"),
                    deadline=clock, status="publication_pending", attempts=attempts,
                    expanded_attempts=sorted(dense), files=[], missing=[], optional_absent=[],
                    training_directory=relative(stream), initial_transitions=job["initial_transitions"], final_transitions=final,
                    accepted_rollout_full_horizon="not_recorded_in_r27; prediction arrays are feedback reference, not reconstructed accepted rollout" if job.get("controller_build") == "S5-r27" else "Only actually saved records are included",
                    schema_sources=["scripts/s1_go2_r11_runtime.py", "scripts/s1_go2_r26_reference.py", "scripts/s1_go2_r11_policy.py"],
                    clock_and_role_notes="Use each result physical_columns/physical_time_note and resolved reference_packet_fields/planner_recording for clocks, foot order and plan roles; no inferred plan_id or model identity. Future nonstandard accepted/reference/gain filenames require explicit discovery updates.",
                    retained_locally=True, automatic_deletion=False, errors=old.get("errors", []))
    sources = {}
    def add(path, attempt=None, required=False, role="diagnostic"):
        key = relative(path)
        if path.is_file():
            sources[key] = dict(path=path, attempt=attempt, role=role)
        elif required:
            evidence["missing"].append(dict(path=key, attempt=attempt, status="missing_on_source_host"))
            if key in previous:
                sources[key] = dict(path=path, attempt=attempt, role=role)
        else:
            evidence["optional_absent"].append(key)
    for name in attempts:
        folder = LARGE / name
        result, resolved = records[name], read(SMALL / (name + "-resolved.json"))
        for filename in ("feedback-actions.jsonl", "planner-records.jsonl", "reference-packets.json", "trajectory.npz",
                         "material-events.jsonl", "material-failures.jsonl", "model.mjb", "model/G1-project-go2.xml"):
            add(folder / filename, name, True)
        if resolved.get('controller_build')=='S5-r29' or (folder/'r29-event-manifest.json').exists():
            for filename in ('planner-calls.jsonl','action-execution.jsonl','r29-event-manifest.json','query-exposure.jsonl'):
                add(folder/filename,name,True,role='r29_projection_diagnostic')
        predictions = sorted(folder.glob("prediction-*.npz"))
        if not predictions:
            add(folder / "prediction-*.npz", name, True)
        for path in predictions + sorted(folder.glob("*.json")) + sorted((folder / "model").rglob("*.xml")):
            add(path, name)
        for filename in ("startup.npz", "prediction-model.xml"):
            add(folder / filename, name)
        for pattern in ("*accepted*.npz", "*reference*.npz", "*feedback*.npz", "*gain*.npz", "*linearization*.npz",
                        "*accepted*.jsonl", "*reference*.jsonl", "*linearization*.jsonl"):
            for path in sorted(folder.glob(pattern)):
                add(path, name, role="additional_recorded_planning_diagnostic")
        add(SMALL / (name + ".json"), name, True, "result_metadata")
        add(SMALL / (name + "-resolved.json"), name, True, "resolved_metadata")
        if name in dense:
            for pattern in ("physical-*.npz", "material-*.npz"):
                paths = sorted(folder.glob(pattern))
                if not paths:
                    add(folder / pattern, name, True, "selected_dense_array")
                for path in paths:
                    add(path, name, True, "selected_dense_array")
        for entry in sources.values():
            if entry["attempt"] == name:
                entry.update(source_commit=result.get("source_commit") or resolved.get("source_commit"),
                             execution_fingerprint=result.get("execution_fingerprint") or resolved.get("execution_fingerprint"),
                             reason=result.get("reason"), planner_recording=resolved.get("planner_recording"),
                             result_schema=relative(SMALL / (name + ".json")),
                             resolved_schema=relative(SMALL / (name + "-resolved.json")))
    # W42 keeps ordinary service stdout/stderr and warning counters. Old
    # missing logs are marked unavailable; never rerun history to create them.
    for name in attempts:
        for path in sorted((LARGE/name).glob("*.log"))+sorted((LARGE/name).glob("MUJOCO_LOG.TXT")):
            add(path,name,role="service_stdout_stderr")
    for pattern in ("*.log","*.txt"):
        for path in sorted(directory.glob(pattern)):
            add(path,role="job_stdout_stderr")
    for path in sorted(stream.glob("*.log")):
        add(path,role="training_stdout_stderr")
    try:
        info, selected, invalid = transitions(stream, transfer, job["initial_transitions"], final,
                                              old.get("transition_selection", {}), clock)
        evidence["transition_selection"] = info
        if selected:
            add(selected, required=True, role="selected_2Hz_transitions")
        else:
            evidence["missing"].append(info)
        if invalid:
            add(invalid, role="unparsed_transition_lines")
    except TimeoutError as error:
        evidence["errors"].append(str(error))
    rollouts = summary.get("training_binary_paths", []) or [relative(x) for x in stream.glob("rollout-*.npz")]
    rollout_count = 0
    for value in rollouts:
        path = local(value)
        match = re.fullmatch(r"rollout-(\d+)-(\d+)\.npz", path.name)
        if match and final is not None and int(match[2]) > job["initial_transitions"] and int(match[1]) <= final:
            add(path, required=True, role="complete_PANN_value_rollout")
            if relative(path) in sources:
                sources[relative(path)]["sample_range_inclusive"] = [int(match[1]), int(match[2])]
            rollout_count += 1
    expected_rollouts = max(0, summary.get("training", {}).get("updates", job["initial_updates"]) - job["initial_updates"])
    if rollout_count < expected_rollouts:
        evidence["missing"].append(dict(status="rollout_batches_unavailable", expected=expected_rollouts, found=rollout_count))
    evidence["rollout_boundary"] = "Complete original NPZ batches intersecting this job; no columns discarded, no weights read; unupdated tail values may exist only in an unselected resume checkpoint"
    manifest, next_id = [], max([int(x.get("file_id", "f0")[1:]) for x in previous.values()] + [0])
    for source_path, entry in sources.items():
        prior = previous.get(source_path, {})
        if not prior:
            next_id += 1
        item = {k: v for k, v in entry.items() if k != "path"}
        item.update(source_path=source_path, file_id=prior.get("file_id", f"f{next_id:06d}"))
        try:
            remaining(clock)
            covered = bundled.get(source_path)
            stamp = entry["path"].stat()
            if covered and (stamp.st_size, stamp.st_mtime_ns) == (covered["source_size"], covered["source_mtime_ns"]):
                packed = dict(prior, **item)
                packed.update(status="prepared", source_size=stamp.st_size,
                    source_mtime_ns=stamp.st_mtime_ns, delivery_via="existing_published_zip",
                    reused_archive={key: covered[key] for key in
                        ("archive", "archive_url", "release_tag", "member", "zip_crc32")},
                    retained_prior_individual_parts=prior.get("retained_prior_individual_parts", prior.get("parts", [])),
                    parts=[], reconstruction="Extract the named member from the existing published ZIP; original bytes retained.")
                evidence["files"].append(packed)
                bundle_delivery["current_files_reused"] += 1
                continue
            packed = package(entry["path"], item, prior, transfer, clock)
            packed.pop("delivery_via", None)
            packed.pop("reused_archive", None)
            for part in packed["parts"]:
                name = re.sub(r"[^A-Za-z0-9._-]", "_", f'diag__{directory.name}__{part.get("transfer_name", Path(part["path"]).name)}')
                part["name"] = name
                row = dict(path=part["path"], name=name, category="selected_diagnostic")
                if allocations.get(name, {}).get("release_tag"):
                    row["release_tag"] = allocations[name]["release_tag"]
                manifest.append(row)
            evidence["files"].append(packed)
        except (OSError, RuntimeError, TimeoutError) as error:
            evidence["files"].append(dict(prior, **item, status="preparation_pending", error=str(error)))
            # Preserve already assigned destinations even if a later source is unavailable.
            for part in prior.get("parts", []):
                if part.get("name") in allocations:
                    manifest.append(allocations[part["name"]])
    if old.get("resolved_prior_collision"):
        evidence["resolved_prior_collision"] = old["resolved_prior_collision"]
    if bundle_delivery:
        evidence["bundle_delivery"] = bundle_delivery
        evidence["delivery_mode"] = "existing_verified_ZIPs_plus_supplemental_assets"
        supplemental = [f for f in evidence["files"] if f.get("delivery_via") != "existing_published_zip"]
        evidence["supplemental_delivery"] = dict(status="publication_pending", source_files=len(supplemental),
            source_bytes=sum(f.get("source_size", 0) for f in supplemental), asset_count=len(manifest),
            transfer_bytes=sum(local(row["path"]).stat().st_size for row in manifest),
            manifest=relative(destination / "manifest.json"), old_archives_reuploaded=False)
    evidence["upload_workers"] = args.upload_workers
    evidence["http_proxy"] = args.http_proxy
    write(destination / "manifest.json", manifest)
    write(destination / "evidence.json", evidence)
    return evidence


def git_delivery(destination, clock, extra=()):
    paths = [relative(destination / x) for x in ("manifest.json", "evidence.json")]
    paths += [relative(SMALL / "index.json")] if (SMALL / "index.json").is_file() else []
    paths += [relative(local(x)) for x in extra if local(x).is_file()]
    def git(*command):
        return subprocess.run(["git", *command], cwd=ROOT, check=True, capture_output=True, text=True,
                              env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
                              timeout=request_timeout(clock.get("deadline_unix_s"), 120))
    with locked(SMALL / "compact-finish.lock"):
        with locked(SMALL / "index.lock"):
            git("add", "--", *paths)
            diff = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=ROOT).returncode
            if diff == 1:
                git("commit", "--only", "-m", f"Record selected diagnostics for {destination.parent.name}", "--", *paths)
            elif diff:
                raise RuntimeError("git diff failed")
        git("fetch", "origin", "main")
        git("merge", "--no-edit", "origin/main")
        git("-c", "pack.window=0", "push", "--no-thin", "origin", "HEAD:main")


def finalize(ctx, verified):
    directory, _, _, destination, clock = ctx
    evidence, manifest = read(destination / "evidence.json"), read(destination / "manifest.json", [])
    with locked(SMALL / "index.lock"):
        assets = {x["name"]: x for x in read(SMALL / "index.json").get("publication", {}).get("assets", [])}
    selected, extra = [], []
    for row in manifest:
        actual = assets.get(row["name"], {})
        state = actual.get("status", "publication_pending")
        okay = state in ("uploaded", "git_recorded") and actual.get("size") == local(row["path"]).stat().st_size
        selected.append(dict(row, status=state if okay else "publication_pending", size=actual.get("size"),
                             url=actual.get("url"), error=actual.get("upload_error")))
        if actual.get("git_path"):
            extra.append(actual["git_path"])
    complete = verified and bool(manifest) and all(x["status"] in ("uploaded", "git_recorded") for x in selected)
    complete &= not evidence["missing"] and all(x.get("status") == "prepared" for x in evidence["files"])
    complete &= evidence.get("transition_selection", {}).get("complete", False)
    if evidence.get("bundle_delivery"):
        _, latest_bundle = published_bundle_sources(directory.name, clock)
        latest_bundle["current_files_reused"] = evidence["bundle_delivery"]["current_files_reused"]
        evidence["bundle_delivery"] = latest_bundle
    for entry in evidence["files"]:
        if entry.get("delivery_via") == "existing_published_zip":
            stamp = local(entry["source_path"]).stat()
            complete &= (stamp.st_size, stamp.st_mtime_ns) == (entry["source_size"], entry["source_mtime_ns"])
        for part in entry.get("parts", []):
            if part.get("direct_source"):
                stamp = local(part["path"]).stat()
                complete &= (stamp.st_size, stamp.st_mtime_ns) == (entry["source_size"], entry["source_mtime_ns"])
    evidence.update(status="published" if complete else "publication_pending", assets=selected,
                    this_invocation_finish_verified=verified,
                    checked_unix_s=time.time(), completion_scope="This manifest only; no claim about unrelated global pending assets or unsaved accepted trajectories")
    if evidence.get("supplemental_delivery"):
        evidence["supplemental_delivery"].update(status="published" if complete else "publication_pending",
            uploaded_assets=sum(x["status"] in ("uploaded", "git_recorded") for x in selected))
        with locked(SMALL / "index.lock"):
            index = read(SMALL / "index.json")
            index.setdefault("publication", {}).setdefault("delivery_windows", {})[
                directory.name + "-supplement"] = dict(status=evidence["status"],
                evidence=relative(destination / "evidence.json"),
                reused_archive_files=evidence["bundle_delivery"]["current_files_reused"],
                supplemental_delivery=evidence["supplemental_delivery"],
                retained_conflicting_individual_assets=True)
            write(SMALL / "index.json", index)
    write(destination / "evidence.json", evidence)
    return evidence, extra


def run(ctx, args):
    directory, job, transfer, destination, clock = ctx
    with locked(transfer / "diagnostics-upload.lock", nonblocking=True):
        evidence = prepare(ctx, args)
        if evidence.get("status") == "published":
            return evidence
        try:
            git_delivery(destination, clock)
            verified = False
            for action in ("upload", "finish"):
                remaining(clock)
                command = [sys.executable, str(ROOT / "scripts/publish_s1_go2_r12.py"), action,
                           "--server-auth", "--manifest", str(destination / "manifest.json"),
                           "--job-name", directory.name + "-diagnostics", "--source-commit", job["source_commit"],
                           "--upload-workers", str(args.upload_workers)]
                if clock.get("deadline_unix_s") is None:
                    command += ["--no-wall-deadline"]
                else:
                    command += ["--deadline-unix", str(clock["deadline_unix_s"])]
                seconds = remaining(clock)
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                        timeout=None if seconds is None else seconds + 1)
                with (transfer / "publisher.log").open("a") as log:
                    log.write(scrub(result.stdout + result.stderr))
                if action == "finish":
                    verified = result.returncode == 0
                if result.returncode:
                    evidence = read(destination / "evidence.json")
                    evidence["errors"].append(dict(action=action, returncode=result.returncode,
                                                    detail=scrub(result.stdout + result.stderr)[-2000:]))
                    write(destination / "evidence.json", evidence)
            evidence, extra = finalize(ctx, verified)
            git_delivery(destination, clock, extra)
        except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
            evidence = pending(ctx, error)
        return evidence


def queue(ctx, args):
    directory, _, transfer, destination, clock = ctx
    session = re.sub(r"[^A-Za-z0-9_-]", "-", "diag-" + directory.name)
    with locked(transfer / "queue.lock"):
        previous = read(destination / "evidence.json")
        if previous.get("status") == "published" and set(args.attempt) <= set(previous.get("expanded_attempts", [])):
            return dict(status="already_published", complete=True, evidence=relative(destination / "evidence.json"))
        active = subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode == 0
        known = set(previous.get("expanded_attempts", [])) | set(read(transfer / "queued-attempts.json", []))
        if active and set(args.attempt) - known:
            return dict(status="publication_pending", session=session,
                        last_error="New --attempt selection was not queued; resume with it after the current session ends")
        if not active:
            remaining(clock)
            write(transfer / "queued-attempts.json", args.attempt)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--job-dir", relative(directory),
                       "--upload-workers", str(args.upload_workers)]
            if clock.get("deadline_unix_s") is None:
                command += ["--no-wall-deadline"]
            if args.delivery_window:
                command += ["--delivery-window", args.delivery_window]
            if args.http_proxy:
                command += ["--http-proxy", args.http_proxy]
            for attempt in args.attempt:
                command += ["--attempt", attempt]
            shell = "source scripts/autodl_d57_env.sh && source scripts/autodl_network.sh academic && exec " + shlex.join(command) + " >> " + shlex.quote(str(transfer / "run.log")) + " 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT), shell], check=True)
    return dict(status="already_running" if active else "queued", session=session,
                deadline=clock, deadline_expired=expired(clock.get("deadline_unix_s")),
                evidence=relative(destination / "evidence.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "queue"))
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--attempt", action="append", default=[], help="Add dense arrays for an attempt in this same job")
    parser.add_argument("--delivery-window", help="Explicitly authorized new delivery window ID; retain the old clock, manifest allocations and uploaded assets.")
    parser.add_argument("--hours", type=float, default=6, help="Initial diagnostic deadline only; resume never extends it")
    parser.add_argument("--no-wall-deadline", action="store_true",
                        help="Complete the selected delivery without a fixed wall-clock deadline; keep request timeouts.")
    parser.add_argument("--upload-workers",type=int,choices=range(1,13),
                        help="Concurrent files per queue; ordinary resume keeps its saved setting.")
    parser.add_argument("--http-proxy",help="Explicit server-local transfer proxy; resume keeps the saved route.")
    args = parser.parse_args()
    if not math.isfinite(args.hours) or not 0 < args.hours <= 6:
        parser.error("--hours must be finite and in (0, 6]")
    if args.delivery_window and not re.fullmatch(r"[A-Za-z0-9_-]+", args.delivery_window):
        parser.error("--delivery-window must be a simple named delivery ID")
    ctx = None
    try:
        ctx = context(args)
        if args.upload_workers is None:
            args.upload_workers = read(ctx[3] / "evidence.json").get("upload_workers",1)
        if args.http_proxy is None:
            args.http_proxy = read(ctx[3] / "evidence.json").get("http_proxy")
        if args.http_proxy:
            for key in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"):
                os.environ[key] = args.http_proxy
            for key in ("all_proxy","ALL_PROXY"):
                os.environ.pop(key,None)
            os.environ["no_proxy"] = os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
        if args.action == "queue":
            value = queue(ctx, args)
        elif args.action == "run":
            value = run(ctx, args)
        else:
            with locked(ctx[2] / "diagnostics-upload.lock", nonblocking=True):
                value = prepare(ctx, args)
        print(json.dumps({k: value[k] for k in ("status", "job", "session", "deadline", "deadline_expired", "complete", "evidence", "last_error") if k in value}, ensure_ascii=False))
        return int(value.get("status") == "publication_pending" and args.action != "prepare")
    except Exception as error:
        if ctx and not isinstance(error, BlockingIOError):
            pending(ctx, error)
        print(json.dumps(dict(status="publication_pending", error=scrub(f"{type(error).__name__}: {error}")), ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
