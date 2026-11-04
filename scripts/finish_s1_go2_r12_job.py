#!/usr/bin/env python3
"""Push compact evidence for a stopped R12 job; retain bulk outputs for the user."""
import argparse
import fcntl
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/terrain_memory/s1-go2-local-readiness-r12.json"
REPO = "kaiwen123-yang/MCP-RL"


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def training_outcomes(episodes, small, binaries, latest_resume, initial_transitions):
    """Summarize the original route criteria and saved policy observations."""
    import numpy as np
    from s1_go2_r11_runtime import PHASES

    outcome = dict(physical_episodes=0, results_available=0, results_unavailable=[],
                   legal_outbound_traversals=0, side_exits=0, far_arrivals=0,
                   return_episodes=0, completed_roundtrips=0, legal_roundtrips=0,
                   legal_return_traversals=0, legal_all_outbound_traversals=0,
                   all_passage_side_exits=0, legality_records=0, physical_failures_by_phase={},
                   saved_observations=0, nonzero_pann_observations=0,
                   return_observations=0, return_nonzero_pann_observations=0,
                   material_loaded_episodes=0,max_forward_displacement_m=None,
                   observation_source="New saved rollouts plus final valid resume tail; no inferred missing observations.")
    for episode in episodes:
        result = read(small / (episode.name + ".json")) or read(episode / "result.json")
        if not result.get("stage_progress"):
            outcome["results_unavailable"].append(episode.name)
            continue
        outcome["results_available"] += 1
        outcome["physical_episodes"] += int(result.get("simulation_s", 0) > 0)
        progress = result["stage_progress"]
        outcome['material_loaded_episodes']+=int(progress['actual_material_loaded'])
        if result.get('transitions'):
            forward=result['max_x_m']-result['transitions'][0]['position_m'][0]
            outcome['max_forward_displacement_m']=forward if outcome['max_forward_displacement_m'] is None else max(outcome['max_forward_displacement_m'],forward)
        traversal = progress["material_traversal"]
        outcome["legal_outbound_traversals"] += int(traversal["completed"])
        outcome["side_exits"] += int(traversal["lateral_exit_s"] is not None)
        outcome["far_arrivals"] += int(progress["far_arrival_s"] is not None)
        outcome["return_episodes"] += int(progress["return_start_s"] is not None)
        outcome["completed_roundtrips"] += result["completed_round_trips"]
        legal=result.get("legal_route")
        if legal:
            outcome["legality_records"] += 1
            outcome["legal_roundtrips"] += legal["completed_roundtrips"]
            outcome["legal_return_traversals"] += legal["legal_return_traversals"]
            outcome["legal_all_outbound_traversals"] += legal["legal_outbound_traversals"]
            outcome["all_passage_side_exits"] += legal["side_exits"]
        reason = result["reason"]
        kind = next((x for x in ("material_domain_exit", "invalid_contact", "body_fall") if reason.startswith(x)), None)
        if kind:
            phase = result["transitions"][-1]["phase"] if result["transitions"] else "unrecorded"
            phases = outcome["physical_failures_by_phase"].setdefault(kind, {})
            phases[phase] = phases.get(phase, 0) + 1

    def observations(values):
        obs = np.asarray(values)
        obs = obs.reshape(-1, obs.shape[-1])
        # Existing shared-observation layout: 80 channels precede the nine phases.
        returning = obs[:, 80 + PHASES.index("return")] > .5
        nonzero = np.any(obs[:, -12:] != 0, axis=1)
        outcome["saved_observations"] += len(obs)
        outcome["nonzero_pann_observations"] += int(nonzero.sum())
        outcome["return_observations"] += int(returning.sum())
        outcome["return_nonzero_pann_observations"] += int((returning & nonzero).sum())

    recorded_samples = set()
    outcome["observation_sources_unavailable"] = []
    for relative in binaries:
        path = ROOT / relative
        if path.name.startswith("rollout-") and path.suffix == ".npz":
            if not path.is_file():
                outcome["observation_sources_unavailable"].append(str(path))
                continue
            first, last = map(int, path.stem.split("-")[1:])
            sample_ids = np.arange(first, last + 1)
            with np.load(path) as data:
                observations(data["obs"][sample_ids > initial_transitions])
            recorded_samples.update(sample_ids.tolist())
    if latest_resume and not (ROOT / latest_resume).is_file():
        outcome["observation_sources_unavailable"].append(str(ROOT / latest_resume))
    elif latest_resume:
        import torch
        saved = torch.load(ROOT / latest_resume, map_location="cpu", weights_only=False)
        tail = saved.get("buffer", [])
        for sample_id, row in enumerate(tail, saved["samples"] - len(tail) + 1):
            if sample_id > initial_transitions and sample_id not in recorded_samples:
                observations(row["obs"].numpy())
    return outcome


def learning_summary(source, initial_transitions, final_transitions):
    """Select this job's update records, never its high-frequency transition log."""
    result = dict(source=str(source.relative_to(ROOT)), updates=[], malformed_lines=[],
                  initial_transitions=initial_transitions, final_transitions=final_transitions)
    if not source.is_file():
        result["status"] = "unavailable"
        return result
    with source.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                samples = row["cumulative_samples"]
            except (ValueError, KeyError, TypeError):
                result["malformed_lines"].append(number)
                continue
            if initial_transitions < samples <= final_transitions:
                result["updates"].append(row)
    result["status"] = "saved"
    return result


def redact(text):
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "https_proxy", "HTTPS_PROXY"):
        if os.environ.get(key):
            text = text.replace(os.environ[key], "[redacted]")
    return text


def compact_records(jobdir, snapshots, small, episodes, job, evaluation, samples, index):
    """Copy named small records and episode JSON, with bounded diagnostic tails."""
    destination = small / "jobs" / jobdir.name
    staged, retained, copied = set(), [], []
    maximum_json_bytes = 2 * 1024 * 1024

    def keep(source, target, limited=False):
        if not source.is_file():
            return None
        size = source.stat().st_size
        if limited and size > maximum_json_bytes:
            retained.append(dict(path=str(source.relative_to(ROOT)), size=size,
                                 reason="large_json_retained_local"))
            return None
        if source != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        staged.add(target)
        copied.append(dict(source=str(source.relative_to(ROOT)),
                           git_path=str(target.relative_to(ROOT)), size=size))
        return str(target.relative_to(ROOT))

    # Explicit names, not arbitrary .jsonl files, logs or a recursive job archive.
    for filename in (
            "job.json", "execution-contract.json", "environment.json", "evaluation.json",
            "run-job.sh", "pip-packages.txt", "gpu-runtime.txt",
            "training-exit-code.txt", "training-finished-at.txt",
            "evaluation-exit-code.txt", "evaluation-finished-at.txt"):
        keep(jobdir / filename, destination / filename)
    for filename in (
            "training-contract.json", "stream-identity.json", "collection.json",
            "stopped.json", "progress.json", "training-deadline.json"):
        keep(snapshots / filename, destination / "training" / filename)
    for source in sorted(snapshots.glob("episode-*.json")):
        keep(source, destination / "training" / source.name)
    learning = snapshots / "learning.json"
    if learning.is_file():
        keep(learning, destination / "training" / learning.name)
    else:
        # Old finish snapshots contain the original learning.jsonl.
        learning = destination / "training" / "learning.json"
        write(learning, learning_summary(snapshots / "learning.jsonl",
                                        job["initial_transitions"], samples))
        staged.add(learning)
    identity = ROOT / "work/r11-frozen" / job["controller_build"] / "identity.json"
    keep(identity, destination / "controller-identity.json")

    known = {row["attempt"]: row for row in index.get("attempts", [])}
    failures = index.get("initialization_failures", [])
    ledger = dict(job=jobdir.name, training_exit_code=job.get("training_exit_code"),
                  evaluation=evaluation, attempts=[])
    unresolved = []
    for episode in episodes:
        result_path = small / (episode.name + ".json")
        result = read(result_path) or read(episode / "result.json")
        row = dict(attempt=episode.name, local_directory=str(episode.relative_to(ROOT)),
                   result_available=bool(result), index_entry=known.get(episode.name),
                   initialization_failures=[x for x in failures if x.get("attempt") == episode.name],
                   compact_files=[])
        for source in sorted(small.glob(episode.name + "*.json")):
            saved = keep(source, source)
            if saved:
                row["compact_files"].append(saved)
        for source in sorted(episode.rglob("*.json")):
            # Per-replan packets are dense raw traces despite their JSON suffix.
            if source.name == "reference-packets.json":
                retained.append(dict(path=str(source.relative_to(ROOT)), size=source.stat().st_size,
                                     reason="dense_reference_trace_retained_local"))
                continue
            # The normal result already exists in small; retain a standalone
            # result.json when an interrupted/alternate writer only saved it here.
            if source == episode / "result.json" and result_path.is_file():
                continue
            saved = keep(source, destination / "episodes" / episode.name / source.relative_to(episode),
                         limited=source.name != "result.json")
            if saved:
                row["compact_files"].append(saved)
        for key in ("status", "success", "reason", "error", "simulation_s", "stage_status"):
            if key in result:
                row[key] = result[key]
        if not result:
            row["result_status"] = "unavailable_see_retained_initialization_or_job_error"
            unresolved.append(episode)
        ledger["attempts"].append(row)
    # Policy-side evaluation JSON may contain useful identities; raw transitions
    # and learned weights remain under the recorded local evaluation directory.
    for source in sorted((jobdir / "evaluation-policy").rglob("*.json")):
        keep(source, destination / source.relative_to(jobdir), limited=True)

    tails = []
    logs = [jobdir / "train.log", jobdir / "evaluation.log"]
    logs += [ROOT / row["log"] for row in evaluation.get("stages", []) if row.get("log")]
    logs += [episode / "agent-server.log" for episode in unresolved]
    for source in dict.fromkeys(logs):
        if not source.is_file():
            continue
        size = source.stat().st_size
        with source.open("rb") as handle:
            handle.seek(max(0, size - 8192))
            tail = handle.read().decode("utf-8", errors="replace")
        tails.append(dict(path=str(source.relative_to(ROOT)), original_bytes=size,
                          tail_bytes=min(size, 8192), truncated=size > 8192, text=redact(tail)))
    ledger["bounded_log_tails"] = tails
    ledger_path = destination / "attempt-ledger.json"
    write(ledger_path, ledger)
    staged.add(ledger_path)
    return staged, dict(root=str(destination.relative_to(ROOT)),
                        ledger=str(ledger_path.relative_to(ROOT)), files=copied,
                        large_json_retained_local=retained,
                        learning=str((destination / "training" / "learning.json").relative_to(ROOT)))



def update_parallel_handoff(contract):
    """Keep both job states in the existing six lines and actual contract state."""
    small=ROOT/contract["outputs"]["compact_root"]
    large=ROOT/contract["outputs"]["large_root"]
    clock=read(large/"w38-training-clock.json")
    if not clock:return
    rows={}
    for stream_id,relative in clock["jobs"].items():
        directory=ROOT/relative
        job=read(directory/"job.json")
        stream=large/"training"/f'{job["group"]}-seed{job["seed"]}'
        collection=read(stream/"collection.json")
        progress=read(stream/"progress.json")
        summary=read(small/"jobs"/(directory.name+".json"))
        evaluation=read(directory/"evaluation.json")
        finished=(directory/"training-finished-at.txt").exists()
        rows[stream_id]=dict(job=directory.name,phase_id=job["phase_id"],seed=job["seed"],
            status="training_ended" if finished else job["status"],
            transitions=collection.get("collected_transitions",job["initial_transitions"]),
            updates=collection.get("updates",job["initial_updates"]),
            new_transitions=collection.get("phase_new_transitions",0),
            new_updates=collection.get("phase_new_updates",0),
            tail=collection.get("unupdated_fragment",0),
            invocation_new_transitions=collection.get("invocation_new_transitions"),
            invocation_wall_s=collection.get("invocation_wall_s"),
            excluded=collection.get("excluded_old_version_transitions",145),
            resume=progress.get("checkpoint",job["resume_checkpoint"]),
            last_completed_update=collection.get("last_completed_update"),
            evaluation=evaluation.get("status","pending_training"),
            evaluation_stages=evaluation.get("stages",[]),
            publication=summary.get("publication_status","pending_computation"),
            outcomes=summary.get("training",{}).get("outcomes"),
            job_directory=relative,training_directory=str(stream.relative_to(ROOT)))
    from s1_go2_r11_runtime import index_file
    with index_file(small,contract) as index:
        for sid,row in rows.items():
            entry=index.setdefault("background_jobs",{}).setdefault(row["job"],{})
            entry.update(stream_id=sid,phase_id=row["phase_id"],live_status=row["status"],
                latest_counts=dict(transitions=row["transitions"],updates=row["updates"],
                    new_transitions=row["new_transitions"],new_updates=row["new_updates"],tail=row["tail"]),
                job_directory=row["job_directory"],training_directory=row["training_directory"],
                latest_resume=row["resume"],deadline_unix_s=clock["deadline_unix_s"])
        index.update(current_task=contract["current_task"],protocol_revision=contract["protocol_revision"],
                     w38_distinct_new_transitions=sum(x["new_transitions"] for x in rows.values()),
                     w38_policy_counts={sid:dict(transitions=r["transitions"],updates=r["updates"]) for sid,r in rows.items()})
    current=read(CONFIG)
    running=any(x["status"]=="launched" for x in rows.values())
    sampled=any(x["new_transitions"]>0 for x in rows.values())
    current["status"]=("joint_sampling_running" if sampled else "launched_waiting_for_samples") if running else "computation_ended"
    current["actual_implementation"].update(built=True,training_started=sampled)
    current["reuse"]["active_controller_status"]="built_and_running" if running else "built_computation_ended"
    current["current_window_scope"].update(current_build_status=current["reuse"]["active_controller_status"],
                                          new_controller_build_required=False)
    current["learning"]["controller_version_boundary"]["status"]="forks_launched"

    current["actual_execution"]=dict(task=current["current_task"],controller_build="S5-r27",
        shared_started_unix_s=clock["started_unix_s"],deadline_unix_s=clock["deadline_unix_s"],
        streams=rows,shared_parent_counted_once=True,
        distinct_new_transitions=sum(x["new_transitions"] for x in rows.values()),
        common_elapsed_wall_s=time.time()-clock["started_unix_s"],
        distinct_new_transitions_per_wall_s=sum(x["new_transitions"] for x in rows.values())/(time.time()-clock["started_unix_s"]),
        no_fused_optimizer_or_policy=True,bulk_retained_local=True,deleted_files=0)
    current["current_window_scope"]["status"]=current["status"]
    write(CONFIG,current)
    plan_path=ROOT/"docs/TERRAIN_MEMORY_PLAN.md"
    plan=plan_path.read_text().splitlines()
    first=next(i for i,line in enumerate(plan) if line.startswith("## 当前交接"))
    slots=[i for i in range(first+1,len(plan)) if plan[i].startswith("- ")][:6]
    details=[]
    for sid in ("w38-a","w38-b"):
        r=rows.get(sid)
        details.append(f"- {sid}：" + (f'{r["job"]}，{r["status"]}；新增{r["new_transitions"]}/{r["new_updates"]}，累计{r["transitions"]}/{r["updates"]}，tail={r["tail"]}、excluded={r["excluded"]}；评价={r["evaluation"]}，紧凑Git={r["publication"]}。' if r else "尚未启动。"))
    from datetime import datetime,timezone,timedelta
    end=datetime.fromtimestamp(clock["deadline_unix_s"],timezone(timedelta(hours=8))).isoformat()
    ability=[]
    for sid,r in rows.items():
        o=r["outcomes"]
        ability.append(sid+("尚无最终任务汇总" if not o else
            f'合规去程/返程/整圈={o.get("legal_all_outbound_traversals")}/{o.get("legal_return_traversals")}/{o.get("legal_roundtrips")}，材料加载回合={o.get("material_loaded_episodes")}'))
    handoff=[
        f'- 当前任务：W38_SMOOTH_APPROACH_PARALLEL_JOINT / S5-r27 已真实修改、构建并启动；共同训练绝对截止北京时间{end}，resume不重置。',
        *details,
        "- 任务能力："+ "；".join(ability)+"。每流仅本phase新update评价E30，同case合格才E60；不把训练更新当合规任务能力。",
        "- 交付：两流compact finish/Git串行且使用现有index锁；全部逐回合JSON及失败/零步/截断、job/stream学习与独立评价进Git；权重、rollout、原始数组留D57，未自动渲染/上传/删除。",
        "- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；各job、training目录与有效resume见当前R12契约actual_execution.streams及正常job/index；同父工程分支，不合并optimizer或重复父历史。"]
    for position,line in zip(slots,handoff):plan[position]=line
    plan_path.write_text("\n".join(plan)+"\n")


def current_small_root():
    return read(CONFIG)['outputs']['compact_root']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-dir", type=Path, required=True)
    p.add_argument("--training-exit-code", type=int, required=True)
    p.add_argument("--hours", type=float, default=6,
                   help="Postprocess ceiling only; finish immediately after compact Git delivery.")
    p.add_argument("--deadline-unix", type=float, help="Keep the original absolute postprocess deadline on resume.")
    a = p.parse_args()
    # Serialize compact snapshots, handoff/config and Git; training keeps its own slot.
    publication_lock=(ROOT/current_small_root()/"compact-finish.lock").open("a")
    fcntl.flock(publication_lock,fcntl.LOCK_EX)
    jobdir = (ROOT / a.job_dir).resolve()
    job = read(jobdir / "job.json")
    current = read(CONFIG)
    frozen_path = jobdir / "execution-contract.json"
    contract = read(frozen_path) or current
    small, large = (ROOT / contract["outputs"][key] for key in ("compact_root", "large_root"))
    name = jobdir.name
    started = time.time()
    deadline = a.deadline_unix if a.deadline_unix is not None else started + a.hours * 3600
    summary_path = small / "jobs" / (name + ".json")
    previous = read(summary_path)
    if (previous.get("cleanup", {}).get("status") == "completed"
            and previous.get("publication_status") == "published"):
        # This historical result was already delivered and its arrays deleted.
        # Preserve its conclusion; never reconstruct it from a later live stream.
        print(json.dumps(dict(job=name, publication_status="published",
                              action="existing_published_historical_record_retained",
                              automatic_cleanup_enabled=False,
                              summary=str(summary_path.relative_to(ROOT))), ensure_ascii=False), flush=True)
        return 0
    policy = dict(
        source=str(CONFIG.relative_to(ROOT)) + "#outputs.automatic_publication",
        configured=current.get("outputs", {}).get("automatic_publication", {}),
        effective=dict(delivery_mode="compact_first_local_bulk", bulk_upload_enabled=False,
                       render_all_episodes=False,
                       cleanup=dict(enabled=False, owner="user", status="manual_after_user_backup")),
        execution_contract=str(frozen_path.relative_to(ROOT)),
        scope="Current delivery policy only; frozen training/evaluation/paths/identity stay unchanged.",
        frozen_delivery_authorization_superseded=True)
    stream = large / "training" / f'{job["group"]}-seed{job["seed"]}'
    initial = set(job.get("initial_attempts", []))
    evaluation = read(jobdir / "evaluation.json") or dict(
        status="unavailable" if job.get("evaluation_command") else "not_scheduled_for_job",
        reason="See retained evaluation.log and exit code" if job.get("evaluation_command") else None,
        attempts=[], stages=[])
    episodes = ([large / x for x in previous["attempts"]] if "attempts" in previous else
                sorted(x for x in large.glob(f'train-{job["group"]}-s{job["seed"]}-ep*')
                       if x.is_dir() and (x.name not in initial if "initial_attempts" in job
                                          else x.stat().st_mtime >= job["started_unix_s"])))
    episodes = sorted({x.name: x for x in episodes + [large / x for x in evaluation.get("attempts", [])]}.values())
    binaries = previous.get("training_binary_paths", [
        str(x.relative_to(ROOT)) for x in sorted(stream.rglob("*"))
        if x.is_file() and x.suffix in (".pt", ".npz") and x.stat().st_mtime >= job["started_unix_s"]])

    # Freeze only the named compact records. Existing snapshots, including old
    # bulk-publication snapshots, remain authoritative on a retry.
    snapshots = jobdir / "stream-records"
    if not snapshots.exists():
        snapshots.mkdir()
        sources = [stream / filename for filename in (
            "training-contract.json", "stream-identity.json", "collection.json",
            "stopped.json", "progress.json")]
        sources += sorted(stream.glob("episode-*.json"))
        sources.append(jobdir / "training-deadline.json")
        for source in sources:
            if source.is_file() and source.stat().st_mtime >= job["started_unix_s"]:
                shutil.copy2(source, snapshots / source.name)
        final_samples = read(snapshots / "collection.json").get("collected_transitions", job["initial_transitions"])
        write(snapshots / "learning.json", learning_summary(
            stream / "learning.jsonl", job["initial_transitions"], final_samples))

    stopped = read(snapshots / "stopped.json")
    if stopped.get("invocation_started_unix_s", 0) < job["started_unix_s"]:
        stopped = {}
    collection = read(snapshots / "collection.json")
    progress = read(snapshots / "progress.json")
    samples = stopped.get("actual_transitions", collection.get("collected_transitions", job["initial_transitions"]))
    updates = stopped.get("updates", collection.get("updates", job["initial_updates"]))
    training_exit = job.get("budget_extension", {}).get("continuation_exit_code", a.training_exit_code)
    training = dict(actual_transitions=samples, updates=updates,
                    target_transitions=job["target_transitions"], target_updates=job["target_updates"],
                    target_reached=bool(stopped and samples >= job["target_transitions"] and updates >= job["target_updates"]),
                    stream_id=job.get("stream_id"),
                    invocation_wall_s=stopped.get("invocation_wall_s",collection.get("invocation_wall_s")),
                    invocation_new_transitions=stopped.get("new_transitions",collection.get("invocation_new_transitions")),
                    initial_transitions=job["initial_transitions"], new_transitions=samples-job["initial_transitions"],
                    initial_updates=job["initial_updates"], new_updates=updates-job["initial_updates"],
                    unupdated_fragment=stopped.get("unupdated_fragment", collection.get("unupdated_fragment")),
                    reason=stopped.get("reason"), last_stage_status=stopped.get("last_stage_status"),
                    phase_id=collection.get("phase_id"), phase_stop=stopped.get("phase_stop"),
                    excluded_old_version_transitions=collection.get("excluded_old_version_transitions"),
                    controller_migration=collection.get("controller_migration"),
                    last_completed_update=collection.get("last_completed_update"),
                    last_episode_reason=stopped.get("last_episode_reason"))
    try:
        training["outcomes"] = training_outcomes(
            [x for x in episodes if x.name.startswith("train-")], small, binaries,
            progress.get("checkpoint"), job["initial_transitions"])
    except Exception as error:
        # Optional aggregate reads must not block delivery of the original JSON.
        # An unavailable aggregate is not a zero or a successful physical result.
        training["outcomes"] = dict(
            status="unavailable", error=redact(f"{type(error).__name__}: {error}")[-4000:],
            source="Retained per-episode JSON and local rollout/resume observation summary",
            recovery="Inspect the original compact episode records and retained local sources.")

    job.update(status="computation_ended",training_exit_code=training_exit,
               evaluation_status=evaluation["status"])
    write(jobdir/"job.json",job)
    from s1_go2_r11_runtime import index_file
    with index_file(small,contract) as locked_index:
        index_snapshot=copy.deepcopy(locked_index)
    staged, evidence = compact_records(
        jobdir, snapshots, small, episodes, dict(job, training_exit_code=training_exit),
        evaluation, samples, index_snapshot)
    policy_path = small / "jobs" / name / "delivery-policy.json"
    write(policy_path, policy)
    staged.add(policy_path)
    latest_resume = progress.get("checkpoint")
    local = dict(
        status="retained_local", upload_status="not_requested", backup_owner="user",
        deletion="User exports to their storage disk, then centrally deletes after backup.",
        repository_root=str(ROOT), job_directory=str(jobdir.relative_to(ROOT)),
        shared_training_directory=str(stream.relative_to(ROOT)),
        episode_directories=[str(x.relative_to(ROOT)) for x in episodes],
        training_binary_paths=binaries,
        parent_resume_checkpoint=job["resume_checkpoint"],
        latest_resume=dict(path=latest_resume, samples=progress.get("samples"),
                           updates=progress.get("updates"), phase_id=progress.get("phase_id"),
                           identity_source=str((snapshots / "progress.json").relative_to(ROOT))),
        last_completed_update=training.get("last_completed_update"),
        availability="No bulk backup or upload is claimed; historical user/deletion state is not inferred.")
    summary = dict(
        job=name, stream_id=job.get("stream_id"), phase_id=job["phase_id"], current_task=job.get("current_task", "S1"),
        source_commit=job["source_commit"], controller_build=job["controller_build"],
        training_exit_code=training_exit, budget_extension=job.get("budget_extension"),
        training_started_unix_s=job["started_unix_s"], training_deadline_unix_s=job["deadline_unix_s"],
        postprocess_started_unix_s=started, deadline_unix_s=deadline,
        training=training, evaluation=evaluation, attempts=[x.name for x in episodes],
        training_binary_paths=binaries, delivery_policy=policy, compact_evidence=evidence,
        publication_scope="compact_git", publication_status="published", errors=[],
        bulk_retention=local, render_status="not_requested", pending_assets=[], render_pending=[],
        cleanup=dict(enabled=False, owner="user", status="manual_after_user_backup", deleted_files=0),
        git_delivery="Prepared for this commit; presence on GitHub records compact delivery only. "
                     "A failed push remains pending locally. Bulk data is not backed up by this push.")
    # Preserve real historical destinations and outcomes, without registering
    # any newly unselected local file as a Release publication obligation.
    for key in ("release_url", "release_urls", "manifest", "publication_resume", "historical_bulk_publication"):
        if key in previous:
            summary[key] = previous[key]
    if previous and previous.get("publication_scope") != "compact_git":
        summary["historical_bulk_publication"] = {
            key: previous[key] for key in (
                "publication_status", "assets_published", "assets_expected", "pending_assets",
                "render", "render_pending", "videos", "cleanup", "errors") if key in previous}

    def save_summary():
        write(summary_path, summary)
        with index_file(small,contract) as index:
            index.setdefault("background_jobs", {})[name] = dict(
                stream_id=job.get("stream_id"),phase_id=job["phase_id"],
                summary=str(summary_path.relative_to(ROOT)), training=training, evaluation=evaluation,
                publication_scope="compact_git", publication_status=summary["publication_status"],
                bulk_retention=summary["bulk_retention"], cleanup=summary["cleanup"])

    save_summary()
    plan_path = ROOT / "docs/TERRAIN_MEMORY_PLAN.md"
    plan = plan_path.read_text().splitlines()
    first = next(i for i, line in enumerate(plan) if line.startswith("## 当前交接"))
    slots = [i for i in range(first + 1, len(plan)) if plan[i].startswith("- ")][:6]
    eval_text = "，".join(f"{x['level']}={x['stage_status']}" for x in evaluation.get("stages", [])) or evaluation["status"]
    outcomes = training["outcomes"]
    handoff = [
        f"- 当前任务：{summary['current_task']}后台作业{name}已结束；训练达标={training['target_reached']}，评价={evaluation['status']}，紧凑Git发布={summary['publication_status']}；大文件留数据盘，用户备份后手动删除。",
        f"- 训练：本job新增{training['new_transitions']}条/{training['new_updates']}更新，累计{samples}/{updates}，旧尾excluded={training['excluded_old_version_transitions']}；停止原因={training.get('reason')}；评价独立结果{eval_text}。",
        f"- 已完成：保留本job全部{len(episodes)}个attempt的可用小JSON、配置、学习/评价记录及失败/零步/中断清单；控制版{job['controller_build']}；不等待视频或大文件上传。",
        f"- 任务表现：最大向前位移{outcomes.get('max_forward_displacement_m')}m、材料加载{outcomes.get('material_loaded_episodes')}回合、合规去程{outcomes.get('legal_all_outbound_traversals')}、合规返程{outcomes.get('legal_return_traversals')}、合规整圈{outcomes.get('legal_roundtrips')}、侧出{outcomes.get('all_passage_side_exits')}；末次原物理原因{training.get('last_episode_reason')}。",
        "- 下一步：先查看本job实际结果以继续研究；紧凑push失败仅普通重试；具体诊断再选取已有视频/数组，不为交付重跑物理，不自动上传或删除大文件。",
        f"- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；摘要{summary_path.relative_to(ROOT)}；checkpoint及job/episode/shared training本地位置见bulk_retention，既有Release历史保留。",
    ]
    if outcomes.get("status") == "unavailable":
        handoff[3] = "- 任务表现：自动汇总不可用；具体错误见job.training.outcomes.error，保留全部可用逐回合JSON，不把缺失指标记为零或成功。"
    for position, line in zip(slots, handoff):
        plan[position] = line
    plan_path.write_text("\n".join(plan) + "\n")
    if job.get("stream_id","").startswith("w38-"):
        update_parallel_handoff(contract)
        staged.add(CONFIG)
    staged.update((summary_path, small / "index.json", plan_path))
    paths = sorted(str(path.relative_to(ROOT)) for path in staged if path.is_file())

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
                              timeout=max(1, min(120, deadline - time.time())))

    try:
        # Stage the shared index under its existing lock; release it before network I/O.
        with (small/"index.lock").open("a") as index_lock:
            fcntl.flock(index_lock,fcntl.LOCK_EX)
            git("add", "--", *paths)
            changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=ROOT).returncode
            if changed:
                git("commit", "--only", "-m", f"Record compact S1 background job {name}", "--", *paths)
        git("-c", "pack.window=0", "push", "--no-thin", "origin", "HEAD:main")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        summary["publication_status"] = "publication_pending"
        detail = getattr(error, "stdout", "") or ""
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        summary["errors"].append(dict(stage="git", error=redact(str(error)), output=redact(detail)[-4000:]))
        save_summary()
        if job.get("stream_id","").startswith("w38-"):
            update_parallel_handoff(contract)
        elif slots:
            plan[slots[0]] = handoff[0].replace("Git发布=published", "Git发布=publication_pending")
            plan_path.write_text("\n".join(plan) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(dict(job=name, training=training, evaluation=evaluation,
                          publication_scope="compact_git", publication_status=summary["publication_status"],
                          bulk_retention=summary["bulk_retention"], cleanup=summary["cleanup"],
                          summary=f"https://github.com/{REPO}/blob/main/{summary_path.relative_to(ROOT)}"),
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

