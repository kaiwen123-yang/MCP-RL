#!/usr/bin/env python3
"""Push compact evidence, then queue requested raw diagnostics for a stopped job."""
import argparse
import fcntl
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
    # The live-launch copy at the job root is not a final collection record.
    keep(snapshots / "collection.json", destination / "collection.json")
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
    identity = ROOT / "work/r11-frozen" / job.get("controller_artifact", job["controller_build"]) / "identity.json"
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
    if job.get("log"):
        logs.append(ROOT / job["log"])
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
    """Keep both streams in normal job/index records and the existing six lines."""
    small=ROOT/contract["outputs"]["compact_root"]
    large=ROOT/contract["outputs"]["large_root"]
    clock_path=contract["execution"]["background_job"].get("training_clock_path")
    clock=read(ROOT/clock_path if clock_path else large/"w38-training-clock.json")
    if not clock:return
    rows={}
    for stream_id,relative in clock["jobs"].items():
        directory=ROOT/relative
        job=read(directory/"job.json")
        stream=(ROOT/job["training_directory"] if job.get("training_directory") else
                large/"training"/f'{job["group"]}-seed{job["seed"]}')
        collection=read(stream/"collection.json")
        progress=read(stream/"progress.json")
        summary=read(small/"jobs"/(directory.name+".json"))
        if job.get("last_resume",{}).get("at_unix_s",0)>summary.get("postprocess_started_unix_s",0):
            summary={}
        evaluation=read(directory/"evaluation.json")
        finished=(directory/"training-finished-at.txt").exists()
        rows[stream_id]=dict(job=directory.name,phase_id=job["phase_id"],seed=job["seed"],
            controller_build=job["controller_build"],reward_version=job["reward_version"],
            deadline_unix_s=job.get("deadline_unix_s"),
            completion_policy=job.get("completion_policy", "legacy_wall_deadline"),
            controller_artifact=job.get("controller_artifact",job["controller_build"]),
            phase_lineage=collection.get("phase_lineage"),
            status="training_ended" if finished else job["status"],
            transitions=collection.get("collected_transitions",job["initial_transitions"]),
            updates=collection.get("updates",job["initial_updates"]),
            new_transitions=collection.get("phase_new_transitions",0),
            new_updates=collection.get("phase_new_updates",0),
            tail=collection.get("unupdated_fragment",0),
            invocation_new_transitions=collection.get("invocation_new_transitions"),
            invocation_wall_s=collection.get("invocation_wall_s"),
            collection_updated_unix_s=collection.get("updated_unix_s"),
            excluded=collection.get("excluded_old_version_transitions",job.get("planned_excluded_old_version_transitions")),
            resume=progress.get("checkpoint",job["resume_checkpoint"]),
            last_completed_update=collection.get("last_completed_update"),
            evaluation=evaluation.get("status","pending_training"),
            evaluation_stages=evaluation.get("stages",[]),
            publication=summary.get("publication_status","pending_computation"),
            outcomes=summary.get("training",{}).get("outcomes"),
            job_directory=relative,training_directory=str(stream.relative_to(ROOT)))
    from s1_go2_r11_runtime import index_file
    phase=next(iter(clock["jobs"])).split("-")[0]
    collected_at=[r["collection_updated_unix_s"] for r in rows.values() if r["collection_updated_unix_s"] is not None]
    elapsed=max(collected_at)-clock["started_unix_s"] if collected_at else None
    with index_file(small,contract) as index:
        for sid,row in rows.items():
            entry=index.setdefault("background_jobs",{}).setdefault(row["job"],{})
            entry.update(stream_id=sid,phase_id=row["phase_id"],live_status=row["status"],
                latest_counts=dict(transitions=row["transitions"],updates=row["updates"],
                    new_transitions=row["new_transitions"],new_updates=row["new_updates"],tail=row["tail"]),
                phase_lineage=row["phase_lineage"],controller_build=row["controller_build"],
                job_directory=row["job_directory"],training_directory=row["training_directory"],
                latest_resume=row["resume"],deadline_unix_s=row["deadline_unix_s"],
                completion_policy=row["completion_policy"])
        index.update(current_task=contract["current_task"],protocol_revision=contract["protocol_revision"])
        index[phase+"_distinct_new_transitions"]=sum(x["new_transitions"] for x in rows.values())
        index[phase+"_policy_counts"]={sid:dict(transitions=r["transitions"],updates=r["updates"]) for sid,r in rows.items()}
        index[phase+"_parallel_state"]=dict(streams=rows,clock=clock_path,
            shared_training_started_unix_s=clock["started_unix_s"],shared_training_deadline_unix_s=clock["deadline_unix_s"],
            measured_training_wall_s=elapsed,
            distinct_new_transitions_per_training_hour=(sum(x["new_transitions"] for x in rows.values())/elapsed*3600 if elapsed and elapsed>0 else None),
            throughput_scope="Ends at latest actual collection update; excludes later evaluation/finish.",
            no_fused_optimizer_or_policy=True,bulk_retained_local=True,deleted_files=0)
    plan_path=ROOT/"docs/TERRAIN_MEMORY_PLAN.md"
    plan=plan_path.read_text().splitlines()
    first=next(i for i,line in enumerate(plan) if line.startswith("## 当前交接"))
    slots=[i for i in range(first+1,len(plan)) if plan[i].startswith("- ")][:6]
    details=[]
    # A job snapshot narrows active_streams to itself. The shared clock plus
    # declared parent map retain both streams, including one not launched yet.
    stream_ids=sorted(set(clock["jobs"]) | set(contract["learning"].get("parent_map",{})))
    for sid in stream_ids:
        r=rows.get(sid)
        details.append(f"- {sid}：" + (f'{r["job"]}，{r["status"]}；新增{r["new_transitions"]}/{r["new_updates"]}，累计{r["transitions"]}/{r["updates"]}，tail={r["tail"]}、excluded={r["excluded"]}；评价={r["evaluation"]}，紧凑Git={r["publication"]}。' if r else "尚未启动。"))
    from datetime import datetime,timezone,timedelta
    end=(datetime.fromtimestamp(clock['deadline_unix_s'],timezone(timedelta(hours=8))).isoformat()
         if clock.get('deadline_unix_s') is not None else None)
    ability=[]
    for sid,r in rows.items():
        o=r["outcomes"]
        ability.append(sid+("尚无最终任务汇总" if not o else
            f'合规去程/返程/整圈={o.get("legal_all_outbound_traversals")}/{o.get("legal_return_traversals")}/{o.get("legal_roundtrips")}，材料加载回合={o.get("material_loaded_episodes")}'))
    handoff=[
        f'- 当前任务：{contract["current_task"]} / {contract["reuse"]["active_controller_build"]}；'
        + (f'共同训练绝对截止北京时间{end}，resume不重置。' if end else
           '后台按各自样本/更新目标结束，无固定墙钟完成期限；原物理时长与技术失败停止保留。'),
        *details,
        "- 任务能力："+ "；".join(ability)+"。每流仅本phase最后实际新update评价E30，同case合格才E60；训练更新不等于合规任务能力。",
        "- 交付：两流compact finish/Git串行，使用原index锁；全部逐回合JSON、2Hz学习转移、12维PANN/value小表与E30关键事件窗先Git；已请求的原始反馈/规划/参考轨迹及必要数组由diagnostics入口后台Release，未选权重/全视频留D57，无自动删除。",
        "- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；各job、training目录与有效resume见正常job/index的两流状态；分别继承自身父分支，optimizer/RNG/归一化不混合。"]
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
    from s1_go2_job_limits import goal_based, request_timeout
    deadline = None if goal_based(contract) else (a.deadline_unix if a.deadline_unix is not None else started + a.hours * 3600)
    summary_path = small / "jobs" / (name + ".json")
    previous = read(summary_path)
    resumed_since_publication=(job.get("last_resume",{}).get("at_unix_s",0)
                               > previous.get("postprocess_started_unix_s",0))
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
        effective=dict(delivery_mode="compact_first_requested_diagnostics_background", bulk_upload_enabled=False,
                       diagnostic_upload_enabled=bool(current.get("outputs", {}).get(
                           "automatic_publication", {}).get("diagnostic_upload_enabled", False)),
                       render_all_episodes=False,
                       cleanup=dict(enabled=False, owner="user", status="manual_after_user_backup")),
        execution_contract=str(frozen_path.relative_to(ROOT)),
        scope="Current delivery policy only; frozen training/evaluation/paths/identity stay unchanged.",
        frozen_delivery_authorization_superseded=True)
    stream = (ROOT / job["training_directory"] if job.get("training_directory") else
              large / "training" / f'{job["group"]}-seed{job["seed"]}')
    initial = set(job.get("initial_attempts", []))
    evaluation = read(jobdir / "evaluation.json") or dict(
        status="unavailable" if job.get("evaluation_command") else "not_scheduled_for_job",
        reason="See retained evaluation.log and exit code" if job.get("evaluation_command") else None,
        attempts=[], stages=[])
    episode_prefix=job.get("episode_prefix",f'train-{job["group"]}-s{job["seed"]}-ep')
    episodes = ([large / x for x in previous["attempts"]]
                if "attempts" in previous and not resumed_since_publication else
                sorted(x for x in large.glob(episode_prefix+"*")
                       if x.is_dir() and (x.name not in initial if "initial_attempts" in job
                                          else x.stat().st_mtime >= job["started_unix_s"])))
    episodes = sorted({x.name: x for x in episodes + [large / x for x in evaluation.get("attempts", [])]}.values())
    prior_binaries=previous.get("training_binary_paths") if not resumed_since_publication else None
    binaries = prior_binaries if prior_binaries is not None else [
        str(x.relative_to(ROOT)) for x in sorted(stream.rglob("*"))
        if x.is_file() and x.suffix in (".pt", ".npz") and x.stat().st_mtime >= job["started_unix_s"]]

    # The explicit starter retained the previous invocation under resume-history
    # before opening a new one. Publication retries reuse the current snapshot.
    snapshots = jobdir / "stream-records"
    if resumed_since_publication and snapshots.exists():
        snapshot_start=read(snapshots/"collection.json").get("invocation_started_unix_s",0)
        if snapshot_start<job["last_resume"]["at_unix_s"]:
            history=jobdir/"resume-history"/("finish-"+str(time.time_ns()))
            history.mkdir(parents=True)
            snapshots.rename(history/"stream-records")
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
                    invocation_started_unix_s=stopped.get("invocation_started_unix_s",collection.get("invocation_started_unix_s")),
                    invocation_wall_s=stopped.get("invocation_wall_s",collection.get("invocation_wall_s")),
                    invocation_new_transitions=stopped.get("new_transitions",collection.get("invocation_new_transitions")),
                    initial_transitions=job["initial_transitions"], new_transitions=samples-job["initial_transitions"],
                    initial_updates=job["initial_updates"], new_updates=updates-job["initial_updates"],
                    unupdated_fragment=stopped.get("unupdated_fragment", collection.get("unupdated_fragment")),
                    reason=stopped.get("reason"), last_stage_status=stopped.get("last_stage_status"),
                    phase_id=collection.get("phase_id"), phase_lineage=collection.get("phase_lineage"),
                    controller_build=job["controller_build"],
                    controller_artifact=contract["reuse"].get("controller_artifact",job["controller_build"]),
                    reward_version=job["reward_version"],
                    phase_stop=stopped.get("phase_stop"),
                    excluded_old_version_transitions=collection.get("excluded_old_version_transitions"),
                    controller_migration=collection.get("controller_migration"),
                    last_completed_update=collection.get("last_completed_update"),
                    last_episode_reason=stopped.get("last_episode_reason"),last_error=stopped.get("last_error"))
    last_update=training["last_completed_update"]
    training["last_update_is_own_current_phase_new_update"]=bool(last_update
        and last_update.get("updates",0)>job["initial_updates"]
        and last_update.get("controller_artifact",last_update.get("controller_build"))==training["controller_artifact"]
        and all(last_update.get(key)==job.get(key) for key in
                ("phase_id","stream_id","controller_build","reward_version")))
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
               evaluation_status=evaluation["status"],
               training_stop_reason=training["reason"],phase_stop=training["phase_stop"],
               controller_artifact=contract["reuse"].get("controller_artifact",job["controller_build"]),
               last_observed_unix_s=collection.get("updated_unix_s"),
               observed_counts={key:collection.get(key) for key in (
                   "collected_transitions","updates","phase_new_transitions","phase_new_updates",
                   "unupdated_fragment","excluded_old_version_transitions")})
    write(jobdir/"job.json",job)
    from s1_go2_r11_runtime import index_file
    with index_file(small,contract) as locked_index:
        index_snapshot=copy.deepcopy(locked_index)
    staged, evidence = compact_records(
        jobdir, snapshots, small, episodes, dict(job, training_exit_code=training_exit),
        evaluation, samples, index_snapshot)
    from export_s1_go2_compact_learning import export_job
    numeric_paths,numeric_evidence=export_job(jobdir,small,large,stream,episodes,
        samples,progress.get("checkpoint"),evaluation)
    staged.update(numeric_paths)
    evidence["numeric_learning"]=numeric_evidence
    from export_s1_go2_event_windows import export_episode
    event_evidence={}
    for stage in evaluation.get("stages",[]):
        if stage.get("level")=="E30" and stage.get("attempt"):
            episode=large/stage["attempt"]
            result=read(small/(episode.name+".json")) or read(episode/"result.json")
            destination=small/"jobs"/name/"numeric-evidence"/"event-windows"/episode.name
            exported=export_episode(episode,destination,result)
            event_summary=exported["summary"]
            event_evidence[episode.name]=dict(
                summary=str((destination/"event-summary.json").relative_to(ROOT)),
                paths=[str(Path(path).relative_to(ROOT)) for path in exported["paths"]],
                status=event_summary.get("status","saved_event_windows"),
                event_times_s={key:(event["time_s"] if event else None)
                               for key,event in event_summary.get("events",{}).items()},
                accepted_violation_count=len(event_summary.get("accepted_violations",[])),
                missing=exported["missing"])
            staged.update(destination.rglob("*.json"))
            staged.update(destination.rglob("*.jsonl"))
    evidence["event_windows"]=event_evidence
    if job["controller_build"] in ("S5-r29","S5-r30","S5-r31"):
        from export_s1_go2_r29_exposure import export_episode as export_exposure
        selected=list(episodes)+[large/x["attempt"] for x in evaluation.get("stages",[]) if x.get("attempt")]
        exposure_paths=[]
        for episode in dict.fromkeys(selected):
            if (episode/"query-exposure.jsonl").exists():
                path=export_exposure(episode,small/"jobs"/name/"numeric-evidence"/"exposure"/episode.name)
                staged.add(path);exposure_paths.append(str(path.relative_to(ROOT)))
        evidence["r29_query_exposure"]=exposure_paths
    policy_path = small / "jobs" / name / "delivery-policy.json"
    write(policy_path, policy)
    staged.add(policy_path)
    latest_resume = progress.get("checkpoint")
    local = dict(
        status="retained_local", upload_status="unselected_bulk_not_requested", backup_owner="user",
        requested_diagnostics="Tracked separately under this job's diagnostics/evidence.json; selected raw arrays are requested uploads, not unrequested bulk.",
        deletion="User exports to their storage disk, then centrally deletes after backup.",
        repository_root=str(ROOT), job_directory=str(jobdir.relative_to(ROOT)),
        training_directory=str(stream.relative_to(ROOT)),
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
        controller_artifact=job["controller_artifact"],last_resume=job.get("last_resume"),
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
    diagnostic_enabled = policy["effective"]["diagnostic_upload_enabled"]
    if diagnostic_enabled:
        summary["diagnostic_publication"] = dict(
            status="tracked_separately", requested=True,
            metadata=str((small / "jobs" / name / "diagnostics" / "evidence.json").relative_to(ROOT)),
            manifest=str((small / "jobs" / name / "diagnostics" / "manifest.json").relative_to(ROOT)),
            entrypoint="scripts/publish_s1_go2_r12_diagnostics.py",
            scope="Original feedback/selected-plan/reference/execution traces and required arrays. Compact publication does not imply raw delivery completed.")
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
        "- 下一步：继续研究并后台上传已请求的原始反馈/选中与参考轨迹及必要数组；状态见同job的diagnostics/evidence.json。紧凑push失败普通重试；不为交付重跑物理，不自动上传未选权重/全视频或删除研究输出。",
        f"- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；摘要{summary_path.relative_to(ROOT)}；checkpoint及job/episode/shared training本地位置见bulk_retention，既有Release历史保留。",
    ]
    if outcomes.get("status") == "unavailable":
        handoff[3] = "- 任务表现：自动汇总不可用；具体错误见job.training.outcomes.error，保留全部可用逐回合JSON，不把缺失指标记为零或成功。"
    for position, line in zip(slots, handoff):
        plan[position] = line
    plan_path.write_text("\n".join(plan) + "\n")
    if job.get("shared_clock_path"):
        update_parallel_handoff(contract)
    staged.update((summary_path, small / "index.json", plan_path))
    paths = sorted(str(path.relative_to(ROOT)) for path in staged if path.is_file())

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
                              timeout=request_timeout(deadline, 120))

    try:
        # Stage the shared index under its existing lock; release it before network I/O.
        with (small/"index.lock").open("a") as index_lock:
            fcntl.flock(index_lock,fcntl.LOCK_EX)
            git("add", "--", *paths)
            changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=ROOT).returncode
            if changed:
                git("commit", "--only", "-m", f"Record compact S1 background job {name}", "--", *paths)
        git("fetch", "origin", "main")
        git("merge", "--no-edit", "origin/main")
        git("-c", "pack.window=0", "push", "--no-thin", "origin", "HEAD:main")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        summary["publication_status"] = "publication_pending"
        detail = getattr(error, "stdout", "") or ""
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        summary["errors"].append(dict(stage="git", error=redact(str(error)), output=redact(detail)[-4000:]))
        save_summary()
        if job.get("shared_clock_path"):
            update_parallel_handoff(contract)
        elif slots:
            plan[slots[0]] = handoff[0].replace("Git发布=published", "Git发布=publication_pending")
            plan_path.write_text("\n".join(plan) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1
    # Raw transfer must not hold the compact/Git lock or delay another stream's
    # result publication. Historical backfill calls the diagnostic tool directly.
    fcntl.flock(publication_lock, fcntl.LOCK_UN)
    publication_lock.close()
    diagnostic_queue = None
    if diagnostic_enabled:
        try:
            diagnostic_command=[sys.executable, str(ROOT / "scripts/publish_s1_go2_r12_diagnostics.py"),
                "queue", "--job-dir", str(jobdir), "--hours", str(a.hours)]
            if goal_based(contract):
                diagnostic_command += ['--no-wall-deadline']
            if job.get("last_resume"):
                # An empty selection would falsely reuse the old published
                # manifest. The uploader retains its original absolute clock.
                diagnostic_known=set(read(small/"jobs"/name/"diagnostics"/"evidence.json").get("attempts",[]))
                for episode in episodes:
                    if episode.name not in diagnostic_known:
                        diagnostic_command += ["--attempt",episode.name]
            queued = subprocess.run(
                diagnostic_command,
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            detail = {}
            for line in reversed(queued.stdout.splitlines()):
                try:
                    detail = json.loads(line)
                    if isinstance(detail, dict):
                        break
                except (ValueError, TypeError):
                    continue
            state = detail.get("status", "queue_pending") if isinstance(detail, dict) else "queue_pending"
            okay = queued.returncode == 0 and state in ("queued", "already_running", "already_published")
            diagnostic_queue = dict(returncode=0 if okay else (queued.returncode or 1),
                                    output=redact(queued.stdout)[-4000:],
                                    complete=okay and state == "already_published",
                                    status=state if okay else "queue_pending")
        except (OSError, subprocess.TimeoutExpired) as error:
            diagnostic_queue = dict(returncode=1, status="queue_pending", complete=False,
                                    output=redact(str(error))[-4000:])
    print(json.dumps(dict(job=name, training=training, evaluation=evaluation,
                          publication_scope="compact_git", publication_status=summary["publication_status"],
                          diagnostic_queue=diagnostic_queue,
                          bulk_retention=summary["bulk_retention"], cleanup=summary["cleanup"],
                          summary=f"https://github.com/{REPO}/blob/main/{summary_path.relative_to(ROOT)}"),
                     ensure_ascii=False), flush=True)
    return diagnostic_queue["returncode"] if diagnostic_queue else 0


if __name__ == "__main__":
    raise SystemExit(main())

