#!/usr/bin/env python3
"""Render and publish the actual outputs of one stopped R12 tmux job."""
import argparse
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-dir", type=Path, required=True)
    p.add_argument("--training-exit-code", type=int, required=True)
    p.add_argument("--hours", type=float, default=4)
    a = p.parse_args()
    jobdir = (ROOT / a.job_dir).resolve()
    job = read(jobdir / "job.json")
    contract = read(CONFIG)
    small, large = (ROOT / contract["outputs"][k] for k in ("compact_root", "large_root"))
    name = jobdir.name
    started = time.time()
    deadline = started + a.hours * 3600
    work_deadline = max(started, deadline - 120)
    summary_path = small / "jobs" / (name + ".json")
    manifest_path = small / "jobs" / (name + "-assets.json")
    previous = read(summary_path)
    records = small / "records"
    records.mkdir(parents=True, exist_ok=True)
    stream = large / "training" / f'{job["group"]}-seed{job["seed"]}'
    initial = set(job.get("initial_attempts", []))
    episodes = ([large / x for x in previous["attempts"]] if "attempts" in previous else
                sorted(x for x in large.glob(f'train-{job["group"]}-s{job["seed"]}-ep*')
                       if x.is_dir() and (x.name not in initial if "initial_attempts" in job
                                          else x.stat().st_mtime >= job["started_unix_s"])))
    binaries = previous.get("training_binary_paths", [str(x.relative_to(ROOT)) for x in sorted(stream.rglob("*"))
                            if x.is_file() and x.suffix in (".pt", ".npz") and x.stat().st_mtime >= job["started_unix_s"]])
    summary = dict(job=name, source_commit=job["source_commit"], controller_build=job["controller_build"],
                   training_exit_code=a.training_exit_code, postprocess_started_unix_s=started,
                   deadline_unix_s=deadline, work_deadline_unix_s=work_deadline,
                   publication_status="publication_pending", render=[], errors=[],
                   attempts=[x.name for x in episodes], training_binary_paths=binaries,
                   manifest=str(manifest_path.relative_to(ROOT)), release_url=f'https://github.com/{REPO}/releases/tag/{contract["outputs"]["release_tag"]}')
    write(summary_path, summary)

    def command(argv, log):
        try:
            with log.open("a") as output:
                output.write(f"\nStarted {time.time()}: {argv[1]}\n")
                output.flush()
                r = subprocess.run(argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   timeout=max(1, work_deadline - time.time()))
            return None if r.returncode == 0 else f"exit_code={r.returncode}; see {log.relative_to(ROOT)}"
        except (OSError, subprocess.TimeoutExpired) as error:
            with log.open("a") as output:
                output.write("\n" + str(error) + "\n")
            return str(error)

    # Freeze small records before rendering; a retry never captures a later job's stream state.
    snapshots = jobdir / "stream-records"
    if not snapshots.exists():
        snapshots.mkdir()
        for source in sorted(stream.rglob("*")):
            if source.is_file() and source.suffix not in (".pt", ".npz") and source.stat().st_mtime >= job["started_unix_s"]:
                target = snapshots / source.relative_to(stream)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    known = {x["attempt"]: x for x in read(small / "index.json").get("attempts", [])}
    for episode in episodes:
        row = dict(attempt=episode.name)
        required = [episode / "trajectory.npz", episode / "model.mjb", small / (episode.name + ".json")]
        if not all(x.exists() for x in required):
            row.update(status="unavailable_no_physics" if not known.get(episode.name, {}).get("physical_started") else "failed",
                       reason="No complete renderable trajectory/model/result; retained available files without rerunning physics.")
        else:
            video = read(episode / "video-asset.json")
            retained = episode / Path(video.get("path", "missing-video")).name
            if video and retained.is_file() and retained.stat().st_size == video.get("size"):
                row["status"] = "existing"
            elif time.time() >= work_deadline:
                row.update(status="pending_deadline", reason="Postprocess deadline reached before rendering.")
            else:
                log = records / f"{name}__render__{episode.name}.txt"
                error = command([sys.executable, "scripts/render_s1_go2_r12.py", "--attempt", episode.name], log)
                row.update(status="failed" if error else "rendered", log=str(log.relative_to(ROOT)))
                if error:
                    row["reason"] = error
        summary["render"].append(row)
        write(summary_path, summary)

    assets_by_name = {x["name"]: x for x in read(manifest_path)} if manifest_path.exists() else {}

    def asset(path, remote_name, category):
        assets_by_name.setdefault(remote_name, dict(path=str(path.relative_to(ROOT)), name=remote_name, category=category))

    for episode in episodes:
        render_status = next(x["status"] for x in summary["render"] if x["attempt"] == episode.name)
        for file in sorted(episode.rglob("*")):
            if file.is_file():
                remote_name = episode.name + "__" + file.relative_to(episode).as_posix().replace("/", "__")
                # Incomplete rendering is retained locally, then replaced by a successful replay on retry.
                if render_status not in ("rendered", "existing") and (file.suffix == ".mp4" or file.name == "video-asset.json"):
                    assets_by_name.pop(remote_name, None)
                    continue
                asset(file, remote_name, "episode")
    for relative in binaries:
        file = ROOT / relative
        asset(file, name + "__training__" + stream.name + "__" + file.relative_to(stream).as_posix().replace("/", "__"), "training")
    for file in sorted(jobdir.rglob("*")):
        if file.is_file():
            asset(file, name + "__" + file.relative_to(jobdir).as_posix().replace("/", "__"), "job_record")
    # A newly built controller is evidence for this job; retain its executable
    # and ordinary build record alongside the already versioned source patch.
    controller = ROOT / "work/r11-frozen" / job["controller_build"]
    for filename in ("agent_server", "identity.json", "build.log"):
        file = controller / filename
        if file.is_file():
            asset(file, "controller__" + job["controller_build"] + "__" + filename, "controller")
    assets = list(assets_by_name.values())
    write(manifest_path, assets)
    for action in ("upload", "finish"):
        log = records / f"{name}__{action}.txt"
        error = command([sys.executable, "scripts/publish_s1_go2_r12.py", action, "--server-auth",
                         "--manifest", str(manifest_path), "--deadline-unix", str(work_deadline)], log)
        if error:
            summary["errors"].append(dict(stage=action, error=error))

    idx = read(small / "index.json")
    registered = {x["name"]: x for x in idx.get("publication", {}).get("assets", [])}
    pending = [x["path"] for x in assets if registered.get(x["name"], {}).get("status") not in ("uploaded", "git_recorded")]
    missing_videos = [x for x in summary["render"] if x["status"] in ("failed", "pending_deadline")]
    stopped = read(snapshots / "stopped.json")
    if stopped.get("invocation_started_unix_s", 0) < job["started_unix_s"]:
        stopped = {}
    collection = read(snapshots / "collection.json")
    samples = stopped.get("actual_transitions", collection.get("collected_transitions", job.get("initial_transitions")))
    updates = stopped.get("updates", collection.get("updates", job.get("initial_updates")))
    summary.update(training=dict(actual_transitions=samples, updates=updates,
                       target_transitions=job["target_transitions"], target_updates=job["target_updates"],
                       target_reached=bool(stopped and samples >= job["target_transitions"] and updates >= job["target_updates"]),
                       reason=stopped.get("reason"), last_stage_status=stopped.get("last_stage_status"),
                       last_episode_reason=stopped.get("last_episode_reason")),
                   pending_assets=pending, render_pending=missing_videos,
                   assets_published=sum(x["path"] not in pending for x in assets), assets_expected=len(assets),
                   videos=[dict(attempt=x["attempt"], url=registered.get(x["attempt"] + "__full-mission-material-overlay.mp4", {}).get("url"))
                           for x in summary["render"] if x["status"] in ("rendered", "existing")])
    summary["release_urls"] = {
        tag: f"https://github.com/{REPO}/releases/tag/{tag}"
        for tag in sorted({registered.get(x["name"], {}).get("release_tag", contract["outputs"]["release_tag"])
                           for x in assets})
    }
    complete = not pending and not missing_videos and not summary["errors"]
    summary.update(publication_status="published" if complete else "publication_pending",
                   git_delivery="This report is prepared for this commit; its presence on GitHub records successful delivery. A push failure is retained locally as pending.")

    def save_summary():
        write(summary_path, summary)
        current = read(small / "index.json")
        current.setdefault("background_jobs", {})[name] = dict(
            summary=str(summary_path.relative_to(ROOT)), training=summary.get("training"),
            publication_status=summary["publication_status"], pending_assets=summary.get("pending_assets", []),
            render_pending=summary.get("render_pending", []))
        publication = current.setdefault("publication", {})
        pending_jobs = publication.setdefault("pending_jobs", {})
        if summary["publication_status"] == "publication_pending":
            pending_jobs[name] = str(summary_path.relative_to(ROOT))
            publication["status"] = "publication_pending"
        else:
            pending_jobs.pop(name, None)
            if not pending_jobs and not publication.get("pending_paths"):
                publication["status"] = "published"
        write(small / "index.json", current)

    save_summary()
    plan_path = ROOT / "docs/TERRAIN_MEMORY_PLAN.md"
    plan = plan_path.read_text().splitlines()
    first = next(i for i, line in enumerate(plan) if line.startswith("## 当前交接"))
    slots = [i for i in range(first + 1, len(plan)) if plan[i].startswith("- ")][:6]
    handoff = [
        f"- 当前任务：W34后台作业{name}已结束训练与自动后处理；训练达标={summary['training']['target_reached']}，发布={summary['publication_status']}。",
        f"- 当前状态：本组实际累计{samples}条/{updates}次更新，目标{job['target_transitions']}/{job['target_updates']}；训练记录不代表正式E30资格。",
        f"- 已完成：本job保留{len(episodes)}个新attempt，视频可用{len(summary['videos'])}，资产已发布{summary['assets_published']}/{len(assets)}；控制版{job['controller_build']}。",
        f"- 当前问题：最后物理原因{stopped.get('last_episode_reason')}；待传{len(pending)}项，待处理视频{len(missing_videos)}项，具体错误见本job摘要。",
        "- 下一步：先查看本job真实路线/支撑结果；有发布待办则从现有文件续传，禁止为补视频重跑物理；再决定继续学习或正式阶梯。",
        f"- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；摘要{summary_path.relative_to(ROOT)}；原Release及本job续传Release见摘要release_urls，用户电脑不另归档。",
    ]
    for position, line in zip(slots, handoff):
        plan[position] = line
    plan_path.write_text("\n".join(plan) + "\n")

    staged = {summary_path, manifest_path, small / "index.json", small / "training-deadline.json", plan_path}
    staged.update(records.glob(name + "__*"))
    for episode in episodes:
        staged.update(small.glob(episode.name + "*.json"))
        staged.update(records.glob(episode.name + "__*"))
    paths = sorted(str(x.relative_to(ROOT)) for x in staged if x.is_file())

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
                              timeout=max(1, min(120, deadline - time.time())))

    try:
        git("add", "--", *paths)
        changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=ROOT).returncode
        if changed:
            git("commit", "--only", "-m", f"Record S1 background job {name} and its publication", "--", *paths)
        git("push", "origin", "HEAD:main")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        summary["publication_status"] = "publication_pending"
        detail = getattr(error, "stdout", "") or ""
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        for key in ("GH_TOKEN", "GITHUB_TOKEN", "https_proxy", "HTTPS_PROXY"):
            if os.environ.get(key):
                detail = detail.replace(os.environ[key], "[redacted]")
        summary["errors"].append(dict(stage="git", error=str(error), output=detail[-4000:]))
        save_summary()
        if slots:
            plan[slots[0]] = handoff[0].replace("发布=published", "发布=publication_pending")
            plan_path.write_text("\n".join(plan) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(dict(job=name, training=summary["training"], publication_status=summary["publication_status"],
                          summary=f"https://github.com/{REPO}/blob/main/{summary_path.relative_to(ROOT)}"), ensure_ascii=False), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
