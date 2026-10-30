#!/usr/bin/env python3
"""Run R12 training, rendering and GitHub publication in one detached tmux job."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/terrain_memory/s1-go2-local-readiness-r12.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", required=True, help="Latest actual checkpoint of the active stream.")
    p.add_argument("--session", default="s1-joint")
    p.add_argument("--hours", type=float, help="Training budget; defaults to the execution contract.")
    p.add_argument("--postprocess-hours", type=float, default=4,
                   help="Separate rendering/publication budget after training.")
    p.add_argument("--network-mode", choices=["academic", "clash", "direct"], default="academic")
    a = p.parse_args()
    contract = json.loads(CONFIG.read_text())
    active = contract["learning"]["active_streams"][0]
    seconds = (a.hours * 3600 if a.hours is not None
               else contract["execution"]["background_job"]["default_maximum_wall_s"])
    if seconds <= 0 or a.postprocess_hours <= 0:
        p.error("Training and postprocess budgets must be positive")
    # Require the persistent server login, independent of this SSH client's token.
    auth_env = dict(os.environ)
    auth_env.pop("GH_TOKEN", None)
    auth_env.pop("GITHUB_TOKEN", None)
    access = subprocess.check_output([
        "bash", "-c",
        'set -e; source scripts/autodl_network.sh "$1"; '
        'gh auth setup-git --hostname github.com >/dev/null; '
        'git var GIT_AUTHOR_IDENT >/dev/null; '
        'gh api repos/kaiwen123-yang/MCP-RL/releases/tags/tm-s1-go2-local-readiness-r12 >/dev/null; '
        'gh api repos/kaiwen123-yang/MCP-RL --jq .permissions.push',
        "bash", a.network_mode,
    ], cwd=ROOT, env=auth_env, text=True, timeout=60)
    if access.strip().splitlines()[-1] != "true":
        raise RuntimeError("The server GitHub login needs push access to MCP-RL.")
    started = time.time()
    deadline = started + seconds
    job_name = "job-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / contract["outputs"]["large_root"] / job_name
    directory.mkdir(parents=True, exist_ok=False)
    command = [
        "bash", "scripts/run_s1_go2_r12_training.sh",
        "--group", active["group"], "--seed", str(active["seed"]),
        "--resume", a.resume, "--background-job",
        "--job-start-unix", str(started), "--deadline-unix", str(deadline),
    ]
    log = directory / "train.log"
    script = directory / "run-job.sh"
    postprocess = [
        "python", "-u", "scripts/finish_s1_go2_r12_job.py",
        "--job-dir", str(directory), "--hours", str(a.postprocess_hours),
    ]
    # Close training output before rendering/uploading the immutable job files.
    script.write_text("\n".join([
        "#!/usr/bin/env bash",
        "set -e",
        "cd " + shlex.quote(str(ROOT)),
        "source scripts/autodl_d57_env.sh",
        "source scripts/autodl_network.sh " + shlex.quote(a.network_mode),
        "unset GH_TOKEN GITHUB_TOKEN",
        "set +e",
        shlex.join(command) + " > " + shlex.quote(str(log)) + " 2>&1",
        "training_exit=$?",
        "set -e",
        "printf '%s\\n' \"$training_exit\" > " + shlex.quote(str(directory / "training-exit-code.txt")),
        "date -u '+%Y-%m-%dT%H:%M:%SZ' > " + shlex.quote(str(directory / "training-finished-at.txt")),
        shlex.join(postprocess) + ' --training-exit-code "$training_exit"',
        "",
    ]))
    collection_path = ROOT / contract["outputs"]["large_root"] / "training" / f'{active["group"]}-seed{active["seed"]}' / "collection.json"
    collection = json.loads(collection_path.read_text()) if collection_path.exists() else {}
    record = dict(
        session=a.session, job_directory=str(directory.relative_to(ROOT)),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        controller_build=contract["reuse"]["active_controller_build"],
        group=active["group"], seed=active["seed"], resume_checkpoint=a.resume,
        started_unix_s=started, deadline_unix_s=deadline,
        initial_transitions=collection.get("collected_transitions"), initial_updates=collection.get("updates"),
        target_transitions=contract["learning"]["target_cumulative_transitions_per_stream"],
        target_updates=contract["learning"]["target_cumulative_updates_per_stream"],
        log=str(log.relative_to(ROOT)), command=command, status="prepared",
        network_mode=a.network_mode, postprocess_hours=a.postprocess_hours,
        initial_attempts=sorted(p.name for p in (ROOT / contract["outputs"]["large_root"]).glob(
            f'train-{active["group"]}-s{active["seed"]}-ep*') if p.is_dir()),
        publication_status="automatic_after_training",
        result_summary=f'{contract["outputs"]["compact_root"]}/jobs/{job_name}.json',
    )
    metadata = directory / "job.json"
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    # The fixed default session name also covers rendering and publication.
    subprocess.run(["tmux", "new-session", "-d", "-s", a.session,
                    "-c", str(ROOT), "bash", str(script)], check=True)
    record["status"] = "launched"
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print("Detach leaves training, rendering and publication running in this tmux session.")
    print("Inspect the job result summary for training outcome, Release URLs and publication errors.")


if __name__ == "__main__":
    main()
