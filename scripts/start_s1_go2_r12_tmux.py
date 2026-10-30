#!/usr/bin/env python3
"""Start the current R12 training stream in a detached AutoDL tmux session."""
import argparse
from datetime import datetime, timezone
import json
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
    p.add_argument("--hours", type=float, help="Job wall-time budget; defaults to the execution contract.")
    a = p.parse_args()
    contract = json.loads(CONFIG.read_text())
    active = contract["learning"]["active_streams"][0]
    seconds = (a.hours * 3600 if a.hours is not None
               else contract["execution"]["background_job"]["default_maximum_wall_s"])
    if seconds <= 0:
        p.error("--hours must be positive")
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
    # All paths/arguments are shell-quoted; no credentials are recorded.
    script.write_text("\n".join([
        "#!/usr/bin/env bash",
        "set -e",
        "cd " + shlex.quote(str(ROOT)),
        "exec > " + shlex.quote(str(log)) + " 2>&1",
        "record_exit() {",
        "  job_exit=$?",
        "  printf '%s\\n' \"$job_exit\" > " + shlex.quote(str(directory / "exit-code.txt")),
        "  date -u '+%Y-%m-%dT%H:%M:%SZ' > " + shlex.quote(str(directory / "finished-at.txt")),
        "}",
        "trap record_exit EXIT",
        "source scripts/autodl_d57_env.sh",
        shlex.join(command),
        "",
    ]))
    record = dict(
        session=a.session, job_directory=str(directory.relative_to(ROOT)),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        controller_build=contract["reuse"]["active_controller_build"],
        group=active["group"], seed=active["seed"], resume_checkpoint=a.resume,
        started_unix_s=started, deadline_unix_s=deadline,
        target_transitions=contract["learning"]["target_cumulative_transitions_per_stream"],
        target_updates=contract["learning"]["target_cumulative_updates_per_stream"],
        log=str(log.relative_to(ROOT)), command=command, status="prepared",
        publication_status="pending_after_training",
    )
    metadata = directory / "job.json"
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    # A fixed default session name prevents accidentally starting this stream twice.
    subprocess.run(["tmux", "new-session", "-d", "-s", a.session,
                    "-c", str(ROOT), "bash", str(script)], check=True)
    record["status"] = "launched"
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print("Detach does not stop this job. Inspect collection.json, learning.jsonl and stopped.json.")
    print("Process exit code alone does not prove the training target or publication is complete.")


if __name__ == "__main__":
    main()
