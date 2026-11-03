#!/usr/bin/env python3
"""Run R12 training, final evaluation and compact Git delivery in one detached tmux job."""
import argparse
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/terrain_memory/s1-go2-local-readiness-r12.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", required=True, help="Latest actual checkpoint of the active stream.")
    p.add_argument("--session", default="s1-joint")
    p.add_argument("--hours", type=float, help="Training budget; defaults to the execution contract.")
    p.add_argument("--postprocess-hours", type=float,
                   help="Compact publication ceiling; defaults to the contract and finishes as soon as pushed.")
    p.add_argument("--network-mode", choices=["academic", "clash", "direct"], default="academic")
    a = p.parse_args()
    contract = json.loads(CONFIG.read_text())
    active = contract["learning"]["active_streams"][0]
    if a.postprocess_hours is None:
        a.postprocess_hours = contract["execution"]["background_job"]["default_postprocess_wall_s"] / 3600
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
    contract["learning"]["phase_id"]=job_name
    (directory / "execution-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       recorded_environment="experiments/terrain_memory/compute-environment.json",
                       controller_build=contract["reuse"]["active_controller_build"],
                       mpc_workers=contract["resources"]["mpc_workers"],
                       torch_threads=contract["resources"]["torch_threads"],
                       environment={key: os.environ.get(key) for key in (
                           "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_ENABLE_X64", "MUJOCO_GL")})
    (directory / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    with (directory / "pip-packages.txt").open("w") as output:
        subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], stdout=output, stderr=subprocess.STDOUT, check=True)
    with (directory / "gpu-runtime.txt").open("w") as output:
        subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    command = [
        "bash", "scripts/run_s1_go2_r12_training.sh",
        "--group", active["group"], "--seed", str(active["seed"]),
        "--resume", a.resume, "--contract", str(directory/"execution-contract.json"), "--background-job",
        "--job-start-unix", str(started), "--deadline-unix", str(deadline),
    ]
    log = directory / "train.log"
    script = directory / "run-job.sh"
    postprocess = [
        "python", "-u", "scripts/finish_s1_go2_r12_job.py",
        "--job-dir", str(directory), "--hours", str(a.postprocess_hours),
    ]
    evaluation = [
        "python", "-u", "scripts/run_s1_go2_r12.py", "--job-dir", str(directory),
    ]
    # Close training/evaluation output before publishing the compact job records.
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
        "set +e",
        shlex.join(evaluation) + " > " + shlex.quote(str(directory / "evaluation.log")) + " 2>&1",
        "evaluation_exit=$?",
        "set -e",
        "printf '%s\\n' \"$evaluation_exit\" > " + shlex.quote(str(directory / "evaluation-exit-code.txt")),
        "date -u '+%Y-%m-%dT%H:%M:%SZ' > " + shlex.quote(str(directory / "evaluation-finished-at.txt")),
        shlex.join(postprocess) + ' --training-exit-code "$training_exit"',
        "",
    ]))
    # The selected resume state is authoritative; bulk outputs stay on the data disk.
    import torch
    resume_path = Path(a.resume)
    if not resume_path.is_absolute():
        resume_path = ROOT / resume_path
    resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
    record = dict(
        session=a.session, job_directory=str(directory.relative_to(ROOT)),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        controller_build=contract["reuse"]["active_controller_build"], current_task=contract["current_task"],
        group=active["group"], seed=active["seed"], resume_checkpoint=a.resume,
        phase_id=job_name,reward_version=contract['learning']['reward_version'],
        planned_excluded_old_version_transitions=contract['learning']['controller_version_boundary']['excluded_old_version_transitions_after_migration'],
        started_unix_s=started, deadline_unix_s=deadline,
        initial_transitions=int(resume_state["samples"]), initial_updates=int(resume_state["updates"]),
        target_transitions=contract["learning"]["target_cumulative_transitions_per_stream"],
        target_updates=contract["learning"]["target_cumulative_updates_per_stream"],
        log=str(log.relative_to(ROOT)), command=command, status="prepared",
        network_mode=a.network_mode, postprocess_hours=a.postprocess_hours,
        initial_attempts=sorted(p.name for p in (ROOT / contract["outputs"]["large_root"]).glob(
            f'train-{active["group"]}-s{active["seed"]}-ep*') if p.is_dir()),
        evaluation_command=evaluation,
        evaluation_hours=contract["control_and_evaluation"]["post_training_evaluation"]["maximum_wall_s"] / 3600,
        evaluation_status="pending_training",
        publication_status="compact_git_after_training_and_evaluation",
        delivery_mode="compact_first_local_bulk",
        delivery_policy_source=str(CONFIG.relative_to(ROOT)) + "#outputs.automatic_publication",
        bulk_upload_enabled=False, automatic_cleanup_enabled=False,
        result_summary=f'{contract["outputs"]["compact_root"]}/jobs/{job_name}.json',
    )
    metadata = directory / "job.json"
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    # The fixed default session name also covers compact Git publication.
    subprocess.run(["tmux", "new-session", "-d", "-s", a.session,
                    "-c", str(ROOT), "bash", str(script)], check=True)
    record["status"] = "launched"
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print("Detach leaves training, final evaluation and compact Git publication running in this tmux session.")
    print("Bulk checkpoints/videos/raw arrays stay local for user backup and manual deletion; finish never cleans them.")
    print("Inspect the job summary for actual outcomes, compact delivery and retained local checkpoint paths.")


if __name__ == "__main__":
    main()
