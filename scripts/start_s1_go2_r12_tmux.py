#!/usr/bin/env python3
"""Launch the declared isolated R12 streams in tmux with one immutable training clock."""
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

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"experiments/terrain_memory/s1-go2-local-readiness-r12.json"

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume",required=True)
    p.add_argument("--stream",help="Explicit declared stream; omitted launches both first forks.")
    p.add_argument("--job-dir",type=Path,help="Continue this existing job/phase with its own saved RNG/tail.")
    p.add_argument("--session")
    p.add_argument("--network-mode",choices=["academic","clash","direct"],default="clash")
    a=p.parse_args()
    current=json.loads(CONFIG.read_text())
    large=ROOT/current["outputs"]["large_root"]
    clock_path=large/"w38-training-clock.json"
    session=a.session or ("s1-"+a.stream if a.stream else "s1-w38")
    if subprocess.run(["tmux","has-session","-t",session],capture_output=True).returncode==0:
        raise ValueError("This tmux session already exists; inspect and continue it, do not duplicate launch.")
    auth_env=dict(os.environ);auth_env.pop("GH_TOKEN",None);auth_env.pop("GITHUB_TOKEN",None)
    subprocess.run(["bash","-c",'set -e; source scripts/autodl_network.sh "$1"; '
                    'gh auth setup-git --hostname github.com >/dev/null; '
                    'git var GIT_AUTHOR_IDENT >/dev/null',
                    "bash",a.network_mode],cwd=ROOT,env=auth_env,check=True)
    import torch
    resume=Path(a.resume)
    if not resume.is_absolute():resume=ROOT/resume
    saved=torch.load(resume,map_location="cpu",weights_only=False)
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    clock=json.loads(clock_path.read_text()) if clock_path.exists() else None
    if a.job_dir:
        directory=(ROOT/a.job_dir).resolve()
        job=json.loads((directory/"job.json").read_text())
        contract=json.loads((directory/"execution-contract.json").read_text())
        active=next(x for x in contract["learning"]["active_streams"] if x["stream_id"]==job["stream_id"])
        if (saved.get("phase_id")!=job["phase_id"] or
            saved.get("stream_id")!=job["stream_id"] or saved.get("fork_identity") is None):
            raise ValueError("Ordinary resume requires this job's own fork checkpoint, not the parent.")
        if time.time()>=job["deadline_unix_s"]:
            raise ValueError("Original shared training deadline has elapsed; do not restart training.")
        jobs=[(directory,job,contract,active)]
    else:
        selected=[x for x in current["learning"]["active_streams"] if not a.stream or x["stream_id"]==a.stream]
        if not selected:raise ValueError("Undeclared stream")
        if clock and any(x["stream_id"] in clock["jobs"] for x in selected):
            raise ValueError("Stream already launched; inspect its tmux/job and use --job-dir with its own resume.")
        if saved.get("fork_identity") is not None:
            raise ValueError("A fork checkpoint continues only its existing --job-dir.")
        jobs=[]
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for active in selected:
            name="job-"+stamp+"-"+active["stream_id"]
            directory=large/name;directory.mkdir(parents=True,exist_ok=False)
            contract=json.loads(json.dumps(current))
            contract["learning"].update(phase_id=name,stream_id=active["stream_id"],
                fork_exploration_seed=active["fork_exploration_seed"],
                training_deadline_path=str((directory/"training-deadline.json").relative_to(ROOT)))
            contract["learning"]["active_streams"]=[active]
            job=dict(session=session,job_directory=str(directory.relative_to(ROOT)),
                source_commit=source,controller_build=contract["reuse"]["active_controller_build"],
                current_task=contract["current_task"],group=active["group"],seed=active["seed"],
                stream_id=active["stream_id"],phase_id=name,resume_checkpoint=str(resume.relative_to(ROOT)),
                reward_version=contract["learning"]["reward_version"],
                planned_excluded_old_version_transitions=145,
                initial_transitions=int(saved["samples"]),initial_updates=int(saved["updates"]),
                target_transitions=contract["learning"]["target_cumulative_transitions_per_stream"],
                target_updates=contract["learning"]["target_cumulative_updates_per_stream"],
                initial_attempts=sorted(x.name for x in large.glob(f'train-{active["group"]}-s{active["seed"]}-ep*')),
                status="prepared",network_mode=a.network_mode,postprocess_hours=6,evaluation_hours=4,
                evaluation_status="pending_training",publication_status="compact_git_after_training_and_evaluation",
                delivery_mode="compact_first_local_bulk",bulk_upload_enabled=False,automatic_cleanup_enabled=False,
                result_summary=f'{contract["outputs"]["compact_root"]}/jobs/{name}.json')
            environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                controller_build=job["controller_build"],mpc_workers=8,torch_threads=1,blas_threads=1,
                environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS","XLA_PYTHON_CLIENT_PREALLOCATE","JAX_ENABLE_X64","MUJOCO_GL")})
            write(directory/"environment.json",environment)
            jobs.append((directory,job,contract,active))
    # Establish once immediately before the first actual detached launch.
    launch_time=time.time()
    if clock is None:
        clock=dict(task=current["current_task"],started_unix_s=launch_time,
            deadline_unix_s=launch_time+current["execution"]["background_job"]["default_maximum_wall_s"],
            jobs={})
    if launch_time>=clock["deadline_unix_s"]:raise ValueError("Shared training deadline elapsed.")
    scripts=[]
    for directory,job,contract,active in jobs:
        job.setdefault("started_unix_s",launch_time)
        job.update(deadline_unix_s=clock["deadline_unix_s"],shared_started_unix_s=clock["started_unix_s"],
                   shared_clock_path=str(clock_path.relative_to(ROOT)),session=session,status="launched")
        command=["bash","scripts/run_s1_go2_r12_training.sh","--group",active["group"],
            "--seed",str(active["seed"]),"--resume",str(resume),"--contract",str(directory/"execution-contract.json"),
            "--background-job","--job-start-unix",str(job["started_unix_s"]),
            "--deadline-unix",str(clock["deadline_unix_s"])]
        evaluation=[sys.executable,"-u","scripts/run_s1_go2_r12.py","--job-dir",str(directory)]
        finish=[sys.executable,"-u","scripts/finish_s1_go2_r12_job.py","--job-dir",str(directory),"--hours","6"]
        logfile=directory/("train-resume-"+str(int(launch_time))+".log" if a.job_dir else "train.log")
        script=directory/"run-job.sh"
        script.write_text("\n".join([
            "#!/usr/bin/env bash","set -e","cd "+shlex.quote(str(ROOT)),
            "source scripts/autodl_d57_env.sh",
            "source scripts/autodl_network.sh "+shlex.quote(a.network_mode),
            "unset GH_TOKEN GITHUB_TOKEN","set +e",
            shlex.join(command)+" > "+shlex.quote(str(logfile))+" 2>&1",
            "training_exit=$?",'echo "$training_exit" > '+shlex.quote(str(directory/"training-exit-code.txt")),
            "date -u '+%Y-%m-%dT%H:%M:%SZ' > "+shlex.quote(str(directory/"training-finished-at.txt")),
            shlex.join(evaluation)+" > "+shlex.quote(str(directory/"evaluation.log"))+" 2>&1",
            "evaluation_exit=$?",'echo "$evaluation_exit" > '+shlex.quote(str(directory/"evaluation-exit-code.txt")),
            "date -u '+%Y-%m-%dT%H:%M:%SZ' > "+shlex.quote(str(directory/"evaluation-finished-at.txt")),
            shlex.join(finish)+' --training-exit-code "$training_exit" > '+shlex.quote(str(directory/"finish.log"))+" 2>&1",""]))
        job.update(command=command,evaluation_command=evaluation,log=str(logfile.relative_to(ROOT)))
        if a.job_dir:job["last_resume"]=dict(checkpoint=str(resume.relative_to(ROOT)),at_unix_s=launch_time,source_commit=source)
        write(directory/"job.json",job)
        if not a.job_dir:write(directory/"execution-contract.json",contract)
        write(directory/"training-deadline.json",dict(
            shared_started_unix_s=clock["started_unix_s"],deadline_unix_s=clock["deadline_unix_s"],
            job=directory.name,stream_id=active["stream_id"]))
        clock["jobs"][active["stream_id"]]=str(directory.relative_to(ROOT))
        scripts.append(script)
    write(clock_path,clock)
    runner=large/("w38-run-"+str(int(launch_time))+".sh")
    lines=["#!/usr/bin/env bash"]
    for i,script in enumerate(scripts):
        lines+=["bash "+shlex.quote(str(script))+" &",f"p{i}=$!"]
    for i in range(len(scripts)):lines+=['wait "$p'+str(i)+'"']
    runner.write_text("\n".join(lines)+"\n")
    subprocess.run(["tmux","new-session","-d","-s",session,"-c",str(ROOT),"bash",str(runner)],check=True)
    from finish_s1_go2_r12_job import update_parallel_handoff
    update_parallel_handoff(current)
    print(json.dumps(dict(session=session,clock=clock,jobs=[j for _,j,_,_ in jobs]),indent=2))

if __name__=="__main__":main()
