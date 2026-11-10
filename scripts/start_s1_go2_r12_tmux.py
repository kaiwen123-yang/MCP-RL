#!/usr/bin/env python3
"""Launch declared isolated R12 streams with explicit goal/legacy wall policy."""
import argparse
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from s1_go2_job_limits import adopt_goal_policy, goal_based, expired, remove_wall_deadline

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"experiments/terrain_memory/s1-go2-local-readiness-r12.json"

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def phase_metadata(parent, active, controller_build, artifact):
    """Describe the actual selected parent, not a historical phase template."""
    return dict(parent_stream_id=parent["parent_stream_id"],parent_phase_id=parent["parent_phase_id"],
        parent_controller_build=parent["parent_controller_build"],active_controller_build=controller_build,
        parent_controller_artifact=parent["parent_controller_artifact"],controller_artifact=artifact,
        implementation_base_controller_build="S5-r30" if controller_build=="S5-r31" else parent["parent_controller_build"],
        parent_checkpoint=parent["resume_checkpoint"],parent_transitions=parent["parent_transitions"],
        parent_updates=parent["parent_updates"],parent_completed_update_transitions=parent["parent_completed_update_transitions"],
        parent_tail_buffer=parent["parent_tail"],excluded_old_version_transitions_after_migration=parent["excluded_old_version_transitions"],
        new_exclusions=0,new_controller_initial_buffer=0,reward_version_change=False,
        parent_reward_version=parent["parent_reward_version"],reward_version=parent["parent_reward_version"],
        restoration="Own actor/critic/optimizer/normalization/RNG/history; no new fork/reseed.",
        first_new_rollout=dict(first_transition=parent["parent_transitions"]+1,
            last_transition=parent["parent_transitions"]+256,update=parent["parent_updates"]+1),
        target_checkpoint=f'checkpoint-{active["target_transitions"]:06d}.pt',
        target_transitions=active["target_transitions"],target_updates=active["target_updates"],
        next_episode_minimum=parent["parent_episode"]+1,parent_directories_read_only=True,
        idempotence="Declare once; ordinary same-job resume retains own clock/RNG/tail.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume",help="Own checkpoint for ordinary resume; new declared phases use their declared parent map.")
    p.add_argument("--new-phase",action="store_true",help="Create the explicitly authorized branch continuations from learning.parent_map.")
    p.add_argument("--stream",help="Explicit declared stream; omitted launches both declared continuations.")
    p.add_argument("--job-dir",type=Path,help="Continue this existing job/phase with its own saved RNG/tail.")
    p.add_argument("--controller-artifact", choices=("S5-r29-qp1", "S5-r29-qp2"),
        help="Explicit S5-r29 numerical QP repair on an existing own job; logical controller/phase stay unchanged.")
    p.add_argument("--no-wall-deadline", action="store_true",
        help="Apply the 2026-11-07 goal-based background policy; retain prior clock values in history.")
    p.add_argument("--workers",type=int,choices=(10,11,12,14),
        help="MPC workers per stream for new jobs only; default from resources.mpc_workers (12). Also used by each new job's evaluation.")
    p.add_argument("--session")
    p.add_argument("--network-mode",choices=["academic","clash","direct"],default="academic")
    a=p.parse_args()
    if a.job_dir and a.workers is not None:
        p.error("--workers applies only to new jobs; ordinary resume retains the existing frozen worker configuration.")
    if a.controller_artifact and not a.job_dir:
        p.error("--controller-artifact applies only to an explicit --job-dir ordinary resume.")
    current=json.loads(CONFIG.read_text())
    no_wall_deadline=a.no_wall_deadline or goal_based(current)
    large=ROOT/current["outputs"]["large_root"]
    clock_path=ROOT/current["execution"]["background_job"]["training_clock_path"]
    session=a.session or ("s1-"+a.stream if a.stream else current["execution"]["background_job"]["session_default"])
    if subprocess.run(["tmux","has-session","-t","="+session],capture_output=True).returncode==0:
        raise ValueError("This tmux session already exists; inspect and continue it, do not duplicate launch.")
    auth_env=dict(os.environ);auth_env.pop("GH_TOKEN",None);auth_env.pop("GITHUB_TOKEN",None)
    subprocess.run(["bash","-c",'set -e; source scripts/autodl_network.sh "$1"; '
                    'gh auth setup-git --hostname github.com >/dev/null; '
                    'git var GIT_AUTHOR_IDENT >/dev/null',
                    "bash",a.network_mode],cwd=ROOT,env=auth_env,check=True)
    import torch
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    clock=json.loads(clock_path.read_text()) if clock_path.exists() else None
    if a.job_dir:
        if a.new_phase or not a.resume:
            raise ValueError("Ordinary resume needs --resume and the existing --job-dir, without --new-phase.")
        resume=Path(a.resume)
        if not resume.is_absolute():resume=ROOT/resume
        saved=torch.load(resume,map_location="cpu",weights_only=False)
        directory=(ROOT/a.job_dir).resolve()
        job=json.loads((directory/"job.json").read_text())
        contract=json.loads((directory/"execution-contract.json").read_text())
        active=next(x for x in contract["learning"]["active_streams"] if x["stream_id"]==job["stream_id"])
        clock_path=ROOT/job["shared_clock_path"]
        clock=json.loads(clock_path.read_text())
        if (saved.get("phase_id")!=job["phase_id"] or
            saved.get("stream_id")!=job["stream_id"] or saved.get("fork_identity") is None
            or saved.get("controller_build")!=job["controller_build"]
            or saved.get("reward_version")!=job["reward_version"]
            or saved.get("group")!=job["group"] or saved.get("seed")!=job["seed"]
            or saved.get("phase_lineage")!=job.get("phase_lineage")):
            raise ValueError("Ordinary resume requires this job's own fork checkpoint, not the parent.")
        if not no_wall_deadline and expired(job["deadline_unix_s"]):
            raise ValueError("Original shared training deadline has elapsed; do not restart training.")
        if saved['samples'] >= job['target_transitions']:
            raise ValueError('This job already reached its sample target; preserve its completed result.')
        previous_artifact=contract["reuse"].get("controller_artifact",job["controller_build"])
        if saved.get('controller_artifact',saved['controller_build']) != previous_artifact:
            raise ValueError('Resume checkpoint artifact must match this job\'s most recent artifact.')
        artifact=a.controller_artifact or previous_artifact
        repair_parents={'S5-r29-qp1':'S5-r29','S5-r29-qp2':'S5-r29-qp1'}
        allowed_repair=(job['controller_build']=='S5-r29'
            and repair_parents.get(artifact)==previous_artifact
            and (artifact=='S5-r29-qp2' or saved['updates']==job['initial_updates']==33))
        if artifact!=previous_artifact and not allowed_repair:
            raise ValueError('Only the declared same-QP numerical artifact upgrade is supported on this job.')
        if artifact!=job["controller_build"]:
            frozen=ROOT/"work/r11-frozen"/artifact
            identity=json.loads((frozen/"identity.json").read_text())
            if (identity.get("build")!=artifact or identity.get("based_on")!=repair_parents.get(artifact)
                    or identity.get("controller_contract_build")!=job["controller_build"]
                    or not (frozen/"agent_server").is_file()):
                raise ValueError("Controller artifact must be the built, separately frozen repair of this logical controller.")
        contract["reuse"]["controller_artifact"]=artifact
        if no_wall_deadline:
            adopt_goal_policy(contract)
        job["controller_artifact"]=artifact
        jobs=[(directory,job,contract,active,resume)]
    else:
        selected=[x for x in current["learning"]["active_streams"] if not a.stream or x["stream_id"]==a.stream]
        if not selected:raise ValueError("Undeclared stream")
        if clock and any(x["stream_id"] in clock["jobs"] for x in selected):
            raise ValueError("Stream already launched; inspect its tmux/job and use --job-dir with its own resume.")
        if not a.new_phase:
            raise ValueError("New branch continuation requires --new-phase and its declared parent_map.")
        if a.resume and len(selected)!=1:
            raise ValueError("Two branch continuations must use their separate parent_map checkpoints.")
        jobs=[]
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for active in selected:
            parent=current["learning"]["parent_map"][active["parent_map_key"]]
            resume=ROOT/parent["resume_checkpoint"]
            if a.resume and (ROOT/a.resume).resolve()!=resume.resolve():
                raise ValueError("The new phase must use this stream's declared parent checkpoint.")
            saved=torch.load(resume,map_location="cpu",weights_only=False)
            if not (saved["stream_id"]==parent["parent_stream_id"]
                    and saved["phase_id"]==parent["parent_phase_id"]
                    and saved["group"]==parent["group"]==active["group"]
                    and saved["seed"]==parent["seed"]==active["seed"]
                    and saved["controller_build"]==parent["parent_controller_build"]
                    and saved["reward_version"]==parent["parent_reward_version"]
                    and saved["samples"]==parent["parent_transitions"]
                    and saved["updates"]==parent["parent_updates"]
                    and saved["episode"]==parent["parent_episode"]
                    and len(saved["buffer"])==parent["parent_tail"]==0
                    and saved["excluded_old_version_transitions"]==parent["excluded_old_version_transitions"]
                    and saved["fork_identity"]==parent["ancestor_fork_identity"]):
                raise ValueError("New phase requires its declared own parent state with unchanged fork ancestry.")
            if (ROOT/active["training_directory"]).resolve()==resume.parent.resolve():
                raise ValueError("The new phase must not write into its parent training directory.")
            name="job-"+stamp+"-"+active["stream_id"]
            directory=large/name;directory.mkdir(parents=True,exist_ok=False)
            contract=json.loads(json.dumps(current))
            if no_wall_deadline:
                adopt_goal_policy(contract)
            workers=a.workers if a.workers is not None else contract["resources"]["mpc_workers"]
            contract["resources"]["mpc_workers"]=workers
            contract["control_and_evaluation"]["post_training_evaluation"]["solver_workers"]=workers
            lineage=dict(parent["phase_lineage"],status="explicit_new_phase_continuation",
                current_phase_id=name,parent_checkpoint=parent["resume_checkpoint"],
                parent_transitions=saved["samples"],parent_updates=saved["updates"],
                parent_episode=saved["episode"],inherited_excluded=saved["excluded_old_version_transitions"],
                new_exclusions=0,new_fork=False,reseeded_after_checkpoint_restore=False,
                restored=["actor","critic","optimizer","RNG","normalization","historical_counts"])
            prefix=f'train-{name}-{active["group"]}-s{active["seed"]}-ep'
            contract["learning"].update(phase_id=name,stream_id=active["stream_id"],
                training_directory=active["training_directory"],episode_prefix=prefix,
                phase_lineage=lineage,
                training_deadline_path=str((directory/"training-deadline.json").relative_to(ROOT)))
            contract["learning"]["target_cumulative_transitions_per_stream"]=active.get("target_transitions",contract["learning"]["target_cumulative_transitions_per_stream"])
            contract["learning"]["target_cumulative_updates_per_stream"]=active.get("target_updates",contract["learning"]["target_cumulative_updates_per_stream"])
            contract["learning"].update(phase_initial_transitions=saved["samples"],phase_initial_updates=saved["updates"])
            contract["learning"]["active_streams"]=[active]
            contract["learning"]["controller_version_boundary"]=phase_metadata(parent,active,
                contract["reuse"]["active_controller_build"],contract["reuse"].get("controller_artifact",contract["reuse"]["active_controller_build"]))
            contract["control_and_evaluation"]["post_training_evaluation"]["checkpoint_selection"]=dict(
                stream_id=active["stream_id"],parent_checkpoint=parent["resume_checkpoint"],
                parent_updates=saved["updates"],parent_transitions=saved["samples"],
                target_updates=active["target_updates"],target_transitions=active["target_transitions"],
                rule="Own final new update with matching stream/phase/controller/artifact/reward; no parent fallback.")

            job=dict(session=session,job_directory=str(directory.relative_to(ROOT)),
                source_commit=source,controller_build=contract["reuse"]["active_controller_build"],
                controller_artifact=contract["reuse"].get("controller_artifact",contract["reuse"]["active_controller_build"]),
                current_task=contract["current_task"],group=active["group"],seed=active["seed"],
                mpc_workers=workers,evaluation_solver_workers=workers,
                stream_id=active["stream_id"],phase_id=name,resume_checkpoint=str(resume.relative_to(ROOT)),
                training_directory=active["training_directory"],episode_prefix=prefix,
                parent_stream_id=parent["parent_stream_id"],phase_lineage=lineage,
                fork_identity=saved["fork_identity"],
                reward_version=contract["learning"]["reward_version"],
                planned_excluded_old_version_transitions=145,
                initial_transitions=int(saved["samples"]),initial_updates=int(saved["updates"]),
                target_transitions=contract["learning"]["target_cumulative_transitions_per_stream"],
                target_updates=contract["learning"]["target_cumulative_updates_per_stream"],
                initial_attempts=sorted(x.name for x in large.glob(prefix+"*")),
                status="prepared",network_mode=a.network_mode,
                postprocess_hours=None if no_wall_deadline else 6,
                evaluation_hours=None if no_wall_deadline else 4,
                evaluation_status="pending_training",publication_status="compact_git_after_training_and_evaluation",
                delivery_mode="compact_first_local_bulk",bulk_upload_enabled=False,automatic_cleanup_enabled=False,
                result_summary=f'{contract["outputs"]["compact_root"]}/jobs/{name}.json')
            if job['controller_build'] == 'S5-r33':
                job.update(selected_mechanism=contract['reuse']['selected_mechanism'],
                    mechanism_evidence=contract['reuse']['mechanism_evidence'],
                    controller_binary_sha256=contract['reuse']['controller_binary_sha256'])
            environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                runtime_root=str(ROOT),pythonpath=os.environ.get("PYTHONPATH"),
                controller_build=job["controller_build"],mpc_workers=workers,
                torch_threads=contract["resources"]["torch_threads"],blas_threads=contract["resources"]["blas_threads"],
                environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS","XLA_PYTHON_CLIENT_PREALLOCATE","JAX_ENABLE_X64","MUJOCO_GL")})
            write(directory/"environment.json",environment)
            jobs.append((directory,job,contract,active,resume))
    # Establish once immediately before the first actual detached launch.
    launch_time=time.time()
    if clock is None:
        clock=dict(task=current["current_task"],started_unix_s=launch_time,
            deadline_unix_s=None if no_wall_deadline else launch_time+current["execution"]["background_job"]["default_maximum_wall_s"],
            jobs={})
    if no_wall_deadline:
        remove_wall_deadline(clock, launch_time)
    if expired(clock["deadline_unix_s"]):raise ValueError("Shared training deadline elapsed.")
    scripts=[]
    for directory,job,contract,active,resume in jobs:
        if a.job_dir:
            history=directory/"resume-history"/str(time.time_ns())
            history.mkdir(parents=True)
            # Preserve the previous invocation's small records in this server
            # job. No checkpoint, physics output, or GPU path is moved/deleted.
            for name in ("job.json","execution-contract.json","run-job.sh",
                    "training-exit-code.txt","training-finished-at.txt",
                    "evaluation-exit-code.txt","evaluation-finished-at.txt",
                    "evaluation.log","finish.log"):
                source_path=directory/name
                if source_path.is_file():shutil.copy2(source_path,history/name)
            if (directory/"stream-records").exists():
                (directory/"stream-records").rename(history/"stream-records")
            from run_s1_go2_r12 import archive_unstarted_evaluation
            old_evaluation=archive_unstarted_evaluation(directory,contract,history)
            for name in ("training-exit-code.txt","training-finished-at.txt"):
                if (directory/name).is_file():(directory/name).replace(history/name)
            if old_evaluation:
                for name in ("evaluation-exit-code.txt","evaluation-finished-at.txt"):
                    if (directory/name).is_file():(directory/name).replace(history/name)
            job["last_resume"]=dict(checkpoint=str(resume.relative_to(ROOT)),at_unix_s=launch_time,
                source_commit=source,evidence_directory=str(history.relative_to(ROOT)),
                samples=saved["samples"],updates=saved["updates"],preserved_tail=len(saved["buffer"]),
                previous_controller_artifact=previous_artifact,controller_artifact=artifact,
                unstarted_evaluation_archive=old_evaluation)
            job["evaluation_status"]=("pending_training" if old_evaluation else job.get("evaluation_status"))
        job.setdefault("started_unix_s",launch_time)
        if no_wall_deadline:
            remove_wall_deadline(job, launch_time)
            job.update(postprocess_hours=None,evaluation_hours=None)
        job.update(deadline_unix_s=clock["deadline_unix_s"],shared_started_unix_s=clock["started_unix_s"],
                   shared_clock_path=str(clock_path.relative_to(ROOT)),session=session,status="launched")
        command=["bash","scripts/run_s1_go2_r12_training.sh","--group",active["group"],
            "--seed",str(active["seed"]),"--resume",str(resume),"--contract",str(directory/"execution-contract.json"),
            "--background-job","--job-start-unix",str(job["started_unix_s"])]
        if clock['deadline_unix_s'] is not None:
            command += ['--deadline-unix',str(clock['deadline_unix_s'])]
        evaluation=[sys.executable,"-u","scripts/run_s1_go2_r12.py","--job-dir",str(directory)]
        finish=[sys.executable,"-u","scripts/finish_s1_go2_r12_job.py","--job-dir",str(directory)]
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
        write(directory/"job.json",job)
        write(directory/"execution-contract.json",contract)
        write(directory/"training-deadline.json",dict(
            shared_started_unix_s=clock["started_unix_s"],deadline_unix_s=clock["deadline_unix_s"],
            job=directory.name,stream_id=active["stream_id"]))
        clock["jobs"][active["stream_id"]]=str(directory.relative_to(ROOT))
        scripts.append(script)
    write(clock_path,clock)
    runner=large/(clock_path.stem.replace("-training-clock","")+"-run-"+str(int(launch_time))+".sh")
    lines=["#!/usr/bin/env bash"]
    for i,script in enumerate(scripts):
        lines+=["bash "+shlex.quote(str(script))+" &",f"p{i}=$!"]
    for i in range(len(scripts)):lines+=['wait "$p'+str(i)+'"']
    runner.write_text("\n".join(lines)+"\n")
    subprocess.run(["tmux","new-session","-d","-s",session,"-c",str(ROOT),"bash",str(runner)],check=True)
    from finish_s1_go2_r12_job import update_parallel_handoff
    update_parallel_handoff(current)
    print(json.dumps(dict(session=session,clock=clock,jobs=[j for _,j,_,_,_ in jobs]),indent=2))

if __name__=="__main__":main()
