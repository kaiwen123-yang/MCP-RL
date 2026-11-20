"""Run the W55 N original evaluation with the frozen case, record and render its outcome."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time

from publish_s1_go2_w54 import ROOT, LARGE, write
from run_s1_go2_w54 import resource_sample

RUN = 'tm-s1-go2-local-readiness-r12'
SMALL = ROOT / 'results/terrain_memory/simulation' / RUN / 'w55/n'
PLAN = 'c8e0b38d8256305b9e5a14907829980c3f1b0054'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def declaration(build, case):
    import torch
    revision = build.removeprefix('S5-')
    label = f'N-{case}-{revision}'
    dest = SMALL / label
    dest.mkdir(parents=True, exist_ok=False)
    seed, episode = (1, 428) if case == 'A' else (2, 417)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    identity=json.loads((ROOT/'work/r11-frozen'/build/'identity.json').read_text())
    parent = LARGE / 'training' / f'w44-{case.lower()}' / f'PANN_PPO_S5-seed{seed}'
    checkpoint = parent / 'checkpoint-014737.pt'
    resume = parent / f'resume-state-ep{episode:05d}.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    complete = torch.load(resume, map_location='cpu', weights_only=False)
    assert (saved['samples'],saved['updates'],saved['seed']) == (14737,57,seed)
    assert (complete['samples'],complete['updates'],complete['episode']) == (14737,57,episode)
    assert saved['normalization'] == complete['normalization'] and not saved['buffer'] and not complete['buffer']
    now = time.time()
    stamp = time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now))
    job_id = f'job-{stamp}-w55-{label.lower()}'
    jobdir = LARGE / 'jobs' / job_id
    jobdir.mkdir(parents=True, exist_ok=False)
    job = dict(job_id=job_id, label=label, route='N', case=case, seed=seed, workers=10,
               affinity='0-9' if case=='A' else '10-19', started_unix_s=now, deadline=None,
               wall_deadline_enabled=False, controller_build=build, controller_binary_sha256=identity['binary_sha256'],
               execution_library_sha256=identity['execution_library_sha256'], controller_artifact_identity=identity,
               checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest(checkpoint),
               resume_checkpoint=str(resume.relative_to(ROOT)), resume_sha256=digest(resume),
               normalization_sha256=hashlib.sha256(json.dumps(saved['normalization'],sort_keys=True).encode()).hexdigest(),
               samples=14737, updates=57, tail=0, excluded=145, new_PPO_samples=0,
               new_PPO_updates=0, donor=0, actor='deterministic mean and frozen exact W44 normalization',
               initial_state='complete original evaluation t=0, virgin terrain', source_commit=commit, plan_commit=PLAN)
    contract = json.loads((ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json').read_text())
    contract.update(current_task='W55_N_ORIGINAL_TASK', protocol_revision='W55_original_frozen_ladder',
                    next_authorized_plan='docs/S1_W55_STABILITY_PLAN.md',
                    w54_fixed_policy=job, w55_fixed_policy=job)
    contract['reuse'].update(active_controller_build=build, controller_artifact=build)
    contract['outputs']['compact_root'] = str(SMALL.relative_to(ROOT))
    contract['execution']['deadline'] = None
    contract['learning'].update(phase_id=job_id, stream_id='w55-'+label.lower(), phase_initial_transitions=14737,
        phase_initial_updates=57, target_cumulative_transitions_per_stream=14737, target_cumulative_updates_per_stream=57,
        training_enabled=False, additional_transitions_from_reviewed_checkpoint=0, additional_updates_from_reviewed_checkpoint=0,
        active_streams=[], evaluation_only=True)
    contract['resources'].update(mpc_workers=10, torch_threads=1, torch_device='cuda')
    contract['execution']['background_job'].update(wall_deadline_enabled=False, default_maximum_wall_s=None)
    contract['control_and_evaluation']['post_training_evaluation'].update(maximum_wall_s=None, wall_deadline_enabled=False)
    for directory in (dest, jobdir):
        write(directory/'job.json',job)
        write(directory/'execution-contract.json',contract)
    return job


def launch(build, case, level):
    case_dir = SMALL / f'N-{case}-{build.removeprefix("S5-")}'
    if level == 'E30':
        job = declaration(build,case)
    else:
        job = json.loads((case_dir/'job.json').read_text())
    jobdir = LARGE/'jobs'/job['job_id']
    dest = case_dir / (level+'.json')
    if dest.exists():
        raise ValueError(f'Existing {dest}; inspect exact job before continuing')
    session = 's1-w55-'+job['label'].lower()+'-'+level.lower()
    command = [sys.executable,'-u',__file__,'--run',str(case_dir),'--level',level]
    shell = 'source scripts/autodl_d57_env.sh && export NUMEXPR_NUM_THREADS=1 && exec taskset -c ' + job['affinity'] + ' ' + shlex.join(command)
    shell += ' > ' + shlex.quote(str(jobdir/(level+'-runner.log'))) + ' 2>&1'
    write(dest,dict(status='declared',tmux=session,job=job['job_id'],level=level,deadline=None,
                    wall_deadline_enabled=False,kill_on_conversation_end=False))
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),shell],check=True)
    subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
    print(json.dumps(dict(job=job['job_id'],tmux=session,ledger=str(dest))))


def run(case_dir, level):
    job = json.loads((case_dir/'job.json').read_text())
    raw_job = LARGE/'jobs'/job['job_id']
    attempt = f'{job["job_id"]}-eval-PANN_PPO_S5-s{job["seed"]}-{level}-attempt01'
    dest = case_dir/(level+'.json')
    state = json.loads(dest.read_text())
    command = [sys.executable,'-u','scripts/run_s1_go2_r12.py','--contract',str(raw_job/'execution-contract.json'),
               '--attempt',attempt,'--evaluation-start-unix',str(job['started_unix_s']),'--checkpoint',str(ROOT/job['checkpoint']),
               '--level',level,'--seed',str(job['seed']),'--workers',str(job['workers']),'--build',job['controller_build'],'--condition','nominal']
    state.update(attempt=attempt,status='running',runner_pid=os.getpid(),started_unix_s=time.time(),command=command)
    write(dest,state)
    usage0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (raw_job/(attempt+'.log')).open('w') as log:
        process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state['pid'] = process.pid
        write(dest,state)
        while process.poll() is None:
            resource_sample(process.pid,raw_job/'resources.jsonl')
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    usage1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    state.update(exit_code=process.returncode,ended_unix_s=time.time(),
                 user_cpu_s=usage1.ru_utime-usage0.ru_utime,system_cpu_s=usage1.ru_stime-usage0.ru_stime)
    result_path = SMALL/(attempt+'.json')
    if result_path.exists():
        result = json.loads(result_path.read_text())
        state.update(status=result['stage_status'],success=result['success'],reason=result['reason'],
                     physical_s=result['simulation_s'],physical_samples=result['physical_samples'],
                     physical_started_unix_s=result['physical_started_unix_s'],physical_ended_unix_s=result['physical_ended_unix_s'])
    else:
        state.update(status='technical_interruption',reason='process_exited_without_completed_result')
    write(dest,state)
    write(raw_job/(level+'-runtime.json'),state)
    if result_path.exists():
        import render_s1_go2_r12 as renderer
        renderer.SMALL = SMALL
        renderer.render(attempt)
        from analyze_s1_w55_n import measure
        measure(case_dir,level)
        state['video'] = json.loads((LARGE/attempt/'video-asset.json').read_text())
        write(dest,state)
        write(raw_job/(level+'-runtime.json'),state)
    return process.returncode


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--launch',choices=('A','B'))
    p.add_argument('--build',default='S5-r54')
    p.add_argument('--level',choices=('E30','E60','E300','E600'),default='E30')
    p.add_argument('--run',type=Path)
    a = p.parse_args()
    if a.launch:
        launch(a.build,a.launch,a.level)
    else:
        sys.exit(run(a.run,a.level))
