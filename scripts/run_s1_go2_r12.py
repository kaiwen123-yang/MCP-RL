#!/usr/bin/env python3
"""Evaluate a frozen R12 policy, or run one job's final nominal E30/E60 ladder."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'experiments/terrain_memory/s1-go2-local-readiness-r12.json'


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')


def evaluate(a, contract):
    from run_s1_go2_r11 import configuration, arguments
    from s1_go2_r11_protocol import case_identity, stable_digest, prerequisite
    from s1_go2_r11_runtime import run, index_file
    from s1_go2_joint_material import JointMaterial

    if time.time() >= a.deadline_unix:
        raise ValueError('The independent R12 evaluation deadline has passed')
    build = a.build or contract['reuse']['active_controller_build']
    workers = a.workers or contract['resources']['mpc_workers']
    c, r9 = configuration(build, a.level, .215, 'tm02_anchor')
    r9['pann']['weights_path'] = str(ROOT / contract['resources']['pann_weights_path'])
    c.update(run_id=contract['run_id'], stage='S1R12', r12_contract=contract,
             campaign_budget=contract['execution'], predecessor_attempt=None,
             window=dict(task=contract['current_task'] + '_FINAL_EVAL',
                         execution_mode='frozen_policy_evaluation',
                         started_unix_s=a.window_start_unix,
                         run_deadline_unix_s=a.deadline_unix,
                         hard_deadline_unix_s=a.deadline_unix))
    c['outputs']['root'] = contract['outputs']['compact_root']
    args = arguments(a.attempt, build, a.condition, 'formal',
                     c['r11_stage']['task_kind'], workers=workers)
    args.config = a.contract
    args.cohort = contract['protocol_revision']
    args.original_robot_dir = str(a.assets / 'go2')
    args.original_scene = str(a.assets / 'scene_flat.xml')
    args.original_resolved = str(a.assets / 'rigid-T1R-resolved.json')
    os.environ['LD_LIBRARY_PATH'] = str(ROOT / 'work/r11-frozen' / build / 'lib')
    os.environ['MJPC_TASKS_DIR'] = str(ROOT / 'work/r11-frozen' / build / 'runtime-tasks')
    stream = None
    normalization = None
    small = ROOT / c['outputs']['root']
    try:
        if a.group == 'FIXED_S5':
            joint = JointMaterial(ROOT, r9, .215, terrain='tm02_anchor')
            joint.group = a.group
        else:
            from s1_go2_r11_policy import PolicyStream, PolicyMaterial
            policy_contract = copy.deepcopy(contract)
            old = json.loads((ROOT / contract['reuse']['runtime_physics_ppo_and_task_settings']).read_text())
            policy_contract['joint_learning'] = copy.deepcopy(old['joint_learning'])
            policy_contract['joint_learning'].update(
                standard_target_transitions_per_stream=contract['learning']['target_cumulative_transitions_per_stream'],
                cross_stream_wait=False)
            policy_output = a.policy_output or (ROOT / contract['outputs']['large_root'] / 'evaluation-policy' / a.attempt)
            stream = PolicyStream(ROOT, a.group, a.seed, policy_contract, policy_output,
                                  training=False, checkpoint=a.checkpoint)
            normalization = stream.normalization
            c['r11_policy'] = True
            joint = PolicyMaterial(ROOT, r9, stream, 0, c['r11_stage']['duration_s'], full_recording=True)
        # Wall clocks and the job contract do not change same-policy qualification.
        case_config = copy.deepcopy(c)
        case_config.pop('window')
        case_config.pop('r12_contract')
        identity, _ = case_identity(case_config, a.group, a.seed, a.condition,
                                    a.checkpoint, normalization)
        identity.update(protocol_revision=contract['formal_evaluation_protocol'], terrain='tm02_anchor',
                        job=contract['learning'].get('phase_id'),stream_id=contract['learning'].get('stream_id'),
                        controller_build=build,reward_version=contract['learning']['reward_version'])
        c.update(case_identity=identity, case_key=stable_digest(identity), evaluation_workers=workers)
        with index_file(small, c) as idx:
            c['predecessor_attempt'] = prerequisite(c, idx, 'formal', 'tm02_anchor')
        result = run(args, c, joint)
    except Exception as error:
        with index_file(small, c) as idx:
            row = next((x for x in idx['attempts'] if x['attempt'] == a.attempt), None)
            if row is not None:
                row.update(status='initialization_failed' if not row['physical_started'] else 'interrupted',
                           stage_status='technical_interruption', error=repr(error), success=False)
        raise
    finally:
        if stream is not None:
            stream.log.close()
            stream.learning.close()
    with index_file(small, c) as idx:
        idx.update(protocol_revision=contract['protocol_revision'], current_task=contract['current_task'])
    return result


def continue_authorized_budget(jobdir, job, contract):
    """Use this job's explicitly extended budget after its original process exits.

    The existing tmux stays alive. No signal or restart is sent to training.
    A saved, naturally truncated same-version tail resumes only at the old
    deadline, with the original job start and unchanged cumulative target.
    """
    extension=job.get('budget_extension')
    if not extension or extension['status']!='scheduled_at_original_deadline':
        return
    large=ROOT/contract['outputs']['large_root']
    stream=large/'training'/f'{job["group"]}-seed{job["seed"]}'
    stopped_path=stream/'stopped.json'
    stopped=json.loads(stopped_path.read_text()) if stopped_path.exists() else {}
    if stopped.get('actual_transitions',0)>=job['target_transitions']:
        extension['status']='not_needed_target_reached'
    elif time.time()>=job['deadline_unix_s']:
        extension['status']='not_started_extended_budget_elapsed'
    elif (stopped.get('last_stage_status')!='budget_truncated' or
          stopped.get('deadline_unix_s')!=extension['previous_deadline_unix_s']):
        extension['status']='not_started_no_original_deadline_boundary'
        extension['reason']='A technical interruption is not an authorized automatic restart.'
    else:
        progress=json.loads((stream/'progress.json').read_text())
        checkpoint=Path(progress['checkpoint'])
        if not checkpoint.is_absolute():checkpoint=ROOT/checkpoint
        for filename in ('stopped.json','progress.json','collection.json','training-contract.json'):
            shutil.copy2(stream/filename,jobdir/('before-budget-extension-'+filename))
        command=[
            'bash','scripts/run_s1_go2_r12_training.sh',
            '--group',job['group'],'--seed',str(job['seed']),
            '--resume',str(checkpoint),'--contract',str(jobdir/'execution-contract.json'),
            '--background-job','--job-start-unix',str(job['started_unix_s']),
            '--deadline-unix',str(job['deadline_unix_s'])]
        extension.update(status='continuation_running',command=command,
            checkpoint=str(checkpoint.relative_to(ROOT)),continuation_started_unix_s=time.time(),
            samples_at_boundary=progress['samples'],updates_at_boundary=progress['updates'],
            preserved_tail=progress['unupdated_fragment'],
            original_training_exit_code=int((jobdir/'training-exit-code.txt').read_text()),
            control_source_commit=job['source_commit'],
            continuation_driver_commit=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            physical_session='New natural session; full physical world is not restored.')
        write(jobdir/'job.json',job)
        print(json.dumps(dict(job=jobdir.name,budget_extension=extension)),flush=True)
        with (jobdir/'training-extension.log').open('w') as log:
            code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        extension.update(status='continuation_finished',continuation_exit_code=code,
                         continuation_finished_unix_s=time.time())
        (jobdir/'training-exit-code.txt').write_text(str(code)+'\n')
        (jobdir/'training-finished-at.txt').write_text(
            time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')
    write(jobdir/'job.json',job)


def evaluate_job(jobdir, contract):
    jobdir = (ROOT / jobdir).resolve()
    contract = json.loads((jobdir / 'execution-contract.json').read_text())
    job = json.loads((jobdir / 'job.json').read_text())
    if contract['current_task'] not in ('W37_CONTACT_CONSISTENT_JOINT','W38_SMOOTH_APPROACH_PARALLEL_JOINT'):
        continue_authorized_budget(jobdir,job,contract)
    existing=jobdir/'evaluation.json'
    if existing.exists():
        previous_record=json.loads(existing.read_text())
        if previous_record.get('status')!='running':
            print(json.dumps(previous_record),flush=True)
            return 0
        started=previous_record['started_unix_s']
    else:
        from datetime import datetime
        started=datetime.fromisoformat((jobdir/'training-finished-at.txt').read_text().strip().replace('Z','+00:00')).timestamp()
    # This branch's evaluation deadline is never renewed by a resumed invocation.
    settings = contract['control_and_evaluation']['post_training_evaluation']
    deadline = started + settings['maximum_wall_s']
    if not settings.get('enabled',True):
        write(jobdir/'evaluation.json',dict(job=jobdir.name,status='cancelled_user_regression_stop',reason=settings.get('disabled_reason'),attempts=[],stages=[]))
        return 0
    record = dict(job=jobdir.name, status='running', started_unix_s=started,
                  deadline_unix_s=deadline, controller_build=job['controller_build'],
                  group=job['group'], seed=job['seed'], stream_id=job.get('stream_id'), phase_id=job['phase_id'], condition=settings['condition'],
                  workers=settings['solver_workers'], checkpoint_selection=settings['checkpoint_selection'],
                  deterministic_policy_mean=True, frozen_normalization=True, ppo_updates=False,
                  attempts=[], stages=[])
    destination = jobdir / 'evaluation.json'
    write(destination, record)
    large = ROOT / contract['outputs']['large_root']
    small = ROOT / contract['outputs']['compact_root']
    stream = large / 'training' / f'{job["group"]}-seed{job["seed"]}'
    try:
        import torch
        stopped_path=stream/'stopped.json'
        stopped=json.loads(stopped_path.read_text()) if stopped_path.exists() else {}
        collection_path=stream/'collection.json'
        collection=json.loads(collection_path.read_text()) if collection_path.exists() else {}
        current_stop=stopped if stopped.get('phase_id')==job['phase_id'] else {}
        job['phase_stop']=current_stop.get('phase_stop')
        job['training_stop_reason']=current_stop.get('reason')
        job['excluded_old_version_transitions']=collection.get('excluded_old_version_transitions')
        write(jobdir/'job.json',job)
        if current_stop.get('reason')=='known_r25_startup_regression':
            record.update(status='skipped_due_to_startup_regression',reason=current_stop['reason'],
                phase_stop=current_stop['phase_stop'],finished_unix_s=time.time(),
                stages=[dict(level=level,stage_status='skipped_due_to_startup_regression',
                    reason=current_stop['reason']) for level in settings['levels']])
            write(destination,record)
            return 0
        # This pointer is written atomically with a completed PPO update, and
        # carries the producing phase/controller/reward. Never scan old filenames.
        last=collection.get('last_completed_update')
        eligible=(last and last['phase_id']==job['phase_id']
            and last['controller_build']==job['controller_build']
            and last['reward_version']==job['reward_version']
            and last.get('stream_id')==job.get('stream_id') and last['updates']>job['initial_updates'] and last['samples']>job['initial_transitions'])
        if not eligible:
            record.update(status='unavailable',reason='new_update_unavailable',
                finished_unix_s=time.time(),stages=[dict(level=level,stage_status='not_run',
                    reason='new_update_unavailable') for level in settings['levels']])
            write(destination,record)
            return 0
        training_exit=int((jobdir/'training-exit-code.txt').read_text())
        if training_exit!=0 or not current_stop or current_stop.get('last_stage_status')=='technical_interruption':
            record.update(status='unavailable',reason='training_not_normally_completed',
                finished_unix_s=time.time(),stages=[dict(level=level,stage_status='not_run',
                    reason='training_not_normally_completed') for level in settings['levels']])
            write(destination,record)
            return 0
        checkpoint=ROOT/last['path']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if not (saved['phase_id']==job['phase_id'] and saved['controller_build']==job['controller_build']
                and saved['reward_version']==job['reward_version']
                and saved.get('stream_id')==job.get('stream_id') and saved['group']==job['group'] and saved['seed']==job['seed']
                and saved['updates']==last['updates'] and saved['samples']==last['samples']
                and len(saved['buffer'])==0):
            raise ValueError('Completed-update checkpoint does not match its recorded phase identity')
        record.update(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_transitions=saved['samples'],
                      checkpoint_updates=saved['updates'], normalization=saved['normalization'])
        del saved
        write(destination, record)
        previous = None
        for level in settings['levels']:
            if previous is not None and previous['stage_status'] != 'passed':
                row = dict(level=level, stage_status='blocked_by_predecessor',
                           reason=f'{previous["level"]}: {previous["stage_status"]}: {previous.get("reason")}')
            elif time.time() >= deadline:
                row = dict(level=level, stage_status='budget_truncated', reason='Evaluation deadline before launch')
            else:
                attempt = f'{jobdir.name}-eval-{job["group"]}-s{job["seed"]}-{level}'
                policy_output = jobdir / 'evaluation-policy' / attempt
                command = [
                    sys.executable, '-u', 'scripts/run_s1_go2_r12.py',
                    '--attempt', attempt, '--build', job['controller_build'],
                    '--group', job['group'], '--seed', str(job['seed']),
                    '--checkpoint', str(checkpoint), '--level', level,
                    '--condition', settings['condition'], '--workers', str(settings['solver_workers']),
                    '--evaluation-start-unix', str(started), '--deadline-unix', str(deadline),
                    '--policy-output', str(policy_output), '--contract', str(jobdir/'execution-contract.json'),
                ]
                row = dict(level=level, attempt=attempt, stage_status='running',
                           started_unix_s=time.time(), command=command)
                record['attempts'].append(attempt)
                record['stages'].append(row)
                write(destination, record)
                log = jobdir / f'evaluation-{level}.log'
                with log.open('w') as output:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    try:
                        code = process.wait(timeout=max(1, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        # The explicit evaluation budget also covers a stalled RPC/initialization.
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            code = process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            code = process.wait()
                        row.update(stage_status='budget_truncated', reason='Independent evaluation wall deadline')
                result_path = small / (attempt + '.json')
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    row.update({key: result.get(key) for key in (
                        'stage_status', 'success', 'reason', 'error', 'simulation_s', 'wall_s',
                        'stage_progress', 'legal_route', 'completed_round_trips', 'case_key', 'qualification')})
                    row['result'] = str(result_path.relative_to(ROOT))
                elif row['stage_status'] == 'running':
                    row.update(stage_status='technical_interruption', reason=f'No completed result; exit_code={code}; see {log.name}')
                row.update(exit_code=code, finished_unix_s=time.time(), log=str(log.relative_to(ROOT)))
                previous = row
                write(destination, record)
                continue
            record['stages'].append(row)
            previous = row
            write(destination, record)
        record['status'] = ('partial' if any(x['stage_status'] in ('technical_interruption', 'budget_truncated')
                                            for x in record['stages']) else 'completed')
    except Exception as error:
        record.update(status='technical_interruption', error=repr(error))
        for level in settings['levels']:
            row = next((x for x in record['stages'] if x['level'] == level), None)
            if row is None:
                record['stages'].append(dict(level=level, stage_status=(
                    'technical_interruption' if level == 'E30' else 'blocked_by_predecessor'),
                    reason=repr(error)))
            elif row['stage_status'] == 'running':
                row.update(stage_status='technical_interruption', reason=repr(error))
    record['finished_unix_s'] = time.time()
    write(destination, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if record['status'] == 'completed' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', default=CONFIG)
    p.add_argument('--job-dir', type=Path, help='Run final-policy E30/conditional E60 after this job training exits.')
    p.add_argument('--attempt')
    p.add_argument('--build')
    p.add_argument('--group', choices=['FIXED_S5', 'PPO_RAW_S5', 'PANN_PPO_S5'], default='PANN_PPO_S5')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--checkpoint')
    p.add_argument('--level', choices=['E30', 'E60', 'E300', 'E600'], default='E30')
    p.add_argument('--condition', choices=['nominal', 'yaw_plus5', 'yaw_minus5'], default='nominal')
    p.add_argument('--workers', type=int)
    p.add_argument('--window-start-unix', '--evaluation-start-unix', dest='window_start_unix', type=float)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--policy-output', type=Path)
    p.add_argument('--assets', type=Path, default=ROOT / 'work/assets')
    a = p.parse_args()
    contract = json.loads((ROOT / a.contract).read_text())
    if a.job_dir is not None:
        return evaluate_job(a.job_dir, contract)
    if a.attempt is None or a.window_start_unix is None or a.deadline_unix is None:
        p.error('A single evaluation needs --attempt, --evaluation-start-unix and --deadline-unix')
    if a.group != 'FIXED_S5' and a.checkpoint is None:
        p.error('Learned evaluation needs its fixed --checkpoint')
    evaluate(a, contract)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
