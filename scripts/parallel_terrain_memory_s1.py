#!/usr/bin/env python3
"""Run remaining independent S1 missions without changing their execution code."""
import argparse
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'experiments/terrain_memory/s1-control-repair.json'
RUN = 'tm-s1-control-repair-20261024'
SMALL = ROOT/'results/terrain_memory/simulation'/RUN
INDEX = SMALL/'index.json'
WORK = ROOT/'work/s1-parallel'


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_runner():
    spec = importlib.util.spec_from_file_location('s1_existing_runner', ROOT/'scripts/run_terrain_memory_s1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args):
    runner = load_runner()
    name = f'{args.terrain}-{args.controller}-seed-{args.seed}'
    cohort = 'parallel-v1-'+name
    command = [sys.executable, str(Path(__file__)), *sys.argv[1:]]

    def start(config, original_args, seed, fingerprint):
        save(WORK/(name+'-start.json'), dict(terrain=args.terrain, controller=args.controller,
            seed=seed, cohort=cohort, execution_fingerprint=fingerprint, pid=os.getpid(),
            command=command,
            requested_utc=datetime.now(timezone.utc).isoformat()))
        if sys.stdin.readline().strip()!='registered':
            raise RuntimeError('The coordinator did not register this physical start.')

    def result(config, original_args, seed, record_path):
        save(WORK/(name+'-result.json'), dict(record=str(record_path)))

    runner.register_start, runner.register_result = start, result
    sys.argv = [str(ROOT/'scripts/run_terrain_memory_s1.py'), '--config', CONFIG,
        '--terrain', args.terrain, '--controller', args.controller, '--seeds', str(args.seed), '--cohort', cohort]
    runner.main()


def coordinate(args):
    WORK.mkdir(exist_ok=True)
    config = json.loads((ROOT/CONFIG).read_text())
    original = json.loads(INDEX.read_text())
    legacy = next(a for a in original['attempts'] if a['status']=='started')
    process_rows = [line.strip().split(None, 2) for line in subprocess.check_output(
        ['ps', '-eo', 'pid,ppid,args'], text=True).splitlines()[1:]]
    parent = next(int(pid) for pid, ppid, cmd in process_rows
                  if cmd.startswith('python scripts/run_terrain_memory_s1.py --matrix --phase B'))
    child = next(int(pid) for pid, ppid, cmd in process_rows if int(ppid)==parent and '--seeds' in cmd)
    legacy_record = ROOT/config['outputs']['compact_root']/legacy['cohort']/(
        f"{legacy['terrain']}-{legacy['controller']}-seed-{legacy['seed']}.json")
    legacy_key = (legacy['terrain'], legacy['controller'], legacy['seed'])
    os.kill(parent, signal.SIGSTOP)
    legacy_done = False
    pause_seconds = 0.
    launches = []

    def transaction(change):
        nonlocal legacy_done, pause_seconds
        stopped = False
        start = time.monotonic()
        if not legacy_done:
            state = Path(f'/proc/{child}/status')
            if state.exists() and '\nState:\tZ' not in state.read_text():
                os.kill(child, signal.SIGSTOP)
                while not any(line.startswith('State:') and ('T ' in line or 'Z ' in line)
                              for line in state.read_text().splitlines()):
                    time.sleep(.001)
                stopped = True
            # The old worker does not participate in the coordinator's writes.
            # If its result exists, let that one final original index write finish.
            if legacy_record.exists():
                if stopped:
                    os.kill(child, signal.SIGCONT)
                    stopped = False
                while state.exists() and '\nState:\tZ' not in state.read_text():
                    time.sleep(.02)
                legacy_done = True
                row = json.loads(legacy_record.read_text())
                row['resource_context'] = dict(mode='serial_then_parallel', maximum_workers=args.workers,
                    timing_comparability='This already-running mission continued during concurrent tasks; see index.parallel_execution for the switch time and short ledger-coordination pauses.')
                save(legacy_record, row)
        index = json.loads(INDEX.read_text())
        change(index)
        index['parallel_execution']['legacy_coordination_pause_s'] = pause_seconds
        save(INDEX, index)
        if stopped:
            os.kill(child, signal.SIGCONT)
            pause_seconds += time.monotonic()-start

    def initialize(index):
        index['parallel_execution'] = dict(
            authorized_utc=datetime.now(timezone.utc).isoformat(), maximum_workers=args.workers,
            maximum_long_prediction_workers=args.long_workers, logical_cpu_count=os.cpu_count(),
            orchestration_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            legacy_matrix_pid=parent, retained_active_pid=child, retained_active_attempt_id=legacy['attempt_id'],
            physical_execution='Unmodified run_terrain_memory_s1.py and s1*.py; same configuration, controller, RNG seeds and physical step.',
            bookkeeping='One coordinator writes the existing index. Workers use separate output cohorts and return their actual result paths. No extra physical tasks or retries.',
            timing_scope='Previously completed attempts were serial. The retained active attempt switches from serial to shared resources at authorized_utc; new workers share CPU/GPU/memory. These wall times are not isolated per-method speed comparisons or NX real time.',
            legacy_coordination_pause_s=0., status='running', command=[sys.executable, *sys.argv])
        index['phase_B_execution_status'] = 'remaining independent missions running concurrently in tmux'

    transaction(initialize)
    queue = [(terrain, method, seed) for terrain in config['phase_B']['terrain_ids']
             for method in config['phase_B']['controller_ids'] for seed in config['phase_B']['paired_optimizer_seeds']
             if (terrain, method, seed)!=legacy_key and
             not (SMALL/'formal-v1'/f'{terrain}-{method}-seed-{seed}.json').exists()]
    queue.sort(key=lambda c: (c[1]!='B3', c[0]!='rigid', c[2]))
    active = {}
    errors = []
    print(json.dumps(dict(stage='parallel_started', retained=legacy_key, queued=queue,
                          workers=args.workers, long_workers=args.long_workers)), flush=True)
    while queue or active or not legacy_done:
        if not legacy_done and legacy_record.exists():
            transaction(lambda index: None)
        for name, item in list(active.items()):
            start_path = WORK/(name+'-start.json')
            if not item['registered'] and start_path.exists():
                request = json.loads(start_path.read_text())

                def register(index):
                    if len(index['attempts'])>=config['budget']['maximum_new_mission_attempts_including_development_failures_and_superseded_runs']:
                        raise RuntimeError('The authorized total physical-attempt budget is exhausted.')
                    attempt = dict(attempt_id=len(index['attempts'])+1, kind='formal', status='started',
                        started_utc=datetime.now(timezone.utc).isoformat(), **request,
                        resource_context=dict(mode='parallel', maximum_workers=args.workers,
                            active_workers_at_registration=len(active)+int(not legacy_done),
                            timing_comparability='shared CPU/GPU/memory; not an isolated controller benchmark'))
                    index['attempts'].append(attempt)
                    index['new_mission_starts'] = len(index['attempts'])
                    item['attempt_id'] = attempt['attempt_id']

                transaction(register)
                item['process'].stdin.write('registered\n')
                item['process'].stdin.flush()
                item['registered'] = True
                launches.append(request)
                print(json.dumps(dict(stage='physical_start', case=name, attempt=item['attempt_id'], pid=request['pid'])), flush=True)
            code = item['process'].poll()
            if code is None:
                continue
            item['log'].close()
            marker = WORK/(name+'-result.json')
            if code!=0:
                errors.append(dict(case=name, process_exit_code=code))
                if item['registered']:
                    def failed(index):
                        attempt = next(a for a in index['attempts'] if a['attempt_id']==item['attempt_id'])
                        attempt.update(status='execution_error', process_exit_code=code)
                    transaction(failed)
                print(json.dumps(dict(stage='execution_error', case=name, process_exit_code=code)), flush=True)
            else:
                record = Path(json.loads(marker.read_text())['record'])
                row = json.loads(record.read_text())
                trace = ROOT/row['trajectory']
                row.update(events_path=str(trace.parent/(name+'-events.json')),
                    plans_path=str(trace.parent/(name+'-plans.json')), model_path=str(trace.parent/'go2.mjb'),
                    actual_execution_record=str(record),
                    resource_context=dict(mode='parallel', maximum_workers=args.workers,
                        timing_comparability='shared CPU/GPU/memory; not an isolated controller benchmark'))
                alias = SMALL/'formal-v1'/(name+'.json')
                save(alias, row)

                def finished(index):
                    attempt = next(a for a in index['attempts'] if a['attempt_id']==item['attempt_id'])
                    attempt.update(status='finished', record=str(alias.relative_to(ROOT)),
                        actual_execution_record=str(record), success=row['success'], termination=row['termination'],
                        duration_s=row['duration_s'], trajectory=row['trajectory'], sha256=row['sha256'])

                transaction(finished)
                print(json.dumps(dict(stage='finished', case=name, success=row['success'], termination=row['termination'],
                    duration_s=row['duration_s'], wall_s=row['wall_s'])), flush=True)
            del active[name]
        long_count = int(not legacy_done and legacy['controller']=='B3')+sum(x['case'][1]=='B3' for x in active.values())
        while queue and len(active)+int(not legacy_done)<args.workers:
            candidates = [i for i, case in enumerate(queue) if case[1]!='B3' or long_count<args.long_workers]
            if not candidates:
                break
            terrain, method, seed = queue.pop(candidates[0])
            name = f'{terrain}-{method}-seed-{seed}'
            log = (WORK/(name+'.log')).open('w')
            command = [sys.executable, str(Path(__file__)), '--worker', '--terrain', terrain,
                       '--controller', method, '--seed', str(seed)]
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
            active[name] = dict(process=process, log=log, case=(terrain, method, seed), registered=False)
            long_count += method=='B3'
        time.sleep(1)
    def complete(index):
        index['parallel_execution'].update(status='finished' if not errors else 'execution_errors',
            finished_utc=datetime.now(timezone.utc).isoformat(), actual_worker_launches=len(launches),
            legacy_coordination_pause_s=pause_seconds, execution_errors=errors)
    transaction(complete)
    if errors:
        raise RuntimeError('Actual worker errors retained; original matrix remains stopped to prevent uncounted retries: '+json.dumps(errors))
    # The existing matrix now sees saved identical-fingerprint formal records,
    # reuses them without resetting a physical task, and triggers its summary.
    os.kill(parent, signal.SIGCONT)
    print('All remaining physical missions finished. Existing matrix resumed to reuse their records and trigger the queued artifacts/publication.', flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--terrain')
    parser.add_argument('--controller')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--long-workers', type=int, default=6)
    args = parser.parse_args()
    worker(args) if args.worker else coordinate(args)
