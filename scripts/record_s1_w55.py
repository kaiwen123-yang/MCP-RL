"""Record original W55 stages and their wall-time overlap from D57 lane outputs."""
import argparse
import fcntl
import json
import time
from pathlib import Path

from publish_s1_go2_w54 import write

ROOT = Path(__file__).resolve().parents[1]
RUN = 'tm-s1-go2-local-readiness-r12'
REL = Path('results/terrain_memory/simulation') / RUN
LARGE = Path('outputs/terrain_memory/simulation') / RUN
WINDOW = ROOT / REL / 'w55'
LEVELS = ('E30', 'E60', 'E300', 'E600')


def read(path):
    return json.loads(path.read_text())


def select(d, fields):
    return {k: d[k] for k in fields if k in d}


def cases(worktree, lane):
    compact = worktree / REL / 'w55' / lane
    rows = {}
    ledger = compact / 'index.json'
    indexed = {r['attempt']: r for r in read(ledger).get('attempts', [])} if ledger.exists() else {}
    for p in sorted(compact.glob('**/E*.json')):
        d = read(p)
        if p.stem not in LEVELS or not d.get('attempt'):
            continue
        row = select(d, ('attempt', 'job', 'level', 'status', 'tmux', 'runner_pid', 'pid',
                         'user_cpu_s', 'system_cpu_s', 'exit_code', 'started_unix_s',
                         'ended_unix_s', 'reason', 'command', 'resource_usage'))
        row['record'] = str(p.relative_to(worktree))
        row['lane'] = lane.upper()
        case_record = indexed.get(row['attempt'], {})
        if case_record.get('physical_started_unix_s') is not None:
            row['physical_started_unix_s'] = case_record['physical_started_unix_s']
        pid = row.get('pid')
        if row['status'] == 'running' and pid:
            row['pid_running_at_snapshot'] = Path('/proc', str(pid)).is_dir()
            live = worktree / LARGE / row['attempt'] / 'live.json'
            if live.is_file():
                row['live'] = read(live)
        rows[row['attempt']] = row
    for p in sorted(compact.glob('job-*-eval-*-attempt*.json')):
        if p.name.endswith('-resolved.json'):
            continue
        d = read(p)
        if p.stem != d.get('attempt') or 'evaluation_stage' not in d or 'physical_samples' not in d:
            continue
        row = rows.setdefault(d['attempt'], {'attempt': d['attempt'], 'lane': lane.upper()})
        row.update(select(d, ('success', 'stage_status', 'reason', 'error', 'simulation_s',
                              'physical_samples', 'physical_started_unix_s', 'physical_ended_unix_s',
                              'max_x_m', 'min_body_height_m', 'completed_round_trips', 'wall_s',
                              'stage_progress', 'case_key', 'source_commit', 'qualification',
                              'samples_500Hz', 'timing', 'record_timing')))
        row['status'] = d['stage_status']
        row['level'] = d['evaluation_stage']['level']
        row['identity'] = select(d['case_identity'], ('seed', 'controller_build', 'job',
             'checkpoint_sha256', 'normalizer_sha256', 'stream_id', 'initial_state'))
        row['legal_route'] = select(d['legal_route'], ('legal_outbound_traversals',
             'legal_return_traversals', 'completed_roundtrips', 'side_exits'))
        row['material'] = select(d['material'], ('max_abs_principal_strain',
             'loaded_ever_cells', 'events', 'domain_exit'))
        row['result'] = str(p.relative_to(worktree))
        row['server_originals'] = str(worktree / LARGE / row['attempt'])
        video = Path(row['server_originals']) / 'video-asset.json'
        if video.is_file():
            row['recorded_video'] = read(video)['path']
    return list(rows.values())


def record(args):
    f, n, c = [p.resolve() for p in (args.f, args.n, args.c)]
    rows = cases(f, 'f') + cases(n, 'n')
    # r56 has an independent writable native build tree. Its branch also contains
    # the r54 history; take only the r56 execution from that tree.
    n56 = n.parent / 'MCP-RL-w55-n-r56'
    if n56.is_dir():
        rows += [row for row in cases(n56, 'n') if '-w55-n-' in row['attempt']
                 and '-r56-eval-' in row['attempt']]
    done = [r for r in rows if 'physical_samples' in r]
    physical = [r for r in done if r['physical_samples'] > 0]
    overlaps = []
    for a in physical:
        for b in physical:
            if a['attempt'] >= b['attempt']:
                continue
            first = max(a['physical_started_unix_s'], b['physical_started_unix_s'])
            last = min(a['physical_ended_unix_s'], b['physical_ended_unix_s'])
            if last > first:
                overlaps.append(dict(attempts=[a['attempt'], b['attempt']], start=first,
                                     end=last, physical_overlap_wall_s=last-first))
    observed = []
    for row in rows:
        begin = row.get('physical_started_unix_s')
        end = row.get('physical_ended_unix_s')
        if end is None and row.get('pid_running_at_snapshot') and row.get('live', {}).get('physical_time_s', 0) > 0:
            end = row['live']['unix_s']
        if begin is not None and end is not None and end > begin:
            observed.append((row['attempt'], begin, end))
    observed_pairs = []
    for i, (a, a0, a1) in enumerate(observed):
        for b, b0, b1 in observed[i+1:]:
            dt = min(a1, b1) - max(a0, b0)
            if dt > 0:
                observed_pairs.append(dict(attempts=[a,b], wall_s=dt))
    passed = [r for r in done if r['success'] and r['stage_status'] == 'passed']
    highest = next((x for x in reversed(LEVELS) if any(r['level'] == x for r in passed)), None)
    now = time.time()
    phase = dict(status=args.status, updated_unix_s=now, deadline=None, wall_deadline_enabled=False,
                 kill_on_conversation_end=False, cases=rows, completed_physical_overlap=overlaps,
                 observed_physical_overlap=observed_pairs,
                 unfinished_interval_end='latest progressing live heartbeat on D57')
    write(WINDOW/'phase.json', phase)
    inventory = {}
    material_dir = c / REL / 'w55/c'
    for p in sorted(material_dir.glob('*-inventory.json')):
        d = read(p)
        inventory[p.stem.removesuffix('-inventory')] = select(d, ('job', 'role', 'ever_loaded',
                              'new_keys', 'unique_full_boundaries', 'reference_and_assembly_wall_s'))
    index_path = WINDOW / 'index.json'
    index = read(index_path) if index_path.exists() else {}
    index.update(schema='s1_w55_actual_execution_v1', window='W55', run_id=RUN,
        implementation_base='c8e0b38d8256305b9e5a14907829980c3f1b0054',
        preceding_results='1ae25272c86f551f8c6b3a65073bc05b24e3d8f6',
        plan='docs/S1_W55_STABILITY_PLAN.md', status=args.status, updated_unix_s=now,
        phase=str((WINDOW/'phase.json').relative_to(ROOT)),
        workspaces={'F': str(f), 'N': str(n), 'C': str(c), 'integration': str(ROOT)},
        initial_cpu_allocation={'N': '0-9', 'F': '10-13', 'C_and_data': '14-21',
                                'system_and_other_reserved_equivalent_cores': 3,
                                'math_library_threads': 1},
        inspected_limits={'cpu_max': '2500000 100000', 'memory_max_bytes': 128849018880,
                          'gpu': 'RTX PRO 6000 Blackwell Server Edition, 97887 MiB'},
        completed_original_attempts=len(done), valid_original_physical_starts_completed=len(physical),
        completed_physical_steps=sum(r['physical_samples'] for r in done),
        completed_simulation_s=sum(r['simulation_s'] for r in done),
        highest_passed_original_stage=highest, E30_passes=sum(r['level']=='E30' for r in passed),
        original_stage_counts={level:sum(r['level']==level for r in done) for level in LEVELS},
        legal_material_outbound=sum(r['legal_route']['legal_outbound_traversals'] for r in done),
        legal_material_return=sum(r['legal_route']['legal_return_traversals'] for r in done),
        legal_round_trips=sum(r['legal_route']['completed_roundtrips'] for r in done),
        effective_physical_overlap_wall_s=sum(r['physical_overlap_wall_s'] for r in overlaps),
        overlap_scope='Pairs of completed original runs, from first to last actual plant step.',
        overlap_including_live_lower_bound_wall_s=sum(r['wall_s'] for r in observed_pairs),
        completed_attempt_cpu_s=sum(r.get('user_cpu_s',0)+r.get('system_cpu_s',0) for r in done),
        material_boundary_inventory=inventory,
        known_nine_cell_replay=str(REL/'reviews/W54-review/pann-replay-summary.json'),
        causal_material_memory_benefit_evaluated=False,
        PPO={'new_samples':0, 'new_updates':0, 'donor':0}, deadline=None,
        wall_deadline_enabled=False, kill_on_conversation_end=False)
    if n56.is_dir():
        index['workspaces']['N_r56'] = str(n56)
    material_summary = WINDOW/'c/summary.json'
    if material_summary.exists():
        summary = read(material_summary)
        fit = read(WINDOW/'c/fit.json')
        fit_start, fit_end = fit['start_unix'], fit['end_unix']
        physical_and_fit = []
        for i, (a, a0, a1) in enumerate(observed):
            for b, b0, b1 in observed[i+1:]:
                overlap = min(a1,b1,fit_end)-max(a0,b0,fit_start)
                if overlap>0:
                    physical_and_fit.append(dict(attempts=[a,b], overlap_wall_s=overlap))
        index['material_candidate'] = dict(summary=str(material_summary.relative_to(ROOT)),
                 candidate_sha256=summary['weights_candidate_sha256'],
                 adopted_in_robot=summary['decision_path']['candidate_used_in_robot'],
                 historical_complete=summary['historical_complete'],
                 independent_holdouts=['W46 B with 18 world cells', 'W55 F-A r53 with 5 preselected event families'],
                 GPU=summary['gpu'], training=summary['training'],
                 fit_unix_interval=[fit_start,fit_end], two_physical_plus_GPU_fit_overlap=physical_and_fit)
    write(index_path, index)
    print(json.dumps({'status':args.status,'completed':len(done),'highest':highest,
                      'physics_steps':index['completed_physical_steps'],
                      'overlap_wall_s':index['effective_physical_overlap_wall_s']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for lane in ('f', 'n', 'c'):
        parser.add_argument('--'+lane, type=Path, required=True, help=f'W55 {lane.upper()} worktree')
    parser.add_argument('--status', required=True, help='Evidence-based execution or handoff status')
    arguments = parser.parse_args()
    WINDOW.mkdir(parents=True, exist_ok=True)
    with (WINDOW/'index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record(arguments)
