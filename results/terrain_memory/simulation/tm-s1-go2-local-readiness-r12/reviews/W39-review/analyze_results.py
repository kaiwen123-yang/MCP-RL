#!/usr/bin/env python3
"""Recompute W39/W38 compact evidence; no runtime imports, network or physics.

Run from any directory. Only writes results-audit.json beside this file.
Records the current Git HEAD when rerun; the published review baseline is retained separately.
"""
from pathlib import Path
from collections import Counter
import json
import math
import statistics as st
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
BASE = REPO / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
REVIEW_BASELINE = 'e51acd535445f617bd0486650e67df4bcc116481'
HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
READ = {}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    result = ([json.loads(line) for line in raw.decode().splitlines() if line.strip()]
              if path.suffix == '.jsonl' else json.loads(raw))
    READ[str(path.relative_to(REPO))] = {
        'bytes': len(raw),
        'jsonl_rows': len(result) if path.suffix == '.jsonl' else None}
    return result


def summary(values):
    return {'n': len(values), 'min': min(values), 'median': st.median(values),
            'mean': st.mean(values), 'max': max(values)} if values else {'n': 0}


def is_natural(result):
    return result['reason'].startswith(('body_fall', 'invalid_contact', 'material_domain_exit'))


def episode(result, seed, caps, events=None):
    passage = result['stage_progress']['material_traversal']
    first_events = (events or {}).get('first_material_events_by_type', {})
    first_load = first_events.get('load', {}).get('event', {}).get('time_s')
    moves = [x for x in result['transitions'] if x['phase'] in ('outbound', 'return')]
    row = {
        'attempt': result['attempt'], 'seed': seed,
        'source': str(BASE.relative_to(REPO) / (result['attempt'] + '.json')),
        'reason': result['reason'], 'reason_family': result['reason'].split(':')[0],
        'natural_physical_terminal': is_natural(result),
        'terminated': result['policy']['terminated'], 'truncated': result['policy']['truncated'],
        'status': result['status'], 'stage_status': result['stage_status'],
        'error': result['error'], 'simulation_s': result['simulation_s'], 'wall_s': result['wall_s'],
        'max_x_m': result['max_x_m'], 'final_xyz_m': result['final_qpos'][:3],
        'final_heading_error_rad': result['final_heading_error_rad'],
        'min_body_height_m': result['min_body_height_m'],
        'first_load_s': first_load,
        'time_first_load_to_terminal_s': result['simulation_s'] - first_load if first_load is not None else None,
        'material_loaded': result['stage_progress']['actual_material_loaded'],
        'base_entry_s': passage['west_entry_s'], 'east_exit_s': passage['east_exit_s'],
        'lateral_exit_s': passage['lateral_exit_s'],
        'base_entry_and_foot_loading': any(x['entry_s'] is not None and x['material_loaded'] for x in result['legal_route']['passages']),
        'first_far_arrival_s': result['stage_progress']['far_arrival_s'],
        'return_start_s': result['stage_progress']['return_start_s'],
        'legal_outbound': result['legal_route']['legal_outbound_traversals'],
        'legal_return': result['legal_route']['legal_return_traversals'],
        'legal_roundtrips': result['legal_route']['completed_roundtrips'],
        'raw_roundtrips': result['completed_round_trips'],
        'final_legal_progress': result['legal_route']['progress'],
        'terminal_reward_settlement': result.get('terminal_reward_settlement'),
        'max_abs_principal_strain': result['material']['max_abs_principal_strain'],
        'max_material_residual': result['material']['max_residual'],
        'material_domain_exit': result['material']['domain_exit'],
        'final_phase': result['transitions'][-1]['phase'],
        'first_outbound_s': moves[0]['time_s'] if moves else None,
        'effective_translation_s': result['policy']['effective_translation_s'],
        'cap_records': len(caps), 'cap_m_s': summary([x['requested_cap_m_s'] for x in caps]),
        'timing': result['timing'], 'material_wall_s': result['material']['wall_s'],
        'all_joints_ever_at_torque_limit': all(abs(x - lim) < 1e-6 for x, lim in zip(result['max_abs_torque_Nm'], [23.7, 23.7, 45.43] * 4)),
        'any_joint_ever_at_torque_limit': any(abs(x - lim) < 1e-6 for x, lim in zip(result['max_abs_torque_Nm'], [23.7, 23.7, 45.43] * 4)),
        'source_commit': result['source_commit'], 'execution_fingerprint': result['execution_fingerprint'],
        'configuration_fingerprint': result['full_configuration_sha256'],
        'material_event_counts': (events or {}).get('material_event_counts'),
    }
    return row


def aggregate(rows):
    nat = [x for x in rows if x['natural_physical_terminal']]
    wall = sum(x['wall_s'] for x in rows)
    timings = {}
    for key in rows[0]['timing']:
        total = sum(x['timing'][key]['total_s'] for x in rows)
        count = sum(x['timing'][key]['count'] for x in rows)
        timings[key] = {'total_s': total, 'count': count, 'mean_s': total / count,
                        'fraction_of_episode_wall_sum': total / wall,
                        'deadline_misses': sum(x['timing'][key]['deadline_misses'] for x in rows)}
    return {
        'episodes': len(rows), 'natural_physical_terminals': len(nat),
        'reason_counts': dict(Counter(x['reason_family'] for x in rows)),
        'full_invalid_contact_reasons': dict(Counter(x['reason'] for x in rows if x['reason_family'] == 'invalid_contact')),
        'all_error_null': all(x['error'] is None for x in rows),
        'natural_lifetime_s': summary([x['simulation_s'] for x in nat]),
        'natural_max_x_m': summary([x['max_x_m'] for x in nat]),
        'all_lifetime_s': summary([x['simulation_s'] for x in rows]),
        'material_loaded_episodes': sum(x['material_loaded'] for x in rows),
        'base_entry_episodes': sum(x['base_entry_s'] is not None for x in rows),
        'base_entry_and_foot_loading_episodes': sum(x['base_entry_and_foot_loading'] for x in rows),
        'loaded_without_base_entry': sum(x['material_loaded'] and x['base_entry_s'] is None for x in rows),
        'base_entry_without_load': sum(x['base_entry_s'] is not None and not x['material_loaded'] for x in rows),
        'lateral_exit_episodes': sum(x['lateral_exit_s'] is not None for x in rows),
        'legal_outbound': sum(x['legal_outbound'] for x in rows),
        'legal_return': sum(x['legal_return'] for x in rows),
        'legal_roundtrips': sum(x['legal_roundtrips'] for x in rows),
        'raw_roundtrips': sum(x['raw_roundtrips'] for x in rows),
        'far_arrivals': sum(x['first_far_arrival_s'] is not None for x in rows),
        'return_starts': sum(x['return_start_s'] is not None for x in rows),
        'natural_terminal_positive_legal_progress': sum(x['final_legal_progress'] > 0 for x in nat),
        'retracted_progress': sum((x['terminal_reward_settlement'] or {}).get('retracted_positive_credit', 0) for x in rows),
        'first_load_to_terminal_s': summary([x['time_first_load_to_terminal_s'] for x in nat if x['first_load_s'] is not None]),
        'physics_s': sum(x['simulation_s'] for x in rows), 'episode_wall_sum_s': wall,
        'timing': timings, 'material_wall_s_nested_in_physics': sum(x['material_wall_s'] for x in rows),
        'unattributed_wall_s': wall - sum(x['total_s'] for x in timings.values()),
        'natural_any_joint_ever_at_torque_limit': sum(x['any_joint_ever_at_torque_limit'] for x in nat),
        'natural_all_joints_ever_at_torque_limit': sum(x['all_joints_ever_at_torque_limit'] for x in nat),
        'max_abs_principal_strain': max(x['max_abs_principal_strain'] for x in rows),
    }


def numeric(transitions, values):
    mapped = {(x['episode'], x['transition_serial']): x for x in values}
    joined = [(t, mapped[(t['episode'], t['transition_serial'])]) for t in transitions]
    reward_errors = []
    components = Counter()
    for t in transitions:
        c = {'progress': 5 * t['signed_route_fraction'], 'elapsed_time': -t['dt_s'] / 60,
             'indentation': -t['indentation_sq_s'] / 60, 'slip': -t['loaded_slip_sq_s'] / 600,
             'legal_laps': 10 * t['completed_circuits'], 'physical_failure': -10 * t['failure'],
             'intrinsic_timeout': -5 * t['intrinsic_timeout']}
        if t.get('action_change') is not None:
            c['cap_change'] = -.01 * t['action_change']
            reward_errors.append(abs(sum(c.values()) - t['reward']))
        components.update(c)
    nonzero = [(t, v) for t, v in joined if v.get('pann_recovery_m') is not None and any(x != 0 for x in v['pann_recovery_m'])]
    terminals = [(t, v) for t, v in joined if t['terminated']]
    truncations = [(t, v) for t, v in joined if t['truncated']]
    return {
        'rows': len(transitions), 'value_rows': len(values),
        'distinct_episode_serial_pairs': len(mapped),
        'distinct_serials': len(set(x['transition_serial'] for x in transitions)),
        'all_reward_action_value_numbers_finite': all(math.isfinite(x) for t, v in joined for x in
            [t['reward'], t['cap_m_s'], t['raw_action'], t['log_prob']] +
            [v[k] for k in ['value', 'next_value', 'bootstrap_value'] if v.get(k) is not None] +
            (v.get('pann_recovery_m') or []) + (v.get('next_pann_recovery_m') or [])),
        'serial_range': [min(x['transition_serial'] for x in transitions), max(x['transition_serial'] for x in transitions)],
        'phase_counts': dict(Counter(x['phase'] for x in transitions)),
        'behavior_updates_counts': dict(Counter(x['behavior_updates'] for x in transitions)),
        'effective_action_rows': sum(x['effective'] for x in transitions),
        'cap_m_s': summary([x['cap_m_s'] for x in transitions]),
        'raw_action': summary([x['raw_action'] for x in transitions]),
        'reward': summary([x['reward'] for x in transitions]),
        'reward_sum': sum(x['reward'] for x in transitions),
        'recorded_reward_components_sum': dict(components),
        'reward_formula_check': {'checked_rows': len(reward_errors), 'max_absolute_error': max(reward_errors) if reward_errors else None,
                                 'missing_action_change_rows': sum(x.get('action_change') is None for x in transitions)},
        'physical_terminals': len(terminals), 'external_truncations': len(truncations),
        'terminal_bootstrap_values_all_zero': all(v['bootstrap_value'] == 0 for t, v in terminals),
        'truncated_bootstrap_equals_next_value': all(v['bootstrap_value'] == v['next_value'] for t, v in truncations),
        'pann_current_nonzero_rows': len(nonzero), 'pann_current_nonzero_fraction': len(nonzero) / len(transitions),
        'pann_nonzero_episode_count': len(set(t['episode'] for t, v in nonzero)),
        'pann_nonzero_serials': [t['transition_serial'] for t, v in nonzero],
        'pann_nonzero_phase_counts': dict(Counter(t['phase'] for t, v in nonzero)),
        'pann_max_abs_m': max(abs(x) for v in values for x in (v.get('pann_recovery_m') or [0])),
        'pann_next_nonzero_rows': sum(any(x != 0 for x in (v.get('next_pann_recovery_m') or [])) for v in values),
        'value': summary([v['value'] for v in values if v.get('value') is not None]),
        'next_value': summary([v['next_value'] for v in values if v.get('next_value') is not None]),
        'missing_value_fields': dict(Counter(k for v in values for k in v.get('missing_fields', []))),
        'all_saved_array_matched': all(v['saved_array_match'] is True for v in values),
        'missing_reward_components': dict(Counter(k for t in transitions for k in t.get('missing_reward_components', []))),
    }


def gae_arithmetic(transitions, values, update, gamma, gae_lambda):
    """Float64 re-evaluation of saved scalars using tensor_gae boundary semantics.

    The production backend uses float32, so sub-micro numerical agreement is
    descriptive; this does not recompute observations, values or actor outputs.
    """
    mapped = {(x['episode'], x['transition_serial']): x for x in values}
    carry = 0.0
    advantages, deltas = [], []
    for t in reversed(transitions):
        v = mapped[(t['episode'], t['transition_serial'])]
        delta = t['reward'] + gamma * (not t['terminated']) * v['next_value'] - v['value']
        carry = delta + gamma * gae_lambda * (not (t['terminated'] or t['truncated'])) * carry
        advantages.append(carry); deltas.append(delta)
    return {'basis': 'Float64 arithmetic on exported saved scalars, matching src/response_ppo/ppo_backend.py:tensor_gae; production uses float32',
            'gamma': gamma, 'gae_lambda': gae_lambda, 'advantage': summary(advantages),
            'recorded_advantage_mean': update['advantage_mean'],
            'absolute_mean_difference': abs(st.mean(advantages) - update['advantage_mean']),
            'td_residual': summary(deltas)}


def build(cohort):
    streams, allrows, alltrans, allvals, jobs, collections = {}, [], [], [], [], []
    for seed, suffix in [(1, 'a'), (2, 'b')]:
        stamp = '20261105T032354Z' if cohort == 'W39' else '20261104T104730Z'
        jobid = f'job-{stamp}-{cohort.lower()}-{suffix}'
        root = BASE / 'jobs' / jobid
        job = read(BASE / 'jobs' / (jobid + '.json'))
        collection = read(root / 'training/collection.json')
        learning = read(root / 'training/learning.json')
        stopped = read(root / 'training/stopped.json')
        progress = read(root / 'training/progress.json')
        ledger = read(root / 'attempt-ledger.json')
        identity = read(root / 'controller-identity.json')
        contract = read(root / 'execution-contract.json')
        index = read(root / 'numeric-evidence/index.json')
        start, end = (252, collection['episode']) if cohort == 'W39' else (217, 251)
        rows, trans, vals, checks, ppo_configs = [], [], [], [], []
        previous = 7569 if cohort == 'W39' else 6545
        for ep in range(start, end + 1):
            attempt = (f'train-{jobid}-PANN_PPO_S5-s{seed}-ep{ep:05d}' if cohort == 'W39'
                       else f'train-PANN_PPO_S5-s{seed}-ep{ep:05d}')
            result = read(BASE / (attempt + '.json'))
            resolved = read(BASE / (attempt + '-resolved.json'))
            ppo_configs.append(resolved['r12_configuration']['r12_contract']['joint_learning']['ppo'])
            caps = read(root / 'episodes' / attempt / 'cap-decisions.json')
            small = read(root / 'training' / f'episode-{ep:05d}.json')
            evpath = root / 'numeric-evidence/episodes' / (attempt + '-events.json')
            events = read(evpath) if evpath.is_file() else None
            t = read(root / 'numeric-evidence/training' / f'episode-{ep:05d}-transitions.jsonl')
            v = read(root / 'numeric-evidence/training' / f'episode-{ep:05d}-pann-values.jsonl')
            row = episode(result, seed, caps, events)
            row.update(episode=ep, reward_sum=sum(x['reward'] for x in t), transitions=len(t),
                       current_pann_nonzero_rows=sum(any(x != 0 for x in y['pann_recovery_m']) for y in v))
            pann_start_times = [x['time_s'] - x['dt_s'] for x, y in zip(t, v)
                                if any(a != 0 for a in y['pann_recovery_m'])]
            row['first_nonzero_current_pann_action_start_s'] = min(pann_start_times) if pann_start_times else None
            row['terminal_transition'] = t[-1]
            row['terminal_value'] = v[-1]
            count = result['policy']['total_transitions'] - previous
            checks.append({'episode': ep, 'result_counter_delta': count, 'episode_record_transitions': small['transitions'],
                           'caps': len(caps), 'transitions': len(t), 'values': len(v),
                           'counts_match': count == small['transitions'] == len(caps) == len(t) == len(v),
                           'startup_regression_status': small['startup_regression_check']['status']})
            previous = result['policy']['total_transitions']
            rows.append(row); trans.extend(t); vals.extend(v)
        nat = [x for x in rows if x['natural_physical_terminal']]
        half = len(nat) // 2
        evaluation = read(root / 'evaluation.json')
        evalid = evaluation['attempts'][0]
        evalresult = read(BASE / (evalid + '.json'))
        read(BASE / (evalid + '-resolved.json'))
        evalcaps = read(root / 'episodes' / evalid / 'cap-decisions.json')
        epath = root / 'numeric-evidence/episodes' / (evalid + '-events.json')
        evalrow = episode(evalresult, seed, evalcaps, read(epath) if epath.is_file() else None)
        eval_numeric_dir = root / 'numeric-evidence/evaluation' / evalid
        ets, evs = [], []
        for f in sorted(eval_numeric_dir.glob('*-transitions.jsonl')): ets.extend(read(f))
        for f in sorted(eval_numeric_dir.glob('*-pann-values.jsonl')): evs.extend(read(f))
        evalrow['numeric'] = numeric(ets, evs) if ets and evs else None
        depfile = root / 'episodes' / evalid / 'pann-action-dependence.json'
        if depfile.is_file():
            deps = read(depfile)
            diffs = [abs(x['with_pann_cap'] - x['zero_feature_cap']) for x in deps]
            evalrow['pann_action_dependence'] = {'rows': len(deps), 'nonzero_cap_differences': sum(x != 0 for x in diffs),
                                                 'max_absolute_cap_difference_m_s': max(diffs) if diffs else None}
        batches = []
        for update in learning['updates']:
            behavior = update['update_count'] - 1
            bt = [x for x in trans if x['behavior_updates'] == behavior]
            bv = [x for x in vals if x['behavior_updates'] == behavior]
            batches.append({'update': update, 'numeric': numeric(bt, bv),
                            'gae_arithmetic': gae_arithmetic(bt, bv, update, ppo_configs[0]['gamma'], ppo_configs[0]['gae_lambda'])})
        out = {
            'job': jobid, 'episode_range': [start, end],
            'training_counts': job['training'],
            'summary': aggregate(rows), 'numeric': numeric(trans, vals),
            'temporal_comparison': {'basis': 'first and last equal-size halves of natural terminal episodes; middle natural episode omitted when odd; all truncations excluded',
                                    'first_half': aggregate(nat[:half]), 'last_half': aggregate(nat[-half:]),
                                    'omitted_middle_attempt': nat[half]['attempt'] if len(nat) % 2 else None},
            'updates': batches, 'evaluation': evalrow,
            'evaluation_metadata': {k:v for k,v in evaluation.items() if k in ['checkpoint','checkpoint_transitions','checkpoint_updates','workers','deterministic_policy_mean','frozen_normalization','ppo_updates','stages']},
            'coverage_checks': checks,
            'all_sample_counts_match': all(x['counts_match'] for x in checks),
            'learning_malformed_lines': learning['malformed_lines'],
            'all_resolved_ppo_configs_identical': all(p == ppo_configs[0] for p in ppo_configs),
            'resolved_ppo_config': ppo_configs[0],
            'episodes': rows,
            'numeric_schema': index['schema'],
            'numeric_index_training': {k:v for k,v in index['training'].items() if k != 'files'},
            'full_observations_exported': index['full_observations_exported'],
            'source_commit_counts': dict(Counter(x['source_commit'] for x in rows)),
            'configuration_fingerprints': dict(Counter(x['configuration_fingerprint'] for x in rows)),
        }
        diagnostic = root / 'diagnostics/evidence.json'
        if diagnostic.is_file():
            d = read(diagnostic); manifest = read(root / 'diagnostics/manifest.json')
            parts = [part for f in d['files'] for part in f['parts']]
            out['diagnostic_upload_snapshot'] = {
                'status': d['status'], 'file_count': len(d['files']), 'manifest_entries': len(manifest),
                'source_bytes': sum(f['source_size'] for f in d['files']),
                'prepared_transfer_bytes': sum(part['size'] for part in parts),
                'prepared_transfer_parts': len(parts),
                'file_status_counts': dict(Counter(f['status'] for f in d['files'])),
                'part_keys': list(parts[0]) if parts else [],
                'expanded_attempts': d['expanded_attempts'], 'missing': d['missing'],
                'errors': d['errors'], 'retained_locally': d['retained_locally'],
                'automatic_deletion': d['automatic_deletion'], 'deadline': d['deadline']}
        streams[suffix] = out
        allrows.extend(rows); alltrans.extend(trans); allvals.extend(vals)
        jobs.append(job); collections.append(collection)
    start = min(j['training_started_unix_s'] for j in jobs)
    end = max(c['updated_unix_s'] for c in collections)
    span = end - start
    return {
        'streams': streams, 'combined': aggregate(allrows),
        'failure_families': {family: aggregate([x for x in allrows if x['reason_family'] == family])
                             for family in ['body_fall', 'invalid_contact', 'material_domain_exit']},
        'natural_loaded_terminal_within_half_second_of_first_load': sum(x['natural_physical_terminal'] and
            x['time_first_load_to_terminal_s'] is not None and x['time_first_load_to_terminal_s'] <= .5 for x in allrows),
        'material_loaded_but_all_actor_pann_zero_episodes': sum(x['material_loaded'] and
            x['current_pann_nonzero_rows'] == 0 for x in allrows),
        'numeric_combined': {
            'rows': len(alltrans), 'pann_current_nonzero_rows': sum(s['numeric']['pann_current_nonzero_rows'] for s in streams.values()),
            'pann_current_nonzero_fraction': sum(s['numeric']['pann_current_nonzero_rows'] for s in streams.values()) / len(alltrans),
            'reward_sum': sum(t['reward'] for t in alltrans)},
        'throughput': {'basis': '2048 distinct new transitions / shared two-stream elapsed training window',
                       'training_start_unix_s': start, 'training_end_unix_s': end,
                       'shared_elapsed_s': span, 'shared_elapsed_h': span / 3600,
                       'transitions_per_h': len(alltrans) * 3600 / span,
                       'includes_final_evaluation': False,
                       'note': 'Cross-version observation, not a controlled worker-count experiment.'}}


def main():
    report = {'schema_version': 'W39_compact_result_audit_v1', 'source_head': HEAD, 'review_baseline': REVIEW_BASELINE,
              'method': 'Read-only standard-library arithmetic on all W39/W38 individual results, resolved configs, cap decisions, compact transition/value rows, per-episode records and job ledgers. No physics or neural-network recomputation.'}
    report['W39'] = build('W39')
    report['W38'] = build('W38')
    report['lineage'] = {'common_ancestor': {'transitions': 6545, 'updates': 25},
                         'each_branch_after_W38': {'transitions': 7569, 'updates': 29},
                         'each_branch_after_W39': {'transitions': 8593, 'updates': 33, 'trained_transitions': 8448, 'historical_excluded': 145, 'tail': 0},
                         'unique_tree_transitions_after_W39': 10641,
                         'unique_tree_formula': '6545 common + 2*1024 W38 + 2*1024 W39',
                         'not_one_combined_policy': True}
    report['limitations'] = [
        'A and B are independent optimizer continuations from a shared W37 ancestor, not independent from-scratch replications.',
        'Target-complete truncations and B episode272 60-second external truncation are not natural physical failures.',
        'W38 action_change reward component is missing in all2048 exported rows; formula check is scoped to W39.',
        'Saved PANN/value tables are available; full observation vectors and optimizer arrays are not present in Git compact evidence.',
        'W39 selected E30 control/physical/plan windows exist in numeric-evidence; the separate code audit covers them. Training-wide accepted/restoration traces are not present in this numeric export.',
        'A/B episode and transition IDs overlap. All joint counts combine per-stream statistics rather than joining bare numeric IDs across streams.',
        'Diagnostics publication_pending means prepared local source manifest, not verified uploaded/read raw assets.',
        'Observed W38/W39 differences confound controller revision, continued learning and stochastic trajectories; they are not an isolated causal repair effect.',
        'Per-episode planning wall is RPC latency in synchronous offline simulation; logical deadline misses do not by themselves prove physical lag caused falls.',
        'Mean/median/max are descriptive; related continuations and repeated states preclude treating every transition as an independent experimental replicate.'
    ]
    report['coverage'] = {'read_files': len(READ), 'read_bytes': sum(x['bytes'] for x in READ.values()),
                          'files': dict(sorted(READ.items()))}
    out = HERE / 'results-audit.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'read_files': len(READ), 'read_bytes': report['coverage']['read_bytes'],
                      'W39_episodes': report['W39']['combined']['episodes'],
                      'W39_reasons': report['W39']['combined']['reason_counts'],
                      'W39_throughput': report['W39']['throughput'],
                      'W39_numeric': report['W39']['numeric_combined']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
