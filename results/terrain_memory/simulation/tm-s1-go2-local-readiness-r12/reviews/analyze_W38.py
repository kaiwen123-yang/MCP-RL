#!/usr/bin/env python3
"""Recompute the historical W38 review from saved compact JSON evidence only.

No simulator/project-runtime imports, builds, physics runs, network access, or live
server queries are performed. The only write is the selected JSON output file.
Paths supplied through the CLI are interpreted relative to the current directory.

From the repository root (the script also infers its containing repository):
    python results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/analyze_W38.py

For an explicit checkout and output, including the optional historical W37 check:
    python path/to/analyze_W38.py --repo /path/to/MCP-RL --output /path/to/W38.json \
        --w37-transitions /path/to/job-20261103T075623Z__stream-records__transitions.jsonl

The optional file is the existing W37 Release asset linked by reviews/W37.json.
It is not downloaded automatically. Omitting it recomputes all W38 metrics but
marks the historical reward-arithmetic calibration as not run. The committed
W38.json includes that calibration. The fixed_head metadata identifies the
original reviewed snapshot; it does not claim the checkout still has that HEAD.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import statistics as st

BASE = Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[5],
                    help='Repository root; defaults to the checkout containing this script.')
parser.add_argument('--output', type=Path,
                    help='Output JSON; defaults to <repo>/<review directory>/W38.json.')
parser.add_argument('--w37-transitions', type=Path,
                    help='Existing W37 transitions.jsonl Release asset, for the optional historical arithmetic check.')
args = parser.parse_args()
REPO = args.repo.expanduser().resolve()
OUT = args.output.expanduser().resolve() if args.output else REPO / BASE / 'reviews/W38.json'
W37_TRANSITIONS = args.w37_transitions.expanduser().resolve() if args.w37_transitions else None
if W37_TRANSITIONS is not None and not W37_TRANSITIONS.is_file():
    parser.error('--w37-transitions must identify an existing local Release-asset file.')
HEAD = '5e00beca0896bb5517bab8c1dcc441303e5c2098'
READ_FILES = set()


def read(path):
    path = Path(path)
    READ_FILES.add(str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path))
    return json.loads(path.read_text())


def natural(r):
    return r['reason'].startswith(('body_fall', 'invalid_contact', 'material_domain_exit'))


def sources(seed, job):
    prefix = str(BASE / 'jobs' / job)
    return {
        'results': str(BASE / f'train-PANN_PPO_S5-s{seed}-ep{{episode:05d}}.json'),
        'episodes_inclusive': [217, 251],
        'caps': prefix + '/episodes/{attempt}/cap-decisions.json',
        'episode_records': prefix + '/training/episode-{episode:05d}.json',
        'collection': prefix + '/training/collection.json',
        'learning': prefix + '/training/learning.json',
        'progress': prefix + '/training/progress.json',
        'stopped': prefix + '/training/stopped.json',
        'environment': prefix + '/environment.json',
        'fork_identity': prefix + '/training/collection.json#fork_identity',
        'job': str(BASE / 'jobs' / (job + '.json')),
        'attempt_ledger': prefix + '/attempt-ledger.json',
    }


def reward_from_compact(r, caps):
    """Undiscounted episode return from telescoping reward and recorded sufficient statistics.

    W38 has at most initial_stand -> outbound. No per-transition reward/GAE claim.
    First cap applied on outbound is the latest actor decision at the phase transition.
    """
    movements = [x for x in r['transitions'] if x['phase'] in ('outbound', 'return')]
    if len(movements) > 1 or any(x['phase'] == 'return' for x in movements):
        raise ValueError('This compact reconstruction is scoped to the observed single outbound leg.')
    applied = .25
    change = 0.0
    if movements:
        start = movements[0]['time_s']
        active = [max((x for x in caps if x['time_s'] <= start), key=lambda x: x['time_s'])]
        active += [x for x in caps if start < x['time_s'] < r['simulation_s']]
        for x in active:
            cap = x['requested_cap_m_s']
            change += ((cap - applied) / .035) ** 2
            applied = cap
    components = {
        'progress': 5 * r['legal_route']['progress'],
        'elapsed_time': -r['simulation_s'] / 60,
        'indentation': -r['material']['integral_indentation_sq_s'] / 60,
        'slip': -r['material']['integral_slip_sq_s'] / 600,
        'cap_change': -.01 * change,
        'legal_laps': 10 * r['legal_route']['completed_roundtrips'],
        'physical_failure': -10 * int(natural(r)),
    }
    return {'basis': 'formula_reconstruction_from_published_compact_records_not_observed_transition_log',
            'components': components, 'total': sum(components.values())}


def episode_metrics(r, caps, seed, job, first_serial, episode):
    passage = r['stage_progress']['material_traversal']
    settlement = r.get('terminal_reward_settlement')
    movement = [x for x in r['transitions'] if x['phase'] in ('outbound', 'return')]
    initial_time = min(r['simulation_s'], movement[0]['time_s']) if movement else r['simulation_s']
    torque = [abs(x - lim) < 1e-6 for x, lim in zip(r['max_abs_torque_Nm'], [23.7, 23.7, 45.43] * 4)]
    return {
        'episode': episode, 'attempt': r['attempt'],
        'source_result': str(BASE / (r['attempt'] + '.json')),
        'source_caps': str(BASE / 'jobs' / job / 'episodes' / r['attempt'] / 'cap-decisions.json'),
        'basis': 'recomputed_from_each_result_JSON_and_each_cap_record',
        'reason': r['reason'], 'reason_family': r['reason'].split(':')[0],
        'true_physical_terminal': natural(r), 'terminated': r['policy']['terminated'],
        'truncated': r['policy']['truncated'], 'stage_status': r['stage_status'],
        'final_phase': r['transitions'][-1]['phase'],
        'simulation_s': r['simulation_s'], 'episode_wall_s': r['wall_s'],
        'max_x_m': r['max_x_m'], 'final_xyz_m': r['final_qpos'][:3],
        'min_body_height_m': r['min_body_height_m'],
        'material_loaded': r['stage_progress']['actual_material_loaded'],
        'loaded_ever_cells': r['material']['loaded_ever_cells'],
        'world_cells': r['material']['world_cells'],
        'base_west_entry_s': passage['west_entry_s'],
        'base_east_exit_s': passage['east_exit_s'], 'side_exit_s': passage['lateral_exit_s'],
        'material_loaded_after_base_entry': any(x['entry_s'] is not None and x['material_loaded'] for x in r['legal_route']['passages']),
        'legal_outbound_traversals': r['legal_route']['legal_outbound_traversals'],
        'legal_return_traversals': r['legal_route']['legal_return_traversals'],
        'legal_roundtrips': r['legal_route']['completed_roundtrips'],
        'raw_roundtrips': r['completed_round_trips'],
        'far_arrival_s': r['stage_progress']['far_arrival_s'],
        'return_start_s': r['stage_progress']['return_start_s'],
        'terminal_settlement': settlement, 'final_legal_progress': r['legal_route']['progress'],
        'first_serial_inferred_from_episode_counter': first_serial,
        'end_serial': r['policy']['total_transitions'], 'cap_records': len(caps),
        'cap_mean_m_s': st.mean(x['requested_cap_m_s'] for x in caps),
        'cap_min_m_s': min(x['requested_cap_m_s'] for x in caps),
        'cap_max_m_s': max(x['requested_cap_m_s'] for x in caps),
        'initial_stand_s': initial_time, 'outbound_s': r['simulation_s'] - initial_time,
        'effective_translation_s': r['policy']['effective_translation_s'],
        'max_abs_material_strain': r['material']['max_abs_principal_strain'],
        'max_material_residual': r['material']['max_residual'],
        'any_joint_ever_at_torque_limit': any(torque), 'all_joints_ever_at_torque_limit': all(torque),
        'planning_s': r['timing']['planning']['total_s'],
        'planning_count': r['timing']['planning']['count'],
        'startup_planning_s': r['timing']['startup_planning']['total_s'],
        'physics_wall_s': r['timing']['physics']['total_s'],
        'material_wall_s': r['material']['wall_s'],
        'observer_observation_wall_s': r['policy']['observer']['observation_wall_s'],
        'observer_query_wall_s': r['policy']['observer']['query_wall_s'],
        'reconstructed_episode_return': reward_from_compact(r, caps),
    }


def summarize(rows):
    nat = [x for x in rows if x['true_physical_terminal']]
    return {
        'basis': 'recomputed_from_all_included_episode_JSONs',
        'episodes': len(rows), 'natural_episodes': len(nat),
        'reason_counts': dict(Counter(x['reason_family'] for x in rows)),
        'invalid_contact_full_reason_counts': dict(Counter(x['reason'] for x in rows if x['reason_family'] == 'invalid_contact')),
        'end_phase_counts': dict(Counter(x['final_phase'] for x in rows)),
        'physics_s': sum(x['simulation_s'] for x in rows),
        'episode_wall_sum_s': sum(x['episode_wall_s'] for x in rows),
        'natural_mean_lifetime_s': st.mean(x['simulation_s'] for x in nat),
        'natural_median_lifetime_s': st.median(x['simulation_s'] for x in nat),
        'natural_mean_max_x_m': st.mean(x['max_x_m'] for x in nat),
        'natural_median_max_x_m': st.median(x['max_x_m'] for x in nat),
        'natural_best_max_x_m': max(x['max_x_m'] for x in nat),
        'natural_mean_final_x_m': st.mean(x['final_xyz_m'][0] for x in nat),
        'material_loaded_episodes': sum(x['material_loaded'] for x in rows),
        'material_loaded_but_base_never_entered': sum(x['material_loaded'] and x['base_west_entry_s'] is None for x in rows),
        'base_entry_count': sum(x['base_west_entry_s'] is not None for x in rows),
        'base_entry_attempts': [x['attempt'] for x in rows if x['base_west_entry_s'] is not None],
        'base_entry_episodes': [x['episode'] for x in rows if x['base_west_entry_s'] is not None],
        'entry_and_interior_foot_loading_count': sum(x['material_loaded_after_base_entry'] for x in rows),
        'entry_and_interior_foot_loading_attempts': [x['attempt'] for x in rows if x['material_loaded_after_base_entry']],
        'entry_and_interior_foot_loading_episodes': [x['episode'] for x in rows if x['material_loaded_after_base_entry']],
        'base_entry_without_any_material_loading_count': sum(x['base_west_entry_s'] is not None and not x['material_loaded'] for x in rows),
        'base_entry_without_any_material_loading_attempts': [x['attempt'] for x in rows if x['base_west_entry_s'] is not None and not x['material_loaded']],
        'base_entry_without_any_material_loading_episodes': [x['episode'] for x in rows if x['base_west_entry_s'] is not None and not x['material_loaded']],
        'side_exit_count': sum(x['side_exit_s'] is not None for x in rows),
        'side_exit_attempts': [x['attempt'] for x in rows if x['side_exit_s'] is not None],
        'side_exit_episodes': [x['episode'] for x in rows if x['side_exit_s'] is not None],
        'legal_outbound_traversals': sum(x['legal_outbound_traversals'] for x in rows),
        'legal_return_traversals': sum(x['legal_return_traversals'] for x in rows),
        'legal_roundtrips': sum(x['legal_roundtrips'] for x in rows),
        'raw_roundtrips': sum(x['raw_roundtrips'] for x in rows),
        'far_arrivals': sum(x['far_arrival_s'] is not None for x in rows),
        'return_starts': sum(x['return_start_s'] is not None for x in rows),
        'natural_terminal_positive_legal_progress': sum(x['final_legal_progress'] > 0 for x in nat),
        'natural_terminal_settlement_records': sum(x['terminal_settlement'] is not None for x in nat),
        'terminal_settlement_retracted_credit_sum': sum((x['terminal_settlement'] or {}).get('retracted_positive_credit', 0) for x in rows),
        'natural_all_joints_ever_at_torque_limit': sum(x['all_joints_ever_at_torque_limit'] for x in nat),
        'natural_any_joint_ever_at_torque_limit': sum(x['any_joint_ever_at_torque_limit'] for x in nat),
        'max_abs_material_strain': max(x['max_abs_material_strain'] for x in rows),
        'planning_total_s': sum(x['planning_s'] for x in rows),
        'planning_count': sum(x['planning_count'] for x in rows),
        'mean_planning_s': sum(x['planning_s'] for x in rows) / sum(x['planning_count'] for x in rows),
        'planning_share_of_episode_wall': sum(x['planning_s'] for x in rows) / sum(x['episode_wall_s'] for x in rows),
        'outbound_phase_activity_s': sum(x['outbound_s'] for x in rows),
        'initial_stand_s': sum(x['initial_stand_s'] for x in rows),
        'return_phase_activity_s': 0.0,
        'reward_return_basis': 'formula reconstruction, not raw transition/GAE inspection',
        'reconstructed_episode_return_sum': sum(x['reconstructed_episode_return']['total'] for x in rows),
        'reconstructed_natural_return_range': [min(x['reconstructed_episode_return']['total'] for x in nat), max(x['reconstructed_episode_return']['total'] for x in nat)],
        'reconstructed_reward_components': {k: sum(x['reconstructed_episode_return']['components'][k] for x in rows) for k in rows[0]['reconstructed_episode_return']['components']},
    }


streams = {}
all_rows = []
collections = []
jobs = []
all_caps = []
for seed, suffix in [(1, 'a'), (2, 'b')]:
    jobid = 'job-20261104T104730Z-w38-' + suffix
    root = REPO / BASE / 'jobs' / jobid
    job = read(REPO / BASE / 'jobs' / (jobid + '.json'))
    collection = read(root / 'training/collection.json')
    learning_file = read(root / 'training/learning.json')
    progress = read(root / 'training/progress.json')
    stopped = read(root / 'training/stopped.json')
    ledger = read(root / 'attempt-ledger.json')
    environment = read(root / 'environment.json')
    execution_contract = read(root / 'execution-contract.json')
    identity = read(root / 'controller-identity.json')
    collections.append(collection)
    jobs.append(job)
    rows, caps, raw_results = [], [], []
    previous = 6545
    check_rows = []
    for episode in range(217, 252):
        result = read(REPO / BASE / f'train-PANN_PPO_S5-s{seed}-ep{episode:05d}.json')
        decisions = read(root / 'episodes' / result['attempt'] / 'cap-decisions.json')
        small = read(root / 'training' / f'episode-{episode:05d}.json')
        row = episode_metrics(result, decisions, seed, jobid, previous + 1, episode)
        count = result['policy']['total_transitions'] - previous
        check_rows.append({'episode': episode, 'counter_delta': count, 'recorded_transitions': small['transitions'], 'caps': len(decisions), 'counts_match': count == small['transitions'] == len(decisions), 'regression_check': small['startup_regression_check']['status']})
        for i, decision in enumerate(decisions):
            next_time = decisions[i + 1]['time_s'] if i + 1 < len(decisions) else result['simulation_s']
            caps.append({'episode': episode, 'serial_inferred': previous + i + 1,
                         **decision, 'successor_time_s': next_time,
                         'physical_terminal': i == len(decisions) - 1 and natural(result),
                         'external_truncation': i == len(decisions) - 1 and result['policy']['truncated']})
        previous = result['policy']['total_transitions']
        rows.append(row)
        raw_results.append(result)
    all_rows.extend(rows)
    all_caps.extend(caps)
    batches = []
    for index, update in enumerate(learning_file['updates']):
        q = caps[index * 256:(index + 1) * 256]
        batches.append({
            'basis': 'learning.json plus cap sequence grouped using matching per-episode sample counts',
            **update, 'first_serial_inferred': q[0]['serial_inferred'], 'last_serial_inferred': q[-1]['serial_inferred'],
            'first_episode': q[0]['episode'], 'last_episode': q[-1]['episode'],
            'last_transition_successor_time_s_inferred': q[-1]['successor_time_s'],
            'physical_terminals': sum(x['physical_terminal'] for x in q),
            'external_truncations': sum(x['external_truncation'] for x in q),
            'cap_mean_m_s': st.mean(x['requested_cap_m_s'] for x in q),
            'cap_std_m_s': st.stdev(x['requested_cap_m_s'] for x in q),
            'cap_min_m_s': min(x['requested_cap_m_s'] for x in q),
            'cap_max_m_s': max(x['requested_cap_m_s'] for x in q),
            'raw_transition_rewards_available': False,
            'reward_mean': None, 'per_batch_nonzero_pann_observations': None,
        })
    attempt = jobid + f'-eval-PANN_PPO_S5-s{seed}-E30'
    evaluation = read(REPO / BASE / (attempt + '.json'))
    eval_caps = read(root / 'episodes' / attempt / 'cap-decisions.json')
    dependence = read(root / 'episodes' / attempt / 'pann-action-dependence.json')
    eval_summary = {
        'basis': 'recomputed from full E30 result and all cap/action-dependence rows',
        'source_result': str(BASE / (attempt + '.json')),
        'source_action_dependence': str(BASE / 'jobs' / jobid / 'episodes' / attempt / 'pann-action-dependence.json'),
        'reason': evaluation['reason'], 'stage_status': evaluation['stage_status'],
        'simulation_s': evaluation['simulation_s'], 'episode_wall_s': evaluation['wall_s'],
        'max_x_m': evaluation['max_x_m'], 'final_xyz_m': evaluation['final_qpos'][:3],
        'min_body_height_m': evaluation['min_body_height_m'],
        'material_loaded': evaluation['stage_progress']['actual_material_loaded'],
        'loaded_ever_cells': evaluation['material']['loaded_ever_cells'],
        'base_west_entry_s': evaluation['stage_progress']['material_traversal']['west_entry_s'],
        'legal_route': evaluation['legal_route'],
        'terminal_reward_settlement': evaluation['terminal_reward_settlement'],
        'cap_count': len(eval_caps), 'cap_mean_m_s': st.mean(x['requested_cap_m_s'] for x in eval_caps),
        'cap_min_m_s': min(x['requested_cap_m_s'] for x in eval_caps), 'cap_max_m_s': max(x['requested_cap_m_s'] for x in eval_caps),
        'pann_same_state_actions': len(dependence),
        'pann_same_state_equal_caps': sum(x['with_pann_cap'] == x['zero_feature_cap'] for x in dependence),
        'pann_same_state_max_cap_difference_m_s': max(abs(x['with_pann_cap'] - x['zero_feature_cap']) for x in dependence),
        'pann_inference_limit': 'Equal caps do not distinguish zero input features from local actor insensitivity; no counterfactual physics trajectory was run.',
        'checkpoint': job['evaluation']['checkpoint'], 'checkpoint_updates': job['evaluation']['checkpoint_updates'],
        'checkpoint_transitions': job['evaluation']['checkpoint_transitions'],
        'deterministic_policy_mean': job['evaluation']['deterministic_policy_mean'],
        'frozen_normalization': job['evaluation']['frozen_normalization'], 'ppo_updates': job['evaluation']['ppo_updates'],
        'workers': job['evaluation']['workers'], 'case_key': evaluation['case_key'],
        'recorded_checkpoint_sha256': evaluation['case_identity']['checkpoint_sha256'],
        'recorded_normalizer_sha256': evaluation['case_identity']['normalizer_sha256'],
        'later_evaluation_stages': [{'level': x['level'], 'stage_status': x['stage_status'], 'reason': x.get('reason')} for x in job['evaluation']['stages'][1:]],
        'E300_E600': 'not_run_not_automatically_scheduled',
        'reconstructed_episode_return': reward_from_compact(evaluation, eval_caps),
    }
    new_records = [x for x in progress['session_records'] if x.get('episode', 0) > 216 and x.get('attempt')]
    streams['w38-' + suffix] = {
        'job': jobid, 'seed_directory_identifier': seed, 'sources': sources(seed, jobid),
        'fork_identity': collection['fork_identity'],
        'ancestry': {'initial_samples': 6545, 'initial_updates': 25, 'new_samples': len(caps), 'new_updates': len(batches),
                     'final_samples': collection['collected_transitions'], 'final_updates': collection['updates'],
                     'trained_samples': collection['completed_update_transitions'], 'excluded_historical_samples': collection['excluded_old_version_transitions'],
                     'new_exclusions': collection['fork_identity']['new_exclusions'], 'active_tail': collection['unupdated_fragment'],
                     'conservation': '7569 collected = 7424 trained + 145 historical excluded + 0 active tail',
                     'last_completed_update': collection['last_completed_update'], 'latest_resume': progress['checkpoint'],
                     'historical_controller_migration_note': 'The inherited controller_migration block is r25->r26 history; fork_identity is the current r26 ep216 -> r27 lineage. No second ep144 migration is implied.'},
        'summary': summarize(rows),
        'first17_natural': summarize(rows[:17]), 'last17_natural': summarize(rows[17:-1]),
        'trend_partition': {'first_episodes_inclusive': [217, 233], 'last_episodes_inclusive': [234, 250], 'excluded_target_boundary_episode': 251, 'causality': 'Descriptive within-stream split, not randomized or significance-tested.'},
        'cap_all': {'basis': 'all 1024 cap-decisions rows', 'count': len(caps), 'mean_m_s': st.mean(x['requested_cap_m_s'] for x in caps), 'std_m_s': st.stdev(x['requested_cap_m_s'] for x in caps), 'min_m_s': min(x['requested_cap_m_s'] for x in caps), 'max_m_s': max(x['requested_cap_m_s'] for x in caps), 'configured_bounds_m_s': [.18, .25], 'at_either_bound_count': sum(abs(x['requested_cap_m_s']-.18) < 1e-8 or abs(x['requested_cap_m_s']-.25) < 1e-8 for x in caps)},
        'learning_batches': batches,
        'pann_training_exposure': {'basis': 'server-computed job outcome summary from local rollouts; original observation arrays not published or independently re-counted here', 'source': str(BASE / 'jobs' / (jobid + '.json')) + '#training.outcomes', 'saved_observations': job['training']['outcomes']['saved_observations'], 'nonzero_pann_observations': job['training']['outcomes']['nonzero_pann_observations'], 'fraction_nonzero': job['training']['outcomes']['nonzero_pann_observations'] / job['training']['outcomes']['saved_observations'], 'return_observations': job['training']['outcomes']['return_observations'], 'observation_sources_unavailable_reported': job['training']['outcomes']['observation_sources_unavailable']},
        'evaluation': eval_summary,
        'timing': {'basis': 'collection, job timestamps and all result timing fields; no live resource query', 'shared_start_unix_s': job['training_started_unix_s'], 'invocation_start_unix_s': collection['invocation_started_unix_s'], 'collection_start_unix_s': collection['collection_started_unix_s'], 'collection_end_unix_s': collection['updated_unix_s'], 'invocation_wall_s': collection['invocation_wall_s'], 'invocation_wall_h': collection['invocation_wall_s']/3600, 'new_samples_per_invocation_hour': 1024*3600/collection['invocation_wall_s'], 'common_training_deadline_unix_s': job['training_deadline_unix_s'], 'eval_start_unix_s': job['evaluation']['started_unix_s'], 'eval_end_unix_s': job['evaluation']['finished_unix_s'], 'eval_wrapper_wall_s': job['evaluation']['finished_unix_s']-job['evaluation']['started_unix_s'], 'cumulative_physics_s_includes_parent_history': collection['physics_s'], 'sum_startup_planning_s': sum(x['startup_planning_s'] for x in rows), 'sum_physics_wall_s': sum(x['physics_wall_s'] for x in rows), 'sum_material_wall_s': sum(x['material_wall_s'] for x in rows), 'sum_observer_observation_wall_s': sum(x['observer_observation_wall_s'] for x in rows), 'sum_observer_query_wall_s': sum(x['observer_query_wall_s'] for x in rows)},
        'resources_recorded': environment,
        'integrity': {'all_35_cap_counts_match_episode_counter_deltas_and_episode_records': all(x['counts_match'] for x in check_rows), 'per_episode_checks': check_rows, 'regression_check_counts': dict(Counter(x['regression_check'] for x in check_rows)), 'all_36_ledger_attempt_results_available': all(x['result_available'] for x in ledger['attempts']), 'ledger_attempts': len(ledger['attempts']), 'ledger_initialization_failure_records': sum(len(x['initialization_failures']) for x in ledger['attempts']), 'new_session_records': len(new_records), 'learning_malformed_lines': learning_file['malformed_lines'], 'all_result_errors_null': all(x['error'] is None for x in raw_results), 'source_commit_counts': dict(Counter(x['source_commit'] for x in raw_results)), 'execution_fingerprint_counts': dict(Counter(x['execution_fingerprint'] for x in raw_results)), 'configuration_fingerprint_counts': dict(Counter(x['full_configuration_sha256'] for x in raw_results)), 'recorded_controller_binary_sha256': identity['binary_sha256'], 'published_raw_policy_transitions': False, 'published_raw_observations': False},
        'publication': {'status': job['publication_status'], 'scope': job['publication_scope'], 'errors': job['errors'], 'compact_file_count': len(job['compact_evidence']['files']), 'pending_assets': job['pending_assets'], 'bulk_retention_status': job['bulk_retention']['status'], 'bulk_upload_status': job['bulk_retention']['upload_status'], 'cleanup': job['cleanup'], 'local_job_directory': job['bulk_retention']['job_directory'], 'local_training_directory': job['bulk_retention']['shared_training_directory']},
        'episodes': rows,
    }

baseline = read(REPO / BASE / 'reviews/W37.json')
w37 = baseline['cohort_comparison']['cohorts']['W37']
shared = min(x['training_started_unix_s'] for x in jobs)
starts = [x['invocation_started_unix_s'] for x in collections]
ends = [x['updated_unix_s'] for x in collections]
train_span = max(ends) - shared
train_overlap = min(ends) - max(starts)
eval_end = max(x['evaluation']['finished_unix_s'] for x in jobs)
full_span = eval_end - shared
aggregate_rate = 2048 * 3600 / train_span

# Recompute the compact-return formula against all existing W37 transition rewards.
# This is existing-data arithmetic, not a simulator execution or a new test suite.
w37_transition_file = W37_TRANSITIONS
calibration = {'basis': 'existing W37 transition records used only to check reconstruction arithmetic', 'observed_W38_transition_rewards': False}
if w37_transition_file is not None:
    old = defaultdict(list)
    for line in w37_transition_file.read_text().splitlines():
        x = json.loads(line)
        old[x['episode']].append(x)
    errors = []
    for ep in range(145, 217):
        r = read(REPO / BASE / f'train-PANN_PPO_S5-s0-ep{ep:05d}.json')
        c = read(REPO / BASE / 'jobs/job-20261103T075623Z/episodes' / r['attempt'] / 'cap-decisions.json')
        reconstructed = reward_from_compact(r, c)['total']
        errors.append(abs(reconstructed - sum(x['reward'] for x in old[ep])))
    calibration.update(episodes=72, maximum_absolute_return_difference=max(errors), source_release_url=baseline['sources']['w37_transitions']['url'])
else:
    calibration['not_run_reason'] = 'Optional existing W37 Release asset was not supplied; use --w37-transitions to repeat the historical arithmetic check.'

combined = summarize(all_rows)
combined['base_entry_attempts'] = [x['attempt'] for x in all_rows if x['base_west_entry_s'] is not None]
combined['base_entry_episodes'] = None  # Episode numbers alone collide across forks.
combined['entry_and_interior_foot_loading_attempts'] = [x['attempt'] for x in all_rows if x['material_loaded_after_base_entry']]
combined['entry_and_interior_foot_loading_episodes'] = None
combined['side_exit_attempts'] = [x['attempt'] for x in all_rows if x['side_exit_s'] is not None]
combined['side_exit_episodes'] = None
combined['base_entry_without_any_material_loading_episodes'] = None
combined['episode_id_fields_note'] = 'Bare episode IDs collide across the two streams. Combined *_episodes ID arrays are null intentionally; use explicit *_count and full stream/seed-qualified *_attempts instead. Within-stream summaries retain their numeric episode lists.'

document = {
    'schema_version': 'W38_existing_evidence_review_v1',
    'review': {'repository': 'kaiwen123-yang/MCP-RL', 'fixed_head': HEAD, 'date_utc': '2026-11-05', 'mode': 'read_only_existing_repository_records', 'training_result_JSONs': 70, 'evaluation_result_JSONs': 2, 'training_cap_rows_read': len(all_caps), 'learning_updates_read': 8, 'new_physics_or_tests_run': False, 'SSH_or_live_server_queries': False, 'repository_modified': False},
    'metric_basis': {'episode_counts_routes_lifetimes_timing': 'recomputed from all individual result JSONs', 'cap_statistics': 'recomputed from all per-episode cap-decisions JSON rows; counts checked against actual episode transition deltas', 'update_statistics': 'read all eight saved learning updates; cap batches reconstructed from exact per-episode sample count alignment', 'PANN_training_nonzero_counts': 'server-computed job summary, not an independent count from unavailable observations', 'reward_episode_totals': ('formula reconstruction from published sufficient statistics, calibrated on all W37 raw transition sums; W38 transition/GAE records were not available' if W37_TRANSITIONS is not None else 'formula reconstruction from published sufficient statistics; optional W37 calibration not run in this invocation; W38 transition/GAE records were not available'), 'return_phase': 'No return or far arrival in either branch; effective_translation counts phase activity, not useful positive progress', 'comparisons': 'descriptive; different controllers/reward versions/worker counts/fork depth prevent causal speed or learning attribution'},
    'streams': streams,
    'combined_new_work': combined,
    'ancestry_across_forks': {'basis': 'both collection.fork_identity plus each branch learning/sample ledger', 'common_parent_samples_count_once': 6545, 'common_parent_updates': 25, 'new_samples_per_fork': [1024, 1024], 'distinct_new_samples': 2048, 'new_update_events': 8, 'new_updates_per_policy': 4, 'each_policy_final': {'collected': 7569, 'updates': 29, 'trained': 7424, 'excluded_historical': 145, 'tail': 0}, 'unique_sample_history_across_tree': 8593, 'tree_count_note': '8593 is shared-parent-plus-two-new-branches data history, not one 8593/33 model. The two policies both end at 7569/29 and do not share optimizer or new on-policy buffer by their recorded fork contract. Checkpoint tensor payloads were not inspected.'},
    'parallel_timing_and_resources': {'basis': 'recomputed from invocation start/end, common job start and evaluation timestamps', 'shared_training_start_unix_s': shared, 'shared_training_start_utc': datetime.fromtimestamp(shared, timezone.utc).isoformat(), 'both_training_completed_unix_s': max(ends), 'both_training_completed_utc': datetime.fromtimestamp(max(ends), timezone.utc).isoformat(), 'training_only_common_elapsed_s': train_span, 'training_only_common_elapsed_h': train_span/3600, 'training_overlap_s': train_overlap, 'training_overlap_h': train_overlap/3600, 'overlap_fraction_of_invocation_span': train_overlap/(max(ends)-min(starts)), 'distinct_new_training_samples_per_common_hour': aggregate_rate, 'training_plus_both_E30_common_elapsed_s': full_span, 'training_plus_both_E30_common_elapsed_h': full_span/3600, 'samples_per_hour_if_including_E30_not_training_throughput': 2048*3600/full_span, 'first_E30_overlap_remaining_second_training_s': max(0, ends[1]-jobs[0]['evaluation']['started_unix_s']), 'both_E30_overlap_s': min(x['evaluation']['finished_unix_s'] for x in jobs)-max(x['evaluation']['started_unix_s'] for x in jobs), 'recorded_workers_per_stream': [8, 8], 'combined_configured_MPC_workers': 16, 'recorded_torch_threads_per_stream': [1, 1], 'recorded_blas_threads_per_stream': [1, 1], 'allocated_vCPU_from_contract': 25, 'GPU_count_from_contract': 1, 'CPU_GPU_utilization_measurements_available': False, 'W37_training_samples_per_hour': w37['transitions_per_h'], 'aggregate_training_throughput_ratio_to_W37': aggregate_rate/w37['transitions_per_h'], 'aggregate_training_elapsed_ratio_to_W37_same_new_sample_count': train_span/w37['phase_elapsed_s'], 'common_deadline_same_for_both': jobs[0]['training_deadline_unix_s']==jobs[1]['training_deadline_unix_s'], 'common_training_budget_s': jobs[0]['training_deadline_unix_s']-shared, 'interpretation': 'W38 under the two-by-eight layout observed about 1.305x aggregate sample throughput, not two times and not eight updates on one policy. Controller, reward and trajectory changes prevent isolating the gain as a parallel-layout causal effect. Both streams overlap for nearly the entire training span. Per-stream planning dominates about 88% of episode wall. No measured evidence establishes that two-by-ten workers is faster or safe against framework thread contention.', 'next_resource_choice': 'Retain the observed 2x8 setting unless a separately justified implementation/resource change needs 2x10. Do not introduce a worker grid or standalone benchmark. 20 solver workers would leave five of 25 allocated vCPU for both runtime/RPC/framework/I/O processes; actual internal thread counts/utilization were not measured.'},
    'W37_comparison': {'source': str(BASE / 'reviews/W37.json'), 'source_review_head': baseline['review']['fixed_head'], 'metrics': w37, 'E30': {k: baseline['final_evaluation'][k] for k in ['simulation_s','max_x_m','final_xy_m','reason']}, 'PANN_nonzero': 62, 'PANN_total': 2048, 'PANN_basis': 'independently counted from eight W37 rollouts in the published prior review'},
    'reward_reconstruction': {'method': 'R_episode = 5*final_legal_progress - simulation_s/60 - indentation_integral/60 - slip_integral/600 - .01*sum((new_applied_cap-old_applied_cap)/.035)^2 +10*legal_roundtrips -10*physical_failure. First applied cap is .25 before outbound; actor decisions during initial stand are not applied until outbound starts.', 'source_policy': 'scripts/s1_go2_r11_policy.py#PolicyMaterial.collect_pending,before_physics,finish', 'source_material': 'scripts/s1_go2_joint_material.py#JointMaterial.__init__,before_physics,after_physics,finish', 'W37_existing_data_arithmetic_check': calibration, 'W38_scope_limit': 'This provides undiscounted episode returns and reward-component totals, not observed per-transition rewards, GAE values, or per-update reward means. Those are explicitly null/unavailable.', 'terminal_settlement_conclusion': 'All 68 true physical terminals have a recorded settlement and zero final legal credit; 67 retract positive credit, while w38-b ep218 already lost credit on a side exit. The two target truncations do not receive physical settlement. The source performs settlement before the final collect_pending.', 'causal_limit': 'Correct terminal accounting did not produce legal task success. No claim that reward cancellation is the physical cause of the residual falls or nonfoot contact.'},
    'data_and_metadata_findings': [
        {'kind': 'evidence_gap', 'finding': 'Low-rate training/evaluation transitions and original observations are not in the W38 compact job directories. PANN per-update counts, actual per-transition reward cancellation and GAE cannot be independently audited from this checkout.', 'minimal_followup': 'Publish existing low-rate transition files or equivalent compact rows under each job, plus per-update nonzero PANN counts computed from retained local rollouts. No new physics, bulk checkpoint/video upload or monitoring framework is needed.'},
        {'kind': 'stale_current_metadata', 'finding': 'AGENTS.md and S5-r27 build spec retain joint_sampling_running/startup qualification wording. The fixed HEAD job/collection/evaluation records show both computations ended. Actual records take precedence.', 'paths': ['AGENTS.md','experiments/terrain_memory/s1-go2-r11-S5-r27.json']},
        {'kind': 'ancestry_disambiguation', 'finding': 'job.training.controller_migration is inherited r25->r26 history. Current r26 ep216 -> r27 fork identity is saved in each collection; do not infer a repeated ep144 restoration or re-exclude 145.'},
        {'kind': 'target_boundary_disambiguation', 'finding': 'w38-b ep251 stops at 2.0 s before outbound at the sample target. Its tiny max x is not recurrence of r25 startup regression. Both ep251 target truncations must be excluded from natural-failure lifetime trends.'},
        {'kind': 'identity_scope', 'finding': 'Within each stream all result configuration/execution fingerprints remain constant, and both streams record the same r27 binary. Different seed/stream fingerprints across forks are expected identity differences, not evidence of changed physics.'},
        {'kind': 'publication', 'finding': 'Both jobs are published compact_git with no errors or pending assets. All 72 actual training/evaluation attempts have results and no initialization failure records. Raw data remains local and automatic cleanup is disabled; publication does not prove user backup.'},
    ],
    'next_experiment_implications': [
        'Do not claim convergence or choose a qualified winner: each branch has four new updates, both final E30 fail before strip entry, and neither branch has any legal traversal/return.',
        'A smoothing of the artificial spatial risk gate plus terminal reward settlement is insufficient to solve the task. Training shows somewhat more entry than W37 but no deeper maximum reach or late-learning improvement; merely repeating the same two-by-1024 schedule has low diagnostic value.',
        'Use existing event-window data to locate the earliest rigid-approach posture/support or feedback loss before changing the controller. New planner/feedback diagnostic logs are referenced by resolved records but not present in compact Git; do not assert a newly observed K spike without reading those retained logs.',
        'Keep the full joint loop, original material .02 domain, torque/contact/fall/route rules and last-update E30/E60 selection. Do not relax thresholds or add isolated-MPC/rigid admission tests, an attempt cap, or parameter/worker matrices.',
        'PANN remains sparsely exposed and there is no return history. Same-state feature zeroing changes none of the 56 evaluation caps; this supports no observed local action effect on these trajectories, not a general claim that material memory cannot help.',
        'Retain 2x8 for comparable observed resource use unless the next justified change explicitly requires otherwise; no data currently support a promised gain from 2x10.',
    ],
}
document['coverage'] = {
    'repository_inventory_at_reviewed_head': {
        'fixed_head': HEAD,
        'tracked_files': 4511,
        'tracked_file_bytes': 454382972,
        'non_UTF8_files': 334,
        'basis': 'Repository-wide byte inventory reported by the parent review at the fixed HEAD; separate from this numerical reproducer input list.',
        'scope_limit': 'A complete file/byte inventory is not a claim of word-by-word semantic understanding of every file. Binary/non-UTF8 contents were not treated as readable text. This W38 review independently parses the specifically listed compact evidence and names its remaining raw-data gaps.',
    },
    'repo_JSON_files_parsed': len([x for x in READ_FILES if not x.startswith('/')]),
    'all_read_sources': sorted(READ_FILES),
    'W38_scope': 'All 70 new training results, both final E30 results, all 2048 training cap decisions, eight saved learning updates, and their compact ledgers/contracts. Historical W37 comparison is identified separately.',
    'review_publication_note': 'review.repository_modified describes the read-only evidence analysis at fixed_head; later saving this review and its reproducer adds no claim of code, environment, physics, or server changes.',
    'dense_arrays_embedded': False,
    'favorable_episode_selection': False,
}
OUT.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
json.loads(OUT.read_text())
print(json.dumps({'file': str(OUT), 'bytes': OUT.stat().st_size, 'training_rows': len(all_rows), 'caps': len(all_caps), 'common_training_hours': train_span/3600, 'common_train_eval_hours': full_span/3600, 'aggregate_training_samples_per_h': aggregate_rate, 'natural_failures': combined['natural_episodes'], 'legal_outbound': combined['legal_outbound_traversals']}, ensure_ascii=False))
