#!/usr/bin/env python3
"""Read committed W49 JSON only; never simulate, contact D57, or download payloads."""
import argparse
import collections
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def compact_call(row):
    b, c = row['boundary'], row['call_record']
    state, mission, solve = b['actual_state'], b['mission_actor'], b['current_solve']
    parts = solve['cost_components']
    return {
        'time_s': b['origin_s'], 'call_id': b['call_id'],
        'xyz_m': state[:3], 'linear_velocity_m_s': state[19:22],
        'command_velocity_m_s': mission['command_velocity'],
        'mission_body_target_m': mission['body_target'],
        'mission_body_target_note': 'Not the active r26 MPC local head reference.',
        'cap_m_s': mission['cap'],
        'actual_r26_userdata_packet': None,
        'actual_r26_packet_availability': 'Context NPZ is an external original, not embedded in Git JSON.',
        'J': solve['J'], 'J0': solve['J0'], 'Jref': solve['Jref'],
        'cost_components': parts,
        'cost_component_fractions': {k: v / solve['J'] if solve['J'] else None for k, v in parts.items()},
        'cost_recompute_error': solve['cost_recompute_error'],
        'risk': solve['risk'], 'candidate_source': solve['candidate_source'],
        'return_reason': solve['return_reason'], 'stop_reason': solve['stop_reason'],
        'policy_age_s': c['policy_age_s'],
        'p0_generation_time_s': b['P0']['generation_time_s'],
        'external_snapshot_scope': mission['external_snapshot_scope'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True, help='Path to the MCP-RL repository root')
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('analysis.json'))
    args = ap.parse_args()
    base = args.repo / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
    w49 = base / 'w49'
    out = {
        'schema': 'independent_w49_committed_evidence_review_v1',
        'reviewed_head': 'de6606dc8363e8ab1c816fa682cace752a06318a',
        'plan_base': 'b142001624cfea5877c57e63f45e0f29c9dc4735',
        'scope': 'Git JSON/source read-only review; no D57 login, physics, GPU stepping, or external payload download.',
        'limits': [
            'Full 3042 boundary JSONL rows and full trajectories are external originals; Git embeds only 231 event calls.',
            'Counts over all calls below come from committed run summaries, not independent full-row recounts.',
            'Complete event context/policy/accepted NPZ payloads are outside Git; manifests do not substitute for payload inspection.',
            'No interpolation across missing timeline periods; absent values remain null.',
            'Current remote release existence is separately checked by the root reviewer.',
        ],
        'originals': {}, 'deliveries': {},
    }
    streams = {}
    for suffix, seed in [('a', 1), ('b', 2)]:
        name = f'job-20261113T145020Z-w49-{suffix}-eval-PANN_PPO_S5-s{seed}-E30-attempt01'
        result = read(base / (name + '.json'))
        evidence = read(w49 / 'original-evidence' / (name + '.json'))
        resolved = read(base / (name + '-resolved.json'))
        stream = read(w49 / f'w49-{suffix}.json')
        streams[suffix] = stream
        rec = resolved['r29_recording']
        calls = rec['event_calls']
        last = max(calls, key=lambda x: x['boundary']['origin_s'])
        timeline = []
        for t in [0, 2, 4, 6, 10, 20, 30]:
            matches = [x for x in calls if abs(x['boundary']['origin_s'] - t) < 1e-6]
            found = max(matches, key=lambda x: x['boundary']['call_id']) if matches else None
            timeline.append({'requested_s': t, 'record': compact_call(found) if found else None,
                             'availability': 'embedded_event_boundary' if found else 'not_embedded_in_git'})
        event_timeline = []
        requested = [4.4, 4.6, 4.7, 4.8, 6.2, 7.3, 7.6, 12.2, 29.9, 30]
        for t in requested:
            matches = [x for x in calls if abs(x['boundary']['origin_s'] - t) < 1e-6]
            if matches:
                event_timeline.append(compact_call(matches[-1]))
        selected = [
            'success', 'reason', 'stage_status', 'simulation_s', 'wall_s', 'max_x_m',
            'min_body_height_m', 'final_qpos', 'final_qvel', 'max_joint_limit_excess_rad',
            'max_abs_torque_Nm', 'legal_route', 'stage_progress', 'timing',
            'samples_500Hz', 'physical_samples', 'transitions', 'mujoco_warning_counts',
            'agent_server_warning_lines', 'qualification',
        ]
        o = {k: result[k] for k in selected}
        o['evidence_source'] = str((w49 / 'original-evidence' / (name + '.json')).relative_to(args.repo))
        o['result_source'] = str((base / (name + '.json')).relative_to(args.repo))
        o['resolved_source'] = str((base / (name + '-resolved.json')).relative_to(args.repo))
        o['material'] = {k: result['material'][k] for k in [
            'max_abs_principal_strain', 'max_stress_original_MPa', 'max_residual',
            'domain_exit', 'world_cells', 'loaded_ever_cells', 'events',
            'integral_indentation_sq_s', 'integral_slip_sq_s',
        ]}
        o['active_translation_duration_s'] = result['policy']['effective_translation_s']
        o['net_x_progress_per_active_s'] = (result['final_qpos'][0] - result['transitions'][1]['position_m'][0]) / result['policy']['effective_translation_s']
        o['final_speed_m_s'] = math.sqrt(sum(x*x for x in result['final_qvel'][:3]))
        o['clearance_above_body_fall_height_m'] = result['min_body_height_m'] - 0.12
        for k in ['planner_calls', 'positive_installs', 'positive_same_problem_cost_descent',
                  'positive_risk_priority_cost_increase', 'internal_stagnation', 'work_completed_calls',
                  'prediction_pairs', 'prediction_unavailable', 'diagnostic_events',
                  'prediction_error_extrema', 'same_attempt_record_readback']:
            o[k] = evidence[k]
        o['internal_stagnation_interpretation'] = 'Literal count of native_search_stagnation; does not count backward_failed or w49_single_first_direction_finished.'
        o['positive_install_interpretation'] = 'Only alpha_/k_guided_ sources with policy_updated and chosen_alpha>0; not every installation.'
        o['embedded_event_calls'] = len(calls)
        o['external_object_count'] = len(rec['objects'])
        o['external_object_bytes'] = rec['actual_bytes']
        o['event_windows'] = rec['merged_windows']
        o['embedded_event_histograms'] = {key: dict(collections.Counter(x['call_record'][key] for x in calls))
                                          for key in ['candidate_source', 'return_reason', 'stop_reason']}
        o['timeline_requested'] = timeline
        o['timeline_available_event_samples'] = event_timeline
        o['last_event_call'] = compact_call(last)
        o['r26_configuration'] = resolved['r12_configuration']['r26_reference']
        o['allowed_ground_body_names'] = resolved['model']['allowed_ground_body_names']
        o['runtime_scope'] = {k: resolved[k] for k in ['physics_dt_s', 'logical_feedback_dt_s', 'logical_planning_dt_s', 'workers', 'synchronous_slow_simulation', 'mujoco_version']}
        o['stream_receipt'] = stream
        out['originals'][suffix] = o
    phases = read(w49 / 'phase.json')
    start_a = streams['a']['stages'][0]['started_unix_s']
    start_b = streams['b']['stages'][0]['started_unix_s']
    end_a = streams['a']['stages'][0]['ended_unix_s']
    end_b = streams['b']['stages'][0]['ended_unix_s']
    out['resources'] = {
        'declared': phases['resources'],
        'original_overlap_s': min(end_a, end_b) - max(start_a, start_b),
        'original_overlap_minutes': (min(end_a, end_b) - max(start_a, start_b)) / 60,
        'finite_timing': {},
        'interpretation': 'Two CPU original tasks truly overlapped; GPU C ended before original A/B began. Affinity is not utilization telemetry.',
    }
    for name in ['task-a-attempt01', 'task-a-attempt02-gdb', 'task-a', 'task-a-model', 'task-b', 'task-c', 'deployment-attempt01', 'deployment']:
        row = read(w49 / (name + '.json'))
        out['resources']['finite_timing'][name] = {k: row[k] for k in ['started_unix_s', 'ended_unix_s', 'wall_s', 'affinity', 'status'] if k in row}
    out['finite_accounting'] = phases['finite_counters'] if 'finite_counters' in phases else {k:v for k,v in phases.items() if k.startswith('finite')}
    out['query_task_a'] = read(w49 / 'task-a.json')
    out['predictor_reconstruction'] = read(w49 / 'task-a-model.json')
    out['technical_attempts'] = [read(w49 / f'{name}.json') for name in ['task-a-attempt01', 'task-a-attempt02-gdb', 'deployment-attempt01']]
    gpu = read(w49 / 'task-c.json')
    g = {k: gpu[k] for k in [
        'status', 'gpu_world_steps', 'warmup_world_steps', 'versions', 'dispatch',
        'initialization_scope', 'old_held_comparison', 'initial_input_comparison',
        'host_device_ctrl_equal', 'snapshot_coverage', 'persistent_input_audit',
        'equivalent_native_ranking', 'full_native_cost_implemented_on_gpu',
        'original_task_qualified', 'options', 'artifact', 'unavailable', 'timing_scope',
    ]}
    g['conditions'] = []
    for c in gpu['conditions']:
        g['conditions'].append({
            'condition': c['condition'], 'first_difference': c['first_difference'],
            'all_initial_declared_inputs_equal_S0': all(v['bitwise_equal'] for v in c['initial_vs_S0'].values()),
            'all_10_steps_sorted_recorded_contacts_equal': all(s['contacts']['0'] == s['contacts']['1'] for s in c['substeps']),
            'step1_world_comparison': c['substeps'][0]['world_comparison'],
            'step10_world_comparison': c['substeps'][-1]['world_comparison'],
            'max_physical_differences_over_10': {k: max(s['world_comparison'][k].get('max_abs', 0) for s in c['substeps'])
                                                for k in ['qpos', 'qvel', 'qacc', 'qfrc_constraint']},
        })
    g['reset_same_world_comparisons'] = {k: [s for s in rows if s['substep'] in [0, 1, 10]]
                                         for k, rows in gpu['same_world_reset_comparisons'].items()}
    out['gpu'] = g
    for d in sorted((w49 / 'delivery').iterdir()):
        if not (d / 'sources.json').exists():
            continue
        sources = read(d / 'sources.json')
        out['deliveries'][d.name] = {
            'receipt': read(d / 'delivery.json'),
            'archives': read(d / 'manifest.json'),
            'source_file_count': len(sources),
            'uncompressed_source_bytes': sum(s['size'] for s in sources),
            'external_originals_read': False,
        }
    out['video_catalog'] = read(w49 / 'videos/catalog.json')
    out['video_remote_receipt'] = read(w49 / 'videos/remote-verification.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'originals': len(out['originals']),
                      'event_calls': sum(x['embedded_event_calls'] for x in out['originals'].values()),
                      'overlap_minutes': out['resources']['original_overlap_minutes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
