#!/usr/bin/env python3
"""Freeze W47's one donor and prescribed fixtures from server-local originals.

This is an inventory calculation, with no simulator, optimizer, GPU or network
entry point. Missing original data never triggers a replacement-source search.
"""
import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REL = Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
SMALL = ROOT / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47'
NAME = 'project-s3-formal-v1-single-yaw_plus5-trajectory.npz'
EXPECTED_SHA = '079f57e1ebad260c9cdac842dc089012e29cb46992ab64d2c31cd9d380b61625'
EXPECTED_SIZE = 28100505


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_identity(path):
    path = Path(path)
    try:
        label = str(path.relative_to(ROOT))
    except ValueError:
        label = str(path)
    if not path.is_file():
        return {'path': label, 'status': 'absent'}
    return {'path': label, 'status': 'present', 'size': path.stat().st_size,
            'sha256': digest(path)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_manifest():
    index_path = ROOT / 'results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/index.json'
    resolved_path = index_path.parent / 'project-s3-formal-v1-single-yaw_plus5-resolved.json'
    idx = json.loads(index_path.read_text())
    item = next(a for a in idx['assets'] if a['name'] == NAME)
    remote = next(a for a in idx['publication']['asset_verification'] if a['name'] == NAME)
    assert (item['size'], item['sha256']) == (EXPECTED_SIZE, EXPECTED_SHA)
    command = ['find', '/root/autodl-tmp', '-type', 'f', '(', '-name', NAME,
               '-o', '-path', '*/project-s3-formal-v1-single-yaw_plus5/trajectory.npz', ')', '-print']
    found = subprocess.check_output(command, text=True).splitlines()
    original = Path(item['path'])
    if original.is_file() and str(original) not in found:
        found.append(str(original))
    resolved = json.loads(resolved_path.read_text())
    candidates = [file_identity(Path(p)) for p in found]
    verified = [p for p in candidates if p['size'] == EXPECTED_SIZE and p['sha256'] == EXPECTED_SHA]
    return {
        'schema': 's1_w47_fixed_donor_source_v1',
        'declared_source': NAME,
        'expected_size': EXPECTED_SIZE, 'expected_sha256': EXPECTED_SHA,
        'historical_manifest': file_identity(index_path),
        'historical_original_path': str(original), 'historical_original_path_exists': original.is_file(),
        'historical_release_metadata': remote,
        'resolved': file_identity(resolved_path),
        'source_project_commit': resolved['source_commit'],
        'source_author_commit': resolved['author_commit'],
        'search': {'command': command, 'matched_paths': found,
                   'roots_are_execution_server_only': True},
        'candidate_files': candidates, 'verified_files': verified,
        'status': 'available' if verified else 'donor_unavailable',
        'reason': None if verified else 'prescribed_R8_trajectory_original_absent_on_execution_server',
        'payload_schema_and_time_phase_verification': 'not_performed_no_original' if not verified else 'pending',
        'saved_cadence_settings': [
            {'time_s': s['time_s'], 'gait': s['task_parameters']['Gait'],
             'cadence_hz': s['task_parameters']['Cadence'], 'amplitude_m': s['task_parameters']['Amplitude']}
            for s in resolved['applied_settings']],
        'mapping_rule': {
            'elapsed_origin': 'original episode physical start, not mission-phase start',
            'source_node': 'nearest original 2ms node at same episode elapsed; ties earlier',
            'required_span_s': .350,
            'mission': 'same moving mission phase; trajectory.phase is mission enum, never gait phase',
            'gait': 'source resolved cadence/GetPhase reset reconstruction; phase difference bounded by rounding and FP64 arithmetic only',
            'no_time_shift_stretch_phase_or_seed_search': True,
            'robot_transform': 'xy translation and yaw quaternion composition only; velocity follows free-joint coordinates',
            'world_material_goal_observer': 'current context unchanged',
            'source_x_role': 'initial mismatch and held-transition defect diagnosis, not an independently optimized state',
            'input': 'actual R8 q_target; K=0; H from frozen current incumbent; original Clamp/H-QP',
            'source_payload_absent_means_mapping_not_claimed': True,
        },
        'downloaded_bytes': 0, 'substitute_sources': 0,
    }


def latest_context(suffix, seed, event_time, fallback):
    directory = ROOT / REL / f'job-20261112T133729Z-w46-{suffix}-eval-PANN_PPO_S5-s{seed}-E30-attempt01'
    call_path = directory / 'planner-calls.jsonl'
    rows = [json.loads(line) for line in call_path.open()]
    row = [r for r in rows if r['origin'] <= event_time + 1e-10][-1]
    origin = row['origin']
    names = [f'accepted-reference-gain-H-plan{row["incumbent_scored_id"]:06d}.npz',
             f'accepted-reference-gain-H-plan{row["installed_plan_id"]:06d}.npz',
             f'r33-candidates-call{row["call_id"]:06d}.npz']
    names = list(dict.fromkeys(names))
    files = [file_identity(directory / n) for n in names]
    archive_directory = ROOT / REL / f'w46-delivery/w46-{suffix}'
    archives = []
    archive_matches = []
    for path in sorted(archive_directory.glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            entries = archive.namelist()
            matches = [n for n in entries if Path(n).name in names]
            archives.append({'path': str(path.relative_to(ROOT)), 'entry_count': len(entries)})
            archive_matches += [{'archive': str(path.relative_to(ROOT)), 'member': n} for n in matches]
    with np.load(directory / 'trajectory.npz') as z:
        delta_tau = np.abs(np.diff(z['torque'][:, [2, 5, 8, 11]], axis=0))
        event_index = int(np.nonzero(np.any(delta_tau >= 90.86 - 1e-8, axis=1))[0][0]) + 1
        actual_event = float(z['time'][event_index])
        state_index = int(np.argmin(abs(z['time'] - origin)))
        event = {'first_knee_limit_reversal_time_s': actual_event,
                 'prior_torque_Nm': z['torque'][event_index - 1, [2, 5, 8, 11]].tolist(),
                 'event_torque_Nm': z['torque'][event_index, [2, 5, 8, 11]].tolist(),
                 'state_time_s': float(z['time'][state_index]),
                 'state_origin_abs_error_s': float(abs(z['time'][state_index] - origin)),
                 'mission_phase_enum': int(z['phase'][state_index])}
    predictions = []
    prior_ranges = []
    for path in sorted(directory.glob('accepted-reference-gain-H-plan*.npz')):
        with np.load(path) as z:
            plan_origin = float(z['header'][2])
            if plan_origin >= origin - 1e-8:
                continue
            times = z['accepted_times']
            node = int(np.argmin(abs(times - origin)))
            gap = float(abs(times[node] - origin))
            prior_ranges.append({'path': str(path.relative_to(ROOT)), 'origin_s': plan_origin,
                                 'last_prediction_time_s': float(times[-1])})
            if gap < 1e-8:
                predictions.append({'file': file_identity(path), 'plan_origin_s': plan_origin,
                                    'node': node, 'node_time_s': float(times[node]), 'abs_gap_s': gap})
    reference = directory / f'r29-current-plan.bin.r35-reference-{row["r35_reference_epoch"]}.bin'
    missing = [x['path'] for x in files if x['status'] == 'absent']
    return {
        'requested_case': 'W46-' + suffix.upper(), 'source_round': 'W46',
        'declared_event_s': event_time, 'actual_event': event,
        'selection': 'last real planner call before or at first recorded knee limit reversal; no unbounded backward context search',
        'source_call': row['call_id'], 'source_time_s': origin,
        'event_minus_context_s': actual_event - origin,
        'source_plan': row['installed_plan_id'], 'input_incumbent_plan': row['incumbent_scored_id'],
        'deployed_relation': {k: row[k] for k in ['return_reason', 'policy_updated', 'policy_generation_call',
                                                'policy_generation_time', 'reused_policy_age_s', 'candidate_source']},
        'context_gait_values': row['gait_context'],
        'scalar_calls_file': file_identity(call_path),
        'trajectory_file': file_identity(directory / 'trajectory.npz'),
        'frozen_reference_file': file_identity(reference),
        'required_policy_files': files,
        'delivery_archive_inventory': archives, 'required_members_found_in_archives': archive_matches,
        'missing_policy_files': missing,
        'context_status': 'unavailable' if missing else 'available',
        'matching_previous_predictions': predictions,
        'closest_saved_prior_predictions': sorted(prior_ranges, key=lambda x: x['origin_s'])[-3:],
        'perturbation_status': 'available' if predictions else 'perturbation_unavailable',
        'fallback': fallback if missing else None,
        'latest_W46_coverage': not missing and bool(predictions),
    }


def historical_fixture(label, requested_case=None):
    directory = ROOT / REL / 'w46-mechanisms' / label
    data = json.loads((directory / 'input.json').read_text())
    files = {n: file_identity(directory / n) for n in [
        'input.json', 'input-state.npz', 'input-policy.bin', 'baseline-policy.bin',
        'context.bin', 'model.xml', 'perturbation.npz']}
    with np.load(directory / 'perturbation.npz') as z:
        delta = z['delta']
    prediction = ROOT / data['perturbation_source']['path']
    return {
        'label': label, 'requested_case': requested_case,
        'source_round': 'W45', 'source_case': 'A' if label[0] == 'A' else 'B',
        'source_call': data['source_call'], 'source_plan': data['source_plan'],
        'source_time_s': data['origin_s'], 'source_directory': data['source'],
        'common_context': 'saved deployed output policy and reconstructed context, not exact historical pre-call replay',
        'files': files, 'matching_prediction_source': dict(data['perturbation_source'], file=file_identity(prediction)),
        'tangent_delta_abs_max_position': float(np.max(abs(delta[:18]))),
        'tangent_delta_abs_max_velocity': float(np.max(abs(delta[18:]))),
        'fallback': requested_case is not None,
        'latest_W46_coverage': False,
    }


def freeze():
    source = source_manifest()
    latest = [latest_context('a', 1, 6.860, 'A148'), latest_context('b', 2, 6.150, 'A153')]
    fixtures = [historical_fixture('B155'), historical_fixture('B158')]
    for row in latest:
        if row['fallback']:
            fixtures.append(historical_fixture(row['fallback'], row['requested_case']))
        else:
            fixtures.append({'requested_case': row['requested_case'], 'source_call': row['source_call'],
                             'source_time_s': row['source_time_s'], 'preparation': 'requires_available_originals'})
    manifest = {'schema': 's1_w47_b_fixture_manifest_v1', 'created_unix_s': time.time(),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'plan': file_identity(ROOT / 'docs/S1_W47_STABILITY_PLAN.md'),
                'latest_context_inventory': latest, 'frozen_fixtures': fixtures,
                'latest_W46_A_B_both_covered': all(r['latest_W46_coverage'] for r in latest),
                'work': {'native_solves': 0, 'native_rollouts': 0, 'held_transitions': 0,
                         'native_physics_substeps': 0, 'plant_episodes': 0, 'PPO_updates': 0}}
    write(SMALL / 'task-b-source-manifest.json', source)
    write(SMALL / 'task-b-fixture-manifest.json', manifest)
    print(json.dumps({'source_status': source['status'], 'latest_W46_A_B_both_covered': manifest['latest_W46_A_B_both_covered'],
                      'fixtures': [f.get('label', f['requested_case']) for f in fixtures]}))


def complete():
    source_path = SMALL / 'task-b-source-manifest.json'
    fixtures_path = SMALL / 'task-b-fixture-manifest.json'
    source = json.loads(source_path.read_text())
    fixtures = json.loads(fixtures_path.read_text())
    if source['status'] != 'donor_unavailable':
        raise RuntimeError('Available donor requires actual semantic implementation and bounded solves; inventory cannot declare a mechanism outcome')
    result = {'schema': 's1_w47_b_result_v1', 'status': 'completed_bounded_availability_negative',
              'source_manifest': file_identity(source_path), 'fixture_manifest': file_identity(fixtures_path),
              'donor_status': source['status'], 'donor_reason': source['reason'],
              'mechanism_verdict': 'unidentified: no original donor input; no candidate-missing, rollout-defect or objective-conflict claim',
              'latest_W46_A_B_both_covered': fixtures['latest_W46_A_B_both_covered'],
              'missing_latest_contexts': [r['requested_case'] for r in fixtures['latest_context_inventory'] if not r['latest_W46_coverage']],
              'missing_matching_perturbations': [r['requested_case'] for r in fixtures['latest_context_inventory'] if r['perturbation_status'] != 'available'],
              'implementation': 'server-local source/context inventory; no unsupported donor or online implementation',
              'formal_solves': 'not_started_source_unavailable', 'original_E30_admission_from_B': False,
              'work': fixtures['work'], 'deadline_unix_s': None, 'wall_deadline_enabled': False,
              'finished_unix_s': time.time()}
    write(SMALL / 'task-b.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'complete'])
    args = parser.parse_args()
    freeze() if args.action == 'freeze' else complete()
