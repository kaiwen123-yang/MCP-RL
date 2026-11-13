#!/usr/bin/env python3
"""Audit committed W47 control evidence; no network, simulation, or optimization."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
from collections import Counter


def maxabs(values):
    return max((abs(float(x)) for x in values), default=0.0)


def apply_patch_text(original, patch):
    old = original.splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    output = []
    pos = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith('@@'):
            i += 1
            continue
        match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        assert match
        start = int(match[1]) - 1
        output.extend(old[pos:start])
        pos = start
        i += 1
        while i < len(lines) and not lines[i].startswith(('@@', '--- ')):
            line = lines[i]
            if line[:1] in (' ', '-'):
                assert old[pos] == line[1:], (pos, old[pos], line)
                if line.startswith(' '):
                    output.append(old[pos])
                pos += 1
            elif line.startswith('+'):
                output.append(line[1:])
            else:
                raise ValueError(line)
            i += 1
    output.extend(old[pos:])
    return ''.join(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, type=pathlib.Path)
    ap.add_argument('--r35-source', type=pathlib.Path)
    ap.add_argument('--output', required=True, type=pathlib.Path)
    args = ap.parse_args()
    repo = args.repo
    directory = repo / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47'
    provenance = []
    def read_json(relative):
        p = directory / relative
        raw = p.read_bytes()
        provenance.append(dict(path=str(p.relative_to(repo)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)
    saved = read_json('task-a/saved-array-summary.json')
    paired = read_json('task-a/paired-evidence.json')
    build = read_json('task-a/diagnostic-build.json')
    phase = read_json('phase.json')
    results = []
    all_candidates = []
    zero_first = []
    record_error = Counter()
    all_node_recon_errors = []
    for label in ['A148', 'A153', 'B155', 'B158']:
        fixture = read_json('task-a/' + label + '.json')
        row = {'label': label, 'source_round': fixture['source_round'], 'historical_replay': fixture['historical_replay'], 'branches': {}}
        assert fixture['source_round'] == 'W45' and fixture['historical_replay'] is False
        for branch, b in fixture['branches'].items():
            nominal = b['nominal']
            candidates = b['candidates']
            pos = [c for c in candidates if c['alpha'] > 0]
            zero = [c for c in candidates if c['alpha'] == 0]
            assert len(pos) == 45 and len(zero) == 5
            attempts = {a['attempt']: a for a in b['search_attempts']}
            for c in candidates:
                attempt = attempts[int(c['slot'] / 10) + 1]
                alpha = c['alpha']
                dv = attempt['dV_unscaled']
                errors = dict(
                    J_difference=abs(c['actual_delta_J'] - (c['J'] - nominal['J'])),
                    cost_parts=abs(sum(c['parts_delta'].values()) - c['actual_delta_J']),
                    cost_sum=abs(c['J0'] + c['Jref'] - c['J']),
                    prediction=abs(c['predicted_decrease'] - (-alpha * (dv[0] + alpha * dv[1]))),
                )
                for key, value in errors.items():
                    record_error[key] = max(record_error[key], value)
                assert c['reference_state_error'] == 0 and c['reference_time_error'] == 0
            row['branches'][branch] = dict(
                nominal_J=nominal['J'], nominal_scale=nominal['scale'],
                positive_count=len(pos), zero_count=len(zero),
                predicted_descent_count=sum(c['predicted_decrease'] > 0 for c in pos),
                actual_ascent_count=sum(c['actual_delta_J'] > 0 for c in pos),
                failed_count=sum(c['failed'] for c in candidates),
                nonzero_violation_count=sum(c['violation'] != 0 for c in candidates),
                min_positive_actual_delta_J=min(c['actual_delta_J'] for c in pos),
                min_positive_predicted_decrease=min(c['predicted_decrease'] for c in pos),
                max_positive_predicted_decrease=max(c['predicted_decrease'] for c in pos),
                positive_actual_over_predicted_range=[
                    min(c['actual_delta_J'] / c['predicted_decrease'] for c in pos),
                    max(c['actual_delta_J'] / c['predicted_decrease'] for c in pos),
                ],
                zero_delta_J_range=[min(c['actual_delta_J'] for c in zero), max(c['actual_delta_J'] for c in zero)],
                all_backwards_complete=all(a['backward_complete'] for a in attempts.values()),
                statuses=dict(Counter(a['status'] for a in attempts.values())),
            )
            all_candidates.extend(pos)
            if branch == 'constant':
                assert all(v == 0 for v in b['same_complete_policy_repeat_envelope'].values())
                for detail in b['first_divergence_details']:
                    for n in detail['nodes']:
                        err = maxabs(r - (u + f) for r, u, f in zip(n['raw'], n['u_ff'], n['raw_feedback']))
                        assert err == 0
                        if n['projected'] is not None:
                            err = maxabs(u - v for u, v in zip(n['projected'], n['saved_native_held']))
                            assert err == n['projection_reconstruction_abs_max']
                            all_node_recon_errors.append(err)
                    if detail['role'] == 'alpha0':
                        assert detail['alpha0_center_action_error'] == 0
                        assert detail['exact_node_time_match']
                        n = next(n for n in detail['nodes'] if n['node'] == detail['first_nonzero_action_node'])
                        assert n['actual_x_equals_reference']
                        assert n['projection_delta_abs_max'] == 0
                        assert n['projection_reconstruction_abs_max'] == 0
                        assert maxabs(n['quaternion_self_tangent']) > 0
                        zero_first.append(dict(label=label, record=detail['record'],
                            node=n['node'], decision_time_s=n['decision_time_s'],
                            first_state_node=detail['first_nonzero_state_node'],
                            first_command_over_1e_6_node=detail['first_action_over_1e_6_rad_node'],
                            first_command_delta_rad=maxabs(a-b for a,b in zip(n['saved_native_held'], n['nominal_native_held'])),
                            quaternion_self_tangent=n['quaternion_self_tangent'],
                            quaternion_norm=n['quaternion_norm'], K_abs_max_at_first_node=n['K_abs_max'],
                            full_horizon_max_command_delta_rad=detail['max_action_delta_rad'],
                            delta_J=detail['actual_delta_J']))
        rr = {r['mode']: r for r in paired['rows'] if r['label'] == label}
        assert set(rr) == {'old', 'corrected'}
        old, corrected = rr['old'], rr['corrected']
        shared = ['input_policy_sha256', 'context_sha256', 'reference_sha256', 'model_sha256', 'state_sha256', 'source_policy_identity']
        assert all(old[k] == corrected[k] for k in shared)
        assert all(v == 0 for v in old['error_vs_saved_alpha0'].values())
        assert all(v == 0 for v in corrected['error_vs_saved_nominal'].values())
        assert corrected['delta_J_vs_saved_nominal'] == 0
        assert old['J'] - corrected['J'] == old['delta_J_vs_saved_nominal']
        for r in rr.values():
            assert r['binary_sha256'] == r['actual_binary_sha256'] == build['variants'][r['mode']]['binary_sha256']
            assert r['path'] == 'ActionIndex/RolloutDiscrete'
            assert r['source_candidate']['alpha'] == 0
            assert r['physics_substeps'] == 1750 and r['held_intervals'] == 175 and r['recorded_state_nodes'] == 176
            assert r['new_optimizer_solves'] == 0 and r['new_PPO_updates'] == 0
            assert not r['rollout_failure'] and r['trunk_violation'] == 0
        row['paired_zero_endpoint'] = dict(
            old_delta_J=old['delta_J_vs_saved_nominal'], corrected_delta_J=corrected['delta_J_vs_saved_nominal'],
            old_max_action_error_vs_nominal_rad=old['error_vs_saved_nominal']['actions'],
            corrected_max_action_error_vs_nominal_rad=corrected['error_vs_saved_nominal']['actions'],
            old_dx_m=old['trajectory_dx_m'], corrected_dx_m=corrected['trajectory_dx_m'],
            error_vs_nominal_all_fields_zero=True, old_exactly_reproduces_archived_zero=True,
            shared_problem_identity_verified=True,
            old_J=old['J'], corrected_J=corrected['J'],
            old_J_parts=old['J_parts'], corrected_J_parts=corrected['J_parts'],
        )
        results.append(row)
    patch_checks = []
    for entry in build['patches']:
        patch_path = repo / entry['path']
        raw = patch_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        record = {'path': entry['path'], 'sha256_matches_build': True}
        if args.r35_source:
            source_relative = next(line[6:] for line in raw.decode().splitlines() if line.startswith('--- a/'))
            original = (args.r35_source / source_relative).read_text()
            corrected = apply_patch_text(original, raw.decode())
            key = 'utilities_source_sha256' if source_relative.endswith('utilities.cc') else 'planner_source_sha256'
            expected = build['variants']['corrected'][key]
            actual = hashlib.sha256(corrected.encode()).hexdigest()
            assert actual == expected, (source_relative, actual, expected)
            record.update(reconstructed_source_sha256=actual, source_matches_build=True)
        patch_checks.append(record)
    assert max(record_error.values()) < 1e-12
    assert len(all_candidates) == 540 and len(zero_first) == 20
    assert sum(r['actual_rollouts'] for r in paired['rows']) == paired['actual_rollouts'] == 8
    assert sum(r['physics_substeps'] for r in paired['rows']) == paired['physics_substeps'] == 14000
    assert phase['actual_new_work']['native_optimizer_solves'] == 0
    result = dict(
        schema='w47_committed_control_audit_v1',
        reviewed_commit='6e9075f48718509d48f51d08d0a8f62a34edfb41',
        independent_new_simulation_or_optimizer_work=0,
        scope='Independent arithmetic of committed JSON, source patch application, and manifest identities. No original binary trajectory arrays reread and no simulation executed.',
        records_verified=dict(positive_candidates=len(all_candidates), zero_first_divergences=len(zero_first), paired_fixed_rollouts=8),
        residual_arithmetic_error_max=dict(record_error),
        interior_node_reconstruction=dict(count=len(all_node_recon_errors), maximum_error=max(all_node_recon_errors)),
        results=results, zero_first_divergence_rows=zero_first,
        patch_checks=patch_checks,
        actual_work_from_records=phase['actual_new_work'],
        sources=provenance,
        limitations=[
            'First-divergence quaternion tangent and native arrays are recorded evidence, not a fresh local MuJoCo primitive recomputation.',
            'Paired identity repair is for the first backward attempt alpha0 in four W45 fixtures only.',
            'Corrected positive-step searches, ActionImpl deployment, receding-horizon closed loop, and material-memory gains remain untested.',
            'Native StateDiff exact identity fix does not suppress distinct nearby rotations or guarantee feedback robustness.',
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['records_verified', 'residual_arithmetic_error_max', 'interior_node_reconstruction', 'patch_checks']}, indent=2))


if __name__ == '__main__':
    main()
