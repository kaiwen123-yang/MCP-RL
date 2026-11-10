#!/usr/bin/env python3
"""Offline W41 controller audit from delivered native logs; no simulation.

The norm formulas are the pinned mjpc/norm.cc cases 0, 2, and 6 at
johnzhang3/mujoco_mpc e00c47a5adb9856af2e0f24231bb3a60d5be23c4.
This script does not claim that a relaxed predictor equals the physical plant.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[6]
RESULTS = ROOT / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
JOBS = {'w41-a': 'job-20261107T100149Z-w41-a',
        'w41-b': 'job-20261107T093906Z-w41-b'}


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def prefix_rows(flat):
    """R29RecordedStep: header/pre/post/PD/force/passive/residual/contact."""
    p = 0
    while p < len(flat):
        ncon = int(flat[p + 4])
        a = flat[p:p + 222 + 8 * ncon]
        assert len(a) == 222 + 8 * ncon
        yield {'node': int(a[0]), 'substep': int(a[1]),
               'before': float(a[2]), 'after': float(a[3]),
               'pre': a[5:42], 'post': a[42:79],
               'raw_pd': a[79:91], 'force': a[91:103],
               'passive': a[103:121], 'residual': a[121:222],
               'contacts': a[222:].reshape(ncon, 8)}
        p += len(a)
    assert p == len(flat)


def cost_terms(xml, weights, residual):
    output = {}
    stages = {}
    start = 0
    for sensor in xml.find('sensor'):
        if sensor.tag != 'user':
            continue
        name = sensor.attrib['name']
        size = int(sensor.attrib['dim'])
        user = np.fromstring(sensor.attrib['user'], sep=' ')
        r = residual[:, start:start + size]
        norm = int(user[0])
        weight = weights[name]
        if norm == 0:
            loss = 0.5 * np.sum(r * r, axis=1)
        elif norm == 2:
            loss = np.sqrt(np.sum(r * r, axis=1) + user[4] ** 2) - user[4]
        elif norm == 6:
            loss = np.sum(np.sqrt(r * r + user[4] ** 2) - user[4], axis=1)
        else:
            raise ValueError(f'unhandled norm {norm}')
        stage = weight * loss
        stages[name] = stage
        output[name] = {'weight': weight, 'norm': norm,
                        'mean_cost': float(stage.mean()),
                        'max_stage_cost': float(stage.max()),
                        'max_abs_residual': float(np.abs(r).max()),
                        'max_stage_node': int(stage.argmax())}
        start += size
    assert start == residual.shape[1] == 101
    total = sum(x['mean_cost'] for x in output.values())
    for x in output.values():
        x['fraction_of_total'] = x['mean_cost'] / total
    return total, output


def check_gains(plan, ctrl, force):
    free_scaled = 0.0
    bound_error = 0.0
    pd_rows = 0
    active_rows = 0
    min_eig = float('inf')
    max_cond = 0.0
    for i in range(len(plan['accepted_actions'])):
        H, K = plan['H'][i], plan['K'][i]
        qxu = plan['base_Qxu'][i].T
        active = plan['base_active'][i].astype(int)
        free = active == 0
        residual = H @ K + qxu
        scale = max(1.0, float(np.max(np.abs(H @ K))), float(np.max(np.abs(qxu))))
        if np.any(free):
            free_scaled = max(free_scaled, float(np.max(np.abs(residual[free]))) / scale)
        x = plan['reference_states'][i]
        pdlo = x[7:19] + (force[:, 0] + 5 * x[25:37]) / 60
        pdhi = x[7:19] + (force[:, 1] + 5 * x[25:37]) / 60
        lo = np.maximum(ctrl[:, 0], pdlo)
        hi = np.minimum(ctrl[:, 1], pdhi)
        for j in np.flatnonzero(active):
            active_rows += 1
            moving = ((active[j] < 0 and pdlo[j] > ctrl[j, 0]) or
                      (active[j] > 0 and pdhi[j] < ctrl[j, 1])) and lo[j] < hi[j]
            expected = np.zeros(K.shape[1])
            if moving:
                pd_rows += 1
                expected[6 + j] = 1.0
                expected[24 + j] = 5.0 / 60.0
            bound_error = max(bound_error, float(np.max(np.abs(K[j] - expected))))
        eig = np.linalg.eigvalsh(H)
        min_eig = min(min_eig, float(eig.min()))
        max_cond = max(max_cond, float(eig.max() / eig.min()))
    return {'free_stationarity_scaled_max': free_scaled,
            'active_row_expected_error_max': bound_error,
            'active_rows': active_rows, 'moving_pd_rows': pd_rows,
            'min_eigenvalue_H': min_eig, 'max_condition_H': max_cond,
            'max_abs_K': float(np.abs(plan['K'][:-1]).max()),
            'regularization_type': int(plan['header'][9])}


def closest(times, t):
    i = int(np.argmin(np.abs(times - t)))
    assert abs(times[i] - t) < 2e-8, (times[i], t)
    return i


def check_first_hold(plan, trajectory, physical, floor, feet):
    rows = list(prefix_rows(plan['accepted_prefix_substeps']))
    hold = [r for r in rows if r['node'] == 0]
    assert len(hold) == 10
    t0 = hold[0]['before']
    idx = closest(trajectory['time'], t0)
    idx1 = closest(trajectory['time'], hold[-1]['after'])
    x0 = np.r_[trajectory['qpos'][idx], trajectory['qvel'][idx]]
    x1 = np.r_[trajectory['qpos'][idx1], trajectory['qvel'][idx1]]
    act = trajectory['q_target'][idx].astype(float)
    first = hold[0]
    force_series = []
    for row in hold:
        pidx = closest(physical[:, 0], row['before'])
        contacts = row['contacts']
        floor_foot = [c for c in contacts
                      if int(c[0]) == floor and int(c[1]) in feet]
        # Floor is a horizontal plane, foot geometries are spheres, and geom0
        # is floor: contact-frame normal is +world z in these selected rows.
        pred_fz = sum(float(c[2]) for c in floor_foot)
        force_series.append({'substep': row['substep'], 'time': row['before'],
                             'predicted_native_floor_foot_fz_N': pred_fz,
                             'actual_total_ground_fz_N': float(physical[pidx, 44]),
                             'actual_nonfoot_fz_N': float(physical[pidx, 47]),
                             'actual_nonfoot_contact_count': int(physical[pidx, 48]),
                             'actual_contacts': physical[pidx, 26:30].astype(int).tolist(),
                             'predicted_pairs': contacts[:, :2].astype(int).tolist(),
                             'max_actuator_force_difference_Nm':
                             float(np.max(np.abs(row['force'] - physical[pidx, 14:26]))),
                             'max_raw_pd_difference_Nm':
                             float(np.max(np.abs(row['raw_pd'] - physical[pidx, 49:61])))})
    err = x1 - hold[-1]['post']
    return {'origin': t0, 'same_initial_state_max_error': float(np.abs(x0 - first['pre']).max()),
            'candidate_initial_state_max_error': float(np.abs(plan['accepted_states'][0] - first['pre']).max()),
            'actual_vs_candidate_first_target_max_rad': float(np.abs(act - plan['accepted_actions'][0]).max()),
            'first_hold_end_s': hold[-1]['after'],
            'first_hold_base_position_error_max_m': float(np.abs(err[:3]).max()),
            'first_hold_joint_position_error_max_rad': float(np.abs(err[7:19]).max()),
            'first_hold_base_velocity_error_max_mixed': float(np.abs(err[19:25]).max()),
            'first_hold_joint_velocity_error_max_rad_s': float(np.abs(err[25:37]).max()),
            'first_hold_force_series': force_series,
            'limitation': 'Targets almost equal, not replayed identically. Actual total ground force includes nonfoot force; it equals foot force only where nonfoot force/count are zero. Material can contribute after loading; strict rigid-contact inference uses pre-material rows only. No same-input claim after this first hold.'}


def executed_feedback(path):
    with (path / 'action-execution.jsonl').open() as f:
        ids = {r['query_id'] for line in f if (r := json.loads(line)).get('applied_action_id') is not None}
    with (path / 'feedback-actions.jsonl').open() as f:
        return [r for line in f if (r := json.loads(line))['query_id'] in ids]


def stall_stats(trajectory, feedback):
    mask = (trajectory['time'] >= 9 - 1e-8) & (trajectory['time'] < 11 - 1e-8)
    ix = np.flatnonzero(mask)
    a, b = int(ix[0]), int(ix[-1])
    actions = [r for r in feedback if 9 - 1e-6 <= r['time'] < 11 - 1e-6]
    return {'start_s': float(trajectory['time'][a]), 'last_sample_s': float(trajectory['time'][b]),
            'net_progress_speed_m_s': float((trajectory['qpos'][b, 0] - trajectory['qpos'][a, 0]) /
                                           (trajectory['time'][b] - trajectory['time'][a])),
            'mean_command_x_m_s': float(trajectory['command_velocity'][mask, 0].mean()),
            'all_four_contact_fraction': float(trajectory['contact'][mask].all(axis=1).mean()),
            'max_foot_x_m': float(trajectory['feet'][mask, :, 0].max()),
            'max_abs_yaw_rad': float(np.abs(trajectory['rpy'][mask, 2]).max()),
            'heading_targets': np.unique(trajectory['heading_target'][mask]).tolist(),
            'executed_feedback_count': len(actions),
            'projection_changed_count': sum(r['projection_changed'] for r in actions),
            'active_bound_count': int(sum(np.count_nonzero(r['active_bounds']) for r in actions)),
            'qp_not_converged_count': sum(not r['qp_converged'] for r in actions)}


def first_large_feedback(feedback, plans):
    for r in feedback:
        if r['time'] <= 2 or max(map(abs, r['raw_K_delta_x'])) <= 1:
            continue
        output = {'time': r['time'], 'plan_id': r['plan_id'],
                  'raw_K_delta_x_max_rad': max(map(abs, r['raw_K_delta_x'])),
                  'projection_changed': r['projection_changed'], 'qp_converged': r['qp_converged']}
        p = plans.get(int(r['plan_id']))
        if p is not None:
            i, j = r['interpolation_nodes']
            fraction = r['interpolation_fraction']
            K = p['K'][i] * (1 - fraction) + p['K'][j] * fraction
            dx = np.array(r['state_delta'])
            calculated = K @ dx
            output['reconstruction_max_error'] = float(np.abs(calculated - r['raw_K_delta_x']).max())
            output['components'] = {}
            for name, sl in [('base_translation', slice(0, 3)), ('base_rotation', slice(3, 6)),
                             ('joint_position', slice(6, 18)), ('base_velocity', slice(18, 24)),
                             ('joint_velocity', slice(24, 36))]:
                output['components'][name] = {'max_abs_state_delta': float(np.abs(dx[sl]).max()),
                                             'max_abs_action_contribution_rad': float(np.abs(K[:, sl] @ dx[sl]).max())}
        return output
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = ap.parse_args()
    out = {'schema': 'w41_offline_controller_audit_v1',
           'base_commit': '9cc42d5a9189a2bfd7e1e6cd99b9300045a16724',
           'norm_source': 'https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/norm.cc',
           'source_inputs': {}, 'streams': {}}
    for stream, job in JOBS.items():
        base = args.raw_root / stream / 'evaluation-E30'
        seed = 1 if stream == 'w41-a' else 2
        resolved_path = RESULTS / f'{job}-eval-PANN_PPO_S5-s{seed}-E30-resolved.json'
        resolved = read_json(resolved_path)
        xml_path = base / 'model/G1-project-go2.xml'
        xml = ET.parse(xml_path).getroot()
        geometries = list(xml.find('worldbody').iter('geom'))
        floor = next(i for i, g in enumerate(geometries) if g.get('name') == 'floor')
        feet = {i for i, g in enumerate(geometries) if g.get('name') in {'FL', 'FR', 'RL', 'RR'}}
        assert floor == 0 and feet == {26, 39, 52, 65}
        ctrl = np.array(resolved['actuator_ctrlrange'])
        force = np.array(resolved['actuator_forcerange'])
        trajectory = dict(np.load(base / 'trajectory.npz'))
        physical = np.load(base / 'physical-000.npz')['physical']
        feedback = executed_feedback(base)
        plans = {}
        entries = []
        paths = sorted(base.glob('accepted-reference-gain-H-plan*.npz'))
        for path in paths:
            p = dict(np.load(path))
            plans[int(p['header'][1])] = p
            J, terms = cost_terms(xml, resolved['applied_settings'][-1]['cost_weights'], p['accepted_residual'])
            saved_J = float(p['header'][-1])
            assert abs(J - saved_J) < 1e-10, (path, J, saved_J)
            gains = check_gains(p, ctrl, force)
            assert gains['regularization_type'] == 0
            assert gains['free_stationarity_scaled_max'] < 1e-10
            assert gains['active_row_expected_error_max'] < 1e-10
            entries.append({'file': path.name, 'plan_id': int(p['header'][1]),
                            'origin_s': float(p['header'][2]), 'saved_cost': saved_J,
                            'recomputed_cost': J, 'cost_error': abs(J - saved_J),
                            'accepted_predicted_base_dx_m': float(p['accepted_states'][-1, 0] - p['accepted_states'][0, 0]),
                            'cost_terms': terms, 'gain_check': gains,
                            'first_hold': check_first_hold(p, trajectory, physical, floor, feet)})
        for p in paths + [xml_path, base / 'trajectory.npz', base / 'physical-000.npz', resolved_path]:
            out['source_inputs'][str(p.relative_to(args.raw_root) if p.is_relative_to(args.raw_root) else p.relative_to(ROOT))] = digest(p)
        out['streams'][stream] = {'applied_locomotion_settings': resolved['applied_settings'][-1],
                                  'stall_9_to_11': stall_stats(trajectory, feedback),
                                  'first_feedback_over_1rad': first_large_feedback(feedback, plans),
                                  'plans': entries}
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(args.output),
                      'plans': sum(len(s['plans']) for s in out['streams'].values()),
                      'max_cost_error': max(p['cost_error'] for s in out['streams'].values() for p in s['plans'])}, indent=2))


if __name__ == '__main__':
    main()
