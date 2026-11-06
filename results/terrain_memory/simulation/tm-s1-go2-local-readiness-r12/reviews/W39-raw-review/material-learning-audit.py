#!/usr/bin/env python3
"""Offline W39 original rollout/material audit. No dynamics or network inference.

Run with --raw-dir pointing at the original Release assets selected by evidence.json.
The program reads saved values, performs arithmetic, and writes its sibling JSON.
Material truth is used only to diagnose recorded exposure; it is never a policy input.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
BASE = REPO / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
HEAD = '07b0ee3e15c9f10e9b1a6a1de3997bec36f39556'
PHASES = ['initial_stand', 'outbound', 'turn', 'return', 'terminal_stand',
          'complete', 'far_hold', 'home_hold', 'turn_home']
LEGS = ['FL', 'FR', 'RL', 'RR']
HALF = np.sqrt(.001) / 2
SCHEMA = {
    'relative_goal_xy': (0, 2), 'base_z': (2, 3), 'base_quaternion_wxyz': (3, 7),
    'relative_heading_sin_cos': (7, 9), 'base_velocity6': (9, 15),
    'joint_position12': (15, 27), 'joint_velocity12': (27, 39),
    'relative_world_feet12': (39, 51), 'foot_velocity12': (51, 63),
    'contact4': (63, 67), 'GRF12': (67, 79), 'applied_cap': (79, 80),
    'mission_phase9': (80, 89), 'gait_sin_cos': (89, 91),
    'finite_task_channels': (91, 93), 'completed_plan_age': (93, 94),
    'query_xy8': (94, 102), 'query_soft_mask4': (102, 106),
    'history20x27': (106, 646), 'recovery4x3': (646, 658),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summary(x):
    x = np.asarray(x, dtype=float).ravel()
    good = x[np.isfinite(x)]
    if not len(good):
        return {'n': len(x), 'nonfinite': int((~np.isfinite(x)).sum())}
    return {'n': len(x), 'nonfinite': int((~np.isfinite(x)).sum()),
            'min': float(good.min()), 'median': float(np.median(good)),
            'mean': float(good.mean()), 'p95': float(np.quantile(good, .95)),
            'p99': float(np.quantile(good, .99)), 'max': float(good.max())}


def interior(xy):
    return ((xy[..., 0] >= .8 + HALF) & (xy[..., 0] <= 2.2 - HALF)
            & (np.abs(xy[..., 1]) <= .6 - HALF))


def footprint(xy):
    """Exact geometric world keys and overlap areas; no constitutive calculation."""
    if not interior(np.asarray(xy)):
        return {}
    lo, hi = np.asarray(xy) - HALF, np.asarray(xy) + HALF
    result = {}
    for ix in range(int(np.floor(lo[0] / .02)), int(np.floor(hi[0] / .02)) + 1):
        for iy in range(int(np.floor(lo[1] / .02)), int(np.floor(hi[1] / .02)) + 1):
            lower = np.maximum(lo, np.array([ix, iy]) * .02)
            upper = np.minimum(hi, (np.array([ix, iy]) + 1) * .02)
            area = float(np.prod(np.maximum(upper - lower, 0)))
            if area > 0:
                result[(ix, iy)] = area
    return result


class Assets:
    def __init__(self, raw):
        self.raw = raw
        self.rows = {}
        self.read = {}
        self.missing = []
        for branch in 'ab':
            path = BASE / 'jobs' / f'job-20261105T032354Z-w39-{branch}' / 'diagnostics/evidence.json'
            self.rows[branch] = read_json(path)['files']

    def select(self, branch, suffix, attempt=None):
        return [r for r in self.rows[branch] if r['source_path'].endswith(suffix)
                and (attempt is None or r.get('attempt') == attempt)]

    def data(self, branch, row):
        paths = [self.raw / p['name'] for p in sorted(row['parts'], key=lambda p: p['part'])]
        if not all(p.exists() for p in paths):
            self.missing.append({'branch': branch, 'id': row['file_id'], 'source': row['source_path']})
            return None
        value = b''.join(p.read_bytes() for p in paths)
        if row['codec'] == 'gzip':
            value = gzip.decompress(value)
        self.read[f'{branch}:{row["file_id"]}'] = {
            'source_path': row['source_path'], 'source_commit': row.get('source_commit'),
            'bytes': len(value), 'parts': [p.name for p in paths]}
        return value

    def npz(self, branch, row, keys=None):
        value = self.data(branch, row)
        if value is None:
            return None
        with np.load(io.BytesIO(value), allow_pickle=False) as npz:
            return {key: npz[key] for key in npz.files if keys is None or key in keys}

    def jsonl(self, branch, row):
        value = self.data(branch, row)
        if value is None:
            return None
        return [json.loads(x) for x in value.splitlines() if x.strip()]


def compact(branch):
    path = BASE / 'jobs' / f'job-20261105T032354Z-w39-{branch}' / 'numeric-evidence/training'
    ts, vs = [], []
    for p in sorted(path.glob('*-transitions.jsonl')):
        ts.extend(read_jsonl(p))
    for p in sorted(path.glob('*-pann-values.jsonl')):
        vs.extend(read_jsonl(p))
    return sorted(ts, key=lambda r: r['transition']), {r['transition']: r for r in vs}


def dimension_statistics(obs):
    records = []
    for i in range(obs.shape[1]):
        x = obs[:, i]
        block = next(k for k, (lo, hi) in SCHEMA.items() if lo <= i < hi)
        records.append({'dimension': i, 'block': block, **summary(x),
                        'std': float(np.std(x)), 'nonzero': int(np.count_nonzero(x))})
    return records


def rollout_audit(branch, assets):
    logs, values = compact(branch)
    logmap = {r['transition']: r for r in logs}
    batches, indexes, inventory = [], [], []
    for row in assets.rows[branch]:
        match = re.search(r'rollout-(\d+)-(\d+)\.npz$', row['source_path'])
        if match is None:
            continue
        data = assets.npz(branch, row)
        if data is None:
            continue
        start, stop = map(int, match.groups())
        batches.append(data)
        indexes.extend(range(start, stop + 1))
        inventory.append({'file_id': row['file_id'], 'first': start, 'last': stop,
                          'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in data.items()}})
    if not batches:
        return {'status': 'original_rollouts_not_available'}
    data = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    order = np.argsort(indexes)
    indexes = np.asarray(indexes)[order]
    data = {k: v[order] for k, v in data.items()}
    obs, nxt = data['obs'].reshape(-1, 658), data['next_obs'].reshape(-1, 658)
    logs = [logmap[int(i)] for i in indexes]
    selected = assets.select(branch, '/transitions-selected.jsonl')
    original_logs = assets.jsonl(branch, selected[0]) if selected else None
    raw_log_differences = Counter()
    if original_logs is not None:
        for row in original_logs:
            other = dict(logmap[row['transition']], **values[row['transition']])
            for key, value in row.items():
                if key in other and value != other[key]:
                    raw_log_differences[key] += 1
    scalar_errors = {}
    for key in ['raw_action', 'reward', 'log_prob', 'value', 'next_value', 'terminated', 'truncated']:
        truth = [values[int(i)][key] if key in ('value', 'next_value') else logmap[int(i)][key] for i in indexes]
        scalar_errors[key] = float(np.max(np.abs(data[key].ravel().astype(float) - np.asarray(truth))))
    pann_error = float(np.max(np.abs(obs[:, -12:] * np.float32(.001)
                          - np.asarray([values[int(i)]['pann_recovery_m'] for i in indexes]))))
    next_pann_error = float(np.max(np.abs(nxt[:, -12:] * np.float32(.001)
                               - np.asarray([values[int(i)]['next_pann_recovery_m'] for i in indexes]))))
    chain = [i for i in range(len(logs)-1) if logs[i]['episode'] == logs[i+1]['episode']]
    chain_error = float(np.max(np.abs(nxt[chain] - obs[np.asarray(chain)+1])))
    phase = np.argmax(obs[:, 80:89], axis=1)
    query = obs[:, 94:102].reshape(-1, 4, 2).astype(float) * 3
    mask = obs[:, 102:106].astype(bool)
    recovery = obs[:, -12:].reshape(-1, 4, 3).astype(float) * .001
    nonzero = np.any(recovery != 0, axis=2)
    base = np.zeros((len(obs), 3))
    base[:, 0] = np.isin(phase, [1, 2, 6]).astype(float) * 3
    base[:, :2] -= obs[:, :2] * 3
    base[:, 2] = obs[:, 2] * .3
    bottom = obs[:, 39:51].reshape(-1, 4, 3) * .3 + base[:, None, :]
    force = obs[:, 67:79].reshape(-1, 4, 3) * (15.206408*9.81/4)
    history = obs[:, 106:646].reshape(-1, 20, 27)
    newest = history[:, -1]
    history_errors = {
        'bottom_z': float(np.max(abs(newest[:, :4] - bottom[:, :, 2]/.05))),
        'grf_z': float(np.max(abs(newest[:, 4:8] - obs[:, 67:79].reshape(-1, 4, 3)[:, :, 2]))),
        'velocity_xy': float(np.max(abs(newest[:, 8:16] - obs[:, 51:63].reshape(-1, 4, 3)[:, :, :2].reshape(-1, 8)))),
        'contact': float(np.max(abs(newest[:, 16:20] - obs[:, 63:67]))),
        'base_velocity': float(np.max(abs(newest[:, 20:26] - obs[:, 9:15]))),
        'applied_cap': float(np.max(abs(newest[:, 26] - obs[:, 79]))),
    }
    event_rows = {}
    for row in assets.select(branch, '/material-events.jsonl'):
        if '-ep' not in row.get('attempt', ''):
            continue
        events = assets.jsonl(branch, row)
        if events is not None:
            event_rows[int(row['attempt'].split('-ep')[-1])] = events
    exposure = []
    for i, log in enumerate(logs):
        ep = log['episode']
        t = log['time_s']-log['dt_s']
        events = event_rows.get(ep)
        loaded_keys = ({tuple(e['world_key']) for e in events if e['time_s'] <= t+1e-8
                        and e['type'] in ('load', 'revisit')} if events is not None else None)
        overlap = [sum(a for key, a in footprint(p).items() if key in loaded_keys)
                   for p in query[i]] if loaded_keys is not None else None
        exposure.append({'transition': int(indexes[i]), 'episode': ep, 'action_start_s': t,
                         'query_inside_legs': int(mask[i].sum()), 'pann_nonzero_legs': int(nonzero[i].sum()),
                         'actual_material_loaded_legs_at_actor_tick': int(np.sum(interior(bottom[i, :, :2]) & (force[i, :, 2] > 0))),
                         'query_ever_loaded_truth_overlap_m2': overlap,
                         'pann_max_abs_m': float(abs(recovery[i]).max())})
    exposure_counts = Counter()
    for row in exposure:
        if row['query_inside_legs'] == 0:
            exposure_counts['all_query_feet_outside_material'] += 1
        elif row['pann_nonzero_legs'] > 0:
            exposure_counts['inside_and_nonzero_pann'] += 1
        elif row['query_ever_loaded_truth_overlap_m2'] is None:
            exposure_counts['inside_zero_pann_no_raw_event_history'] += 1
        elif sum(row['query_ever_loaded_truth_overlap_m2']) == 0:
            exposure_counts['inside_zero_pann_no_previously_loaded_truth_key'] += 1
        else:
            exposure_counts['inside_zero_pann_with_previously_loaded_truth_key'] += 1
    large = np.max(np.abs(obs), axis=1)
    top = np.argsort(large)[-10:][::-1]
    return {'status': 'audited', 'rows': len(obs), 'files': inventory,
            'normalization': 'Fixed engineering normalization from PolicyMaterial.observation; GaussianActor.forward passes this vector directly into Linear/Tanh, without running mean/variance normalization or clipping. No activation/weight inference is made.',
            'raw_selected_log_rows': len(original_logs) if original_logs is not None else None,
            'raw_selected_log_common_field_differences_from_compact': dict(raw_log_differences),
            'all_saved_arrays_finite': {k: bool(np.all(np.isfinite(v))) for k, v in data.items()},
            'scalar_compact_max_abs_difference': scalar_errors,
            'pann_compact_max_abs_difference_m': pann_error,
            'next_pann_compact_max_abs_difference_m': next_pann_error,
            'within_episode_next_obs_to_next_current_obs_max_abs_difference': chain_error,
            'within_episode_chained_rows': len(chain),
            'phase_decode_mismatches': sum(PHASES[p] != log['phase'] for p, log in zip(phase, logs)),
            'query_mask_geometry_mismatches': int(np.count_nonzero(mask != interior(query))),
            'nonzero_pann_on_outside_query_leg': int(np.count_nonzero(nonzero & ~mask)),
            'pann_nonzero_rows': int(np.any(nonzero, axis=1).sum()),
            'next_pann_nonzero_rows': int(np.any(nxt[:, -12:] != 0, axis=1).sum()),
            'next_pann_nonzero_true_terminal_rows': int(np.sum(np.any(nxt[:, -12:] != 0, axis=1) & data['terminated'].ravel().astype(bool))),
            'pann_nonzero_foot_rows': int(nonzero.sum()),
            'pann_negative_values': int(np.count_nonzero(recovery < 0)),
            'pann_horizon_increase_over_1e_12m': int(np.count_nonzero(np.diff(recovery, axis=2) > 1e-12)),
            'pann_max_abs_m': float(np.abs(recovery).max()),
            'quaternion_norm_max_abs_error': float(np.max(np.abs(np.linalg.norm(obs[:, 3:7], axis=1)-1))),
            'phase_one_hot_max_error': float(np.max(abs(obs[:, 80:89].sum(axis=1)-1))),
            'finite_task_channels_all_zero': bool(np.all(obs[:, 91:93] == 0)),
            'plan_age_s': summary(obs[:, 93] * .35),
            'history_newest_to_current_max_abs_normalized_difference': history_errors,
            'raw_event_history_episodes': len(event_rows), 'pann_exposure_row_categories': dict(exposure_counts),
            'pann_exposure_rows': exposure,
            'block_statistics': {k: {'range': [lo, hi], 'absolute_values': summary(abs(obs[:, lo:hi]))}
                                 for k, (lo, hi) in SCHEMA.items()},
            'current_dimension_statistics': dimension_statistics(obs),
            'next_dimension_statistics': dimension_statistics(nxt),
            'largest_current_observation_rows': [dict(transition=int(indexes[i]), episode=logs[i]['episode'],
                action_start_s=logs[i]['time_s']-logs[i]['dt_s'], dimension=int(np.argmax(abs(obs[i]))),
                value=float(obs[i, np.argmax(abs(obs[i]))])) for i in top]}


def compact_domain_audit():
    rows = []
    for p in sorted(BASE.glob('train-job-20261105T032354Z-w39-*-ep*.json')):
        result = read_json(p)
        if not result.get('reason', '').startswith('material_domain_exit'):
            continue
        material = result['material']
        peak = material['rate_extrema']['internal_qdot_s_inv']
        foot = material['rate_extrema']['actual_foot_vz_over_H_s_inv']
        rows.append({'attempt': result['attempt'], 'physical_terminal_s': result['simulation_s'],
                     'max_abs_principal_strain': material['max_abs_principal_strain'],
                     'equivalent_max_normal_indentation_mm': .05*material['max_abs_principal_strain']*1000,
                     'max_newton_residual_original_MPa': material['max_residual'],
                     'peak_qdot_is_terminal_material_state': abs(peak['time_s']-result['simulation_s']-.0002)<1e-8,
                     'peak_qdot_boundary_transition': peak['boundary_transition'],
                     'peak_qdot_leg': LEGS[peak['leg']],
                     'peak_qdot_foot_vz_m_s': peak['foot_vz_over_H_s_inv']*.05,
                     'peak_qdot_e_after': peak['e_after'],
                     'peak_qdot_patch_area_m2': peak['area_after_m2'],
                     'peak_qdot_patch_candidate_force_N': peak['force_after_N'],
                     'largest_loaded_foot_vz_m_s': foot['foot_vz_m_s'],
                     'largest_loaded_foot_vz_time_s': foot['time_s'],
                     'full_rate_extrema': material['rate_extrema']})
    return {'rows': rows, 'n': len(rows),
            'maximum_newton_residual_original_MPa': max(r['max_newton_residual_original_MPa'] for r in rows),
            'terminal_peak_qdot_boundary_counts': dict(Counter(r['peak_qdot_boundary_transition'] for r in rows)),
            'terminal_peak_qdot_leg_counts': dict(Counter(r['peak_qdot_leg'] for r in rows)),
            'largest_loaded_foot_vertical_speed_m_s': summary([abs(r['largest_loaded_foot_vz_m_s']) for r in rows])}


def original_domain_failure_audit(assets):
    records = []
    for branch in 'ab':
        for row in assets.select(branch, '/material-failures.jsonl'):
            attempt = row.get('attempt')
            if attempt is None:
                continue
            result = read_json(BASE / f'{attempt}.json')
            if not result['reason'].startswith('material_domain_exit'):
                continue
            failures = assets.jsonl(branch, row)
            if failures is None:
                continue
            for f in failures:
                bottom = np.asarray(f['bottom_m'])
                velocity = np.asarray(f['velocity_m_s'])
                force = np.asarray(f['force_N'])
                loaded = force[:, 2] > 0
                leg = int(np.argmax(np.where(loaded, -bottom[:, 2], -np.inf)))
                friction = -.5*force[:, 2, None]*velocity[:, :2]/np.sqrt(np.sum(velocity[:, :2]**2, axis=1, keepdims=True)+.01**2)
                rate = result['material']['rate_extrema']['internal_qdot_s_inv']
                records.append({'attempt': attempt, 'physical_terminal_time_s': f['physical_state_time_s'],
                    'material_candidate_state_time_s': f['material_state_time_s'],
                    'candidate_ahead_of_physical_state_s': f['material_state_time_s']-f['physical_state_time_s'],
                    'physical_terminal_matches_compact_result': abs(f['physical_state_time_s']-result['simulation_s'])<1e-9,
                    'failure_strain_matches_compact_result': f['max_abs_principal_strain']==result['material']['max_abs_principal_strain'],
                    'domain_exit_loaded_foot': LEGS[leg], 'bottom_xyz_m': bottom[leg].tolist(),
                    'vertical_velocity_m_s': float(velocity[leg, 2]),
                    'candidate_normal_force_N_not_applied': float(force[leg, 2]),
                    'normal_strain_from_foot_bottom': float(-bottom[leg, 2]/.05),
                    'normal_strain_difference_from_failure_max': float(-bottom[leg, 2]/.05-f['max_abs_principal_strain']),
                    'candidate_friction_formula_max_abs_difference_N': float(abs(force[:, :2]-friction).max()),
                    'peak_internal_qdot_cell_boundary_type': rate['boundary_transition'],
                    'peak_internal_qdot_cell_area_m2': rate['area_after_m2'],
                    'peak_internal_qdot_cell_is_not_unique_domain_cause': 'All displacement-loaded cells under one foot share its imposed e11=bottom_z/H; the cell with largest qdot need not uniquely cause the domain exit.'})
    return {'count': len(records), 'records': records,
            'max_normal_strain_geometry_difference': max(abs(r['normal_strain_difference_from_failure_max']) for r in records) if records else None,
            'max_friction_formula_difference_N': max(r['candidate_friction_formula_max_abs_difference_N'] for r in records) if records else None,
            'max_material_candidate_time_offset_error_vs_0p2ms_s': max(abs(r['candidate_ahead_of_physical_state_s']-.0002) for r in records) if records else None}


def missed_observer_pulses(assets):
    records = []
    for branch, episode, transition, start, stop, ticks in [
        ('a', 269, 8136, 8082, 8337, [15.448, 15.450]),
        ('b', 258, 7808, 7570, 7825, [13.104, 13.106]),
    ]:
        batch = assets.select(branch, f'/rollout-{start:06d}-{stop:06d}.npz')[0]
        original = assets.npz(branch, batch)
        if original is None:
            continue
        obs = original['obs'][transition-start].reshape(-1)
        query = obs[94:102].reshape(4, 2).astype(float)*3
        attempt = f'train-job-20261105T032354Z-w39-{branch}-PANN_PPO_S5-s{1 if branch=="a" else 2}-ep{episode:05d}'
        events = assets.jsonl(branch, assets.select(branch, '/material-events.jsonl', attempt)[0])
        trajectory = assets.npz(branch, assets.select(branch, '/trajectory.npz', attempt)[0])
        if events is None or trajectory is None:
            continue
        log = next(row for row in compact(branch)[0] if row['transition'] == transition)
        action_time = log['time_s']-log['dt_s']
        keys = footprint(query[1])
        prior = [e for e in events if tuple(e['world_key']) in keys and e['time_s'] <= action_time+1e-8]
        points = []
        for t in ticks:
            i = int(np.argmin(abs(trajectory['time']-t)))
            foot = trajectory['feet'][i, 1]
            points.append({'time_s': float(trajectory['time'][i]),
                           'FR_geom_center_xyz_m': foot.tolist(),
                           'FR_bottom_z_m': float(foot[2]-.022),
                           'FR_sole_max_x_m': float(foot[0]+HALF),
                           'FR_footprint_world_keys': [list(k) for k in footprint(foot[:2])],
                           'FR_query_prior_loaded_keys_currently_covered': [list(k) for k in footprint(foot[:2])
                                    if k in {tuple(e['world_key']) for e in prior if e['type'] in ('load', 'revisit')} ]})
        ai = int(np.argmin(abs(trajectory['time']-action_time)))
        phase = int(np.argmax(obs[80:89]))
        base = np.r_[[3 if phase in (1, 2, 6) else 0, 0]-obs[:2].astype(float)*3, float(obs[2])*.3]
        observed_bottom = obs[39:51].reshape(4, 3).astype(float)*.3+base
        inferred_offsets = trajectory['feet'][ai, :, 2]-observed_bottom[:, 2]
        records.append({'branch': branch, 'episode': episode, 'transition': transition,
                        'action_time_s': action_time, 'query_FR_xy_m': query[1].tolist(),
                        'query_pann_4x3_m': (obs[-12:].astype(float)*.001).reshape(4, 3).tolist(),
                        'true_history_events_on_overlapping_keys': prior,
                        'bracketing_2ms_observer_times': points,
                        'radius_plus_floor_z_from_independent_saved_actor_and_geometry_m': inferred_offsets.tolist(),
                        'bottom_offset_used_m': .022,
                        'interpretation': 'At the earlier 2ms observation, the sole had not yet covered the queried virgin x-column; at the later observation, the foot bottom was already above the undeformed surface. Truth-only loading occurred between these observations. No constitutive or network rerun is needed to explain zero PANN for these keys.'})
    return records


def trajectory_audit(branch, episode, assets):
    attempt = f'train-job-20261105T032354Z-w39-{branch}-PANN_PPO_S5-s{1 if branch=="a" else 2}-ep{episode:05d}'
    rows = assets.select(branch, '/trajectory.npz', attempt)
    if not rows:
        return {'status': 'not_in_release_selection'}
    data = assets.npz(branch, rows[0])
    if data is None:
        return {'status': 'not_downloaded'}
    result = read_json(BASE / f'{attempt}.json')
    times = data['time']
    records = []
    for target in [0., 2., 5., 10., 15., 20., 30., 40., 50., float(times[-1])]:
        if target > times[-1]+1e-8:
            continue
        i = int(np.argmin(abs(times-target)))
        records.append({'time_s': float(times[i]), 'diagnostic_time_s': float(data['diagnostic_time'][i]),
                        'base_xyz_m': data['qpos'][i, :3].tolist(),
                        'base_velocity_m_s': data['qvel'][i, :3].tolist(),
                        'body_target_m': data['body_target'][i].tolist(),
                        'command_velocity_m_s': data['command_velocity'][i].tolist(),
                        'cap_m_s': float(data['requested_speed_cap'][i]),
                        'phase': PHASES[int(data['phase'][i])],
                        'feet_xy_m': data['feet'][i, :, :2].tolist()})
    limits = np.array([23.7, 23.7, 45.43]*4)
    terminal_window = times >= times[-1]-.010-1e-8
    terminal = []
    for i in np.flatnonzero(terminal_window):
        terminal.append({'time_s': float(times[i]), 'diagnostic_time_s': float(data['diagnostic_time'][i]),
                         'base_xyz_m': data['qpos'][i, :3].tolist(),
                         'feet_xyz_m': data['feet'][i].tolist(),
                         'foot_geom_center_vz_m_s': data['feet_velocity'][i, :, 2].tolist(),
                         'actual_grf_z_N_at_diagnostic_time': data['actual_grf'][i, :, 2].tolist()})
    late = times >= 10
    return {'status': 'audited', 'attempt': attempt, 'reason': result['reason'],
            'rows': len(times), 'time_mismatched_diagnostic_rows': int(np.count_nonzero(abs(times-data['diagnostic_time'])>1e-9)),
            'max_base_x_m': float(data['qpos'][:, 0].max()), 'min_base_z_m': float(data['qpos'][:, 2].min()),
            'max_abs_base_y_m': float(abs(data['qpos'][:, 1]).max()),
            'native_or_material_contact_rows': int(np.any(data['contact'], axis=1).sum()),
            'any_inside_foot_rows': int(np.any(interior(data['feet'][:, :, :2]), axis=1).sum()),
            'phase_counts': dict(Counter(PHASES[int(x)] for x in data['phase'])),
            'late_10s_onward_requested_cap_m_s': summary(data['requested_speed_cap'][late]),
            'late_10s_onward_command_x_velocity_m_s': summary(data['command_velocity'][late, 0]),
            'late_10s_onward_base_x_velocity_m_s': summary(data['qvel'][late, 0]),
            'late_10s_onward_base_y_m': summary(data['qpos'][late, 1]),
            'rows_unclipped_pd_outside_original_torque_bound': int(np.any(abs(data['unclipped_pd_request'])>limits+1e-8, axis=1).sum()),
            'max_abs_unclipped_pd_Nm': float(abs(data['unclipped_pd_request']).max()),
            'time_points': records, 'terminal_10ms_control_rows': terminal,
            'terminal_trajectory_caveat': 'The runtime appends a terminal row by replacing only time/qpos/qvel/rpy; all other copied fields retain diagnostic_time. Do not pair a copied terminal foot/force field with the terminal state time.'}


def dense_material_audit(branch, episode, assets):
    attempt = f'train-job-20261105T032354Z-w39-{branch}-PANN_PPO_S5-s{1 if branch=="a" else 2}-ep{episode:05d}'
    result = read_json(BASE / f'{attempt}.json')
    selected = [r for r in assets.rows[branch] if r.get('attempt') == attempt]
    failure_rows = [r for r in selected if r['source_path'].endswith('/material-failures.jsonl')]
    failure = assets.jsonl(branch, failure_rows[0]) if failure_rows else None
    mats = [r for r in selected if re.search(r'/material-\d+\.npz$', r['source_path'])]
    if not mats:
        return {'status': 'dense_material_not_in_published_selection', 'attempt': attempt,
                'failure_records': failure}
    chunks = []
    for r in mats:
        loaded = assets.npz(branch, r, keys=['foot', 'rates'])
        if loaded is not None and loaded['foot'].size:
            chunks.append(loaded['foot'])
    if len(chunks) != len(mats):
        return {'status': 'dense_chunks_incomplete', 'attempt': attempt,
                'loaded_chunks': len(chunks), 'selected_chunks': len(mats), 'failure_records': failure}
    foot = np.concatenate(chunks)
    foot = foot[np.argsort(foot[:, 0])]
    physicals = [r for r in selected if re.search(r'/physical-\d+\.npz$', r['source_path'])]
    phys_chunks = []
    for r in physicals:
        loaded = assets.npz(branch, r)
        if loaded is not None:
            phys_chunks.append(loaded['physical'])
    physical = np.concatenate(phys_chunks) if phys_chunks else None
    force = foot[:, 30:42].reshape(-1, 4, 3)
    bottom = foot[:, 6:18].reshape(-1, 4, 3)
    velocity = foot[:, 18:30].reshape(-1, 4, 3)
    applied = foot[:, 0] < result['simulation_s']-1e-9
    soft = foot[:, 2:6].astype(bool)
    contact = soft & (force[:, :, 2] > 0)
    first_loaded = int(np.flatnonzero(np.any(contact, axis=1))[0]) if contact.any() else None
    terminal_leg = int(np.argmax(foot[-1, 42:46]))
    last_start = len(foot)-1
    while last_start > 0 and contact[last_start-1, terminal_leg]:
        last_start -= 1
    alignment = {}
    if physical is not None:
        pindex = {round(float(t), 7): i for i, t in enumerate(physical[:, 0])}
        fi, pi = [], []
        for i, t in enumerate(foot[:, 0]):
            if round(float(t), 7) in pindex:
                fi.append(i); pi.append(pindex[round(float(t), 7)])
        sf = soft[fi]
        difference = physical[pi, 30:42].reshape(-1, 4, 3)-force[fi]
        alignment = {'physical_rows': len(physical), 'material_boundary_rows_with_physical_force_row': len(fi),
                     'material_only_rows': len(foot)-len(fi),
                     'soft_foot_force_max_abs_difference_N': float(abs(difference[sf]).max()) if sf.any() else 0.}
    last = []
    for i in np.flatnonzero(foot[:, 0] >= result['simulation_s']-.005-1e-8):
        last.append({'boundary_time_s': float(foot[i, 0]), 'material_state_time_s': float(foot[i, 1]),
                     'applied_before_terminal': bool(applied[i]), 'soft': soft[i].tolist(),
                     'bottom_z_m': bottom[i, :, 2].tolist(), 'bottom_xy_m': bottom[i, :, :2].tolist(),
                     'bottom_vz_m_s': velocity[i, :, 2].tolist(), 'normal_force_N': force[i, :, 2].tolist(),
                     'indentation_m': foot[i, 42:46].tolist(), 'running_max_strain': float(foot[i, 46]),
                     'running_max_newton_residual_original_MPa': float(foot[i, 47])})
    return {'status': 'audited', 'attempt': attempt, 'reason': result['reason'],
            'rows': len(foot), 'first_load_material_state_s': float(foot[first_loaded, 1]) if first_loaded is not None else None,
            'force_to_physical_alignment': alignment,
            'material_candidate_rows_not_physically_applied': int((~applied).sum()),
            'max_actually_applied_material_foot_normal_N': float(force[applied, :, 2].max()),
            'max_candidate_material_foot_normal_N': float(force[:, :, 2].max()),
            'terminal_leg': LEGS[terminal_leg],
            'terminal_leg_last_loading_boundary_s': float(foot[last_start, 0]),
            'terminal_leg_last_loading_indent_mm': float(-bottom[last_start, terminal_leg, 2]*1000),
            'terminal_leg_last_loading_vz_m_s': float(velocity[last_start, terminal_leg, 2]),
            'terminal_leg_first_load_material_state_to_terminal_physical_time_ms': float((result['simulation_s']-foot[last_start, 1])*1000),
            'terminal_leg_first_load_boundary_to_terminal_boundary_ms': float((result['simulation_s']-foot[last_start, 0])*1000),
            'terminal_leg_candidate_indentation_mm': float(-bottom[-1, terminal_leg, 2]*1000),
            'terminal_leg_candidate_vz_m_s': float(velocity[-1, terminal_leg, 2]),
            'terminal_leg_candidate_normal_N': float(force[-1, terminal_leg, 2]),
            'terminal_leg_last_applied_normal_N': float(force[np.flatnonzero(applied)[-1], terminal_leg, 2]),
            'failure_records': failure, 'terminal_5ms_material_rows': last,
            'interpretation': 'Each domain-exit last material row is solved with the pre-integration foot boundary and then rejected before mj_applyFT and mj_step. Its force is a candidate, not an experienced force. No constitutive replay was performed.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=Path, required=True)
    args = parser.parse_args()
    assets = Assets(args.raw_dir)
    report = {'schema_version': 'W39_raw_material_learning_v1', 'source_head': HEAD,
              'method': 'Read original saved rollout/observation/material arrays and event records; arithmetic only. No physics, optimizer, constitutive or neural-network recomputation.',
              'observation_layout_zero_based_half_open': SCHEMA,
              'foot_order': LEGS,
              'raw_rollouts': {branch: rollout_audit(branch, assets) for branch in 'ab'},
              'zero_pann_with_truth_overlap_original_pulses': missed_observer_pulses(assets),
              'domain_exit_compact_context': compact_domain_audit(),
              'all_domain_exit_original_failures': original_domain_failure_audit(assets),
              'selected_trajectories': {f'{branch}{episode}': trajectory_audit(branch, episode, assets)
                                        for branch, episode in [('a', 254), ('a', 260), ('b', 256), ('b', 261), ('b', 272)]},
              'domain_exit_original_material': {f'{branch}{episode}': dense_material_audit(branch, episode, assets)
                                               for branch, episode in [('a', 254), ('a', 260), ('b', 256)]},
              'read_originals': assets.read, 'missing_requested_originals': assets.missing}
    out = HERE / 'material-learning-audit.json'
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(out), 'original_files_read': len(assets.read),
                      'missing_originals': len(assets.missing),
                      'rollout_status': {k: v['status'] for k, v in report['raw_rollouts'].items()}}))


if __name__ == '__main__':
    main()
