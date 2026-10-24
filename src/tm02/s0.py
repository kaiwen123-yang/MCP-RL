"""Fixed S0 frame-supported contact experiment (SI geometry, MPa material law).

Each job is one complete two-coupon history. The unmodified constitutive law
sees sigma/gamma; physical stress is gamma times its output. No state labels,
training, reaction-force addition, or contact-by-contact restart is used.
"""
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .evaluate import integrate, write_json
from .material import load_weights
from .reference import Reference, IntegrationFailure, newton


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def catalog(config):
    grid = config['context_grid']
    contexts = []
    for i, (gamma, pad, gap, loads, retgap) in enumerate(itertools.product(
            grid['material_scale_gamma'], grid['prefix_loaded_pad'],
            grid['prefix_gap_after_window_s'], grid['future_peak_load_pairs_N'],
            grid['return_gap_after_hind_window_s'])):
        contexts.append(dict(context_id=i, gamma=gamma, prefix_pad=pad,
                             prefix_gap_s=gap, hind_N=loads['hind_P'],
                             return_N=loads['return_Q'], return_gap_s=retgap))
    c = config['current_candidates']
    actions = [dict(candidate_id=i, pad=p, waveform=w, wait_s=d)
               for i, (p, w, d) in enumerate(itertools.product(
                   c['pad'], c['waveform'], c['extra_wait_s']))]
    return contexts, actions


def commands(config, context, action):
    """One common clock, with explicit zero load between all contact events."""
    dt = config['execution']['dt_s']
    decision = 1 + context['prefix_gap_s']
    current = decision + 0.25 + action['wait_s']
    hind = current + 1.25
    ret = hind + 1 + context['return_gap_s']
    finish = ret + 1
    events = [
        dict(phase='prefix_front', start_s=0., end_s=1., pad=context['prefix_pad'],
             peak_N=100., waveform='early', foot=0),
        dict(phase='current_front', start_s=current, end_s=current + 1,
             pad=action['pad'], peak_N=100., waveform=action['waveform'], foot=0),
        dict(phase='hind', start_s=hind, end_s=hind + 1, pad='P',
             peak_N=context['hind_N'], waveform='early', foot=1),
        dict(phase='return_front', start_s=ret, end_s=finish, pad='Q',
             peak_N=context['return_N'], waveform='early', foot=0)]
    t = np.arange(round((finish + 2) / dt) + 1) * dt
    force = np.zeros((len(t), 2))
    foot_force = np.zeros_like(force)
    phase = np.full(len(t), 'post_task_free_recovery', dtype='U24')
    for a, b, label in [(1, decision, 'prefix_gap'),
                         (decision, current, 'approach_and_wait'),
                         (current + 1, hind, 'handoff'),
                         (hind + 1, ret, 'return_gap')]:
        phase[(t >= a - 1e-9) & (t < b - 1e-9)] = label
    resets = np.zeros((len(t), 2), bool)
    for ev in events:
        i = 'PQ'.index(ev['pad'])
        knots = np.array(config['waveforms'][ev['waveform'] + '_time_s_force_fraction'])
        values = np.interp(t - ev['start_s'], knots[:, 0], knots[:, 1], left=0, right=0)
        force[:, i] += ev['peak_N'] * values
        foot_force[:, ev['foot']] += ev['peak_N'] * values
        mask = (t >= ev['start_s'] - 1e-9) & (t < ev['end_s'] - 1e-9)
        phase[mask] = ev['phase']
        resets[round(ev['start_s'] / dt), i] = True
        ev['load_start_s'] = ev['start_s'] + (0.5 if ev['waveform'] == 'late' else 0.)
        ev['unload_s'] = ev['load_start_s'] + 0.5
    stress = np.zeros((len(t), 2, 6))
    stress[:, :, 0] = -force / (config['physical_scope']['area_m2'] * 1e6)
    meta = dict(context=context, action=action, events=events, decision_s=decision,
                current_start_s=current, current_end_s=current + 1,
                hind_start_s=hind, return_start_s=ret, finish_s=finish,
                completion_time_s=finish - decision)
    return dict(time=t, command_force_N=force, commanded_foot_force_N=foot_force,
                prescribed_physical_stress_MPa=stress, phase=phase,
                planned_contact_start=resets), meta


def reference_history(reference, t, command, settings, reset):
    """Reference.path equations with partial-failure output and explicit ablation events."""
    n = len(t)
    e, q, stress = (np.full((n, 6), np.nan) for _ in range(3))
    e[0] = q[0] = stress[0] = 0.
    residual = np.full(n, np.nan)
    iterations = np.full(n, -1, int)
    residual[0], iterations[0] = 0., 0
    de_dq = np.linalg.solve(reference.ceq + reference.cov, reference.cov)
    failure = None
    for i in range(1, n):
        if reset[i]:
            e[i] = q[i] = stress[i] = 0.
            residual[i], iterations[i] = 0., 0
            continue
        dt = t[i] - t[i-1]
        origin = np.linalg.solve(reference.ceq + reference.cov, command[i])
        def both(qi):
            return reference.residual_jacobian(qi, origin + de_dq @ qi, q[i-1], dt, de_dq)
        try:
            q[i], residual[i], iterations[i] = newton(
                lambda qi: both(qi)[0], lambda qi: both(qi)[1], q[i-1], settings, i)
        except (IntegrationFailure, np.linalg.LinAlgError) as error:
            failure = dict(step=i, time_s=float(t[i]), error=str(error))
            break
        e[i] = origin + de_dq @ q[i]
        stress[i] = reference.stress(e[i], q[i])
    return dict(strain=e, q=q, stress=stress, residual=residual,
                newton_iterations=iterations), failure


def feet(config, arrays, meta):
    """Continuous piecewise-linear travel; contact height is solved material height.

    Mode 0 contact, 1 raising, 2 clear flight/parking, 3 lowering. Clearance
    applies to clear flight, not the necessary passage through smaller gaps
    while raising/lowering. Speeds are checked on every actual time interval.
    """
    t = arrays['time']
    surface = -arrays['compression_m']
    physical = config['physical_scope']
    limits = config['contact']['limits']
    height = limits['flight_height_above_virgin_surface_m']
    lift = limits['min_raise_or_lower_duration_s']
    speed = limits['max_horizontal_foot_speed_m_s']
    centers = np.array([p['center_x_m'] for p in physical['pads']])
    position = np.zeros((len(t), 2, 2))  # foot, (world x, world z)
    position[:, 0, 0] = centers['PQ'.index(meta['context']['prefix_pad'])]
    position[:, 1, 0] = physical['hind_staging_x_m']
    position[:, :, 1] = height
    mode = np.full((len(t), 2), 2, np.int8)

    def interval(a, b):
        return (t >= a - 1e-9) & (t <= b + 1e-9)

    def translate(foot, a, x0, x1):
        duration = abs(x1 - x0) / speed
        if duration > 0:
            mask = interval(a, a + duration)
            position[mask, foot, 0] = x0 + (x1 - x0) * (t[mask] - a) / duration
        position[t >= a + duration - 1e-9, foot, 0] = x1

    def touch(foot, pad, a, b, approach=True):
        if approach:
            mask = interval(a - lift, a)
            u = (t[mask] - (a - lift)) / lift
            position[mask, foot, 1] = (1 - u) * height + u * surface[mask, pad]
            mode[mask, foot] = 3
        mask = interval(a, b)
        position[mask, foot, 1] = surface[mask, pad]
        mode[mask, foot] = 0
        mask = interval(b, b + lift)
        u = (t[mask] - b) / lift
        position[mask, foot, 1] = (1 - u) * surface[mask, pad] + u * height
        mode[mask, foot] = 1
        mode[np.isclose(t, b), foot] = 0

    prefix_pad = 'PQ'.index(meta['context']['prefix_pad'])
    current_pad = 'PQ'.index(meta['action']['pad'])
    touch(0, prefix_pad, 0., 0.5, approach=False)
    translate(0, 0.5 + lift, centers[prefix_pad], physical['front_staging_x_m'])
    translate(0, meta['decision_s'], physical['front_staging_x_m'], centers[current_pad])
    touch(0, current_pad, meta['current_start_s'], meta['current_end_s'])
    translate(0, meta['current_end_s'] + lift, centers[current_pad], centers[1])
    touch(0, 1, meta['return_start_s'], meta['return_start_s'] + 0.5)
    touch(1, 0, meta['hind_start_s'], meta['hind_start_s'] + 0.5)

    # All possible foot/coupon gaps are saved; lateral overlap is explicit.
    halfwidth = np.sqrt(physical['area_m2']) / 2
    overlap = np.abs(position[:, :, 0, None] - centers) <= halfwidth + 1e-12
    gap = position[:, :, 1, None] - surface[:, None, :]
    velocity = np.diff(position, axis=0) / np.diff(t)[:, None, None]
    arrays.update(foot_position_xz_m=position, foot_velocity_xz_m_s=velocity,
                  foot_motion_mode=mode, foot_pad_overlap=overlap, foot_pad_gap_m=gap)


def metrics(config, arrays, meta, failures):
    t = arrays['time']
    limits = config['contact']['limits']
    force, strain, delta = (arrays[k] for k in ('command_force_N', 'strain', 'compression_m'))
    step_work = (force[1:] + force[:-1]) / 2 * np.diff(delta, axis=0)
    # With free lateral stress this is the same boundary work as V sigma:depsilon.
    stress = arrays['physical_stress_MPa']
    tensor_work = (config['physical_scope']['area_m2'] * config['physical_scope']['thickness_m'] * 1e6
                   * np.sum((stress[1:] + stress[:-1]) / 2 * np.diff(strain, axis=0)
                            * np.array([1, 1, 1, 2, 2, 2]), axis=(1, 2)))
    arrays['signed_contact_work_increment_J'] = step_work
    arrays['positive_contact_work_increment_J'] = np.maximum(0., step_work)
    arrays['volume_stress_work_increment_J'] = tensor_work
    tensors = np.zeros((*strain.shape[:2], 3, 3))
    tensors[:, :, 0, 0], tensors[:, :, 1, 1], tensors[:, :, 2, 2] = (strain[:, :, i] for i in range(3))
    for component, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)), 3):
        tensors[:, :, a, b] = tensors[:, :, b, a] = strain[:, :, component]
    principal = np.full(strain.shape[:2], np.nan)
    valid = np.isfinite(tensors).all(axis=(-1, -2))
    principal[valid] = np.max(np.abs(np.linalg.eigvalsh(tensors[valid])), axis=-1)
    arrays['max_absolute_principal_strain'] = principal
    # Physical checks are time-resolved experimental observables, not a test suite.
    violation = np.zeros(len(t), bool)
    violations = []
    def record(name, bad, observed):
        hit = np.flatnonzero(bad)
        if len(hit):
            violation[hit] = True
            violations.append(dict(kind=name, first_time_s=float(t[hit[0]]),
                                   first_phase=str(arrays['phase'][hit[0]]),
                                   timepoints=int(len(hit)), observed_extreme=float(observed)))
    record('normal_force', np.any((force < -1e-10) | (force > limits['max_normal_force_N'] + 1e-10), axis=1), np.nanmax(force))
    record('material_compression', np.any(delta > limits['max_compression_m'], axis=1), np.nanmax(delta))
    record('principal_strain', np.any(principal > limits['max_absolute_principal_strain'], axis=1), np.nanmax(principal))
    v = arrays['foot_velocity_xz_m_s']
    for axis, key in enumerate(('horizontal', 'vertical')):
        speeds = np.abs(v[:, :, axis])
        bad = np.r_[False, np.any(speeds > limits[f'max_{key}_foot_speed_m_s'] + 1e-10, axis=1)]
        record(key + '_speed', bad, np.nanmax(speeds))
    gaps, overlap = arrays['foot_pad_gap_m'], arrays['foot_pad_overlap']
    record('penetration', np.any(overlap & (gaps < -1e-10), axis=(1, 2)), np.nanmin(gaps[overlap]))
    clear = arrays['foot_motion_mode'] == 2
    clear_gaps = arrays['foot_position_xz_m'][:, :, 1] - np.max(-delta, axis=1)[:, None]
    record('flight_clearance', np.any(clear & (clear_gaps < limits['min_flight_clearance_m'] - 1e-10), axis=1),
           np.nanmin(clear_gaps[clear]))
    # During every nonzero force, that same foot must occupy the loaded surface.
    foot_force = arrays['commanded_foot_force_N']
    active_gap = np.min(np.where(overlap, np.abs(gaps), np.inf), axis=2)
    contact_error = np.where(foot_force > 1e-8, active_gap, 0.)
    record('loaded_contact_gap', np.any(contact_error > 1e-10, axis=1), np.nanmax(contact_error))
    for failure in failures:
        if failure:
            violation[t >= failure['time_s'] - 1e-9] = True
    arrays['physical_violation'] = violation

    def mask(a, b):
        return (t >= a - 1e-9) & (t <= b + 1e-9)
    def score(a, b):
        samples = mask(a, b)
        steps = (t[:-1] >= a - 1e-9) & (t[1:] <= b + 1e-9)
        complete = bool(np.isfinite(delta[samples]).all())
        return dict(feasible=bool(complete and not np.any(violation[samples])),
                    peak=float(np.max(np.maximum(0., delta[samples]))) if complete else None,
                    work=float(np.maximum(0., step_work[steps]).sum()) if complete else None,
                    signed_work=float(step_work[steps].sum()) if complete else None)
    remaining = score(meta['decision_s'], meta['finish_s'])
    current = score(meta['current_start_s'], meta['current_end_s'])
    myopic = score(meta['decision_s'], meta['current_end_s'])
    future = score(meta['hind_start_s'], meta['finish_s'])
    row = dict(**meta['context'], **meta['action'], completion_time_s=meta['completion_time_s'],
               integration_complete=all(f is None for f in failures),
               full_history_feasible=bool(not np.any(violation)),
               remaining_feasible=remaining['feasible'], myopic_feasible=myopic['feasible'],
               max_remaining_pad_compression_m=remaining['peak'],
               positive_remaining_contact_work_J=remaining['work'], signed_remaining_contact_work_J=remaining['signed_work'],
               current_contact_peak_m=current['peak'], future_contact_peak_m=future['peak'],
               myopic_positive_work_J=myopic['work'], myopic_completion_time_s=meta['current_end_s']-meta['decision_s'],
               maximum_force_N=float(np.nanmax(force)), maximum_full_compression_m=float(np.nanmax(delta)),
               maximum_principal_strain=float(np.nanmax(principal)),
               maximum_horizontal_speed_m_s=float(np.nanmax(np.abs(v[:, :, 0]))),
               maximum_vertical_speed_m_s=float(np.nanmax(np.abs(v[:, :, 1]))),
               minimum_flight_clearance_m=float(np.nanmin(clear_gaps[clear])),
               max_force_compatibility_error_N=float(np.nanmax(np.abs(-stress[:, :, 0] * config['physical_scope']['area_m2'] * 1e6 - force))),
               max_boundary_volume_work_difference_J=float(np.nanmax(np.abs(step_work.sum(1)-tensor_work))),
               failure=failures, violations=violations)
    return row


_MODELS = {}


def worker_init():
    affinity = sorted(os.sched_getaffinity(0))
    index = multiprocessing.current_process()._identity[0] - 1
    os.sched_setaffinity(0, {affinity[index % len(affinity)]})
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')


def run_one(job):
    config, base, source, out, context, action, predictor = job
    folder = Path(out) / 'trajectories' / predictor['id']
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"c{context['context_id']:02d}-a{action['candidate_id']:02d}"
    record_path, data_path = folder / (stem + '.json'), folder / (stem + '.npz')
    if record_path.exists():
        return json.loads(record_path.read_text())
    begin = time.monotonic()
    arrays, meta = commands(config, context, action)
    command = arrays['prescribed_physical_stress_MPa'] / context['gamma']
    resets = arrays['planned_contact_start'] if predictor['id'] == 'reference_reset_planning' else np.zeros(command.shape[:2], bool)
    arrays['intentional_planning_reset'] = resets
    predictions, failures = [], []
    if predictor['type'] == 'frozen_learned_material':
        key = predictor['id']
        if key not in _MODELS:
            _MODELS[key] = load_weights(Path(source) / predictor['weights_relative_to_source_run'])
        w, norm = _MODELS[key]
    else:
        reference = Reference(base['reference_material']['parameters'])
    for pad in range(2):
        if predictor['type'] == 'frozen_learned_material':
            result, failure = integrate(w, norm, arrays['time'], command[:, pad], base['numerics'], stress_control=True)
        else:
            result, failure = reference_history(reference, arrays['time'], command[:, pad], base['numerics'], resets[:, pad])
        if failure:
            failure.update(pad='PQ'[pad], phase=str(arrays['phase'][failure['step']]))
        predictions.append(result)
        failures.append(failure)
    for name in ('strain', 'q', 'stress', 'residual', 'newton_iterations'):
        arrays[name] = np.stack([p[name] for p in predictions], axis=1)
    arrays['physical_stress_MPa'] = arrays.pop('stress') * context['gamma']
    arrays['equivalent_stress_MPa'] = command
    arrays['physical_residual_MPa'] = arrays.pop('residual') * context['gamma']
    arrays['compression_m'] = -config['physical_scope']['thickness_m'] * arrays['strain'][:, :, 0]
    feet(config, arrays, meta)
    row = metrics(config, arrays, meta, failures)
    row.update(predictor=predictor['id'], elapsed_s=time.monotonic()-begin,
               trajectory=data_path.relative_to(Path(out)).as_posix(),
               timepoints=len(arrays['time']), max_physical_residual_MPa=float(np.nanmax(arrays['physical_residual_MPa'])))
    np.savez_compressed(data_path, **arrays, metadata_json=json.dumps(meta), failures_json=json.dumps(failures))
    write_json(record_path, row)
    return row


def simulate(config, base, source, out):
    contexts, actions = catalog(config)
    jobs = [(config, base, str(source), str(out), ctx, act, pred)
            for pred in config['predictors'] for ctx in contexts for act in actions]
    with ProcessPoolExecutor(max_workers=config['execution']['workers'],
                             mp_context=multiprocessing.get_context('spawn'), initializer=worker_init) as pool:
        for i, row in enumerate(pool.map(run_one, jobs, chunksize=1), 1):
            if i % 24 == 0 or not row['integration_complete']:
                print(json.dumps(dict(completed=i, total=len(jobs), predictor=row['predictor'],
                                      context=row['context_id'], candidate=row['candidate_id'],
                                      integration_complete=row['integration_complete'],
                                      remaining_feasible=row['remaining_feasible'])), flush=True)


def collect(config, out):
    contexts, actions = catalog(config)
    return [json.loads((Path(out) / 'trajectories' / pred['id'] /
                        f"c{ctx['context_id']:02d}-a{act['candidate_id']:02d}.json").read_text())
            for pred in config['predictors'] for ctx in contexts for act in actions]
