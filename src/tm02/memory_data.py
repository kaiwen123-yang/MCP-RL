"""Frozen observable arrays for the fixed three-arm material-memory study."""
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .reference import Reference, random_path


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _grid(end, dt):
    """Keep dt exactly, with a final shorter increment when the endpoint requires it."""
    count = int(np.floor(end/dt+1e-10))
    time = np.arange(count+1, dtype=float)*dt
    if abs(time[-1]-end) < 1e-10:
        time[-1] = end
    else:
        time = np.r_[time, end]
    return time


def _envelope(time, points, offset=0.):
    points = np.asarray(points, dtype=float)
    return np.interp(time-offset, points[:, 0], points[:, 1], left=0., right=0.)


def _random(config, seed, evaluation=False):
    data = copy.deepcopy(config['data'])
    data['evaluation' if evaluation else 'train']['strain_seed'] = int(seed)
    return random_path(data, evaluation)


def _training(study, base, base_out, out):
    destination = out/'training'
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination/'data.json'
    if index_path.exists():
        return json.loads(index_path.read_text(encoding='utf-8'))
    reference = Reference(base['reference_material']['parameters'])
    original = np.load(base_out/'observations.npz')
    original_reference = np.load(base_out/'reference.npz')
    extra = study['arms'][1]
    random_source = destination/'random1000-reference.npz'
    if not random_source.exists():
        times = [original['train_time']]
        strains = [original['train_strain']]
        stresses = [original_reference['train_stress']]
        internal = [original_reference['train_q']]
        residuals = [original_reference['train_residual']]
        iterations = [original_reference['train_newton_iterations']]
        knots = {key: original[key] for key in original.files if key.startswith('train_knot_')}
        seeds = list(base['data']['train']['strain_seeds'])
        for index, seed in enumerate(range(extra['additional_strain_seed_start'],
                                          extra['additional_strain_seed_start']+
                                          extra['additional_strain_seed_count']), 100):
            time, strain, kt, ke = _random(base, seed)
            response = reference.path(time, strain, base['numerics'])
            times.append(time[None])
            strains.append(strain[None])
            stresses.append(response['stress'][None])
            internal.append(response['q'][None])
            residuals.append(response['residual'][None])
            iterations.append(response['newton_iterations'][None])
            knots[f'train_knot_time_{index:03d}'] = kt
            knots[f'train_knot_strain_{index:03d}'] = ke
            seeds.append(seed)
            if (index+1) % 100 == 0:
                print(f'Frozen random training reference {index+1}/1000.', flush=True)
        np.savez_compressed(random_source, time=np.concatenate(times),
                            strain=np.concatenate(strains), stress=np.concatenate(stresses),
                            q=np.concatenate(internal), residual=np.concatenate(residuals),
                            newton_iterations=np.concatenate(iterations), seeds=np.asarray(seeds),
                            **knots)
    random = np.load(random_source)
    recovery_source = destination/'recovery50-reference.npz'
    recovery_metadata_path = destination/'recovery50.json'
    rc = study['recovery_training']
    if not recovery_source.exists():
        histories, commands, descriptions = [], [], []
        time = _grid(rc['total_time_s'], rc['dt_s'])
        index = 0
        for mode, count in rc['mode_counts'].items():
            direction = np.asarray(rc['stress_directions'][mode])
            for local_index in range(count):
                seed = rc['parameter_seed_start']+index
                rng = np.random.default_rng(seed)
                preload = float(rng.uniform(*rc['preload_amplitude_uniform_MPa']))
                probe = float(rng.uniform(*rc['probe_amplitude_uniform_MPa']))
                delay = rc['zero_stress_recovery_delays_s'][
                    local_index % len(rc['zero_stress_recovery_delays_s'])]
                recovery_start = rc['preload_unit_envelope'][-1][0]
                probe_start = recovery_start+delay
                amplitude = preload*_envelope(time, rc['preload_unit_envelope'])
                amplitude += probe*_envelope(time, rc['probe_unit_envelope'], probe_start)
                command = amplitude[:, None]*direction
                histories.append(reference.path(time, command, base['numerics'], stress_control=True))
                commands.append(command)
                descriptions.append(dict(index=index, mode=mode, parameter_seed=seed,
                    preload_component_amplitude_MPa=preload, probe_component_amplitude_MPa=probe,
                    recovery_delay_s=delay, recovery_start_s=recovery_start,
                    probe_start_s=probe_start, initial_state='virgin equilibrium'))
                index += 1
        np.savez_compressed(recovery_source,
            **{key: np.stack([row[key] for row in histories]) for key in histories[0]},
            prescribed_stress=np.stack(commands))
        _json(recovery_metadata_path, descriptions)
    recovery = np.load(recovery_source)
    noises = {}
    noise_config = study['training_noise']
    increments = study['training']['increments_per_history']
    for realization in noise_config['realizations']:
        noise_id = realization['id']
        random_noise = np.stack([np.random.default_rng(seed).normal(
            noise_config['mean_MPa'], noise_config['std_MPa'], (increments, 6))
            for seed in range(realization['random_seed_start'],
                              realization['random_seed_start']+realization['random_path_count'])])
        if noise_id == 0:
            random_noise[:100] = original['training_noise']
        recovery_noise = np.stack([np.random.default_rng(seed).normal(
            noise_config['mean_MPa'], noise_config['std_MPa'], (increments, 6))
            for seed in range(realization['recovery_seed_start'],
                              realization['recovery_seed_start']+rc['history_count'])])
        noises[noise_id] = random_noise, recovery_noise
    records = []
    for arm in study['arms']:
        name = arm['id']
        if name == 'A_random100':
            arrays = dict(train_time=original['train_time'], train_strain=original['train_strain'],
                          ideal_stress=original['ideal_stress'], train_seeds=random['seeds'][:100])
            count = 100
        elif name == 'B_random1000':
            arrays = dict(train_time=random['time'], train_strain=random['strain'],
                          ideal_stress=random['stress'][:, 1:].copy(), train_seeds=random['seeds'])
            arrays['ideal_stress'][:100] = original['ideal_stress']
            count = 1000
        else:
            arrays = dict(train_time=np.concatenate((original['train_time'][:50], recovery['time'])),
                train_strain=np.concatenate((original['train_strain'][:50], recovery['strain'])),
                ideal_stress=np.concatenate((original['ideal_stress'][:50], recovery['stress'][:, 1:])),
                train_seeds=np.r_[random['seeds'][:50],
                                  np.arange(rc['parameter_seed_start'], rc['parameter_seed_start']+50)])
            count = 50
        for noise_id, (random_noise, recovery_noise) in noises.items():
            noise = random_noise[:count] if name != 'C_mixed100' else np.concatenate(
                (random_noise[:50], recovery_noise))
            arrays[f'training_noise{noise_id}'] = noise
            observed = arrays['ideal_stress']+noise
            if noise_id == 0:
                shared = min(count, 100)
                observed[:shared] = original['noisy_stress'][:shared]
            arrays[f'noise{noise_id}_stress'] = observed
        path = destination/name/'observations.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        records.append(dict(arm=name, histories=arm['histories'], increments_per_history=increments,
            sample_count_including_initial_state=increments+1,
            observations=str(path.relative_to(out)).replace('\\', '/'), sha256=_sha256(path),
            actual_noise=[dict(realization=i, mean_MPa=float(arrays[f'training_noise{i}'].mean()),
                              std_MPa=float(arrays[f'training_noise{i}'].std())) for i in noises],
            shared_first_50_array_sha256={key: _array_hash(value[:50])
                for key, value in arrays.items() if key != 'train_seeds'}))
    index = dict(arms=records, source_run=base_out.name,
        source_observations_sha256=_sha256(base_out/'observations.npz'),
        source_reference_sha256=_sha256(base_out/'reference.npz'),
        reference_q_scope='Only in separate reference archives; absent from training observations.',
        normalization='Use saved second-round ideal/noisy weight norms unchanged; not estimated here.',
        initial_state='Each history is independent virgin equilibrium; no within-history resets.',
        noise_realizations=noise_config['realizations'],
        reference_arrays={path.name: _sha256(path) for path in (random_source, recovery_source)})
    _json(index_path, index)
    return index


def _final_histories(study, base, revisit):
    """Yield prescribed boundaries and their fixed physical-condition metadata."""
    families = {row['id']: row for row in study['final_evaluation']['families']}
    dt = study['final_evaluation']['default_dt_s']
    family = families['F1_random_original']
    for index in range(family['conditions']):
        seed = family['strain_seed_start']+index
        config = copy.deepcopy(base)
        config['data']['evaluation'].update(increments=family['increments'], dt_s=family['dt_s'])
        time, strain, kt, ke = _random(config, seed, True)
        identity = f"{family['id']}-{index:03d}"
        yield dict(history_id=identity, condition_id=identity, family=family['id'],
                   mode=family['mode'], cluster_id=str(seed), strain_seed=seed), time, strain, dict(
                       knot_time=kt, knot_strain=ke)
    family = families['F2_low_amplitude_rate']
    index = 0
    for seed in family['base_path_seeds']:
        config = copy.deepcopy(base)
        f1 = families['F1_random_original']
        config['data']['evaluation'].update(increments=f1['increments'], dt_s=f1['dt_s'])
        base_time, _, kt, ke = _random(config, seed, True)
        spline = CubicSpline(kt, ke, bc_type=base['data']['spline_boundary_condition'])
        for amplitude, rate in itertools.product(family['amplitude_factors'], family['time_scale_factors']):
            endpoint = f1['increments']*f1['dt_s']*rate
            time = _grid(endpoint, dt)
            strain = amplitude*spline(time/rate)
            identity = f"{family['id']}-{index:03d}"
            yield dict(history_id=identity, condition_id=identity, family=family['id'],
                mode=family['mode'], cluster_id=str(seed), base_path_seed=seed,
                amplitude_factor=amplitude, time_scale_factor=rate), time, strain, dict(
                    base_knot_time=kt, base_knot_strain=ke, base_sample_time=base_time)
            index += 1
    family = families['F3_strain_relaxation']
    for index, (direction_id, amplitude, hold) in enumerate(itertools.product(
            range(len(family['directions'])), family['component_amplitudes'], family['hold_times_s'])):
        ramp, unload = family['ramp_time_s'], family['unload_time_s']
        relaxed = ramp+hold
        post = relaxed+unload
        time = _grid(post+family['post_unload_strain_zero_hold_s'], dt)
        amplitude_path = amplitude*_envelope(time, [[0, 0], [ramp, 1], [relaxed, 1], [post, 0]])
        strain = amplitude_path[:, None]*np.asarray(family['directions'][direction_id])
        identity = f"{family['id']}-{index:03d}"
        yield dict(history_id=identity, condition_id=identity, family=family['id'],
            mode=family['mode'], cluster_id=identity, direction_id=direction_id,
            component_amplitude=amplitude, hold_time_s=hold, relaxation_start_s=ramp,
            relaxation_end_s=relaxed, tail_start_s=ramp+0.8*hold, tail_end_s=relaxed,
            post_unload_start_s=post), time, strain, {}
    family = families['F4_stress_creep_recovery']
    for index, (peak, hold, ramp) in enumerate(itertools.product(
            family['peak_component_stress_MPa'], family['hold_times_s'], family['ramp_times_s'])):
        recovery_start = 2*ramp+hold
        endpoint = recovery_start+family['zero_external_stress_recovery_s']
        time = _grid(endpoint, dt)
        amplitude = peak*_envelope(time, [[0, 0], [ramp, 1], [ramp+hold, 1], [recovery_start, 0]])
        command = amplitude[:, None]*np.asarray(family['direction'])
        identity = f"{family['id']}-{index:03d}"
        yield dict(history_id=identity, condition_id=identity, family=family['id'],
            mode=family['mode'], cluster_id=identity, peak_component_stress_MPa=peak,
            hold_time_s=hold, ramp_time_s=ramp, recovery_start_s=recovery_start,
            recovery_end_s=endpoint), time, command, {}
    family = families['F5_compression_shear_cycles']
    prefix = revisit['preloading_histories'][0]
    prefix_knots = np.asarray(prefix['points_t_s_sigma11_MPa'], dtype=float)
    prefix_knots[:, 1] /= np.max(np.abs(prefix_knots[:, 1]))
    index = 0
    for seed in family['prefix_seeds']:
        prefix_peak = float(np.random.default_rng(seed).uniform(0.75, 5.5))
        for peak, ratio, cycles in itertools.product(family['normal_peaks_MPa'],
                family['shear_to_normal_ratios'], family['cycle_counts']):
            cycle_start = prefix['duration_s']
            recovery_start = cycle_start+cycles*family['cycle_period_s']
            endpoint = recovery_start+family['zero_external_stress_recovery_s']
            time = _grid(endpoint, dt)
            command = np.zeros((len(time), 6))
            command[:, 0] = prefix_peak*_envelope(time, prefix_knots)
            for cycle in range(cycles):
                envelope = _envelope(time, family['cycle_unit_envelope'],
                                     cycle_start+cycle*family['cycle_period_s'])
                command[:, 0] -= peak*envelope
                command[:, 3] += peak*ratio*((-1)**cycle)*envelope
            identity = f"{family['id']}-{index:03d}"
            yield dict(history_id=identity, condition_id=identity, family=family['id'],
                mode=family['mode'], cluster_id=str(seed), prefix_seed=seed,
                prefix_peak_MPa=prefix_peak, peak_component_stress_MPa=peak,
                shear_to_normal_ratio=ratio, cycle_count=cycles, cycle_start_s=cycle_start,
                recovery_start_s=recovery_start, recovery_end_s=endpoint), time, command, {}
            index += 1
    family = families['F6_revisit_pairs']
    for index, (preload_peak, delay, probe_peak) in enumerate(itertools.product(
            family['preload_peaks_MPa'], family['recovery_delays_s'], family['common_probe_peaks_MPa'])):
        condition_id = f"{family['id']}-{index:03d}"
        preload = copy.deepcopy(revisit['preloading_histories'][0])
        probe = copy.deepcopy(revisit['common_reload_probe'])
        for phase, peak in ((preload, preload_peak), (probe, probe_peak)):
            points = np.asarray(phase['points_t_s_sigma11_MPa'], dtype=float)
            points[:, 1] *= peak/np.max(np.abs(points[:, 1]))
            phase['points_t_s_sigma11_MPa'] = points.tolist()
        for candidate in revisit['current_action_candidates']:
            phases = [preload, candidate,
                      dict(duration_s=delay, points_t_s_sigma11_MPa=[[0, 0], [delay, 0]]), probe]
            times, loads, labels = [0.], [0.], [0]
            for phase_id, phase in enumerate(phases):
                local = _grid(phase['duration_s'], dt)[1:]
                offset = times[-1]
                times.extend((offset+local).tolist())
                loads.extend(_envelope(local, phase['points_t_s_sigma11_MPa']).tolist())
                labels.extend([phase_id]*len(local))
            time = np.asarray(times)
            command = np.zeros((len(time), 6))
            command[:, 0] = loads
            recovery_start = preload['duration_s']+candidate['duration_s']
            yield dict(history_id=condition_id+'-'+candidate['id'], condition_id=condition_id,
                family=family['id'], mode=family['mode'], cluster_id=condition_id,
                candidate=candidate['id'], preload_peak_MPa=preload_peak,
                probe_peak_MPa=probe_peak, recovery_delay_s=delay,
                recovery_start_s=recovery_start, recovery_end_s=recovery_start+delay,
                probe_start_s=recovery_start+delay), time, command, dict(phase=np.asarray(labels))


def prepare_data(study, base_config, revisit_config, base_out, out):
    """Save training and 700 reference arrays before any new model training."""
    base_out, out = Path(base_out), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = out/'data.json'
    if frozen.exists():
        return json.loads(frozen.read_text(encoding='utf-8'))
    training = _training(study, base_config, base_out, out)
    destination = out/'final_reference'
    destination.mkdir(exist_ok=True)
    reference = Reference(base_config['reference_material']['parameters'])
    histories = []
    for metadata, time, command, extras in _final_histories(study, base_config, revisit_config):
        path = destination/(metadata['history_id']+'.npz')
        stress_control = metadata['mode'] == 'stress_control'
        if not path.exists():
            response = reference.path(time, command, base_config['numerics'], stress_control=stress_control)
            np.savez_compressed(path, **response, command=command,
                **({'prescribed_stress': command} if stress_control else {'prescribed_strain': command}),
                **extras)
        metadata.update(arrays=path.name, sha256=_sha256(path), sample_count=len(time),
                        increments=len(time)-1, initial_time_s=float(time[0]),
                        final_time_s=float(time[-1]), dt_min_s=float(np.diff(time).min()),
                        dt_max_s=float(np.diff(time).max()))
        histories.append(metadata)
        if len(histories) % 25 == 0:
            print(f'Frozen complete final reference {len(histories)}/700.', flush=True)
    catalog = dict(histories=histories, physical_condition_count=len({r['condition_id'] for r in histories}),
        complete_reference_histories=len(histories), reference_material=base_config['reference_material'],
        numerics=base_config['numerics'], initial_state='Independent virgin equilibrium per complete history.',
        statistical_units='F2 base path and F5 prefix seed clusters; F6 paired candidates share condition_id.',
        frozen_before_training=True)
    _json(destination/'catalog.json', catalog)
    record = dict(run_id=study['run_id'], training=training,
        final_reference_catalog='final_reference/catalog.json',
        final_reference_catalog_sha256=_sha256(destination/'catalog.json'),
        physical_condition_count=catalog['physical_condition_count'],
        complete_reference_histories=catalog['complete_reference_histories'],
        frozen_before_training=True,
        study_config_sha256=hashlib.sha256(json.dumps(study, sort_keys=True).encode()).hexdigest())
    _json(frozen, record)
    print('All training/noise arrays and 600 final conditions / 700 complete histories frozen.', flush=True)
    return record
