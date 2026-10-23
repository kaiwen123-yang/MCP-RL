"""Complete, freely evolved responses for the fixed R2 memory study.

Catalog arrays are frozen before training.  This module is called only after
the twelve final-loss choices are frozen; known diagnostic catalogs are scored
separately.  A catalog path resolves its ``arrays`` entries against its parent.
Each model record supplies model_id, arm, cell, condition, noise_realization,
seed and an absolute weights path.  No reference internal state enters a
deployment call, including the prescribed-strain diagnostic replay.
"""
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np

from .evaluate import integrate, write_json
from .material import COMPONENTS, forces, load_weights
from .reference import Reference


IDENTITY = ('model_id', 'arm', 'cell', 'condition', 'noise_realization', 'seed')
STRAIN_METRICS = ('stress_MAE_MPa', 'stress_RMSE_MPa',
                  'hydrostatic_MAE_MPa', 'hydrostatic_RMSE_MPa',
                  'deviatoric_component_MAE_MPa', 'deviatoric_component_RMSE_MPa',
                  'relaxation_stress_MAE_MPa', 'relaxation_stress_RMSE_MPa',
                  'tail_stress_MAE_MPa', 'tail_stress_RMSE_MPa',
                  'post_unload_stress_MAE_MPa', 'post_unload_stress_RMSE_MPa',
                  'recovery_stress_MAE_MPa', 'recovery_stress_RMSE_MPa',
                  'reload_stress_MAE_MPa', 'reload_stress_RMSE_MPa')
STRESS_METRICS = ('full_history_strain_MAE', 'full_history_strain_RMSE',
                  'full_history_epsilon11_MAE', 'full_history_epsilon11_RMSE',
                  'peak_compression', 'reference_peak_compression',
                  'peak_compression_error', 'absolute_peak_compression_error',
                  'relative_peak_compression_error',
                  'reload_strain_MAE', 'reload_strain_RMSE', 'reload_epsilon11_MAE',
                  'recovery_strain_MAE', 'recovery_strain_RMSE', 'recovery_epsilon11_MAE',
                  'free_recovery_residual_epsilon11', 'reference_free_recovery_residual_epsilon11',
                  'free_recovery_residual_epsilon11_error',
                  'absolute_free_recovery_residual_epsilon11_error',
                  'recovery_integrated_compression_s', 'reference_recovery_integrated_compression_s',
                  'recovery_integrated_compression_error_s',
                  'absolute_recovery_integrated_compression_error_s',
                  'recovery_epsilon11_absolute_error_integral_s',
                  'probe_integrated_compression_s', 'reference_probe_integrated_compression_s',
                  'probe_integrated_compression_error_s',
                  'absolute_probe_integrated_compression_error_s')
COMPONENT_METRICS = tuple(f'{prefix}_{c}{suffix}'
                          for c in COMPONENTS
                          for prefix, suffix in (('stress_MAE', '_MPa'), ('stress_RMSE', '_MPa'),
                                                 ('strain_MAE', ''), ('strain_RMSE', '')))
METRICS = STRAIN_METRICS + STRESS_METRICS + COMPONENT_METRICS
PAIR_METRICS = ('candidate_center_bias', 'absolute_candidate_center_bias',
                'action_difference_error', 'absolute_action_difference_error',
                'absolute_response_error', 'expected_regret', 'worst_case_regret')


def _csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def _npz(path):
    with np.load(path) as saved:
        return {k: saved[k] for k in saved.files}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _window(t, item, prefix):
    start = item.get(prefix + '_start_s')
    if start is None:
        return None
    return (t >= start - 1e-10) & (t <= item.get(prefix + '_end_s', float(t[-1])) + 1e-10)


def response_metrics(result, truth, item, complete, tolerance=1e-6):
    """Physical-unit errors; incomplete histories retain curves but no full score."""
    values = {key: None for key in METRICS}
    if not complete:
        return values
    t = result['time']
    if item['mode'] == 'strain_control':
        error = result['stress'] - truth['stress']
        full = error[1:]
        hydro = full[:, :3].mean(axis=-1)
        dev = full.copy()
        dev[:, :3] -= hydro[:, None]
        values.update(stress_MAE_MPa=float(np.abs(full).mean()),
                      stress_RMSE_MPa=float(np.sqrt(np.mean(full ** 2))),
                      hydrostatic_MAE_MPa=float(np.abs(hydro).mean()),
                      hydrostatic_RMSE_MPa=float(np.sqrt(np.mean(hydro ** 2))),
                      deviatoric_component_MAE_MPa=float(np.abs(dev).mean()),
                      deviatoric_component_RMSE_MPa=float(np.sqrt(np.mean(dev ** 2))))
        for i, component in enumerate(COMPONENTS):
            values[f'stress_MAE_{component}_MPa'] = float(np.abs(full[:, i]).mean())
            values[f'stress_RMSE_{component}_MPa'] = float(np.sqrt(np.mean(full[:, i] ** 2)))
        for prefix in ('relaxation', 'tail', 'post_unload', 'recovery', 'reload'):
            mask = _window(t, item, 'probe' if prefix == 'reload' else prefix)
            if mask is not None:
                values[f'{prefix}_stress_MAE_MPa'] = float(np.abs(error[mask]).mean())
                values[f'{prefix}_stress_RMSE_MPa'] = float(np.sqrt(np.mean(error[mask] ** 2)))
        return values
    error = result['strain'] - truth['strain']
    full = error[1:]
    values.update(full_history_strain_MAE=float(np.abs(full).mean()),
                  full_history_strain_RMSE=float(np.sqrt(np.mean(full ** 2))),
                  full_history_epsilon11_MAE=float(np.abs(full[:, 0]).mean()),
                  full_history_epsilon11_RMSE=float(np.sqrt(np.mean(full[:, 0] ** 2))))
    for i, component in enumerate(COMPONENTS):
        values[f'strain_MAE_{component}'] = float(np.abs(full[:, i]).mean())
        values[f'strain_RMSE_{component}'] = float(np.sqrt(np.mean(full[:, i] ** 2)))
    compression = np.maximum(0., -result['strain'][:, 0])
    reference = np.maximum(0., -truth['strain'][:, 0])
    probe = _window(t, item, 'probe')
    peak_mask = np.ones(len(t), bool) if probe is None else probe
    peak, refpeak = float(compression[peak_mask].max()), float(reference[peak_mask].max())
    values.update(peak_compression=peak, reference_peak_compression=refpeak,
                  peak_compression_error=peak - refpeak,
                  absolute_peak_compression_error=abs(peak - refpeak),
                  relative_peak_compression_error=abs(peak - refpeak) / refpeak if abs(refpeak) > tolerance else None)
    recovery = _window(t, item, 'recovery')
    for prefix, mask in (('recovery', recovery), ('reload', probe)):
        if mask is not None:
            values[f'{prefix}_strain_MAE'] = float(np.abs(error[mask]).mean())
            values[f'{prefix}_strain_RMSE'] = float(np.sqrt(np.mean(error[mask] ** 2)))
            values[f'{prefix}_epsilon11_MAE'] = float(np.abs(error[mask, 0]).mean())
    for prefix, mask in (('recovery', recovery), ('probe', probe)):
        if mask is not None:
            predicted = float(np.trapezoid(compression[mask], t[mask]))
            target = float(np.trapezoid(reference[mask], t[mask]))
            values[f'{prefix}_integrated_compression_s'] = predicted
            values[f'reference_{prefix}_integrated_compression_s'] = target
            values[f'{prefix}_integrated_compression_error_s'] = predicted - target
            values[f'absolute_{prefix}_integrated_compression_error_s'] = abs(predicted - target)
    if recovery is not None:
        epsilon = float(result['strain'][recovery, 0][-1])
        reference_epsilon = float(truth['strain'][recovery, 0][-1])
        values.update(free_recovery_residual_epsilon11=epsilon,
                      reference_free_recovery_residual_epsilon11=reference_epsilon,
                      free_recovery_residual_epsilon11_error=epsilon - reference_epsilon,
                      absolute_free_recovery_residual_epsilon11_error=abs(epsilon - reference_epsilon),
                      recovery_epsilon11_absolute_error_integral_s=float(
                          np.trapezoid(np.abs(error[recovery, 0]), t[recovery])))
    return values


def _worker_environment():
    # One CPU per worker; retain the original Newton/material code.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    affinity = sorted(os.sched_getaffinity(0))
    worker = multiprocessing.current_process()._identity[0] - 1
    os.sched_setaffinity(0, {affinity[worker % len(affinity)]})
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')


def _evaluate_one(arguments):
    family, *run_arguments = arguments
    return evaluate_catalog(*run_arguments, summarize=False, _family=family)


def evaluate_catalog(models, catalog_path, numerics, output_dir, compact_dir, reuse=None,
                     tolerance=1e-6, workers=1, summarize=True, _family=None):
    """Run every supplied model/history cell with its independently evolved state.

    ``reuse`` maps (model_id, history_id) to {prediction_file, failure}; these
    are exact old predictions selected by the caller, never nearest matches.
    Existing per-cell JSON records are actual resumable completed executions.
    """
    begin = time.monotonic()
    catalog_path, output_dir, compact_dir = map(Path, (catalog_path, output_dir, compact_dir))
    catalog_definition = json.loads(catalog_path.read_text(encoding='utf-8'))
    catalog = catalog_definition['histories']
    if _family is not None:
        catalog = [item for item in catalog if item['family'] == _family]
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_dir.mkdir(parents=True, exist_ok=True)
    reuse = {} if reuse is None else reuse
    records, executed, resumed, reused = [], 0, 0, 0
    if workers > 1:
        # Spawn avoids inheriting an initialized JAX runtime. Each worker owns
        # one model/family task containing intact histories. Families share a
        # model directory but have disjoint filenames; summaries run once here.
        families = sorted({item['family'] for item in catalog})
        arguments = [(family, [model], catalog_path, numerics, output_dir, compact_dir,
                      {k: v for k, v in reuse.items() if k[0] == model['model_id']}, tolerance)
                     for model in models for family in families]
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_environment) as pool:
            for result in pool.map(_evaluate_one, arguments, chunksize=1):
                records.extend(result['records'])
                executed += result['executed']
                resumed += result['resumed']
                reused += result['reused']
    for model in ([] if workers > 1 else models):
        identity = {k: model[k] for k in IDENTITY}
        weight_path = Path(model['weights'])
        weights, norm = load_weights(weight_path)
        weight_hash = _hash(weight_path)
        modeldir = output_dir / model['model_id']
        modeldir.mkdir(exist_ok=True)
        for item in catalog:
            history_id = item['history_id']
            recordfile = modeldir / (history_id + '.json')
            if recordfile.exists():
                records.append(json.loads(recordfile.read_text(encoding='utf-8')))
                resumed += 1
                continue
            truth = _npz(catalog_path.parent / item['arrays'])
            t = truth['time']
            stress_control = item['mode'] == 'stress_control'
            command = truth['prescribed_stress'] if stress_control else truth['strain']
            key = model['model_id'], history_id
            integration_begin = time.monotonic()
            if key in reuse:
                saved = reuse[key]
                reused_source = Path(saved['prediction_file'])
                source = modeldir / (history_id + '.npz')
                shutil.copy2(reused_source, source)
                result, failure = _npz(source), saved['failure']
                reused += 1
            else:
                result, failure = integrate(weights, norm, t, command, numerics, stress_control=stress_control)
                source = modeldir / (history_id + '.npz')
                np.savez_compressed(source, **result)
                executed += 1
            integration_elapsed = time.monotonic() - integration_begin if key not in reuse else 0.
            complete = failure is None
            finite = np.isfinite(result['strain']).all(axis=-1) & np.isfinite(result['stress']).all(axis=-1)
            last = int(np.flatnonzero(finite)[-1])
            parameter_keys = [k for k in item if k not in
                              ('history_id', 'condition_id', 'source_history_id', 'family', 'mode',
                               'cluster_id', 'candidate', 'arrays', 'parameters')
                              and not k.endswith(('_start_s', '_end_s', 'sha256'))]
            parameters = dict(item.get('parameters', {}), **{k: item[k] for k in parameter_keys})
            record = dict(identity, history_id=history_id, family=item['family'],
                          condition_id=item['condition_id'], cluster_id=item.get('cluster_id'),
                          candidate=item.get('candidate'), mode=item['mode'],
                          parameters=parameters,
                          weights_sha256=weight_hash, integration_complete=complete,
                          failure=failure, completed_increments=int(finite[1:].sum()),
                          requested_increments=len(t) - 1, last_complete_time_s=float(t[last]),
                          integration_elapsed_s=integration_elapsed,
                          actual_endpoint_s=float(t[-1]),
                          max_residual_MPa=float(np.nanmax(result['residual'])),
                          max_Newton_iterations=int(np.max(result['newton_iterations'])),
                          prediction_file=str(source.resolve()), prediction_sha256=_hash(source),
                          prediction_relative_path=source.relative_to(output_dir).as_posix(),
                          reused_existing_prediction=key in reuse,
                          reused_prediction_source=str(Path(reuse[key]['prediction_file']).resolve()) if key in reuse else None,
                          reference_file=str((catalog_path.parent / item['arrays']).resolve()),
                          reference_relative_path=Path(os.path.relpath(
                              catalog_path.parent / item['arrays'], output_dir)).as_posix())
            record.update({f'parameter_{k}': v for k, v in parameters.items()})
            record.update({k: v for k, v in item.items() if k.endswith(('_start_s', '_end_s'))})
            record.update(response_metrics(result, truth, item, complete, tolerance))
            write_json(recordfile, record)
            records.append(record)
        print(json.dumps(dict(catalog=catalog_path.name, model_id=model['model_id'], family=_family,
                              recorded_histories=len(catalog),
                              failures=sum(not r['integration_complete'] for r in records
                                           if r['model_id'] == model['model_id']))), flush=True)
    if not summarize:
        return dict(records=records, executed=executed, resumed=resumed, reused=reused)
    _csv(output_dir / 'complete-history-metrics.csv', records)
    primary = (*IDENTITY, 'family', 'history_id', 'condition_id', 'cluster_id', 'candidate',
               'mode', 'parameters', 'integration_complete', 'failure', 'last_complete_time_s',
               'stress_MAE_MPa', 'stress_RMSE_MPa', 'full_history_strain_MAE',
               'full_history_strain_RMSE', 'absolute_peak_compression_error',
               'recovery_strain_RMSE', 'reload_strain_RMSE',
               'absolute_free_recovery_residual_epsilon11_error',
               'absolute_recovery_integrated_compression_error_s')
    _csv(compact_dir / 'primary-history-metrics.csv', [{key: row[key] for key in primary} for row in records])
    summaries = family_summaries(records)
    paired = paired_arm_changes(records)
    _csv(compact_dir / 'family-model-summary.csv', summaries)
    _csv(output_dir / 'paired-arm-history-changes.csv', paired)
    _csv(compact_dir / 'paired-arm-family-summary.csv', paired_family_summaries(paired))
    pairs = candidate_pairs(records, tolerance, output_dir)
    _csv(compact_dir / 'candidate-pairs.csv', pairs)
    _csv(compact_dir / 'candidate-summary.csv', candidate_summaries(pairs))
    pair_changes = paired_arm_changes(pairs, PAIR_METRICS, key='condition_id')
    _csv(compact_dir / 'paired-arm-candidate-changes.csv', pair_changes)
    _csv(compact_dir / 'paired-arm-candidate-summary.csv', paired_family_summaries(pair_changes))
    summary = dict(model_count=len(models),
                   physical_conditions=len({r.get('physical_condition_id', r['condition_id']) for r in catalog}),
                   complete_reference_histories=len({r['arrays'] for r in catalog}),
                   boundary_replays_per_model=len(catalog), model_history_cells=len(records),
                   successful_histories=sum(r['integration_complete'] for r in records),
                   failures=[r for r in records if not r['integration_complete']],
                   newly_executed_histories=executed, reused_old_histories=reused,
                   reused_old_histories_total=sum(r['reused_existing_prediction'] for r in records),
                   accumulated_integration_process_seconds=sum(r['integration_elapsed_s'] for r in records),
                   resumed_histories=resumed, elapsed_s=time.monotonic() - begin,
                   catalog_sha256=_hash(catalog_path), catalog=str(catalog_path.resolve()),
                   metrics_file=str((output_dir / 'complete-history-metrics.csv').resolve()),
                   family_model_summary=summaries,
                   statistical_units='Fixed physical conditions; F2/F5 additionally summarized by five base/prefix seeds. '
                                     'Models, noise realizations, candidates and time samples are not independent conditions.',
                   mean_definition='Equal complete-history weighting within each family/model; F6 pairs remain single physical conditions. '
                                   'No pooled six-family headline mean. Incomplete full-family means are null.',
                   tie_tolerance_absolute_strain=tolerance,
                   tie_choice='Explicit equal-probability choice of early and late; expected and worst reference regret.',
                   near_tie_rule='Reference gaps at or below tolerance are decayed-memory near ties, not strong ordering evidence.')
    if catalog_definition.get('known_diagnostics'):
        summary.update(known_original_histories=32, original_random_histories=20,
                       original_revisit_histories=12, original_revisit_candidate_pairs=6,
                       physical_conditions=26,
                       physical_condition_note='Twenty old random paths plus six old same-prefix candidate pairs; '
                                               'the two boundary replays do not create new physical conditions.')
    write_json(output_dir / 'evaluation.json', summary)
    return summary


def family_summaries(records):
    rows = []
    groups = sorted({(r['family'], r['model_id']) for r in records})
    for family, model_id in groups:
        group = [r for r in records if r['family'] == family and r['model_id'] == model_id]
        full = all(r['integration_complete'] for r in group)
        row = {k: group[0][k] for k in IDENTITY}
        row.update(family=family, physical_conditions=len({r['condition_id'] for r in group}),
                   histories=len(group), successful_histories=sum(r['integration_complete'] for r in group),
                   all_histories_complete=full)
        for metric in METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            row[f'mean_{metric}'] = float(np.mean(values)) if full and values else None
            row[f'completed_median_{metric}'] = float(np.median(values)) if values else None
            row[f'completed_p90_{metric}'] = float(np.quantile(values, .9)) if values else None
            row[f'completed_worst_{metric}'] = float(np.max(values)) if values else None
        rows.append(row)
    return rows


def paired_arm_changes(records, metrics=METRICS, key='history_id'):
    """A minus B/C: positive values mean a smaller metric in the latter arm.

    The caller normally supplies twelve selected models. For the 60-model
    known diagnostic, initialization is additionally matched across arms.
    """
    records = [r for r in records if r['arm'] != 'always_early']
    counts = {}
    for r in records:
        counts.setdefault((r['arm'], r['cell']), set()).add(r['seed'])
    by_initialization = any(len(seeds) > 1 for seeds in counts.values())
    def item_key(row):
        return row['family'], row['cell'], row[key], row['seed'] if by_initialization else None
    indexed = {(r['arm'], *item_key(r)): r for r in records}
    rows = []
    for first in records:
        if first['arm'] != 'A_random100':
            continue
        for arm in ('B_random1000', 'C_mixed100'):
            other = indexed[(arm, *item_key(first))]
            row = dict(family=first['family'], cell=first['cell'], condition=first['condition'],
                       noise_realization=first['noise_realization'], comparison='A_minus_' + arm[0],
                       history_id=first.get('history_id'), condition_id=first['condition_id'],
                       cluster_id=first.get('cluster_id'), candidate=first.get('candidate'),
                       A_seed=first['seed'], other_seed=other['seed'],
                       A_complete=first['integration_complete'], other_complete=other['integration_complete'],
                       paired_complete=first['integration_complete'] and other['integration_complete'])
            for metric in metrics:
                a, b = first[metric], other[metric]
                row['A_' + metric], row['other_' + metric] = a, b
                row['A_minus_other_' + metric] = a - b if a is not None and b is not None else None
            rows.append(row)
    return rows


def paired_family_summaries(rows):
    summary = []
    if not rows:
        return summary
    metrics = [k for k in rows[0] if k.startswith('A_minus_other_')]
    groups = sorted({(r['family'], r['cell'], r['comparison']) for r in rows})
    for family, cell, comparison in groups:
        group = [r for r in rows if (r['family'], r['cell'], r['comparison']) == (family, cell, comparison)]
        # Complete-history averages are nested within condition first, then
        # base-seed cluster; two candidates never become two sampling units.
        clustered = family.startswith(('F2_', 'F5_'))
        units = sorted({str(r['cluster_id']) if clustered else r['condition_id'] for r in group})
        for metric in metrics:
            usable = [r for r in group if r[metric] is not None]
            if not usable:
                continue
            unit_values = []
            for unit in units:
                members = [r for r in group if (str(r['cluster_id']) if clustered else r['condition_id']) == unit]
                values = [r[metric] for r in members]
                unit_values.append(float(np.mean(values)) if all(x is not None for x in values) else None)
            finite = [v for v in unit_values if v is not None]
            all_complete = len(finite) == len(unit_values)
            summary.append(dict(family=family, cell=cell, comparison=comparison,
                                metric=metric.removeprefix('A_minus_other_'),
                                averaging_unit='base_seed_cluster' if clustered else 'physical_condition',
                                prescribed_units=len(units), complete_units=len(finite),
                                complete_model_history_pairs=sum(r['paired_complete'] for r in group),
                                A_minus_other_mean=float(np.mean(finite)) if all_complete else None,
                                completed_unit_median=float(np.median(finite)) if finite else None,
                                completed_unit_min=float(np.min(finite)) if finite else None,
                                completed_unit_max=float(np.max(finite)) if finite else None,
                                units_lower_metric_in_other=sum(v > 0 for v in finite),
                                units_higher_metric_in_other=sum(v < 0 for v in finite)))
    return summary


def candidate_pairs(records, tolerance, prediction_root):
    """F6 and known stress-revisit pairs, scored on common-probe compression."""
    def rank(delta):
        return 'tie' if abs(delta) <= tolerance else ('early' if delta < 0 else 'late')
    candidates = [r for r in records if r['candidate'] and r['mode'] == 'stress_control']
    groups = sorted({(r['family'], r['condition_id'], r['model_id']) for r in candidates})
    rows = []
    for family, condition_id, model_id in groups:
        group = {r['candidate']: r for r in candidates
                 if (r['family'], r['condition_id'], r['model_id']) == (family, condition_id, model_id)}
        early, late = group['early'], group['late']
        row = {k: early[k] for k in IDENTITY}
        row.update(family=family, condition_id=condition_id, cluster_id=early['cluster_id'],
                   integration_complete=early['integration_complete'] and late['integration_complete'],
                   reference_early=None, reference_late=None, reference_center=None,
                   reference_early_minus_late=None, reference_rank=None, reference_near_tie=None,
                   predicted_early=None, predicted_late=None, predicted_center=None,
                   predicted_early_minus_late=None, predicted_rank='integration_failure',
                   response_ratio=None, rank_matches=None, strong_ordering_match=None,
                   choice_probability_early=None, expected_reference_outcome=None,
                   **{metric: None for metric in PAIR_METRICS})
        # Reference outcomes remain available even for failed learned paths.
        outcomes = []
        for source in (early, late):
            truth = _npz(Path(prediction_root) / source['reference_relative_path'])
            # The original reference file stores phase=3 for the probe. New
            # catalogs include a dedicated probe_start_s in per-cell records.
            probe_start = source['probe_start_s']
            mask = truth['time'] >= probe_start - 1e-10
            outcomes.append(float(np.maximum(0., -truth['strain'][mask, 0]).max()))
        re, rl = outcomes
        delta, center, oracle = re - rl, (re + rl) / 2, min(re, rl)
        row.update(reference_early=re, reference_late=rl, reference_center=center,
                   reference_early_minus_late=delta, reference_rank=rank(delta),
                   reference_near_tie=abs(delta) <= tolerance)
        if row['integration_complete']:
            ye, yl = early['peak_compression'], late['peak_compression']
            choice = rank(ye - yl)
            probability = {'early': 1., 'late': 0., 'tie': .5}[choice]
            outcome = probability * re + (1 - probability) * rl
            row.update(predicted_early=ye, predicted_late=yl, predicted_center=(ye + yl) / 2,
                       candidate_center_bias=(ye + yl) / 2 - center,
                       absolute_candidate_center_bias=abs((ye + yl) / 2 - center),
                       predicted_early_minus_late=ye - yl, action_difference_error=ye - yl - delta,
                       absolute_action_difference_error=abs(ye - yl - delta),
                       absolute_response_error=(abs(ye - re) + abs(yl - rl)) / 2,
                       response_ratio=(ye - yl) / delta if abs(delta) > tolerance else None,
                       predicted_rank=choice, rank_matches=choice == rank(delta),
                       strong_ordering_match=choice == rank(delta) if abs(delta) > tolerance else None,
                       choice_probability_early=probability, expected_reference_outcome=outcome,
                       expected_regret=outcome - oracle,
                       worst_case_regret=(max(re, rl) if choice == 'tie' else outcome) - oracle)
        rows.append(row)
    # A single constant-policy row per physical pair, independent of model or seed.
    for family, condition_id in sorted({(r['family'], r['condition_id']) for r in rows}):
        reference = next(r for r in rows if (r['family'], r['condition_id']) == (family, condition_id))
        row = dict(reference)
        row.update(model_id='always_early', arm='always_early', cell='always_early',
                   condition='constant_policy', noise_realization=None, seed=None,
                   integration_complete=True, predicted_rank='early', choice_probability_early=1.,
                   rank_matches=reference['reference_rank'] == 'early',
                   strong_ordering_match=reference['reference_rank'] == 'early' if not reference['reference_near_tie'] else None,
                   expected_reference_outcome=reference['reference_early'],
                   **{metric: None for metric in PAIR_METRICS})
        for key in ('predicted_early', 'predicted_late', 'predicted_center', 'predicted_early_minus_late', 'response_ratio'):
            row[key] = None
        row['expected_regret'] = row['worst_case_regret'] = reference['reference_early'] - min(reference['reference_early'], reference['reference_late'])
        rows.append(row)
    return rows


def candidate_summaries(pairs):
    rows = []
    for family, model_id in sorted({(r['family'], r['model_id']) for r in pairs}):
        group = [r for r in pairs if (r['family'], r['model_id']) == (family, model_id)]
        row = {k: group[0][k] for k in IDENTITY}
        row.update(family=family, physical_pairs=len(group),
                   successful_pairs=sum(r['integration_complete'] for r in group),
                   reference_near_ties=sum(r['reference_near_tie'] for r in group),
                   reference_strict_early=sum(r['reference_rank'] == 'early' for r in group),
                   reference_strict_late=sum(r['reference_rank'] == 'late' for r in group),
                   predicted_ties=sum(r['predicted_rank'] == 'tie' for r in group),
                   strong_ordering_matches=sum(r['strong_ordering_match'] is True for r in group),
                   strong_ordering_conditions=sum(not r['reference_near_tie'] for r in group))
        for metric in PAIR_METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            row['mean_' + metric] = float(np.mean(values)) if len(values) == len(group) else None
            row['completed_median_' + metric] = float(np.median(values)) if values else None
            row['completed_max_' + metric] = float(np.max(values)) if values else None
        rows.append(row)
    return rows


def prepare_known_catalog(base_out, frozen_revisit, destination, protocol):
    """Copy the actual 20 old paths and point both replays at the same 12 arrays."""
    base_out, frozen_revisit, destination = map(Path, (base_out, frozen_revisit, destination))
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(base_out / 'observations.npz') as obs, np.load(base_out / 'reference.npz') as ref:
        histories = []
        for i, t in enumerate(obs['evaluation_time']):
            hid = f'known-random-{i:02d}'
            file = destination / (hid + '.npz')
            np.savez_compressed(file, time=t, strain=obs['evaluation_strain'][i], stress=ref['evaluation_stress'][i])
            histories.append(dict(history_id=hid, family='D1_known_random', condition_id=hid,
                                  cluster_id=i, mode='strain_control', arrays=file.name, old_path_id=i))
    for preload in protocol['preloading_histories']:
        for delay in protocol['recovery_delays_s']:
            for candidate in protocol['current_action_candidates']:
                hid = f"{preload['id']}-delay-{delay:g}-{candidate['id']}"
                source = destination / (hid + '-reference.npz')
                shutil.copy2(frozen_revisit / (hid + '-reference.npz'), source)
                recovery = preload['duration_s'] + candidate['duration_s']
                for mode, suffix, family in (('strain_control', 'strain', 'D2_known_revisit_strain'),
                                             ('stress_control', 'stress', 'D3_known_revisit_stress')):
                    histories.append(dict(history_id=hid + '-' + suffix, source_history_id=hid,
                                          family=family, condition_id=f"{preload['id']}-delay-{delay:g}-{suffix}",
                                          physical_condition_id=f"{preload['id']}-delay-{delay:g}",
                                          cluster_id=None, candidate=candidate['id'], mode=mode,
                                          arrays=source.name, recovery_start_s=recovery,
                                          recovery_end_s=recovery + delay, probe_start_s=recovery + delay,
                                          parameters=dict(preload=preload['id'], delay_s=delay)))
    path = destination / 'catalog.json'
    write_json(path, dict(histories=histories, known_diagnostics=True,
                         known_original_histories=32, original_random_histories=20,
                         original_revisit_histories=12, original_revisit_candidate_pairs=6,
                         physical_conditions=26, boundary_replays_per_model=44))
    return path


def existing_known_predictions(models, catalog_path, base_out, second_revisit):
    """Reuse the ten archived A models' 200 random responses and ideal0 revisits."""
    catalog = json.loads(Path(catalog_path).read_text(encoding='utf-8'))['histories']
    base_out, second_revisit = Path(base_out), Path(second_revisit)
    selection = json.loads((base_out / 'training.json').read_text(encoding='utf-8'))['selection']
    reused = {}
    for model in models:
        if model['arm'] != 'A_random100' or model['cell'] not in ('ideal', 'noise0'):
            continue
        for item in catalog:
            condition, seed = model['condition'], model['seed']
            if item['family'] == 'D1_known_random':
                stem = f"new-{condition}-seed-{seed:02d}-heldout-{item['old_path_id']:02d}"
                npz = base_out / 'predictions' / (stem + '.npz')
                record = json.loads(npz.with_suffix('.json').read_text(encoding='utf-8'))
            elif item['family'] == 'D3_known_revisit_stress' and seed == selection[condition]['seed']:
                npz = second_revisit / (item['source_history_id'] + '-new_' + condition + '.npz')
                record = json.loads(npz.with_suffix('.json').read_text(encoding='utf-8'))
            else:
                continue
            reused[model['model_id'], item['history_id']] = dict(prediction_file=str(npz), failure=record['failure'])
    return reused


def local_recovery_quantities(models, reference_parameters, output_dir, compact_dir):
    """Zero-state observable poles, derived from physical tensor derivatives.

    sigma = Ceq e + Cov(e-q), V qdot = Cov(e-q).
    Under sigma=0, e=Lq with L=(Ceq+Cov)^-1 Cov and
    qdot=-Dq, D=V^-1 Cov(I-L). The strain-observable decay poles are
    eigenvalues of D; their residues are mapped by L. Isotropic scalar times
    eta*(M_eq+M_ov)/(M_eq*M_ov) are reported for M=K,G only at zero state.
    """
    output_dir, compact_dir = Path(output_dir), Path(compact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_dir.mkdir(parents=True, exist_ok=True)
    zero = jnp.zeros(6, dtype=jnp.float64)
    rows, matrices = [], {}
    reference = Reference(reference_parameters)
    entries = [(dict(model_id='reference', arm='reference', cell='reference', condition='reference',
                     noise_realization=None, seed=None), reference.ceq, reference.cov, reference.v0)]
    for model in models:
        w, norm = load_weights(model['weights'])
        total = np.asarray(jax.jacfwd(lambda e: forces(w, norm, e, zero, zero)[0])(zero))
        cov = -np.asarray(jax.jacfwd(lambda q: forces(w, norm, zero, q, zero)[0])(zero))
        viscosity = np.asarray(jax.jacfwd(lambda rate: forces(w, norm, zero, zero, rate)[2])(zero))
        entries.append((model, total - cov, cov, viscosity))
    for model, ceq, cov, viscosity in entries:
        response = np.linalg.solve(ceq + cov, cov)
        decay = np.linalg.solve(viscosity, cov @ (np.eye(6) - response))
        poles, modes = np.linalg.eig(decay)
        observable_modes = response @ modes
        row = {k: model[k] for k in IDENTITY}
        for prefix, matrix, suffix in (('eq', ceq, 'MPa'), ('ov', cov, 'MPa'), ('viscosity', viscosity, 'MPa_s')):
            row[f'K_{prefix}_{suffix}'] = float(matrix[:3, :3].sum() / 9)
            row[f'G_{prefix}_{suffix}'] = float(np.mean(np.diag(matrix)[3:]) / 2)
        for mode, letter in (('volumetric', 'K'), ('deviatoric', 'G')):
            eq, ov, eta = row[f'{letter}_eq_MPa'], row[f'{letter}_ov_MPa'], row[f'{letter}_viscosity_MPa_s']
            row[f'free_recovery_{mode}_time_s'] = eta * (eq + ov) / (eq * ov)
        row.update(free_recovery_pole_real_per_s=poles.real.tolist(),
                   free_recovery_pole_imag_per_s=poles.imag.tolist(),
                   strain_mode_amplitudes=np.linalg.norm(observable_modes, axis=0).tolist())
        rows.append(row)
        prefix = model['model_id'] + '/'
        matrices.update({prefix + key: value for key, value in
                         dict(Ceq=ceq, Cov=cov, viscosity=viscosity, zero_stress_strain_map=response,
                              free_recovery_decay_matrix=decay, poles=poles, strain_modes=observable_modes).items()})
    np.savez_compressed(output_dir / 'local-zero-state-matrices.npz', **matrices)
    _csv(compact_dir / 'local-zero-state-recovery.csv', rows)
    write_json(output_dir / 'local-zero-state-recovery.json', dict(
        records=rows,
        definition='Ceq and Cov are stress derivatives of physical energy gradients; V is the physical '
                   'dissipative-force derivative with respect to rate, all at e=q=qdot=0. '
                   'The six-vector uses physical shear components; gradient shear entries are divided by two.',
        derivation='sigma=Ceq*e+Cov*(e-q)=0 gives e=L*q, L=solve(Ceq+Cov,Cov); '
                   'V*qdot=Cov*(e-q) gives qdot=-D*q, D=solve(V,Cov*(I-L)). '
                   'For isotropic K/G, tau=eta*(M_eq+M_ov)/(M_eq*M_ov).',
        scope='Local zero-state observable free-recovery quantities only; nonlinear whole histories still use '
              'the unchanged learned potentials and reference material. Internal q coordinates are not labels or compared as identified truth.'))
    return rows


def _study_models(out, selected):
    """The persisted final-loss selection is the only source of main models."""
    out = Path(out)
    training = json.loads((out / 'training.json').read_text(encoding='utf-8'))
    records = training['selection'].values() if selected else training['runs']
    models = []
    for row in records:
        model_id = row['model_id'] if selected else f"{row['arm']}--{row['cell']}--seed-{row['seed']:02d}"
        models.append(dict(row, model_id=model_id, weights=str((out / row['weights']).resolve())))
    return models


def evaluate_study(study, base_config, out, base_out, root, workers=8):
    """Execute the frozen 700-history catalog for the twelve final-loss models."""
    out, root = Path(out), Path(root)
    compact = root / study['outputs']['compact_reproduction_root'] / out.name / 'final'
    return evaluate_catalog(_study_models(out, selected=True), out / 'final_reference' / 'catalog.json',
                            base_config['numerics'], out / 'final_predictions', compact,
                            tolerance=study['final_evaluation']['metrics']['ranking_tolerance_absolute_strain'],
                            workers=workers)


def diagnose_study(study, base_config, out, base_out, root, workers=8):
    """All sixty models on old known histories, including exact output reuse."""
    out, base_out, root = map(Path, (out, base_out, root))
    protocol = json.loads((root / 'experiments/tm02/r2-revisit.json').read_text(encoding='utf-8'))
    transfer_root = root / protocol['outputs']['large_root']
    frozen = transfer_root / base_config['comparison']['frozen_run']
    second = transfer_root / study['base']['run']
    destination = out / 'diagnostics'
    compact = root / study['outputs']['compact_reproduction_root'] / out.name / 'diagnostics'
    models = _study_models(out, selected=False)
    catalog = prepare_known_catalog(base_out, frozen, destination / 'reference', protocol)
    reuse = existing_known_predictions(models, catalog, base_out, second)
    result = evaluate_catalog(models, catalog, base_config['numerics'], destination / 'predictions', compact,
                              reuse=reuse,
                              tolerance=study['final_evaluation']['metrics']['ranking_tolerance_absolute_strain'],
                              workers=workers)
    result['local_zero_state_model_records'] = len(local_recovery_quantities(
        models, base_config['reference_material']['parameters'], destination, compact))
    write_json(destination / 'predictions' / 'evaluation.json', result)
    return result
