"""Paired free integration of frozen and multi-history TM02 material models.

The deployment call receives only time, imposed strain, learned weights, and
their saved normalization. Reference stress is opened after free integration.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .evaluate import integrate, write_csv, write_json
from .material import COMPONENTS, load_weights


INVARIANTS = ('strain_trace', 'strain_trace_square', 'strain_trace_fourth',
              'rate_trace', 'rate_trace_square', 'rate_trace_fourth')
ERROR_METRICS = ('MAE_MPa', 'RMSE_MPa', 'hydrostatic_MAE_MPa',
                 'hydrostatic_RMSE_MPa', 'deviatoric_component_MAE_MPa',
                 'deviatoric_component_RMSE_MPa')


def _invariants(e):
    tensor = np.zeros((*e.shape[:-1], 3, 3))
    tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2] = np.moveaxis(e[..., :3], -1, 0)
    tensor[..., 0, 1] = tensor[..., 1, 0] = e[..., 3]
    tensor[..., 0, 2] = tensor[..., 2, 0] = e[..., 4]
    tensor[..., 1, 2] = tensor[..., 2, 1] = e[..., 5]
    square = tensor @ tensor
    return np.stack((np.trace(tensor, axis1=-2, axis2=-1),
                     np.trace(square, axis1=-2, axis2=-1),
                     np.sum(square * square, axis=(-2, -1))), axis=-1)


def support_features(t, e):
    """Differences are always taken along time, never between histories."""
    rate = np.diff(e, axis=-2) / np.diff(t, axis=-1)[..., None]
    return np.concatenate((_invariants(e[..., 1:, :]), _invariants(rate)), axis=-1)


def training_support(t, e):
    features = support_features(t, e)
    # Include the prescribed virgin strain in the strain bounds, as in training.
    strain = _invariants(e).reshape(-1, 3)
    rate = features[..., 3:].reshape(-1, 3)
    lower = np.r_[strain.min(axis=0), rate.min(axis=0)]
    upper = np.r_[strain.max(axis=0), rate.max(axis=0)]
    return lower, upper


def _support_masks(features, bounds):
    inside = (features >= bounds[0]) & (features <= bounds[1])
    masks = {name: inside[:, i] for i, name in enumerate(INVARIANTS)}
    masks['observable_invariant_box'] = np.all(inside, axis=-1)
    return masks


def _metrics(prediction, target, complete, masks):
    error = prediction[1:] - target[1:]
    result = {name: None for name in ERROR_METRICS}
    for component in COMPONENTS:
        result.update({f'{metric}_{component}_MPa': None for metric in ('MAE', 'RMSE')})
    result['hydrostatic_fraction_of_component_squared_error'] = None
    for name, mask in masks.items():
        for side, selected in (('inside', mask), ('outside', ~mask)):
            result[f'{name}_{side}_increments'] = int(selected.sum())
            for metric in ERROR_METRICS:
                result[f'{name}_{side}_{metric}'] = None
    if not complete:
        return result
    hydro = error[:, :3].mean(axis=-1)
    deviatoric = error.copy()
    deviatoric[:, :3] -= hydro[:, None]

    def measures(indices):
        return dict(MAE_MPa=float(np.mean(np.abs(error[indices]))),
                    RMSE_MPa=float(np.sqrt(np.mean(error[indices] ** 2))),
                    hydrostatic_MAE_MPa=float(np.mean(np.abs(hydro[indices]))),
                    hydrostatic_RMSE_MPa=float(np.sqrt(np.mean(hydro[indices] ** 2))),
                    deviatoric_component_MAE_MPa=float(np.mean(np.abs(deviatoric[indices]))),
                    deviatoric_component_RMSE_MPa=float(np.sqrt(np.mean(deviatoric[indices] ** 2))))

    result.update(measures(np.ones(len(error), dtype=bool)))
    for k, component in enumerate(COMPONENTS):
        result[f'MAE_{component}_MPa'] = float(np.mean(np.abs(error[:, k])))
        result[f'RMSE_{component}_MPa'] = float(np.sqrt(np.mean(error[:, k] ** 2)))
    squared_error = float(np.sum(error ** 2))
    result['hydrostatic_fraction_of_component_squared_error'] = (
        float(3 * np.sum(hydro ** 2) / squared_error) if squared_error else 0.)
    for name, mask in masks.items():
        for side, selected in (('inside', mask), ('outside', ~mask)):
            if selected.any():
                result.update({f'{name}_{side}_{key}': value
                               for key, value in measures(selected).items()})
    return result


def _paths(config, out, root):
    frozen = root / config['outputs']['large_root'] / config['comparison']['frozen_run']
    return {'old': frozen, 'new': out}


def prediction_stem(cohort, condition, seed, role, path_id):
    return f'{cohort}-{condition}-seed-{seed:02d}-{role}-{path_id:02d}'


def _evaluate_models(config, out, root, cohort, role):
    directories = _paths(config, out, root)
    modeldir = directories[cohort]
    training = json.loads((modeldir / 'training.json').read_text())
    with np.load(modeldir / 'observations.npz') as observed:
        bounds = training_support(observed['train_time'], observed['train_strain'])
    observed_dir = out if role == 'heldout' else directories['old']
    with np.load(observed_dir / 'observations.npz') as observed:
        times, strains = observed['evaluation_time'], observed['evaluation_strain']
    if role == 'known':
        times, strains = times[None], strains[None]
        path_seeds = [config['comparison']['known_hard_path']['original_strain_seed']]
    else:
        path_seeds = config['data']['evaluation']['strain_seeds']
    features = support_features(times, strains)
    predictiondir = out / 'predictions'
    predictiondir.mkdir(exist_ok=True)
    records = []
    for row in training['runs']:
        condition, seed = row['condition'], row['seed']
        w, norm = load_weights(modeldir / row['weights'])
        weight_hash = hashlib.sha256((modeldir / row['weights']).read_bytes()).hexdigest()
        for path_id, (clock, strain) in enumerate(zip(times, strains)):
            stem = prediction_stem(cohort, condition, seed, role, path_id)
            jsonpath = predictiondir / f'{stem}.json'
            if jsonpath.exists():
                records.append(json.loads(jsonpath.read_text()))
                continue
            result, failure = integrate(w, norm, clock, strain, config['numerics'])
            np.savez_compressed(predictiondir / f'{stem}.npz', **result)
            # Scoring labels are accessed only after this complete free rollout.
            with np.load(observed_dir / 'reference.npz') as truth:
                target = truth['evaluation_stress']
                if role == 'heldout':
                    target = target[path_id]
            record = dict(cohort=cohort, condition=condition, seed=seed, role=role,
                          path_id=path_id, path_seed=path_seeds[path_id],
                          selected_by_final_training_loss=seed == training['selection'][condition]['seed'],
                          model_source_run=modeldir.name, weights=row['weights'],
                          weights_sha256=weight_hash,
                          iterations=row['iterations'],
                          final_training_loss_MPa=row['final_training_loss_MPa'],
                          stress_scale_MPa=float(norm['s_s']),
                          integration_complete=failure is None, failure=failure,
                          completed_increments=int(np.isfinite(result['stress'][1:]).all(axis=-1).sum()),
                          requested_increments=len(clock) - 1,
                          max_residual_MPa=float(np.nanmax(result['residual'])),
                          max_Newton_iterations=int(np.max(result['newton_iterations'])),
                          prediction_file=f'predictions/{stem}.npz')
            record.update(_metrics(result['stress'], target, failure is None,
                                   _support_masks(features[path_id], bounds)))
            write_json(jsonpath, record)
            records.append(record)
        print(json.dumps(dict(evaluation_cohort=cohort, condition=condition, seed=seed,
                              role=role, completed_model_path_records=len(records),
                              failures=sum(not r['integration_complete'] for r in records))), flush=True)
    return records


def evaluate_frozen(config, out, compact, root):
    """Run old final models while the new models train; no old optimization."""
    begin = time.monotonic()
    records = _evaluate_models(config, out, root, 'old', 'heldout')
    write_json(out / 'frozen-evaluation.json', dict(records=len(records),
               successful_predictions=sum(r['integration_complete'] for r in records),
               elapsed_s=time.monotonic() - begin))
    return records


def _flat_records(records):
    return [dict(row, failure=json.dumps(row['failure']) if row['failure'] else '')
            for row in records]


def _distribution(values):
    if not values:
        return dict(count=0, median=None, p90=None, worst=None)
    return dict(count=len(values), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), worst=float(np.max(values)))


def _improvement_distribution(values):
    if not values:
        return dict(count=0, median=None, p10=None, p90=None, smallest=None, largest=None)
    return dict(count=len(values), median=float(np.median(values)),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)),
                smallest=float(np.min(values)), largest=float(np.max(values)))


def _initialization_summary(records):
    rows = []
    for cohort in ('old', 'new'):
        for condition in ('ideal', 'noisy'):
            group = [r for r in records if r['cohort'] == cohort and r['condition'] == condition]
            for seed in sorted({r['seed'] for r in group}):
                paths = [r for r in group if r['seed'] == seed]
                complete = all(r['integration_complete'] for r in paths)
                row = dict(cohort=cohort, condition=condition, seed=seed,
                           selected_by_final_training_loss=paths[0]['selected_by_final_training_loss'],
                           physical_paths=len(paths),
                           successful_paths=sum(r['integration_complete'] for r in paths),
                           all_paths_complete=complete,
                           final_training_loss_MPa=paths[0]['final_training_loss_MPa'])
                for metric in ERROR_METRICS:
                    values = [r[metric] for r in paths if r['integration_complete']]
                    # Whole-ensemble summaries are missing when any path failed.
                    row[f'mean_path_{metric}'] = float(np.mean(values)) if complete else None
                    for statistic, value in _distribution(values).items():
                        row[f'completed_path_{metric}_{statistic}'] = value
                rows.append(row)
    return rows


def _paired_rows(records):
    rows = []
    for condition in ('ideal', 'noisy'):
        selected = {(r['cohort'], r['path_id']): r for r in records
                    if r['condition'] == condition and r['selected_by_final_training_loss']}
        ids = sorted({r['path_id'] for r in selected.values()})
        for path_id in ids:
            old, new = selected['old', path_id], selected['new', path_id]
            complete = old['integration_complete'] and new['integration_complete']
            row = dict(condition=condition, path_id=path_id, path_seed=old['path_seed'],
                       old_seed=old['seed'], new_seed=new['seed'],
                       old_complete=old['integration_complete'], new_complete=new['integration_complete'],
                       paired_complete=complete)
            metric_names = [*ERROR_METRICS, *[f'{m}_{c}_MPa' for c in COMPONENTS for m in ('MAE', 'RMSE')]]
            for metric in metric_names:
                row[f'old_{metric}'], row[f'new_{metric}'] = old[metric], new[metric]
                row[f'improvement_{metric}'] = old[metric] - new[metric] if complete else None
            rows.append(row)
    return rows


def _selected_summary(pairs):
    summary = {}
    for condition in ('ideal', 'noisy'):
        group = [r for r in pairs if r['condition'] == condition]
        successful = [r for r in group if r['paired_complete']]
        detail = dict(physical_paths=len(group), complete_pairs=len(successful),
                      failed_path_ids=[r['path_id'] for r in group if not r['paired_complete']],
                      old_selected_seed=group[0]['old_seed'], new_selected_seed=group[0]['new_seed'])
        for metric in ERROR_METRICS:
            detail[metric] = dict(
                old_completed_path_distribution=_distribution([r[f'old_{metric}'] for r in group if r['old_complete']]),
                new_completed_path_distribution=_distribution([r[f'new_{metric}'] for r in group if r['new_complete']]),
                paired_improvement_distribution=_improvement_distribution([r[f'improvement_{metric}'] for r in successful]),
                improved_paths=sum(r[f'improvement_{metric}'] > 0 for r in successful),
                worsened_paths=sum(r[f'improvement_{metric}'] < 0 for r in successful),
                equal_paths=sum(r[f'improvement_{metric}'] == 0 for r in successful),
                new_worst_path_id=max((r for r in group if r['new_complete']),
                                     key=lambda r: r[f'new_{metric}'])['path_id']
                                     if any(r['new_complete'] for r in group) else None)
        summary[condition] = detail
    return summary


def _ensemble_summary(initializations):
    summary = {}
    for cohort in ('old', 'new'):
        for condition in ('ideal', 'noisy'):
            group = [r for r in initializations if r['cohort'] == cohort and r['condition'] == condition]
            complete = [r for r in group if r['all_paths_complete']]
            summary[f'{cohort}_{condition}'] = dict(
                initializations=len(group), all_paths_complete=len(complete),
                failed_initialization_seeds=[r['seed'] for r in group if not r['all_paths_complete']],
                mean_path_error_distribution={metric: _distribution([r[f'mean_path_{metric}'] for r in complete])
                                              for metric in ERROR_METRICS})
    return summary


def _support_table(config, out, compact, root):
    directories = _paths(config, out, root)
    with np.load(out / 'observations.npz') as observed:
        heldout = support_features(observed['evaluation_time'], observed['evaluation_strain'])
    with np.load(directories['old'] / 'observations.npz') as observed:
        known = support_features(observed['evaluation_time'][None], observed['evaluation_strain'][None])
    rows, arrays = [], {}
    for cohort, directory in directories.items():
        with np.load(directory / 'observations.npz') as observed:
            bounds = training_support(observed['train_time'], observed['train_strain'])
        arrays[f'{cohort}_train_lower'], arrays[f'{cohort}_train_upper'] = bounds
        for role, features in (('heldout', heldout), ('known', known)):
            inside = (features >= bounds[0]) & (features <= bounds[1])
            arrays[f'{cohort}_{role}_inside'] = inside
            for path_id, values in enumerate(features):
                for k, name in enumerate(INVARIANTS):
                    rows.append(dict(cohort=cohort, role=role, path_id=path_id, invariant=name,
                                     training_min=float(bounds[0][k]), training_max=float(bounds[1][k]),
                                     evaluation_min=float(values[:, k].min()), evaluation_max=float(values[:, k].max()),
                                     inside_increments=int(inside[path_id, :, k].sum()),
                                     outside_increments=int((~inside[path_id, :, k]).sum()),
                                     total_increments=len(values)))
    arrays.update(heldout_features=heldout, known_features=known,
                  invariant_names=np.asarray(INVARIANTS))
    np.savez_compressed(out / 'training-support.npz', **arrays)
    write_csv(compact / 'training-support.csv', rows)


def _known_old_records(config, out, root):
    """Score the preserved first-round arrays; never integrate old known paths again."""
    frozen = _paths(config, out, root)['old']
    training = json.loads((frozen / 'training.json').read_text())
    with np.load(frozen / 'observations.npz') as observed:
        bounds = training_support(observed['train_time'], observed['train_strain'])
        features = support_features(observed['evaluation_time'], observed['evaluation_strain'])
    with np.load(frozen / 'reference.npz') as truth:
        target = truth['evaluation_stress']
    records = []
    for row in training['runs']:
        stem = f"{row['condition']}-seed-{row['seed']:02d}"
        original = json.loads((frozen / 'predictions' / f'{stem}.json').read_text())
        with np.load(frozen / 'predictions' / f'{stem}.npz') as predicted:
            values = predicted['stress']
        record = dict(cohort='old', condition=row['condition'], seed=row['seed'], role='known', path_id=0,
                      path_seed=config['comparison']['known_hard_path']['original_strain_seed'],
                      selected_by_final_training_loss=row['seed'] == training['selection'][row['condition']]['seed'],
                      model_source_run=frozen.name, weights=row['weights'],
                      iterations=row['iterations'], final_training_loss_MPa=row['final_training_loss_MPa'],
                      integration_complete=original['integration_complete'], failure=original['failure'],
                      reused_original_prediction=True,
                      prediction_file=str(frozen / 'predictions' / f'{stem}.npz'))
        record.update(_metrics(values, target, record['integration_complete'], _support_masks(features, bounds)))
        records.append(record)
    return records


def _selected_arrays(config, out, root, heldout, known):
    with np.load(out / 'observations.npz') as observed, np.load(out / 'reference.npz') as reference:
        arrays = dict(time=observed['evaluation_time'], strain=observed['evaluation_strain'],
                      reference_stress=reference['evaluation_stress'])
    for cohort in ('old', 'new'):
        for condition in ('ideal', 'noisy'):
            rows = sorted((r for r in heldout if r['cohort'] == cohort and r['condition'] == condition
                           and r['selected_by_final_training_loss']), key=lambda r: r['path_id'])
            arrays[f'{cohort}_{condition}_stress'] = np.stack([
                np.load(out / r['prediction_file'])['stress'] for r in rows])
            arrays[f'{cohort}_{condition}_seed'] = rows[0]['seed']
    np.savez_compressed(out / 'selected-heldout-full-responses.npz', **arrays)
    frozen = _paths(config, out, root)['old']
    with np.load(frozen / 'observations.npz') as observed, np.load(frozen / 'reference.npz') as reference:
        known_arrays = dict(time=observed['evaluation_time'], strain=observed['evaluation_strain'],
                            reference_stress=reference['evaluation_stress'])
    for row in known:
        if row['selected_by_final_training_loss']:
            filename = Path(row['prediction_file'])
            if row['cohort'] == 'new':
                filename = out / filename
            known_arrays[f"{row['cohort']}_{row['condition']}_stress"] = np.load(filename)['stress']
            known_arrays[f"{row['cohort']}_{row['condition']}_seed"] = row['seed']
    np.savez_compressed(out / 'selected-known-full-responses.npz', **known_arrays)


def evaluate(config, out, compact, root):
    from .multihistory_figures import figures
    begin = time.monotonic()
    compact.mkdir(parents=True, exist_ok=True)
    heldout = _evaluate_models(config, out, root, 'old', 'heldout')
    heldout += _evaluate_models(config, out, root, 'new', 'heldout')
    known_new = _evaluate_models(config, out, root, 'new', 'known')
    known = _known_old_records(config, out, root) + known_new
    write_csv(compact / 'all-heldout-model-path-errors.csv', _flat_records(heldout))
    # Reused old predictions have different provenance columns, all retained.
    fields = list(dict.fromkeys(key for row in known for key in row))
    write_csv(compact / 'known-diagnostic-errors.csv',
              [{key: r.get(key) for key in fields} for r in _flat_records(known)])
    initializations = _initialization_summary(heldout)
    write_csv(compact / 'all-initialization-summary.csv', initializations)
    pairs = _paired_rows(heldout)
    known_pairs = _paired_rows(known)
    write_csv(compact / 'selected-paired-path-errors.csv', pairs)
    write_csv(compact / 'known-diagnostic-selected-comparison.csv', known_pairs)
    _support_table(config, out, compact, root)
    _selected_arrays(config, out, root, heldout, known)
    training = json.loads((out / 'training.json').read_text())
    frozen_training = json.loads((_paths(config, out, root)['old'] / 'training.json').read_text())
    run = json.loads((out / 'run.json').read_text())
    index = dict(kind='TM02 v1 section 5.2.2 / Figure 14 multi-history extension reimplementation',
                 run=out.name, paper='https://arxiv.org/html/2401.14270v1',
                 condition_scope={'ideal': 'original-method multi-history extension',
                                  'noisy': 'project extension'},
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                 run_provenance=run, frozen_comparison=config['comparison'],
                 training_runs=len(training['runs']),
                 iterations_each=config['training']['iterations_per_initialization'],
                 total_optimizer_updates=training['total_optimizer_updates'],
                 selected=training['selection'], frozen_selected=frozen_training['selection'],
                 selection_rule=training['rule'],
                 independent_heldout_histories=config['data']['evaluation']['paths'],
                 increments_per_heldout_history=config['data']['evaluation']['increments'],
                 new_heldout_model_path_predictions=len(heldout),
                 successful_heldout_predictions=sum(r['integration_complete'] for r in heldout),
                 new_known_diagnostic_predictions=len(known_new),
                 successful_new_known_predictions=sum(r['integration_complete'] for r in known_new),
                 old_known_predictions_reused=len(known) - len(known_new),
                 selected_paired_summary=_selected_summary(pairs),
                 initialization_summary=_ensemble_summary(initializations),
                 known_diagnostic_selected_comparison=known_pairs,
                 failures=[{k: r[k] for k in ('cohort', 'condition', 'seed', 'role', 'path_id', 'failure')}
                           for r in heldout + known_new if not r['integration_complete']],
                 error_definition='MPa; equal weight for six independent physical tensor components; time zero excluded. Hydrostatic error is mean of three normal stress errors. Deviatoric error subtracts that mean from normal components; shear unchanged.',
                 normalization_comparison='No cross-model NRMSE comparison; every model uses its original saved normalization.',
                 support_definition='Marginal strain and within-history strain-rate invariant ranges, with conjunction recorded as an observable invariant box; this is not sufficient internal-state coverage.',
                 statistical_units='20 independent held-out loading histories; 25 initializations per condition/cohort. Predictions and time increments are not independent physical histories.',
                 deployment_inputs=['strain', 'time increments', 'own previous internal state'],
                 auxiliary_RNN_in_deployment=False, reference_internal_variables_in_deployment=False,
                 evaluation_stress_labels_in_deployment=False, numerics=config['numerics'],
                 evaluation_elapsed_s=time.monotonic() - begin,
                 training_elapsed_s=training['elapsed_s'], raw_artifacts=str(out),
                 prediction_arrays='predictions/*.npz',
                 selected_full_responses=['selected-heldout-full-responses.npz', 'selected-known-full-responses.npz'],
                 scope='Fixed-budget material evidence only; no robot-control benefit claim. Known hard history is a previously inspected diagnostic, not a new blind test.')
    write_json(compact / 'index.json', index)
    figures(config, out, compact, heldout, initializations, pairs, known, training)
    print('Paired multi-history predictions, physical-error tables and figures saved.', flush=True)
    return index
