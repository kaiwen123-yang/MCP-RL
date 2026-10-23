"""Twenty-four new material histories compared with the frozen 36 histories."""
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time

import numpy as np


MODELS = ('reference', 'old_ideal', 'old_noisy', 'new_ideal', 'new_noisy')
METRICS = (
    'peak_compression', 'integrated_compression_s',
    'peak_compression_error', 'absolute_peak_compression_error',
    'relative_peak_compression_error_percent',
    'integrated_compression_error_s', 'absolute_integrated_compression_error_s',
    'full_history_strain_MAE', 'full_history_strain_RMSE',
    'reload_strain_MAE', 'reload_strain_RMSE', 'reload_epsilon11_MAE',
    'recovery_strain_MAE', 'recovery_strain_RMSE', 'recovery_epsilon11_MAE',
    'recovery_end_compression', 'recovery_end_compression_error',
    'recovery_integrated_compression_s', 'recovery_integrated_compression_error_s',
    'max_residual_MPa',
)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_metrics(result, truth, recovery_start, probe_start, failure):
    """The probe includes its start boundary, as in the frozen protocol."""
    values = {key: None for key in METRICS}
    values['completed_increments'] = int(np.isfinite(result['strain'][1:, 0]).sum())
    if failure is not None:
        return values
    t = result['time']
    probe = t >= probe_start-1e-10
    recovery = (t >= recovery_start-1e-10) & (t <= probe_start+1e-10)
    error = result['strain']-truth['strain']
    compression = np.maximum(0, -result['strain'][:, 0])
    reference = np.maximum(0, -truth['strain'][:, 0])
    peak = float(compression[probe].max())
    true_peak = float(reference[probe].max())
    integral = float(np.trapezoid(compression[probe], t[probe]))
    true_integral = float(np.trapezoid(reference[probe], t[probe]))
    recovery_integral = float(np.trapezoid(compression[recovery], t[recovery]))
    reference_recovery_integral = float(np.trapezoid(reference[recovery], t[recovery]))
    values.update(
        peak_compression=peak, integrated_compression_s=integral,
        peak_compression_error=peak-true_peak,
        absolute_peak_compression_error=abs(peak-true_peak),
        relative_peak_compression_error_percent=100*abs(peak-true_peak)/true_peak,
        integrated_compression_error_s=integral-true_integral,
        absolute_integrated_compression_error_s=abs(integral-true_integral),
        full_history_strain_MAE=float(np.abs(error[1:]).mean()),
        full_history_strain_RMSE=float(np.sqrt(np.mean(error[1:]**2))),
        reload_strain_MAE=float(np.abs(error[probe]).mean()),
        reload_strain_RMSE=float(np.sqrt(np.mean(error[probe]**2))),
        reload_epsilon11_MAE=float(np.abs(error[probe, 0]).mean()),
        recovery_strain_MAE=float(np.abs(error[recovery]).mean()),
        recovery_strain_RMSE=float(np.sqrt(np.mean(error[recovery]**2))),
        recovery_epsilon11_MAE=float(np.abs(error[recovery, 0]).mean()),
        recovery_end_compression=float(compression[recovery][-1]),
        recovery_end_compression_error=float(compression[recovery][-1]-reference[recovery][-1]),
        recovery_integrated_compression_s=recovery_integral,
        recovery_integrated_compression_error_s=recovery_integral-reference_recovery_integral,
        max_residual_MPa=float(result['residual'].max()),
    )
    return values


def candidate_pairs(protocol, records):
    """Separate common amplitude bias, action response and decision consequence."""
    tolerance = protocol['metrics']['ranking_tie_tolerance_absolute_strain']

    def rank(difference):
        return 'tie' if abs(difference) <= tolerance else ('early' if difference < 0 else 'late')

    rows = []
    for preload in protocol['preloading_histories']:
        for delay in protocol['recovery_delays_s']:
            group = {(r['model'], r['candidate']): r for r in records
                     if r['preload'] == preload['id'] and r['delay_s'] == delay}
            reference_early = group['reference', 'early']['peak_compression']
            reference_late = group['reference', 'late']['peak_compression']
            refcenter = (reference_early+reference_late)/2
            refdelta = reference_early-reference_late
            refrank = rank(refdelta)
            oracle = min(reference_early, reference_late)
            for model in (*MODELS, 'always_early'):
                row = dict(
                    preload=preload['id'], delay_s=delay, model=model,
                    reference_early=reference_early, reference_late=reference_late,
                    reference_center=refcenter, reference_early_minus_late=refdelta,
                    reference_rank=refrank, predicted_early=None, predicted_late=None,
                    predicted_center=None, candidate_center_bias=None,
                    absolute_candidate_center_bias=None,
                    predicted_early_minus_late=None, action_difference_error=None,
                    absolute_action_difference_error=None,
                    response_ratio=None, predicted_rank=None, rank_matches=False,
                    choice_probability_early=None, expected_reference_outcome=None,
                    expected_regret=None, worst_case_regret=None,
                    integration_complete=True,
                )
                if model == 'always_early':
                    chosen, probability_early = 'early', 1.
                else:
                    early, late = group[model, 'early'], group[model, 'late']
                    if not (early['integration_complete'] and late['integration_complete']):
                        row.update(predicted_rank='integration_failure', integration_complete=False)
                        rows.append(row)
                        continue
                    ye, yl = early['peak_compression'], late['peak_compression']
                    center, delta = (ye+yl)/2, ye-yl
                    chosen = rank(delta)
                    probability_early = {'early': 1., 'late': 0., 'tie': .5}[chosen]
                    row.update(
                        predicted_early=ye, predicted_late=yl, predicted_center=center,
                        candidate_center_bias=center-refcenter,
                        absolute_candidate_center_bias=abs(center-refcenter),
                        predicted_early_minus_late=delta,
                        action_difference_error=delta-refdelta,
                        absolute_action_difference_error=abs(delta-refdelta),
                        response_ratio=delta/refdelta if abs(refdelta) > tolerance else None,
                    )
                outcome = probability_early*reference_early+(1-probability_early)*reference_late
                worst = max(reference_early, reference_late) if chosen == 'tie' else outcome
                row.update(
                    predicted_rank=chosen, rank_matches=chosen == refrank,
                    choice_probability_early=probability_early,
                    expected_reference_outcome=outcome,
                    expected_regret=outcome-oracle, worst_case_regret=worst-oracle,
                )
                rows.append(row)
    return rows


def _summaries(records, pairs):
    summaries = {}
    for model in (*MODELS[1:], 'always_early'):
        rp = [r for r in pairs if r['model'] == model]
        completed_pairs = [r for r in rp if r['integration_complete']]
        item = dict(
            complete_pairs=len(completed_pairs), total_pairs=len(rp),
            rank_matches=sum(r['rank_matches'] for r in rp),
            predicted_ties=sum(r['predicted_rank'] == 'tie' for r in rp),
            integration_failure_pairs=sum(not r['integration_complete'] for r in rp),
        )
        for metric in ('expected_regret', 'worst_case_regret',
                       'absolute_candidate_center_bias', 'action_difference_error',
                       'absolute_action_difference_error'):
            values = [r[metric] for r in completed_pairs if r[metric] is not None]
            item[metric] = (dict(mean=float(np.mean(values)), median=float(np.median(values)),
                                 max=float(np.max(values))) if values else None)
        complete = [r for r in records if r['model'] == model and r['integration_complete']]
        item['complete_histories'] = len(complete) if model != 'always_early' else None
        for metric in ('absolute_peak_compression_error', 'relative_peak_compression_error_percent',
                       'absolute_integrated_compression_error_s', 'reload_epsilon11_MAE',
                       'recovery_epsilon11_MAE', 'full_history_strain_RMSE'):
            values = [r[metric] for r in complete]
            item[metric] = (dict(mean=float(np.mean(values)), median=float(np.median(values)),
                                 max=float(np.max(values))) if values else None)
        summaries[model] = item
    return summaries


def revisit(config, rc, out, root, run):
    """Reuse exact frozen input/reference arrays; integrate only the two new models."""
    import jax
    from .evaluate import integrate
    from .material import load_weights

    begin = time.monotonic()
    frozen_run = config['comparison']['frozen_run']
    frozen = root/rc['outputs']['large_root']/frozen_run
    frozen_compact = root/rc['outputs']['compact_root']/frozen_run
    destination = root/rc['outputs']['large_root']/run
    compact = root/rc['outputs']['compact_root']/run
    training = json.loads((out/'training.json').read_text(encoding='utf-8'))
    frozen_index = json.loads((frozen_compact/'index.json').read_text(encoding='utf-8'))
    with (frozen_compact/'all_rollouts.csv').open(newline='', encoding='utf-8') as stream:
        frozen_records = {(r['history'], r['model']): r for r in csv.DictReader(stream)}
    destination.mkdir(parents=True, exist_ok=True)
    compact.mkdir(parents=True, exist_ok=True)
    models = {f'new_{condition}': load_weights(out/row['weights'])
              for condition, row in training['selection'].items()}
    source_files = [root/'scripts/run_tm02_r2.py', *sorted((root/'src/tm02').glob('*.py'))]
    metadata = dict(
        run=run, command=' '.join(sys.argv), protocol=rc,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_sha256={p.relative_to(root).as_posix(): _sha256(p) for p in source_files},
        config_sha256=_sha256(root/'experiments/tm02/r2-multihistory.json'),
        protocol_sha256=_sha256(root/config['revisit_comparison']['protocol']),
        host=platform.node(), python=sys.version, jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        frozen_run=frozen_run, frozen_archive=config['comparison']['release_url'],
        frozen_archive_sha256=config['comparison']['archive_sha256'],
        selected=training['selection'], frozen_selected=frozen_index['selected'],
        selected_weights_sha256={c: _sha256(out/r['weights']) for c, r in training['selection'].items()},
        numerics=config['numerics'], additional_training_runs=0,
        source_run=out.relative_to(root).as_posix(), raw_artifacts=destination.relative_to(root).as_posix(),
    )
    _json(destination/'run.json', metadata)
    records, rollouts, artifacts = [], {}, []
    new_executions, resumed_histories = 0, 0
    for preload in rc['preloading_histories']:
        for delay in rc['recovery_delays_s']:
            for candidate in rc['current_action_candidates']:
                hid = f"{preload['id']}-delay-{delay:g}-{candidate['id']}"
                with np.load(frozen/f'{hid}-reference.npz') as saved:
                    truth = {key: saved[key] for key in saved.files}
                # Commands, times and phase labels come from the frozen actual history.
                t, command, phase = truth['time'], truth['prescribed_stress'], truth['phase']
                recovery_start = preload['duration_s']+candidate['duration_s']
                probe_start = recovery_start+delay
                for model in MODELS:
                    if model in ('reference', 'old_ideal', 'old_noisy'):
                        old_name = model.removeprefix('old_')
                        artifact = frozen/f'{hid}-{old_name}.npz'
                        with np.load(artifact) as saved:
                            result = {key: saved[key] for key in saved.files}
                        old_record = frozen_records[hid, old_name]
                        failure = json.loads(old_record['failure']) if old_record['failure'] else None
                        reused = True
                    else:
                        artifact = destination/f'{hid}-{model}.npz'
                        record_file = destination/f'{hid}-{model}.json'
                        if record_file.exists():
                            previous = json.loads(record_file.read_text(encoding='utf-8'))
                            failure = previous['failure']
                            with np.load(artifact) as saved:
                                result = {key: saved[key] for key in saved.files}
                            resumed_histories += 1
                        else:
                            w, norm = models[model]
                            result, failure = integrate(w, norm, t, command, config['numerics'], stress_control=True)
                            np.savez_compressed(artifact, **result, prescribed_stress=command, phase=phase)
                            new_executions += 1
                        reused = False
                    row = dict(
                        history=hid, preload=preload['id'], delay_s=delay,
                        candidate=candidate['id'], model=model, reused_frozen_history=reused,
                        integration_complete=failure is None, failure=failure,
                        **response_metrics(result, truth, recovery_start, probe_start, failure),
                        artifact=artifact.relative_to(root).as_posix(), artifact_sha256=_sha256(artifact),
                    )
                    records.append(row)
                    rollouts[hid, model] = result
                    artifacts.append(dict(history=hid, model=model, path=row['artifact'], sha256=row['artifact_sha256']))
                    if not reused:
                        _json(record_file, row)
                        print(json.dumps({key: row[key] for key in
                              ('history', 'model', 'integration_complete', 'peak_compression', 'failure')}), flush=True)
    pairs = candidate_pairs(rc, records)
    _csv(compact/'all_rollouts.csv', [{**row, 'failure': json.dumps(row['failure']) if row['failure'] else ''}
                                     for row in records])
    _csv(compact/'candidate_comparison.csv', pairs)
    _json(destination/'artifacts.json', artifacts)
    _figures(rc, compact, rollouts, pairs)
    summary = dict(
        kind='Project material revisit comparison: frozen single-history and new multi-history models',
        run=run, complete_physical_histories=12, candidate_pairs=6,
        reused_reference_histories=12, reused_old_model_histories=24,
        new_model_histories=24, total_material_histories=len(records),
        newly_executed_this_invocation=new_executions, resumed_new_histories=resumed_histories,
        successful_new_histories=sum(r['integration_complete'] for r in records if r['model'].startswith('new_')),
        selected=training['selection'], frozen_selected=frozen_index['selected'],
        selection_rule=training['rule'], additional_training_runs=0,
        deployment_inputs=['prescribed stress', 'time increments', 'own previous internal state'],
        auxiliary_RNN_in_deployment=False, reference_internal_variables_in_deployment=False,
        ranking_tie_tolerance=rc['metrics']['ranking_tie_tolerance_absolute_strain'],
        tie_choice='Uniform choice over early/late, expected and worst regret under reference peak compression; reporting convention, not deployed policy',
        outcome='Positive peak max(0,-epsilon_11) during common reload probe, minimized',
        recovery_metrics='Zero-stress delay after the common candidate window; include both interval endpoints',
        summaries=_summaries(records, pairs), pairs=pairs,
        failures=[r for r in records if not r['integration_complete']],
        raw_artifacts=destination.relative_to(root).as_posix(),
        original_method_index=(root/config['outputs']['compact_root']/run/'index.json').relative_to(root).as_posix(),
        frozen_index=(frozen_compact/'index.json').relative_to(root).as_posix(),
        metadata=metadata, elapsed_s=time.monotonic()-begin,
        scope='All six reference pairs favor early. Agreement with them cannot demonstrate history-dependent action selection or robot-control benefit; compare always-early regret explicitly.',
    )
    _json(compact/'index.json', summary)
    print('Compared 24 new selected-model histories with 36 frozen histories; no frozen rollout recomputed.', flush=True)
    return summary


def _figures(protocol, compact, rollouts, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .figures import style, save

    style()
    colors = dict(reference='#222222', old_ideal='#0072B2', new_ideal='#0072B2',
                  old_noisy='#D55E00', new_noisy='#D55E00')
    styles = dict(reference='-', old_ideal=':', new_ideal='--', old_noisy=':', new_noisy='--')
    labels = dict(reference='Reference', old_ideal='Old ideal', new_ideal='New ideal',
                  old_noisy='Old noisy', new_noisy='New noisy')
    for preload in protocol['preloading_histories']:
        for delay in protocol['recovery_delays_s']:
            fig, axes = plt.subplots(4, 2, figsize=(11.2, 9.2), sharex=False, layout='constrained')
            for col, candidate in enumerate(protocol['current_action_candidates']):
                hid = f"{preload['id']}-delay-{delay:g}-{candidate['id']}"
                ref = rollouts[hid, 'reference']
                t = ref['time']
                recovery_start = preload['duration_s']+candidate['duration_s']
                probe_start = recovery_start+delay
                recovery = (t >= recovery_start-1e-10) & (t <= probe_start+1e-10)
                probe = t >= probe_start-1e-10
                axes[0, col].plot(t, ref['prescribed_stress'][:, 0], color=colors['reference'])
                axes[0, col].set_title(f"{candidate['id'].capitalize()} loading")
                axes[0, col].set_ylabel(r'Stress $\sigma_{11}$ (MPa)')
                for model in MODELS:
                    strain = rollouts[hid, model]['strain'][:, 0]
                    for row, mask in ((1, slice(None)), (2, recovery), (3, probe)):
                        axes[row, col].plot(t[mask], -100*strain[mask], color=colors[model],
                                            ls=styles[model], label=labels[model])
                axes[1, col].set_ylabel('Full compression (%)')
                axes[2, col].set_ylabel('Recovery compression (%)')
                axes[3, col].set_ylabel('Probe compression (%)')
                for row in (0, 1):
                    axes[row, col].axvspan(probe_start, t[-1], color='#999999', alpha=.12)
                for row in range(4):
                    axes[row, col].grid(alpha=.16)
                    axes[row, col].set_xlabel('Time (s)')
            handles, names = axes[1, 0].get_legend_handles_labels()
            fig.legend(handles, names, loc='outside lower center', ncol=5)
            fig.suptitle(f"{preload['id']}, recovery {delay:g} s: full histories and recovery/probe details\n"
                         'New noisy: project extension; shaded full-history interval: common probe')
            save(fig, compact, f"history-{preload['id']}-delay-{delay:g}")
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 8.2), layout='constrained')
    metrics = ('candidate_center_bias', 'predicted_early_minus_late', 'response_ratio')
    ylabel = ('Candidate center bias (strain)', 'Early minus late peak (strain)', 'Predicted/reference action gap')
    for col, preload in enumerate(protocol['preloading_histories']):
        for row, (metric, axis_label) in enumerate(zip(metrics, ylabel)):
            ax = axes[row, col]
            for model in MODELS:
                group = [p for p in pairs if p['preload'] == preload['id'] and p['model'] == model]
                if metric == 'candidate_center_bias' and model == 'reference':
                    continue
                x = [p['delay_s'] for p in group]
                y = [np.nan if p[metric] is None else p[metric] for p in group]
                ax.plot(x, y, color=colors[model], ls=styles[model],
                        marker='o' if model.startswith('new_') else 'x', label=labels[model])
            ax.set_xscale('log')
            ax.set_xlabel('Recovery delay (s)')
            ax.set_ylabel(axis_label)
            ax.grid(alpha=.18)
            if row == 0:
                ax.set_title(preload['id'])
            if row == 1:
                tol = protocol['metrics']['ranking_tie_tolerance_absolute_strain']
                ax.axhspan(-tol, tol, color='#999999', alpha=.2)
            ax.axhline(1 if row == 2 else 0, color='#999999', lw=.7)
    handles, names = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=5)
    fig.suptitle('Amplitude bias and action response are distinct outcomes; gray action-gap band: original tie tolerance')
    save(fig, compact, 'candidate-bias-and-action-response')
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout='constrained')
    reference_pairs = [p for p in pairs if p['model'] == 'reference']
    tick_labels = [f"{p['preload'].removeprefix('preload_')}\n{p['delay_s']:g} s" for p in reference_pairs]
    policies = (*MODELS[1:], 'always_early')
    for ax, metric in zip(axes, ('expected_regret', 'worst_case_regret')):
        for j, model in enumerate(policies):
            rows = [p for p in pairs if p['model'] == model]
            x = np.arange(len(rows))+(j-2)*.13
            values = [np.nan if p[metric] is None else p[metric] for p in rows]
            color = '#222222' if model == 'always_early' else colors[model]
            label = 'Always early' if model == 'always_early' else labels[model]
            marker = 'x' if model == 'always_early' else ('o' if model.startswith('new_') else 's')
            ax.scatter(x, values, color=color, label=label, marker=marker, s=30,
                       facecolors='none' if model.startswith('old_') else color)
        ax.set_xticks(np.arange(len(tick_labels)), tick_labels)
        ax.set_ylabel('Peak-compression regret (strain)')
        ax.set_title(metric.replace('_', ' ').capitalize())
        ax.grid(axis='y', alpha=.2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=5)
    fig.suptitle('Six fixed pairs; predicted ties use equal-probability early/late choice\n'
                 'Always-early reaches the reference minimum in all six pairs')
    save(fig, compact, 'decision-regret-and-constant-baseline')
