"""Physical-unit figures and complete response atlases from actual study results."""
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
import multiprocessing
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image


ARMS = ('A_random100', 'B_random1000', 'C_mixed100')
CELLS = ('ideal', 'noise0', 'noise1', 'noise2')
COLORS = {'A_random100': '#0072B2', 'B_random1000': '#D55E00',
          'C_mixed100': '#009E73', 'reference': '#242424', 'always_early': '#777777'}
STYLES = {'A_random100': '-', 'B_random1000': '--', 'C_mixed100': ':', 'reference': '-.'}
MARKERS = {'A_random100': 'o', 'B_random1000': '^', 'C_mixed100': 's', 'always_early': 'D'}
COMPONENTS = ('11', '22', '33', '12', '13', '23')
FAMILY_TITLES = {
    'F1_random_original': 'F1: original random',
    'F2_low_amplitude_rate': 'F2: low amplitude / rate',
    'F3_strain_relaxation': 'F3: strain relaxation',
    'F4_stress_creep_recovery': 'F4: creep / free recovery',
    'F5_compression_shear_cycles': 'F5: compression / shear cycles',
    'F6_revisit_pairs': 'F6: revisit candidate pairs',
}


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _number(row, key):
    value = row[key]
    return None if value in ('', None) else float(value)


def _style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .18, 'grid.linewidth': .5,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'lines.linewidth': 1.2, 'savefig.facecolor': 'white'})


def _save(fig, directory, name, caption, grayscale=False):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory/(name+'.png'), dpi=200, bbox_inches='tight')
    fig.savefig(directory/(name+'.svg'), bbox_inches='tight')
    if grayscale:
        with Image.open(directory/(name+'.png')) as picture:
            picture.convert('L').save(directory/(name+'-grayscale.png'))
    plt.close(fig)
    return dict(name=name, png=str(directory/(name+'.png')), svg=str(directory/(name+'.svg')),
                caption=caption)


def _legend(arms=ARMS, reference=False):
    handles = [Line2D([0], [0], color=COLORS[arm], ls=STYLES[arm],
                      marker=MARKERS[arm], markersize=4, label=arm[0]) for arm in arms]
    if reference:
        handles.insert(0, Line2D([0], [0], color=COLORS['reference'], ls=STYLES['reference'],
                                 label='Reference'))
    return handles


def _by_condition(rows, metric):
    """Two F6 candidate histories remain one physical condition in dot plots."""
    values = []
    for condition in sorted({r['condition_id'] for r in rows}):
        group = [r for r in rows if r['condition_id'] == condition]
        numbers = [_number(r, metric) for r in group]
        values.append(dict(condition_id=condition, cluster_id=group[0]['cluster_id'],
                           value=float(np.mean(numbers)) if all(v is not None for v in numbers) else None))
    return values


def _dots(ax, x, values, color, marker, clustered=False):
    usable = [r for r in values if r['value'] is not None]
    jitter = np.linspace(-.12, .12, len(values))
    locations = {r['condition_id']: float(offset) for r, offset in zip(values, jitter)}
    ax.scatter([x+locations[r['condition_id']] for r in usable], [r['value'] for r in usable],
               s=11, color=color, marker=marker, alpha=.35, linewidths=0)
    if clustered:
        clusters = sorted({r['cluster_id'] for r in values})
        for offset, cluster in zip(np.linspace(-.15, .15, len(clusters)), clusters):
            members = [r['value'] for r in values if r['cluster_id'] == cluster]
            if all(v is not None for v in members):
                ax.scatter(x+offset, np.mean(members), color=color, edgecolor='black',
                           marker='D', s=34, linewidths=.7, zorder=4)
    return len(usable), len(values)


def _error_figures(records, paired, compact):
    artifacts = []
    for cell in CELLS:
        for statistic in ('MAE', 'RMSE'):
            for is_paired in (False, True):
                fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.7), layout='constrained')
                for ax, family in zip(axes.flat, FAMILY_TITLES):
                    strain_control = family.startswith(('F1_', 'F2_', 'F3_'))
                    metric = f'stress_{statistic}_MPa' if strain_control else f'full_history_strain_{statistic}'
                    unit = 'MPa' if strain_control else 'strain'
                    clustered = family.startswith(('F2_', 'F5_'))
                    counts = []
                    if is_paired:
                        metric = 'A_minus_other_'+metric
                        for index, (comparison, arm) in enumerate(zip(('A_minus_B', 'A_minus_C'), ARMS[1:])):
                            group = [r for r in paired if r['family'] == family and
                                     r['cell'] == cell and r['comparison'] == comparison]
                            counts.append(_dots(ax, index, _by_condition(group, metric),
                                               COLORS[arm], MARKERS[arm], clustered))
                        ax.set_xticks([0, 1], ['A - B', 'A - C'])
                        ax.axhline(0, color='#333333', lw=.8)
                        ax.set_ylabel(f'Paired {statistic} reduction ({unit})')
                    else:
                        for index, arm in enumerate(ARMS):
                            group = [r for r in records if r['family'] == family and
                                     r['cell'] == cell and r['arm'] == arm]
                            counts.append(_dots(ax, index, _by_condition(group, metric),
                                               COLORS[arm], MARKERS[arm], clustered))
                        ax.set_xticks([0, 1, 2], ['A: random 100', 'B: random 1000', 'C: mixed 100'], rotation=12)
                        ax.set_ylim(bottom=0)
                        ax.set_ylabel(f'Full-history {statistic} ({unit})')
                    note = '5 base-seed clusters' if clustered else '100 prescribed conditions'
                    ax.set_title(FAMILY_TITLES[family]+'\n'+note)
                    ax.text(.03, .97, 'Complete: '+', '.join(f'{a}/{b}' for a, b in counts),
                            transform=ax.transAxes, va='top', fontsize=7)
                    ax.yaxis.set_major_locator(MaxNLocator(5))
                    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
                prefix = 'paired-changes' if is_paired else 'physical-errors'
                title = f'{cell}: '+('positive values favor B/C over A' if is_paired else 'fixed-condition material errors')
                fig.suptitle(title, fontsize=12)
                caption = ('Each small point is one prescribed physical condition; F6 averages the two '
                    'candidate-history errors within each condition. F2/F5 diamonds are the five complete '
                    'base-seed cluster means. No IID confidence intervals or significance claims. '
                    'Incomplete conditions remain in the displayed denominators and have no fabricated error. '
                    'A and C have 100 training histories; B has 1000 and greater computation.')
                fig.supxlabel('Small points: conditions; outlined diamonds: F2/F5 base-seed means. No IID inference.', fontsize=8)
                artifacts.append(_save(fig, compact, f'{prefix}-{statistic.lower()}-{cell}', caption, True))
    return artifacts


def _training_figures(training, out, compact):
    styles = ('-', '--', ':', '-.', (0, (4, 1, 1, 1, 1, 1)))
    fig, axes = plt.subplots(3, 4, figsize=(11.6, 8.1), sharex=True, layout='constrained')
    for i, arm in enumerate(ARMS):
        for j, cell in enumerate(CELLS):
            ax = axes[i, j]
            selected = training['selection'][arm+'--'+cell]
            rows = sorted((r for r in training['runs'] if r['arm'] == arm and r['cell'] == cell),
                          key=lambda r: r['seed'])
            for row in rows:
                with np.load(out/row['loss_curve']) as saved:
                    total = saved['stress_MAE']+saved['biot_MAE']
                    updates = saved['iteration']+1
                chosen = row['seed'] == selected['seed']
                ax.semilogy(updates, total, color=COLORS[arm], ls=styles[row['seed']],
                            alpha=1. if chosen else .45, lw=1.5 if chosen else .8)
                ax.scatter(updates[-1], row['final_training_loss_MPa'],
                           marker='*' if chosen else 'o', color=COLORS[arm],
                           s=35 if chosen else 10, zorder=4)
            ax.set_title(f"{arm[0]} / {cell}; selected seed {selected['seed']}")
            if j == 0:
                ax.set_ylabel('Stress MAE + Biot MAE (MPa, log)')
            if i == 2:
                ax.set_xlabel('Complete-dataset Adam update')
            ax.set_xlim(0, 20500)
    handles = [Line2D([0], [0], color='#444444', ls=styles[seed], label=f'Seed {seed}') for seed in range(5)]
    fig.legend(handles=handles, ncols=5, loc='outside lower center', frameon=False)
    fig.suptitle('All 60 complete training-loss curves; 10 archived models reused', fontsize=12)
    return _save(fig, compact, 'all-training-losses',
        'Every recorded update is plotted without smoothing. Stars mark final-weight training losses '
        'for the selected seed in each cell, using training loss only. Lines are pre-update recorded losses. '
        'A/C use 100 histories per Adam update, B uses 1000; update count is not equal computation.', True)


def _local_figures(out, compact, training):
    rows = _json(out/'diagnostics/local-zero-state-recovery.json')['records']
    reference = next(r for r in rows if r['arm'] == 'reference')
    fields = [('K_eq_MPa', 'Equilibrium bulk modulus (MPa)'),
              ('G_eq_MPa', 'Equilibrium shear modulus (MPa)'),
              ('free_recovery_volumetric_time_s', 'Local volumetric recovery time (s)'),
              ('free_recovery_deviatoric_time_s', 'Local deviatoric recovery time (s)')]
    fig, axes = plt.subplots(4, 4, figsize=(11.5, 9.7), layout='constrained')
    for i, (metric, ylabel) in enumerate(fields):
        for j, cell in enumerate(CELLS):
            ax = axes[i, j]
            ax.axhline(reference[metric], color=COLORS['reference'], ls='-.', lw=1)
            for index, arm in enumerate(ARMS):
                group = sorted((r for r in rows if r['arm'] == arm and r['cell'] == cell), key=lambda r: r['seed'])
                for offset, row in zip(np.linspace(-.12, .12, 5), group):
                    chosen = row['seed'] == training['selection'][arm+'--'+cell]['seed']
                    ax.scatter(index+offset, row[metric], color=COLORS[arm],
                               marker='*' if chosen else MARKERS[arm], s=55 if chosen else 24)
            ax.set_xticks([0, 1, 2], ['A', 'B', 'C'])
            if i == 0:
                ax.set_title(cell)
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle('Zero-state observable material quantities: every initialization', fontsize=12)
    fig.supxlabel('Dots: five fixed initializations per cell; stars: training-loss choices; dash-dot: reference.', fontsize=8)
    return _save(fig, compact, 'local-material-recovery',
        'All 60 material models are shown. Recovery times derive from the zero-state physical tensor '
        'Hessians and the zero-stress observable poles; they do not replace nonlinear history evolution '
        'with a constant relaxation time. Five initializations share one dataset per noise cell.', True)


def _candidate_figures(pairs, compact, tolerance):
    artifacts = []
    fixed = sorted({r['condition_id'] for r in pairs})
    reference = {r['condition_id']: r for r in pairs if r['arm'] == 'always_early'}
    reference_ranks = [reference[k]['reference_rank'] for k in fixed]
    categories = {'early': 0, 'tie': 1, 'late': 2, 'integration_failure': 3}
    cmap = ListedColormap(['#0072B2', '#BBBBBB', '#D55E00', '#CC79A7'])
    for cell in CELLS:
        fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), layout='constrained')
        for arm in ARMS:
            group = [r for r in pairs if r['cell'] == cell and r['arm'] == arm]
            complete = [r for r in group if r['integration_complete'] == 'True']
            x = [_number(r, 'reference_center') for r in complete]
            axes[0].scatter(x, [_number(r, 'candidate_center_bias') for r in complete], s=15,
                            color=COLORS[arm], marker=MARKERS[arm], alpha=.6, label=arm[0])
            axes[1].scatter([_number(r, 'reference_early_minus_late') for r in complete],
                            [_number(r, 'predicted_early_minus_late') for r in complete], s=15,
                            color=COLORS[arm], marker=MARKERS[arm], alpha=.6)
            axes[2].scatter([fixed.index(r['condition_id']) for r in complete],
                            [_number(r, 'expected_regret') for r in complete], s=15,
                            color=COLORS[arm], marker=MARKERS[arm], alpha=.6)
            axes[2].scatter([fixed.index(r['condition_id']) for r in complete],
                            [_number(r, 'worst_case_regret') for r in complete], s=23,
                            facecolors='none', edgecolors=COLORS[arm], marker=MARKERS[arm], linewidths=.6)
        axes[0].axhline(0, color='#333333', lw=.8)
        axes[0].set(xlabel='Reference candidate center (strain)', ylabel='Predicted center - reference (strain)',
                    title='Amplitude bias')
        axes[0].legend(frameon=False)
        bounds = axes[1].get_xlim()+axes[1].get_ylim()
        low, high = min(bounds), max(bounds)
        axes[1].plot([low, high], [low, high], color='#555555', ls='-.', lw=.8)
        axes[1].axvspan(-tolerance, tolerance, color='#DDDDDD', alpha=.6)
        axes[1].axhspan(-tolerance, tolerance, color='#DDDDDD', alpha=.6)
        axes[1].set_xscale('symlog', linthresh=tolerance)
        axes[1].set_yscale('symlog', linthresh=tolerance)
        axes[1].set(xlabel='Reference early - late (strain)', ylabel='Predicted early - late (strain)',
                    title='Action difference; linear band +/- 1e-6')
        axes[2].scatter(np.arange(len(fixed)), [_number(reference[k], 'expected_regret') for k in fixed],
                        marker='x', color=COLORS['always_early'], s=13, label='Always early')
        axes[2].set(xlabel='Prescribed physical-pair index', ylabel='Reference-outcome regret (strain)',
                    title='Filled: expected; outline: worst case')
        axes[2].set_ylim(bottom=0)
        axes[2].legend(frameon=False)
        fig.suptitle(f'{cell}: all 100 fixed revisit pairs', fontsize=12)
        artifacts.append(_save(fig, compact, 'revisit-response-'+cell,
            'Each point is one physical candidate pair. Center bias is separated from early-minus-late '
            'response. Both action-difference axes use symlog with an unchanged +/-1e-6 strain linear '
            'band. No ratio is shown for near-zero reference gaps. Ties choose early/late with equal '
            'probability for expected and worst-case regret; the always-early baseline uses the same references.', True))
        grid = [reference_ranks]
        labels = ['Reference']
        for arm in ARMS:
            lookup = {r['condition_id']: r for r in pairs if r['cell'] == cell and r['arm'] == arm}
            grid.append([lookup[k]['predicted_rank'] for k in fixed])
            labels.append(arm[0])
        grid.append(['early']*len(fixed))
        labels.append('Always early')
        numbers = np.asarray([[categories[value] for value in row] for row in grid])
        fig, ax = plt.subplots(figsize=(11.6, 3.0), layout='constrained')
        ax.imshow(numbers, aspect='auto', interpolation='none', cmap=cmap,
                  norm=BoundaryNorm(np.arange(-.5, 4.5), cmap.N))
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel('Prescribed physical-pair index (same order as response and regret plot)')
        ax.set_title(f'{cell}: complete action choices at fixed 1e-6 strain tie tolerance')
        ax.grid(False)
        rank_markers = ('o', 'D', '^', 'x')
        for category, marker in enumerate(rank_markers):
            ys, xs = np.nonzero(numbers == category)
            ax.scatter(xs, ys, marker=marker, s=13, c='white' if marker != 'x' else 'black',
                       edgecolors='black' if marker != 'x' else None, linewidths=.4)
        legend = [Line2D([0], [0], marker=marker, color='none', markerfacecolor=cmap(index),
                         markeredgecolor='#333333', markersize=7, label=label)
                  for index, (label, marker) in enumerate((('Early', 'o'), ('Near-tie / tie', 'D'),
                                                          ('Late', '^'), ('Integration failure', 'x')))]
        fig.legend(handles=legend, loc='outside lower center', ncols=4, frameon=False)
        for y, row in enumerate(grid):
            counts = [row.count(key) for key in ('early', 'tie', 'late', 'integration_failure')]
            ax.text(1.005, 1-(y+.5)/len(grid), '/'.join(map(str, counts)), transform=ax.transAxes,
                    va='center', fontsize=7)
        artifacts.append(_save(fig, compact, 'revisit-choices-'+cell,
            'Rows preserve all 100 physical pairs; the color categories are early, tie, late and actual '
            'integration failure. Counts are early/tie/late/failure. Reference near-ties describe decayed '
            'memory and are not strong ordering evidence. These fixed-grid counts are not IID trials.', True))
    return artifacts


def _history_figure(history_id, rows, training, destination, prediction_root, diagnostic=False):
    with np.load(prediction_root/rows[0]['reference_relative_path']) as saved:
        reference = {key: saved[key] for key in ('time', 'strain', 'stress')}
    response = 'stress' if rows[0]['mode'] == 'strain_control' else 'strain'
    symbol, unit = ('sigma', 'MPa') if response == 'stress' else ('epsilon', 'strain')
    fig, axes = plt.subplots(6, 4, figsize=(12.6, 11.4), sharex=True, sharey='row', layout='constrained')
    failures = []
    for column, cell in enumerate(CELLS):
        group = sorted((r for r in rows if r['cell'] == cell), key=lambda r: (r['arm'], int(r['seed'])))
        for row in group:
            arm = row['arm']
            selected_seed = training['selection'][arm+'--'+cell]['seed']
            chosen = int(row['seed']) == selected_seed
            with np.load(prediction_root/row['prediction_relative_path']) as saved:
                time, prediction = saved['time'], saved[response]
            for component in range(6):
                axes[component, column].plot(time, prediction[:, component],
                    color=COLORS[arm], ls=STYLES[arm],
                    alpha=.3 if diagnostic and not chosen else .95,
                    lw=.65 if diagnostic and not chosen else 1.25)
            if row['integration_complete'] != 'True':
                failures.append(f"{arm[0]}/{cell}/s{row['seed']}@{float(row['last_complete_time_s']):g}s")
        for component, index in enumerate(COMPONENTS):
            ax = axes[component, column]
            ax.plot(reference['time'], reference[response][:, component],
                    color=COLORS['reference'], ls=STYLES['reference'], lw=1.)
            ax.set_xlim(reference['time'][0], reference['time'][-1])
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
            if column == 0:
                ax.set_ylabel(f'{symbol}_{index} ({unit})')
            if component == 0:
                ax.set_title(cell)
            if component == 5:
                ax.set_xlabel('Time (s)')
    fig.suptitle(history_id+'\n'+('All 60 models; faint lines retain the other four initializations per cell'
                                 if diagnostic else '12 final-training-loss selections; independent full-history state evolution'),
                 fontsize=11)
    fig.legend(handles=_legend(reference=True), loc='outside lower center', ncols=4, frameon=False)
    if failures:
        fig.supxlabel('Actual incomplete curves: '+', '.join(failures), fontsize=6)
    return _save(fig, destination, history_id,
        f'All six {response} components for the entire prescribed history, including every saved time point. '
        'Each cell overlays A/B/C with the shared reference. '+
        ('All five initializations per arm are shown, with the training-loss choice emphasized. ' if diagnostic else '')+
        ('Actual incomplete curves: '+', '.join(failures) if failures else 'Every displayed model history completed.'))


def _atlas(records, training, destination, prediction_root, compact, examples, diagnostic=False):
    artifacts = []
    groups = {}
    for row in records:
        groups.setdefault(row['history_id'], []).append(row)
    ordered = sorted(groups.items())
    # Each process owns one complete history figure and its distinct files.
    # Summary figures and copying the fixed compact examples stay in the parent.
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_style) as executor:
        results = executor.map(_history_figure,
            (history_id for history_id, rows in ordered), (rows for history_id, rows in ordered),
            repeat({'selection': training['selection']}), repeat(destination), repeat(prediction_root),
            repeat(diagnostic), chunksize=1)
        for index, ((history_id, rows), artifact) in enumerate(zip(ordered, results), 1):
            artifact['history_id'] = history_id
            artifacts.append(artifact)
            if history_id in examples:
                for suffix in ('.png', '.svg'):
                    shutil.copyfile(destination/(history_id+suffix), compact/(history_id+suffix))
            if index % 25 == 0:
                print(f"Rendered {'diagnostic' if diagnostic else 'final'} complete-history figure {index}/{len(groups)}.",
                      flush=True)
    return artifacts


def study_figures(study, out, compact):
    """Render only completed real training/evaluation artifacts; no placeholder data."""
    out, compact = Path(out), Path(compact)
    destination = out/'figures'
    compact_figures = compact/'figures'
    compact_figures.mkdir(parents=True, exist_ok=True)
    _style()
    training = _json(out/'training.json')
    final_root = out/'final_predictions'
    diagnostic_root = out/'diagnostics/predictions'
    final = _csv(final_root/'complete-history-metrics.csv')
    diagnostic = _csv(diagnostic_root/'complete-history-metrics.csv')
    paired = _csv(final_root/'paired-arm-history-changes.csv')
    candidates = _csv(compact/'final/candidate-pairs.csv')
    artifacts = _error_figures(final, paired, compact_figures)
    artifacts.append(_training_figures(training, out, compact_figures))
    artifacts.append(_local_figures(out, compact_figures, training))
    artifacts.extend(_candidate_figures(candidates, compact_figures,
                      study['final_evaluation']['metrics']['ranking_tolerance_absolute_strain']))
    # These examples are fixed by protocol order, before any error-based inspection.
    examples = {family+'-000'+('-early' if family.startswith('F6_') else '') for family in FAMILY_TITLES}
    final_atlas = _atlas(final, training, destination/'final-history-curves', final_root, compact_figures, examples)
    diagnostic_atlas = _atlas(diagnostic, training, destination/'known-history-curves', diagnostic_root,
                              compact_figures, set(), True)
    manifest = dict(summary_figures=artifacts, final_history_curves=final_atlas,
        known_history_curves=diagnostic_atlas, compact_history_examples=sorted(examples),
        final_physical_conditions=len({r['condition_id'] for r in final}),
        final_complete_histories=len({r['history_id'] for r in final}),
        final_model_history_predictions=len(final), known_model_history_predictions=len(diagnostic),
        final_actual_failures=sum(r['integration_complete'] != 'True' for r in final),
        known_actual_failures=sum(r['integration_complete'] != 'True' for r in diagnostic),
        units='MPa for prescribed-strain stress errors; strain for prescribed-stress response errors.',
        design='Every physical condition retained. F6 candidates aggregate within condition for summary points; '
               'F2/F5 show the five seed clusters. No IID confidence intervals or means-only bar plots.',
        rendering='PNG previews and SVG vector output, fixed colorblind palette with redundant line/marker codes; '
                  'compact summaries include grayscale previews. Complete histories are not temporally downsampled.',
        atlas_rendering_processes=8, atlas_process_start_method='spawn')
    (destination/'figures.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    final_summary = _json(final_root/'evaluation.json')
    diagnostic_summary = _json(diagnostic_root/'evaluation.json')
    def evaluation_index(summary, path):
        return dict(evaluation=path, models=summary['model_count'],
                    physical_conditions=summary['physical_conditions'],
                    complete_reference_histories=summary['complete_reference_histories'],
                    boundary_replays_per_model=summary['boundary_replays_per_model'],
                    model_history_cells=summary['model_history_cells'],
                    successful_histories=summary['successful_histories'],
                    actual_failures=len(summary['failures']))
    index = dict(kind='TM02 material-memory applicability study', run=out.name,
        training=dict(study_models=training['study_models'], new_models=training['new_models'],
            reused_models=training['reused_models'], selected_models=training['selected_models'],
            total_new_optimizer_updates=training['total_optimizer_updates'],
            actual_history_gradient_evaluations=training['history_gradient_evaluations'],
            new_training_elapsed_s=training['elapsed_s'], selection=training['selection']),
        final_evaluation=evaluation_index(final_summary, 'final_predictions/evaluation.json'),
        known_diagnostics=evaluation_index(diagnostic_summary, 'diagnostics/predictions/evaluation.json'),
        figures=dict(large_index='figures/figures.json', summary_figures=len(artifacts),
                     final_history_figures=len(final_atlas), known_history_figures=len(diagnostic_atlas),
                     compact_directory='figures', compact_examples=sorted(examples)),
        scope='Fixed nonlinear Table 1 reference material, unchanged model/loss/scales, '
              'three data arms and three independent training-noise realizations. '
              'B uses greater computation. This project extension does not establish robot, MPC, or RL benefit.',
        output_directory=str(out), compact_directory=str(compact))
    (compact/'index.json').write_text(json.dumps(index, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Rendered {len(artifacts)} summary figures, {len(final_atlas)} final and '
          f'{len(diagnostic_atlas)} known complete-history figures.', flush=True)
    return manifest
