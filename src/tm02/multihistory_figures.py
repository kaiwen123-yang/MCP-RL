"""Figures drawn from the saved, complete paired material trajectories."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .material import COMPONENTS


COLORS = {'ideal': '#0072B2', 'noisy': '#D55E00'}


def _style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 8,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'savefig.dpi': 200, 'lines.linewidth': 1.3})


def _save(fig, directory, name):
    fig.savefig(directory / f'{name}.png', dpi=200)
    fig.savefig(directory / f'{name}.svg')
    plt.close(fig)


def _correlation(out, compact, condition):
    with np.load(out / 'selected-heldout-full-responses.npz') as arrays:
        truth = arrays['reference_stress'][:, 1:]
        prediction = arrays[f'new_{condition}_stress'][:, 1:]
        seed = int(arrays[f'new_{condition}_seed'])
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2), layout='constrained')
    for k, ax in enumerate(axes.flat):
        target = truth[..., k].ravel()
        values = prediction[..., k].ravel()
        valid = np.isfinite(values)
        low = min(float(target.min()), float(values[valid].min())) if valid.any() else float(target.min())
        high = max(float(target.max()), float(values[valid].max())) if valid.any() else float(target.max())
        ax.plot([low, high], [low, high], color='#555555', linewidth=.9, label='Identity')
        ax.scatter(target[valid], values[valid], s=3, alpha=.28, color=COLORS[condition],
                   marker='o' if condition == 'ideal' else '^', rasterized=True)
        ax.set_xlabel('Reference stress (MPa)')
        ax.set_ylabel('Free prediction (MPa)')
        ax.set_title(f'$\\sigma_{{{COMPONENTS[k]}}}$')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.15)
    complete = int(np.isfinite(prediction).all(axis=(1, 2)).sum())
    label = 'Fig.14 ideal multi-history reimplementation' if condition == 'ideal' else 'Noisy multi-history project extension'
    fig.suptitle(f'{label}; final-loss-selected seed {seed}\n'
                 f'20 fixed unseen histories × 250 increments; {complete}/20 complete; auxiliary RNN discarded')
    _save(fig, compact, f'fig14-{condition}-heldout-correlation')


def _responses(out, compact, config, role):
    filename = 'selected-heldout-full-responses.npz' if role == 'heldout' else 'selected-known-full-responses.npz'
    with np.load(out / filename) as loaded:
        arrays = dict(loaded)
    npaths = len(arrays['time']) if role == 'heldout' else 1
    for path_id in range(npaths):
        clock = arrays['time'][path_id] if role == 'heldout' else arrays['time']
        target = arrays['reference_stress'][path_id] if role == 'heldout' else arrays['reference_stress']
        for condition in ('ideal', 'noisy'):
            fig, axes = plt.subplots(3, 2, figsize=(9, 7), sharex=True, layout='constrained')
            old = arrays[f'old_{condition}_stress']
            new = arrays[f'new_{condition}_stress']
            if role == 'heldout':
                old, new = old[path_id], new[path_id]
            old_seed, new_seed = int(arrays[f'old_{condition}_seed']), int(arrays[f'new_{condition}_seed'])
            for k, ax in enumerate(axes.flat):
                ax.plot(clock, target[:, k], color='#202020', label='Reference')
                ax.plot(clock, old[:, k], color='#888888', linestyle='--', label=f'Old seed {old_seed}')
                ax.plot(clock, new[:, k], color=COLORS[condition], linestyle=':', linewidth=1.7,
                        label=f'New seed {new_seed}')
                ax.set_ylabel(f'$\\sigma_{{{COMPONENTS[k]}}}$ (MPa)')
                ax.grid(alpha=.18)
            for ax in axes[-1]:
                ax.set_xlabel('Time (s)')
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
            suffix = 'ideal original-method extension' if condition == 'ideal' else 'noisy project extension'
            if role == 'heldout':
                seed = config['data']['evaluation']['strain_seeds'][path_id]
                name = f'heldout-path-{path_id:02d}-{condition}-full-response'
                title = f'Unseen history {path_id:02d}, strain seed {seed}; {suffix}'
            else:
                name = f'known-diagnostic-{condition}-full-response'
                title = f'Previously inspected difficult history; {suffix}'
            old_complete, new_complete = np.isfinite(old).all(), np.isfinite(new).all()
            fig.suptitle(title + '\nFinal training loss selects models; '
                         + f'complete old/new: {old_complete}/{new_complete}')
            _save(fig, compact, name)


def _training(out, compact, training):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, layout='constrained')
    for column, condition in enumerate(('ideal', 'noisy')):
        chosen = training['selection'][condition]['seed']
        for record in (r for r in training['runs'] if r['condition'] == condition):
            seed = record['seed']
            with np.load(out / 'final' / f'{condition}-seed-{seed:02d}-loss.npz') as curve:
                steps = curve['iteration'][::10] + 1
                for row, term in enumerate(('stress_MAE', 'biot_MAE')):
                    axes[row, column].semilogy(steps, curve[term][::10],
                        color=COLORS[condition] if seed == chosen else '#aaaaaa',
                        alpha=1 if seed == chosen else .45, linewidth=1.4 if seed == chosen else .55,
                        zorder=3 if seed == chosen else 1)
        for row, term in enumerate(('Stress fit', 'Biot consistency')):
            axes[row, column].set_title(f'{condition}: {term}')
            axes[row, column].set_ylabel('MAE (MPa)')
            axes[row, column].grid(alpha=.18)
    for ax in axes[-1]:
        ax.set_xlabel('Full-dataset Adam updates')
    fig.suptitle('All 25 initializations per condition; final-loss-selected model highlighted\n'
                 '100 complete histories per update; display every 10th update, all 20,000 saved')
    _save(fig, compact, 'multihistory-training-curves')


def _initializations(compact, summaries):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), layout='constrained')
    groups = [(cohort, condition) for condition in ('ideal', 'noisy') for cohort in ('old', 'new')]
    metrics = [('RMSE_MPa', 'Stress RMSE'), ('hydrostatic_RMSE_MPa', 'Hydrostatic RMSE'),
               ('deviatoric_component_RMSE_MPa', 'Deviatoric component RMSE')]
    counts = []
    for position, (cohort, condition) in enumerate(groups):
        all_rows = [r for r in summaries if r['cohort'] == cohort and r['condition'] == condition]
        rows = [r for r in all_rows if r['all_paths_complete']]
        counts.append(f'{len(rows)}/{len(all_rows)}')
        for ax, (metric, label) in zip(axes, metrics):
            values = [r[f'mean_path_{metric}'] for r in rows]
            if values:
                ax.boxplot([values], positions=[position], widths=.36, showfliers=False,
                           medianprops={'color': '#202020'}, whiskerprops={'color': '#777777'})
                offsets = np.linspace(-.13, .13, len(rows))
                ax.scatter(position + offsets, values, s=18,
                           color=COLORS[condition] if cohort == 'new' else '#777777',
                           marker='o' if cohort == 'old' else '^', alpha=.8, zorder=3)
            ax.set_ylabel(f'Mean across 20 path errors (MPa)')
            ax.set_title(label)
            ax.grid(axis='y', alpha=.18)
    for ax in axes:
        ax.set_xticks(range(4), [f'{cohort}\n{condition}' for cohort, condition in groups])
    fig.suptitle('Initialization axis: one dot per final model, 20-path mean per model\n'
                 'Box: median/IQR; whiskers: 1.5 IQR. Complete models: ' + ', '.join(counts))
    _save(fig, compact, 'all-initialization-distributions')


def _paired(compact, pairs):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), layout='constrained')
    metrics = [('RMSE_MPa', 'Stress RMSE'), ('hydrostatic_RMSE_MPa', 'Hydrostatic RMSE'),
               ('deviatoric_component_RMSE_MPa', 'Deviatoric component RMSE')]
    for row, condition in enumerate(('ideal', 'noisy')):
        selected = sorted((p for p in pairs if p['condition'] == condition and p['paired_complete']),
                          key=lambda p: p['path_id'])
        for ax, (metric, label) in zip(axes[row], metrics):
            path_ids = np.array([p['path_id'] for p in selected])
            improvement = np.array([p[f'old_{metric}'] - p[f'new_{metric}'] for p in selected])
            ax.axhline(0, color='#555555', linewidth=.9)
            ax.scatter(path_ids, improvement, color=COLORS[condition],
                       marker='o' if condition == 'ideal' else '^', s=27, zorder=3)
            for offset, index in enumerate(np.flatnonzero(improvement < 0)):
                ax.annotate(f'{improvement[index]:.4f}', (path_ids[index], improvement[index]),
                            xytext=(0, 10 + 12 * (offset % 2)), textcoords='offset points',
                            ha='center', va='bottom', fontsize=7, color=COLORS[condition])
            ax.set_xlim(-.7, 19.7)
            ax.set_xticks(range(20))
            ax.tick_params(axis='x', labelrotation=90, labelsize=7)
            ax.margins(y=.18)
            ax.set_xlabel('Fixed held-out path ID')
            ax.set_ylabel('Improvement: old − new (MPa)')
            ax.set_title(f'{condition}: {label}\n{len(selected)}/20 complete pairs')
            ax.grid(alpha=.18)
    fig.suptitle('Path axis: 20 paired loading histories, selected using training loss only\n'
                 'Positive improvement means lower new-model error; negative values are labeled')
    _save(fig, compact, 'selected-paired-path-errors')


def _heatmaps(config, compact, records):
    seeds = config['training']['initialization_seeds']
    npaths = config['data']['evaluation']['paths']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    maximum = max((r['RMSE_MPa'] for r in records if r['integration_complete']), default=1.)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#cccccc')
    for row, condition in enumerate(('ideal', 'noisy')):
        for column, cohort in enumerate(('old', 'new')):
            ax = axes[row, column]
            data = np.full((len(seeds), npaths), np.nan)
            for record in records:
                if record['condition'] == condition and record['cohort'] == cohort:
                    data[seeds.index(record['seed']), record['path_id']] = record['RMSE_MPa']
            image = ax.imshow(data, cmap=cmap, vmin=0, vmax=maximum, aspect='auto', interpolation='nearest')
            failed = np.argwhere(~np.isfinite(data))
            if len(failed):
                ax.scatter(failed[:, 1], failed[:, 0], marker='x', color='#202020', s=15)
            ax.set_title(f'{cohort} {condition}: {np.isfinite(data).sum()}/{data.size} complete')
            ax.set_xlabel('Fixed held-out path ID')
            ax.set_ylabel('Initialization seed')
            ax.set_xticks(range(0, npaths, 2))
            ax.set_yticks(range(0, len(seeds), 4), seeds[::4])
    fig.colorbar(image, ax=axes.ravel().tolist(), label='Full-path stress RMSE (MPa)', shrink=.83)
    fig.suptitle('All 2,000 model–path results; shared physical error scale\n'
                 'Gray ×: integration failure retained; physical conditions remain 20 histories')
    _save(fig, compact, 'all-model-path-rmse')


def figures(config, out, compact, records, summaries, pairs, known, training):
    _style()
    for condition in ('ideal', 'noisy'):
        _correlation(out, compact, condition)
    _responses(out, compact, config, 'heldout')
    _responses(out, compact, config, 'known')
    _training(out, compact, training)
    _initializations(compact, summaries)
    _paired(compact, pairs)
    _heatmaps(config, compact, records)
