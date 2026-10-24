"""Source-backed S0 time histories, task consequences, and complete animation."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
import numpy as np

from .evaluate import write_json
from .s0 import catalog, collect

COLORS = ['#0072B2', '#D55E00']
LABELS = {'reference_continuous': 'Full oracle', 'reference_reset_planning': 'Reset',
          'A_random100--ideal': 'A ideal', 'B_random1000--ideal': 'B ideal',
          'B_random1000--noise0': 'B noise 0', 'B_random1000--noise1': 'B noise 1',
          'B_random1000--noise2': 'B noise 2', 'myopic_oracle': 'Myopic',
          'retrospective_best_constant': 'Best constant*'}


def policies_from_csv(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for key in ('context_id', 'selected_id'):
            r[key] = int(r[key]) if r[key] else None
        for key in ('gamma', 'wait_budget_s', 'actual_peak_m', 'actual_time_s',
                    'actual_positive_work_J', 'actual_compression_regret_m'):
            r[key] = float(r[key]) if r[key] else None
        r['actual_task_failure'] = r['actual_task_failure'] == 'True'
    return rows


def load_trace(out, policy):
    with np.load(Path(out) / policy['actual_trajectory']) as saved:
        return {k: saved[k] for k in saved.files}


def save_figure(fig, directory, name):
    fig.canvas.draw()
    fig.savefig(directory / (name + '.png'), dpi=180)
    fig.savefig(directory / (name + '.pdf'))
    # Layout observations are printed, not a new audit/status document.
    from matplotlib.text import Text
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text().strip():
            box = item.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1+1 or box.y1 > fig.bbox.y1+1:
                outside.append(item.get_text())
    print(json.dumps(dict(figure=name, text_outside_canvas=outside)), flush=True)
    plt.close(fig)


def representative(config, out, compact, context, lookup):
    strategies = ['reference_continuous', 'myopic_oracle', 'retrospective_best_constant']
    fig, axes = plt.subplots(5, 3, figsize=(12, 10), sharex='col', sharey='row', layout='constrained')
    for column, strategy in enumerate(strategies):
        row = lookup[strategy, context['context_id'], 1.5]
        trace = load_trace(out, row)
        meta = json.loads(str(trace['metadata_json']))
        t = trace['time']
        for pad in range(2):
            style = '-' if pad == 0 else '--'
            axes[0, column].plot(t, trace['command_force_N'][:, pad], style, color=COLORS[pad], label='PQ'[pad])
            axes[1, column].plot(t, trace['compression_m'][:, pad]*1000, style, color=COLORS[pad])
            axes[2, column].plot(t, trace['q'][:, pad, 0]*1000, style, color=COLORS[pad])
            axes[3, column].plot(t, trace['foot_position_xz_m'][:, pad, 0]*1000, style,
                                 color=COLORS[pad], label=['Front', 'Hind'][pad])
            axes[4, column].plot(t, trace['foot_position_xz_m'][:, pad, 1]*1000, style, color=COLORS[pad])
        a = meta['action']
        axes[0, column].set_title(f"{LABELS[strategy]}: ID {row['selected_id']}\n{a['pad']}, {a['waveform']}, wait {a['wait_s']:g} s\n"
                                 f"T={row['actual_time_s']:.2f} s; peak={row['actual_peak_m']*1000:.5f} mm\nW+={row['actual_positive_work_J']:.6f} J", fontsize=9)
        axes[0, column].legend(ncol=2, loc='upper right', fontsize=8)
        axes[3, column].legend(ncol=2, loc='upper right', fontsize=8)
        for ax in axes[:, column]:
            ax.axvspan(0, meta['decision_s'], color='#dddddd', alpha=0.35)
            ax.axvspan(meta['finish_s'], t[-1], color='#dddddd', alpha=0.35)
            for ev in meta['events'][1:]:
                ax.axvline(ev['start_s'], color='#888888', lw=0.5, ls=':')
            ax.grid(alpha=0.18)
            ax.set_xlim(0, t[-1])
        axes[4, column].set_xlabel('Global time (s)')
    for ax, label in zip(axes[:, 0], ['Normal load (N)', 'Compression (mm)', 'Internal q11 (10^-3)',
                                     'Foot lateral x (mm)', 'Foot height z (mm)']):
        ax.set_ylabel(label)
    gamma_note = 'original scale' if context['gamma'] == 1 else 'synthetic scale'
    fig.suptitle(f"Complete frame-supported task | gamma={context['gamma']:g} ({gamma_note}) | prefix {context['prefix_pad']}\n"
                 'Prefix gap 0.1 s; hind P 50 N; return Q 150 N; return gap 0.5 s; wait budget 1.5 s\n'
                 'Shaded past prefix and final recovery are excluded from objectives. *Retrospective single constant across 36 contexts.', fontsize=10)
    name = f"complete-context-{context['context_id']:02d}"
    save_figure(fig, compact / 'figures', name)
    return name


def animate(config, out, compact, context, policy):
    """Render every phase from t=0 through the final two seconds of recovery."""
    trace = load_trace(out, policy)
    meta = json.loads(str(trace['metadata_json']))
    t = trace['time']
    magnification = 20
    # Only downward deformation is magnified; positive flight height stays actual.
    def display_z(z):
        return np.where(z < 0, z * magnification, z) * 1000
    fig = plt.figure(figsize=(12, 7.2), layout='constrained')
    grid = fig.add_gridspec(3, 2, width_ratios=(1.2, 1))
    scene = fig.add_subplot(grid[:2, 0])
    info_ax = fig.add_subplot(grid[2, 0])
    axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]
    info_ax.axis('off')
    scene.set_xlim(-28, 128)
    scene.set_ylim(-20, 32)
    scene.set_xlabel('World lateral position x (mm)')
    scene.set_ylabel('Displayed vertical position (mm)')
    scene.axhline(0, color='0.55', lw=0.8, ls='--')
    scene.plot([-22, 122], [28, 28], color='#666666', lw=5)
    scene.text(50, 30, 'External load-bearing frame', ha='center', fontsize=10)
    width_mm = np.sqrt(config['physical_scope']['area_m2'])*1000
    pads, heads, stems = [], [], []
    for i, center in enumerate((0, 100)):
        block = Rectangle((center-width_mm/2, -18), width_mm, 18, color=COLORS[i], alpha=0.45)
        scene.add_patch(block)
        pads.append(block)
        scene.text(center, -19.3, 'PQ'[i], ha='center', va='center')
        head = Rectangle((center-8, 0), 16, 1.2, facecolor=COLORS[i], edgecolor='black')
        scene.add_patch(head)
        heads.append(head)
        stems.append(scene.plot([center, center], [1.2, 28], color=COLORS[i], lw=2, ls='-' if i == 0 else '--')[0])
    lines = []
    series = [trace['command_force_N'], trace['compression_m']*1000,
              np.r_[np.zeros((1, 2)), np.cumsum(trace['positive_contact_work_increment_J'], axis=0)]]
    labels = ['Actual load (N)', 'Actual compression (mm)', 'Cumulative positive work (J)']
    cursors = []
    for axis, values, label in zip(axes, series, labels):
        for pad in range(2):
            axis.plot(t, values[:, pad], color=COLORS[pad], alpha=0.18, lw=1)
            lines.append(axis.plot([], [], color=COLORS[pad], ls='-' if pad == 0 else '--', label='PQ'[pad])[0])
        axis.set_xlim(t[0], t[-1])
        axis.set_ylim(min(0, np.min(values))*1.05, max(np.max(values)*1.12, 1e-8))
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        cursors.append(axis.axvline(0, color='black', lw=0.8))
    axes[0].legend(ncol=2, loc='upper right')
    axes[-1].set_xlabel('Global time (s); includes prefix and final recovery')
    text = info_ax.text(0, 0.95, '', va='top', fontsize=10, family='monospace')
    fig.suptitle('S0 complete contact task: frame-supported plungers, not quadruped locomotion\n'
                 f"gamma=0.025 is synthetic | downward deformation x{magnification}; flight height x1 | all readouts in actual units", fontsize=12)
    video = Path(out) / 'complete-task-gamma0025.mp4'
    writer = FFMpegWriter(fps=25, codec='libx264', bitrate=2200,
                         extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    frames = np.unique(np.r_[np.arange(0, len(t), 4), len(t)-1])
    with writer.saving(fig, str(video), dpi=120):
        for i in frames:
            for pad in range(2):
                pads[pad].set_height(18 + display_z(-trace['compression_m'][i, pad]))
            for foot in range(2):
                x = trace['foot_position_xz_m'][i, foot, 0]*1000
                z = float(display_z(trace['foot_position_xz_m'][i, foot, 1]))
                heads[foot].set_xy((x-8, z))
                stems[foot].set_data([x, x], [z+1.2, 28])
            for j, values in enumerate(series):
                for pad in range(2):
                    lines[2*j+pad].set_data(t[:i+1], values[:i+1, pad])
                cursors[j].set_xdata([t[i], t[i]])
            values = trace['compression_m'][i]*1000
            forces = trace['command_force_N'][i]
            text.set_text(f"t={t[i]:5.2f} s | {trace['phase'][i]}\n"
                          f"P: {forces[0]:6.1f} N   delta={values[0]:.5f} mm\n"
                          f"Q: {forces[1]:6.1f} N   delta={values[1]:.5f} mm\n"
                          f"Front=solid; hind=dashed stems\n"
                          f"Full oracle ID {policy['selected_id']}; remaining T={policy['actual_time_s']:.2f} s\n"
                          f"Remaining peak={policy['actual_peak_m']*1000:.5f} mm; W+={policy['actual_positive_work_J']:.6f} J")
            writer.grab_frame()
            if i in (frames[0], frames[len(frames)//2], frames[-1]):
                fig.savefig(compact / 'figures' / f'animation-frame-{i:04d}.png', dpi=120)
    plt.close(fig)
    return dict(path=str(video), context_id=context['context_id'], candidate_id=policy['selected_id'],
                full_global_time_s=float(t[-1]), frames=len(frames), fps=25,
                deformation_magnification=magnification, flight_height_magnification=1,
                display='Downward displacement below virgin surface is magnified x20; actual mm/N/s are shown at every frame.')


def figures(config, out, compact):
    directory = compact / 'figures'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    contexts, actions = catalog(config)
    policies = policies_from_csv(compact / 'policies.csv')
    lookup = {(r['policy'], r['context_id'], r['wait_budget_s']): r for r in policies}
    representatives = [c for c in contexts if c['prefix_gap_s'] == 0.1 and c['hind_N'] == 50
                       and c['return_N'] == 150 and c['return_gap_s'] == 0.5]
    names = [representative(config, out, compact, c, lookup) for c in representatives]
    models = list(LABELS)
    budgets = config['evaluation']['extra_wait_budgets_s']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), layout='constrained', sharex=True)
    for i, gamma in enumerate((1, 0.025)):
        for j, budget in enumerate(budgets):
            ax = axes[i, j]
            for k, model in enumerate(models):
                group = [r for r in policies if r['gamma'] == gamma and r['wait_budget_s'] == budget and r['policy'] == model]
                values = np.array([r['actual_compression_regret_m']*1000 for r in group if not r['actual_task_failure']])
                jitter = np.linspace(-0.18, 0.18, len(values))
                ax.scatter(k+jitter, values, s=13, alpha=0.7, c=COLORS[i], marker='o' if i == 0 else '^')
            ax.axhline(config['evaluation']['compression_tie_tolerance_m']*1000, color='#555555', ls=':', label='Declared tie tolerance')
            ax.set_xticks(range(len(models)), [LABELS[m] for m in models], rotation=55, ha='right')
            ax.set_title(f'gamma={gamma:g} | extra-wait budget {budget:g} s')
            ax.set_ylabel('True peak-compression regret (mm)')
            ax.grid(axis='y', alpha=0.2)
            ax.set_ylim(bottom=-0.000002)
    axes[0, 0].legend(loc='upper left')
    fig.suptitle('Every policy is evaluated in the continuous reference plant\n36 deterministic contexts per panel; regret to the strict feasible minimum. *One retrospective constant ID per panel.', fontsize=11)
    save_figure(fig, directory, 'actual-policy-regret')
    names.append('actual-policy-regret')

    candidates = collect(config, out)
    # Pareto visualization retains all twelve actions and encodes the third axis by color.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, gamma in zip(axes, (1, 0.025)):
        ctx = next(c for c in representatives if c['gamma'] == gamma and c['prefix_pad'] == 'P')
        group = [r for r in candidates if r['predictor'] == 'reference_continuous' and r['context_id'] == ctx['context_id']]
        times = [r['completion_time_s'] for r in group]
        peaks = [r['max_remaining_pad_compression_m']*1000 for r in group]
        works = [r['positive_remaining_contact_work_J'] for r in group]
        scatter = ax.scatter(times, peaks, c=works, cmap='viridis', s=70, edgecolors='black', linewidths=0.4)
        fig.colorbar(scatter, ax=ax, label='Positive contact work (J)', shrink=0.8)
        ax.set_xlabel('Remaining completion time (s)')
        ax.set_ylabel('Maximum remaining compression (mm)')
        ax.set_title(f'gamma={gamma:g}; context {ctx["context_id"]}')
        ax.grid(alpha=0.2)
    fig.suptitle('All twelve fixed actions: time, compression and work\nPrefix P, gap 0.1 s; future loads 50/150 N; return gap 0.5 s. Exact Pareto membership: candidates.csv.', fontsize=10)
    save_figure(fig, directory, 'candidate-tradeoffs')
    names.append('candidate-tradeoffs')
    animation_context = next(c for c in representatives if c['gamma'] == 0.025 and c['prefix_pad'] == 'P')
    video_path = Path(out) / 'complete-task-gamma0025.mp4'
    if not video_path.exists():
        animation = animate(config, out, compact, animation_context,
                            lookup['reference_continuous', animation_context['context_id'], 1.5])
    else:
        animation = dict(path=str(video_path), reused_existing_animation=True)
    index = json.loads((compact / 'index.json').read_text())
    index['figures'] = dict(stems=names, animation=animation,
                           representative_context_ids=[c['context_id'] for c in representatives])
    write_json(compact / 'index.json', index)
    print(json.dumps(index['figures']), flush=True)
