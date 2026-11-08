#!/usr/bin/env python3
"""Summarize selected original W41 arrays and plot the two final CPU E30s.

Usage: python analyze-raw-arrays.py --raw-root /path/to/read-selected-raw/output
No physics or policy inference is rerun. Plot smoothing is display-only.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def trajectory_summary(z):
    t = z['time']
    out = dict(samples=len(t), terminal_s=float(t[-1]),
               max_base_x_m=float(z['qpos'][:, 0].max()),
               final_base_xyz_m=z['qpos'][-1, :3].tolist(), windows={})
    for lo, hi in [(1, 8), (9, 11), (11, 13), (15, 20), (30, 60), (40, 60)]:
        m = (t >= lo) & (t <= hi)
        if m.sum() < 2:
            continue
        tt, q = t[m], z['qpos'][m]
        elapsed = float(tt[-1] - tt[0])
        dx = float(q[-1, 0] - q[0, 0])
        out['windows'][f'{lo}-{hi}s'] = dict(
            sampled_elapsed_s=elapsed, dx_m=dx, net_speed_m_s=dx / elapsed,
            mean_command_vx_m_s=float(z['command_velocity'][m, 0].mean()),
            four_contact_fraction=float(np.all(z['contact'][m], axis=1).mean()),
            max_foot_x_m=float(z['feet'][m, :, 0].max()),
            median_requested_cap_m_s=float(np.median(z['requested_speed_cap'][m])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema='W41_selected_raw_arrays_v1',
                  source='Original release members listed in raw-*-read-coverage.json',
                  scope='16 complete training rollouts, two final E30s and B344 trajectory; no physics or actor rerun.',
                  streams={}, trajectories={})
    for stream in ['w41-a', 'w41-b']:
        files = sorted((args.raw_root / stream / 'training').glob('rollout-*.npz'))
        arrays = [np.load(p) for p in files]
        obs = np.concatenate([z['obs'].reshape(-1, z['obs'].shape[-1]) for z in arrays])
        row = dict(rollout_files=[str(p.relative_to(args.raw_root)) for p in files],
                   transitions=len(obs), obs_dim=obs.shape[1],
                   all_arrays_finite=all(np.isfinite(z[k]).all() for z in arrays for k in z.files),
                   terminated=sum(int(z['terminated'].sum()) for z in arrays),
                   truncated=sum(int(z['truncated'].sum()) for z in arrays),
                   current_pann_nonzero=int(np.any(obs[:, -12:] != 0, axis=1).sum()),
                   max_abs_pann_m=float(np.max(abs(obs[:, -12:] * .001))),
                   max_abs_observation=float(np.max(abs(obs))))
        evaldir = args.raw_root / stream / 'evaluation-E30'
        dependence = json.loads((evaldir / 'pann-action-dependence.json').read_text())
        delta = np.array([x['with_pann_cap'] - x['zero_feature_cap'] for x in dependence])
        exposure = [json.loads(line) for line in (evaldir / 'query-exposure.jsonl').read_text().splitlines()]
        row['evaluation_pann'] = dict(actor_decisions=len(dependence),
            immediate_zero_feature_action_changed=int(np.count_nonzero(delta)),
            max_abs_cap_difference_m_s=float(abs(delta).max()),
            query_rows_including_terminal=len(exposure),
            query_inside_any_foot_rows=sum(any(x['inside']) for x in exposure),
            query_nonzero_any_foot_rows=sum(any(x['feature_nonzero']) for x in exposure),
            interpretation='Same-state immediate actor comparison only; not a closed-loop causal memory ablation.')
        report['streams'][stream] = row
    for stream, case in [('w41-a', 'evaluation-E30'), ('w41-b', 'evaluation-E30'), ('w41-b', 'ep00344')]:
        with np.load(args.raw_root / stream / case / 'trajectory.npz') as z:
            report['trajectories'][f'{stream}/{case}'] = trajectory_summary(z)
    (HERE / 'raw-array-summary.json').write_text(json.dumps(report, indent=2) + '\n')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(3, 2, figsize=(12, 8.5), sharex='col', constrained_layout=True)
    for col, stream in enumerate(['w41-a', 'w41-b']):
        path = args.raw_root / stream / 'evaluation-E30'
        z = np.load(path / 'trajectory.npz'); t = z['time']
        manifest = json.loads((path / 'r29-event-manifest.json').read_text())
        load = next(x['time_s'] for x in manifest['events'] if x['kind'] == 'first_material_load')
        axs[0, col].plot(t, z['qpos'][:, 0], label='Actual base x', color='#176B99', lw=1.8)
        axs[0, col].axhspan(.8, 2.2, color='#E1B86A', alpha=.15, label='Material strip (base criterion)')
        axs[0, col].axhline(.8, color='#946528', ls=':', lw=1)
        axs[0, col].axhline(2.2, color='#946528', ls=':', lw=1)
        axs[0, col].set_ylim(-.1, 2.35)
        axs[0, col].set_title(f'{stream.upper()} final E30 | fall at {t[-1]:.3f} s', fontweight='bold')
        axs[0, col].set_ylabel('Base position x (m)')
        axs[1, col].plot(t, z['command_velocity'][:, 0], color='#DC7936', label='Command vx', lw=1.4)
        n = 251
        smooth = np.convolve(z['qvel'][:, 0], np.ones(n) / n, mode='valid')
        axs[1, col].plot(t[n // 2:len(t) - n // 2], smooth, color='#176B99', label='Actual vx (0.5 s mean)', lw=1.2)
        axs[1, col].axhline(0, color='#667078', lw=.7)
        axs[1, col].set_ylabel('Longitudinal speed (m/s)')
        axs[2, col].step(t, z['contact'].sum(axis=1), where='post', color='#477A55', lw=.85)
        axs[2, col].set_yticks(range(5)); axs[2, col].set_ylim(-.15, 4.25)
        axs[2, col].set_ylabel('Feet in contact (0-4)'); axs[2, col].set_xlabel('Simulation time (s)')
        for ax in axs[:, col]:
            ax.axvline(load, color='#8A547A', ls='--', lw=1, label='First material load')
            ax.grid(axis='y', alpha=.18); ax.set_xlim(0, t[-1])
        axs[0, col].legend(loc='upper left', fontsize=8)
        axs[1, col].legend(loc='lower left', fontsize=8)
    fig.suptitle('W41 CPU: completed training, persistent locomotion and contact failure', fontsize=14, fontweight='bold')
    fig.savefig(HERE / 'cpu-trajectory-overview.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(output=str(HERE / 'raw-array-summary.json'),
                          plot=str(HERE / 'cpu-trajectory-overview.png'))))


if __name__ == '__main__':
    main()
