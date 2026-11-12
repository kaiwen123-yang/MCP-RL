#!/usr/bin/env python3
"""Recompute existing W45 originals. No simulation, training, or server calls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
RESULTS = ROOT / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'


def first_time(time, mask):
    indices = np.flatnonzero(mask)
    return float(time[indices[0]]) if len(indices) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / 'trajectory-originals.json').read_text())
    checks = []
    for row in manifest:
        path = args.raw_dir / row['local_name']
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != row['size'] or digest != row['sha256']:
            raise ValueError(f'Original digest/size mismatch: {row["local_name"]}')
        checks.append(dict(local_name=row['local_name'], size=len(raw), sha256=digest))
    summary = dict(snapshot='3e3f69c440d6a33c14131563e5f485d88da3a803',
                   verified_originals=checks, streams={}, physical_experiments_started=0)
    trajectories = {}
    for suffix, seed in [('a', 1), ('b', 2)]:
        stream = f'w45-{suffix}'
        result_path = RESULTS / f'job-20261112T103825Z-{stream}-eval-PANN_PPO_S5-s{seed}-E30-attempt01.json'
        result = json.loads(result_path.read_text())
        with np.load(args.raw_dir / f'{stream}-trajectory.npz', allow_pickle=False) as z:
            a = {key: z[key] for key in z.files}
        trajectories[stream] = a
        time, qpos, rpy = a['time'], a['qpos'], a['rpy']
        half = np.sqrt(.001) / 2
        feet = a['feet']
        inside = ((feet[:, :, 0] - half >= .8) & (feet[:, :, 0] + half <= 2.2)
                  & (np.abs(feet[:, :, 1]) + half <= .6))
        loaded = inside & (a['actual_grf'][:, :, 2] > 0)
        dependence = json.loads((args.raw_dir / stream / 'pann-action-dependence.json').read_text())
        caps = json.loads((args.raw_dir / stream / 'cap-decisions.json').read_text())
        delta = [abs(x['with_pann_cap'] - x['zero_feature_cap']) for x in dependence]
        row = dict(
            source_result=str(result_path.relative_to(ROOT)),
            trajectory_nodes=len(time), arrays={k: list(v.shape) for k, v in a.items()},
            numeric_arrays_finite=all(np.isfinite(v).all() for v in a.values()),
            simulation_s=float(time[-1]), wall_s=result['wall_s'],
            wall_per_simulation_s=result['wall_s'] / float(time[-1]),
            reason=result['reason'], final_xyz_m=qpos[-1, :3].tolist(),
            final_rpy_rad=rpy[-1].tolist(), max_body_x_m=float(qpos[:, 0].max()),
            minimum_body_height_m=float(qpos[:, 2].min()),
            first_body_strip_entry_s=first_time(time, qpos[:, 0] >= .8),
            first_body_strip_east_exit_s=first_time(time, qpos[:, 0] >= 2.2),
            first_abs_roll_above_0_5_s=first_time(time, abs(rpy[:, 0]) > .5),
            tilt_threshold_note='0.5 rad is descriptive only; original fall threshold remains 1.2 rad.',
            first_full_foot_inside_positive_Fz_500Hz_s=first_time(time, loaded.any(axis=1)),
            loaded_500Hz_nodes=int(loaded.any(axis=1).sum()),
            loaded_500Hz_nodes_by_leg=loaded.sum(axis=0).tolist(),
            first_no_foot_contact_after_stand_500Hz_s=first_time(time, (time > 2) & ~a['contact'].any(axis=1)),
            flight_note='No-foot contact is diagnostic, not an added physical failure condition.',
            original_legal_route=result['legal_route'],
            material_max_strain=result['material']['max_abs_principal_strain'],
            material_domain_exit=result['material']['domain_exit'],
            material_loaded_ever_cells=result['material']['loaded_ever_cells'],
            cap_decisions=len(caps), cap_min_m_s=min(x['requested_cap_m_s'] for x in caps),
            cap_max_m_s=max(x['requested_cap_m_s'] for x in caps),
            pann_action_dependence_records=len(dependence),
            nonzero_cap_change_when_zeroing_pann=sum(x != 0 for x in delta),
            max_cap_change_when_zeroing_pann=max(delta),
            last_actor_decision_s=caps[-1]['time_s'],
            planning=result['timing']['planning'], observer=result['policy']['observer'],
            samples=14737, updates=57, new_PPO_samples=0, new_PPO_updates=0,
        )
        if suffix == 'a':
            foot = np.concatenate([np.load(args.raw_dir / stream / f'material-{i:04d}.npz')['foot'] for i in [12, 13]])
            columns = result['material']['foot_columns']
            force = foot[:, [columns.index(f'force_{j}_z') for j in range(4)]]
            ft = foot[:, columns.index('boundary_time_s')]
            mask = (force > 0).any(axis=1)
            starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
            ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
            row['material_5000Hz_selected_last_chunks'] = dict(
                chunks=[12, 13], interval_s=[float(ft[0]), float(ft[-1])],
                first_positive_force_s=first_time(ft, mask),
                positive_force_rows=int(mask.sum()),
                represented_hold_duration_s=float(mask.sum() * .0002),
                max_force_by_leg_N=force.max(axis=0).tolist(),
                max_principal_strain=float(foot[:, columns.index('max_strain')].max()),
                positive_runs=[dict(first_boundary_s=float(ft[i]), last_boundary_s=float(ft[j]), samples=int(j-i+1)) for i, j in zip(starts, ends)],
                limitation='Only chunks 12 and 13 read directly; full episode loaded status also checked against trajectory/result. Boundary timestamps are pre-integration force times.',
            )
        assert abs(row['simulation_s'] - result['simulation_s']) < 1e-10
        assert abs(row['max_body_x_m'] - result['max_x_m']) < 1e-12
        assert not result['success'] and row['first_body_strip_east_exit_s'] is None
        assert all(result['legal_route'][k] == 0 for k in ['completed_roundtrips', 'legal_outbound_traversals', 'legal_return_traversals'])
        summary['streams'][stream] = row
    (args.output_dir / 'trajectory-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex='col', constrained_layout=True)
        for col, (stream, a) in enumerate(trajectories.items()):
            time = a['time']
            axes[0, col].plot(time, a['qpos'][:, 0], color='#206c91')
            axes[0, col].axhline(.8, color='#6d747a', linestyle='--', label='Material strip begins (body x)')
            axes[0, col].set_ylim(-.05, .91)
            axes[0, col].set_title(f'{stream.upper()}: fall at {time[-1]:.3f} s; 0 legal traversals')
            axes[0, col].set_ylabel('Body x (m)')
            axes[1, col].plot(time, a['rpy'][:, 0], color='#ab403a', label='Roll')
            axes[1, col].plot(time, a['rpy'][:, 1], color='#d58b26', label='Pitch')
            axes[1, col].axhline(1.2, color='#6d747a', linestyle='--', label='Original fall threshold')
            axes[1, col].set_ylim(-.7, 1.3)
            axes[1, col].set_ylabel('Orientation (rad)')
            axes[2, col].plot(time, a['contact'].sum(axis=1), color='#327c6a', linewidth=.65)
            axes[2, col].set_ylabel('Feet in contact')
            axes[2, col].set_yticks(range(5))
            axes[2, col].set_xlabel('Simulation time (s)')
            for ax in axes[:, col]:
                ax.axvspan(0, 2, color='#8999a5', alpha=.12)
                ax.grid(alpha=.16)
                if stream == 'w45-a':
                    ax.axvline(13.048, color='#633c91', linewidth=1, label='First material force (13.048 s)')
            axes[0, col].legend(fontsize=7, loc='upper left')
            axes[1, col].legend(fontsize=7, loc='upper left')
        fig.suptitle('W45 / fixed update 57: numerical acceptance repaired, task stability unresolved', fontsize=13)
        fig.savefig(args.output_dir / 'w45-final-trajectories.png', dpi=170)
        plt.close(fig)
    print(json.dumps({s: {k: r[k] for k in ['simulation_s', 'max_body_x_m', 'final_rpy_rad', 'nonzero_cap_change_when_zeroing_pann']} for s, r in summary['streams'].items()}, indent=2))


if __name__ == '__main__':
    main()
