#!/usr/bin/env python3
"""Independent W53 F saved-array analysis; no model, solver or physics calls."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--originals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.originals / 'w53-f'
    tr = np.load(args.originals / 'trajectory.npz')
    parts = [np.load(p) for p in sorted(folder.glob('control-*.npz'))]
    c = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    t = c['time']
    n = len(t)
    mass = float(np.load(folder / 'mpc-000001.npz')['mass'])
    first_swing = (t >= 2.052 - 1e-8) & (t < 2.254 - 1e-8)
    failure = json.loads((folder / 'solver-failure.json').read_text())
    prefix = f"failed-mpc-{failure['time_s']:.6f}"
    A = sparse.load_npz(folder / (prefix + '-A.npz')).toarray()
    bounds = np.load(folder / (prefix + '.npz'))
    lo, hi = bounds['lower'], bounds['upper']
    contradictions = []
    for k in range(14):
        fs = slice(180 + 12*k, 180 + 12*(k+1))
        row = 12 + 52*k
        zero = (lo[row+40:row+52] == 0) & (hi[row+40:row+52] == 0)
        block = A[row+28:row+40, fs].copy()
        block[:, zero] = 0
        for j in range(12):
            rr = row + 28 + j
            if np.linalg.norm(block[j]) == 0 and (lo[rr] > 0 or hi[rr] < 0):
                contradictions.append(dict(stage=k, joint_index=j,
                    original_matrix_row=rr, remaining_row_norm=0.,
                    lower_Nm=float(lo[rr]), upper_Nm=float(hi[rr])))

    touchdowns = []
    for leg, name in enumerate(('FL', 'FR', 'RL', 'RR')):
        starts = np.flatnonzero(~c['planned_contact'][:-1, leg] & c['planned_contact'][1:, leg]) + 1
        for i in starts:
            later = np.flatnonzero(c['observed_contact'][i:, leg])
            actual = None if not len(later) else i + later[0]
            same_stance_end = np.flatnonzero(~c['planned_contact'][i:, leg])
            end = n if not len(same_stance_end) else i + same_stance_end[0]
            anchor = c['feet_position_reference'][i, leg]
            row = dict(leg=name, scheduled_touchdown_s=float(t[i]),
                actual_clearance_at_scheduled_touchdown_mm=float(1000*(tr['feet'][i, leg, 2] - .022)),
                frozen_stance_anchor_clearance_mm=float(1000*(anchor[2] - .022)),
                observed_touchdown_s=None if actual is None else float(t[actual]),
                delay_s=None if actual is None else float(t[actual] - t[i]),
                max_anchor_change_during_same_stance_m=float(np.max(abs(c['feet_position_reference'][i:end, leg] - anchor))))
            if actual is not None and actual < end:
                pos = tr['feet'][actual, leg]
                err = anchor - pos
                row.update(anchor_at_observed_touchdown_clearance_mm=float(1000*(c['feet_position_reference'][actual, leg, 2] - .022)),
                    actual_clearance_at_observed_touchdown_mm=float(1000*(pos[2] - .022)),
                    position_feedback_vertical_acceleration_m_s2=float(400*err[2]),
                    full_saved_vertical_acceleration_reference_m_s2=float(c['feet_acceleration_reference'][actual, leg, 2]),
                    observed_Fz_N=float(c['measured_grf'][actual, leg, 2]))
            touchdowns.append(row)

    masking = []
    for tt in (2.260, 2.280, 2.300, 2.302, 2.320):
        i = int(np.argmin(abs(t-tt)))
        removed = np.sum(c['mpc_force'][i][~c['active_wbc_contact'][i]], axis=0)/mass
        masking.append(dict(time_s=float(t[i]),
            planned_contact=c['planned_contact'][i].tolist(),
            observed_contact=c['observed_contact'][i].tolist(),
            active_contact=c['active_wbc_contact'][i].tolist(),
            planned_Fz_N=c['mpc_force'][i, :, 2].tolist(),
            removed_upward_acceleration_m_s2=float(removed[2]),
            actual_body_acceleration_reference_z_m_s2=float(c['body_acceleration_reference'][i, 2]),
            algebraic_reference_without_force_mask_z_m_s2=float(c['body_acceleration_reference'][i, 2]+removed[2])))

    future_jumps = []
    for path in sorted(folder.glob('mpc-*.npz')):
        a = np.load(path)
        contact, feet, times = a['contact_schedule'], a['feet'], a['times']
        for leg, name in enumerate(('FL', 'FR', 'RL', 'RR')):
            if not contact[0, leg]:
                continue
            for k in np.flatnonzero(~contact[:-1, leg] & contact[1:, leg]) + 1:
                change = feet[k, leg] - feet[k-1, leg]
                future_jumps.append(dict(file=path.name, plan_time_s=float(times[0]),
                    leg=name, later_node=int(k), earlier_time_s=float(times[k-1]), later_time_s=float(times[k]),
                    change_m=change.tolist(), xy_change_norm_m=float(np.linalg.norm(change[:2])),
                    before_m=feet[k-1, leg].tolist(), after_m=feet[k, leg].tolist()))

    valid_dynamics = []
    for path in sorted(folder.glob('mpc-*.npz'))[-4:]:
        a = np.load(path)
        ti = float(a['times'][0])
        i = int(np.argmin(abs(t-ti)))
        j = 11  # FR calf generalized-velocity index (actuator index 5).
        inertia = float(a['M'][j] @ c['qacc'][i])
        reaction = float(a['J'].reshape(12, 18)[:, j] @ c['force'][i].ravel())
        valid_dynamics.append(dict(time_s=ti, h_FR_calf_Nm=float(a['h'][j]),
            inertia_term_FR_calf_Nm=inertia, reaction_term_FR_calf_Nm=reaction,
            saved_tau_FR_calf_Nm=float(c['tau_total'][i, 5]),
            reconstructed_tau_FR_calf_Nm=float(a['h'][j]+inertia-reaction),
            actual_lower_Nm=float(c['torque_lower'][i, 5]), actual_upper_Nm=float(c['torque_upper'][i, 5])))

    intervals = []
    for begin, end in [(0, 2), (2.052, 2.254), (2.254, 2.302), (2.302, 2.332), (2.332, 2.552), (2.552, 3.12)]:
        mask = (t >= begin-1e-8) & (t < end-1e-8)
        force_means = {key: float(np.mean(np.sum(c[key][mask, :, 2], axis=1))) for key in ('mpc_force', 'force', 'measured_grf')}
        intervals.append(dict(begin_s=begin, end_s=end, samples=int(mask.sum()),
            start_body_z_m=float(tr['qpos'][:n][mask, 2][0]), end_body_z_m=float(tr['qpos'][:n][mask, 2][-1]),
            mean_total_Fz_N=force_means,
            peak_WBC_torque_limit_fraction=float(np.max(abs(c['tau_total'][mask])/np.tile([23.7, 23.7, 45.43], 4)))))

    answer = dict(schema='w54_independent_F_saved_array_audit_v1',
        scope='Full saved trajectory, all WBC controls, all 157 saved MPC plans and failed arrays; no physics, model calls, solver reruns, FD, or training.',
        sample_alignment_max_error_s=float(np.max(abs(t-tr['time'][:n]))),
        WBC_samples=n, MPC_files=len(list(folder.glob('mpc-*.npz'))),
        failed_matrix_shape=list(A.shape), contradictions=contradictions,
        touchdown_anchors=touchdowns, body_reference_force_mask=masking,
        future_unstarted_swing_touchdown_transitions=len(future_jumps),
        maximum_future_XY_jump=max(future_jumps, key=lambda r: r['xy_change_norm_m']),
        future_touchdown_jumps=future_jumps,
        first_swing_peak_WBC_torque_fraction=float(np.max(abs(c['tau_total'][first_swing])/np.tile([23.7,23.7,45.43],4))),
        max_initial_PD_mapping_error_Nm=float(np.max(abs(c['tau_total']-c['requested_pd']))),
        max_floating_dynamics_residual=float(np.max(abs(c['dynamics_residual']))),
        last_exact_model_WBC_dynamics=valid_dynamics,
        intervals=intervals,
        limits='Without-force-mask values are algebraic references, not simulated counterfactual outcomes. Future foot jumps are prediction artifacts, not actual foot slip. Removing static contradictions does not prove dynamic feasibility at failure state. Early swing tracking error itself is not fully causally isolated.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(answer, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), samples=n, plans=answer['MPC_files'],
        contradictions=len(contradictions), touchdown_events=len(touchdowns),
        future_transitions=len(future_jumps), max_future_jump_m=answer['maximum_future_XY_jump']['xy_change_norm_m'])))


if __name__ == '__main__':
    main()
