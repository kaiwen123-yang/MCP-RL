#!/usr/bin/env python3
"""Inspect saved W39 replay inputs; no dynamics, optimization or GPU execution."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = json.loads(args.index.read_text())
    results = {}
    for branch in ('a', 'b'):
        rows = [r for r in records if r['branch'] == branch and 'E30' in r['attempt']]
        def file(name):
            return Path(next(r for r in rows if r['name'] == name)['parts'][0]['local_path'])
        resolved_row = next(r for r in rows if r['name'].endswith('-resolved.json'))
        resolved = json.loads(gzip.decompress(Path(resolved_row['parts'][0]['local_path']).read_bytes()))
        with np.load(file('trajectory.npz')) as a, np.load(file('prediction-000.npz')) as b:
            live = a['time'] == a['diagnostic_time']
            t = a['time'][live]
            origins = b['plan_origin']
            at = np.searchsorted(t, origins)
            at = np.minimum(at, len(t)-1)
            left = np.maximum(at-1, 0)
            at = np.where(abs(t[left]-origins) < abs(t[at]-origins), left, at)
            record = {
                'attempt': rows[0]['attempt'],
                'model_binary_sha256': hashlib.sha256(file('model.mjb').read_bytes()).hexdigest(),
                'model_hash_matches_resolved': hashlib.sha256(file('model.mjb').read_bytes()).hexdigest() == resolved['model_mjb_sha256'],
                'valid_control_rows': int(live.sum()),
                'terminal_rows_with_copied_diagnostics': int((~live).sum()),
                'feedback_reference_plans': int(len(origins)),
                'saved_reference_shape': list(b['plan_states'].shape),
                'saved_reference_action_shape': list(b['plan_actions'].shape),
                'reference_timestep_median_s': float(np.median(np.diff(b['plan_times'], axis=1))),
                'reference_horizon_min_s': float(np.min(b['plan_times'][:,-1] - b['plan_times'][:,0])),
                'reference_horizon_max_s': float(np.max(b['plan_times'][:,-1] - b['plan_times'][:,0])),
                'max_reference_origin_to_control_clock_s': float(np.max(abs(t[at]-origins))),
                'finite_userdata_all_control_rows': bool(np.isfinite(a['support_reference_userdata'][live]).all()),
                'userdata_shape': list(a['support_reference_userdata'].shape),
                'mocap_pos_shape': list(a['mocap_pos'].shape),
                'mocap_quat_shape': list(a['mocap_quat'].shape),
                'physics_dt_s': resolved['physics_dt_s'],
                'feedback_dt_s': resolved['logical_feedback_dt_s'],
                'planning_dt_s': resolved['logical_planning_dt_s'],
                'predictor_numerics': resolved['prediction_settings']['numerics'],
                'task_setting_changes': resolved['applied_settings'],
                'source_runtime_sha256': resolved['source_sha256']['scripts/s1_go2_r11_runtime.py'],
                'original_planner_private_state_not_saved': ['complete accepted rollout', 'complete feedback K horizon', 'backward H/C/multipliers', 'warm-start policy buffers', 'task residual private phase/current gait state'],
            }
        results[branch] = record
    out = {
        'source_commit': '07b0ee3e15c9f10e9b1a6a1de3997bec36f39556',
        'scope': 'saved-input inventory for prospective GPU tests; no tests were run here',
        'same_E30_model_binary': results['a']['model_binary_sha256'] == results['b']['model_binary_sha256'],
        'streams': results,
        'replay_contract': {
            'usable_now': 'q/v, time, original target, mocap, all96 userdata, resolved actuator limits/model and applied settings; supports explicitly reconstructed matched CPU/GPU fixtures',
            'reference_role': 'prediction NPZ is policy feedback reference, not accepted optimizer rollout',
            'backend_model': 'reproduce predictor MakeDifferentiable, coldstart, custom callbacks/patches and10 substeps; model.mjb alone does not install these callbacks',
            'task_phase': 'reconstruct and disclose transition history or serialize residual task state for new fixtures; q/v alone is insufficient for a cost/planner comparison',
            'whole_planner': 'choose and record identical initial policy/warm-start and task state for matched CPU/GPU calls; do not claim exact historical W39 replay',
            'version': 'r28 CPU versus r28 GPU, or corrected r29 CPU versus corrected r29 GPU; never compare old Qux proposal to revised metric projection',
            'plant_boundary': 'offline material truth used only to diagnose existing data; do not feed hidden material state into predictor/actor',
        },
    }
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k != 'streams'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
