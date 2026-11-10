"""Recompute the W43 CPU final report from already downloaded published data.

This program performs offline JSON/NPZ analysis and local SHA256 checks only.
It does not download assets, contact the server, run physics, or load checkpoint
weights. The release metadata input is the captured public GitHub response used
for the review, normalized to job -> {id,url,assets:[name,size,state,digest,url]}.

Required local names are a/b-e30-trajectory.npz, a/b-e30-planner-calls.jsonl.gz,
the corresponding a-ep00406..00410 / b-ep00394..00398 files, material-events and
material-failures JSONL gzip files for all twelve episodes, and a/b-training-
rollout-013714-013969.npz plus rollout-013970-014225.npz. See publication-audit.json
for source paths, asset names, URLs, sizes and the exact selected scope.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True, help='MCP-RL repository checkout')
parser.add_argument('--raw', type=Path, required=True, help='Directory containing the 52 reviewed raw assets')
parser.add_argument('--release-metadata', type=Path, required=True, help='Captured normalized release-assets JSON; no network fetch is performed')
parser.add_argument('--output', type=Path, help='Output directory; defaults to the repository W43 review directory')
args = parser.parse_args()
REPO = args.repo.resolve()
RAW = args.raw.resolve()
R = REPO / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
OUT = args.output or R / 'reviews/W43-final-review'
OUT.mkdir(parents=True, exist_ok=True)
remote = json.loads(args.release_metadata.read_text())
downloads = []

def plain(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))

def dist(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), min=float(a.min()), median=float(np.median(a)), p95=float(np.quantile(a, 0.95)), max=float(a.max())) if len(a) else dict(n=0)
summary = {'reviewed_commit': '6a94c94c8e21e3b50fcca40ab4363165dee2987e', 'scope': 'All 10 W43 training compact/resolved files, 2 final update55 E30 compact/resolved files, all 12 downloaded trajectory and planner-call originals, 1024 Git transition and PANN records and 4 downloaded complete rollout arrays. No new physics, server access or checkpoint tensor inspection.', 'streams': {}, 'episodes': []}
pub = {'reviewed_commit': summary['reviewed_commit'], 'metadata_scope': 'Two W43 job diagnostics Releases, every manifest name/state/size compared; not all repository history.', 'streams': {}, 'downloaded_and_sha256_verified': [], 'not_inspected': 'Full high-rate physical/material/prediction, full feedback/actions/planner-records and selected matrices are outside this CPU evidence subtask; solver audit has its own declared subset. Checkpoint tensors/full videos are not selected by the ordinary delivery policy.'}
for side, job in [('a', 'job-20261110T035658Z-w43-a'), ('b', 'job-20261110T035700Z-w43-b')]:
    J = R / 'jobs' / job
    j = json.loads((J / 'job.json').read_text())
    ev = json.loads((J / 'evaluation.json').read_text())
    col = json.loads((J / 'training/collection.json').read_text())
    learn = json.loads((J / 'training/learning.json').read_text())
    evidence = json.loads((J / 'diagnostics/evidence.json').read_text())
    manifest = json.loads((J / 'diagnostics/manifest.json').read_text())
    meta = remote[job]
    names = {a['name']: a for a in meta['assets']}
    local_assets = {a['name']: a for a in evidence['assets']}
    mismatch = []
    for f in evidence['files']:
        source = f['source_path']
        base = source.rsplit('/', 1)[-1]
        if base not in ['trajectory.npz', 'planner-calls.jsonl', 'material-events.jsonl', 'material-failures.jsonl'] and (not base.startswith('rollout-')):
            continue
        attempt = f.get('attempt') or ''
        label = 'e30' if '-eval-' in attempt else 'ep' + attempt.rsplit('-ep', 1)[-1] if '-ep' in attempt else 'training'
        suffix = base + ('.gz' if f['codec'] == 'gzip' else '')
        path = RAW / f'{side}-{label}-{suffix}'
        if len(f['parts']) != 1:
            raise ValueError('This W43 review expects the published single-part asset: ' + source)
        asset = names[f['parts'][0]['name']]
        with path.open('rb') as handle:
            sha = hashlib.file_digest(handle, 'sha256').hexdigest()
        if sha != asset['digest'].removeprefix('sha256:') or path.stat().st_size != asset['size']:
            raise ValueError('Published digest/size mismatch: ' + path.name)
        downloads.append({'job': job, 'attempt': attempt, 'source_path': source, 'asset': asset, 'sha256': sha, 'sha256_matches_remote': True, 'downloaded_bytes': path.stat().st_size})
    for f in manifest:
        a = names.get(f['name'])
        l = local_assets.get(f['name'])
        if not a or a['state'] != 'uploaded' or a['size'] != l['size']:
            mismatch.append({'name': f['name'], 'remote': a, 'expected_size': l['size'] if l else None})
    pub['streams'][side] = {'job': job, 'release_url': meta['url'], 'release_id': meta['id'], 'manifest_count': len(manifest), 'remote_asset_count': len(meta['assets']), 'remote_bytes': sum((a['size'] for a in meta['assets'])), 'manifest_comparison_mismatches': mismatch, 'evidence_status': evidence['status'], 'missing_sources': evidence['missing'], 'optional_absent': evidence['optional_absent'], 'historical_publication_errors': [{'action': x['action'], 'returncode': x['returncode'], 'detail': x['detail']} for x in evidence['errors']], 'contains_checkpoint_asset': any(('.pt' in a['name'] for a in meta['assets'])), 'source': 'jobs/' + job + '/diagnostics/evidence.json'}
    ts = []
    ps = []
    for f in sorted((J / 'numeric-evidence/training').glob('*transitions.jsonl')):
        ts.extend((json.loads(l) for l in f.read_text().splitlines()))
    for f in sorted((J / 'numeric-evidence/training').glob('*pann-values.jsonl')):
        ps.extend((json.loads(l) for l in f.read_text().splitlines()))
    ts.sort(key=lambda t: t['transition'])
    ps.sort(key=lambda t: t['transition'])
    ids = [t['transition'] for t in ts]
    pids = [t['transition'] for t in ps]
    roll = []
    roll_mismatches = []
    for f in sorted(RAW.glob(f'{side}-training-rollout*.npz')):
        a = np.load(f, allow_pickle=False)
        start = int(f.name.split('rollout-')[1].split('-')[0])
        count = len(a['reward'])
        tr = [t for t in ts if start <= t['transition'] < start + count]
        pv = [t for t in ps if start <= t['transition'] < start + count]
        errs = {}
        for k in ['raw_action', 'reward', 'terminated', 'truncated', 'log_prob']:
            expected = np.array([t[k] for t in tr])
            observed = a[k].reshape(count)
            errs[k] = float(np.max(np.abs(observed.astype(float) - expected.astype(float))))
        for k in ['value', 'next_value']:
            expected = np.array([t[k] for t in pv])
            errs[k] = float(np.max(np.abs(a[k].reshape(count) - expected)))
        errs['pann_obs_vs_export_max'] = float(np.max(np.abs(a['obs'][:, :, -12:].reshape(count, 12) * 0.001 - np.array([p['pann_recovery_m'] for p in pv]))))
        errs['pann_next_obs_vs_export_max'] = float(np.max(np.abs(a['next_obs'][:, :, -12:].reshape(count, 12) * 0.001 - np.array([p['next_pann_recovery_m'] for p in pv]))))
        roll.append({'file': f.name, 'start': start, 'count': count, 'arrays': {k: list(a[k].shape) for k in a.files}, 'all_finite': all((np.all(np.isfinite(a[k])) for k in a.files)), 'comparison_max_abs_error': errs})
    st = {'job': job, 'initial_samples': j['initial_transitions'], 'initial_updates': j['initial_updates'], 'final_samples': col['collected_transitions'], 'final_updates': col['updates'], 'tail': col['unupdated_fragment'], 'excluded': col['excluded_old_version_transitions'], 'last_episode': col['episode'], 'resume_file': f"resume-state-ep{col['episode']:05d}.pt", 'invocation_wall_s': col['invocation_wall_s'], 'training_exit_code': j['training_exit_code'], 'training_stop_reason': j['training_stop_reason'], 'deadline_unix_s': j['deadline_unix_s'], 'wall_deadline_enabled': j['wall_deadline_enabled'], 'transition_rows': len(ts), 'unique_transition_ids': len(set(ids)), 'transition_range': [min(ids), max(ids)], 'contiguous_exact_phase_range': ids == list(range(13714, 14226)), 'pann_ids_match': ids == pids, 'behavior_updates_counts': dict(collections.Counter((t['behavior_updates'] for t in ts))), 'terminated_rows': sum((t['terminated'] for t in ts)), 'truncated_rows': sum((t['truncated'] for t in ts)), 'failure_rows': sum((t['failure'] for t in ts)), 'reward_sum': sum((t['reward'] for t in ts)), 'cap_m_s': dist([t['cap_m_s'] for t in ts]), 'pann_current_max_abs': max((abs(v) for p in ps for v in p['pann_recovery_m'])), 'pann_next_max_abs': max((abs(v) for p in ps for v in p['next_pann_recovery_m'])), 'saved_array_match_all': all((p['saved_array_match'] for p in ps)), 'updates': [{k: u[k] for k in ['update_count', 'cumulative_samples', 'samples', 'optimizer_steps', 'loss', 'loss/clip', 'loss/vf', 'loss/ent', 'behavior_logprob_max_error', 'approx_kl', 'clip_fraction']} for u in learn['updates']], 'rollout_verification': roll, 'evaluation': {k: ev[k] for k in ['status', 'checkpoint', 'checkpoint_transitions', 'checkpoint_updates', 'deterministic_policy_mean', 'frozen_normalization', 'ppo_updates', 'training_end']}, 'evaluation_stages': [{k: s[k] for k in ['level', 'stage_status', 'reason', 'success', 'error', 'exit_code', 'simulation_s', 'wall_s'] if k in s} for s in ev['stages']]}
    st['cap_by_behavior_update'] = {str(u): dist([t['cap_m_s'] for t in ts if t['behavior_updates'] == u]) | {'mean': float(np.mean([t['cap_m_s'] for t in ts if t['behavior_updates'] == u]))} for u in sorted({t['behavior_updates'] for t in ts})}
    summary['streams'][side] = st
    paths = sorted(R.glob(f'train-{job}-*.json')) + [R / f"{job}-eval-PANN_PPO_S5-s{(1 if side == 'a' else 2)}-E30.json"]
    for p in paths:
        if p.name.endswith('-resolved.json'):
            continue
        d = json.loads(p.read_text())
        resolved = json.loads(p.with_name(p.stem + '-resolved.json').read_text())
        label = 'e30' if '-eval-' in p.name else 'ep' + p.stem.rsplit('-ep', 1)[1]
        a = np.load(RAW / f'{side}-{label}-trajectory.npz', allow_pickle=False)
        t = a['time']
        out = next((x for x in d['transitions'] if x['phase'] == 'outbound'))
        mask = t >= out['time_s'] - 1e-09
        q = a['qpos'][mask]
        c = a['contact'][mask]
        f = a['feet'][mask]
        v = a['qvel'][mask]
        cmd = a['command_velocity'][mask]
        fz = a['actual_grf'][mask, :, 2]
        diff = np.diff(c.astype(np.int8), axis=0)
        mo = {'trajectory_samples': len(t), 'physics_s': float(t[-1]), 'max_base_x_m': float(a['qpos'][:, 0].max()), 'final_base_xyz_m': a['qpos'][-1, :3].tolist(), 'outbound_start_s': out['time_s'], 'outbound_dx_m': float(q[-1, 0] - q[0, 0]), 'outbound_dz_m': float(q[-1, 2] - q[0, 2]), 'outbound_mean_vx_m_s': float(v[:, 0].mean()), 'outbound_mean_command_vx_m_s': float(cmd[:, 0].mean()), 'outbound_four_feet_contact_fraction': float(np.mean(np.all(c, axis=1))), 'outbound_contact_liftoffs': int(np.sum(diff < 0)), 'outbound_contact_touchdowns': int(np.sum(diff > 0)), 'outbound_joint_target_change_max_rad': float(np.max(np.abs(a['q_target'][mask] - a['q_target'][mask][0]))), 'max_foot_x_m': float(a['feet'][:, :, 0].max()), 'minimum_foot_normal_force_N': fz.min(axis=0).tolist(), 'maximum_foot_normal_force_N': fz.max(axis=0).tolist(), 'foot_z_range_m': [f[:, :, 2].min(axis=0).tolist(), f[:, :, 2].max(axis=0).tolist()], 'outbound_support_active_samples': int(a['support_active'][mask].sum()), 'outbound_raw_pd_max_abs_Nm': float(np.abs(a['unclipped_pd_request'][mask]).max()), 'outbound_raw_pd_minus_applied_torque_max_abs_Nm': float(np.abs(a['unclipped_pd_request'][mask] - a['torque'][mask]).max()), 'nonfoot_ground_contact_max': int(a['nonfoot_ground_contacts'].max()), 'relative_height_residual_outbound_first_last': a['relative_height_residual'][mask][[0, -1]].tolist(), 'sampling_limit': 'trajectory is 500Hz endpoint telemetry; does not by itself prove no saturation/contact changes inside each ten 0.2ms physics substeps.'}
        mo['outbound_requested_cap_m_s'] = dist(a['requested_speed_cap'][mask]) | {'mean': float(a['requested_speed_cap'][mask].mean())}
        warnings_path = J / 'episodes' / p.stem / 'mujoco-warnings.json'
        warnings = json.loads(warnings_path.read_text()) if warnings_path.exists() else None
        calls = [json.loads(l) for l in gzip.open(RAW / f'{side}-{label}-planner-calls.jsonl.gz', 'rt')]
        new = [c for c in calls if c['epoch_reset_reason'] != 'same_problem']
        bad = [c['call_id'] for c in new if c['search_attempts'] and (c['search_attempts'][0]['lambda_used'], c['search_attempts'][0]['rate_used']) != (1, 1)]
        ats = [a for c in calls for a in c['search_attempts']]
        oc = [c for c in calls if c['origin'] >= out['time_s'] - 1e-09]
        oa = [a for c in oc for a in c['search_attempts']]
        accepted = [a for a in oa if a['status'] == 'accepted_native_descent']
        opt = {'calls': len(calls), 'deployment': dict(collections.Counter((c['deployment_status'] for c in calls))), 'return_reasons': dict(collections.Counter((c['return_reason'] for c in calls))), 'epoch_resets': len(new), 'bad_epoch_first_attempts': bad, 'new_epochs_without_attempts': [c['call_id'] for c in new if not c['search_attempts']], 'first_attempt_lambda_rate_counts': dict(collections.Counter((str((c['search_attempts'][0]['lambda_used'], c['search_attempts'][0]['rate_used'])) for c in new if c['search_attempts']))), 'attempt_lambda_used_counts': dict(collections.Counter((str(a['lambda_used']) for a in ats))), 'outbound_deployment': dict(collections.Counter((c['deployment_status'] for c in oc))), 'outbound_k_unscaled': dist([a['k_unscaled_max'] for a in oa]), 'outbound_Qu_unregularized': dist([a['Qu_unregularized_max'] for a in oa]), 'outbound_native_accepted_decrease': dist([a['native_decrease'] for a in accepted]), 'outbound_proxy_decrease_on_accepted': dist([a['proxy_full_step_decrease'] for a in accepted]), 'online_wall_s': dist([c['wall_s'] for c in calls if c['origin'] > 0]), 'startup_stop_calls': [{'call': c['call_id'], 'stopped': c['epoch_stopped'], 'attempts': c['backward_attempts'], 'lambda_after': c['lambda_after'], 'rate_after': c['rate_after']} for c in calls if c['origin'] == 0 and c['call_id'] >= 12], 'outbound_nominal_cost_first_last': [oa[0]['nominal_current_cost'], oa[-1]['nominal_current_cost']]}
        md = {k: d['material'][k] for k in ['world_cells', 'loaded_ever_cells', 'events', 'max_abs_principal_strain', 'max_stress_original_MPa', 'domain_exit']}
        md['event_log_uncompressed_bytes'] = len(gzip.open(RAW / f'{side}-{label}-material-events.jsonl.gz', 'rb').read())
        md['failure_log_uncompressed_bytes'] = len(gzip.open(RAW / f'{side}-{label}-material-failures.jsonl.gz', 'rb').read())
        summary['episodes'].append({'stream': side, 'name': p.stem, 'kind': 'evaluation' if label == 'e30' else 'training', 'source_file': str(p.relative_to(REPO)), 'source_commit': d['source_commit'], 'reason': d['reason'], 'error': d['error'], 'success': d['success'], 'stage_status': d['stage_status'], 'wall_s': d['wall_s'], 'compact_fields_match_raw': {'final_qpos': np.array_equal(d['final_qpos'], a['qpos'][-1]), 'max_x_m': d['max_x_m'] == mo['max_base_x_m'], 'physics_s': abs(d['simulation_s'] - t[-1]) < 1e-09}, 'motion': mo, 'optimizer': opt, 'material': md, 'warnings': warnings, 'legal_route': d['legal_route'], 'policy_total_transitions': d.get('policy', {}).get('total_transitions')})
summary['aggregate'] = {'training_episodes': sum((e['kind'] == 'training' for e in summary['episodes'])), 'evaluation_episodes': sum((e['kind'] == 'evaluation' for e in summary['episodes'])), 'training_end_reasons': dict(collections.Counter((e['reason'] for e in summary['episodes'] if e['kind'] == 'training'))), 'result_level_errors': sum((e['error'] is not None for e in summary['episodes'])), 'all_calls': sum((e['optimizer']['calls'] for e in summary['episodes'])), 'training_calls': sum((e['optimizer']['calls'] for e in summary['episodes'] if e['kind'] == 'training')), 'all_bad_reset_calls': sum((len(e['optimizer']['bad_epoch_first_attempts']) for e in summary['episodes'])), 'all_liftoffs': sum((e['motion']['outbound_contact_liftoffs'] for e in summary['episodes'])), 'all_touchdowns': sum((e['motion']['outbound_contact_touchdowns'] for e in summary['episodes'])), 'all_material_events': sum((e['material']['events'] for e in summary['episodes'])), 'distinct_new_transitions': sum((s['transition_rows'] for s in summary['streams'].values()))}
for x in downloads:
    pub['downloaded_and_sha256_verified'].append({k: x[k] for k in ['job', 'attempt', 'source_path', 'sha256', 'sha256_matches_remote', 'downloaded_bytes']} | {'asset_name': x['asset']['name'], 'asset_url': x['asset']['url']})
pub['downloaded_count'] = len(downloads)
pub['downloaded_bytes'] = sum((x['downloaded_bytes'] for x in downloads))
(OUT / 'cpu-summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=plain) + '\n')
(OUT / 'publication-audit.json').write_text(json.dumps(pub, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(summary['aggregate'], indent=2))
for e in summary['episodes']:
    print(e['name'], e['motion']['outbound_dx_m'], e['optimizer']['deployment'])
print('downloaded_bytes', pub['downloaded_bytes'], 'release_bytes', [(s, p['remote_bytes']) for s, p in pub['streams'].items()])
