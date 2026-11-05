#!/usr/bin/env python3
"""Export saved R12 low-rate learning evidence; never run policy or physics."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASES = ['initial_stand', 'outbound', 'turn', 'return', 'terminal_stand', 'complete', 'far_hold', 'home_hold', 'turn_home']


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def lines(path):
    if path.is_file():
        with path.open() as handle:
            for number, text in enumerate(handle, 1):
                if text.strip():
                    yield number, json.loads(text)


def relative(path):
    return str(path.relative_to(ROOT))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def saved_values(stream, initial, final, latest_resume):
    """Read behavior values/features, without recomputing a network forward pass."""
    result, sources = {}, []
    for path in sorted(stream.glob('rollout-*.npz')):
        first, last = map(int, path.stem.split('-')[1:])
        if last <= initial or first > final:
            continue
        with np.load(path) as data:
            sources.append(relative(path))
            for offset, sample in enumerate(range(first, last + 1)):
                if not initial < sample <= final:
                    continue
                obs = data['obs'][offset].reshape(-1)
                nxt = data['next_obs'][offset].reshape(-1)
                result[sample] = dict(
                    value=float(data['value'][offset].item()), next_value=float(data['next_value'][offset].item()),
                    phase=PHASES[int(np.argmax(obs[80:89]))], next_phase=PHASES[int(np.argmax(nxt[80:89]))],
                    pann_recovery_m=(obs[-12:].astype(float) * .001).tolist(),
                    next_pann_recovery_m=(nxt[-12:].astype(float) * .001).tolist(),
                    raw_action=float(data['raw_action'][offset].item()),
                    log_prob=float(data['log_prob'][offset].item()),
                    reward=float(data['reward'][offset].item()),
                    terminated=bool(data['terminated'][offset].item()),
                    truncated=bool(data['truncated'][offset].item()),
                    source=relative(path), row=offset)
    if latest_resume and (ROOT / latest_resume).is_file():
        import torch
        path = ROOT / latest_resume
        saved = torch.load(path, map_location='cpu', weights_only=False)
        tail = saved.get('buffer', [])
        if tail:
            sources.append(relative(path))
        for sample, row in enumerate(tail, saved['samples'] - len(tail) + 1):
            if not initial < sample <= final or sample in result:
                continue
            obs = row['obs'].numpy().reshape(-1)
            nxt = row['next_obs'].numpy().reshape(-1)
            result[sample] = dict(
                value=float(row['value'].item()), next_value=float(row['next_value'].item()),
                phase=PHASES[int(np.argmax(obs[80:89]))], next_phase=PHASES[int(np.argmax(nxt[80:89]))],
                pann_recovery_m=(obs[-12:].astype(float) * .001).tolist(),
                next_pann_recovery_m=(nxt[-12:].astype(float) * .001).tolist(),
                raw_action=float(row['raw_action'].item()), log_prob=float(row['log_prob'].item()),
                reward=float(row['reward'].item()), terminated=bool(row['terminated'].item()),
                truncated=bool(row['truncated'].item()), source=relative(path), buffer_sample=sample)
    return result, sources


def export_low_rate(source, destination, identity, initial, final, values, updates, fixed_update=None, phase_events=None):
    grouped = defaultdict(list)
    value_tables = defaultdict(list)
    count, matched, missing = 0, 0, Counter()
    reward_missing = Counter()
    files = []
    for line_number, original in lines(source):
        row = dict(original)
        # A stream's own log may include an interrupted tail. Preserve all rows,
        # and identify membership rather than silently removing discarded data.
        sample = row['transition']
        if fixed_update is None and sample <= initial:
            continue
        row.update({key: row.get(key, value) for key, value in identity.items()})
        row['source_line'] = line_number
        saved = values.get(sample)
        saved_matches = bool(saved and all(
            np.float32(saved[key]) == np.float32(row[key]) for key in ('raw_action', 'log_prob', 'reward'))
            and all(saved[key] == row[key] for key in ('terminated', 'truncated')))
        row['in_saved_training_state'] = (None if fixed_update is not None else saved_matches)
        recorded_update = row.get('behavior_updates', row.get('behavior_update_count'))
        if recorded_update is None:
            recorded_update = fixed_update if fixed_update is not None else max(
                [identity['initial_updates']] + [x['update_count'] for x in updates if x['cumulative_samples'] < sample])
            row['behavior_version_source'] = ('frozen_evaluation_checkpoint' if fixed_update is not None
                                               else 'saved_update_boundaries_before_this_transition')
        else:
            row['behavior_version_source'] = 'recorded_at_sampling'
        row['behavior_updates'] = recorded_update
        table = {key: row.get(key) for key in ('episode', 'transition', 'transition_serial', 'time_s', 'phase_id', 'stream_id')}
        table['behavior_updates'] = recorded_update
        table['saved_array_match'] = saved_matches if fixed_update is None else None
        if phase_events:
            # Evaluation did not save observation arrays in W38. Its saved route
            # event clock still identifies the phases at the two interval ends.
            for key, clock in [('phase', row['time_s']-row['dt_s']), ('next_phase', row['time_s'])]:
                events = [event for event in phase_events if event['time_s'] <= clock+1e-9]
                if row.get(key) is None and events:
                    row[key] = events[-1]['phase']
            table['phase_source'] = 'saved_result_route_phase_event_clock'
        fields = ('value', 'next_value', 'phase', 'next_phase', 'pann_recovery_m', 'next_pann_recovery_m')
        absent = []
        for key in fields:
            table[key] = row.pop(key, None)
            if table[key] is None and saved_matches:
                table[key] = saved[key]
            if table[key] is None:
                absent.append(key)
                missing[key] += 1
        table['source'] = dict(log=relative(source), line=line_number,
                               saved_array=saved['source'] if saved_matches else None,
                               saved_array_row=saved.get('row') if saved_matches else None)
        table['bootstrap_value'] = (0.0 if row['terminated'] else table['next_value'])
        table['missing_fields'] = absent
        row['phase'], row['next_phase'] = table['phase'], table['next_phase']
        row['missing_reward_components'] = ['action_change'] if 'action_change' not in row else []
        reward_missing.update(row['missing_reward_components'])
        episode = f"episode-{row['episode']:05d}"
        grouped[episode].append(row)
        value_tables[episode].append(table)
        count += 1
        matched += int(saved_matches)
    for episode in grouped:
        for name, records in [('transitions', grouped[episode]), ('pann-values', value_tables[episode])]:
            path = destination / f'{episode}-{name}.jsonl'
            write_rows(path, records)
            files.append(path)
    return files, dict(source=relative(source), source_available=source.is_file(), rows=count,
        saved_array_matched_rows=matched, missing_fields=dict(missing), missing_reward_components=dict(reward_missing),
        files=[relative(path) for path in files])


def episode_events(episode, small):
    result_path = small / (episode.name + '.json')
    result = read(result_path) or read(episode / 'result.json')
    counts, first = Counter(), {}
    for number, row in lines(episode / 'material-events.jsonl'):
        kind = row['type']
        counts[kind] += 1
        if kind not in first:
            first[kind] = dict(source_line=number, event=row)
    return dict(attempt=episode.name, result_source=relative(result_path),
        result_available=bool(result),
        outcome={key: result[key] for key in ('status', 'success', 'reason', 'error', 'simulation_s',
            'stage_status', 'completed_round_trips', 'stage_progress', 'legal_route',
            'terminal_reward_settlement') if key in result},
        route_phase_events=result.get('transitions'),
        material_event_counts=dict(counts), first_material_events_by_type=first,
        material_event_source=relative(episode / 'material-events.jsonl'),
        material_event_source_available=(episode / 'material-events.jsonl').is_file(),
        material_domain_exit=result.get('material', {}).get('domain_exit'),
        terminal=dict(time_s=result.get('simulation_s'), reason=result.get('reason'),
                      final_qpos=result.get('final_qpos'), final_qvel=result.get('final_qvel')),
        scope='Saved result/route/material-event summary, not an inferred dense contact or action trace.')


def export_job(jobdir, small, large, stream, episodes, final_samples, latest_resume, evaluation):
    job = read(jobdir / 'job.json')
    destination = small / 'jobs' / jobdir.name / 'numeric-evidence'
    identity = {key: job.get(key) for key in ('stream_id', 'phase_id', 'controller_build', 'reward_version',
        'source_commit', 'initial_updates')}
    identity.update(job=jobdir.name, group=job['group'], seed=job['seed'])
    values, sources = saved_values(stream, job['initial_transitions'], final_samples, latest_resume)
    updates = [row for _, row in lines(stream / 'learning.jsonl')]
    files, training = export_low_rate(stream / 'transitions.jsonl', destination / 'training', identity,
        job['initial_transitions'], final_samples, values, updates)
    evaluations = {}
    for attempt in evaluation.get('attempts', []):
        source = jobdir / 'evaluation-policy' / attempt / 'transitions.jsonl'
        result = read(small / (attempt + '.json'))
        paths, info = export_low_rate(source, destination / 'evaluation' / attempt, identity,
            0, final_samples, {}, [], fixed_update=evaluation.get('checkpoint_updates'),
            phase_events=result.get('transitions'))
        files.extend(paths)
        evaluations[attempt] = info
    for episode in episodes:
        path = destination / 'episodes' / (episode.name + '-events.json')
        write(path, episode_events(episode, small))
        files.append(path)
    summary = dict(identity=identity, training_directory=relative(stream), initial_transitions=job['initial_transitions'],
        final_transitions=final_samples, training=training, evaluation=evaluations, array_sources=sources,
        schema=dict(transition_clock='time_s is transition end in that episode; action held over [time_s-dt_s,time_s]',
            actor_nominal_rate_Hz=2, truncated_last_interval_retained=True,
            PANN='Causal observer recovery at 0.002/0.2/0.5 s, each foot, metres; saved normalized obs[-12:] times0.001.',
            foot_order=['FL', 'FR', 'RL', 'RR'], recovery_horizons_s=[.002, .2, .5],
            phase_channels='obs[80:89] in original fixed nine-phase order', phases=PHASES,
            value='Saved behavior value/next_value; no network recomputation. bootstrap_value is zero on physical terminated, next_value on external truncated.',
            reward_components='Original recorded increments; action_change absent in W38 logs is reported missing, not inferred from reward.',
            behavior_version='Recorded at sampling when available, otherwise saved update boundaries (new update applies after its last sampled transition).',
            evaluation_W38_missing='Original fixed-policy W38 evaluation saved transition scalars but no rollout values/features; unavailable fields remain null.'),
        files=[relative(path) for path in files], new_physics=False, bulk_retained_local=True,
        deleted_files=0, full_observations_exported=False)
    manifest = destination / 'index.json'
    write(manifest, summary)
    files.append(manifest)
    return files, dict(index=relative(manifest), training_rows=training['rows'],
        training_saved_array_matched_rows=training['saved_array_matched_rows'],
        evaluation_rows=sum(x['rows'] for x in evaluations.values()),
        training_missing_fields=training['missing_fields'], file_count=len(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path, required=True)
    args = parser.parse_args()
    jobdir = (ROOT / args.job_dir).resolve()
    job = read(jobdir / 'job.json')
    contract = read(jobdir / 'execution-contract.json')
    small, large = (ROOT / contract['outputs'][key] for key in ('compact_root', 'large_root'))
    stream = ROOT / job['training_directory'] if job.get('training_directory') else large / 'training' / f"{job['group']}-seed{job['seed']}"
    previous = read(small / 'jobs' / (jobdir.name + '.json'))
    evaluation = read(jobdir / 'evaluation.json')
    episodes = [large / name for name in previous['attempts']]
    collection, progress = read(stream / 'collection.json'), read(stream / 'progress.json')
    paths, summary = export_job(jobdir, small, large, stream, episodes,
        collection['collected_transitions'], progress.get('checkpoint'), evaluation)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
