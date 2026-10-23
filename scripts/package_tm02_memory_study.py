#!/usr/bin/env python3
"""Package the completed fixed TM02 material-memory study and real artifacts.

Creates local archive and checksum assets. Upload and remote asset verification
are separate operations; this command does not claim a release is published.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n',
                    encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_training_throughput(out, compact):
    """Measure adjacent saved checkpoints within each actual execution segment."""
    profiles = {
        'train.log': {(10, 100): ('A_random100', 25),
                      (20, 1000): ('B_random1000', 25)},
        'train-micro100.log': {(20, 1000): ('B_random1000', 100)},
        'train-fullbatch.log': {(20, 1000): ('B_random1000', 1000),
                               (20, 100): ('C_mixed100', 100)}}
    intervals, logs = [], []
    for name, groups in profiles.items():
        path = out / name
        if not path.exists():
            continue
        previous, previous_key, segment = None, None, 0
        checkpoints = 0
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.startswith('{'):
                continue
            current = json.loads(line)
            if 'group_start' not in current:
                continue
            key = current['independent_runs'], current['paths_per_update']
            arm, microbatch = groups[key]
            checkpoints += 1
            if key != previous_key or current['iteration'] <= previous['iteration']:
                # The first checkpoint contains initialization/compilation. Its
                # cumulative elapsed time is not a measured steady-state chunk.
                segment += 1
            else:
                updates = current['iteration'] - previous['iteration']
                seconds = current['elapsed_s'] - previous['elapsed_s']
                intervals.append(dict(
                    source_log=name, segment=segment, arm=arm,
                    independent_models=key[0], histories_per_update=key[1],
                    path_microbatch_size=microbatch,
                    from_iteration=previous['iteration'], to_iteration=current['iteration'],
                    full_dataset_updates_per_model=updates,
                    from_checkpoint_elapsed_s=previous['elapsed_s'],
                    to_checkpoint_elapsed_s=current['elapsed_s'],
                    checkpoint_interval_s=seconds,
                    seconds_per_250_full_dataset_updates=seconds * 250 / updates,
                    full_dataset_updates_per_model_per_s=updates / seconds,
                    model_optimizer_updates_per_s=updates * key[0] / seconds,
                    complete_history_gradient_evaluations_per_s=updates * key[0] * key[1] / seconds))
            previous, previous_key = current, key
        logs.append(dict(file=name, sha256=sha256(path), training_checkpoints=checkpoints,
                         execution_segments=segment,
                         first_checkpoints_excluded_from_timing=segment))
    table = compact / 'training-throughput.csv'
    with table.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(intervals[0]))
        writer.writeheader()
        writer.writerows(intervals)
    summaries = []
    for arm, models, paths, micro in sorted({
            (r['arm'], r['independent_models'], r['histories_per_update'],
             r['path_microbatch_size']) for r in intervals}):
        rows = [r for r in intervals if (r['arm'], r['independent_models'],
                r['histories_per_update'], r['path_microbatch_size']) == (arm, models, paths, micro)]
        timing = [r['seconds_per_250_full_dataset_updates'] for r in rows]
        summaries.append(dict(arm=arm, independent_models=models,
            histories_per_update=paths, path_microbatch_size=micro,
            measured_checkpoint_intervals=len(rows),
            observed_from_iteration=min(r['from_iteration'] for r in rows),
            observed_to_iteration=max(r['to_iteration'] for r in rows),
            measured_full_dataset_updates_per_model=sum(r['full_dataset_updates_per_model'] for r in rows),
            measured_interval_seconds=sum(r['checkpoint_interval_s'] for r in rows),
            median_seconds_per_250_updates=float(np.median(timing)),
            mean_seconds_per_250_updates=float(np.mean(timing)),
            min_seconds_per_250_updates=float(np.min(timing)),
            max_seconds_per_250_updates=float(np.max(timing))))
    baseline = {r['arm']: r['median_seconds_per_250_updates'] for r in summaries
                if r['path_microbatch_size'] == 25}
    for row in summaries:
        if row['arm'] in baseline and row['path_microbatch_size'] != 25:
            row['median_speedup_vs_same_arm_micro25'] = (
                baseline[row['arm']] / row['median_seconds_per_250_updates'])
    return dict(table=table.relative_to(ROOT).as_posix(), logs=logs, groups=summaries,
        timing_definition='Differences between adjacent checkpoint elapsed_s values inside the same log '
                          'and the same (independent models, histories per update) execution segment. '
                          'The first checkpoint of every segment is excluded; no interval crosses a restart.',
        scope='Measured saved-checkpoint intervals only; no extrapolated future timing. '
              'Checkpoint kernel timing excludes serialization outside the timed training call.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/tm02/r2-memory-study.json')
    parser.add_argument('--run', default='tm02-memory-study-20261023')
    parser.add_argument('--release-tag', default='tm02-memory-study-evidence-20261023')
    parser.add_argument('--executed-command', action='append', default=[],
                        help='Additional command actually executed; repeat as needed.')
    parser.add_argument('--execution-log', action='append', default=[],
                        help='Actual execution log to include; repeat as needed.')
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = read_json(config_path)
    base_path = ROOT / config['base']['config']
    base = read_json(base_path)
    protocol_path = ROOT / 'experiments/tm02/r2-revisit.json'
    out = ROOT / base['outputs']['large_root'] / args.run
    compact = ROOT / config['outputs']['compact_reproduction_root'] / args.run
    transfer_compact = ROOT / config['outputs']['compact_transfer_root'] / args.run
    index_path = compact / 'index.json'
    index = read_json(index_path)
    training = read_json(out / 'training.json')
    execution = read_json(out / 'run.json')
    frozen_data = read_json(out / 'data.json')
    references = read_json(out / 'final_reference/catalog.json')['histories']
    final_summary = read_json(out / 'final_predictions/evaluation.json')
    diagnostic_summary = read_json(out / 'diagnostics/predictions/evaluation.json')

    # Count the requested scientific deliverables, retaining failed predictions.
    runs = training['runs']
    selected = list(training['selection'].values())
    final_records = [read_json(p) for p in sorted((out / 'final_predictions').glob('*/*.json'))]
    diagnostic_records = [read_json(p) for p in
                          sorted((out / 'diagnostics/predictions').glob('*/*.json'))]
    final_arrays = list((out / 'final_predictions').glob('*/*.npz'))
    diagnostic_arrays = list((out / 'diagnostics/predictions').glob('*/*.npz'))
    curve_lengths = []
    for record in runs:
        with np.load(out / record['loss_curve']) as curve:
            curve_lengths.append(len(curve['iteration']))
    training_arrays = {}
    for arm in config['arms']:
        path = out / 'training' / arm['id'] / 'observations.npz'
        with np.load(path) as observed:
            training_arrays[arm['id']] = dict(
                histories=len(observed['train_time']),
                sample_count=observed['train_time'].shape[1],
                strain_shape=list(observed['train_strain'].shape),
                stress_shapes={cell: list(observed[f'{cell}_stress'].shape)
                               for cell in ('ideal', 'noise0', 'noise1', 'noise2')},
                noise_shapes={f'noise{noise}': list(observed[f'training_noise{noise}'].shape)
                              for noise in config['training']['noise_realizations_for_noisy']},
                sha256=sha256(path))
    completion = dict(
        study_model_records=len(runs),
        reused_models=sum(record['reused'] for record in runs),
        new_models=sum(not record['reused'] for record in runs),
        final_weights=sum((out / r['weights']).is_file() for r in runs),
        optimizer_states=sum((out / r['optimizer']).is_file() for r in runs),
        training_curves=len(curve_lengths),
        loss_curve_lengths=dict(Counter(str(n) for n in curve_lengths)),
        selected_models=len(selected),
        selected_potential_weights=sum((out / r['potential_weights']).is_file() for r in selected),
        model_update_counts=dict(Counter(str(r['iterations']) for r in runs)),
        new_model_optimizer_updates=sum(r['iterations'] for r in runs if not r['reused']),
        training_arrays=training_arrays,
        physical_final_conditions=len({r['condition_id'] for r in references}),
        final_reference_histories=len(references),
        final_reference_arrays=sum((out / 'final_reference' / r['arrays']).is_file() for r in references),
        final_prediction_records=len(final_records), final_prediction_arrays=len(final_arrays),
        final_model_directories=len({p.parent.name for p in final_arrays}),
        failed_final_predictions=sum(not r['integration_complete'] for r in final_records),
        diagnostic_prediction_records=len(diagnostic_records),
        diagnostic_prediction_arrays=len(diagnostic_arrays),
        diagnostic_model_directories=len({p.parent.name for p in diagnostic_arrays}),
        failed_diagnostic_predictions=sum(not r['integration_complete'] for r in diagnostic_records),
        final_evaluation=final_summary, known_diagnostics=diagnostic_summary,
        failed_rollouts_retained=True)
    tc = config['training']
    fc = config['final_evaluation']
    diagnostic_expected = tc['study_models_including_reused'] * (20 + 2 * 12)
    complete = (
        completion['study_model_records'] == completion['final_weights']
        == completion['optimizer_states'] == completion['training_curves']
        == tc['study_models_including_reused']
        and completion['reused_models'] == tc['reused_models']
        and completion['new_models'] == tc['new_training_runs']
        and completion['selected_models'] == completion['selected_potential_weights'] == tc['selected_models']
        and all(r['iterations'] == tc['iterations_per_model'] for r in runs)
        and all(n == tc['iterations_per_model'] for n in curve_lengths)
        and completion['new_model_optimizer_updates'] == tc['new_model_optimizer_updates']
        and all(training_arrays[a['id']]['histories'] == a['histories']
                and training_arrays[a['id']]['sample_count'] == tc['increments_per_history'] + 1
                for a in config['arms'])
        and completion['physical_final_conditions'] == fc['physical_condition_count']
        and completion['final_reference_histories'] == completion['final_reference_arrays']
        == fc['complete_reference_histories']
        and completion['final_prediction_records'] == completion['final_prediction_arrays']
        == fc['learned_model_history_predictions']
        and completion['final_model_directories'] == tc['selected_models']
        and completion['diagnostic_prediction_records'] == completion['diagnostic_prediction_arrays']
        == diagnostic_expected
        and completion['diagnostic_model_directories'] == tc['study_models_including_reused'])
    if not complete:
        raise SystemExit('Fixed study artifacts are incomplete; no release archive created.\n'
                         + json.dumps(completion, indent=2))

    throughput = write_training_throughput(out, compact)
    execution_changes = execution['execution_changes']
    discarded_update_bound = sum(
        sum(not r['reused'] and r['arm'] == change['arm'] for r in runs)
        * config['execution']['checkpoint_every_iterations'] for change in execution_changes)
    training_compute = {key: training[key] for key in (
        'total_optimizer_updates', 'elapsed_s', 'history_gradient_evaluations',
        'arm_compute', 'compute_scope')}
    training_compute.update(
        total_optimizer_updates_definition='Retained training trajectories of the 50 new models: '
                                           '20,000 updates each. This excludes discarded work at execution changes.',
        retained_training_trajectory_optimizer_updates=training['total_optimizer_updates'],
        discarded_model_optimizer_update_attempts_upper_bound=discarded_update_bound,
        discarded_work_bound_definition='At most one checkpoint interval per affected model per recorded '
                                       'microbatch switch. This is an upper bound, not a measured count.',
        interrupted_wall_time_scope='execution_changes.seconds_since_checkpoint_upper_bound includes '
                                   'downtime between interruption and continuation; it is not actual compute time.',
        other_arithmetic_scope='Compilation and final training-loss/gradient evaluations are additional '
                               'work outside retained Adam update counts.',
        throughput=throughput)

    work = ROOT / 'work/terrain-memory'
    work.mkdir(parents=True, exist_ok=True)
    archive = work / f'{args.run}-evidence.tar.gz'
    checksum = work / f'{args.run}-evidence.sha256'
    source_files = [ROOT / 'scripts/run_tm02_memory_study.py', Path(__file__),
                    *sorted((ROOT / 'src/tm02').glob('*.py'))]
    config_files = [config_path, base_path, protocol_path]
    release_url = f"{config['base']['release_url'].rsplit('/tag/', 1)[0]}/tag/{args.release_tag}"
    download = release_url.replace('/tag/', '/download/')
    metadata = dict(
        command=' '.join(sys.argv),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        package_source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        completion=completion,
        actual_commands=list(dict.fromkeys([
            *[invocation['command'] for invocation in execution['invocations']],
            *args.executed_command])),
        execution_provenance=(out / 'run.json').relative_to(ROOT).as_posix(),
        execution_provenance_sha256=sha256(out / 'run.json'),
        hardware={key: execution[key] for key in ('host', 'hardware', 'cpu', 'python', 'jax')},
        training_compute=training_compute,
        execution_changes=execution_changes,
        seeds=dict(initialization=tc['initialization_seeds'],
                   original_training_histories=base['data']['train']['strain_seeds'],
                   random_augmentation={key: config['arms'][1][key] for key in (
                       'additional_strain_seed_start', 'additional_strain_seed_count')},
                   training_noise=config['training_noise']['realizations'],
                   recovery_parameter_seed_start=config['recovery_training']['parameter_seed_start'],
                   final_families=[{key: value for key, value in family.items()
                                    if 'seed' in key or key == 'id'}
                                   for family in fc['families']]),
        packaged_current_source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files},
        packaged_config_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in config_files},
        frozen_data=frozen_data,
        frozen_artifact_dependency=config['base'],
        release_url=release_url,
        archive_scope='Only this study run, its compact results, source and configurations. '
                      'Ten reused models and reused diagnostic arrays are actual files inside this run.',
        archive_index_snapshot='The archive contains the index before its own digest and byte count are added. '
                               'The external checksum authenticates the archive; artifact-files.json hashes '
                               'every other included member, including that index snapshot.',
        publication='Local assets prepared; upload and remote asset hash have not been verified by this command.')
    index.pop('artifact_archive_sha256', None)
    index.pop('artifact_archive_bytes', None)
    index['artifact_archive'] = f'{download}/{archive.name}'
    index['artifact_checksum'] = f'{download}/{checksum.name}'
    index['training']['performance'] = throughput
    index['artifact_package'] = metadata
    write_json(index_path, index)

    manifest_path = out / 'artifact-files.json'
    trees = [out, compact]
    if transfer_compact.is_dir():
        trees.append(transfer_compact)
    files = sorted({p for directory in trees for p in directory.rglob('*')
                    if p.is_file() and p != manifest_path} | set(source_files + config_files))
    members = {p.relative_to(ROOT).as_posix(): p for p in files}
    for log in args.execution_log:
        path = Path(log).resolve()
        members[f'evidence/logs/{path.name}'] = path
    manifest = {name: dict(bytes=path.stat().st_size, sha256=sha256(path))
                for name, path in members.items()}
    write_json(manifest_path, manifest)
    members[manifest_path.relative_to(ROOT).as_posix()] = manifest_path
    with tarfile.open(archive, 'w:gz') as tar:
        for name, path in members.items():
            tar.add(path, arcname=name, recursive=False)
    digest = sha256(archive)
    checksum.write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    index['artifact_archive_sha256'] = digest
    index['artifact_archive_bytes'] = archive.stat().st_size
    write_json(index_path, index)
    print(json.dumps(dict(archive=str(archive), checksum=str(checksum), sha256=digest,
                         bytes=archive.stat().st_size, completion=completion), indent=2))


if __name__ == '__main__':
    main()
