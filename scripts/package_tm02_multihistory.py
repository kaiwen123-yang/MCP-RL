#!/usr/bin/env python3
"""Package the completed fixed TM02 multi-history experiment and its provenance.

Run in the existing TM02 environment after training, paired evaluation and revisit:
work/envs/tm02/bin/python scripts/package_tm02_multihistory.py --release-tag TAG
This creates local release assets; uploading them is a separate operation.
"""
import argparse
from collections import Counter
import hashlib
import io
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
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/tm02/r2-multihistory.json')
    parser.add_argument('--run', default='tm02-multihistory-20261023')
    parser.add_argument('--release-tag')
    parser.add_argument('--executed-command', action='append', default=[],
                        help='Additional command actually executed for this run; repeat as needed.')
    parser.add_argument('--execution-log', action='append', default=[],
                        help='Actual execution log to include; repeat as needed.')
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    protocol_path = ROOT / config['revisit_comparison']['protocol']
    protocol = read_json(protocol_path)
    out = ROOT / config['outputs']['large_root'] / args.run
    transfer = ROOT / protocol['outputs']['large_root'] / args.run
    compact = ROOT / config['outputs']['compact_root'] / args.run
    transfer_compact = ROOT / protocol['outputs']['compact_root'] / args.run
    index = read_json(compact / 'index.json')
    transfer_index = read_json(transfer_compact / 'index.json')
    training = read_json(out / 'training.json')
    training_run = read_json(out / 'run.json')
    transfer_run = read_json(transfer / 'run.json')

    # These counts describe the actual scientific artifacts, including failures.
    runs = training['runs']
    final_weights = list((out / 'final').glob('*-seed-??.npz'))
    optimizer_states = list((out / 'final').glob('*-optimizer.pkl'))
    curves = list((out / 'final').glob('*-loss.npz'))
    curve_lengths = []
    for path in curves:
        with np.load(path) as curve:
            curve_lengths.append(len(curve['iteration']))
    prediction_records = [read_json(p) for p in sorted((out / 'predictions').glob('*.json'))]
    prediction_counts = Counter(f"{r['cohort']}_{r['role']}" for r in prediction_records)
    prediction_arrays = list((out / 'predictions').glob('*.npz'))
    revisit_records = [read_json(p) for p in sorted(transfer.glob('*-new_*.json'))]
    revisit_arrays = list(transfer.glob('*-new_*.npz'))
    with np.load(out / 'observations.npz') as observed:
        training_histories, training_points = observed['train_time'].shape
        heldout_histories, heldout_points = observed['evaluation_time'].shape
    with np.load(out / 'reference.npz') as reference:
        reference_training_histories = len(reference['train_time'])
        reference_heldout_histories = len(reference['evaluation_time'])
    expected_runs = config['training']['total_training_runs']
    updates = config['training']['iterations_per_initialization']
    expected_predictions = {
        'old_heldout': config['comparison']['old_models_on_new_paths'],
        'new_heldout': config['comparison']['new_models_on_new_paths'],
        'new_known': config['comparison']['known_hard_path']['new_models'],
    }
    completion = dict(
        training_records=len(runs), final_weights=len(final_weights),
        final_optimizer_states=len(optimizer_states), training_curves=len(curves),
        model_update_counts=dict(Counter(str(r['iterations']) for r in runs)),
        loss_curve_lengths=dict(Counter(str(n) for n in curve_lengths)),
        actual_model_optimizer_updates=sum(r['iterations'] for r in runs),
        prediction_records=dict(prediction_counts), prediction_arrays=len(prediction_arrays),
        failed_predictions=sum(not r['integration_complete'] for r in prediction_records),
        new_revisit_records=len(revisit_records), new_revisit_arrays=len(revisit_arrays),
        failed_new_revisit_histories=sum(not r['integration_complete'] for r in revisit_records),
        training_histories=training_histories, increments_per_training_history=training_points-1,
        physical_heldout_histories=heldout_histories, increments_per_heldout_history=heldout_points-1,
        reference_training_histories=reference_training_histories,
        reference_heldout_histories=reference_heldout_histories,
        failed_rollouts_retained=True,
    )
    complete = (
        len(runs) == len(final_weights) == len(optimizer_states) == len(curves) == expected_runs
        and all(r['iterations'] == updates for r in runs)
        and all(n == updates for n in curve_lengths)
        and dict(prediction_counts) == expected_predictions
        and len(prediction_arrays) == sum(expected_predictions.values())
        and len(revisit_records) == len(revisit_arrays) == config['revisit_comparison']['new_selected_model_histories']
        and training_histories == reference_training_histories == config['data']['train']['paths_per_condition']
        and training_points-1 == config['data']['train']['increments']
        and heldout_histories == reference_heldout_histories == config['data']['evaluation']['paths']
        and heldout_points-1 == config['data']['evaluation']['increments']
    )
    if not complete:
        raise SystemExit('Fixed experiment artifacts are incomplete; no release archive created.\n'
                         + json.dumps(completion, indent=2))

    work = ROOT / 'work/terrain-memory'
    archive = work / f'{args.run}-evidence.tar.gz'
    checksum = work / f'{args.run}-evidence.sha256'
    work.mkdir(parents=True, exist_ok=True)
    source_files = [ROOT / 'scripts/run_tm02_r2.py', Path(__file__),
                    ROOT / 'scripts/run_tm02_r2_server.sh', *sorted((ROOT / 'src/tm02').glob('*.py'))]
    config_files = [ROOT / args.config, protocol_path]
    metadata = dict(
        command=' '.join(sys.argv),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        package_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        completion=completion,
        actual_commands=list(dict.fromkeys([training_run['command'], transfer_run['command'],
                                            *args.executed_command])),
        execution_provenance={'training': training_run, 'revisit': transfer_run},
        seeds=dict(initialization=config['training']['initialization_seeds'],
                   training_history=config['data']['train']['strain_seeds'],
                   stress_noise=config['data']['stress_noise']['seeds_by_training_path'],
                   heldout_history=config['data']['evaluation']['strain_seeds'],
                   known_diagnostic=config['comparison']['known_hard_path']['original_strain_seed']),
        packaged_current_source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files},
        packaged_config_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in config_files},
        frozen_artifact_dependency=config['comparison'],
        evaluation_cpu={row['field'].rstrip(':'): row['data'] for row in
            json.loads(subprocess.check_output(['lscpu', '-J'], text=True))['lscpu']
            if row['field'] in ('Model name:', 'CPU(s):', 'Thread(s) per core:')},
        archive_index_snapshot='Both compact indexes are included before adding this archive SHA256 and byte count. '
                               'The external checksum sidecar authenticates the archive; its internal file manifest '
                               'authenticates the included index snapshots and other files.',
        publication='Local assets prepared; release upload has not been verified by this packaging command.',
    )
    frozen_archive = work / config['comparison']['archive_filename']
    metadata['local_frozen_archive'] = dict(path=frozen_archive.relative_to(ROOT).as_posix(),
        bytes=frozen_archive.stat().st_size, sha256=sha256(frozen_archive))
    if args.release_tag:
        release_url = f"{config['comparison']['release_url'].rsplit('/tag/', 1)[0]}/tag/{args.release_tag}"
        download = release_url.replace('/tag/', '/download/')
        metadata['release_url'] = release_url
        archive_location, checksum_location = f'{download}/{archive.name}', f'{download}/{checksum.name}'
    else:
        archive_location = archive.relative_to(ROOT).as_posix()
        checksum_location = checksum.relative_to(ROOT).as_posix()
    for current in (index, transfer_index):
        current.pop('artifact_archive_sha256', None)
        current.pop('artifact_archive_bytes', None)
        current['artifact_archive'] = archive_location
        current['artifact_checksum'] = checksum_location
    index['artifact_package'] = metadata
    index['revisit_comparison'] = dict(
        index=(transfer_compact / 'index.json').relative_to(ROOT).as_posix(),
        new_histories=transfer_index['new_model_histories'],
        successful_new_histories=transfer_index['successful_new_histories'],
        summaries=transfer_index['summaries'], scope=transfer_index['scope'])
    transfer_index['artifact_package'] = dict(completion=completion,
        original_method_index=(compact / 'index.json').relative_to(ROOT).as_posix(),
        archive_index_snapshot=metadata['archive_index_snapshot'])
    write_json(compact / 'index.json', index)
    write_json(transfer_compact / 'index.json', transfer_index)

    manifest_path = out / 'artifact-files.json'
    trees = (out, transfer, compact, transfer_compact)
    files = sorted({p for directory in trees for p in directory.rglob('*')
                    if p.is_file() and p != manifest_path} | set(source_files + config_files))
    members = {p.relative_to(ROOT).as_posix(): p for p in files}
    for name in args.execution_log:
        path = Path(name).resolve()
        members[f'evidence/logs/{path.name}'] = path
    manifest = {name: dict(bytes=path.stat().st_size, sha256=sha256(path))
                for name, path in members.items()}
    # Preserve the committed versions recorded at execution, alongside current source.
    source_snapshots = {}
    for execution in (training_run, transfer_run):
        commit = execution['source_commit']
        for name, expected_hash in execution['source_sha256'].items():
            name = name.replace('\\', '/')
            destination = f'evidence/source/{commit}/{name}'
            data = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
            source_snapshots[destination] = data
            source_hash = hashlib.sha256(data).hexdigest()
            manifest[destination] = dict(bytes=len(data), sha256=source_hash,
                                         execution_recorded_sha256=expected_hash,
                                         matches_execution_record=source_hash == expected_hash)
    config_snapshot = json.dumps(training_run['config'], indent=2, allow_nan=False).encode('utf-8') + b'\n'
    source_snapshots['evidence/training-config-at-execution.json'] = config_snapshot
    manifest['evidence/training-config-at-execution.json'] = dict(
        bytes=len(config_snapshot), sha256=hashlib.sha256(config_snapshot).hexdigest())
    write_json(manifest_path, manifest)
    members[manifest_path.relative_to(ROOT).as_posix()] = manifest_path
    with tarfile.open(archive, 'w:gz') as tar:
        for name, path in members.items():
            tar.add(path, arcname=name, recursive=False)
        for name, data in source_snapshots.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    digest = sha256(archive)
    checksum.write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    for current, directory in ((index, compact), (transfer_index, transfer_compact)):
        current['artifact_archive_sha256'] = digest
        current['artifact_archive_bytes'] = archive.stat().st_size
        write_json(directory / 'index.json', current)
    print(json.dumps(dict(archive=str(archive), checksum=str(checksum), sha256=digest,
                          bytes=archive.stat().st_size, completion=completion), indent=2))


if __name__ == '__main__':
    main()
