"""Time two CUDA launch layouts on the registered W59 development snapshots."""
import gzip
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path

import numpy as np

import compare_s1_w59_m1_gpu as replay
from s1_w59_m1_hybrid import BUILD, HybridObserver


LIBRARIES = (BUILD / 'propagate.so', BUILD / 'propagate-one-column.so')
PERCENTILES = (25, 50, 75, 90, 95, 99, 100)
TIMINGS = ('full_M1_ms', 'synchronous_GPU_propagation_ms', 'CPU_force_correction_ms')


def different_outputs(observers, traces):
    grounds = [o.ground for o in observers]
    arrays = (
        ('observer', observers, ('last_forces', 'last_prior_forces', 'last_correction', 'last_areas')),
        ('ground', grounds, ('e', 'q', 'stress', 'loaded', 'ever_loaded', 'last_normal',
                             'last_area', 'last_owner', 'last_load_event_id', 'last_release')),
        ('trace', traces, ('ids', 'owners', 'active')),
    )
    result = []
    for location, pair, names in arrays:
        for name in names:
            a, b = (np.asarray(getattr(obj, name)) for obj in pair)
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                result.append(f'{location}.{name}')
    values = (
        ('observer', observers, ('last_packet_events', 'relaxing', 'load_time', 'release_id')),
        ('ground', grounds, ('domain_exit', 'event_serial', 'max_strain', 'max_stress', 'max_rate',
                             'max_residual')),
    )
    for location, pair, names in values:
        for name in names:
            if getattr(pair[0], name) != getattr(pair[1], name):
                result.append(f'{location}.{name}')
    return result


def compare_snapshot(index, source, observers, traces):
    episode = source['snapshot_id'].rsplit('/', 1)[0]
    saved_path = replay.RAW / episode / 'w59-legal-requests' / Path(source['snapshot']).name
    blob = saved_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != source['sha256']:
        raise ValueError(f"Snapshot changed: {source['snapshot_id']}")
    serialized = gzip.decompress(blob)
    elapsed = np.zeros((6, 2, len(TIMINGS)))
    differences = []
    for rep in range(len(elapsed)):
        for observer in observers:
            saved = pickle.loads(serialized)
            vars(observer.ground).update(saved['belief'])
            vars(observer).update(saved['observer'])
        packet = observers[0].packet
        inputs = tuple(np.asarray([row[j] for row in packet]) for j in range(4))
        inputs += (np.asarray(observers[0].force_measurements),)
        for observer in observers:
            observer.packet = []
            observer.force_measurements = []
        for i in ((0, 1) if (index+rep)%2 == 0 else (1, 0)):
            traces[i].clear()
            observer = observers[i]
            native = observer.native
            propagation = native.propagation_wall_s
            correction = native.correction_wall_s
            elapsed[rep, i, 0] = replay.publish(observer, *inputs)
            elapsed[rep, i, 1] = 1000*(native.propagation_wall_s-propagation)
            elapsed[rep, i, 2] = 1000*(native.correction_wall_s-correction)
        fields = different_outputs(observers, traces)
        if fields:
            differences.append(dict(repetition=rep, fields=fields))
    return dict(id=source['snapshot_id'], stratum=source['stratum'], input_sha256=source['sha256'],
                columns=len(traces[0].ids), cell_ticks=traces[0].active.size,
                publication_events=len(observers[0].last_packet_events), differences=differences,
                samples_ms=elapsed.tolist(), warm_medians_ms=np.median(elapsed[1:], axis=0).tolist(),
                warm_full_M1_quartiles_ms=np.percentile(elapsed[1:, :, 0], (25, 75), axis=0).tolist())


def episode_columns(case, largest_snapshot):
    packet_path = BUILD / 'replay' / f'{case}-packets.npz'
    if not packet_path.exists():
        return dict(packets=None, reason='Main paired replay has no finished packet array')
    with np.load(packet_path) as source:
        columns = source['columns'][:, 0]
        elapsed = source['times_ms']
    outside = columns > largest_snapshot
    nonempty = columns[columns > 0]
    bins = [0, 1, 33, 65, 129, 257, 513, 1025, np.iinfo(np.int32).max]
    counts, edges = np.histogram(columns, bins=bins)
    return dict(packets=len(columns), empty=int(len(columns)-len(nonempty)),
                min_nonempty=int(nonempty.min()) if nonempty.size else None,
                percentiles=np.percentile(columns, PERCENTILES).tolist() if columns.size else [],
                bin_edges=edges.tolist(), bin_counts=counts.tolist(),
                above_snapshot_max=dict(max_snapshot_columns=largest_snapshot, packets=int(outside.sum()),
                    cpu_and_default_gpu_total_s=(elapsed[outside].sum(axis=0)/1000).tolist()))


def main():
    dest = BUILD / 'launch-layouts'
    dest.mkdir(exist_ok=True)
    start = replay.now()
    parameters = replay.read(replay.PARAMETERS)['reference_material']['parameters']
    observers = [HybridObserver(parameters, replay.WEIGHTS, mode='M1', propagation_library=lib)
                 for lib in LIBRARIES]
    try:
        traces = [replay.Trace(o) for o in observers]
        rows = [compare_snapshot(i, source, observers, traces) for i, source in enumerate(replay.anchors())]
    finally:
        for observer in observers:
            observer.close()
    medians = np.array([row['warm_medians_ms'] for row in rows])
    sum_medians = medians.sum(axis=0)
    warm_full = np.array([row['samples_ms'] for row in rows])[:, 1:, :, 0]
    trial_totals = warm_full.sum(axis=0)
    paths = (*LIBRARIES, BUILD / 'correction.so')
    fatbin = {}
    for path in LIBRARIES:
        image = subprocess.check_output(['objcopy', '-O', 'binary', '--only-section=.nv_fatbin',
                                         str(path), '/dev/stdout'])
        fatbin[path.name] = hashlib.sha256(image).hexdigest()
    largest_snapshot = max(row['columns'] for row in rows)
    result = dict(
        started=start, finished=replay.now(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        binaries={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, cuda_fatbin_sha256=fatbin,
        backend_order=['GPU_16_columns_per_block', 'GPU_1_column_per_block'], times_order=TIMINGS,
        protocol='Same registered development snapshots as the CPU/GPU comparison. Restore the same state '
                 'and alternate backend order across snapshots and repetitions. One warmup then five '
                 'paired timings. Includes all ten completed inputs, full M1 observe and publication. '
                 'Restoration and output comparisons outside the timer.',
        registered_snapshots=len(rows), nonempty_snapshots=sum(row['columns'] > 0 for row in rows),
        differing_state_fields_over_all_trials=sum(len(r['fields']) for row in rows for r in row['differences']),
        sum_anchor_medians_ms=sum_medians.tolist(), ratio_of_sums=(sum_medians[0]/sum_medians[1]).tolist(),
        full_M1_sum_by_warm_trial_ms=trial_totals.tolist(),
        full_M1_warm_trials_quartiles_ms=np.percentile(trial_totals, (25, 50, 75), axis=0).tolist(),
        full_M1_warm_trial_ratios=(trial_totals[:, 0]/trial_totals[:, 1]).tolist(),
        by_episode_column_distribution={k:episode_columns(k, largest_snapshot) for k in ('A', 'B')},
        episode_percentile_levels=PERCENTILES, rows=rows, new_physical_steps=0, online_5902_changed=False,
    )
    replay.write(dest / 'summary.json', result)
    print(json.dumps({k:result[k] for k in ('started', 'finished', 'registered_snapshots',
          'differing_state_fields_over_all_trials', 'sum_anchor_medians_ms', 'ratio_of_sums',
          'full_M1_warm_trials_quartiles_ms')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
