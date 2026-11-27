"""Compare M1 with the recorded controller queries and the TM02 plant event ledger."""
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from math import ceil
from pathlib import Path

import numpy as np


KINDS = ('first_load', 'release', 'same_foot_reload', 'cross_foot_revisit')


def load_queries(raw):
    """The final successor at 60 s also queries M1. At t=0 there is no published MPC plan."""
    solves = raw/'w55-f'
    with (solves/'mpc-solves.jsonl').open() as stream:
        plans = [json.loads(line) for line in stream]
    times = [plan['time_s'] for plan in plans]
    result = {}
    records = raw.parent/'evaluation-policy'/raw.name/'transitions.jsonl'
    with records.open() as stream:
        for line in stream:
            row = json.loads(line)
            t = row['time_s']
            # The cap policy reads latest_plan before the force controller publishes a solve at t.
            previous = bisect_left(times, t)-1
            if previous < 0:
                raise ValueError(f'Missing MPC plan before recorded query at {t}')
            with np.load(solves/plans[previous]['file']) as plan:
                j = np.abs(plan['times']-(t+0.2)).argmin()
                xy = plan['foot_center_plan'][j, :, :2].copy()
            packet = round(t/.002)
            result[packet] = (t, xy, np.asarray(row['next_pann_recovery_m']).reshape(12))
    return result


def tick(time_s):
    return round(time_s/.0002)


def plant_events(raw):
    with (raw/'material-events.jsonl').open() as stream:
        for line in stream:
            entry = json.loads(line)
            kind = entry['type']
            if kind == 'load':
                kind = 'first_load'
            elif kind == 'revisit':
                kind = 'cross_foot_revisit' if entry['cross_leg'] else 'same_foot_reload'
            yield (entry['event_id'], kind, tuple(entry['world_key']), tick(entry['time_s']),
                   entry['leg'], entry['previous_load_event_id'])


def observer_events(replay_log, index):
    with gzip.open(replay_log, 'rt') as stream:
        for line in stream:
            backend, packet, event, kind, key, t, leg, previous = json.loads(line)
            if backend == index:
                yield event, kind, tuple(key), tick(t), leg, previous


def event_keys(events):
    loads = {}
    by_kind = defaultdict(Counter)
    cross_with_source = Counter()
    for event, kind, key, time_tick, leg, previous in events:
        identity = (key, time_tick, leg)
        by_kind[kind][identity] += 1
        if kind == 'cross_foot_revisit':
            previous_key, previous_tick, previous_leg = loads[previous]
            cross_with_source[(identity, previous_key, previous_tick, previous_leg)] += 1
        if kind != 'release':
            loads[event] = identity
    return by_kind, cross_with_source


def overlap(reference, observed):
    return dict(plant=reference.total(), observer=observed.total(),
                matched=(reference & observed).total())


def cross_time_alignment(plant_cross, observed_cross):
    series = []
    for counter in (plant_cross, observed_cross):
        grouped = defaultdict(list)
        for ((key, arrival_tick, leg), _, _, previous_leg), count in counter.items():
            grouped[(key, previous_leg, leg)].extend([arrival_tick]*count)
        series.append(grouped)
    original, estimate = series
    groups = original.keys() & estimate.keys()
    matched = sum(min(len(original[g]),len(estimate[g])) for g in groups)
    # A timing discrepancy is unambiguous for these groups without time tolerances.
    unique = [g for g in groups if len(original[g]) == len(estimate[g]) == 1]
    errors = [abs(original[g][0]-estimate[g][0]) for g in unique]
    return dict(same_cell_and_foot_pair=matched, plant=plant_cross.total(),
                observer=observed_cross.total(), unique_pair_groups=len(unique),
                exact_tick_in_unique_groups=errors.count(0),
                within_one_tick_in_unique_groups=sum(x<=1 for x in errors),
                max_abs_ticks_in_unique_groups=max(errors, default=None))


def compare_plant_ledger(raw, replay_log):
    plant, plant_cross = event_keys(plant_events(raw))
    out = {}
    for index, backend in enumerate(('cpu', 'hybrid')):
        observed, observed_cross = event_keys(observer_events(replay_log, index))
        out[backend] = {kind: overlap(plant[kind], observed[kind]) for kind in KINDS}
        out[backend]['cross_revisit_with_previous_load'] = overlap(plant_cross, observed_cross)
        out[backend]['cross_revisit_arrival_clocks'] = cross_time_alignment(plant_cross, observed_cross)
    return out


def export_recorded_inputs(raw, columns, packet_trace, output):
    """Store the exact sole motions and physical flags needed alongside the published force trace."""
    with np.load(packet_trace) as sample:
        force = sample['actual_Fz'].reshape(-1, 4)
        soft = sample['is_material'].reshape(-1, 4)
        packet_ends = sample['time']
    count = len(force)
    motion = np.empty((count, 26), dtype=np.float64)
    contact = np.empty((count, 4), dtype=bool)
    offset = 0
    for source in sorted(raw.glob('material-[0-9][0-9][0-9][0-9].npz')):
        with np.load(source) as z:
            foot = z['foot']
        if not len(foot):
            continue
        batch = slice(offset, offset+len(foot))
        motion[batch, :2] = foot[:, :2]
        motion[batch, 2:] = foot[:, 6:30]
        if not np.array_equal(foot[:, 2:6].astype(bool), soft[batch]):
            raise ValueError('Material mask differs from the measured trace')
        offset += len(foot)
    if offset != count or not np.array_equal(motion[9::10, 1], packet_ends):
        raise ValueError('Motion and publication clocks differ')
    legs = ('FL','FR','RL','RR')
    force_columns = [columns.index(f'grf_{leg}_z') for leg in legs]
    contact_columns = [columns.index('contact_'+leg) for leg in legs]
    offset = 0
    for source in sorted(raw.glob('physical-[0-9][0-9][0-9].npz')):
        with np.load(source) as z:
            physics = z['physical']
        batch = slice(offset, offset+len(physics))
        if not np.array_equal(physics[:, :2], motion[batch, :2]) or not np.array_equal(physics[:, force_columns], force[batch]):
            raise ValueError('Physical force or source clocks differ from the published comparison')
        contact[batch] = physics[:, contact_columns].astype(bool)
        offset += len(physics)
    if offset != count:
        raise ValueError('Physical contact trace is incomplete')
    np.savez_compressed(output, clock_s=motion[:, :2], sole_bottom_m=motion[:, 2:14].reshape(-1,4,3),
                        sole_velocity_m_s=motion[:, 14:26].reshape(-1,4,3), physical_contact=contact)


def whole_episode_projection(result):
    """Apply measured ratios to the corresponding original production timer scopes."""
    source = result['original_reference']
    cpu = result['native_timing']['cpu']['observer']
    hybrid = result['native_timing']['hybrid']['observer']
    obs0 = source['archived_observer_wall_s']
    query0 = source['archived_query_wall_s']
    old = source['archived_E60_wall_s']
    obs1 = obs0 * hybrid['observation_wall_s']/cpu['observation_wall_s']
    query1 = query0 * hybrid['query_wall_s']/cpu['query_wall_s']
    estimated = old-obs0-query0+obs1+query1
    return dict(original_E60_s=old, original_observe_s=obs0, original_query_s=query0,
                projected_observe_s=obs1, projected_query_s=query1, projected_E60_s=estimated,
                projected_s_saved=old-estimated, projected_whole_E60_speedup=old/estimated)


def sizes_and_times(packet_file):
    with np.load(packet_file) as data:
        columns = data['columns'][:, 0]
        times = data['times_ms']
    result = {}
    for label, lo, hi in [('empty',0,0),('1_to_32',1,32),('33_to_64',33,64),
                          ('65_to_128',65,128),('129_and_above',129,int(columns.max()))]:
        subset = times[(columns >= lo) & (columns <= hi)]
        if len(subset):
            cpu, hybrid = subset.sum(axis=0)*1.e-3
            result[label] = dict(windows=len(subset), cpu_s=cpu, hybrid_s=hybrid,
                                 total_speedup=cpu/hybrid, hybrid_faster_windows=int((subset[:, 1]<subset[:, 0]).sum()))
    return result


def verify_actor_queries(raw, query_file):
    expected = load_queries(raw)
    actual = json.loads(query_file.read_text())
    differences = dict(cpu=0, hybrid=0)
    nonzero = 0
    different_times = different_xy = 0
    for row in actual:
        t, xy, recorded = expected[row['packet']]
        different_times += int(row['time_s'] != t)
        different_xy += int(not np.array_equal(row['xy_m'], xy))
        nonzero += int(np.count_nonzero(recorded))
        for side in differences:
            # Original cap actor normalizes metres to millimetres and casts once to float32.
            published = (np.asarray(row[side+'_m'])/.001).astype(np.float32)*np.float32(.001)
            differences[side] += int(np.count_nonzero(published != recorded))
    return dict(original_queries=len(expected), completed=len(actual), different_times=different_times,
                different_footprints=different_xy, height_values=12*len(actual), original_nonzero=nonzero,
                different_values_at_original_publication_precision=differences)


def build_summary(single, paired, layouts, raw_root):
    cases = {}
    for case in ('A', 'B'):
        initial = json.loads((single/(case+'-summary.json')).read_text())
        final = json.loads((paired/(case+'-summary.json')).read_text())
        raw = raw_root/final['episode']
        events = compare_plant_ledger(raw, paired/(case+'-events.jsonl.gz'))
        cases[case] = dict(
            episode=final['episode'], complete=final['complete'],
            initial_single_worker_M1_publication=initial['totals'],
            full_M1=final['recovery']['complete_M1_including_queries'],
            actor_query_check=verify_actor_queries(raw,paired/(case+'-queries.json')),
            contact_threshold_edge_differences=final['threshold_signed_edge_mismatches'],
            max_abs_interbackend_Fz_N=final['force_max_abs_N'],
            material_cell_tick_comparison=final['exact_5kHz_cell_trace'],
            query_count=final['recovery']['completed_paired_queries'],
            final_publication_by_column_count=sizes_and_times(paired/(case+'-packets.npz')),
            soft_force_MAE_N=final['material_force_measured_MAE_N'],
            original_vs_observer_event_counts=events,
            projection=whole_episode_projection(final))
    layout = json.loads(layouts.read_text())
    sums = {k:sum(c['full_M1'][k] for c in cases.values()) for k in ('cpu_s','hybrid_s')}
    sums['speedup'] = sums['cpu_s']/sums['hybrid_s']
    complete_times = []
    for case in cases:
        with np.load(paired/(case+'-packets.npz')) as data:
            windows = data['times_ms']
        for row in json.loads((paired/(case+'-queries.json')).read_text()):
            windows[row['packet']-1] += row['times_ms']
        complete_times.append(windows)
    sums['percentiles_50_95_ms_cpu_hybrid'] = np.percentile(np.concatenate(complete_times),[50,95],axis=0).tolist()
    old = max(c['projection']['original_E60_s'] for c in cases.values())
    new = max(c['projection']['projected_E60_s'] for c in cases.values())
    root = Path(__file__).resolve().parents[1]
    study = root/'work/w59/m1-gpu-study'
    libraries = {'frozen_cpu_m1':root/'work/w57-m/material.so',
                 'host_correction':study/'correction.so',
                 'device_layout_16':study/'propagate.so',
                 'device_layout_1':study/'propagate-one-column.so'}
    device = []
    telemetry = study/'device.csv'
    if telemetry.is_file():
        with telemetry.open() as stream:
            device = [row for row in csv.DictReader(stream) if row['phase']=='paired']
    hardware = study/'device.json'
    return dict(source_runtime='4ad3a5f28b99ad9ccc84455915fdffa9dfcb57af',
                paired_shared_gpu_samples=dict(
                    hardware=json.loads(hardware.read_text()) if hardware.is_file() else None,
                    scope='Device-wide 30 s samples during the overlapping A/B workload; includes other jobs.',
                    count=len(device), first=device[0]['server_time'] if device else None,
                    last=device[-1]['server_time'] if device else None,
                    sm_percentiles_0_50_100=np.percentile([float(x['sm_percent']) for x in device],[0,50,100]).tolist() if device else [],
                    memory_used_MiB_range=[min(float(x['memory_used_MiB']) for x in device),max(float(x['memory_used_MiB']) for x in device)] if device else []),
                measured_libraries_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest()
                                           for k,v in libraries.items()},
                registration='Twelve W59 development snapshots and two completed 5902 PPO-full fallback E60 recordings, A/B 0021. No policy or model was fitted.',
                offline_schedule='Single-worker A then B on CPU 19 for the initial layout. Matched 12-snapshot layout selection. Final A and B each alternate CPU/GPU on identical real inputs and run together on CPUs 19 and 20. Original extra physics slots used: zero. Math threads: one per worker. GPU shared.',
                paired_gpu_library=final['propagation_library']['sha256'],
                frozen_material_weights=dict(archive='https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm02-memory-study-evidence-20261023',
                    member='selected/B_random1000--ideal-potentials.npz',
                    sha256=hashlib.sha256((root/'work/assets/B_random1000--ideal-potentials.npz').read_bytes()).hexdigest()),
                recording_layout='In the W59 measurement Release asset, inputs/<case>-inputs.npz provides every original substep start/end, sole bottom positions, sole velocities and physical contact flags. paired-workers/<case>-packets.npz stores actual_Fz and is_material on the same clocks. The query JSON contains every original query time, four xy positions, archived and both recomputed heights. The input folder also contains the original plant and controller contact ledgers.',
                cases=cases, full_M1_combined=sums,
                physical_event_comparison='Exact 0.2 ms right-boundary tick, global world-cell key, event kind and leg. Cross-foot lineage additionally requires the preceding load on the same key at the same tick and with the same previous foot.',
                projection_scope='Each underlying recorded E60 keeps its own original plant, controller and I/O time. Only its measured observation and recovery query times are scaled using matching paired internal timers. This estimates a possible online wall change; the plant and the live policy were not rerun.',
                paired_wave_projection=dict(original_s=old, projected_s=new, projected_s_saved=old-new,
                    projected_paired_waves_to_save_one_hour=ceil(3600/(old-new)) if old>new else None),
                layout_probe=dict(full_M1_candidate_over_reference_speedup=layout['ratio_of_sums'][0],
                                  differing_fields=layout['differing_state_fields_over_all_trials']),
                new_physics_steps=0, new_training_events=0, backend_5902='unchanged production CPU M1')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--layouts', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_summary(args.single, args.paired, args.layouts, args.raw_root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
