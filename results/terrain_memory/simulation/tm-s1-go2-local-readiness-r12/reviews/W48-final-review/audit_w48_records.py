#!/usr/bin/env python3
"""Offline audit of W48 committed record metadata; no simulation/network/payloads."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


def maxabs(a):
    return max(map(abs, a), default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo
    base = repo / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
    sources = []

    def read(p):
        p = Path(p)
        b = p.read_bytes()
        sources.append(dict(path=str(p.relative_to(repo)), bytes=len(b), sha256=hashlib.sha256(b).hexdigest()))
        return json.loads(b)

    phase = read(base / 'w48/phase.json')
    comparison = read(base / 'w48/original-comparison.json')
    with (base / 'w47/task-c/planner-calls.csv').open() as f:
        prior = {(r['case'], int(r['call_id'])): r for r in csv.DictReader(f)}
    metrics = ['base_position_m', 'base_orientation_rad', 'joint_position_rad',
               'base_linear_velocity_m_s', 'base_angular_velocity_rad_s', 'joint_velocity_rad_s']
    cases = {}
    starts = []
    ends = []
    shared_states = []
    for case, seed in [('a', 1), ('b', 2)]:
        name = f'job-20261113T090632Z-w48-{case}-eval-PANN_PPO_S5-s{seed}-E30-attempt01'
        resolved = read(base / (name + '-resolved.json'))
        evidence = read(base / 'w48/original-evidence' / (name + '.json'))
        stream = read(base / 'w48' / ('w48-' + case + '.json'))
        rec = resolved['r29_recording']
        rows = rec['event_calls']
        by_call = {x['boundary']['call_id']: x for x in rows}
        assert len(by_call) == len(rows)
        source_catalog = read(base / 'w48/delivery' / ('job-20261113T090632Z-w48-' + case) / 'sources.json')
        catalog = {x['source_path']: x for x in source_catalog}
        object_by_file = {x['file']: x for x in rec['objects']}
        assert len(object_by_file) == len(rec['objects'])
        assert len({x['content_sha256'] for x in rec['objects']}) == len(rec['objects'])
        assert sum(x['bytes'] for x in rec['objects']) == rec['actual_bytes'] == evidence['object_bytes']
        for obj in rec['objects']:
            matching = [x for p, x in catalog.items() if p.endswith('/' + name + '/' + obj['file'])]
            assert len(matching) == 1
            assert matching[0]['sha256'] == obj['file_sha256'] and matching[0]['size'] == obj['bytes']
        counts = collections.Counter()
        ages = collections.Counter()
        source_missing = []
        arithmetic_errors = []
        aligned_time_errors = []
        cost_errors = []
        prior_cost_difference_calls = []
        mismatch = []
        compact_rows = []
        for row in rows:
            c, call = row['boundary'], row['call_record']
            cid = c['call_id']
            assert c['origin_s'] == call['origin'] and cid == call['call_id']
            assert c['installed_plan_id'] == call['installed_plan_id']
            assert set(row['objects']) == {'policy', 'accepted', 'p0', 'context', 'reference'}
            for role, meta in row['objects'].items():
                assert meta == object_by_file[meta['file']]
            assert c['installed_policy_sha256'] == row['objects']['policy']['content_sha256']
            assert c['accepted_rollout_sha256'] == row['objects']['accepted']['content_sha256']
            assert c['P0']['policy_sha256'] == row['objects']['p0']['content_sha256']
            assert c['actual_context_sha256'] == row['objects']['context']['content_sha256']
            assert c['r35_reference_sha256'] == row['objects']['reference']['content_sha256']
            old = prior[(case, cid)]
            for key in ['candidate_source', 'stop_reason', 'return_reason']:
                if old[key] != call[key]: mismatch.append((cid, key))
            for key in ['origin', 'chosen_alpha', 'installed_plan_id', 'native_new_rollouts',
                        'candidate_records', 'derivative_passes', 'transition_fd_calls', 'risk_geometry_fd_calls']:
                if float(old[key]) != call[key]: mismatch.append((cid, key))
            if (old['policy_updated'] == 'True') != call['policy_updated']:
                mismatch.append((cid, 'policy_updated'))
            if float(old['chosen_actual']) != call['chosen_actual']:
                prior_cost_difference_calls.append(cid)
            current = c['current_solve']
            err = sum(current['cost_components'].values()) - current['J']
            assert err == current['cost_recompute_error']
            assert abs(current['J0'] + current['Jref'] - current['J']) < 1e-8 * max(1, abs(current['J']))
            cost_errors.append(abs(err))
            p = c['previous_prediction']
            counts[p['status']] += 1
            summary = dict(call_id=cid, origin_s=c['origin_s'], plan_id=c['installed_plan_id'],
                           candidate=call['candidate_source'], alpha=call['chosen_alpha'],
                           chosen_actual=call['chosen_actual'], stop=call['stop_reason'],
                           current_cost=current['J'], Jref=current['Jref'])
            if p['status'] == 'available':
                assert p['source_call'] == cid - 1
                assert p['policy_sha256'] == c['P0']['policy_sha256']
                assert p['source_plan'] == c['P0']['plan_id']
                counts['node_' + str(p['node'])] += 1
                aligned_time_errors.append(abs(c['origin_s'] - p['node_time_s']))
                x, y = c['actual_state'], p['predicted_state']
                # Translation and hinge positions are Euclidean; quaternion tangents are
                # retained evidence, not independently recomputed by a MuJoCo engine here.
                q = [x[i] - y[i] for i in range(3)] + [x[i] - y[i] for i in range(7, 19)]
                qr = p['base_position_m'] + p['joint_position_rad']
                dv = [x[i] - y[i] for i in range(19, 37)]
                arithmetic_errors += [abs(a - b) for a, b in zip(q, qr)]
                arithmetic_errors += [abs(a - b) for a, b in zip(dv, p['velocity_difference'])]
                assert p['position_tangent'][:3] == p['base_position_m']
                assert p['position_tangent'][3:6] == p['base_orientation_rad']
                assert p['position_tangent'][6:] == p['joint_position_rad']
                assert p['velocity_difference'][:3] == p['base_linear_velocity_m_s']
                assert p['velocity_difference'][3:6] == p['base_angular_velocity_rad_s']
                assert p['velocity_difference'][6:] == p['joint_velocity_rad_s']
                if p['source_call'] in by_call:
                    src = by_call[p['source_call']]['boundary']
                    assert src['installed_policy_sha256'] == p['policy_sha256']
                    assert src['accepted_rollout_sha256'] == p['accepted_sha256']
                    ages[round(c['origin_s'] - src['origin_s'], 9)] += 1
                    counts['source_call_also_in_committed_event_subset'] += 1
                else: source_missing.append(p['source_call'])
                summary.update(prediction_source_call=p['source_call'], prediction_source_plan=p['source_plan'],
                               prediction_node=p['node'], errors={k: maxabs(p[k]) for k in metrics})
            compact_rows.append(summary)
        assert not mismatch, mismatch
        assert max(arithmetic_errors) == 0 and max(aligned_time_errors) == 0
        events = []
        for event in rec['events']:
            t = event['time_s']
            if event['kind'] not in ('phase_transition', 'first_actual_held_jump_over_1rad',
                                     'first_actual_knee_limit_reversal', 'terminal'): continue
            pre = [r for r in compact_rows if r['origin_s'] <= t + 1e-9]
            chosen = pre[-1]
            assert all(any(w['start_s'] <= t <= w['end_available_s'] for w in rec['merged_windows'])
                       for _ in [0])
            events.append(dict(event=event, preceding_saved_boundary=chosen,
                               boundary_to_event_s=t-chosen['origin_s'],
                               scope='Errors are at preceding planner boundary, not automatically at event time.'))
        first = evidence['first_nonzero_same_time_prediction_error']
        assert by_call[first['call_id']]['boundary'] == first
        shared_states.append(first['actual_state'])
        starts.append(stream['stages'][0]['started_unix_s'])
        ends.append(stream['stages'][0]['ended_unix_s'])
        cases[case.upper()] = dict(
            original_status=evidence['status'], original_result=evidence['original_result'],
            calls=evidence['planner_calls'], prediction_pairs=evidence['prediction_pairs'],
            startup_unavailable=evidence['prediction_unavailable'],
            positive_installs=evidence['positive_installs'],
            positive_cost_descent=evidence['positive_same_problem_cost_descent'],
            positive_risk_increase=evidence['positive_risk_priority_cost_increase'],
            stagnation=evidence['internal_stagnation'],
            work=evidence['work_completed_calls'],
            full_call_error_extrema_as_committed=evidence['prediction_error_extrema'],
            saved_event_calls=len(rows), saved_objects=len(rec['objects']), object_bytes=rec['actual_bytes'],
            event_subsets_independently_verified=dict(counts),
            source_ages_s=[dict(age_s=k, count=v) for k,v in sorted(ages.items())],
            previous_sources_before_saved_window=source_missing,
            maximum_time_alignment_error_s=max(aligned_time_errors),
            max_Euclidean_position_velocity_arithmetic_error=max(arithmetic_errors),
            max_event_cost_recompute_error=max(cost_errors),
            event_prior_W46_choice_identity_mismatches=mismatch,
            event_prior_W46_chosen_actual_difference_calls=prior_cost_difference_calls,
            key_events=events, event_boundary_rows=compact_rows,
            initial_same_attempt_record_readback=evidence['same_attempt_record_readback'],
            runtime_process=resolved['w48_native_process'],
        )
    assert shared_states[0] == shared_states[1]
    work = {key: sum(x['work'][key] for x in cases.values()) for key in cases['A']['work']}
    assert work == phase['actual_work_final']['original_work']
    assert sum(x['prediction_pairs'] for x in cases.values()) == 1904
    assert sum(x['calls'] for x in cases.values()) == 1944
    gpu = read(base / 'w48/task-c/duplicates.json')
    assert len(gpu['rows']) == gpu['duplicate_groups'] == 6
    gpu_metrics = ['max_base_position_m', 'max_orientation_rad', 'max_joint_position_rad',
                   'max_linear_velocity_m_s', 'max_angular_velocity_rad_s',
                   'max_joint_velocity_rad_s', 'max_held_difference_rad']
    for row in gpu['rows']:
        assert row['native']['initial_state_bitwise_equal'] and row['gpu']['initial_state_bitwise_equal']
        assert row['native']['first_state_node'] is None
        assert row['native']['first_held_node'] is None
        assert all(row['native'][k] == 0 for k in gpu_metrics)
        assert row['gpu']['first_state_node'] == row['gpu']['first_held_node'] == 1
        assert row['gpu']['first_state_after_origin_s'] == .002
    delivery = {}
    all_current_sources = []
    for label in ['finite', 'job-20261113T090632Z-w48-a', 'job-20261113T090632Z-w48-b']:
        d = read(base / 'w48/delivery' / label / 'delivery.json')
        cat = read(base / 'w48/delivery' / label / 'sources.json')
        all_current_sources.extend(cat)
        assert d['status'] == 'published_size_digest_verified'
        assert sum(a['size'] for a in d['assets']) == d['total_bytes']
        for a in d['assets']:
            assert a['verified'] and a['remote_digest'] == 'sha256:' + a['sha256']
        delivery[label] = dict(status=d['status'], assets=len(d['assets']), bytes=d['total_bytes'])
    covered_paths = {x['source_path'] for x in all_current_sources}
    covered_hashes = {x['sha256'] for x in all_current_sources}
    pending = []
    for label in ['w48-a', 'w48-b']:
        d = read(base / 'w48/delivery' / label / 'delivery.json')
        cat = read(base / 'w48/delivery' / label / 'sources.json')
        manifest = read(base / 'w48/delivery' / label / 'manifest.json')
        pending.append(dict(label=label, status=d['status'], upload_exit=d.get('upload_exit_code'),
                            archive_count=len(manifest), sources=len(cat), source_bytes=sum(x['size'] for x in cat),
                            exact_paths_covered_by_later_manifests=sum(x['source_path'] in covered_paths for x in cat),
                            byte_identity_covered_by_later_manifests=sum(x['sha256'] in covered_hashes for x in cat),
                            uncovered_nonempty_sources=[x for x in cat if x['size'] and x['sha256'] not in covered_hashes]))
    tech = read(base / 'w48/startup-technical.json')
    r8 = read(base / 'w48/task-c/r8-source.json')
    out = dict(schema='w48_committed_records_gpu_audit_v1',
        reviewed_commit='61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6',
        scope='Independent arithmetic/source review of committed JSON/CSV and event manifests. No server, physics, GPU, payload download or raw trajectory replay.',
        cases=cases, total_work=work,
        cases_first_20ms_actual_states_equal=True,
        simultaneous_original_process_interval_s=max(0, min(ends)-max(starts)),
        simultaneous_scope='Overlap of recorded original-evaluation child lifetimes, not measured per-core utilization.',
        GPU=dict(duplicate_groups=6, native_duplicates_exact_all_saved_times=True,
                 GPU_first_differences_all_at_2ms=True,
                 max_reported_duplicate_errors={k:max(row['gpu'][k] for row in gpu['rows']) for k in gpu_metrics},
                 rows=gpu['rows'], missing=gpu['missing_fields'], new_physics=0, new_GPU_allocations=0,
                 wall_s=gpu['wall_s']),
        R8=r8, scientific_delivery=delivery,
        scientific_assets=sum(x['assets'] for x in delivery.values()),
        scientific_bytes=sum(x['bytes'] for x in delivery.values()),
        initial_technical_delivery_manifests=pending,
        technical_startup=tech,
        comparison_metadata=comparison,
        source_files=sources,
        limitations=['Full 1904 boundary JSONL is in server/Release originals; only 237 available event boundaries were independently audited here.',
                     'Quaternion tangent values and paired trajectory exact-equality claims are submitted evidence, not independent native replays.',
                     '20ms prior-prediction error includes intervening closed-loop execution and does not identify a unique dynamics cause.',
                     'Missing per-substep/raw GPU initialization fields prevent a unique numerical vs hidden-state attribution.',
                     'Delivery metadata from Git is not a new live GitHub asset lookup; old technical manifests require separate remote reconciliation.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(calls=1944,prediction_pairs=1904,event_calls=sum(x['saved_event_calls'] for x in cases.values()),
                         event_available=sum(x['event_subsets_independently_verified']['available'] for x in cases.values()),
                         event_objects=sum(x['saved_objects'] for x in cases.values()),
                         original_overlap_s=out['simultaneous_original_process_interval_s'],
                         assets=out['scientific_assets'],bytes=out['scientific_bytes'],
                         pending_technical=out['initial_technical_delivery_manifests']),ensure_ascii=False))


if __name__ == '__main__':
    main()
