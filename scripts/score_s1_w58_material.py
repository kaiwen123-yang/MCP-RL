"""Score future loading and free recovery conditional on actual held-out robot motion."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from collect_s1_w58_material import BG, MODES, OUT, RAW, ROOT, SPLITS, SPEC_PATH, WEIGHTS, WORK
from collect_s1_w58_material import dump_gz, file_sha256, measured_episode, read_gz
from s1_w55_c_data import DT, R12, write
from s1_w60_cpu_observer import CPUColumns
from score_s1_w57_material import forward_bootstrap, reload, truth_at


def release_information(groups, blocks):
    requests = [dict(event_id=q['event_id'], target=q['release_publication_tick'], keys=[q['key']])
                for q in groups['M0'] if 'release_state' in q]
    truth = truth_at(blocks, requests)
    result = {}
    for mode, queries in groups.items():
        for q in queries:
            if 'release_state' not in q:
                continue
            eid = q['event_id']
            true_state = truth[eid][0]
            prior = q['release_state']
            source = q['release']
            observed = q['source_observed_history']
            matched = (observed['last_owner'] == source['leg'] and not observed['loaded']
                       and observed['release_time_s'] is not None and
                       abs(observed['release_time_s']-source['time_s']) <= DT+1e-8)
            result[mode, eid] = dict(source_leg=source['leg'], target_leg=q['leg'],
                cross_foot=bool(source['leg'] != q['leg']), source_release_s=source['time_s'],
                physical_free_s=q['event']['time_s']-source['time_s'], source_binding_one_tick=matched,
                source_foot_normal_force_before_N=source['normal_before_N'],
                target_cell_normal_force_N=q['event']['normal_N'],
                source_depth_error_m=float(-.05*(prior[0]-true_state[0])),
                source_q11_error=float(prior[2]-true_state[2]))
    return result


def add_release_metrics(rows, source):
    for row in rows:
        item = source[row['mode'], row['event_id']].copy()
        e = row['errors']
        e['source_depth_m'] = item.pop('source_depth_error_m')
        e['source_internal_q11'] = item.pop('source_q11_error')
        row.update(item)


def summarize_errors(rows):
    table = {}
    for key in sorted({k for r in rows for k in r['errors']}):
        v = np.array([r['errors'][key] for r in rows if key in r['errors']], dtype=float)
        table[key] = dict(n=len(v), bias=float(np.mean(v)), mae=float(np.mean(np.abs(v))),
                          abs_p95=float(np.quantile(np.abs(v), .95)))
    return table


def primary(rows):
    return [row for row in rows if row['cross_foot']]


def score(episode):
    import torch
    start, cpu = time.time(), time.process_time()
    source = RAW / episode
    blocks, foot, _ = measured_episode(source)
    groups = {m:read_gz(WORK / f'{episode}-{m}-starts.json.gz') for m in MODES}
    law = CPUColumns(WEIGHTS)
    # Two independent forecast batches use the existing measured-boundary CUDA ABI.
    # Keep small batches on the low priority stream of this process.
    stream = torch.cuda.Stream(priority=0)
    with torch.cuda.stream(stream):
        future, ft = forward_bootstrap(episode, groups, foot, blocks, source, law)
        released, rt = reload(episode, groups, foot, law)
    enrich = release_information(groups, blocks)
    add_release_metrics(released, enrich)
    result = dict(episode=episode, forward=future, release_to_reload=released)
    dump_gz(OUT / f'{episode}-forecasts.json.gz', result)
    write(OUT / f'{episode}-forecast-cost.json', dict(episode=episode, forward=ft, free=rt,
        end_to_end_including_data_assembly_and_io_s=time.time()-start, host_cpu_core_s=time.process_time()-cpu,
        compilation_s_this_run=0., reused_library_sha256={p:file_sha256(ROOT/'work/w57-m'/p)
                for p in ('forecast_cpu.so','forecast_gpu.so')},
        max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        affinity=sorted(os.sched_getaffinity(0)), cuda_stream_priority=stream.priority, **BG))


def development():
    """Use archived source states and exact equal-input forecast rows on the two development recordings."""
    w57 = ROOT.parent / 'MCP-RL-w57'
    w56 = ROOT.parent / 'MCP-RL-w56-r60/outputs' / R12
    data = {}
    for case, episode in zip(('a','b'), SPLITS['development']):
        base = w56 / episode
        blocks, foot, signatures = measured_episode(base)
        groups = {m:read_gz(w57/'outputs'/R12/'w57/m'/f'{case}-{m}-starts.json.gz') for m in ('M0','M1')}
        archive = read_gz(w57/'results'/R12/'w57/m'/f'{case}-held-forecasts.json.gz')
        old = {(row['mode'],row['event_id']):row for row in archive['release_to_reload']}
        release = release_information(groups, blocks)
        rows = []
        for a, b in zip(groups['M0'], groups['M1']):
            if a['event_id'] != b['event_id']:
                raise ValueError('Paired histories have distinct event IDs')
            if 'release_state' not in a:
                continue
            eid = a['event_id']
            aa, bb = a['source_observed_history'], b['source_observed_history']
            use_prior = bb['ever_loaded'] and bb['release_time_s'] is not None and not bb['loaded'] and not aa['loaded']
            chosen = 'M0' if use_prior else 'M1'
            release['M2', eid] = release[chosen, eid].copy()
            # The M2 owner/event timestamps always come from the supported estimator.
            release['M2', eid]['source_binding_one_tick'] = release['M1', eid]['source_binding_one_tick']
            for m, src in [('M0','M0'), ('M1','M1'), ('M2',chosen)]:
                row = dict(old[src,eid], mode=m, errors=old[src,eid]['errors'].copy())
                row['M2_selected_release_prior'] = bool(use_prior)
                rows.append(row)
        add_release_metrics(rows, release)
        per_mode = {}
        for m in MODES:
            targets = primary([r for r in rows if r['mode']==m])
            gaps = [v['physical_free_s'] for v in targets]
            per_mode[m] = dict(cross_foot=summarize_errors(targets), events=len(targets),
                gap_min_median_max_s=np.quantile(gaps,[0,.5,1]).tolist() if gaps else None,
                matched_causal_release=sum(r['source_binding_one_tick'] for r in targets),
                chose_prior=sum(r['M2_selected_release_prior'] for r in targets))
        data[episode] = dict(families=per_mode, chunk_foot_sha256=signatures,
            raw_foot_substeps=len(foot), source_event_sha256=file_sha256(base/'material-events.jsonl'))
    write(OUT/'development.json', dict(role='W56 A/B development only', split_commit='3164877e',
        computation='Identical source state and recorded free-boundary inputs imply bit-identical M2 forecast '
                    'to the selected archived M0/M1 forecast. True source state sampled at the same 500 Hz publication.',
        by_episode=data, **BG))


def groups_for_score(data, mode):
    groups=defaultdict(list)
    for row in data['forward']:
        if row['mode'] != mode:
            continue
        groups['query_20p2ms/'+row['family']].append(row)
        if row['family'] != 'release':
            groups['query_20p2ms/all_nonrelease'].append(row)
        if row['continuously_unloaded_after_actual_release']:
            groups['query_20p2ms/continuous_free_recovery'].append(row)
    for row in data['release_to_reload']:
        if row['mode'] != mode:
            continue
        owner='cross_foot' if row['cross_foot'] else 'same_foot'
        groups['source_release/'+owner].append(row)
        if row['physical_free_s'] >= 6:
            groups['source_release/free_at_least_6s'].append(row)
        if row['source_binding_one_tick']:
            groups['source_release/matched_'+owner].append(row)
    return groups


def compare_prefix(one, other, signatures):
    p, q = signatures[one], signatures[other]
    count = 0
    for a,b in zip(p,q):
        if a != b:
            break
        count += 1
    def chunk(episode, index):
        root=(ROOT.parent/'MCP-RL-w56-r60/outputs'/R12 if episode in SPLITS['development'] else RAW)
        files=sorted((root/episode).glob('material-[0-9]*.npz'))
        with np.load(files[index]) as z:
            return z['foot'][:,6:42]
    if count < min(len(p),len(q)):
        a,b=chunk(one,count),chunk(other,count)
        common=min(len(a),len(b))
        diff=np.flatnonzero(np.any(a[:common]!=b[:common],axis=1))
        steps=count*5000+(int(diff[0]) if len(diff) else common)
    else:
        steps=count*5000
    return dict(other_episode=other, identical_5kHz_motion_and_force_prefix_steps=steps,
                prefix_s=DT*steps)


def finish():
    calibration, = SPLITS['calibration']
    dev = json.loads((OUT/'development.json').read_text())['by_episode']
    signatures = {ep:v['chunk_foot_sha256'] for ep,v in dev.items()}
    sources = [*SPLITS['calibration'], *SPLITS['retained']]
    replays, tables, bands, counts, costs, overlaps, adoption, material_rows = {}, {}, {}, {}, {}, {}, {}, {}
    for ep in sources:
        meta = json.loads((OUT/f'{ep}-replay.json').read_text())
        signatures[ep] = meta['chunk_foot_sha256']
        replays[ep] = meta
        overlaps[ep] = [compare_prefix(ep,other,signatures) for other in signatures if other!=ep]
        data=read_gz(OUT / f'{ep}-forecasts.json.gz')
        costs[ep]=json.loads((OUT/f'{ep}-forecast-cost.json').read_text())
        material_rows[ep] = str((OUT/f'{ep}-forecasts.json.gz').relative_to(ROOT))
        tables[ep], counts[ep] = {}, {}
        for m in MODES:
            groups=groups_for_score(data,m)
            tables[ep][m],counts[ep][m] = {}, {}
            for group, rows in groups.items():
                stats=summarize_errors(rows)
                counts[ep][m][group] = len(rows)
                for key,values in stats.items():
                    index=(m,group,key)
                    v=np.abs([r['errors'][key] for r in rows if key in r['errors']])
                    if ep==calibration:
                        bands[index]=float(np.quantile(v,.95))
                    radius=bands.get(index)
                    values['band_full_width']=None if radius is None else 2*radius
                    values['covered']=None if radius is None else int(np.count_nonzero(v<=radius))
                    values['coverage']=None if radius is None else float(np.mean(v<=radius))
                tables[ep][m][group]=stats
        if ep in SPLITS['retained']:
            t=tables[ep]
            present=all('source_release/cross_foot' in t[m] for m in MODES)
            d={m:t[m]['query_20p2ms/all_nonrelease']['future_force_N']['mae'] for m in MODES}
            def cross(m,k):
                return t[m]['source_release/cross_foot'][k]['mae']
            adoption[ep]=dict(cross_foot_release_rows=present,
                cross_foot_force_improves=present and cross('M2','next_cell_force_N')<cross('M1','next_cell_force_N'),
                cross_foot_preload_depth_preserved=present and cross('M2','preload_depth_m')<=cross('M1','preload_depth_m'),
                nonrelease_20p2ms_force_preserved=d['M2']<=d['M1']+.1*abs(d['M0']-d['M1']),
                current_force_preserved=meta['current_max_M2_minus_M1_N']<=1e-10)
    write(OUT/'errors-and-coverage.json',dict(physical_task='offline conditional forecast', counts=counts,
        bands_from=calibration, intervals='Symmetric absolute-error bands, full width in the units of each metric.',
        groups='Each within-family query is a representative interval/cell. Long and source-matched subsets overlap '
               'the corresponding primary group; their counts must not be added.',by_episode=tables))
    artifacts=[]
    for ep in sources:
        for m in MODES:
            p=WORK/f'{ep}-{m}-starts.json.gz'
            artifacts.append(dict(path=str(p),size=p.stat().st_size,sha256=file_sha256(p)))
    selection=ROOT/'results'/R12/'w58/m/selection.json'
    selection.parent.mkdir(parents=True,exist_ok=True)
    write(selection, dict(schema='W58-M2-release-selection-v1',
        selected='M2' if all(all(c.values()) for c in adoption.values()) else 'M1',
        spec=str(SPEC_PATH.relative_to(ROOT)), source_episodes=SPLITS, conditions_by_retained_episode=adoption,
        matched_input_scope='Recorded complete histories in one nominal material setting, including distinct '
          'W57 training acquisitions. Hardware and cross-domain performance have not been evaluated.',
        raw_artifacts=artifacts, per_event_forecasts=material_rows,
        conditional_error_table=str((OUT/'errors-and-coverage.json').relative_to(ROOT)),
        development_evidence=str((OUT/'development.json').relative_to(ROOT)),
        identical_observation_prefixes=overlaps,
        actual_GPU_paths=sum(c[k]['paths'] for c in costs.values() for k in ('forward','free')),
        actual_GPU_column_steps=sum(c[k]['logical_column_steps'] for c in costs.values() for k in ('forward','free')),
        replay_cpu_core_s=sum(v['host_cpu_core_s'] for v in replays.values()),
        forecast_cpu_core_s=sum(v['host_cpu_core_s'] for v in costs.values()),
        estimator_only_new_robot_episodes=0, new_PPO_distinct=0, **BG))
    print(selection.read_text(), flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['development','score','finish','follow'])
    p.add_argument('--episode',choices=SPLITS['calibration']+SPLITS['retained'])
    a=p.parse_args()
    if a.stage=='development':
        development()
    elif a.stage=='score':
        score(a.episode)
    elif a.stage=='finish':
        finish()
    else:
        if not (OUT/'development.json').exists():
            development()
        for ep in SPLITS['calibration']+SPLITS['retained']:
            while not (OUT/f'{ep}-replay.json').is_file():
                time.sleep(20)
            if not (OUT/f'{ep}-forecast-cost.json').exists():
                print('Scoring',ep,flush=True)
                score(ep)
        finish()
