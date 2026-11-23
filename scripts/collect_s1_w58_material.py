"""Extract three causal states at equal target clocks from complete archived robot recordings."""
import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from collect_s1_w57_material import PARAMETERS, state_at, targets
from s1_w55_c_data import DT, R12, ROOT, WEIGHTS, write
from s1_w58_material import W58MaterialObserver

SPEC_PATH = ROOT / 'experiments/terrain_memory/w58/material.json'
SPEC = json.loads(SPEC_PATH.read_text())
SPLITS = SPEC['recorded_episodes']
MODES = ('M0', 'M1', 'M2')
RAW = ROOT.parent / 'MCP-RL-w57/outputs' / R12
WORK = ROOT / 'work/w58-m2'
OUT = ROOT / 'results' / R12 / 'w58/m2'
BG = SPEC['background']


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def measured_episode(base):
    blocks = sorted(base.glob('material-[0-9]*.npz'))
    foot = []
    for path in blocks:
        with np.load(path) as z:
            foot.append(z['foot'])
    measured = np.concatenate(foot)
    return blocks, measured, [digest(v[:, 6:42]) for v in foot]


def dump_gz(path, value):
    temp = path.with_suffix('.pending')
    with gzip.open(temp, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
    temp.replace(path)


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def extract(episode):
    started, cpu_start = time.time(), time.process_time()
    base = RAW / episode
    blocks, foot, fingerprints = measured_episode(base)
    rows, eligible = targets(base, foot)
    groups = {m: deepcopy(rows) for m in MODES}
    queries, releases = defaultdict(list), defaultdict(list)
    for k, row in enumerate(rows):
        queries[row['start']].append(k)
        if row['release']:
            release_tick = round(row['release']['time_s']/DT)
            publication = 10*((release_tick+9)//10)
            if publication <= row['event_tick']:
                releases[publication].append(k)
    obs = W58MaterialObserver(PARAMETERS, WEIGHTS)
    current = {m: [] for m in MODES}
    max_contact_discrepancy = 0.
    for tick in range(0, len(foot), 10):
        grounds = {'M0': obs.prior.ground, 'M1': obs.contact_ground, 'M2': obs.ground}
        for mode in MODES:
            for i in queries.get(tick, ()):
                row = groups[mode][i]
                row.update(observed_state=state_at(grounds[mode], row['keys']),
                           publication_id=obs.windows, publication_s=obs.time)
            for i in releases.get(tick, ()):
                row = groups[mode][i]
                history_observer = obs.prior if mode == 'M0' else obs
                row.update(release_publication_tick=tick,
                           release_state=state_at(grounds[mode], [row['key']])[0],
                           source_observed_history=history_observer.history(row['key']))
                if mode == 'M2':
                    row['released_path_difference'] = obs.released_state_sensitivity(row['key'])
        for v in foot[tick:tick+10]:
            obs.completed(v[0], v[1], v[6:18].reshape(4,3), v[18:30].reshape(4,3), v[32:42:3])
        obs.observe(v[1], None, None)
        if not (tick+10) % 25000:
            print(f'{episode} boundary {obs.time:.1f} s; {obs.windows} publications', flush=True)
        owner = obs.last_areas[-1] > 0
        if owner.any():
            actual = v[32:42:3][owner]
            before = obs.last_correction[owner, 5]
            fused = obs.last_forces[-1, owner]
            current['M0'].extend((obs.prior.last_forces[-1, owner] - actual).tolist())
            current['M1'].extend((before - actual).tolist())
            current['M2'].extend((fused - actual).tolist())
            max_contact_discrepancy = max(max_contact_discrepancy, float(np.max(np.abs(fused-before))))
    for mode in MODES:
        dump_gz(WORK / f'{episode}-{mode}-starts.json.gz', groups[mode])
    summary = dict(episode=episode, ended_physical_s=obs.time, foot_substeps=len(foot),
        publications=obs.windows, event_families_raw_counts=eligible, sampled_event_count=len(rows),
        matched_release_publications=sum(map(len, releases.values())),
        released_path_selected=sum(v.get('released_path_difference') is not None for v in groups['M2']),
        chunk_foot_sha256=fingerprints, event_stream_sha256=file_sha256(base/'material-events.jsonl'),
        online_current_foot_observations=len(current['M1']),
        current_force_MAE_N={m:float(np.mean(np.abs(current[m]))) for m in MODES},
        current_max_M2_minus_M1_N=max_contact_discrepancy,
        observer=obs.timing(), build_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in
            (WEIGHTS, ROOT/'work/w56-r60/columns.so', ROOT/'work/w57-m/material.so')},
        started_unix_s=started, ended_unix_s=time.time(), host_cpu_core_s=time.process_time()-cpu_start,
        max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        process_id=os.getpid(), affinity=sorted(os.sched_getaffinity(0)), **BG)
    write(OUT / f'{episode}-replay.json', summary)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', choices=SPLITS['calibration']+SPLITS['retained'])
    args=parser.parse_args()
    WORK.mkdir(exist_ok=True, parents=True)
    OUT.mkdir(exist_ok=True, parents=True)
    extract(args.episode)
