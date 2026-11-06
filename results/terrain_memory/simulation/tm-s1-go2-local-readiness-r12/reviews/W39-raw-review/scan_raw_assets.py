#!/usr/bin/env python3
"""Read and inventory downloaded original assets, without simulating physics.

Usage: python scan_raw_assets.py --index /path/raw-index.json --output coverage.json
The index is built from the two committed diagnostics/evidence.json manifests.
Source byte equality and parsing are coverage checks, not proof of model validity.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def inspect(record):
    parts = record['parts']
    out = {k: record[k] for k in ('branch', 'file_id', 'attempt', 'name', 'source_path', 'source_size', 'codec')}
    out['transfer_bytes'] = sum(p['size'] for p in parts)
    out['asset_urls'] = [p['url'] for p in parts]
    if not record['selected']:
        return out | {'status': 'inventoried_not_downloaded', 'reason': 'training model binary; two E30 binaries inspected separately'}
    if len(parts) != 1:
        return out | {'status': 'multipart_not_inspected'}
    p = Path(parts[0]['local_path'])
    if not p.exists() or p.stat().st_size != parts[0]['size']:
        return out | {'status': 'not_yet_available'}
    n = record['name']
    try:
        if n.endswith('.npz'):
            arrays = {}
            with np.load(p, allow_pickle=False) as z:
                for k in z.files:
                    a = z[k]
                    d = {'shape': list(a.shape), 'dtype': str(a.dtype), 'elements': int(a.size)}
                    if a.dtype.kind in 'fci':
                        d['nonfinite'] = int(np.count_nonzero(~np.isfinite(a)))
                    arrays[k] = d
            out['arrays'] = arrays
            out['decoded_source_bytes'] = p.stat().st_size
        elif record['codec'] == 'gzip':
            data = gzip.decompress(p.read_bytes())
            out['decoded_source_bytes'] = len(data)
            if n.endswith('.jsonl'):
                count = 0
                keys = collections.Counter()
                for line in data.splitlines():
                    if line.strip():
                        v = json.loads(line)
                        count += 1
                        if isinstance(v, dict):
                            keys.update(v.keys())
                out.update(jsonl_rows=count, top_level_key_counts=dict(keys))
            elif n.endswith('.json'):
                v = json.loads(data)
                out['top_level_type'] = type(v).__name__
                out['top_level_size'] = len(v) if isinstance(v, (dict, list)) else None
            elif n.endswith('.xml'):
                out['xml_root'] = ET.fromstring(data).tag
            else:
                data.decode('utf-8')
        elif n.endswith('.mjb'):
            h = hashlib.sha256()
            with p.open('rb') as f:
                while b := f.read(1024*1024):
                    h.update(b)
            out.update(decoded_source_bytes=p.stat().st_size, binary_sha256=h.hexdigest(), binary_interpretation='bytes read; no MuJoCo execution')
        elif n.endswith('.xml'):
            data = p.read_bytes()
            out.update(decoded_source_bytes=len(data), xml_root=ET.fromstring(data).tag)
        else:
            data = p.read_bytes()
            out['decoded_source_bytes'] = len(data)
            if n.endswith('.json'):
                json.loads(data)
            elif n.endswith('.jsonl'):
                out['jsonl_rows'] = sum(bool(line.strip()) and json.loads(line) is not None for line in data.splitlines())
            else:
                out['uninterpreted_bytes'] = True
        out['source_size_matches'] = out['decoded_source_bytes'] == record['source_size']
        out['status'] = 'read' if out['source_size_matches'] else 'source_size_mismatch'
    except Exception as exc:
        out.update(status='read_error', error=f'{type(exc).__name__}: {exc}')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = json.loads(args.index.read_text())
    # Resume inventory only for immutable completed records, never skip errors.
    previous = {}
    if args.output.exists():
        old = json.loads(args.output.read_text())
        previous = {(x['branch'], x['file_id']): x for x in old.get('files', []) if x['status'] == 'read'}
    result = [previous.get((r['branch'], r['file_id'])) or inspect(r) for r in records]
    counts = collections.Counter(r['status'] for r in result)
    nonfinite = collections.Counter()
    for r in result:
        for k, a in r.get('arrays', {}).items():
            if a.get('nonfinite'):
                nonfinite[r['name'] + ':' + k] += a['nonfinite']
    summary = {
        'source_commit': '07b0ee3e15c9f10e9b1a6a1de3997bec36f39556',
        'scope': 'W39 immutable Release diagnostics; parse/array coverage differs from causal or semantic review',
        'physics_experiments_run': False,
        'status_counts': dict(counts),
        'manifest_source_bytes': sum(r['source_size'] for r in result),
        'manifest_transfer_bytes': sum(r['transfer_bytes'] for r in result),
        'read_transfer_bytes': sum(r['transfer_bytes'] for r in result if r['status'] == 'read'),
        'read_source_bytes': sum(r['source_size'] for r in result if r['status'] == 'read'),
        'npz_files_read': sum('arrays' in r for r in result if r['status'] == 'read'),
        'npz_numeric_nonfinite': sum(a.get('nonfinite', 0) for r in result for a in r.get('arrays', {}).values()),
        'npz_elements_read': sum(a['elements'] for r in result for a in r.get('arrays', {}).values()),
        'npz_nonfinite_by_field': dict(nonfinite),
        'nonfinite_interpretation': 'material-final.npz:last_release uses NaN for never released cells (src/tm02/s1_contact.py); this sentinel is not an actor/state integration NaN. Any other nonfinite field requires separate review.',
        'jsonl_rows_read': sum(r.get('jsonl_rows', 0) for r in result),
        'excluded_scope': '62 training model.mjb binaries inventoried, not downloaded/read; historical unrecorded accepted/K not recoverable by upload',
        'files': result,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
