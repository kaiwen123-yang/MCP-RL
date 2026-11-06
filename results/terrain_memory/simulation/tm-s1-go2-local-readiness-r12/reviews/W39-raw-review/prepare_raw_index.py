#!/usr/bin/env python3
"""Map the immutable W39 Release manifests to an existing local asset directory.

This only writes an index. It does not download, run physics, or delete anything.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    args = p.parse_args()
    raw = args.raw.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    base = here.parent.parent
    records = []
    for branch in ('a', 'b'):
        evidence = json.loads((base / 'jobs' / f'job-20261105T032354Z-w39-{branch}' / 'diagnostics/evidence.json').read_text())
        assets = {a['name']: a for a in evidence['assets']}
        for f in evidence['files']:
            name = Path(f['source_path']).name
            attempt = f.get('attempt') or 'training'
            parts = []
            for part in f['parts']:
                a = assets[part['name']]
                parts.append({'part': part['part'], 'name': a['name'], 'size': a['size'], 'url': a['url'], 'local_path': str(raw / a['name'])})
            records.append({'branch': branch, 'file_id': f['file_id'], 'attempt': attempt, 'name': name, 'source_path': f['source_path'], 'source_size': f['source_size'], 'codec': f['codec'], 'parts': parts, 'selected': name != 'model.mjb' or 'E30' in attempt})
    (raw / 'raw-index.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'files': len(records), 'selected': sum(r['selected'] for r in records), 'index': str(raw / 'raw-index.json')}))


if __name__ == '__main__':
    main()
