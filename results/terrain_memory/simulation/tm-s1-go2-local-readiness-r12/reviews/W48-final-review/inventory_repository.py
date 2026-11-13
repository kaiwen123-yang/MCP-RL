#!/usr/bin/env python3
"""Read and hash every tracked file; inventory the W48 delta without simulation."""
import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--base', default='068d8388feba972dd0e8cbd5bbbb1a7fd3ee71bf')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    def git(*words):
        return subprocess.check_output(['git', '-C', str(repo), *words])
    head = git('rev-parse', 'HEAD').decode().strip()
    paths = git('ls-files', '-z').decode().rstrip('\0').split('\0')
    changed = set(git('diff', '--name-only', '-z', args.base, head).decode().rstrip('\0').split('\0'))
    totals = collections.Counter()
    kinds = collections.defaultdict(collections.Counter)
    tree_digest = hashlib.sha256()
    delta = []
    errors = []
    for rel in paths:
        path = repo / rel
        try:
            digest = hashlib.sha256()
            size = 0
            with path.open('rb') as f:
                while chunk := f.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            sha = digest.hexdigest()
            tree_digest.update(rel.encode() + b'\0' + str(size).encode() + b'\0' + sha.encode() + b'\n')
            totals['files'] += 1
            totals['bytes'] += size
            ext = path.suffix or '(no suffix)'
            kinds[ext]['files'] += 1
            kinds[ext]['bytes'] += size
            if rel in changed:
                row = {'path': rel, 'bytes': size, 'sha256': sha}
                if ext in {'.json', '.jsonl'}:
                    text = path.read_text()
                    if ext == '.json':
                        parsed = json.loads(text)
                        row['json_type'] = type(parsed).__name__
                        row['top_level_entries'] = len(parsed) if isinstance(parsed, (dict, list)) else None
                    else:
                        row['jsonl_rows'] = sum(bool(line.strip()) and json.loads(line) is not None for line in text.splitlines())
                    row['structured_parse'] = 'passed'
                elif ext in {'.py', '.md', '.patch', '.cc', '.h', '.txt'}:
                    row['utf8_lines'] = len(path.read_text().splitlines())
                delta.append(row)
        except Exception as exc:
            errors.append({'path': rel, 'error': repr(exc)})
    result = {
        'schema': 'w48_repository_coverage_v1',
        'base': args.base, 'head': head,
        'method': 'Every tracked file streamed in full for SHA256; changed JSON/JSONL parsed in full. This is byte/structure coverage, not a claim to semantic understanding of every byte or replay of every Release.',
        'totals': dict(totals), 'tracked_tree_manifest_sha256': tree_digest.hexdigest(),
        'extensions': dict(sorted(kinds.items())), 'delta_files': len(delta),
        'delta_bytes': sum(row['bytes'] for row in delta), 'delta': delta,
        'errors': errors,
        'excluded': ['Git object database', 'untracked scratch', 'remote Release payloads except separately evidenced selected originals'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('head', 'totals', 'delta_files', 'delta_bytes', 'errors')}))


if __name__ == '__main__':
    main()
