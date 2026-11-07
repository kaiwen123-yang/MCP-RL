#!/usr/bin/env python3
"""Read selected original ZIP members over HTTP Range and verify their CRC/size.

No server outputs or release assets are modified. Downloaded coverage is explicit;
listing a ZIP is not reported as reading all of its members.
"""
import argparse
import concurrent.futures
import io
import json
from pathlib import Path, PurePosixPath
import threading
import time
import urllib.request
import zipfile
import zlib


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        start = max(0, size - 2 * 1024 * 1024)
        self.tail_start = start
        self.tail = self.fetch(start, size)

    def fetch(self, start, end):
        if end <= start:
            return b''
        for attempt in range(4):
            try:
                request = urllib.request.Request(self.url, headers={
                    'Range': f'bytes={start}-{end-1}', 'User-Agent': 'W40-evidence-reader'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise IOError('Server did not honor the requested byte range')
                    expected = f'bytes {start}-{end-1}/{self.size}'
                    if response.headers.get('Content-Range') != expected:
                        raise IOError('Unexpected Content-Range')
                    data = response.read()
                    self.url = response.url  # Signed redirect stays in memory only.
                if len(data) != end-start:
                    raise IOError('Incomplete HTTP range')
                return data
            except Exception:
                if attempt == 3:
                    raise IOError(f'Failed range {start}:{end}') from None
                time.sleep(attempt+1)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos+n)
        start = self.pos
        self.pos = end
        if start >= self.tail_start:
            return self.tail[start-self.tail_start:end-self.tail_start]
        return self.fetch(start, end)


def selected(member):
    path = PurePosixPath(member)
    if path.name in ('feedback-actions.jsonl', 'planner-calls.jsonl'):
        return True
    if path.name.startswith('rollout-') and path.suffix == '.npz':
        return True
    if str(path.parent) in ('w40-a/ep00314', 'w40-b/evaluation-E30'):
        return path.name in ('planner-records.jsonl', 'reference-packets.json',
            'action-execution.jsonl', 'r29-event-manifest.json', 'trajectory.npz',
            'physical-000.npz', 'material-events.jsonl', 'material-failures.jsonl',
            'query-exposure.jsonl', 'qp-failure.json')
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    lock = threading.Lock()
    coverage = dict(source_manifest=str(args.manifest), source_commit=manifest['source_commit'],
        total_manifest_files=manifest['source_files'], total_archive_bytes=manifest['archive_bytes'],
        scope='Selected original members only; size and ZIP CRC verified after full decompression.',
        files=[], errors=[])

    def save():
        args.coverage.parent.mkdir(parents=True, exist_ok=True)
        args.coverage.write_text(json.dumps(coverage, indent=2)+'\n')

    def archive(item):
        rows = [row for row in item['files'] if selected(row['member'])]
        if not rows:
            return
        try:
            source = RemoteZip(item['url'], item['size'])
            with zipfile.ZipFile(source) as zipped:
                for row in rows:
                    member = row['member']
                    path = args.output/member
                    if not path.resolve().is_relative_to(args.output.resolve()):
                        raise ValueError('Unsafe ZIP member path')
                    expected_crc = int(row['zip_crc32'], 16)
                    if path.is_file():
                        data = path.read_bytes()
                        valid = len(data) == row['source_size'] and zlib.crc32(data) == expected_crc
                    else:
                        valid = False
                    if not valid:
                        data = zipped.read(member)
                        if len(data) != row['source_size'] or zlib.crc32(data) != expected_crc:
                            raise IOError('Member differs from manifest')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                    with lock:
                        coverage['files'].append(dict(member=member, archive=item['name'],
                            url=item['url'], bytes=len(data), crc32=row['zip_crc32'],
                            compressed_bytes=row['compressed_size'], status='read_and_crc_verified'))
                        save()
                        print(json.dumps(dict(member=member, bytes=len(data), completed=len(coverage['files']))), flush=True)
        except Exception as error:
            with lock:
                coverage['errors'].append(dict(archive=item['name'], error=str(error)))
                save()
                print(json.dumps(coverage['errors'][-1]), flush=True)
            raise

    requested = [r for a in manifest['archives'] for r in a['files'] if selected(r['member'])]
    coverage['requested_files'] = len(requested)
    coverage['requested_source_bytes'] = sum(r['source_size'] for r in requested)
    coverage['requested_compressed_bytes'] = sum(r['compressed_size'] for r in requested)
    save()
    print(json.dumps({k:v for k,v in coverage.items() if k not in ('files','errors')}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(archive, manifest['archives']))
    coverage['status'] = 'complete' if len(coverage['files']) == len(requested) else 'partial'
    save()


if __name__ == '__main__':
    main()
