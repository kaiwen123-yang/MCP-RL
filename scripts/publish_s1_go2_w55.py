#!/usr/bin/env python3
"""Publish immutable W55 originals from D57 and record verified GitHub receipts."""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import publish_s1_go2_r12 as transport
from publish_s1_go2_w54 import write

ROOT = Path(__file__).resolve().parents[1]
RUN = 'tm-s1-go2-local-readiness-r12'
TAG = RUN + '-w55'
SMALL = ROOT / 'results/terrain_memory/simulation' / RUN / 'w55'
STAGING = ROOT / 'outputs/terrain_memory/simulation' / RUN / 'w55-delivery'
CHUNK = 8 * 1024**2
PART_SIZE = 256 * 1024**2


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def manifest_row(path, category, checksum):
    return dict(path=str(path.relative_to(ROOT)), name=path.name, category=category,
                size=path.stat().st_size, sha256=checksum, release_tag=TAG)


def prepare(name, roots, directory):
    staging = STAGING / name
    staging.mkdir(parents=True, exist_ok=True)
    members = {}
    for root in roots:
        root = root.resolve(strict=True)
        for path in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if path.is_file():
                member = str(path.relative_to(root.parent))
                if member in members and members[member] != path:
                    raise ValueError(f'Two distinct source files have member name {member}')
                members[member] = path
    if not members:
        raise ValueError('No generated original files found in the supplied paths')

    catalog, assets, parts, group = [], [], [], []
    size = 0
    for member, path in members.items():
        if path.suffix.lower() == '.mp4':
            # Keep a playable asset without adding a second copy inside the archive.
            target = staging / f'w55-{name}-{path.name}'
            if not target.exists():
                target.symlink_to(path)
            if target.resolve() != path:
                raise ValueError(f'Video name already refers to another original: {target}')
            checksum = digest(path)
            assets.append(manifest_row(target, 'full_recorded_video', checksum))
            catalog.append(dict(source_path=str(path), asset=target.name,
                                size=path.stat().st_size, sha256=checksum))
            continue
        if group and size + path.stat().st_size > PART_SIZE:
            parts.append(group)
            group, size = [], 0
        group.append((member, path))
        size += path.stat().st_size
    if group:
        parts.append(group)
    for number, files in enumerate(parts, 1):
        archive = staging / f'w55-{name}-originals-{number:04d}.zip'
        temp = archive.with_suffix('.partial')
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for member, path in files:
                h = hashlib.sha256()
                with path.open('rb') as source, bundle.open(member, 'w', force_zip64=True) as target:
                    while block := source.read(CHUNK):
                        h.update(block)
                        target.write(block)
                catalog.append(dict(source_path=str(path), asset=archive.name,
                                    member=member, size=path.stat().st_size, sha256=h.hexdigest()))
        os.replace(temp, archive)
        if archive.stat().st_size < 2 * 1024**3:
            assets.append(manifest_row(archive, 'execution_originals_zip', digest(archive)))
        else:
            # GitHub caps individual assets at 2 GiB. Retain the unbroken source ZIP.
            with archive.open('rb') as stream:
                part = 0
                while stream.tell() < archive.stat().st_size:
                    part += 1
                    destination = archive.with_name(f'{archive.name}.part{part:03d}')
                    checksum = hashlib.sha256()
                    with destination.open('wb') as target:
                        for _ in range(PART_SIZE // CHUNK):
                            block = stream.read(CHUNK)
                            if not block:
                                break
                            target.write(block)
                            checksum.update(block)
                    assets.append(manifest_row(destination, 'concatenate_numbered_zip_parts',
                                               checksum.hexdigest()))
    write(directory / 'sources.json', catalog)
    write(directory / 'manifest.json', assets)


def publish(name, roots):
    directory = SMALL / 'delivery' / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'manifest.json'
    if not manifest.exists():
        prepare(name, roots, directory)
    # Reuse the established authenticated D57 uploader and its remote size/digest readback.
    # The W55 release is created once against a pushed W55 commit before uploading.
    transport.RUN = TAG
    transport.SMALL = SMALL
    for action in ('upload', 'finish'):
        sys.argv = [__file__, action, '--manifest', str(manifest), '--server-auth',
                    '--upload-workers', '1', '--no-wall-deadline']
        transport.main()
    indexed = json.loads((SMALL / 'index.json').read_text())['publication']['assets']
    by_name = {r['name']: r for r in indexed}
    verified = []
    for local in json.loads(manifest.read_text()):
        remote = by_name[local['name']]
        if (remote['status'] != 'uploaded' or remote.get('remote_state') != 'uploaded'
                or remote.get('remote_digest') != 'sha256:' + local['sha256']
                or remote['size'] != local['size']):
            raise RuntimeError('Unconfirmed Release asset: ' + local['name'])
        verified.append({k: remote[k] for k in ('name', 'size', 'url', 'remote_digest',
                                                'remote_asset_id', 'remote_verified_unix_s')})
    write(directory / 'delivery.json', dict(status='published', assets=verified,
          original_files=len(json.loads((directory / 'sources.json').read_text())),
          remote_check='authenticated size, uploaded state and SHA-256', deadline=None,
          wall_deadline_enabled=False, checked_unix_s=time.time()))
    print(json.dumps(dict(group=name, verified_assets=len(verified))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', action='append', default=[], type=Path)
    args = parser.parse_args()
    STAGING.mkdir(parents=True, exist_ok=True)
    with (STAGING / 'publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        publish(args.name, args.source)
