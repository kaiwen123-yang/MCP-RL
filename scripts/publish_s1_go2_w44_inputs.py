#!/usr/bin/env python3
"""Publish W44 reconstructed problems, new comparisons and r33 artifact only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = 'terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w44-mechanisms'
REPO = 'kaiwen123-yang/MCP-RL'
PART_MEMBER_BYTES = 5 * 1024**3 // 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    out = ROOT / 'outputs' / RUN
    compact = ROOT / 'results' / RUN
    delivery_path = compact / 'delivery.json'
    delivery = dict(status='preparing', started_unix_s=time.time(), deadline_unix_s=None,
        source_commit=args.source_commit, release_tag=args.tag, originals_retained=True,
        automatic_cleanup=False, part_member_limit_bytes=PART_MEMBER_BYTES,
        compression='ZIP_DEFLATED level1', parts=[],
        scope='Only W44 reconstructed fixtures and new common-context R32/S/N/P outputs, retained invalid-context-batch1, compact evidence, and r33 patch/spec/binary/build identity. Invalid-context outputs remain labelled excluded from mechanism comparison. Original W43 files are referenced by provenance/hash, not downloaded or reuploaded.',
        verification_scope='Each remote part uploaded state, name, size and SHA256 digest; no remote archive download or full semantic reread.')

    def save():
        delivery_path.write_text(json.dumps(delivery, indent=2) + '\n')

    save()
    try:
        files = [p for p in out.rglob('*') if p.is_file() and not p.is_symlink()
                 and p.suffix != '.zip']
        files += [ROOT / 'work/r11-frozen/S5-r33' / name
                  for name in ('agent_server', 'identity.json', 'build.log')]
        files += [ROOT / 'experiments/terrain_memory' / ('s1-go2-r11-S5-r33' + suffix)
                  for suffix in ('.json', '.patch')]
        files += [p for p in compact.rglob('*') if p.is_file() and p != delivery_path]
        members = []
        for path in sorted(set(files)):
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            members.append(dict(path=str(path.relative_to(ROOT)), size=path.stat().st_size, sha256=digest))
        groups = []
        group = []
        member_bytes = 0
        for item in members:
            if item['size'] > PART_MEMBER_BYTES:
                raise ValueError('W44 source exceeds the declared whole-member part limit: ' + item['path'])
            if group and member_bytes + item['size'] > PART_MEMBER_BYTES:
                groups.append(group)
                group = []
                member_bytes = 0
            group.append(item)
            member_bytes += item['size']
        if group:
            groups.append(group)
        for number, group in enumerate(groups, 1):
            archive = out / f'{args.tag}-part{number:02d}.zip'
            manifest = dict(source_commit=args.source_commit, part=number, total_parts=len(groups),
                member_bytes=sum(item['size'] for item in group), members=group)
            # Resuming reuses each already prepared immutable part. Changed
            # evidence needs a new tag; no local or remote archive is replaced.
            if not archive.exists():
                with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED,
                                     compresslevel=1, allowZip64=True) as bundle:
                    bundle.writestr('w44-source-manifest.json', json.dumps(manifest, indent=2) + '\n')
                    for item in group:
                        bundle.write(ROOT / item['path'], item['path'])
            else:
                with zipfile.ZipFile(archive) as bundle:
                    recorded = json.loads(bundle.read('w44-source-manifest.json'))
                if recorded != manifest:
                    raise ValueError('Prepared W44 part belongs to different evidence; preserve it and use a new tag: ' + archive.name)
            with archive.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            delivery['parts'].append(dict(part=number, archive=str(archive.relative_to(ROOT)),
                name=archive.name, size=archive.stat().st_size, sha256=digest,
                member_bytes=manifest['member_bytes'], source_count=len(group), members=group,
                status='prepared'))
            save()
        delivery.update(status='prepared', source_count=len(members), asset_count=len(groups),
            source_bytes=sum(item['size'] for item in members),
            total_size_bytes=sum(part['size'] for part in delivery['parts']))
        save()
        if subprocess.run(['gh', 'release', 'view', args.tag, '--repo', REPO],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120).returncode:
            subprocess.run(['gh', 'release', 'create', args.tag, '--repo', REPO,
                '--target', args.source_commit, '--title', 'W44 r33 motion optimization evidence and artifact',
                '--notes', 'Four W43 saved problems reconstructed into shared comparison contexts; new R32/S/N/P native results, explicitly excluded invalid-context first batch, and frozen r33 artifact. Local direction evidence does not establish closed-loop material traversal. Original W43 assets are retained without duplicate upload. Each compressed part has its complete member manifest.'],
                check=True, timeout=120)
        def release():
            return json.loads(subprocess.check_output(['gh', 'api',
                f'repos/{REPO}/releases/tags/{args.tag}'], text=True, timeout=120))
        remote = release()
        assets = {row['name']: row for row in remote['assets']}
        delivery['release_url'] = remote['html_url']
        for part in delivery['parts']:
            asset = assets.get(part['name'])
            if asset is None:
                subprocess.run(['gh', 'release', 'upload', args.tag,
                    str(ROOT / part['archive']), '--repo', REPO], check=True, timeout=600)
                remote = release()
                assets = {row['name']: row for row in remote['assets']}
                asset = assets[part['name']]
            if (asset['state'] != 'uploaded' or asset['size'] != part['size']
                    or asset.get('digest') != 'sha256:' + part['sha256']):
                raise RuntimeError('Remote W44 part state/size/digest differ; existing asset retained: ' + part['name'])
            part.update(status='published', verified_unix_s=time.time(), asset_id=asset['id'],
                url=asset['browser_download_url'], remote_size=asset['size'], remote_digest=asset['digest'])
            save()
        delivery.update(status='published', verified_unix_s=time.time())
    except Exception as error:
        delivery.update(status='publication_pending', error=repr(error))
        save()
        raise
    save()
    print(json.dumps(dict(status=delivery['status'], source_count=delivery['source_count'],
        asset_count=delivery['asset_count'], total_size_bytes=delivery['total_size_bytes'],
        release_url=delivery['release_url'], parts=[{key:part[key] for key in (
            'name','size','sha256','url','remote_digest')} for part in delivery['parts']]), indent=2))


if __name__ == '__main__':
    main()
