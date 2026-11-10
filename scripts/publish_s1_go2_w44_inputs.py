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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    out = ROOT / 'outputs' / RUN
    compact = ROOT / 'results' / RUN
    archive = out / (args.tag + '.zip')
    delivery_path = compact / 'delivery.json'
    delivery = dict(status='preparing', started_unix_s=time.time(), deadline_unix_s=None,
        source_commit=args.source_commit, release_tag=args.tag, originals_retained=True,
        automatic_cleanup=False,
        scope='Only W44 reconstructed fixtures and new common-context R32/S/N/P outputs, compact evidence, and r33 patch/spec/binary/build identity. Original W43 files are referenced by provenance/hash, not downloaded or reuploaded.',
        verification_scope='Remote uploaded state, name, size and SHA256 digest; no remote archive download or full semantic reread.')

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
        # A resumed publication reads the already prepared archive. Its exact
        # member hashes remain inspectable inside; changing evidence needs a new tag.
        if not archive.exists():
            with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
                bundle.writestr('w44-source-manifest.json', json.dumps(dict(
                    source_commit=args.source_commit, members=members), indent=2) + '\n')
                for item in members:
                    bundle.write(ROOT / item['path'], item['path'])
        else:
            with zipfile.ZipFile(archive) as bundle:
                recorded = json.loads(bundle.read('w44-source-manifest.json'))
            if recorded != dict(source_commit=args.source_commit, members=members):
                raise ValueError('Prepared W44 archive belongs to different evidence; preserve it and use a new tag.')
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        delivery.update(status='prepared', archive=str(archive.relative_to(ROOT)),
            size=archive.stat().st_size, sha256=digest, source_count=len(members), members=members)
        save()
        if subprocess.run(['gh', 'release', 'view', args.tag, '--repo', REPO],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120).returncode:
            subprocess.run(['gh', 'release', 'create', args.tag, '--repo', REPO,
                '--target', args.source_commit, '--title', 'W44 r33 motion optimization evidence and artifact',
                '--notes', 'Four W43 saved problems reconstructed into shared comparison contexts; new R32/S/N/P native results and frozen r33 artifact. Local direction evidence does not establish closed-loop material traversal. Original W43 assets are retained without duplicate upload.'],
                check=True, timeout=120)
        def release():
            return json.loads(subprocess.check_output(['gh', 'api',
                f'repos/{REPO}/releases/tags/{args.tag}'], text=True, timeout=120))
        remote = release()
        asset = next((row for row in remote['assets'] if row['name'] == archive.name), None)
        if asset is None:
            subprocess.run(['gh', 'release', 'upload', args.tag, str(archive), '--repo', REPO],
                           check=True, timeout=600)
        remote = release()
        asset = next(row for row in remote['assets'] if row['name'] == archive.name)
        if asset['state'] != 'uploaded' or asset['size'] != delivery['size'] or asset.get('digest') != 'sha256:' + digest:
            raise RuntimeError('Remote W44 asset state/size/digest differ; existing asset retained.')
        delivery.update(status='published', verified_unix_s=time.time(), asset_id=asset['id'],
            url=asset['browser_download_url'], remote_size=asset['size'], remote_digest=asset['digest'],
            release_url=remote['html_url'])
    except Exception as error:
        delivery.update(status='publication_pending', error=repr(error))
        save()
        raise
    save()
    print(json.dumps({key:delivery[key] for key in (
        'status','archive','size','sha256','source_count','release_url','url','remote_digest')}, indent=2))


if __name__ == '__main__':
    main()
