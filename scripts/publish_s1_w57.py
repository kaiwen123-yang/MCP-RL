"""Direct D57 publication of W57 originals, videos and verified GitHub Release receipts."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

from publish_s1_go2_w54 import write

ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-local-readiness-r12'
REPO='kaiwen123-yang/MCP-RL'
TAG=RUN+'-w57'
SMALL=ROOT/'results/terrain_memory/simulation'/RUN/'w57'
LARGE=ROOT/'outputs/terrain_memory/simulation'/RUN
SPOOL=Path('/dev/shm')
PART=512*1024**2


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def network_gh(route, *args):
    command = ['bash', '-c', 'source scripts/autodl_network.sh "$1" >/dev/null && shift && exec gh "$@"',
               'w57-release', route, *map(str, args)]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def transient_transfer(message):
    message = message.lower()
    return message.rstrip().endswith('eof') or any(reason in message for reason in (
        'tls handshake timeout', 'connection reset', 'unexpected eof', 'http 502',
        'http 503', 'http 504', 'i/o timeout', 'connection timed out',
        'error connecting to api.github.com', 'error connecting to uploads.github.com'))


def gh(*args):
    # The PC tunnel is only for authenticated small GETs after D57 route failures.
    for route in ('direct', 'academic', 'direct', 'pc'):
        result = network_gh(route, *args)
        if result.returncode == 0:
            return result.stdout.strip()
        error = result.stderr.strip()
        if not transient_transfer(error):
            raise RuntimeError(error)
        print(f'GitHub metadata GET on {route} failed: {error}', flush=True)
    raise RuntimeError(error)


def release_assets(name=None):
    release = read(SMALL/'release.json')
    endpoint = f'repos/{REPO}/releases/{release["databaseId"]}/assets'
    assets = {}
    page = 1
    while True:
        # Keep successful pages when only a later GET loses its connection.
        batch = json.loads(gh('api', '-H', 'Cache-Control: no-cache',
                              f'{endpoint}?per_page=100&page={page}'))
        assets.update((row['name'], row) for row in batch)
        if (name and name in assets) or len(batch) < 100:
            return release, assets
        page += 1


def asset_by_id(asset_id):
    endpoint = f'repos/{REPO}/releases/assets/{asset_id}'
    return json.loads(gh('api', '-H', 'Cache-Control: no-cache', endpoint))


def preserve_starter(row, name, size, checksum):
    """Release this name only for the unfinished upload of the exact local part."""
    if row is None or row['state'] != 'starter' or row.get('digest'):
        return row
    asset_id = row['id']
    part = Path(name)
    retained = f'{part.stem}-incomplete-upload-{asset_id}{part.suffix}'
    endpoint = f'repos/{REPO}/releases/assets/{asset_id}'
    # Every W57 sender holds publish.lock and waits synchronously for its upload client.
    # Give a fresh server-side state one normal publisher poll to settle, then reread the ID.
    current = asset_by_id(asset_id)
    while current['name'] == name and current['state'] == 'starter' and not current.get('digest'):
        updated = datetime.fromisoformat(current['updated_at'].replace('Z', '+00:00')).timestamp()
        wait = 30 - (time.time()-updated)
        if wait <= 0:
            break
        time.sleep(min(30, wait + 0.1))
        current = asset_by_id(asset_id)
    if current['name'] == retained:
        return release_assets(name)[1].get(name)
    if current['name'] != name:
        raise ValueError(f'Unexpected name for GitHub asset {asset_id}: {current["name"]}')
    if current['state'] != 'starter' or current.get('digest'):
        return current
    if current['size'] not in (0, size) or current['uploader']['login'] != REPO.split('/')[0]:
        raise ValueError(f'Unfinished asset differs from the declared W57 upload: {name} ({asset_id})')
    note = SMALL/'dispatch'/'release-interruptions'/(name+'.json')
    receipt = read(note) if note.exists() else dict(name=name, expected_size=size, expected_digest=checksum,
                                                    retained={}, deadline=None, wall_deadline_enabled=False,
                                                    kill_on_conversation_end=False)
    if (receipt['expected_size'], receipt['expected_digest']) != (size, checksum):
        raise ValueError(f'Original bytes changed during W57 publication: {name}')
    receipt['retained'][str(asset_id)] = dict(before={k:current.get(k) for k in
        ('name','state','size','digest','created_at','updated_at')}, retained_name=retained,
        observed_unix_s=time.time(), confirmed=False)
    write(note, receipt)
    for route in ('direct', 'academic'):
        edit = network_gh(route, 'api', '--method', 'PATCH', endpoint, '-f', f'name={retained}')
        final = asset_by_id(asset_id)
        if final['name'] == retained:
            receipt['retained'][str(asset_id)].update(confirmed=True, confirmed_unix_s=time.time(),
                                                      state=final['state'], digest=final.get('digest'))
            write(note, receipt)
            print(f'Preserved interrupted asset {asset_id} as {retained}', flush=True)
            return None
        if final['name'] == name and (final['state'] != 'starter' or final.get('digest')):
            return final
        if final['name'] != name or route == 'academic' or not transient_transfer(edit.stderr):
            raise RuntimeError(f'Could not preserve incomplete GitHub asset {asset_id}: {edit.stderr.strip()}')


def transfer(path, name, release, existing):
    size = path.stat().st_size
    checksum = 'sha256:'+sha(path)
    original = existing.get(name)
    row = preserve_starter(original, name, size, checksum)
    if row is None:
        # An earlier native upload can leave a digestless reservation. Try the other D57 route first.
        routes = ('academic','direct') if original and original['state']=='starter' else ('direct','academic')
        for route in routes:
            proc = network_gh(route, 'release', 'upload', TAG, str(path), '-R', REPO)
            _, assets = release_assets(name)
            fresh = assets.get(name)
            row = preserve_starter(fresh, name, size, checksum)
            if row is not None:
                break
            if fresh is not None and fresh['state'] == 'starter':
                if route == routes[-1]:
                    raise RuntimeError(f'Interrupted GitHub upload pending a new D57 connection: {name}')
                continue
            error = proc.stderr.strip()
            if proc.returncode == 0 or not transient_transfer(error) or route == routes[-1]:
                raise RuntimeError(f'GitHub has no asset {name} after upload on {route}: {error}')
            print(f'{name} absent after interrupted upload on {route}: {error}', flush=True)
        existing[name] = row
    if row['size']!=size or row.get('digest')!=checksum or row['state']!='uploaded':
        raise ValueError(f'Size/digest/state do not match the local original: {name}')
    verified = dict(name=name,size=size,digest=checksum,url=row['browser_download_url'],
                    asset_id=row['id'],release=release['url'],verified_unix_s=time.time())
    note = SMALL/'dispatch'/'release-interruptions'/(name+'.json')
    if note.exists():
        receipt = read(note)
        receipt['replacement'] = verified
        write(note, receipt)
    return verified


def originals(label, paths, video=None):
    dest=SMALL/'delivery'/(label+'.json')
    release,existing=release_assets()
    prior=read(dest) if dest.exists() else None
    retention = (prior or {}).get('local_retention', {})
    if retention.get('removed_mpc_files') or retention.get('H_process_20261124'):
        # This episode's published ZIPs precede the verified local inode cleanup.
        # Reuse those exact assets when finalizing or explicitly republishing it.
        for receipt in prior['assets']:
            actual=existing.get(receipt['name'])
            if actual is None or (actual['size'],actual.get('digest'),actual['state'])!=(
                    receipt['size'],receipt['digest'],'uploaded'):
                raise ValueError('The exact archived asset must be recovered before republishing '+receipt['name'])
        return
    canonical=[]
    for prefix,path in paths:
        if path.is_dir():
            canonical += [(prefix+'/'+str(f.relative_to(path)),f) for f in sorted(path.rglob('*'))
                          if f.is_file() and f!=video]
        elif path.is_file():
            canonical.append((prefix,path))
        else:
            raise FileNotFoundError(path)
    groups=[];part=[];size=0
    for name,f in canonical:
        n=f.stat().st_size
        if size+n>PART and part:
            groups.append(part);part=[];size=0
        part.append((name,f));size+=n
    if part:groups.append(part)
    recorded=prior if prior is not None else dict(label=label,tag=TAG,assets=[],
        roots={k:str(p) for k,p in paths},original_file_count=len(canonical),originals_retained=True,
        compact_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    done={a['name']:a for a in recorded['assets']}
    def already_verified(name):
        if name not in done or name not in existing:return False
        old=done[name];cur=existing[name]
        return old['size']==cur['size'] and old['digest']==cur.get('digest') and cur['state']=='uploaded'
    def save(receipt):
        done[receipt['name']]=receipt
        recorded['assets']=list(done.values())
        write(dest,recorded)
        print(receipt['name'],receipt['size'],receipt['digest'],flush=True)
    with tempfile.TemporaryDirectory(prefix='w57-upload-',dir=SPOOL) as tmp:
        for number,group in enumerate(groups,1):
            name=f'{label}-originals-{number:03d}.zip'
            if already_verified(name):continue
            temporary=Path(tmp)/name
            with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
                for member,f in group:
                    z.write(f,member)
            save(transfer(temporary,name,release,existing))
            temporary.unlink()  # Temporary upload spool; every source stays at the recorded D57 path.
        if video:
            name=label+'-full.mp4'
            if not already_verified(name):
                link=Path(tmp)/name
                link.symlink_to(video.resolve())
                save(transfer(link,name,release,existing))
    recorded.update(status='verified',release=release['url'],verified_unix_s=time.time())
    write(dest,recorded)


def deliver_episode(attempt):
    import render_s1_go2_r12 as renderer
    out=LARGE/attempt
    if not (out/'trajectory.npz').exists():
        raise FileNotFoundError(f'No recorded trajectory for {attempt}')
    renderer.SMALL=SMALL/'episodes'
    with open(ROOT/'work/w57/render.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        alternate=ROOT/'work/w57/rendered'/attempt
        # On D57 the personal mount can exhaust its inode quota with bytes still free.
        # Keep the original trajectory read-only and place new derived files on scratch.
        dest=out
        if not (out/'video-asset.json').exists() and (
                (alternate/'video-asset.json').exists() or os.statvfs(out).f_favail<2):
            dest=alternate
        if not (dest/'video-asset.json').exists():
            renderer.render(attempt,output_dir=dest)
    video=dest/'full-mission-material-overlay.mp4'
    paths=[('episode',out),('job',LARGE/'jobs'/attempt)]
    if dest!=out:
        paths.append(('episode/video-asset.json',dest/'video-asset.json'))
    actor=LARGE/'evaluation-policy'/attempt
    if actor.exists():paths.append(('frozen-speed-policy',actor))
    originals(attempt,paths,video)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('episode').add_argument('attempt')
    b=sub.add_parser('bundle')
    b.add_argument('name');b.add_argument('paths',nargs='+',type=Path)
    sub.add_parser('ready-episodes')
    sub.add_parser('follow-episodes')
    args=parser.parse_args()
    lockpath=ROOT/'work/w57/publish.lock'
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    if args.action in ('episode','bundle'):
        with lockpath.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if args.action=='episode':
                deliver_episode(args.attempt)
            else:
                originals(args.name,[(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
                                     path.resolve()) for path in args.paths])
        return
    while True:
        for source in sorted((SMALL/'attempts').glob('*.json')):
            state=read(source)
            if 'ended_unix_s' not in state or not (SMALL/'episodes'/(source.stem+'.json')).exists():
                continue
            dest=SMALL/'delivery'/source.name
            # Let a finished PPO node publish between complete episodes, even with a backlog.
            with lockpath.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not dest.exists() or read(dest).get('status')!='verified':
                    deliver_episode(source.stem)
        queue=SMALL/'dispatch/frozen-evaluations.json'
        if args.action=='ready-episodes' or queue.exists() and read(queue)['stage']=='queue_ended':
            break
        time.sleep(30)


if __name__=='__main__':main()
