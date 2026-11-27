"""Publish W59 artifacts on D57 and check the server's size, state and SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import subprocess
import sys
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import publish_s1_w57 as release
from publish_s1_go2_w54 import write

ROOT = Path(__file__).resolve().parents[1]
REL = Path('terrain_memory/simulation')/release.RUN
STATE = ROOT/'results'/REL/'w59'
RAW = ROOT/'outputs'/REL
WORK = ROOT/'work/w59'
TAG = release.RUN+'-w59'
release.SMALL, release.LARGE, release.TAG, release.ROOT = STATE, RAW, TAG, ROOT


def github(*args):
    for route in ('academic','direct','clash'):
        p = release.network_gh(route,*args)
        if p.returncode == 0:
            return p.stdout.strip()
        if not release.transient_transfer(p.stderr):
            break
    raise RuntimeError(p.stderr.strip())


release.gh = github


def ensure_release():
    path = STATE/'release.json'
    if path.is_file():
        return json.loads(path.read_text())
    tag_endpoint = f'repos/{release.REPO}/releases/tags/{TAG}'
    query = release.network_gh('academic','api',tag_endpoint)
    if query.returncode and ('404' in query.stderr or 'Not Found' in query.stderr):
        target = github('api',f'repos/{release.REPO}/commits/main','--jq','.sha')
        notes = WORK/'release-notes.md'
        notes.write_text('Go2/TM02 W59 的真实物理记录、完整视频与模型 checkpoint。各项结果、技术中断与 SHA-256 回执见 main 的 w59 运行索引。\n')
        github('release','create',TAG,'-R',release.REPO,'--target',target,'--title','S1 W59 · Go2 material memory and action-conditioned control','--notes-file',str(notes))
    row = json.loads(github('api',tag_endpoint))
    info = dict(databaseId=row['id'], url=row['html_url'], tagName=row['tag_name'], targetCommitish=row['target_commitish'])
    write(path,info)
    return info


def bundle_assets(name=None):
    info = ensure_release()
    endpoint = f'repos/{release.REPO}/releases/{info["databaseId"]}'
    header = ('-H', 'Cache-Control: no-cache')
    try:
        rows = json.loads(github('api',*header,endpoint,'--jq','.assets'))
        assets = {r['name']:r for r in rows}
        if name is not None and name in assets:
            return info, assets
        # The embedded list may be capped. Verify that there is no additional
        # REST asset before using this snapshot to conclude that a name is absent.
        beyond = f'{endpoint}/assets?per_page=1&page={len(assets)+1}'
        if github('api',*header,beyond,'--jq','length') == '0':
            return info, assets
    except RuntimeError as error:
        print('Release detail unavailable; using the paginated asset list:',error,flush=True)
    return release.release_assets(name)


def publish(label, paths, video=None):
    # ZIP bytes, asset order and remote verification stay identical to the serial publisher.
    info, assets = bundle_assets()
    groups, count = release.groups_for(paths, video, release.PART)
    target = STATE/'delivery'/(label+'.json')
    old = json.loads(target.read_text()) if target.exists() else {}
    done = {v['name']:v for v in old.get('assets',[])}
    receipts = []
    parts = [(f'{label}-originals-{i:03d}.zip', group) for i,group in enumerate(groups,1)]
    if video is not None:
        parts.append((label+'-full.mp4', None))

    def receipt(state):
        write(target,dict(label=label,release=info['url'],status=state,original_file_count=count,
                          roots={k:str(v) for k,v in paths},assets=receipts,originals_retained=True))

    def prepare(directory, part):
        name, group = part
        previous = done.get(name)
        signature = release.source_signature(group) if group else None
        remote = assets.get(name,{})
        if (previous and previous.get('source_signature')==signature
                and remote.get('digest')==previous['digest'] and remote.get('size')==previous['size']
                and remote.get('state')=='uploaded'):
            return previous, None, signature, None
        if group is None:
            path = directory/name
            path.symlink_to(video.resolve())
            return None, path, signature, None
        start = time.monotonic()
        path = release.prepare_part(directory,name,group)
        return None, path, signature, (start,time.monotonic())

    def room_for_next(directory, group):
        if group is None:
            return True
        # Current ZIP is already charged to tmpfs. Keep serial staging when there is
        # not room for a second ZIP, including ZIP headers and incompressible data.
        size = sum(p.stat().st_size + 2*len(member.encode()) + 256 for member,p in group)
        budget = size + size//100 + 1024**2
        sample = WORK/'last-resource.json'
        if sample.is_file():
            room = json.loads(sample.read_text()).get('usable_memory_bytes')
            if room is not None and room < budget + 3*1024**3:
                return False
        return shutil.disk_usage(directory).free >= budget

    with tempfile.TemporaryDirectory(prefix='w59-parts-',dir='/dev/shm') as tmp:
        directory = Path(tmp)
        with ThreadPoolExecutor(max_workers=1) as packer:
            pending = None
            previous_transfer = None
            for i,part in enumerate(parts):
                name, group = part
                cached, path, signature, packing = (pending.result() if pending is not None
                                                    else prepare(directory,part))
                pending = None
                if i+1 < len(parts) and room_for_next(directory,parts[i+1][1]):
                    pending = packer.submit(prepare,directory,parts[i+1])
                if cached is None:
                    start = time.monotonic()
                    result = release.transfer(path,name,info,assets,lookup_assets=bundle_assets)
                    end = time.monotonic()
                    result['source_signature'] = signature
                    if packing is not None:
                        overlap = 0.
                        if previous_transfer is not None:
                            overlap = max(0., min(packing[1],previous_transfer[1])
                                          - max(packing[0],previous_transfer[0]))
                        result.update(prepare_wall_s=packing[1]-packing[0],
                                      preparation_overlapped_transfer_s=overlap,
                                      transfer_verify_wall_s=end-start)
                    previous_transfer = (start,end)
                    path.unlink()
                else:
                    result = cached
                    previous_transfer = None
                receipts.append(result)
                if group is not None:
                    receipt('partial')
    receipt('verified')


def render_video(name, dest):
    # The publisher and ahead-of-upload renderer can both reach the same closed trial.
    # Keep one writer of its complete movie. The recorder is never invoked.
    locks=WORK/'video-locks'
    locks.mkdir(parents=True,exist_ok=True)
    with (locks/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (dest/'video-asset.json').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/render_s1_go2_r12.py'),
                '--attempt',name,'--output-dir',str(dest),'--w59-nvenc'],cwd=ROOT,check=True)


def episode(name):
    receipt = json.loads((STATE/'attempts'/(name+'.json')).read_text())
    if not receipt.get('ended_unix_s'):
        raise ValueError('Episode still owns its recorder')
    raw = RAW/name
    result = STATE/'episodes'/(name+'.json')
    paths = [('job', RAW/'jobs'/name)]
    if raw.is_dir():
        paths.append(('episode',raw))
    for p in (result, result.with_name(name+'-resolved.json')):
        if p.is_file():
            paths.append(('receipt/'+p.name,p))
    actor = RAW/'evaluation-policy'/name
    if actor.is_dir():
        paths.append(('frozen-speed-policy',actor))
    video = None
    if (raw/'trajectory.npz').is_file() and result.is_file() and receipt.get('task_mode') != 'branch':
        render_video(name,raw)
        video = raw/'full-mission-material-overlay.mp4'
    publish(name,paths,video)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode')
    p.add_argument('--label')
    p.add_argument('files',nargs='*',type=Path)
    args = p.parse_args()
    os.environ.setdefault('S1_RELEASE_STREAMS','2')
    ensure_release()
    if args.episode:
        episode(args.episode)
    elif args.label and args.files:
        publish(args.label,[(path.name,path) for path in args.files])
