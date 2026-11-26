"""Publish W59 artifacts on D57 and check the server's size, state and SHA-256."""
import argparse
import json
import os
from pathlib import Path
import tempfile

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


def publish(label, paths, video=None):
    # The canonical data stay on D57. Stage one upload part at a time in RAM.
    # This avoids a second persistent copy of every physical trajectory.
    info, assets = release.release_assets()
    groups, count = release.groups_for(paths, video, release.PART)
    target = STATE/'delivery'/(label+'.json')
    old = json.loads(target.read_text()) if target.exists() else {}
    done = {v['name']:v for v in old.get('assets',[])}
    receipts = []
    def send(name, group=None):
        remote = assets.get(name,{})
        past = done.get(name)
        signature = release.source_signature(group) if group else None
        if past and past.get('source_signature')==signature and remote.get('digest')==past['digest'] and remote.get('size')==past['size'] and remote.get('state')=='uploaded':
            return past
        with tempfile.TemporaryDirectory(prefix='w59-part-',dir='/dev/shm') as tmp:
            if group is None:
                file = Path(tmp)/name
                file.symlink_to(video.resolve())
            else:
                file = release.prepare_part(Path(tmp),name,group)
            return dict(release.transfer(file,name,info,assets), source_signature=signature)
    def receipt(state):
        write(target,dict(label=label,release=info['url'],status=state,original_file_count=count,
                          roots={k:str(v) for k,v in paths},assets=receipts,originals_retained=True))
    for i,group in enumerate(groups,1):
        receipts.append(send(f'{label}-originals-{i:03d}.zip',group))
        receipt('partial')
    if video is not None:
        receipts.append(send(label+'-full.mp4'))
    receipt('verified')


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
        import render_s1_go2_r12 as renderer
        renderer.SMALL, renderer.LARGE = STATE/'episodes', RAW
        if not (raw/'video-asset.json').is_file():
            renderer.render(name)
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
