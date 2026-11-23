"""Deliver completed W58 records from D57, render complete trajectories, and verify GitHub digests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import publish_s1_w57 as release
from publish_s1_go2_w54 import write
from inspect_s1_w58_predecessor import REL, read, BG

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work/w58'
R=ROOT/'results'/REL/'w58'
RAW=ROOT/'outputs'/REL
TAG=REL.name+'-w58'
release.SMALL=R
release.LARGE=RAW
release.TAG=TAG
release.ROOT=ROOT


def identity():
    info=R/'release.json'
    if info.exists():return read(info)
    args=['gh','api',f'repos/{release.REPO}/releases/tags/{TAG}']
    result=subprocess.run(args,capture_output=True,text=True,cwd=ROOT)
    if result.returncode:
        if 'HTTP 404' not in result.stderr:raise RuntimeError(result.stderr)
        source=read(WORK/'runtime.json')['commit']
        notes=WORK/'release-notes.md'
        notes.write_text('Go2/TM02 W58 的完整触地记录、材料离线预测、独立资格、机制路线与冻结策略原件。每个已上传文件具有 D57 大小与 GitHub SHA-256 回执。当前状态见 main 上的 W58 运行索引。\n')
        release.gh('release','create',TAG,'-R',release.REPO,'--target',source,'--title','S1 W58 · Go2 material memory and contact control','--notes-file',notes)
    row=json.loads(release.gh('api',f'repos/{release.REPO}/releases/tags/{TAG}'))
    data=dict(databaseId=row['id'],url=row['html_url'],tagName=row['tag_name'],targetCommitish=row['target_commitish'])
    write(info,data)
    return data


def verified(name):
    p=R/'delivery'/(name+'.json')
    return p.exists() and read(p).get('status')=='verified'


def episode(name):
    out=RAW/name
    small=R/'episodes'
    paths=[('job',RAW/'jobs'/name)]
    if out.is_dir():paths.append(('episode',out))
    for suffix in ('.json','-resolved.json'):
        p=small/(name+suffix)
        if p.is_file():paths.append(('receipt/'+p.name,p))
    actor=RAW/'evaluation-policy'/name
    if actor.is_dir():paths.append(('frozen-speed-policy',actor))
    video=None
    if (out/'trajectory.npz').is_file() and (small/(name+'.json')).is_file():
        import render_s1_go2_r12 as renderer
        renderer.SMALL=small
        renderer.LARGE=RAW
        if not (out/'video-asset.json').is_file():renderer.render(name)
        video=out/'full-mission-material-overlay.mp4'
    release.originals(name,paths,video)


def publish_ready():
    identity()
    label='w58-g-p2-complete-development-replays'
    if not verified(label):
        files=[*sorted((R/'g').rglob('*-landings.jsonl.gz')),R/'p2/saved-requests.jsonl.gz',
               R/'p2/saved-support-calibration.json',R/'p2/common-force-candidates.json']
        release.originals(label,[(str(p.relative_to(R)),p) for p in files])
    # Submit the complete release-state extraction of each completed history as soon as ready.
    m2=ROOT.parent/'MCP-RL-w58-m2'
    for meta in sorted((R/'m2').glob('*-replay.json')):
        name=meta.name.removesuffix('-replay.json')
        label='w58-material-'+name
        paths=[m2/'work/w58-m2'/f'{name}-{k}-starts.json.gz' for k in ('M0','M1','M2')]
        if all(p.is_file() for p in paths) and not verified(label):
            release.originals(label,[(p.name,p) for p in paths]+[(meta.name,meta)])
    for final in sorted((R/'learning').glob('seed-*/*/final.json')):
        src=WORK/'campaigns'/final.relative_to(R/'learning').parent
        name='w58-joint-'+src.parent.name+'-'+src.name
        if not verified(name):release.originals(name,[('learning',src),('seed-initialization.pt',src.parent/'common-initialization.pt')])
    for p in sorted((R/'attempts').glob('*.json')):
        state=read(p)
        if 'ended_unix_s' in state and not verified(p.stem):episode(p.stem)
    completed=R/'outcome.json'
    if completed.exists() and not verified('w58-final-execution'):
        files=[(name,WORK/name) for name in ('runtime.json','throughput.jsonl') if (WORK/name).is_file()]
        files.append(('pool-jobs',WORK/'pool/jobs'))
        release.originals('w58-final-execution',files)


def main():
    import fcntl
    lock=(WORK/'delivery.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            publish_ready()
            receipts=[read(p) for p in (R/'delivery').glob('*.json')]
            assets={a['name']:a for r in receipts for a in r['assets']}
            data=dict(verified_bundles=sum(r.get('status')=='verified' for r in receipts),
                      verified_assets=len(assets),verified_bytes=sum(a['size'] for a in assets.values()),
                      release=read(R/'release.json')['url'],**BG)
            p=R/'publication.json'
            if not p.exists() or any(read(p).get(k)!=v for k,v in data.items()):
                write(p,dict(data,last_verified_unix_s=time.time()))
            if (R/'outcome.json').exists():
                _,remote=release.release_assets()
                for name,a in assets.items():
                    if (remote[name]['state'],remote[name]['size'],remote[name].get('digest'))!=('uploaded',a['size'],a['digest']):
                        raise ValueError('Final Release metadata mismatch: '+name)
                write(R/'delivery-ended.json',dict(data,final_verification_unix_s=time.time(),all_remote_digests_matched=True))
                return
        except (RuntimeError,subprocess.CalledProcessError) as error:
            print('D57 delivery pending',repr(error),flush=True)
        time.sleep(45)


if __name__=='__main__':main()

