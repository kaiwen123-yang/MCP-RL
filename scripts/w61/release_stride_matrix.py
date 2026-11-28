"""Publish one finite B/D stride matrix with checksummed full observations from D57."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import time


def read_asset(repo,tag,name):
    record=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/releases/tags/{tag}']))
    return next((a for a in record['assets'] if a['name']==name),None)


def archive(directory,asset,runs):
    with tarfile.open(asset,'w:gz',compresslevel=1) as stream:
        for run in runs:
            stream.add(directory/run,arcname=run)


def main(args):
    for name in args.runs:
        marker=args.raw/name/'result.json'
        while not marker.is_file():
            time.sleep(12)
    module=importlib.util.spec_from_file_location('stride_evidence',args.renderer)
    metrics=importlib.util.module_from_spec(module);module.loader.exec_module(metrics)
    args.output.mkdir(parents=True,exist_ok=True)
    compact=args.output/'result.json'
    if not compact.exists():
        start=time.monotonic()
        rows=[metrics.summarize((args.raw/name).resolve()) for name in args.runs]
        compact.write_text(json.dumps(dict(lane=args.lane,case_count=len(rows),
            completed=sum(r['completed'] for r in rows),rendering_wall_s=time.monotonic()-start,runs=rows),
            indent=2,allow_nan=False)+'\n')
    if (args.output/'release.json').exists():
        return
    asset=args.raw/args.asset
    remote=read_asset(args.repo,args.tag,asset.name)
    if remote is None:
        if not asset.exists():
            archive(args.raw,asset,args.runs)
        subprocess.run(['gh','release','upload',args.tag,'--repo',args.repo,str(asset)],check=True)
        remote=read_asset(args.repo,args.tag,asset.name)
    with asset.open('rb') as f:
        digest='sha256:'+hashlib.file_digest(f,'sha256').hexdigest()
    assert (remote['size'],remote['digest'],remote['state'])==(asset.stat().st_size,digest,'uploaded')
    receipt={k:remote[k] for k in ('name','id','size','digest','state','browser_download_url')}
    (args.output/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
    subprocess.run(['git','add','--sparse',str(args.output)],check=True)
    subprocess.run(['git','commit','-m',f'Deliver {args.lane} {args.output.name} outcomes and verified originals'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('raw','output','renderer'):
        p.add_argument('--'+k,type=Path,required=True)
    for k in ('tag','repo','asset','lane'):
        p.add_argument('--'+k,required=True)
    p.add_argument('runs',nargs='+');main(p.parse_args())
