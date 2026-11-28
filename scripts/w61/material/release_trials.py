"""Upload only naturally completed new A/C simulator trials; verify GitHub size and SHA-256."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

A=Path('/root/autodl-tmp/MCP-RL-w61-material')
C=Path('/root/autodl-tmp/MCP-RL-w61-mpc')
sys.path.insert(0,str(C/'scripts'))
from w61.mpc.video import render
from w61_gait_rollout import DEFAULT_MODEL
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61')
PREFIX='repos/kaiwen123-yang/MCP-RL/releases'
RID=461339357
TAG='tm-s1-go2-local-readiness-r12-w61'
STUDIES={'w61-a1-forefoot-controls-v1':(A,'material',24),
         'w61-a-m1-full-chain-v1':(A,'material',6),
         'w61-c-transfer-wall-v1':(C,'mpc',8),
         'w61-c-material-history-v1':(C,'mpc',8),
         'w61-c-mppi-candidates-v2':(C,'mpc',6),
         'w61-c-mppi-wall-v2':(C,'mpc',4)}


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,data):
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def api(args,env=None):
    return subprocess.check_output(['gh','api',*args],text=True,env=env,stderr=subprocess.PIPE)


def inventory():
    pages=json.loads(api(['--paginate','--slurp',f'{PREFIX}/{RID}/assets?per_page=100']))
    return {x['name']:x for page in pages for x in page}


def push(path):
    checksum=digest(path);size=path.stat().st_size
    row=remote.get(path.name)
    args=['-X','POST',f'https://uploads.github.com/repos/kaiwen123-yang/MCP-RL/releases/{RID}/assets?name={path.name}',
        '-H','Content-Type: '+('video/mp4' if path.suffix=='.mp4' else 'application/zip'),'--input',str(path)]
    if row is None:
        for attempt in range(4):
            env=dict(os.environ)
            if attempt==3:
                for key in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):env[key]='http://127.0.0.1:17897'
            try:
                row=json.loads(api(args,env));break
            except subprocess.CalledProcessError as exc:
                print('upload',attempt+1,path.name,exc.stderr.strip()[:170],flush=True)
                try:row=inventory().get(path.name)
                except subprocess.CalledProcessError:row=None
                if row is not None:break
                if attempt==3:raise
    if row['digest']!='sha256:'+checksum or row['size']!=size:
        raise ValueError('Remote content differs at '+path.name)
    remote[path.name]=row
    return dict(name=path.name,id=row['id'],size=size,sha256=checksum,url=row['browser_download_url'])


def trial_assets(study,case,source,staging):
    staging.mkdir(parents=True,exist_ok=True)
    stem=study+'-'+case
    mp4=staging/(stem+'.mp4')
    clock=source/'w61-release-video-clock.json'
    if not clock.is_file() or not mp4.is_file():
        write(clock,render(source,mp4,label=case))
    bundle=staging/(stem+'-originals.zip')
    if not bundle.exists():
        pending=bundle.with_suffix('.pending')
        with zipfile.ZipFile(pending,'w',allowZip64=True) as stream:
            for f in sorted(source.rglob('*')):
                if f.is_file():
                    stream.write(f,f.relative_to(source),compress_type=zipfile.ZIP_STORED if f.suffix=='.npz' else zipfile.ZIP_DEFLATED)
        pending.replace(bundle)
    return [mp4,bundle]


def stage_list(base,lane,study,staging,rows):
    use=[dict(path=str(staging/r['name']),size=r['size'],sha256=r['sha256']) for r in rows]
    write(base/'work/w61'/lane/('h-retirable-'+study+'.json'),use)


def weights():
    study='w61-a2-physical-response-v1'
    raw=A/'work/w61/material'/study
    dest=A/REL/'material'/study
    receipt=dest/'release.json'
    if receipt.is_file():return
    if len(list(dest.glob('fit-*.json')))!=3:return
    staging=raw/'release';staging.mkdir(exist_ok=True)
    out=staging/(study+'-checkpoints.zip')
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED) as stream:
        for f in sorted(raw.glob('physical-response-*.pt')):stream.write(f,f.name)
    row=push(out);write(receipt,dict(tag=TAG,model_and_optimizer_rng=[row]))
    stage_list(A,'material',study,staging,[row])
    print('verified',out.name,row['size'],flush=True)


def main():
    global remote
    remote=inventory()
    weights()
    while True:
        ended=0
        for study,(base,lane,target) in STUDIES.items():
            compact=base/REL/lane/study
            if not compact.exists():continue
            raw=base/'work/w61'/lane/study
            staging=raw/'release'
            recfile=compact/'release.json'
            saved=json.loads(recfile.read_text())['trials'] if recfile.exists() else {}
            for f in sorted(compact.glob('*.json')):
                if f.stem in saved or f.name in ('release.json','summary.json'):continue
                val=json.loads(f.read_text());result=val.get('result')
                if result is None:continue
                source=Path(val.get('original_directory',val.get('raw_directory','')))
                if not (source/'trajectory.npz').is_file():continue
                assets=trial_assets(study,f.stem,source,staging)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    rows=list(pool.map(push,assets))
                saved[f.stem]=dict(assets=rows,clock=json.loads((source/'w61-release-video-clock.json').read_text()))
                write(recfile,dict(tag=TAG,registered=target,completed_verified=len(saved),trials=saved))
                # Only stageable delivery paths enter this list, and only after both remote matches.
                stage_list(base,lane,study,staging,[r for x in saved.values() for r in x['assets']])
                print('verified',study,f.stem,flush=True)
            ended+=len(saved)==target
        if ended==len(STUDIES):return
        time.sleep(10)


if __name__=='__main__':main()
