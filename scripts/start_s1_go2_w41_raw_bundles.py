#!/usr/bin/env python3
"""Upload the24 prepared W41 ZIPs through the private server upload proxy."""
import argparse,fcntl,json,shlex,shutil,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-local-readiness-r12'
SMALL=ROOT/'results/terrain_memory/simulation'/RUN
DEST=SMALL/'raw-bundles/w41-20261108'
OUT=ROOT/'outputs/terrain_memory/simulation'/RUN/'raw-bundles/w41-20261108'
PROXY='http://127.0.0.1:18911'
TAG=RUN+'-w41-raw-bundles-20261108'
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    session='w41-zip-upload'
    if subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode==0:
        raise RuntimeError('The W41 ZIP transfer is already active; do not duplicate it.')
    previous=json.loads((OUT/'upload-job.json').read_text()) if (OUT/'upload-job.json').exists() else None
    if previous and not args.resume:raise RuntimeError('Use --resume to retain the previous upload records and reuse all24 packages.')
    if args.resume and not previous:raise RuntimeError('No existing ZIP delivery to resume.')
    history=None
    if previous:
        history=OUT/'upload-history'/str(time.time_ns());history.mkdir(parents=True)
        for name in ['upload-job.json','run-upload.sh','upload-a.log','upload-b.log','finalize.log','upload-a-exit-code.txt','upload-b-exit-code.txt','finalize-exit-code.txt']:
            if (OUT/name).exists():shutil.copy2(OUT/name,history/name)
    manifest=json.loads((DEST/'manifest.json').read_text())
    if manifest['archive_count']!=24:raise RuntimeError('Expected twelve independent ZIPs for each W41 stream')
    for b in manifest['archives']:
        p=ROOT/b['path']
        if p.stat().st_size!=b['size'] or p.stat().st_size>=2*1024**3:raise RuntimeError('Prepared archive size differs: '+b['name'])
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    existing=subprocess.run(['gh','release','view',TAG,'--repo','kaiwen123-yang/MCP-RL'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if existing.returncode:
        subprocess.run(['gh','release','create',TAG,'--repo','kaiwen123-yang/MCP-RL','--target',source,
            '--title','W41 A/B full raw evidence: twelve ZIPs per stream','--notes',
            '24 independent lossless ZIPs covering the existing requested W41 A/B source manifests. Original files and previously uploaded individual assets are retained. See the repository raw-bundles/w41-20261108 manifest for members and verification.','--latest=false'],check=True,cwd=ROOT)
    commands=[]
    for sid in ['a','b']:
        commands.append([sys.executable,'-u','scripts/publish_s1_go2_r12.py','upload','--server-auth','--no-wall-deadline',
            '--manifest',str(DEST/('upload-manifest-'+sid+'.json')),'--upload-workers','12','--asset-timeout-s','7200'])
    clock=previous['clock'] if previous else dict(started_unix_s=time.time(),deadline_unix_s=None,completion_policy='all24archives_verified_or_explicit_technical_failure')
    job=dict(session=session,source_commit=source,clock=clock,archive_count=24,per_stream_archives=12,
        transfer_workers_per_stream=12,total_transfer_workers=24,proxy=PROXY,
        proxy_profile='Dedicated W41-upload Clash profile outside repository; subscription credentials not recorded',
        commands=commands,originals_retained=True,existing_release_assets_retained=True,automatic_delete=False,
        release_tag=TAG,status='launched',invocation_started_unix_s=time.time(),
        prior_invocation_record=str((history/'upload-job.json').relative_to(ROOT)) if history else None)
    write(OUT/'upload-job.json',job);write(DEST/'upload-job.json',job)
    lines=['#!/bin/bash','set -e','cd '+shlex.quote(str(ROOT)),
        'source scripts/autodl_d57_env.sh','unset GH_TOKEN GITHUB_TOKEN',
        'export http_proxy='+PROXY+' https_proxy='+PROXY+' HTTP_PROXY='+PROXY+' HTTPS_PROXY='+PROXY,
        'export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1',
        'unset all_proxy ALL_PROXY','set +e']
    for sid,command in zip(['a','b'],commands):
        lines += [shlex.join(command)+' > '+shlex.quote(str(OUT/('upload-'+sid+'.log')))+' 2>&1 &','pid_'+sid+'=$!']
    for sid in ['a','b']:lines += ['wait "$pid_'+sid+'"','echo "$?" > '+shlex.quote(str(OUT/('upload-'+sid+'-exit-code.txt')))]
    lines += [shlex.join([sys.executable,'-u','scripts/finalize_s1_go2_w41_raw_bundles.py'])+' > '+shlex.quote(str(OUT/'finalize.log'))+' 2>&1',
        'echo "$?" > '+shlex.quote(str(OUT/'finalize-exit-code.txt'))]
    script=OUT/'run-upload.sh';script.write_text('\n'.join(lines)+'\n')
    subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(script)],check=True)
    print(json.dumps(job,indent=2))
if __name__=='__main__':main()
