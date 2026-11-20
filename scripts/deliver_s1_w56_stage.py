"""Publish one naturally finished W56 case's originals and full video from D57."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

import publish_s1_go2_r12 as transport
from publish_s1_go2_w55 import prepare
from run_s1_w56_f import ROOT, REL, RUN, WORK, commit, read, write

TAG=RUN+'-w56'
STAGING=ROOT/'outputs/terrain_memory/simulation'/RUN/'w56-delivery'


def deliver(workspace, label, level):
    status=read(workspace/REL/'w56'/label[0].lower()/label/(level+'.json'))
    attempt=status['attempt']
    raw=workspace/'outputs/terrain_memory/simulation'/RUN
    episode=raw/attempt
    job=raw/'jobs'/status['job']
    group=(label+'-'+level).lower()
    dest=ROOT/REL/'w56/delivery'/group
    staging=STAGING/group
    dest.mkdir(parents=True,exist_ok=True)
    staging.mkdir(parents=True,exist_ok=True)
    with (STAGING/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=dest/'manifest.json'
        if not manifest.exists():
            video=read(episode/'video-asset.json')
            if Path(video['path']).stat().st_size != video['size']:
                raise ValueError('Recorded video size changed')
            snapshot=staging/'job-snapshot'
            snapshot.mkdir(exist_ok=True)
            resources=job/'resources.jsonl'
            if resources.is_file():
                with resources.open() as source,(snapshot/'resources.jsonl').open('w') as target:
                    for line in source:
                        if json.loads(line)['root_pid']==status['pid']:
                            target.write(line)
            write(snapshot/'stage.json',status)
            inputs=[episode,snapshot]
            names=('job.json','execution-contract.json',attempt+'.log',level+'-runner.log',level+'-runtime.json')
            inputs.extend(job/name for name in names if (job/name).is_file())
            policy=raw/'evaluation-policy'/attempt
            if policy.is_dir():
                inputs.append(policy)
            prepare(group,inputs,dest,prefix='w56',staging_root=STAGING,tag=TAG)
        if not (staging/'index.json').exists():
            write(staging/'index.json',{})
        transport.RUN=TAG
        transport.SMALL=staging
        for action in ('upload','finish'):
            sys.argv=[__file__,action,'--manifest',str(manifest),'--server-auth',
                      '--upload-workers','1','--no-wall-deadline']
            transport.main()
        assets={row['name']:row for row in read(staging/'index.json')['publication']['assets']}
        verified=[]
        for item in read(manifest):
            remote=assets[item['name']]
            if (remote['status'],remote['remote_state'],remote['size'],remote['remote_digest']) != (
                    'uploaded','uploaded',item['size'],'sha256:'+item['sha256']):
                raise ValueError('Release asset readback mismatch: '+item['name'])
            verified.append({key:remote[key] for key in ('name','size','url','remote_digest','remote_asset_id',
                                                         'remote_verified_unix_s','remote_state')})
        write(dest/'delivery.json',dict(status='published',attempt=attempt,assets=verified,
              original_files=len(read(dest/'sources.json')),checked_unix_s=time.time(),
              deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False))
        commit([manifest,dest/'sources.json',dest/'delivery.json'],f'Verify W56 {label} {level} Release assets')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--case',required=True)
    p.add_argument('--level',required=True)
    args=p.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    deliver(args.workspace.resolve(),args.case,args.level)
