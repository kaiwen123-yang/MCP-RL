"""Release each new range of true PPO batches/checkpoints once when its declared W57 node closes."""
from pathlib import Path
import subprocess
import sys
import time

from publish_s1_w57 import ROOT, SMALL, read

CAMPAIGN=ROOT/'work/w57-r/pilot02'


def publish(label, files):
    dest=SMALL/'delivery'/(label+'.json')
    if dest.exists() and read(dest).get('status')=='verified':return
    subprocess.run([sys.executable,'-u','scripts/publish_s1_w57.py','bundle',label,
                    *(str(p) for p in files)],cwd=ROOT,check=True)


def recorded_nodes(head):
    run=CAMPAIGN/head
    previous=0
    for path in sorted(run.glob('node-*.json')):
        node=read(path);version=node['version']
        label=f'w57-pilot02-{head}-updates-{previous+1:02d}-{version:02d}'
        files=[path,run/(path.stem+'-matched-loads.jsonl.gz')]
        if not all(p.is_file() for p in files):return
        if not previous:files += [run/'run.json',run/'policy-000.pt',run/'collector-A.json',run/'collector-B.json']
        for v in range(previous,version):
            files += [run/f'policy-{v+1:03d}.pt',run/f'update-{v+1:03d}.json']
            for case in ('A','B'):
                files += [run/f'batch-{v:03d}-{case}.jsonl',run/f'ready-{v:03d}-{case}.json']
        publish(label,files)
        previous=version


def main():
    while True:
        for head in ('full','blind'):
            recorded_nodes(head)
        final=CAMPAIGN/'final.json'
        if final.exists():
            files=[final]
            for head in ('full','blind'):
                run=CAMPAIGN/head
                if run.is_dir():
                    files += [run/'frozen-evaluation.json',run/'published.json',run/'dispatch.jsonl']
                    for case in ('A','B'):
                        files += [run/f'episodes-{case}.jsonl',run/f'episode-ends-{case}.jsonl']
            publish('w57-pilot02-final-learning-state',files)
            return
        time.sleep(45)


if __name__=='__main__':main()
