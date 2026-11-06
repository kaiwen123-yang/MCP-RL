#!/usr/bin/env python3
"""Publish the actual bounded GPU task result and its saved numerical arrays."""
import fcntl,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
OUT=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w40-gpu'
SMALL=ROOT/REL/'gpu/w40-gpu'
def run(*args):
    return subprocess.check_output(args,cwd=ROOT,text=True).strip()
def main():
    SMALL.mkdir(parents=True,exist_ok=True)
    job=json.loads((OUT/'job.json').read_text())
    result=json.loads((OUT/'result.json').read_text()) if (OUT/'result.json').exists() else {'status':'technical_interruption_before_result'}
    for name in ('job.json','clock.json'):
        (SMALL/name).write_bytes((OUT/name).read_bytes())
    (SMALL/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    delivery={'status':'preparing','arrays':[],'retained_locally':True,'automatic_delete':False}
    tag='tm-s1-go2-w40-gpu-'+str(int(job['clock']['started_unix_s']))
    try:
        if subprocess.run(['gh','release','view',tag,'--repo','kaiwen123-yang/MCP-RL'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            subprocess.run(['gh','release','create',tag,'--repo','kaiwen123-yang/MCP-RL',
                '--target',job['source_commit'],'--title','W40 GPU predictor numerical evidence',
                '--notes','Independent bounded GPU task. See result.json for actual completed layer and numerical differences; no policy qualification or main hot swap.'],cwd=ROOT,check=True)
        assets=json.loads(run('gh','release','view',tag,'--repo','kaiwen123-yang/MCP-RL','--json','assets'))['assets']
        existing={x['name']:x for x in assets}
        for path in sorted(OUT.glob('*.npz')):
            if path.name not in existing:
                subprocess.run(['gh','release','upload',tag,str(path),'--repo','kaiwen123-yang/MCP-RL'],cwd=ROOT,check=True)
            delivery['arrays'].append(dict(name=path.name,size=path.stat().st_size,local=str(path.relative_to(ROOT))))
        assets=json.loads(run('gh','release','view',tag,'--repo','kaiwen123-yang/MCP-RL','--json','assets'))['assets']
        byname={x['name']:x for x in assets}
        for item in delivery['arrays']:
            a=byname[item['name']];item.update(url=a['url'],remote_size=a['size'],verified=a['size']==item['size'])
        delivery.update(status='published' if all(x['verified'] for x in delivery['arrays']) else 'size_mismatch',release_tag=tag)
    except Exception as error:
        delivery.update(status='publication_interrupted',error=str(error),release_tag=tag)
    (SMALL/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
    # Reuse the existing short Git serialization lock; no network-transfer lock.
    with (ROOT/REL/'compact-finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(['git','add',str(SMALL.relative_to(ROOT))],cwd=ROOT,check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',str(SMALL.relative_to(ROOT))],cwd=ROOT).returncode:
            subprocess.run(['git','commit','-m','Publish actual W40 GPU numerical task results','--',str(SMALL.relative_to(ROOT))],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
if __name__=='__main__':main()
