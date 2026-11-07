#!/usr/bin/env python3
"""Publish only new W41 predictor evidence after computation exits; retain originals."""
import fcntl,json,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
OUT=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu'
SMALL=ROOT/REL/'gpu/w41-gpu'
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    SMALL.mkdir(parents=True,exist_ok=True)
    for name in ['job.json','clock.json','result.json','mjwarp-result.json']:
        if (OUT/name).exists():(SMALL/name).write_bytes((OUT/name).read_bytes())
    delivery=dict(status='preparing',queued_unix_s=time.time(),automatic_delete=False,originals_retained=True,
        old_W40_assets_reused_without_reupload=True,assets=[])
    tag='tm-s1-go2-w41-gpu-evidence'
    try:
        files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.npz','.json','.jsonl','.log','.xml','.gz')]
        native=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w41-dynamics/native-v2'
        files += [p for p in native.iterdir() if p.is_file()]
        # New source arrays only. ZIP_STORED retains compressed NPZ bytes and
        # avoids recompressing old/raw research or creating a computer archive.
        packs=[];group=[];size=0
        for p in sorted(files):
            if group and size+p.stat().st_size>1500000000:packs.append(group);group=[];size=0
            group.append(p);size+=p.stat().st_size
        if group:packs.append(group)
        assets=[]
        for i,group in enumerate(packs,1):
            path=OUT/f'w41-new-evidence-{i:02d}.zip'
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                for p in group:z.write(p,str(p.relative_to(ROOT)))
            assets.append(path);delivery['assets'].append(dict(name=path.name,size=path.stat().st_size,sources=[str(p.relative_to(ROOT)) for p in group]))
        exists=subprocess.run(['gh','release','view',tag,'--repo','kaiwen123-yang/MCP-RL'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if not exists:subprocess.run(['gh','release','create',tag,'--repo','kaiwen123-yang/MCP-RL','--target','main','--title','W41 held dynamics and GPU evidence','--notes','Actual completed numerical and independent predictor stages; see result.json for remaining work or first failure. No hardware capability claim.'],check=True,cwd=ROOT)
        published=json.loads(subprocess.check_output(['gh','release','view',tag,'--repo','kaiwen123-yang/MCP-RL','--json','assets'],text=True))['assets'];byname={x['name']:x for x in published}
        for path in assets:
            if path.name not in byname:subprocess.run(['gh','release','upload',tag,str(path),'--repo','kaiwen123-yang/MCP-RL'],check=True,cwd=ROOT)
        published=json.loads(subprocess.check_output(['gh','release','view',tag,'--repo','kaiwen123-yang/MCP-RL','--json','assets'],text=True))['assets'];byname={x['name']:x for x in published}
        for x in delivery['assets']:
            a=byname[x['name']];x.update(remote_size=a['size'],url=a['url'],digest=a.get('digest'),verified=a['size']==x['size'])
        delivery.update(status='published' if all(x['verified'] for x in delivery['assets']) else 'size_mismatch',release_tag=tag)
    except Exception as e:delivery.update(status='publication_interrupted',error=repr(e),release_tag=tag)
    write(SMALL/'delivery.json',delivery)
    with (ROOT/REL/'compact-finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(['git','add',str(SMALL.relative_to(ROOT))],cwd=ROOT,check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',str(SMALL.relative_to(ROOT))],cwd=ROOT).returncode:
            subprocess.run(['git','commit','-m','Publish actual W41 GPU evidence and delivery state','--',str(SMALL.relative_to(ROOT))],cwd=ROOT,check=True)
        subprocess.run(['git','fetch','origin','main'],cwd=ROOT,check=True)
        subprocess.run(['git','merge','--no-edit','origin/main'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
if __name__=='__main__':main()
