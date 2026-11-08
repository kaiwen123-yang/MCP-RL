#!/usr/bin/env python3
"""Publish W41 additions with immutable, invocation-specific asset names."""
import argparse,fcntl,json,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
OUT=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu'
SMALL=ROOT/REL/'gpu/w41-gpu'
PART_BYTES=1400000000
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--continuation');args=parser.parse_args()
    SMALL.mkdir(parents=True,exist_ok=True)
    previous=json.loads((SMALL/'delivery.json').read_text()) if (SMALL/'delivery.json').exists() else {}
    for name in ['job.json','clock.json','result.json','mjwarp-result.json']:
        if (OUT/name).exists():(SMALL/name).write_bytes((OUT/name).read_bytes())
    delivery=dict(status='preparing',queued_unix_s=time.time(),automatic_delete=False,originals_retained=True,
        old_W40_assets_reused_without_reupload=True,assets=list(previous.get('assets',[])),
        prior_assets_retained=True,active_continuation=args.continuation)
    tag='tm-s1-go2-w41-gpu-evidence'
    try:
        baseline={}
        if args.continuation:
            history=OUT/'resume-history'/args.continuation
            baseline=json.loads((history/'baseline-sources.json').read_text())
        extensions={'.npz','.json','.jsonl','.log','.xml','.gz','.mjb','.bin'}
        files=[]
        for p in OUT.rglob('*'):
            relative=p.relative_to(OUT)
            if not p.is_file() or p.suffix not in extensions:continue
            if relative.parts[0] in ('resume-history','publication-parts'):continue
            if p.name.startswith('publication') and p.suffix=='.log':continue
            stamp=p.stat();old=baseline.get(str(p.relative_to(ROOT)))
            if old and (stamp.st_size,stamp.st_mtime_ns)==(old['size'],old['mtime_ns']):continue
            files.append(p)
        if args.continuation:
            # A new failure packet belongs to this continuation, while copied
            # G1 reports/logs and their original arrays stay in the old ZIP.
            files+=list(history.glob('first-*-failure-input.npz'))
        else:
            native=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w41-dynamics/native-v2'
            files += [p for p in native.iterdir() if p.is_file()]
        result=json.loads((OUT/'result.json').read_text())
        attempts=[x['attempt'] for x in result.get('stages',{}).get('G4',[])]
        if (OUT/'mjwarp-result.json').exists():
            attempts += [x['attempt'] for x in json.loads((OUT/'mjwarp-result.json').read_text()).get('evaluations',[])]
        for attempt in attempts:
            folder=OUT.parents[1]/attempt
            if folder.exists():files += [p for p in folder.rglob('*') if p.is_file() and p.suffix in extensions]
            for suffix in ['.json','-resolved.json']:
                p=ROOT/REL/(attempt+suffix)
                if p.exists():files.append(p)
        files=sorted(set(files))
        stem='w41-'+args.continuation+'-evidence' if args.continuation else 'w41-new-evidence'
        transport=OUT/'publication-parts'/(args.continuation or 'initial')
        transport.mkdir(parents=True,exist_ok=True);manifest_path=transport/'manifest.json'
        if manifest_path.exists():
            additions=json.loads(manifest_path.read_text())
        else:
            entries=[]
            for i,p in enumerate(files):
                source=str(p.relative_to(ROOT));size=p.stat().st_size
                if size<=PART_BYTES:
                    entries.append(dict(path=p,member=source,source=source,size=size,source_size=size,offset=0))
                else:
                    # Split only the transport copy of a large completed stream.
                    # Concatenating parts reconstructs its original bytes.
                    with p.open('rb') as src:
                        offset=0;part=0
                        while offset<size:
                            part+=1;target=transport/f'file{i:05d}.part{part:04d}'
                            n=min(PART_BYTES,size-offset)
                            with target.open('xb') as dst:
                                left=n
                                while left:
                                    b=src.read(min(4*1024**2,left))
                                    if not b:raise EOFError("Completed source truncated while preparing transport: "+source)
                                    dst.write(b);left-=len(b)
                            entries.append(dict(path=target,member=source+f'.part{part:04d}',source=source,size=n,source_size=size,offset=offset))
                            offset+=n
            packs=[];group=[];size=0
            for row in entries:
                if group and size+row['size']>PART_BYTES:packs.append(group);group=[];size=0
                group.append(row);size+=row['size']
            if group:packs.append(group)
            additions=[]
            for i,group in enumerate(packs,1):
                path=OUT/f'{stem}-{i:02d}.zip'
                mapping=[{k:v for k,v in row.items() if k!='path'} for row in group]
                # Never truncate an existing local archive or replace a remote asset.
                with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                    for row in group:z.write(row['path'],row['member'])
                    z.writestr('manifests/'+path.stem+'.json',json.dumps(mapping,indent=2))
                additions.append(dict(name=path.name,path=str(path.relative_to(ROOT)),size=path.stat().st_size,
                    sources=sorted({row['source'] for row in group}),member_mapping=mapping))
            write(manifest_path,additions)
        names={x['name'] for x in additions}
        delivery['assets']=[x for x in delivery['assets'] if x['name'] not in names]+additions
        delivery.update(new_asset_names=sorted(names),continuation_manifest=str(manifest_path.relative_to(ROOT)),
            unchanged_G1_sources_reuploaded=False if args.continuation else None)
        exists=subprocess.run(['gh','release','view',tag,'--repo','kaiwen123-yang/MCP-RL'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if not exists:subprocess.run(['gh','release','create',tag,'--repo','kaiwen123-yang/MCP-RL','--target','main','--title','W41 held dynamics and GPU evidence','--notes','Actual completed stages and retained failure records; see result.json for current scope.'],check=True,cwd=ROOT)
        def remote():
            release=json.loads(subprocess.check_output(['gh','api',f'repos/kaiwen123-yang/MCP-RL/releases/tags/{tag}'],text=True))
            pages=json.loads(subprocess.check_output(['gh','api',f'repos/kaiwen123-yang/MCP-RL/releases/{release["id"]}/assets?per_page=100','--paginate','--slurp'],text=True))
            return {a['name']:a for page in pages for a in page}
        byname=remote()
        for item in additions:
            actual=byname.get(item['name'])
            if actual:
                if actual['state']!='uploaded' or actual['size']!=item['size']:
                    raise RuntimeError('Existing asset retained without replacement: '+item['name'])
                continue
            subprocess.run(['gh','release','upload',tag,str(ROOT/item['path']),'--repo','kaiwen123-yang/MCP-RL'],check=True,cwd=ROOT)
        byname=remote()
        for item in delivery['assets']:
            actual=byname[item['name']]
            item.update(remote_size=actual['size'],url=actual['browser_download_url'],digest=actual.get('digest'),
                verified=actual['state']=='uploaded' and actual['size']==item['size'])
        delivery.update(status='published' if all(x['verified'] for x in delivery['assets']) else 'size_mismatch',release_tag=tag)
    except Exception as e:delivery.update(status='publication_interrupted',error=repr(e),release_tag=tag)
    write(SMALL/'delivery.json',delivery)
    with (ROOT/REL/'compact-finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(['git','add',str(SMALL.relative_to(ROOT))],cwd=ROOT,check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',str(SMALL.relative_to(ROOT))],cwd=ROOT).returncode:
            subprocess.run(['git','commit','-m','Publish actual W41 GPU continuation evidence without replacing prior assets','--',str(SMALL.relative_to(ROOT))],cwd=ROOT,check=True)
        subprocess.run(['git','fetch','origin','main'],cwd=ROOT,check=True);subprocess.run(['git','merge','--no-edit','origin/main'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
if __name__=='__main__':main()
