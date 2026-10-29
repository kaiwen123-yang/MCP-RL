#!/usr/bin/env python3
"""Package completed R11 execution artifacts on the execution host only."""
import hashlib,json,tarfile
from pathlib import Path
from s1_go2_r11_protocol import ROOT,CONTRACT,RUN
from s1_go2_r11_runtime import index_file
SMALL=ROOT/CONTRACT['outputs']['compact_root']
LARGE=ROOT/CONTRACT['outputs']['large_root']
DEST=LARGE/'release-assets'
LIMIT=256*1024**2

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def asset(path,name,scope):
    return dict(name=name,path=str(path.relative_to(ROOT)),size=path.stat().st_size,
        sha256=digest(path),scope=scope,
        url=f'https://github.com/kaiwen123-yang/MCP-RL/releases/download/{RUN}/{name}')

def pack(label,base,files):
    directory=DEST/label
    directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'manifest.json'
    if manifest.exists():return json.loads(manifest.read_text())
    groups=[];current=[];size=0
    for path in files:
        count=path.stat().st_size
        if current and size+count>LIMIT:groups.append(current);current=[];size=0
        current.append(path);size+=count
    if current:groups.append(current)
    records=[]
    for part,paths in enumerate(groups,1):
        name=f'{label}-arrays-{part:02d}.tar.gz'
        archive=directory/name
        with tarfile.open(archive,'w:gz',compresslevel=2,dereference=True) as tar:
            for path in paths:tar.add(path,arcname=path.relative_to(base),recursive=False)
        record=asset(archive,name,'Original completed execution files; server-side packaging')
        record['members']=[dict(path=str(path.relative_to(base)),size=path.stat().st_size,
            sha256=digest(path)) for path in paths]
        records.append(record)
    manifest.write_text(json.dumps(records,indent=2)+'\n')
    return records

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    with index_file(SMALL,CONTRACT) as state:rows=list(state['attempts'])
    records=[]
    for row in rows:
        if not row.get('result'):continue
        name=row['attempt'];directory=LARGE/name
        # Every original chunk and log stays on the execution host.
        files=sorted(p for p in directory.rglob('*') if p.is_file() and
            p.suffix!='.mp4' and p.name!='video-asset.json')
        records.extend(pack(name,directory,files))
        video=directory/'video-asset.json'
        if video.exists():
            metadata=json.loads(video.read_text());path=Path(metadata['path'])
            records.append(asset(path,name+'-full-mission.mp4',metadata['scope']))
    # Source-derived diagnostics contain no additional robot integration.
    for name in ['r10-source-analysis','observer-retained-dev11']:
        directory=LARGE/name
        if directory.exists():
            files=sorted(p for p in directory.rglob('*') if p.is_file())
            records.extend(pack(name,directory,files))
    for directory in sorted((ROOT/'work/r11-frozen').iterdir()):
        if not directory.is_dir():continue
        names=['agent_server','identity.json','SOURCE_COMMIT','lib/libmujoco.so.3.2.3']
        records.extend(pack('controller-'+directory.name,directory,[directory/n for n in names]))
    with index_file(SMALL,CONTRACT) as state:
        state['assets']=records
        state['asset_packaging']=dict(location='execution_host_only',
            files=len(records),bytes=sum(r['size'] for r in records),
            original_outputs_retained=True,local_computer_artifact_downloads=0)
    print(json.dumps(dict(assets=len(records),bytes=sum(r['size'] for r in records))),flush=True)
if __name__=='__main__':main()
