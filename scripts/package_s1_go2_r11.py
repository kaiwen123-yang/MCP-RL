#!/usr/bin/env python3
"""Package completed R11 execution artifacts on the execution host only."""
import argparse,gzip,hashlib,io,json,sys,tarfile,zlib
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

def build_memory_archive(base,files,deduplicate=False):
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',filename='',mtime=0,compresslevel=2) as compressed:
        with tarfile.open(fileobj=compressed,mode='w',dereference=True) as tar:
            seen={}
            for path in files:
                info=tar.gettarinfo(str(path),arcname=str(path.relative_to(base)))
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                if deduplicate:
                    key=(digest(path),info.mode)
                    if key in seen:
                        info.type=tarfile.LNKTYPE;info.linkname=seen[key];info.size=0
                        tar.addfile(info);continue
                    seen[key]=info.name
                with path.open('rb') as stream:tar.addfile(info,stream)
    buffer.seek(0)
    return buffer

def pack(label,base,files,deduplicate=False):
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
    if deduplicate:groups=[files]
    records=[]
    for part,paths in enumerate(groups,1):
        name=f'{label}-arrays-{part:02d}.tar.gz'
        payload=build_memory_archive(base,paths,deduplicate=deduplicate)
        size=payload.getbuffer().nbytes
        checksum=hashlib.sha256(payload.getbuffer()).hexdigest()
        record=dict(name=name,size=size,sha256=checksum,archive_mode='memory_tar_gzip_v1',
            source_directory=str(base.relative_to(ROOT)),manifest_path=str(manifest.relative_to(ROOT)),
            scope='Original completed execution files; deterministic archive built in execution-host RAM and uploaded without a derived disk copy',
            creation_runtime=dict(python=sys.executable,zlib=zlib.ZLIB_RUNTIME_VERSION),
            url=f'https://github.com/kaiwen123-yang/MCP-RL/releases/download/{RUN}/{name}')
        record['deduplicate']=deduplicate
        record['members']=[dict(path=str(path.relative_to(base)),size=path.stat().st_size,
            sha256=digest(path)) for path in paths]
        payload.close();records.append(record)
    manifest.write_text(json.dumps(records,indent=2)+'\n')
    return records

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-training',action='store_true');args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    with index_file(SMALL,CONTRACT) as state:rows=list(state['attempts'])
    records=[]
    for row in rows:
        if not row.get('result'):continue
        name=row['attempt'];directory=LARGE/name
        # Every original chunk and log stays on the execution host.
        files=sorted(p for p in directory.rglob('*') if p.is_file() and
            p.suffix!='.mp4' and p.name!='video-asset.json')
        original=pack(name,directory,files)
        records.extend(original)
        covered={m['path']:m for item in original for m in item['members']}
        supplemental=[p for p in files if str(p.relative_to(directory)) not in covered]
        if supplemental:records.extend(pack(name+'-retained-supplement',directory,supplemental))
        video=directory/'video-asset.json'
        if video.exists():
            metadata=json.loads(video.read_text());path=Path(metadata['path'])
            records.append(asset(path,name+'-full-mission.mp4',metadata['scope']))
            for variant in sorted(directory.glob('*.mp4')):
                if variant.resolve()!=path.resolve():
                    records.append(asset(variant,name+'-retained-'+variant.name,
                        'Earlier rendering retained for completeness; the canonical full-mission asset and index contain current qualification status.'))
    # Source-derived diagnostics contain no additional robot integration.
    for name in ['r10-source-analysis','observer-retained-dev11']:
        directory=LARGE/name
        if directory.exists():
            files=sorted(p for p in directory.rglob('*') if p.is_file())
            original=pack(name,directory,files)
        records.extend(original)
        covered={m['path']:m for item in original for m in item['members']}
        supplemental=[p for p in files if str(p.relative_to(directory)) not in covered]
        if supplemental:records.extend(pack(name+'-retained-supplement',directory,supplemental))
    if args.include_training and (LARGE/'training').exists():
        # Final publication only, after all requested training/resume activity ends.
        for directory in sorted((LARGE/'training').iterdir()):
            if directory.is_dir():
                records.extend(pack('training-'+directory.name,directory,
                    sorted(p for p in directory.rglob('*') if p.is_file())))
    for directory in sorted((ROOT/'work/r11-frozen').iterdir()):
        if not directory.is_dir():continue
        names=['agent_server','identity.json','SOURCE_COMMIT','lib/libmujoco.so.3.2.3']
        records.extend(pack('controller-'+directory.name,directory,[directory/n for n in names]))
    frozen=ROOT/'work/r11-frozen'
    sources=sorted(p for p in frozen.rglob('*') if p.is_file() and
        p.relative_to(frozen).parts[1] in ('mjpc','runtime-tasks'))
    records.extend(pack('controller-source-snapshots',frozen,sources,deduplicate=True))
    evidence=sorted(p for p in (ROOT/'work').glob('r11-*') if p.is_file() and
        p.suffix in ('.log','.json') and p.name not in
        ('r11-final-package.log','r11-publish-payload.json'))
    evidence.extend(sorted(LARGE.glob('dev*/video-asset.json')))
    records.extend(pack('execution-logs-and-analysis',ROOT,evidence))
    with index_file(SMALL,CONTRACT) as state:
        state['publication_coverage']=dict(attempts=len(rows),
            task_arrays=sum('-E30-arrays-' in r['name'] or '-E60-arrays-' in r['name'] or '-E30-repro-arrays-' in r['name'] for r in records),
            canonical_videos=sum(r['name'].endswith('-full-mission.mp4') for r in records),
            frozen_controller_versions=len([p for p in frozen.iterdir() if p.is_dir()]),
            controller_source_files=len(sources),execution_logs_and_analysis_files=len(evidence),
            source_snapshot_tar_hardlinks='Byte-identical source/model files use standard tar hardlinks; all original versioned paths are retained.')
        state['assets']=records
        state['asset_packaging']=dict(location='execution_host_only',
            files=len(records),bytes=sum(r['size'] for r in records),
            original_outputs_retained=True,local_computer_artifact_downloads=0,
            new_archive_storage='execution_host_RAM',existing_disk_archives_preserved=True)
    print(json.dumps(dict(assets=len(records),bytes=sum(r['size'] for r in records))),flush=True)
if __name__=='__main__':main()
