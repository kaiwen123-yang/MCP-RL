#!/usr/bin/env python3
"""Package the existing W40 raw-evidence selection into12independent ZIPs on D57."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,time,zipfile,subprocess
ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-local-readiness-r12'
LARGE=ROOT/'outputs/terrain_memory/simulation'/RUN
SMALL=ROOT/'results/terrain_memory/simulation'/RUN
OUT=LARGE/'raw-bundles/w40-20261107'
DEST=SMALL/'raw-bundles/w40-20261107'
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def member(row,sid):
    source=Path(row['source_path']);attempt=row.get('attempt')
    if attempt:
        case='ep'+attempt.rsplit('-ep',1)[1] if '-ep' in attempt else 'evaluation-'+attempt.rsplit('-',1)[1]
        prefix=f'w40-{sid}/{case}/'
        episode=LARGE/attempt
        path=ROOT/source
        if path.is_relative_to(episode):return prefix+path.relative_to(episode).as_posix()
        if source.name==attempt+'.json':return prefix+'result.json'
        if source.name==attempt+'-resolved.json':return prefix+'resolved.json'
        return prefix+source.name
    return f'w40-{sid}/training/'+source.name
def main():
    OUT.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
    plans=[];clocks={};seen=set()
    for sid in ('a','b'):
        job=f'job-20261106T103028Z-w40-{sid}'
        ev=json.loads((SMALL/'jobs'/job/'diagnostics/evidence.json').read_text())
        if ev['missing']:raise RuntimeError('Requested source missing: '+job)
        clocks[sid]=json.loads((LARGE/job/'diagnostics-transfer/delivery-windows/w40-raw-20261107-clash/deadline.json').read_text())
        bins=[dict(stream_id='w40-'+sid,job=job,part=i+1,parts=6,estimated_bytes=0,files=[]) for i in range(6)]
        for row in sorted(ev['files'],key=lambda x:sum(p['size'] for p in x['parts']),reverse=True):
            path=ROOT/row['source_path'];stamp=path.stat()
            if stamp.st_size!=row['source_size'] or stamp.st_mtime_ns!=row['source_mtime_ns']:
                raise RuntimeError('Closed source changed: '+row['source_path'])
            arc=member(row,sid)
            if arc in seen:raise RuntimeError('Duplicate archive member: '+arc)
            seen.add(arc)
            item={k:row[k] for k in ('source_path','source_size','source_mtime_ns','attempt','role','source_commit','execution_fingerprint') if k in row}
            item['member']=arc
            target=min(bins,key=lambda x:x['estimated_bytes'])
            target['files'].append(item);target['estimated_bytes']+=sum(p['size'] for p in row['parts'])
        plans+=bins
    deadline=min(c['deadline_unix_s'] for c in clocks.values())
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(DEST/'packaging-plan.json',dict(schema='w40_12_zip_plan_v1',source_commit=source_commit,
       clocks=clocks,deadline_unix_s=deadline,archive_count=12,files=len(seen),
       source_bytes=sum(f['source_size'] for b in plans for f in b['files']),
       compression='ZIP_STORED for existingNPZ/gzip; deflate level1 for other original files; original file bytes preserved',
       archive_layout='Six independent ZIPs per stream, short member paths map back to original source paths',
       package_workers=2,upload_workers=12,bundles=[{k:v for k,v in b.items() if k!='files'} for b in plans]))
    def pack(b):
        if time.time()>=deadline:raise TimeoutError('Existing delivery deadline reached')
        name=f'{b["stream_id"]}-raw-{b["part"]:02d}-of-06.zip'
        final=OUT/name;partial=OUT/(name+'.partial')
        if final.exists() or partial.exists():raise RuntimeError('Preserve existing archive; inspect before rerunning: '+name)
        begin=time.time();files=[]
        with zipfile.ZipFile(partial,'w',allowZip64=True) as z:
            for row in b['files']:
                p=ROOT/row['source_path']
                codec=zipfile.ZIP_STORED if p.suffix in ('.npz','.gz','.png','.mp4') else zipfile.ZIP_DEFLATED
                z.write(p,arcname=row['member'],compress_type=codec,compresslevel=1)
                info=z.getinfo(row['member'])
                files.append(dict(row,zip_crc32=f'{info.CRC:08x}',compressed_size=info.compress_size))
            z.writestr('manifests/'+name[:-4]+'.json',json.dumps(dict(
                schema='w40_raw_bundle_v1',archive=name,stream_id=b['stream_id'],job=b['job'],
                part=b['part'],parts=6,source_commit=source_commit,files=files),
                indent=2,ensure_ascii=False),compress_type=zipfile.ZIP_DEFLATED,compresslevel=1)
        partial.rename(final)
        with zipfile.ZipFile(final) as z:
            infos={i.filename:i for i in z.infolist()}
            if len(infos)!=len(files)+1 or any(infos[f['member']].file_size!=f['source_size'] for f in files):
                raise RuntimeError('ZIP directory does not match selected sources')
        if final.stat().st_size>=2*1024**3:raise RuntimeError('Archive exceeds GitHub asset limit: '+name)
        result=dict(name=name,path=str(final.relative_to(ROOT)),stream_id=b['stream_id'],job=b['job'],
            file_count=len(files),source_bytes=sum(f['source_size'] for f in files),
            size=final.stat().st_size,wall_s=time.time()-begin,files=files)
        write(DEST/(name+'.json'),result)
        print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:bundles=list(pool.map(pack,plans))
    result=dict(schema='w40_12_zip_delivery_v1',status='packaged_pending_upload',
        source_commit=source_commit,deadline_unix_s=deadline,archive_count=len(bundles),
        source_files=sum(x['file_count'] for x in bundles),source_bytes=sum(x['source_bytes'] for x in bundles),
        archive_bytes=sum(x['size'] for x in bundles),archives=bundles,
        originals_retained=True,individual_remote_assets_retained=True,computer_archive=False)
    write(DEST/'manifest.json',result)
    write(DEST/'upload-manifest.json',[dict(path=b['path'],name=b['name'],category='W40_requested_raw_zip') for b in bundles])
    print(json.dumps({k:v for k,v in result.items() if k!='archives'}),flush=True)
if __name__=='__main__':main()
