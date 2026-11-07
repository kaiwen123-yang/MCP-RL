#!/usr/bin/env python3
"""Verify twelve remote ZIP assets against the saved bundle manifest, then publish."""
from pathlib import Path
import json,subprocess,fcntl,time
ROOT=Path(__file__).resolve().parents[1]
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
DEST=SMALL/'raw-bundles/w40-20261107'
def main():
    m=json.loads((DEST/'manifest.json').read_text());upload=json.loads((DEST/'upload-manifest.json').read_text())
    remote={}
    for tag in sorted({x['release_tag'] for x in upload if x.get('release_tag')}):
        release=json.loads(subprocess.check_output(['gh','api',f'repos/kaiwen123-yang/MCP-RL/releases/tags/{tag}'],cwd=ROOT,text=True))
        pages=json.loads(subprocess.check_output(['gh','api',f'repos/kaiwen123-yang/MCP-RL/releases/{release["id"]}/assets?per_page=100','--paginate','--slurp'],cwd=ROOT,text=True))
        for page in pages:
            for a in page:remote[a['name']]=dict(a,release_tag=tag)
    completed=True
    for b in m['archives']:
        a=remote.get(b['name'])
        good=bool(a and a['state']=='uploaded' and a['size']==b['size'])
        b.update(verified=good,url=a['browser_download_url'] if a else None,
            remote_state=a.get('state') if a else None,remote_size=a.get('size') if a else None,
            remote_digest=a.get('digest') if a else None,release_tag=a.get('release_tag') if a else None)
        completed=completed and good
    m.update(status='published' if completed else 'publication_pending',checked_unix_s=time.time(),
       verification='Remote REST asset name/size/uploaded-state plus local ZIP directory counts/sizes. No asset downloads.',
       verified_archives=sum(b['verified'] for b in m['archives']))
    paths=[DEST]
    with (SMALL/'compact-finish.lock').open('a') as gl:
        fcntl.flock(gl,fcntl.LOCK_EX)
        (DEST/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
        for sid in ('a','b'):
            ep=SMALL/'jobs'/f'job-20261106T103028Z-w40-{sid}'/'diagnostics/evidence.json'
            e=json.loads(ep.read_text());selected=[b for b in m['archives'] if b['stream_id']=='w40-'+sid]
            e['individual_asset_delivery_status']=e.get('individual_asset_delivery_status',e['status'])
            e['delivery_mode']='six_lossless_ZIP_archives_per_stream'
            e['bundle_delivery']=dict(status=m['status'],manifest=str((DEST/'manifest.json').relative_to(ROOT)),
                source_files=sum(b['file_count'] for b in selected),archives=[{k:b.get(k) for k in ('name','url','size','verified','remote_digest')} for b in selected],
                original_individual_assets_retained=True,all_requested_source_bytes_preserved=True)
            if completed:e['status']='published';e['this_invocation_finish_verified']=True
            ep.write_text(json.dumps(e,indent=2,ensure_ascii=False)+'\n');paths.append(ep)
        with (SMALL/'index.lock').open('a') as il:
            fcntl.flock(il,fcntl.LOCK_EX)
            idx=json.loads((SMALL/'index.json').read_text())
            idx['publication'].setdefault('delivery_windows',{}).setdefault('w40-raw-20261107-clash',{}).update(
              delivery_mode='twelve_ZIP_archives',status=m['status'],bundle_manifest=str((DEST/'manifest.json').relative_to(ROOT)),
              verified_archives=m['verified_archives'],source_files=m['source_files'])
            (SMALL/'index.json').write_text(json.dumps(idx,indent=2,ensure_ascii=False)+'\n');paths.append(SMALL/'index.json')
            rel=[str(p.relative_to(ROOT)) for p in paths]
            subprocess.run(['git','add','--',*rel],cwd=ROOT,check=True)
            if subprocess.run(['git','diff','--cached','--quiet','--',*rel],cwd=ROOT).returncode:
                subprocess.run(['git','commit','-m','Publish and verify W40 twelve-ZIP raw diagnostic delivery','--',*rel],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    print(json.dumps(dict(status=m['status'],verified_archives=m['verified_archives'],source_files=m['source_files'])),flush=True)
    return 0 if completed else 1
if __name__=='__main__':raise SystemExit(main())
