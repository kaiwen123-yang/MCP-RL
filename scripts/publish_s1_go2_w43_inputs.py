#!/usr/bin/env python3
"""Publish only W43 solver inputs and the final new artifact; retain all originals."""
import hashlib,json,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w43-optimizer-inputs'
OUT=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w43-optimizer-inputs'
def main():
    tag='w43-r32-inputs-20261110'
    archive=OUT/'w43-r32-inputs-and-artifact.zip'
    delivery=dict(status='preparing',started_unix_s=time.time(),deadline_unix_s=None,
        release_tag=tag,originals_retained=True,automatic_cleanup=False,
        raw_scope='Only this W43 invocation: final and first-failure saved inputs, logs, numeric snapshots, r32 patch/spec/binary/build identity. No historical Release download or reupload.')
    def save():(SMALL/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
    save()
    try:
        files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in {'.npz','.json','.jsonl','.log','.bin','.search','.patch'}]
        files += [ROOT/'work/r11-frozen/S5-r32'/name for name in ('agent_server','identity.json','build.log')]
        files += [ROOT/'experiments/terrain_memory'/('s1-go2-r11-S5-r32'+suffix) for suffix in ('.json','.patch')]
        files += [p for p in SMALL.glob('*.json') if p.name!='delivery.json']
        if not archive.exists():
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                for p in sorted(set(files)):z.write(p,str(p.relative_to(ROOT)))
        digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
        delivery.update(status='queued',archive=str(archive.relative_to(ROOT)),size=archive.stat().st_size,
            sha256=digest,source_count=len(set(files)))
        save()
        repo='kaiwen123-yang/MCP-RL'
        if subprocess.run(['gh','release','view',tag,'--repo',repo],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            subprocess.run(['gh','release','create',tag,'--repo',repo,'--target','865af7a91fb773219d45903d74a544308e2ebdb9',
                '--title','W43 r32 actual solver input evidence and artifact',
                '--notes','Normal r32 build and saved W42 input sequences; includes retained first-build wiring failure and sparse A4s native failure. Not a task-qualification claim.'],check=True)
        release=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/releases/tags/{tag}'],text=True))
        old=next((a for a in release['assets'] if a['name']==archive.name),None)
        if old is None:
            subprocess.run(['gh','release','upload',tag,str(archive),'--repo',repo],check=True)
        release=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/releases/tags/{tag}'],text=True))
        asset=next(a for a in release['assets'] if a['name']==archive.name)
        if asset['state']!='uploaded' or asset['size']!=delivery['size'] or asset.get('digest')!='sha256:'+digest:
            raise RuntimeError('Remote name/state/size/digest did not match; existing asset preserved.')
        delivery.update(status='published',verified_unix_s=time.time(),asset_id=asset['id'],
            url=asset['browser_download_url'],remote_size=asset['size'],remote_digest=asset['digest'],
            release_url=release['html_url'],verification='Remote metadata only; no archive download.')
    except Exception as e:
        delivery.update(status='publication_pending',error=repr(e))
    save()
    print(json.dumps(delivery,indent=2))
if __name__=='__main__':main()
