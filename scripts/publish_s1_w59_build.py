"""Publish the signed-off r59 native/plant ABIs and original saved-input evidence."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

from run_s1_w59_n import ROOT,WORK,REL,commit
from publish_s1_go2_w54 import write

RUN='tm-s1-go2-local-readiness-r12'
TAG=RUN+'-w56'
REPO='kaiwen123-yang/MCP-RL'
DEST=ROOT/REL.parent/'delivery/n-r59-build'
STAGING=ROOT/'outputs/terrain_memory/simulation'/RUN/'w56-delivery/n-r59-build'
NAME='w56-n-r59-build-originals-0001.zip'
OLD=ROOT.parent/'MCP-RL-w55-n-r56'

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    STAGING.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
    with (STAGING/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        archive=STAGING/NAME
        catalog=DEST/'sources.json'
        manifest=DEST/'manifest.json'
        build=ROOT/'work/r11-frozen/S5-r59';identity=json.loads((build/'identity.json').read_text())
        payload=[build/name for name in ('agent_server','identity.json','lib/libw55_support.so','lib/libw55_support_python.so')]
        checks=['binary_sha256','native_support_library_sha256','execution_library_sha256']
        for path,key in zip([payload[0],*payload[2:]],checks):
            if digest(path)!=identity[key]:raise ValueError(f'Signed-off {key} differs from build')
        if digest(WORK/'support.cc') != identity['execution_core_sha256']:raise ValueError('Compiled core changed')
        payload += [WORK/name for name in ('support.cc','support.h','build.log','validation.log','first-proposal.json')]
        payload += sorted((WORK/'validation').glob('*'))
        if not manifest.exists():
            members=[];prior=ROOT/'results/terrain_memory/simulation'/RUN/'w55/delivery'
            published={}
            for src in sorted(prior.glob('*/sources.json')):
                for item in json.loads(src.read_text()):
                    published[item['sha256']]=(src,item)
            reused=[];seen={}
            temp=archive.with_suffix('.partial')
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                for path in payload:
                    sha=digest(path)
                    if sha in published:
                        old,row=published[sha];reused.append(dict(source_path=str(path),sha256=sha,size=path.stat().st_size,
                          existing_catalog=str(old.relative_to(ROOT)),existing_asset=row['asset'],member=row.get('member')));continue
                    if sha in seen:
                        reused.append(dict(source_path=str(path),sha256=sha,size=path.stat().st_size,existing_asset=NAME,member=seen[sha]));continue
                    member=str(path.relative_to(ROOT/'work'));seen[sha]=member;z.write(path,member)
                    members.append(dict(source_path=str(path),member=member,asset=NAME,sha256=sha,size=path.stat().st_size))
            temp.replace(archive)
            write(catalog,dict(new_members=members,reused=reused,prior_material_inputs={
              'mapped_PD_saved_input':'w55/delivery/n-r56-build/sources.json (RR_calf_upper_counterexample-input.bin)',
              'original_XML_and_policy_inputs':'w55/delivery/n-a-r56-e30/sources.json (G1-project-go2.xml, call-000809-candidate-08.bin, original contexts)',
              'predictor_library_sha256':identity['mujoco_library_sha256']}))
            write(manifest,[dict(name=NAME,size=archive.stat().st_size,sha256=digest(archive),path=str(archive.relative_to(ROOT)),release_tag=TAG)])
        local=json.loads(manifest.read_text())[0]
        # Metadata only. The archive stays on D57.
        release=json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/releases/tags/{TAG}']))
        endpoint=f"repos/{REPO}/releases/{release['id']}/assets?per_page=100"
        def lookup():
            pages=json.loads(subprocess.check_output(['gh','api',endpoint,'--paginate','--slurp']))
            return next((a for page in pages for a in page if a['name']==NAME),None)
        asset=lookup()
        if asset is None:
            subprocess.run(['gh','release','upload',TAG,str(archive),'--repo',REPO],check=True)
            asset=lookup()
        if not asset or (asset['name'],asset['size'],asset['digest'],asset['state'])!=(NAME,local['size'],'sha256:'+local['sha256'],'uploaded'):
            raise RuntimeError('Remote asset size, digest or state differs')
        now=time.time()
        evidence=json.loads(catalog.read_text())
        row=dict(name=NAME,remote_digest=asset['digest'],size=asset['size'],remote_asset_id=asset['id'],remote_state=asset['state'],
                 url=asset['browser_download_url'],remote_verified_unix_s=now)
        write(DEST/'delivery.json',dict(status='published',assets=[row],original_files=len(evidence['new_members']),
              reused_files=len(evidence['reused']),source_commit=identity['source_commit'],checked_unix_s=now,
              deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False))
        commit('Verify W56 N/r59 build, dual-ABI libraries and saved native replay evidence',[catalog,manifest,DEST/'delivery.json'])
        print(json.dumps(row),flush=True)

if __name__=='__main__':main()
