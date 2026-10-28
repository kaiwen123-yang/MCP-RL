#!/usr/bin/env python3
"""Publish R10 directly over the execution host's AutoDL academic proxy."""
import argparse,base64,json,os,subprocess,sys,time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[1]
REPO='kaiwen123-yang/MCP-RL';TAG='tm-s1-go2-soft-landing-r10'
SMALL=ROOT/'results/terrain_memory/simulation'/TAG


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','record-draft','status','upload','finish']);a=p.parse_args()
    credentials=json.loads(sys.stdin.read());token=credentials['token']
    env=dict(os.environ,GH_TOKEN=token,GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='1',
        GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
        GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
        GIT_AUTHOR_NAME=credentials['name'],GIT_AUTHOR_EMAIL=credentials['email'],
        GIT_COMMITTER_NAME=credentials['name'],GIT_COMMITTER_EMAIL=credentials['email'])
    def command(program,*args):
        return subprocess.check_output([program,*args],env=env,cwd=ROOT,text=True).strip()
    def gh(*args):return command('/usr/bin/gh',*args)
    def git(*args):return command('git',*args)
    def api(endpoint):return json.loads(gh('api',endpoint))
    def tagged_release():
        return next(row for row in api(f'repos/{REPO}/releases?per_page=100') if row['tag_name']==TAG)
    def push():
        git('fetch','origin','main');git('merge-base','--is-ancestor','origin/main','HEAD')
        git('push','origin','HEAD:main');head=git('rev-parse','HEAD')
        actual=api(f'repos/{REPO}/git/ref/heads/main')['object']['sha']
        if actual!=head:raise RuntimeError('Published main does not identify the submitted commit')
        return head
    def all_assets(release):
        result=[];page=1
        while True:
            rows=api(f"repos/{REPO}/releases/{release['id']}/assets?per_page=100&page={page}")
            result.extend(rows)
            if len(rows)<100:return {row['name']:row for row in result}
            page+=1
    idx=json.loads((SMALL/'index.json').read_text())
    if a.action=='prepare':
        paths=['AGENTS.md','README.md','docs/TERRAIN_MEMORY_PLAN.md','experiments/terrain_memory/s1-go2-soft-landing-r10.json',
            'scripts/finalize_s1_go2_r10.py','scripts/publish_s1_go2_r10.py']
        paths += [str(f.relative_to(ROOT)) for f in (ROOT/'experiments/terrain_memory').glob('s1-go2-r10-*.patch')]
        paths += [str(f.relative_to(ROOT)) for f in SMALL.iterdir() if f.suffix in ('.json','.csv','.png','.svg')]
        git('add','--',*paths);git('commit','-m','Publish bounded R10 material execution results and all unexecuted cases')
        head=push()
        gh('release','create',TAG,'--repo',REPO,'--target',head,'--draft',
            '--title','S1 R10: material execution prerequisite blocked after bounded development',
            '--notes-file',str(ROOT/'work/r10-release-notes.md'))
        release=tagged_release()
        idx['publication']=dict(status='draft_uploading',release_id=release['id'],release_url=release['html_url'],
            evidence_commit=head,transport='AutoDL /etc/network_turbo academic acceleration',
            upload_started_unix_s=time.time())
        (SMALL/'index.json').write_text(json.dumps(idx,indent=2)+'\n')
        print(json.dumps(idx['publication']),flush=True);return
    release=tagged_release() if a.action=='record-draft' else api(f"repos/{REPO}/releases/{idx['publication']['release_id']}")
    if a.action=='record-draft':
        idx['publication']=dict(status='draft_uploading',release_id=release['id'],release_url=release['html_url'],
            evidence_commit=git('rev-parse','HEAD'),transport='AutoDL /etc/network_turbo academic acceleration',
            upload_started_unix_s=time.time())
        (SMALL/'index.json').write_text(json.dumps(idx,indent=2)+'\n')
        print(json.dumps(idx['publication']),flush=True);return
    actual=all_assets(release)
    if a.action=='status':
        verified=[item for item in idx['assets'] if item['name'] in actual and
            actual[item['name']]['state']=='uploaded' and actual[item['name']]['size']==item['size'] and
            actual[item['name']].get('digest')=='sha256:'+item['sha256']]
        print(json.dumps(dict(verified_assets=len(verified),total_assets=len(idx['assets']),
            verified_bytes=sum(x['size'] for x in verified),remaining_bytes=sum(x['size'] for x in idx['assets'])-sum(x['size'] for x in verified))),flush=True)
        return
    if a.action=='upload':
        idx['publication']['transport_history']=['AutoDL academic acceleration: intermittent TLS handshake failures',
            'User authorized local VPN over SSH reverse tunnel for continuation']
        idx['publication']['transport']=env['R10_PUBLICATION_TRANSPORT']
        (SMALL/'index.json').write_text(json.dumps(idx,indent=2)+'\n')
        notes=ROOT/'work/r10-release-notes.md'
        notes.write_text(notes.read_text().replace('Uploads use AutoDL academic acceleration directly from the execution host.',
            'Uploads started with AutoDL academic acceleration; after repeated TLS handshake failures, the user authorized continuation through a local VPN SSH tunnel. Files are read on the execution host and are not archived on the user computer.'))
        folder=OUT=ROOT/'outputs/terrain_memory/simulation'/TAG/'release-assets'
        folder.mkdir(exist_ok=True);todo=[]
        for item in idx['assets']:
            if item['size']>=2*1024**3:raise RuntimeError('Oversized asset: '+item['name'])
            if item['name'] in actual:
                prior=actual[item['name']]
                if prior['state']=='starter' and prior['size']==item['size'] and prior.get('digest') is None and release['draft']:
                    gh('api','--method','DELETE',f"repos/{REPO}/releases/assets/{prior['id']}")
                    print(json.dumps(dict(replacing_interrupted_draft_upload=item['name'])),flush=True)
                elif prior['state']=='uploaded' and prior['size']==item['size'] and prior.get('digest')=='sha256:'+item['sha256']:
                    continue
                else:
                    raise RuntimeError('Existing asset differs: '+item['name'])
            link=folder/item['name']
            if not link.exists():link.symlink_to(ROOT/item['path'])
            todo.append((link,item['size']))
        begin=time.time();done=0;size=0
        print(json.dumps(dict(already_verified_assets=len(idx['assets'])-len(todo),
            already_verified_bytes=sum(item['size'] for item in idx['assets'])-sum(item[1] for item in todo),
            remaining_assets=len(todo),remaining_bytes=sum(item[1] for item in todo))),flush=True)
        upload_url=release['upload_url'].split('{')[0]
        connections=threading.local()
        def upload(item):
            link,count=item
            if not hasattr(connections,'session'):
                session=requests.Session();session.trust_env=False
                session.proxies={'http':env['http_proxy'],'https':env['https_proxy']}
                session.headers.update(Authorization='Bearer '+token,Accept='application/vnd.github+json',
                    **{'X-GitHub-Api-Version':'2022-11-28','User-Agent':'MCP-RL-R10-publication'})
                connections.session=session
            try:
                with link.open('rb') as data:
                    response=connections.session.post(upload_url,params={'name':link.name},data=data,
                        headers={'Content-Type':'application/octet-stream'},timeout=(60,600))
            except (requests.ConnectionError,requests.Timeout) as error:
                connections.session.close();del connections.session
                print(json.dumps(dict(interrupted_file=link.name,transport_error=type(error).__name__)),flush=True)
                return None
            response.raise_for_status()
            return count
        iterator=iter(todo)
        interrupted=0
        with ThreadPoolExecutor(3) as pool:
            pending={pool.submit(upload,item) for item in list(todo)[:3]}
            iterator=iter(todo[3:])
            while pending:
                finished,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in finished:
                    transferred=future.result()
                    if transferred is None:interrupted+=1
                    else:size+=transferred;done+=1
                    if done%10==0 or done==len(todo):
                        print(json.dumps(dict(uploaded=done,total=len(todo),bytes=size,elapsed_s=time.time()-begin)),flush=True)
                    item=next(iterator,None)
                    if item is not None:pending.add(pool.submit(upload,item))
        print(json.dumps(dict(successful_uploads=done,interrupted_files=interrupted,bytes=size)),flush=True)
        if interrupted:raise SystemExit(2)
        return
    checks=[]
    for item in idx['assets']:
        asset=actual[item['name']];expected=f"https://github.com/{REPO}/releases/download/{TAG}/{item['name']}"
        good=(asset['state']=='uploaded' and asset['size']==item['size'] and
              asset.get('digest')=='sha256:'+item['sha256'])
        if not good:raise RuntimeError('Final original-byte metadata mismatch: '+item['name'])
        checks.append(dict(name=asset['name'],url=asset['browser_download_url'],size=asset['size'],digest=asset['digest'],verified=True))
    payload=ROOT/'work/r10-release-publish.json'
    payload.write_text(json.dumps(dict(draft=False,body=(ROOT/'work/r10-release-notes.md').read_text())))
    gh('api','--method','PATCH',f"repos/{REPO}/releases/{release['id']}",'--input',str(payload))
    public=api(f'repos/{REPO}/releases/tags/{TAG}')
    if public['draft']:raise RuntimeError('Release is still a draft')
    public_assets=all_assets(public)
    for check in checks:
        asset=public_assets[check['name']]
        expected=f"https://github.com/{REPO}/releases/download/{TAG}/{check['name']}"
        if asset['browser_download_url']!=expected or asset['size']!=check['size'] or asset.get('digest')!=check['digest']:
            raise RuntimeError('Published URL/size/digest mismatch: '+check['name'])
        check['url']=asset['browser_download_url']
    tag=api(f'repos/{REPO}/git/ref/tags/{TAG}')['object']
    while tag['type']=='tag':tag=api(f"repos/{REPO}/git/tags/{tag['sha']}")['object']
    idx['status']='blocked_material_execution_published_verified'
    idx['publication'].update(status='published_verified',release_url=public['html_url'],
        published_tag_commit=tag['sha'],all_assets_verified=True,asset_count=len(checks),
        asset_bytes=sum(x['size'] for x in checks),maximum_asset_bytes=max(x['size'] for x in checks),
        assets=checks,verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        elapsed_since_upload_start_s=time.time()-idx['publication']['upload_started_unix_s'],
        verification='Authenticated GitHub URL/size/digest against original execution-host SHA256; no asset downloads')
    (SMALL/'index.json').write_text(json.dumps(idx,indent=2)+'\n')
    git('add',str((SMALL/'index.json').relative_to(ROOT)),'scripts/publish_s1_go2_r10.py',
        'scripts/finalize_s1_go2_r10.py',*[str(f.relative_to(ROOT)) for f in SMALL.glob('*-cap-decisions.json')])
    git('commit','-m','Verify published R10 asset URLs sizes and SHA256 digests')
    final=push()
    print(json.dumps(dict(final_commit=final,release=public['html_url'],tag_commit=tag['sha'],
        verified_assets=len(checks),asset_bytes=idx['publication']['asset_bytes'],
        max_asset_bytes=idx['publication']['maximum_asset_bytes'])),flush=True)


if __name__=='__main__':main()
