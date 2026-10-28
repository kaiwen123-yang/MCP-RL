#!/usr/bin/env python3
"""Publish R10 directly over the execution host's AutoDL academic proxy."""
import argparse,base64,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO='kaiwen123-yang/MCP-RL';TAG='tm-s1-go2-soft-landing-r10'
SMALL=ROOT/'results/terrain_memory/simulation'/TAG


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','upload','finish']);a=p.parse_args()
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
        release=api(f'repos/{REPO}/releases/tags/{TAG}')
        idx['publication']=dict(status='draft_uploading',release_id=release['id'],release_url=release['html_url'],
            evidence_commit=head,transport='AutoDL /etc/network_turbo academic acceleration',
            upload_started_unix_s=time.time())
        (SMALL/'index.json').write_text(json.dumps(idx,indent=2)+'\n')
        print(json.dumps(idx['publication']),flush=True);return
    release=api(f'repos/{REPO}/releases/tags/{TAG}');actual=all_assets(release)
    if a.action=='upload':
        folder=OUT=ROOT/'outputs/terrain_memory/simulation'/TAG/'release-assets'
        folder.mkdir(exist_ok=True);todo=[]
        for item in idx['assets']:
            if item['size']>=2*1024**3:raise RuntimeError('Oversized asset: '+item['name'])
            if item['name'] in actual:
                prior=actual[item['name']]
                if prior['size']!=item['size'] or prior.get('digest')!='sha256:'+item['sha256']:
                    raise RuntimeError('Existing asset differs: '+item['name'])
                continue
            link=folder/item['name']
            if not link.exists():link.symlink_to(ROOT/item['path'])
            todo.append((link,item['size']))
        begin=time.time();done=0;size=0
        def upload(item):
            link,count=item;gh('release','upload',TAG,'--repo',REPO,str(link));return count
        with ThreadPoolExecutor(6) as pool:
            for future in as_completed([pool.submit(upload,item) for item in todo]):
                size+=future.result();done+=1
                if done%10==0 or done==len(todo):
                    print(json.dumps(dict(uploaded=done,total=len(todo),bytes=size,elapsed_s=time.time()-begin)),flush=True)
        return
    checks=[]
    for item in idx['assets']:
        asset=actual[item['name']];expected=f"https://github.com/{REPO}/releases/download/{TAG}/{item['name']}"
        good=(asset['state']=='uploaded' and asset['size']==item['size'] and
              asset.get('digest')=='sha256:'+item['sha256'] and asset['browser_download_url']==expected)
        if not good:raise RuntimeError('Final original-byte metadata mismatch: '+item['name'])
        checks.append(dict(name=asset['name'],url=asset['browser_download_url'],size=asset['size'],digest=asset['digest'],verified=True))
    payload=ROOT/'work/r10-release-publish.json'
    payload.write_text(json.dumps(dict(draft=False,body=(ROOT/'work/r10-release-notes.md').read_text())))
    gh('api','--method','PATCH',f"repos/{REPO}/releases/{release['id']}",'--input',str(payload))
    public=api(f'repos/{REPO}/releases/tags/{TAG}')
    if public['draft']:raise RuntimeError('Release is still a draft')
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
    git('add',str((SMALL/'index.json').relative_to(ROOT)))
    git('commit','-m','Verify published R10 asset URLs sizes and SHA256 digests')
    final=push()
    print(json.dumps(dict(final_commit=final,release=public['html_url'],tag_commit=tag['sha'],
        verified_assets=len(checks),asset_bytes=idx['publication']['asset_bytes'],
        max_asset_bytes=idx['publication']['maximum_asset_bytes'])),flush=True)


if __name__=='__main__':main()
