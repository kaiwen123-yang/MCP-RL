#!/usr/bin/env python3
"""Publish R11 directly from the execution host; verify metadata without downloading assets."""
import argparse,base64,hashlib,json,os,re,subprocess,sys,time,threading
import requests
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from s1_go2_r11_protocol import ROOT,CONTRACT,RUN
from s1_go2_r11_runtime import index_file
from package_s1_go2_r11 import build_memory_archive
REPO='kaiwen123-yang/MCP-RL'
SMALL=ROOT/CONTRACT['outputs']['compact_root']
LINKS=ROOT/CONTRACT['outputs']['large_root']/'release-upload-links'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['draft','upload','status','finish'])
    args=parser.parse_args()
    credentials=json.loads(sys.stdin.read())
    env=dict(os.environ,GH_TOKEN=credentials['token'],GITHUB_TOKEN=credentials['token'],
        GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='2',
        GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
        GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(
            ('x-access-token:'+credentials['token']).encode()).decode(),
        GIT_CONFIG_KEY_1='http.proxy',GIT_CONFIG_VALUE_1=os.environ.get('https_proxy',''))
    def command(argv):
        p=subprocess.run(argv,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if p.returncode:
            text=p.stderr.lower()
            category='network' if any(s in text for s in ['connect','timeout','timed out','tls','ssl','resolve']) else 'command'
            raise RuntimeError(json.dumps(dict(program=argv[0],exit_code=p.returncode,category=category,
                http_codes=re.findall(r'http[ /](\d{3})',text),network_detail=next((x for x in ['TLS handshake timeout','context deadline exceeded','connection reset by peer','EOF','i/o timeout','no such host'] if x.lower() in text),None))))
        return p.stdout.strip()
    def gh(*argv):return command(['gh',*argv])
    def git(*argv):return command(['git',*argv])
    def api(endpoint,*argv):return json.loads(gh('api',endpoint,*argv))
    def state():
        with index_file(SMALL,CONTRACT) as j:return json.loads(json.dumps(j))
    def release():
        j=state()
        if j.get('publication',{}).get('release_id'):
            return api(f"repos/{REPO}/releases/{j['publication']['release_id']}")
        rows=api(f'repos/{REPO}/releases?per_page=100')
        return next((r for r in rows if r['tag_name']==RUN),None)
    def assets(r):
        result={}
        page=1
        while True:
            rows=api(f"repos/{REPO}/releases/{r['id']}/assets?per_page=100&page={page}")
            result.update({r['name']:r for r in rows})
            if len(rows)<100:return result
            page+=1
    if args.action=='draft':
        r=release()
        if r is None:
            note=ROOT/'work/r11-release-notes.md'
            if not note.exists():note.write_text('R11 campaign is in progress. This draft contains retained development evidence; no S5 freeze or learned-policy result is implied.\n')
            gh('release','create',RUN,'--repo',REPO,'--draft','--target',git('rev-parse','HEAD'),
                '--title','S1 R11: staged material continuation','--notes-file',str(note))
            r=release()
        with index_file(SMALL,CONTRACT) as j:
            j['publication']=dict(status='draft_uploading',release_id=r['id'],release_url=r['html_url'],
                evidence_commit=git('rev-parse','HEAD'),upload_started_unix_s=time.time(),
                transport=os.environ.get('R11_PUBLICATION_TRANSPORT','AutoDL academic acceleration'))
        print(json.dumps(dict(release_id=r['id'],draft=r['draft'])),flush=True)
        return
    r=release()
    if r is None:raise RuntimeError('No R11 draft exists')
    remote=assets(r);expected=state().get('assets',[])
    def matches(item,actual):
        return actual.get('state')=='uploaded' and actual['size']==item['size'] and actual.get('digest')=='sha256:'+item['sha256']
    if args.action=='status':
        verified=[i for i in expected if i['name'] in remote and matches(i,remote[i['name']])]
        print(json.dumps(dict(verified=len(verified),total=len(expected),bytes=sum(i['size'] for i in verified))),flush=True)
        return
    if args.action=='upload':
        LINKS.mkdir(parents=True,exist_ok=True)
        todo=[]
        for item in expected:
            if item['size']>=2*1024**3:raise RuntimeError('Asset exceeds release limit: '+item['name'])
            prior=remote.get(item['name'])
            if prior and matches(item,prior):continue
            if prior:
                if r['draft'] and prior['state']=='starter':
                    gh('api',f"repos/{REPO}/releases/assets/{prior['id']}",'--method','DELETE')
                else:raise RuntimeError('Existing published bytes differ: '+item['name'])
            if item.get('archive_mode')!='memory_tar_gzip_v1':
                link=LINKS/item['name']
                if not link.exists():link.symlink_to(ROOT/item['path'])
                item['_upload_path']=str(link)
            todo.append(item)
        # Old gh upload produced repeated EOF/TLS errors through the authorized tunnel.
        # Authenticated metadata and Git remain on gh; original file bytes stream via HTTP/1.
        connections=threading.local()
        upload_url=r['upload_url'].split('{')[0]
        def upload(item):
            begin=time.time();name=item['name']
            if not hasattr(connections,'session'):
                session=requests.Session();session.trust_env=False
                session.proxies={'http':env['http_proxy'],'https':env['https_proxy']}
                session.headers.update(Authorization='Bearer '+credentials['token'],
                    Accept='application/vnd.github+json',**{'X-GitHub-Api-Version':'2022-11-28',
                    'User-Agent':'MCP-RL-R11-publication'})
                connections.session=session
            try:
                if item.get('archive_mode')=='memory_tar_gzip_v1':
                    base=ROOT/item['source_directory']
                    data=build_memory_archive(base,[base/member['path'] for member in item['members']],
                        deduplicate=item.get('deduplicate',False))
                    if data.getbuffer().nbytes!=item['size'] or hashlib.sha256(data.getbuffer()).hexdigest()!=item['sha256']:
                        raise RuntimeError('Completed source bytes or archive runtime changed: '+name)
                else:
                    data=Path(item['_upload_path']).open('rb')
                with data:
                    response=connections.session.post(upload_url,params={'name':name},
                        data=data,headers={'Content-Type':'application/octet-stream'},timeout=(45,900))
                response.raise_for_status()
                actual=response.json()
                return dict(name=name,uploaded=True,wall_s=time.time()-begin,
                    size=actual['size'],digest=actual.get('digest'))
            except requests.RequestException as error:
                connections.session.close();del connections.session
                return dict(name=name,uploaded=False,wall_s=time.time()-begin,
                    error=type(error).__name__,network_detail=next((label for token,label in [('CERTIFICATE_VERIFY_FAILED','certificate_verify_failed'),('UNEXPECTED_EOF','unexpected_tls_eof'),('EOF occurred','unexpected_tls_eof'),('WRONG_VERSION_NUMBER','wrong_tls_version'),('timed out','timeout'),('Connection refused','connection_refused')] if token in str(error)),None),http_status=error.response.status_code if error.response is not None else None)
        failed=0;completed=0;iterator=iter(todo)
        with ThreadPoolExecutor(8) as pool:
            pending={pool.submit(upload,path) for path in [next(iterator,None) for _ in range(min(8,len(todo)))]}
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    record=future.result();completed+=1;failed+=not record['uploaded']
                    print(json.dumps(record),flush=True)
                    if failed<4:
                        path=next(iterator,None)
                        if path is not None:pending.add(pool.submit(upload,path))
        with index_file(SMALL,CONTRACT) as j:
            j['publication']['last_upload_batch']=dict(requested=len(todo),completed=completed,failed=failed,
                transport=os.environ.get('R11_PUBLICATION_TRANSPORT','AutoDL academic acceleration'),time_unix_s=time.time())
        if failed:raise SystemExit(2)
        return
    checked=[]
    for item in expected:
        actual=remote.get(item['name'])
        if not actual or not matches(item,actual):raise RuntimeError('Unverified asset: '+item['name'])
        checked.append(dict(name=item['name'],size=actual['size'],digest=actual['digest'],
            url=f'https://github.com/{REPO}/releases/download/{RUN}/{item["name"]}'))
    git('fetch','origin','main')
    git('merge-base','--is-ancestor','origin/main','HEAD')
    git('push','origin','HEAD:main')
    head=git('rev-parse','HEAD')
    if api(f'repos/{REPO}/git/ref/heads/main')['object']['sha']!=head:raise RuntimeError('Main commit mismatch')
    payload=ROOT/'work/r11-publish-payload.json'
    payload.write_text(json.dumps(dict(draft=False,target_commitish=head,
        body=(ROOT/'work/r11-release-notes.md').read_text())))
    gh('api',f"repos/{REPO}/releases/{r['id']}",'--method','PATCH','--input',str(payload))
    public=api(f'repos/{REPO}/releases/tags/{RUN}')
    actual_assets=assets(public)
    for item in checked:
        actual=actual_assets[item['name']]
        if actual['browser_download_url']!=item['url'] or actual['size']!=item['size'] or actual.get('digest')!=item['digest']:
            raise RuntimeError('Published asset metadata mismatch: '+item['name'])
    tag=api(f'repos/{REPO}/git/ref/tags/{RUN}')['object']
    while tag['type']=='tag':tag=api(f"repos/{REPO}/git/tags/{tag['sha']}")['object']
    if tag['sha']!=head:raise RuntimeError('Release tag does not name the final evidence commit')
    with index_file(SMALL,CONTRACT) as j:
        j['publication'].update(status='published_verified',evidence_commit=head,published_tag_commit=tag['sha'],
            release_url=public['html_url'],all_assets_verified=True,assets=checked,asset_count=len(checked),
            asset_bytes=sum(x['size'] for x in checked),verified_unix_s=time.time(),
            verification='Authenticated URL, size and SHA256 metadata against original execution-host files; no asset downloads')
    git('add',str((SMALL/'index.json').relative_to(ROOT)))
    git('commit','-m','Verify R11 release asset URLs sizes and digests')
    git('push','origin','HEAD:main')
    final=git('rev-parse','HEAD')
    if api(f'repos/{REPO}/git/ref/heads/main')['object']['sha']!=final:raise RuntimeError('Final metadata commit mismatch')
    print(json.dumps(dict(final_commit=final,evidence_commit=head,release_url=public['html_url'],verified_assets=len(checked))),flush=True)
if __name__=='__main__':main()
