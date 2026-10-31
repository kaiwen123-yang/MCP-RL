#!/usr/bin/env python3
"""Publish retained R12 files directly from the execution host.

No physics, downloads, hashes or Git mutations. Use --server-auth for the
server gh login, or the existing credential JSON on stdin for manual publication.
"""
import argparse, fcntl, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-local-readiness-r12'
REPO='kaiwen123-yang/MCP-RL'
SMALL=ROOT/'results/terrain_memory/simulation'/RUN
LARGE=ROOT/'outputs/terrain_memory/simulation'/RUN
ENV=Path(sys.executable).parent

def server_credentials(deadline, release_tags, manifest=None, manifest_path=None, job_name=None, source_commit=None):
    """Read the server's existing GitHub login without writing its token to disk."""
    token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if not token:
        token=subprocess.check_output(['gh','auth','token','--hostname','github.com'],
            text=True,stderr=subprocess.PIPE,timeout=30).strip()
    env=dict(os.environ,GH_TOKEN=token,GODEBUG='http2client=0')
    def api(endpoint,*flags,missing_ok=False):
        remaining=deadline-time.time()
        if remaining<=0:raise TimeoutError('Publication deadline reached')
        try:
            output=subprocess.check_output(['gh','api',endpoint,*flags],
                env=env,text=True,stderr=subprocess.PIPE,timeout=min(120,remaining))
        except subprocess.CalledProcessError as error:
            if missing_ok and 'HTTP 404' in error.stderr:return None
            raise RuntimeError('GitHub API failed: '+error.stderr.replace(token,'[credential]').strip()) from None
        return json.loads(output)
    releases={};remote_assets={}
    def load_release(tag,create=False):
        if tag in releases:return
        release=api(f'repos/{REPO}/releases/tags/{tag}',missing_ok=create)
        if release is None:
            release=api(f'repos/{REPO}/releases','--method','POST',
                '-f',f'tag_name={tag}','-f',f'target_commitish={source_commit}',
                '-f',f'name={tag}','-f',f'body=Retained S1 outputs for {job_name}; source {source_commit}.',
                '-f','make_latest=false')
        pages=api(f'repos/{REPO}/releases/{release["id"]}/assets?per_page=100','--paginate','--slurp')
        releases[tag]=release
        remote_assets[tag]=[row for page in pages for row in page]
    for tag in sorted(release_tags):load_release(tag)
    if manifest is not None and job_name:
        # Saved destinations are immutable on retry. Reserve both remote assets
        # and this manifest's assigned names before placing any new files.
        occupied={tag:{row['name'] for row in rows} for tag,rows in remote_assets.items()}
        for row in manifest:
            if row.get('release_tag') and (ROOT/row['path']).stat().st_size:
                occupied.setdefault(row['release_tag'],set()).add(row['name'])
        part=1
        try:
            for row in manifest:
                if row.get('release_tag') or not (ROOT/row['path']).stat().st_size:continue
                while True:
                    tag=f'{RUN}-{job_name}-part{part:02d}'
                    load_release(tag,create=True)
                    names=occupied.setdefault(tag,{x['name'] for x in remote_assets[tag]})
                    if row['name'] in names or len(names)<900:
                        row['release_tag']=tag
                        names.add(row['name'])
                        break
                    part+=1
        finally:
            # Persist partial assignment even if a later Release cannot be
            # created; the next finish invocation resumes the same destinations.
            manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    return dict(token=token,releases=releases,remote_assets=remote_assets)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['upload','finish'])
    p.add_argument('--attempt',action='append',default=[])
    p.add_argument('--dependencies',action='store_true')
    p.add_argument('--asset',action='append',default=[])
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--server-auth',action='store_true')
    p.add_argument('--manifest',type=Path,help='Explicit job asset list: path,name,category.')
    p.add_argument('--job-name',help='Route unassigned job assets into Releases of at most900files.')
    p.add_argument('--source-commit',help='Actual job source commit for newly created Release tags.')
    args=p.parse_args()
    if args.job_name and not (args.server_auth and args.manifest and args.source_commit):
        p.error('--job-name requires --server-auth,--manifest and --source-commit')
    manifest=json.loads(args.manifest.read_text()) if args.manifest else None
    registered=json.loads((SMALL/'index.json').read_text()).get('publication',{}).get('assets',[])
    if manifest is not None:
        retained={row['name']:row.get('release_tag',RUN) for row in registered}
        for row in manifest:
            if row['name'] in retained:row.setdefault('release_tag',retained[row['name']])
    destination_rows=manifest if manifest is not None else registered
    if args.server_auth:
        tags={row.get('release_tag',RUN) for row in destination_rows}|{RUN}
        credentials=server_credentials(args.deadline_unix,tags,
            manifest=manifest if args.action=='upload' else None,manifest_path=args.manifest,
            job_name=args.job_name,source_commit=args.source_commit)
        releases=credentials['releases']
        remotes={tag:{row['name']:row for row in rows} for tag,rows in credentials['remote_assets'].items()}
    else:
        credentials=json.load(sys.stdin)
        releases={RUN:credentials['release_metadata']}
        remotes={RUN:{row['name']:row for row in credentials['remote_assets']}}
    destinations={row['name']:row.get('release_tag',RUN) for row in destination_rows}
    env=dict(os.environ,GODEBUG='http2client=0',GH_TOKEN=credentials['token'],PATH=str(ENV)+os.pathsep+os.environ['PATH'])
    proxy=env.get('https_proxy',env.get('HTTPS_PROXY',''))
    def update(change):
        with (SMALL/'index.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            value=json.loads((SMALL/'index.json').read_text());change(value)
            (SMALL/'index.json').write_text(json.dumps(value,indent=2)+'\n')
    release=releases[RUN]
    remote=remotes[RUN]
    if args.action=='upload' and not args.server_auth:
        for row in json.loads((SMALL/'index.json').read_text()).get('publication',{}).get('assets',[]):
            if row['status']=='uploaded' and row['name'] not in remote:
                remote[row['name']]=dict(row,state='uploaded')
    def metadata(j):
        q=j.setdefault('publication',{});q.update(status='publication_pending',release_id=release['id'],release_url=release['html_url'],release_tag=RUN)
        q.setdefault('release_destinations',{}).update({tag:dict(id=row['id'],url=row['html_url']) for tag,row in releases.items()})
    update(metadata)
    if args.action=='finish':
        def finish(j):
            q=j['publication']; missing=[]
            registered={row['path'] for row in q.get('assets',[])}
            selected={row['name'] for row in manifest} if manifest is not None else None
            if manifest is not None:
                names={row['name'] for row in q.get('assets',[])}
                missing.extend(row['path'] for row in manifest if row['name'] not in names)
            else:
                for folder in LARGE.iterdir():
                    if folder.is_dir() and folder.name!='release-upload-links':
                        missing.extend(str(file.relative_to(ROOT)) for file in folder.rglob('*')
                            if file.is_file() and str(file.relative_to(ROOT)) not in registered)
            for row in q.get('assets',[]):
                if selected is not None and row['name'] not in selected:continue
                actual=remotes[row.get('release_tag',RUN)].get(row['name'])
                if row.get('status')=='git_recorded':continue
                okay=actual and actual['state']=='uploaded' and actual['size']==row['size']
                row['status']='uploaded' if okay else 'publication_pending'
                if okay:row['url']=actual['browser_download_url']
                else:missing.append(row['path'])
            missing=sorted(set(missing+[row['path'] for row in q.get('assets',[]) if row['status'] not in ('uploaded','git_recorded')]))
            q.update(status='published' if not missing and not q.get('pending_jobs') else 'publication_pending',pending_paths=missing,
                pending_categories=['missing Release assets'] if missing else [],
                scope='Registered R12 assets; current job checked against its explicit manifest' if manifest is not None else 'All retained R12 files',
                asset_count=sum(x['status']=='uploaded' for x in q['assets']),
                asset_bytes=sum(x['size'] for x in q['assets'] if x['status']=='uploaded'),
                verification='One normal name/size/upload-state comparison; no full hash pass or downloads',checked_unix_s=time.time())
            print(json.dumps(dict(status=q['status'],asset_count=q['asset_count'],asset_bytes=q['asset_bytes'],pending_paths=missing)),flush=True)
        update(finish);return
    inputs=[]
    for attempt in args.attempt:
        folder=LARGE/attempt
        for file in sorted(folder.rglob('*')):
            if file.is_file():inputs.append((file,attempt+'__'+file.relative_to(folder).as_posix().replace('/','__'),'episode'))
    if args.dependencies:
        for file in sorted((ROOT/'work/assets/go2').rglob('*')):
            if file.is_file():inputs.append((file,'dependency__go2__'+file.relative_to(ROOT/'work/assets/go2').as_posix().replace('/','__'),'dependency'))
        for name in ['scene_flat.xml','rigid-T1R-resolved.json']:
            inputs.append((ROOT/'work/assets'/name,'dependency__'+name,'dependency'))
    if manifest is not None:
        inputs.extend((ROOT/row['path'],row['name'],row['category']) for row in manifest)
    expected=[]
    for file,name,category in inputs:
        size=file.stat().st_size
        if size>=2*1024**3:raise RuntimeError('Asset requires transfer splitting: '+str(file))
        tag=destinations.get(name,RUN)
        expected.append(dict(name=name,path=str(file.relative_to(ROOT)),size=size,category=category,release_tag=tag,
            url=f'https://github.com/{REPO}/releases/download/{tag}/{name}',status='publication_pending'))
    if args.asset:expected=[row for row in expected if row['name'] in args.asset]
    def register(j):
        q=j['publication'];rows={x['name']:x for x in q.get('assets',[])}
        for row in expected:
            old=rows.get(row['name']);actual=remotes[row['release_tag']].get(row['name'])
            if actual and actual['size']==row['size'] and actual['state']=='uploaded':row['status']='uploaded'
            elif actual:raise RuntimeError('Existing asset differs; retained without replacement: '+row['name'])
            rows[row['name']]=row
        q['assets']=list(rows.values())
    update(register)
    def upload(row):
        actual=remotes[row['release_tag']].get(row['name'])
        if actual:return
        path=ROOT/row['path']
        if row['size']==0:
            target=SMALL/'records'/(row['name']+'.txt' if path.suffix=='.log' else row['name']);target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path,target)
            def empty(j):
                for item in j['publication']['assets']:
                    if item['name']==row['name']:
                        item['status']='git_recorded';item['git_path']=str(target.relative_to(ROOT));item['url']=f'https://github.com/{REPO}/blob/main/{target.relative_to(ROOT)}'
            update(empty);return
        begin=time.time()
        # Actual gh and requests uploads failed with 408/TLS EOF on this proxy.
        # Native curl pins HTTP/1 and TLS1.2; credential header is stdin only.
        config='header = "Authorization: Bearer '+env['GH_TOKEN']+'"\nheader = "Accept: application/vnd.github+json"\nheader = "Content-Type: application/octet-stream"\n'
        destination=releases[row['release_tag']]
        url=destination['upload_url'].split('{')[0]+'?name='+row['name']
        response=subprocess.run(['curl','--config','-','--http1.1','--tls-max','1.2','--silent','--show-error','--fail-with-body',
            '--connect-timeout','30','--max-time',str(min(300,int(args.deadline_unix-time.time()))),
            '--request','POST','--data-binary','@'+str(path),url],input=config,cwd=ROOT,env=env,text=True,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if response.returncode:raise RuntimeError('curl upload failed: '+(response.stderr.replace(proxy,'[existing proxy]') if proxy else response.stderr)+' '+response.stdout[:250])
        actual=json.loads(response.stdout)
        if actual['size']!=row['size']:raise RuntimeError('Uploaded size differs: '+row['name'])
        def uploaded(j):
            for item in j['publication']['assets']:
                if item['name']==row['name']:item['status']='uploaded'
        update(uploaded)
        print(json.dumps(dict(name=row['name'],size=row['size'],status='uploaded',wall_s=round(time.time()-begin,2))),flush=True)
    for row in expected:
        if time.time()>=args.deadline_unix:break
        try:
            upload(row)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            message=str(error)
            def failed(j):
                for item in j['publication']['assets']:
                    if item['name']==row['name']:
                        item.update(status='publication_pending',upload_error=message)
            update(failed)
            print(json.dumps(dict(name=row['name'],status='publication_pending',error=message)),flush=True)
        if time.time()>=args.deadline_unix:break
if __name__=='__main__':main()

