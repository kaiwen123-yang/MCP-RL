#!/usr/bin/env python3
"""Publish R8 originals directly from the execution host and verify metadata."""
import argparse,base64,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from run_s1_go2_wb_baseline import ROOT,index_file
CONFIG='experiments/terrain_memory/s1-go2-wb-support.json'
REPO='kaiwen123-yang/MCP-RL'

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','upload','finalize']);p.add_argument('--attempt')
    p.add_argument('--notes-file');args=p.parse_args()
    c=json.loads((ROOT/CONFIG).read_text());small=ROOT/c['outputs']['root'];tag=c['run_id']
    token=sys.stdin.read().strip();env=dict(os.environ,GH_TOKEN=token,GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_COUNT='1',
        GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
        GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+token).encode()).decode())
    gh_path=ROOT.parent/'terrain-memory-s0/work/tools/gh_2.103.2_linux_amd64/bin/gh'
    def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,env=env,text=True).strip()
    def gh(*a):return subprocess.check_output([str(gh_path),*a],cwd=ROOT,env=env,text=True).strip()
    if args.action=='prepare':
        commit=git('rev-parse','HEAD');git('push','origin','HEAD:main')
        url=gh('release','create',tag,'--repo',REPO,'--target',commit,'--draft',
            '--title','S1 R8: Go2 held-control prediction and continuous support',
            '--notes','Execution is in progress. Frozen author long missions and project development are reported separately. Final outcomes and complete asset verification follow this bounded campaign.')
        with index_file(small,c) as idx:idx['publication']=dict(status='draft_in_progress',release_url=url,evidence_commit=commit)
        print(url,flush=True);return
    idx=json.loads((small/'index.json').read_text())
    release=json.loads(gh('api',f'repos/{REPO}/releases/tags/{tag}'))
    pages=json.loads(gh('api','--paginate','--slurp',f"repos/{REPO}/releases/{release['id']}/assets?per_page=100"))
    actual={a['name']:a for page in pages for a in page}
    if args.action=='upload':
        assets=[a for a in idx['assets'] if not args.attempt or a['name'].startswith(args.attempt+'-')]
        folder=ROOT/'outputs/terrain_memory/simulation'/tag/'release-assets';folder.mkdir(exist_ok=True)
        todo=[]
        for a in assets:
            if a['size']>=c['outputs']['max_release_asset_bytes_exclusive']:raise ValueError('Asset exceeds declared Release limit: '+a['name'])
            if a['name'] in actual:
                b=actual[a['name']]
                if b['size']!=a['size'] or b['digest']!='sha256:'+a['sha256']:raise ValueError('Published bytes differ: '+a['name'])
                continue
            link=folder/a['name']
            if not link.exists():link.symlink_to(Path(a['path']))
            todo.append(link)
        def upload(path):
            gh('release','upload',tag,'--repo',REPO,str(path))
            print(json.dumps(dict(uploaded=path.name,bytes=path.stat().st_size)),flush=True)
        with ThreadPoolExecutor(2) as pool:list(pool.map(upload,todo))
        return
    expected=idx['assets']
    checks=[]
    for a in expected:
        b=actual[a['name']]
        if b['state']!='uploaded' or b['size']!=a['size'] or b['digest']!='sha256:'+a['sha256']:
            raise ValueError('Release does not match original execution-host bytes: '+a['name'])
    commit=git('rev-parse','HEAD');git('push','origin','HEAD:main')
    gh('release','edit',tag,'--repo',REPO,'--target',commit,'--notes-file',args.notes_file,'--draft=false')
    # Draft URLs can contain an untagged identifier. Fetch after publication.
    release=json.loads(gh('api',f'repos/{REPO}/releases/tags/{tag}'))
    pages=json.loads(gh('api','--paginate','--slurp',f"repos/{REPO}/releases/{release['id']}/assets?per_page=100"))
    final={a['name']:a for page in pages for a in page}
    for a in expected:
        b=final[a['name']]
        checks.append(dict(name=b['name'],url=b['browser_download_url'],size=b['size'],digest=b['digest'],
            verified=b['state']=='uploaded' and b['size']==a['size'] and b['digest']=='sha256:'+a['sha256']))
    if not all(a['verified'] for a in checks):raise ValueError('Final published metadata differs')
    with index_file(small,c) as idx:
        idx['publication'].update(status='published_verified',release_url=release['html_url'],release_id=release['id'],
            evidence_commit=commit,draft=release['draft'],asset_count=len(checks),asset_bytes=sum(a['size'] for a in checks),
            all_assets_verified=True,asset_verification=checks,verified_utc=datetime.now(timezone.utc).isoformat(),
            method='Original execution-host files; authenticated direct upload; final URL/size/digest verified without downloading assets.')
    git('add',str((small/'index.json').relative_to(ROOT)));git('commit','-m','Verify final R8 Release URLs sizes and digests')
    git('push','origin','HEAD:main');print(json.dumps(dict(commit=git('rev-parse','HEAD'),release=release['html_url'],assets=len(checks))),flush=True)
if __name__=='__main__':main()
