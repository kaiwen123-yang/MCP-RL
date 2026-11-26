"""Fast-forward W59 compact Git history directly from D57 using the GitHub Git database API."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REPO='repos/kaiwen123-yang/MCP-RL'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def api(endpoint,field,payload=None,method='POST'):
    env=os.environ.copy()
    # api.github.com is reachable directly from D57 when its git HTTPS route returns 503.
    for name in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
        env.pop(name,None)
    cmd=['gh','api',REPO+endpoint,'--jq',field,'-H','Cache-Control: no-cache']
    data=None
    if payload is not None:
        cmd+=['--method',method,'--input','-']
        data=json.dumps(payload,ensure_ascii=False).encode()
    return subprocess.check_output(cmd,input=data,cwd=ROOT,env=env).decode().strip()


def person(value):
    name,email,epoch,sign,hr,minute=re.fullmatch(r'(.*) <([^>]*)> (\d+) ([+-])(\d\d)(\d\d)',value).groups()
    delta=timedelta(hours=int(hr),minutes=int(minute))
    zone=timezone(delta if sign=='+' else -delta)
    return dict(name=name,email=email,date=datetime.fromtimestamp(int(epoch),zone).isoformat())


def publish_commit(sha):
    headers,message=git('cat-file','commit',sha).decode().split('\n\n',1)
    commit=dict(message=message,parents=[])
    for line in headers.splitlines():
        field,value=line.split(' ',1)
        if field=='parent':
            commit['parents'].append(value)
        elif field in ('author','committer'):
            commit[field]=person(value)
        elif field=='tree':
            commit[field]=value
        else:
            raise ValueError(f'Commit {sha} has an additional header; use native Git')
    parent=commit['parents'][0]
    base=git('rev-parse',parent+'^{tree}').decode().strip()
    if base!=commit['tree']:
        records=git('diff-tree','-r','--no-renames','--no-abbrev','--no-commit-id','--raw','-z',parent,sha).split(b'\0')[:-1]
        entries=[]
        for metadata,path in zip(records[::2],records[1::2],strict=True):
            old_mode,mode,_,blob,status=metadata[1:].decode().split()
            entry=dict(path=path.decode(),type='blob',mode=old_mode if status=='D' else mode)
            if status=='D':
                entry['sha']=None
            else:
                # Only compact, already committed UTF-8 sources/results take this path.
                entry['content']=git('cat-file','blob',blob).decode()
            entries.append(entry)
        actual=api('/git/trees','.sha',dict(base_tree=base,tree=entries))
        if actual!=commit['tree']:
            raise ValueError(f'Tree differs for local commit {sha}')
    if api('/git/commits','.sha',commit)!=sha:
        raise ValueError(f'Commit differs from local object {sha}')


def publish(target):
    target=git('rev-parse',target+'^{commit}').decode().strip()
    original=api('/git/ref/heads/main','.object.sha')
    if original!=target:
        subprocess.run(['git','merge-base','--is-ancestor',original,target],cwd=ROOT,check=True)
        commits=git('rev-list','--reverse','--topo-order',original+'..'+target).decode().splitlines()
        for sha in commits:
            publish_commit(sha)
        current=api('/git/ref/heads/main','.object.sha')
        if current!=target:
            subprocess.run(['git','merge-base','--is-ancestor',current,target],cwd=ROOT,check=True)
            if api('/git/refs/heads/main','.object.sha',dict(sha=target,force=False),'PATCH')!=target:
                raise ValueError('GitHub main did not reach the verified tree')
    tip=api('/git/ref/heads/main','.object.sha')
    if tip!=target:
        subprocess.run(['git','merge-base','--is-ancestor',target,tip],cwd=ROOT,check=True)
    track=git('rev-parse','refs/remotes/origin/main').decode().strip()
    if track!=tip and subprocess.run(['git','merge-base','--is-ancestor',track,tip],cwd=ROOT).returncode==0:
        git('update-ref','refs/remotes/origin/main',tip,track)
    return target,tip


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('commit',nargs='?',default='HEAD')
    print(*publish(parser.parse_args().commit),flush=True)
