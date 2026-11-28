"""Non-force GitHub publication from D57's Git object store, including compact binary evidence."""
import argparse,base64,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from publish_s1_w59_git import api,person


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def publish_commit(sha):
    raw=git('cat-file','commit',sha).decode()
    headers,message=raw.split('\n\n',1)
    parents=[];info=dict(message=message,parents=parents)
    for line in headers.splitlines():
        key,val=line.split(' ',1)
        if key=='parent':parents.append(val)
        elif key in ('author','committer'):info[key]=person(val)
        elif key=='tree':info[key]=val
        else:raise ValueError('Additional Git header '+key)
    def delta(par):
        return git('diff-tree','--no-commit-id','-r','--no-renames','--no-abbrev','--raw','-z',par,sha).split(b'\0')[:-1]
    base,changes=min(((p,delta(p)) for p in parents),key=lambda x:len(x[1]))
    if changes:
        entries=[]
        for meta,p in zip(changes[::2],changes[1::2],strict=True):
            old,mode,_,oid,status=meta[1:].decode().split()
            entry=dict(path=p.decode(),type='blob',mode=old if status=='D' else mode)
            if status=='D':entry['sha']=None
            else:
                data=git('cat-file','blob',oid)
                try:
                    decoded=data.decode('utf-8')
                    if '\0' in decoded:raise ValueError()
                    entry['content']=decoded
                except (UnicodeDecodeError,ValueError):
                    actual=api('/git/blobs','.sha',dict(content=base64.b64encode(data).decode(),encoding='base64'))
                    if actual!=oid:raise ValueError('Git blob SHA differs for '+entry['path'])
                    entry['sha']=actual
            entries.append(entry)
        tree=api('/git/trees','.sha',dict(base_tree=git('rev-parse',base+'^{tree}').decode().strip(),tree=entries))
        if tree!=info['tree']:raise ValueError('Published tree SHA differs for '+sha)
    if api('/git/commits','.sha',info)!=sha:raise ValueError('Published commit SHA differs for '+sha)


def publish(target):
    head=git('rev-parse',target+'^{commit}').decode().strip()
    old=api('/git/ref/heads/main','.object.sha')
    if subprocess.call(['git','merge-base','--is-ancestor',old,head],cwd=ROOT)!=0:
        raise ValueError('Read and integrate the current remote parent before publishing')
    for h in git('rev-list','--reverse','--topo-order',old+'..'+head).decode().splitlines():
        publish_commit(h)
        print('verified commit',h,flush=True)
    current=api('/git/ref/heads/main','.object.sha')
    if current!=head:
        subprocess.run(['git','merge-base','--is-ancestor',current,head],cwd=ROOT,check=True)
        if api('/git/refs/heads/main','.object.sha',dict(sha=head,force=False),'PATCH')!=head:
            raise ValueError('Unexpected GitHub ref response')
    print('remote main',api('/git/ref/heads/main','.object.sha'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('commit')
    publish(parser.parse_args().commit)
