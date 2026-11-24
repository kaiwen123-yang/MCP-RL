"""Receive H media proofs and expose only this frozen process-file cohort."""
import gzip,json,os,sqlite3,sys
from pathlib import Path
from inventory import CTL,now,save,BG
from packet import affinity,status,report,waves,db_open
from tmux_archive import stream,store_retire
def aliases():
    from policy import RAW,COMPACT
    a={}
    for f in (COMPACT/'attempts').glob('*.json'):
        p=RAW/f.stem
        if p.is_symlink():a[str(p)]=str(p.resolve())
    with db_open() as db:
        for path,target in db.execute("select path,link from files where kind='link'"):
            if path.endswith('/outputs') and os.path.realpath(path)==str(RAW.parents[2]):a[path]=str(RAW.parents[2])
    return a
def meta():
    names=('roots.json','inventory.done.json','cohort-summary.json','catalog.done.json','hash-progress.json','tmux-archive.json','real-throughput.json','archive-complete-proof.json','postcheck.json')
    return {n:json.loads((CTL/n).read_text()) for n in names if (CTL/n).is_file()}
def main():
    affinity();op=sys.argv[1]
    if op=='status':report(status())
    elif op=='meta':report(meta())
    elif op=='aliases':report(aliases())
    elif op=='waves':waves(json.load(sys.stdin))
    elif op=='pack':stream(json.load(sys.stdin))
    elif op in ('policy','exclusions'):
        from shutil import copyfileobj
        with (CTL/('policy.jsonl.gz' if op=='policy' else 'retained-process-scope.tsv.gz')).open('rb') as f:copyfileobj(f,sys.stdout.buffer,1024*1024)
    elif op=='verified':
        from policy import Policy
        q=json.load(sys.stdin);complete=json.loads((CTL/'catalog.done.json').read_text())
        if q['volume_serial']!='6063F4DA' or not q['all_objects_verified'] or q['files']!=complete['selected'] or q['logical_bytes']!=complete['selected_bytes'] or complete['scoped_hash_failures']:
            raise ValueError('Complete snapshot does not match the H proof')
        with db_open() as db:Policy(db,fresh=True)
        save('archive-complete-proof.json',{**q,'at_D57':now(),**BG});report({'accepted':True})
    elif op=='retire':
        if not (CTL/'archive-complete-proof.json').is_file():raise RuntimeError('Waiting for the complete H archive receipt')
        report(store_retire(json.load(sys.stdin)))
    elif op=='source-reasons':
        from policy import Policy
        req=json.load(sys.stdin);out=[]
        with db_open() as db:
            policy=Policy(db)
            for i in req['ids']:
                row=db.execute('select path,root,kind,tracked,nlink,opened,state from files where id=? and id in(select id from cohort)',(i,)).fetchone()
                if row:out.append((i,policy.reason(*row)))
        report(out)
    elif op=='eligible-again':
        from policy import Policy
        ids=json.load(sys.stdin)['ids'];out=[]
        with db_open() as db:
            policy=Policy(db)
            for i in ids:
                v=db.execute('select path,root,kind,tracked,nlink,opened,state from files where id=? and id in(select id from cohort)',(i,)).fetchone()
                if v and policy.reason(*v).startswith('remove:'):out.append(i)
        report(out)
    elif op=='postcheck':
        from verify_process import postcheck
        report(postcheck())
    elif op=='H-progress':
        q=json.load(sys.stdin);save('H-progress.json',q);report({'accepted':True})
    elif op=='final-H':
        q=json.load(sys.stdin);save('H-final.json',q);report({'accepted':True})
    elif op=='git-receipts':
        f=CTL/'git-receipts.json';report(json.loads(f.read_text()) if f.exists() else {})
    elif op=='close':
        q=json.load(sys.stdin)
        if not (CTL/'postcheck.json').is_file() or not q.get('catalog_sha256'):raise ValueError('Missing postcheck or H catalog digest')
        save('receiver-closed.json',{**q,'at':now(),**BG});report({'closed':True})
    else:raise ValueError(op)
if __name__=='__main__':main()
