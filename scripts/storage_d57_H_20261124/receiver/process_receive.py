"""Receive and verify the finite D57 process-data archive, then retire just its proven source files."""
import concurrent.futures as cf
import gzip, hashlib, io, json, msvcrt, os, sqlite3, subprocess, sys, time, zipfile
from pathlib import Path
from archive_common import ROOT, OPS, CAT, INC, OLD, BG, now, check_volume, request, rem, write, sha
from catalog_db import connect, initialize
from retire_client import load_policy, send_batch

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

def status(phase, **extra):
    write(CAT/'current.json', dict(at=now(),phase=phase,terminal=False,**BG,**extra))

def pull(op, name):
    target=CAT/name
    if target.exists():return
    pending=INC/(name+'.part')
    with pending.open('wb') as out:
        res=rem(op,out=out)
        out.flush();os.fsync(out.fileno())
    if res.returncode:raise RuntimeError(res.stderr.decode(errors='replace')[-700:])
    with gzip.open(pending,'rb') as f:
        while f.read(2**20):pass
    os.replace(pending,target)

def ingest(c):
    if c.execute("select 1 from meta where k='all_waves'").fetchone():return
    while True:
        check_volume();j=request('meta')
        if 'catalog.done.json' in j:break
        status('hashing_on_D57',scan=j.get('cohort-summary.json'),hash=j.get('hash-progress.json'));time.sleep(6)
    scope=j['catalog.done.json']
    if scope['scoped_hash_failures']:raise RuntimeError('D57 reported a changed source; inspect hash_misses')
    write(CAT/'scope.json',j)
    pull('policy','policy.jsonl.gz');pull('exclusions','retained-process-scope.tsv.gz')
    c.execute('create table if not exists aliases(path text primary key,target text)')
    c.executemany('insert or replace into aliases values(?,?)',request('aliases').items());c.commit()
    first=c.execute('select coalesce(max(id),0)+1 from waves').fetchone()[0]
    cols='id,path,root,kind,n,mt,ct,dev,ino,nl,alloc,mode,tr,link,op,h,state,capture'
    while first<=scope['waves']:
        res=rem('waves',dict(first=first,last=min(first+7,scope['waves'])))
        if res.returncode:raise RuntimeError(res.stderr.decode(errors='replace')[-750:])
        with zipfile.ZipFile(io.BytesIO(res.stdout)) as z:
            for info in z.infolist():
                i=int(info.filename.split('.')[0]);assert i==first
                rows=json.loads(gzip.decompress(z.read(info)))
                c.executemany('insert or ignore into files('+cols+') values('+','.join(['?']*18)+')',rows)
                c.execute('insert or ignore into waves values(?,?)',(i,now()));first+=1
        c.commit();status('receiving_source_manifest',waves=first-1,total_waves=scope['waves'])
    counts=c.execute('select count(*),sum(n) from files').fetchone()
    assert counts==(scope['selected'],scope['selected_bytes'])
    c.execute('insert or ignore into objects(h,n,rep) select h,n,min(id) from files group by h,n');c.commit()
    wanted=dict(c.execute('select h,n from objects where archive is null'))
    records=[];archives={}
    def import_rows(stream):
        for h,n,rel,member,size,mt,digest in stream:
            if wanted.get(h)!=n:continue
            del wanted[h];records.append((rel,member,h));archives[rel]=(rel,size,mt,digest,'reused_H_archive',0,None)
    with sqlite3.connect(OLD.as_uri()+'?mode=ro',uri=True) as prev:
        import_rows(prev.execute('select o.h,o.n,o.archive,o.member,a.n,a.mt,a.h from objects o join archives a on o.archive=a.path where a.ok=1'))
    old22=ROOT/'00_catalog/migration-20261122/archive-index.sqlite'
    if wanted and old22.exists():
        with sqlite3.connect(old22.as_uri()+'?mode=ro',uri=True) as prev:
            import_rows(prev.execute('select c.sha256,c.bytes,a.path,c.member,a.bytes,a.mtime_ns,a.sha256 from contents c join archives a on a.id=c.archive_id'))
    c.executemany('insert or ignore into archives values(?,?,?,?,?,?,?)',archives.values())
    c.executemany('update objects set archive=?,member=? where h=?',records)
    load_policy(c)
    c.execute('update files set retention=(select reason from classification where classification.id=files.id)')
    c.execute('create index if not exists member_archive on objects(archive)')
    c.execute("insert or replace into meta values('all_waves','yes')");c.commit()
    j=dict(at=now(),files=counts[0],logical_bytes=counts[1],unique_objects=c.execute('select count(*) from objects').fetchone()[0],
           reuse_objects=len(records),old_H_archives=len(archives),new_distinct_objects=len(wanted),new_source_bytes=sum(wanted.values()),**BG)
    write(CAT/'manifest-ready.json',j);print(now(),'manifest',j,flush=True)

def prior_proof(row):
    rel,n,mt,digest=row;check_volume();p=ROOT/rel
    if not p.is_file() or p.stat().st_size!=n:return (rel,False,'container missing or different size',0,0)
    before=p.stat();got=sha(p)
    if got!=digest:return(rel,False,'container digest differs',0,0)
    with sqlite3.connect((CAT/'archive-index.sqlite').as_uri()+'?mode=ro',uri=True,timeout=90) as db:
        members=db.execute('select h,n,member from objects where archive=?',(rel,)).fetchall()
    packed=[m for m in members if m[2] is not None]
    for h,size,member in members:
        if member is None and (size,h)!=(n,digest):raise ValueError('Unbound H standalone object: '+rel)
    if packed:
        with zipfile.ZipFile(p) as z:
            for h,size,member in packed:
                v=z.getinfo(member)
                if v.file_size!=size:raise ValueError('Previous member size mismatch '+rel+'/'+member)
                with z.open(v) as f:
                    if hashlib.file_digest(f,'sha256').hexdigest()!=h:raise ValueError('Previous member digest mismatch '+rel+'/'+member)
    end=p.stat();check_volume()
    if (before.st_size,before.st_mtime_ns)!=(end.st_size,end.st_mtime_ns):raise ValueError('H archive changed during readback '+rel)
    return(rel,True,got,end.st_mtime_ns,len(members))

def verify_old(c):
    rows=c.execute("select path,n,mt,h from archives where kind='reused_H_archive' and ok=0 order by path").fetchall()
    allcount=len(rows);proof=CAT/'reused-H-readback.jsonl'
    with cf.ThreadPoolExecutor(max_workers=2) as pool,proof.open('a',encoding='utf8',buffering=1) as out:
        for i,(rel,ok,detail,mt,members) in enumerate(pool.map(prior_proof,rows),1):
            out.write(json.dumps(dict(path=rel,ok=ok,proof=detail,mtime_ns=mt,members=members,at=now()),separators=(',',':'))+'\n')
            if ok:c.execute('update archives set ok=1,mt=?,checked=? where path=?',(mt,now(),rel))
            else:
                check_volume()
                c.execute('update archives set ok=-1,checked=? where path=?',(now(),rel))
                c.execute('update objects set archive=null,member=null,batch=null where archive=?',(rel,))
            c.commit()
            if i%10==0 or i==allcount:status('parallel_transfer_and_previous_H_verification',verified_this_pass=i,pending_this_pass=allcount-i)
    write(CAT/'previous-proof-complete.json',dict(at=now(),verified=c.execute("select count(*) from archives where kind='reused_H_archive' and ok=1").fetchone()[0],**BG))

def full_archive_check(c):
    f=CAT/'archive-complete-proof.json'
    if f.exists():return read(f)
    if not (CAT/'transfer-complete.json').exists():raise ValueError('Transfer incomplete')
    bad=c.execute("""select count(*) from files f left join objects o on f.h=o.h and f.n=o.n left join archives a on o.archive=a.path where f.h is null or f.state like 'missing%' or coalesce(a.ok,0)!=1""").fetchone()[0]
    if bad:raise RuntimeError('Selected files without a matching H receipt: '+str(bad))
    rows=c.execute("select path,n,mt,h from archives where ok=1 and exists(select 1 from objects o where o.archive=archives.path) order by path").fetchall()
    new=set(x[0] for x in c.execute("select path from archives where kind='new_D57_pack' and ok=1"))
    for i,(rel,n,mt,h) in enumerate(rows):
        check_volume();p=ROOT/rel;s=p.stat()
        if (s.st_size,s.st_mtime_ns)!=(n,mt):raise IOError('H archive metadata changed '+rel)
        if rel in new and sha(p)!=h:raise IOError('H new pack readback differs '+rel)
        if i%24==0:status('final_H_readback_before_deletion',checked=i,total=len(rows))
    files,n=c.execute('select count(*),sum(n) from files').fetchone()
    data=dict(files=files,logical_bytes=n,volume_serial='6063F4DA',all_objects_verified=True,archives=len(rows),
              new_packs=len(new),new_H_bytes=c.execute("select coalesce(sum(n),0) from archives where kind='new_D57_pack' and ok=1").fetchone()[0],
              reused_H_archives=len(rows)-len(new),manifest_sha256=hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),at=now(),**BG)
    write(f,data);return data

def retire(c):
    p=full_archive_check(c);check_volume();request('verified',p)
    while True:
        if not send_batch(c):break
        if time.monotonic()-retire.previous>35:
            vals=c.execute("select count(*),coalesce(sum(n),0),coalesce(sum(alloc),0) from files where server_status='removed'").fetchone()
            status('retiring_verified_D57_files',removed_files=vals[0],removed_logical_bytes=vals[1],removed_allocated=vals[2],archive_proof=p)
            retire.previous=time.monotonic()
    waiting=[a[0] for a in c.execute("select id from retire_queue where state='archive:pending_release'")]
    if waiting:
        ids=[]
        for i in range(0,len(waiting),1000):ids.extend(request('eligible-again',dict(ids=waiting[i:i+1000])))
        if ids:
            c.executemany("update retire_queue set state='pending' where id=?",[(i,) for i in ids]);c.commit();return retire(c)
    write(CAT/'retirement-done.json',dict(at=now(),states=c.execute("select q.state,count(*),sum(f.n),sum(f.alloc) from retire_queue q join files f using(id) group by q.state").fetchall(),**BG))
retire.previous=0

def main():
    check_volume()
    lf=(OPS/'receiver.lock').open('a+b');lf.seek(0);msvcrt.locking(lf.fileno(),msvcrt.LK_NBLCK,1)
    c=connect();initialize(c);ingest(c)
    tr=None
    if not (CAT/'transfer-complete.json').is_file():
        tr=subprocess.Popen([sys.executable,'-u',str(OPS/'transfer_packs.py')],cwd=OPS,creationflags=subprocess.CREATE_NO_WINDOW)
    verify_old(c)
    if tr and tr.wait()!=0:raise RuntimeError('Transfer process needs attention, read archive-run.log and batches.err')
    retire(c)
    from process_finish import finish
    finish(c)
if __name__=='__main__':
    try:main()
    except Exception as e:
        print(now(),repr(e),flush=True)
        if CAT.is_dir():status('needs_attention',reason=repr(e))
        raise

