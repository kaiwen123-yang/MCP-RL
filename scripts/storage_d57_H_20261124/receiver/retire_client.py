"""Allow only actual H-proved content into D57's second, live-checked retirement pass."""
from archive_common import *
from catalog_db import connect,initialize
import gzip,hashlib,collections

def load_policy(c):
    f=CAT/'policy.jsonl.gz'
    if not f.is_file():return False
    c.executescript("""CREATE TABLE IF NOT EXISTS classification(id INTEGER PRIMARY KEY,reason TEXT NOT NULL);
                       CREATE TABLE IF NOT EXISTS retire_queue(id INTEGER PRIMARY KEY,state TEXT DEFAULT 'pending');
                       CREATE INDEX IF NOT EXISTS retire_qstate ON retire_queue(state);""")
    if c.execute("select 1 from meta where k='policy_ready'").fetchone():return True
    with gzip.open(f,'rt') as src:
        batch=[]
        for line in src:
            batch.append(json.loads(line))
            if len(batch)==8000:
                c.executemany('insert or replace into classification values(?,?)',batch)
                c.executemany('insert or ignore into retire_queue(id) values(?)',[(i,) for i,r in batch if r.startswith('remove:')]);c.commit();batch=[]
        c.executemany('insert or replace into classification values(?,?)',batch)
        c.executemany('insert or ignore into retire_queue(id) values(?)',[(i,) for i,r in batch if r.startswith('remove:')])
    c.execute("insert or replace into meta values('policy_ready','yes')");c.commit()
    return True

def proofs(c):
    p=CAT/'previous-proofs.jsonl'
    if not p.exists():return
    last=c.execute("select v from meta where k='proof_offset'").fetchone()
    with p.open('rb') as f:
        f.seek(int(last[0]) if last else 0)
        while True:
            at=f.tell();l=f.readline()
            if not l.endswith(b'\n'):break
            d=json.loads(l)
            if not d['ok']:
                c.execute('update archives set ok=-1,checked=? where path=?',(d['verified_at'],d['path']))
                c.execute('update objects set archive=NULL,member=NULL,batch=NULL where archive=?',(d['path'],))
            else:
                c.execute('update archives set ok=1,mt=?,checked=? where path=? and n=? and h=?',
                          (d['mtime_ns'],d['verified_at'],d['path'],d['bytes'],d['sha256']))
        c.execute("insert or replace into meta values('proof_offset',?)",(str(at),));c.commit()

def additional_prior(c):
    if not (CAT/'previous-proof-complete.json').exists():return
    row=c.execute("select path,n,h from archives a where ok=0 and exists(select 1 from objects o where o.archive=a.path) limit 1").fetchone()
    if not row:return
    p,n,h=row;target=ROOT/p;check_volume();ok=target.is_file() and target.stat().st_size==n and sha(target)==h
    if ok:c.execute('update archives set ok=1,mt=?,checked=? where path=?',(target.stat().st_mtime_ns,now(),p))
    else:
        c.execute('update archives set ok=-1,checked=? where path=?',(now(),p))
        c.execute('update objects set archive=NULL,member=NULL,batch=NULL where archive=?',(p,))
    c.commit()

def send_batch(c):
    raw=c.execute("""select f.id,f.h,f.n,o.archive,o.member,a.n,a.mt,a.h
        from retire_queue q join files f on f.id=q.id join objects o on o.h=f.h and o.n=f.n
        join archives a on a.path=o.archive where q.state='pending' and a.ok=1 and f.state not like 'captured%'
        limit 1200""").fetchall()
    if not raw:return 0
    total=0;ar={};files=[]
    for i,h,s,p,member,n,mt,ah in raw:
        if p not in ar:
            target=ROOT/p;st=target.stat()
            if (st.st_size,st.st_mtime_ns)!=(n,mt):raise IOError('Verified H archive has changed: '+p)
            ar[p]=dict(sha256=ah,size=n,mtime_ns=mt)
        files.append([i,h,p,member]);total+=s
        if total>=384*2**20:break
    check_volume()
    envelope=dict(volume_serial='6063F4DA',volume_label='AI_DATA_8TB',archives=ar,files=files,issued_at=now())
    key=hashlib.sha256(json.dumps(files,separators=(',',':')).encode()).hexdigest()
    dest=OPS/'retirement-receipts';dest.mkdir(exist_ok=True)
    proof=dest/(key+'.json.gz')
    if not proof.exists():
        with gzip.open(proof,'wt',compresslevel=1) as f:json.dump(envelope,f,separators=(',',':'))
    check_volume()
    result=request('retire',envelope)
    if set(i for i,_ in result)!=set(i[0] for i in files):raise ValueError('Unmatched retirement acknowledgment')
    write(dest/(key+'.result.json'),dict(archive_request_sha256=sha(proof),source_outcomes=result,completed_at=now()))
    c.executemany('update retire_queue set state=? where id=?',[(s,i) for i,s in result])
    c.executemany('update files set server_status=? where id=?',[(s,i) for i,s in result]);c.commit()
    return len(result)

def run():
    c=connect();initialize(c)
    last=0
    while True:
        check_volume();proofs(c);additional_prior(c)
        if not load_policy(c):time.sleep(5);continue
        n=send_batch(c)
        if time.monotonic()-last>=15 or not n:
            data=c.execute("select q.state,count(*),sum(f.n),sum(f.alloc) from retire_queue q join files f on f.id=q.id group by q.state").fetchall()
            write(CAT/'retirement-progress.json',dict(at=now(),status='running',groups=data,**BG))
            print(now(),data,flush=True);last=time.monotonic()
        if not n:
            if (CAT/'transfer-complete.json').exists():
                proofs(c)
                pending=c.execute("""select count(*) from retire_queue q join files f on f.id=q.id
                    join objects o on f.h=o.h join archives a on o.archive=a.path
                    where q.state='pending' and a.ok=1 and f.state not like 'captured%'""").fetchone()[0]
                if pending==0:break
            time.sleep(6)
    write(CAT/'retirement-done.json',dict(at=now(),**BG))
if __name__=='__main__':run()

