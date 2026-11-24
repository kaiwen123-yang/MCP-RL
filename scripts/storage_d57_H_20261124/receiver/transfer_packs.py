"""Parallel resumable SSH packs; H readback and per-member SHA-256 precede every archive acknowledgment."""
from archive_common import *
from catalog_db import connect,initialize
import collections,concurrent.futures as cf,gzip,math,threading,zipfile,statistics

SIZES=[2,4,6,8,12,16,20,24]
VERIFY=threading.BoundedSemaphore(3)
PLANS=OPS/'pack-plans'
class PacketFailure(Exception):pass

def plan(c, max_groups):
    rows=c.execute('select rep,h,n from objects where archive is null and batch is null and rep is not null order by rep limit 13500').fetchall()
    if not rows:return
    group=[];total=0
    def add():
        nonlocal total,group
        i=c.execute('select coalesce(max(id),0)+1 from batches').fetchone()[0]
        key=f'40_support_evidence/20261124-d57-process/new-{i:06d}.zip'
        c.execute("insert into batches values(?,?,'ready',?,?,NULL)",(i,key,total,len(group)))
        with gzip.open(PLANS/f'{i:06d}.json.gz','wt',compresslevel=1) as f:json.dump(group,f,separators=(',',':'))
        c.executemany('update objects set batch=? where h=? and archive is null',[(i,r[1]) for r in group]);c.commit()
        group=[];total=0
    planned=0
    for row in rows:
        if group and (total+row[2]>160*2**20 or len(group)>=1600):
            add();planned+=1
            if planned>=max_groups:return
        group.append(row);total+=row[2]
    if group:add()

def verify(p,objects):
    expected={i:(digest,n) for i,digest,n in objects}
    with zipfile.ZipFile(p) as z:
        meta=json.loads(z.read('_verification.json'))
        good=meta['files'];missing=meta['omitted']
        if {r[0] for r in good}|{r[0] for r in missing}!=set(expected):raise PacketFailure('SSH packet member manifest differs')
        if len(good)+len(missing)!=len(expected):raise PacketFailure('Repeated SSH packet member')
        for i,n,h,member,revision in good:
            if member!=f'objects/{i}-{h}' or (not revision and (h,n)!=expected[i]):raise PacketFailure('Source version disagreement')
            info=z.getinfo(member)
            if info.file_size!=n:raise PacketFailure('H member length mismatch')
            with z.open(info) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=h:raise PacketFailure('H member digest mismatch')
    return meta

def receive(i,rel):
    check_volume();target=ROOT/rel;part=INC/(target.name+'.part')
    with gzip.open(PLANS/f'{i:06d}.json.gz','rt') as f:objects=json.load(f)
    started=time.time()
    if target.is_file():source=target
    else:
        source=part
        # Valid already-complete bytes can outlive the bookkeeping process.
        complete=part.exists() and zipfile.is_zipfile(part)
        if not complete:
            offset=part.stat().st_size if part.exists() else 0
            prefix=sha(part) if offset else ''
            with part.open('ab') as out:
                resp=rem('pack',dict(objects=objects,offset=offset,prefix=prefix),out)
                out.flush();os.fsync(out.fileno())
            if resp.returncode:
                err=resp.stderr.decode(errors='replace')[-1100:]
                if 'Different retained ZIP prefix' in err or 'prefix exceeded' in err:
                    # Preserve for diagnosis; disassociate only this unverified incoming stream.
                    quarantine=part.with_name(part.name+f'.superseded-{time.time_ns()}')
                    os.replace(part,quarantine)
                raise PacketFailure(err)
    with VERIFY:
        check_volume();meta=verify(source,objects);digest=sha(source)
        if source!=target:
            if target.exists():raise PacketFailure('A concurrent writer created the same pack')
            os.replace(part,target)
        st=target.stat()
        data=dict(id=i,path=rel,bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=digest,
                  nfiles=len(meta['files']),source_bytes=sum(r[1] for r in meta['files']),
                  started=started,verified=time.time(),verified_at=now(),volume_serial='6063F4DA')
        write(OPS/'pack-receipts'/f'{i:06d}.json',data)
    return data,meta,objects

def replace_entry(c,i,h,n,why,cap=None):
    original=c.execute('select h,n from files where id=?',(i,)).fetchone()
    c.execute('insert into revisions values(?,?,?,?,?,?,?)',(i,*original,n,h,why,now()))
    c.execute("update files set n=?,h=?,state=?,capture=coalesce(?,capture),server_status='retained:live_version' where id=?",(n,h,why,cap,i))
    if original[0]!=h:
        alt=c.execute("select id from files where h=? and id!=? and state not like 'missing%' limit 1",(original[0],i)).fetchone()
        c.execute("update objects set rep=?,batch=NULL where h=? and rep=? and archive is null",(alt[0] if alt else None,original[0],i))
    return original

def accept(c,pack,meta,objects):
    i=pack['id'];rel=pack['path'];lookup={x[0]:(x[1],x[2]) for x in objects}
    c.execute('insert or replace into archives values(?,?,?,?,?,1,?)',(rel,pack['bytes'],pack['mtime_ns'],pack['sha256'],'new_D57_pack',pack['verified_at']))
    for ident,n,h,member,rev in meta['files']:
        if rev:replace_entry(c,ident,h,n,'captured_during_pack')
        c.execute('insert into objects(h,n,rep,archive,member,batch) values(?,?,?,?,?,?) on conflict(h) do update set archive=CASE WHEN objects.archive IS NULL THEN excluded.archive ELSE objects.archive END,member=CASE WHEN objects.archive IS NULL THEN excluded.member ELSE objects.member END',(h,n,ident,rel,member,i))
    c.execute("update batches set status='verified',err=NULL where id=?",(i,));c.commit()
    if meta['omitted']:
        for ident, reason in meta['omitted']:
            h, n = lookup[ident]
            c.execute("update files set state=?,server_status='retained:source_changed' where id=?",
                      ('missing:'+reason,ident))
            other = c.execute("select min(id) from files where h=? and state not like 'missing%' and id!=?",(h,ident)).fetchone()[0]
            c.execute("update objects set rep=?,batch=NULL where h=? and rep=? and archive is null",(other,h,ident))
        c.commit()

def disk_observation(active):
    total=0
    for i,rel in active.values():
        p=INC/(Path(rel).name+'.part')
        if p.is_file():total+=p.stat().st_size
    return total

def main():
    check_volume();PLANS.mkdir(exist_ok=True);(OPS/'pack-receipts').mkdir(exist_ok=True)
    c=connect();initialize(c)
    c.execute('create table if not exists revisions(id integer,old_sha text,old_bytes integer,new_bytes integer,new_sha text,reason text,at text)')
    c.execute("update batches set status='ready' where status='transferring'");c.commit()
    limit=4;peak=0;failures=collections.Counter();active={};prior={};blacklist={}
    last=0;previous=None;last_bytes=0;windows=[];cooldown=0;log=OPS/'actual-transfer.jsonl'
    with cf.ThreadPoolExecutor(max_workers=24) as pool:
        while True:
            check_volume()
            for future in [f for f in active if f.done()]:
                ident,rel=active.pop(future)
                try:
                    ack,meta,objects=future.result();accept(c,ack,meta,objects)
                    print(now(),'H verified',ident,ack['source_bytes'],ack['bytes'],ack['nfiles'],flush=True)
                except Exception as e:
                    failures[ident]+=1
                    c.execute("update batches set status=?,err=? where id=?",('review' if failures[ident]>=3 else 'ready',str(e)[:850],ident));c.commit()
                    print(now(),'packet pending',ident,repr(e),flush=True);limit=max(2,limit//2);cooldown=time.monotonic()+90
            ready=c.execute("select count(*) from batches where status='ready'").fetchone()[0]
            if ready<2*limit:
                plan(c,max(4,2*limit-ready))
                ready=c.execute("select count(*) from batches where status='ready'").fetchone()[0]
            if not active and not ready and c.execute("select 1 from meta where k='all_waves'").fetchone():
                verifying=c.execute("select 1 from archives where ok=0 limit 1").fetchone()
                nc=1 if verifying else c.execute("""select count(*) from files f left join objects o on f.h=o.h and f.n=o.n
                    left join archives a on o.archive=a.path where f.kind='file' and f.h is not null and f.state not like 'missing%' and coalesce(a.ok,0)!=1""").fetchone()[0]
                if not nc:
                    write(CAT/'transfer-complete.json',dict(completed_at=now(),peak_streams=peak,**BG));break
                if not ready and c.execute("select count(*) from batches where status='review'").fetchone()[0]:
                    raise RuntimeError('Some transfers require diagnosis; see batches.err and retained incoming files')
            for ident,rel in c.execute("select id,path from batches where status='ready' order by id limit ?",(max(0,limit-len(active)),)).fetchall():
                active[pool.submit(receive,ident,rel)]=(ident,rel)
                c.execute("update batches set status='transferring' where id=?",(ident,))
            c.commit();peak=max(peak,len(active))
            if time.monotonic()-last>=25:
                server=request('status')
                amount=c.execute("select coalesce(sum(n),0) from archives where kind='new_D57_pack' and ok=1").fetchone()[0]+disk_observation(active)
                elapsed=server['now']-previous['now'] if previous else 0
                throughput=max(0.,amount-last_bytes)/max(1.,elapsed)/2**20 if previous else 0.
                cores=(server['cpu_total']-previous['cpu_total'])/max(.01,elapsed) if previous else 0.
                rates={k:(v[0]-previous['physics'][k][0])/elapsed for k,v in server.get('physics',{}).items() if previous and k in previous.get('physics',{}) and elapsed>0}
                sample=dict(at=now(),streams=len(active),limit=limit,peak_streams=peak,MiB_per_s=round(throughput,3),
                    total_used_cores=round(cores,2),io_pressure_pct=server['io_pressure'],working_GiB=server['working_GiB'],
                    physics_s_per_wall_s=rates,new_verified_bytes=c.execute("select coalesce(sum(n),0) from archives where kind='new_D57_pack' and ok=1").fetchone()[0],
                    verified_packs=c.execute("select count(*) from batches where status='verified'").fetchone()[0],**BG)
                with log.open('a',encoding='utf8') as f:f.write(json.dumps(sample,separators=(',',':'))+'\n')
                write(CAT/'transfer-progress.json',sample)
                same=previous and len(active)>=limit and time.monotonic()>cooldown and throughput>0
                if same:
                    windows.append(throughput)
                    if len(windows)>=3:
                        speed=statistics.median(windows[-3:])
                        lower=max([n for n in prior if n<limit],default=0)
                        if lower and speed<1.04*prior[lower]:
                            blacklist[limit]=time.monotonic()+300;limit=lower;cooldown=time.monotonic()+100;windows=[]
                        elif cores<22 and server['working_GiB']<105 and server['io_pressure']<11 and server['cpu_pressure']<12:
                            prior[limit]=speed
                            nxt=next((n for n in SIZES if n>limit),None)
                            if nxt and time.monotonic()>blacklist.get(nxt,0):
                                limit=nxt;cooldown=time.monotonic()+35;windows=[]
                elif not active:windows=[]
                if previous and (cores>23.5 or server['working_GiB']>111 or server['cpu_pressure']>25 or server['io_pressure']>32):
                    lower=max([n for n in SIZES if n<limit],default=2)
                    limit=lower;cooldown=time.monotonic()+60;windows=[]
                previous=server;last_bytes=amount;last=time.monotonic()
            time.sleep(1.5 if active else 5)
    c.close()
if __name__=='__main__':main()

