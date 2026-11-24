"""Inventory the finite process cohort and hash only eligible or awaiting-publication output versions."""
import concurrent.futures as cf,collections,os,time,json,gzip,hashlib,sqlite3,sys
from pathlib import Path
from inventory import CTL,BG,save,now
from policy import Policy,SOURCE_PIN
from hash_inventory import open_regular,stable_sig,gz
def main():
    db=sqlite3.connect(CTL/'inventory.sqlite',timeout=90);db.execute('pragma journal_mode=WAL')
    if 'capture' not in {x[1] for x in db.execute('pragma table_info(files)')}:db.execute('alter table files add column capture text')
    db.execute('create table if not exists cohort(id integer primary key,reason text)')
    db.execute('create table if not exists hash_misses(id integer primary key, reason text)')
    if not (CTL/'cohort-summary.json').exists():
        pol=Policy(db,fresh=True); totals=collections.defaultdict(lambda:[0,0,0]);batch=[]
        with gzip.open(CTL/'retained-process-scope.tsv.gz','wt',compresslevel=1) as excludes:
            for i,p,root,k,tr,nl,op,state,n,alloc,dev,ino in db.execute("select id,path,root,kind,tracked,nlink,opened,state,size,allocated,dev,ino from files where kind='file'"):
                why=pol.reason(p,root,k,tr,nl,op,state)
                if (dev,ino) in pol.live_inodes:why='retain:currently_open_inode'
                v=totals[why];v[0]+=1;v[1]+=n;v[2]+=alloc
                if why.startswith(('remove:','archive:')):
                    batch.append((i,why))
                    if len(batch)>=2000:db.executemany('insert or replace into cohort values(?,?)',batch);batch=[]
                elif '/outputs/' in p or p.startswith('/autodl-fs/data/MCP-RL/W57/') or '/w57-raw-scratch/' in p:
                    excludes.write(json.dumps([p,why,n],separators=(',',':'))+'\n')
        db.executemany('insert or replace into cohort values(?,?)',batch);db.commit()
        save('cohort-summary.json',dict(at=now(),groups=dict(totals),scientific_pins=sorted(SOURCE_PIN),**BG))
    db.execute("attach database ? as prev",(str(CTL.parent/'.mcp-rl-h-archive-20261123/inventory.sqlite'),))
    db.execute("""update files as f set digest=(select o.digest from prev.files o where o.path=f.path),state='prior_exact_source_signature'
       where f.id in(select id from cohort) and f.digest is null and exists(select 1 from prev.files o where o.path=f.path and o.kind='file' and o.digest is not null and o.size=f.size and o.mtime=f.mtime and o.ctime=f.ctime and o.dev=f.dev and o.ino=f.ino and o.nlink=f.nlink and o.state not like 'captured%' and o.capture is null)""");db.commit()
    def measure(r):
        i,p,*signature=r
        try:
            f,s=open_regular(p)
            with f:
                if stable_sig(s)!=tuple(signature):return (i,None,'identity_changed')
                h=hashlib.file_digest(f,'sha256').hexdigest()
                if stable_sig(os.fstat(f.fileno()))!=tuple(signature):return(i,None,'changed_during_hash')
            return i,h,'source_hashed'
        except OSError as e:return i,None,str(e)
    totals=db.execute('select count(*),sum(f.size) from cohort c join files f using(id)').fetchone()
    print(now(),'selected',totals,flush=True)
    done=0;ts=time.monotonic();ids=db.execute("select f.id,path,size,mtime,ctime,dev,ino,nlink from cohort c join files f using(id) where f.digest is null and f.id not in(select id from hash_misses) order by f.id").fetchall()
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        for i in range(0,len(ids),512):
            rr=list(pool.map(measure,ids[i:i+512]));db.executemany('update files set digest=?,state=? where id=?',[(h,s,j) for j,h,s in rr if h])
            db.executemany('insert or replace into hash_misses values(?,?)',[(j,s) for j,h,s in rr if not h]);db.commit();done+=len(rr)
            if time.monotonic()-ts>12:save('hash-progress.json',dict(at=now(),new_checked=done,new_total=len(ids),**BG));ts=time.monotonic()
    waves=CTL/'waves';waves.mkdir(exist_ok=True)
    cur=db.execute("select f.id,path,root,kind,size,mtime,ctime,dev,ino,nlink,allocated,mode,tracked,link,opened,digest,state,capture,c.reason from cohort c join files f using(id) where f.digest is not null order by f.id")
    num=0
    with gzip.open(CTL/'policy.jsonl.gz.tmp','wt',compresslevel=1) as out:
        while chunk:=cur.fetchmany(4096):
            num+=1;gz(waves/f'{num:05d}.json.gz',[a[:-1] for a in chunk])
            for a in chunk:out.write(json.dumps([a[0],a[-1] if a[-1].startswith('remove:') else 'remove:requires_completed_release_at_retirement'],separators=(',',':'))+'\n')
    os.replace(CTL/'policy.jsonl.gz.tmp',CTL/'policy.jsonl.gz')
    save('catalog.done.json',dict(at=now(),selected=totals[0],selected_bytes=totals[1],scoped_hash_failures=db.execute('select count(*) from hash_misses').fetchone()[0],waves=num,**BG))
    print(now(),'hashed',num,'waves',flush=True)
if __name__=='__main__':main()
