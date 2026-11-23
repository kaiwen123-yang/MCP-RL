"""Hash stable files once and capture changing inputs separately; export resumable manifest waves."""
import concurrent.futures as cf
import fcntl,gzip,hashlib,json,os,sqlite3,stat,tempfile,time
from pathlib import Path
from inventory import BASE,CTL,TZ,BG,now,save,sid

WAVES=CTL/'waves';VERSIONS=CTL/'versions'
def gz(p,obj):
    tmp=p.with_name(p.name+'.pending')
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as f:json.dump(obj,f,ensure_ascii=True,separators=(',',':'))
    os.replace(tmp,p)
def stable_sig(st):return (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_dev,st.st_ino,st.st_nlink)
def open_regular(p):
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    f=os.fdopen(fd,'rb');s=os.fstat(fd)
    if not stat.S_ISREG(s.st_mode):
        f.close();raise ValueError('changed file type')
    return f,s
def freeze(i,p):
    # A still-open append log gets an exact finite prefix, verified a second time.
    f,s=open_regular(p)
    with f,tempfile.NamedTemporaryFile(dir=VERSIONS,prefix=f'{i}-',delete=False) as out:
        tmp=Path(out.name);n=s.st_size;remaining=n;h=hashlib.sha256()
        while remaining:
            b=f.read(min(2**20,remaining))
            if not b:break
            out.write(b);h.update(b);remaining-=len(b)
        f.seek(0);confirm=hashlib.sha256();rem=n
        while rem:
            b=f.read(min(2**20,rem))
            if not b:break
            confirm.update(b);rem-=len(b)
    if remaining or rem or h.digest()!=confirm.digest():
        tmp.unlink();raise OSError('mutable source changed within the captured prefix')
    digest=h.hexdigest();dst=VERSIONS/(str(i)+'-'+digest)
    os.replace(tmp,dst)
    return digest,'captured_version',sid(s),str(dst)
def measure(row):
    i,p,size,mt,ct,dev,ino,nlink,allocated,mode,opened=row
    try:
        if opened:return (i,*freeze(i,p))
        f,s=open_regular(p)
        with f:
            if stable_sig(s)!=(size,mt,ct,dev,ino,nlink):return (i,*freeze(i,p))
            h=hashlib.file_digest(f,'sha256').hexdigest()
            if stable_sig(s)!=stable_sig(os.fstat(f.fileno())):return (i,*freeze(i,p))
        return (i,h,'source_hashed',sid(s),None)
    except (OSError,ValueError) as e:
        return (i,None,'changed_or_unavailable:'+type(e).__name__+':'+str(e)[:230],None,None)
def run():
    lock=(CTL/'hash.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (CTL/'inventory.done.json').exists():time.sleep(4)
    WAVES.mkdir(exist_ok=True);VERSIONS.mkdir(exist_ok=True)
    db=sqlite3.connect(CTL/'inventory.sqlite',timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    names={r[1] for r in db.execute('pragma table_info(files)')}
    if 'capture' not in names:db.execute('ALTER TABLE files ADD COLUMN capture TEXT')
    db.execute('CREATE TABLE IF NOT EXISTS waves(id INTEGER PRIMARY KEY, first_id INTEGER,last_id INTEGER,kind TEXT)')
    db.commit()
    def export(ids,kind):
        number=db.execute('select coalesce(max(id),0)+1 from waves').fetchone()[0]
        # IDs are unique within a wave; order is also the recovery index.
        rows=[]
        for k in range(0,len(ids),750):
            sel=ids[k:k+750];cols='id,path,root,kind,size,mtime,ctime,dev,ino,nlink,allocated,mode,tracked,link,opened,digest,state,capture'
            rows.extend(db.execute('SELECT '+cols+' FROM files WHERE id IN ('+','.join('?'*len(sel))+')',sel))
        gz(WAVES/f'{number:05d}.json.gz',rows)
        db.execute('insert into waves values(?,?,?,?)',(number,min(ids),max(ids),kind))
        db.executemany('insert into exported values(?)',((i,) for i in ids));db.commit()
        save('hash-progress.json',dict(at=now(),waves=number,latest=kind,**BG))
    db.execute('CREATE TABLE IF NOT EXISTS exported(id INTEGER PRIMARY KEY)');db.commit()
    if not (CTL/'prior-matched.json').exists():
        with gzip.open(CTL/'inventory.json.gz','rt') as f:old=json.load(f)
        with gzip.open(CTL/'sha256.json.gz','rt') as f:hs=json.load(f)
        assert len(hs)==len(old)
        mp={str(BASE/x[0]):(tuple([x[1],x[2],x[3],x[5],x[4],x[6],x[7]]),h) for x,h in zip(old,hs)}
        del old,hs
        hits=[]
        for i,p,n,mt,ct,dev,ino,nl,a,opened in db.execute("SELECT id,path,size,mtime,ctime,dev,ino,nlink,allocated,opened FROM files WHERE kind='file' AND state='snapshot'"):
            v=mp.get(p)
            if v and not opened and (n,mt,ct,dev,ino,nl,a)==v[0]:hits.append((v[1],i))
        db.executemany("UPDATE files SET digest=?,state='verified_prior_signature' WHERE id=?",hits);db.commit()
        save('prior-matched.json',dict(at=now(),matches=len(hits),**BG));del hits,mp
    # Publish old verified matches and link metadata first so reclamation can proceed during new hashing.
    for kind,where in [('old',"state='verified_prior_signature'"),('metadata',"kind!='file'")]:
        ids=[i for i, in db.execute('SELECT id FROM files WHERE '+where+' AND id NOT IN (select id from exported) ORDER BY id')]
        for j in range(0,len(ids),12000):export(ids[j:j+12000],kind)
    cached={}
    cols='id,path,size,mtime,ctime,dev,ino,nlink,allocated,mode,opened'
    remaining=db.execute("SELECT "+cols+" FROM files WHERE kind='file' AND id NOT IN (select id from exported) ORDER BY CASE WHEN path LIKE '/autodl-fs/%' THEN 2 WHEN path LIKE '%/work/envs/%' OR path LIKE '%/.git/%' THEN 1 ELSE 0 END,id")
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        while data:=remaining.fetchmany(3200):
            pending=[];source={};derived={}
            for row in data:
                key=(row[4],row[5],row[6],row[2],row[3])
                if row[7]>1 and not row[10] and key in cached:derived[row[0]]=cached[key]
                else:source[row[0]]=row;pending.append(row)
            out=[]
            for i,h,s,actual,cap in pool.map(measure,pending):
                row=source[i]
                if s=='source_hashed' and row[7]>1:cached[(row[4],row[5],row[6],row[2],row[3])]=h
                if actual and s=='captured_version':
                    db.execute('UPDATE files SET size=?,mtime=?,ctime=?,dev=?,ino=?,nlink=?,allocated=?,mode=? WHERE id=?',(*actual,i))
                out.append((h,s,cap,i))
            out.extend((h,'source_hardlink',None,i) for i,h in derived.items())
            db.executemany('UPDATE files SET digest=?,state=?,capture=? WHERE id=?',out)
            export([r[0] for r in data],'hashed')
    stats=db.execute("SELECT state,count(*),sum(size) FROM files WHERE kind='file' GROUP BY state").fetchall()
    gz(CTL/'scan-issues.json.gz',list(db.execute('SELECT * FROM issues')))
    save('hash.done.json',dict(at=now(),states=stats,waves=db.execute('select count(*) from waves').fetchone()[0],**BG))
    db.close()
if __name__=='__main__':run()
