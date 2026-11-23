"""Seal the two physical episodes that were still writing at the inventory watermark, with final versions."""
import fcntl,gzip,hashlib,json,os,sqlite3,stat,subprocess,time
from pathlib import Path
from inventory import BASE,ROOT,CTL,BG,TZ,now,save,sid
from hash_inventory import freeze,measure,gz

REPO=BASE/'MCP-RL-w57'; REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
SMALL=REPO/'results'/REL/'w57';RAW=REPO/'outputs'/REL
def read(p):return json.loads(p.read_text())
def owner(p,roots):
    for root in roots:
        if p==root or root in p.parents:return str(root)
    raise ValueError('Tail resolves outside project roots '+str(p))
def git_bundle():
    path=CTL/'project-history.pack'
    if path.is_file() and (CTL/'git-snapshot.json').is_file():return
    bare=CTL/'history.git'
    subprocess.run(['git','init','--bare','-q',str(bare)],check=True)
    alt=bare/'objects/info/alternates';alt.write_text(str(ROOT/'.git/objects')+'\n')
    pairs=[l.split() for l in subprocess.check_output(['git','-C',str(ROOT),'for-each-ref','--format=%(objectname) %(refname)'],text=True).splitlines()]
    curr=read(CTL/'roots.json')['registered'];allheads={}
    for name in curr:
        head=subprocess.check_output(['git','-C',name,'rev-parse','HEAD'],text=True).strip()
        pairs.append([head,'refs/archive-worktrees/'+Path(name).name]);allheads[name]=head
    subprocess.run(['git','--git-dir',str(bare),'update-ref','--stdin'],input=''.join('update '+ref+' '+h+'\n' for h,ref in pairs),text=True,check=True)
    # D57 has a real shallow boundary. Preserve it explicitly so Git can validate
    # and restore its locally available graph without fetching missing ancestors.
    shallow=(ROOT/'.git/shallow').read_bytes() if (ROOT/'.git/shallow').is_file() else b''
    if shallow:(bare/'shallow').write_bytes(shallow)
    tmp=CTL/'project-history.pack.pending'
    with tmp.open('wb') as out:
        subprocess.run(['git','--git-dir',str(bare),'-c','pack.threads=2','-c','pack.windowMemory=32m',
                        'pack-objects','--stdout','--revs','--delta-base-offset'],
                       input=('\n'.join(sorted(set(h for h,_ in pairs)))+'\n').encode(),stdout=out,check=True)
    os.replace(tmp,path)
    with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    save('git-snapshot.json',dict(at=now(),file=path.name,bytes=path.stat().st_size,sha256=h,refs={ref:d for d,ref in pairs},shallow=shallow.decode().splitlines(),worktrees=allheads,**BG))
def main():
    lock=(CTL/'tail.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (CTL/'catalog.done.json').exists():return
    git_bundle()
    while not (CTL/'hash.done.json').exists():time.sleep(5)
    db=sqlite3.connect(CTL/'inventory.sqlite',timeout=90)
    db.execute('pragma journal_mode=wal')
    db.execute('create table if not exists tail_alias(id integer primary key,path text unique,previous_id integer)')
    meta=read(CTL/'roots.json');cut=meta['start_unix'];roots=sorted(map(Path,meta['roots']),key=lambda p:-len(str(p)))
    active=[]
    for f in sorted((SMALL/'attempts').glob('*.json')):
        v=read(f)
        if v['started_unix_s']<=cut and v.get('ended_unix_s',float('inf'))>=cut:active.append(f.stem)
    while any('ended_unix_s' not in read(SMALL/'attempts'/(n+'.json')) for n in active):time.sleep(10)
    paths=set()
    for n in active:
        for p in (RAW/n,RAW/'jobs'/n,RAW/'evaluation-policy'/n):
            if p.is_dir():
                base=p.resolve()
                for dr,dirs,files in os.walk(base,followlinks=False):
                    paths.add(Path(dr));paths.update(Path(dr)/s for s in dirs+files)
        for folder in ('attempts','episodes','mechanics'):
            paths.update((SMALL/folder).glob(n+'*.json'))
    for p in [SMALL/'index.json',SMALL/'episodes/index.json',SMALL/'m/closed-loop-short.json',
              REPO/'results'/REL/'current.json',REPO/'docs/S1_DAILY_WORKLOG.md']:
        if p.is_file():paths.add(p)
    # Revisit entries that vanished/replaced during the initial metadata/hash pass.
    paths.update(Path(p) for p, in db.execute("select path from files where kind='file' and digest is null"))
    cols='id,path,root,kind,size,mtime,ctime,dev,ino,nlink,allocated,mode,tracked,link,opened,digest,state,capture'
    added=[];missing=[]
    for p in sorted(paths):
        try:s=p.lstat()
        except FileNotFoundError:missing.append(str(p));continue
        old=db.execute('select id,size,mtime,ctime,dev,ino,nlink,digest,kind from files where path=?',(str(p),)).fetchone()
        if old and (s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_dev,s.st_ino,s.st_nlink)==old[1:7] and (old[7] or old[8]!='file'):continue
        if old and not stat.S_ISREG(s.st_mode):continue
        if db.execute('select 1 from tail_alias where path=?',(str(p),)).fetchone():continue
        i=db.execute('select coalesce(max(id),0)+1 from files').fetchone()[0]
        kind='file' if stat.S_ISREG(s.st_mode) else 'dir' if stat.S_ISDIR(s.st_mode) else 'link' if stat.S_ISLNK(s.st_mode) else 'special'
        physical=str(p);cap=None;h=None;st=sid(s);state='closed_tail';link=os.readlink(p) if kind=='link' else None
        if kind=='file':
            if old:
                h,state,st,cap=freeze(i,str(p));physical=cap
            else:
                _,h,state,st,cap=measure((i,str(p),*sid(s),0))
                if not h:raise RuntimeError('Unable to capture final tail '+str(p)+': '+state)
        row=(i,physical,owner(p,roots),kind,*st,0,link,1,h,state,cap)
        db.execute('insert into files('+cols+') values('+','.join('?'*18)+')',row)
        db.execute('insert into tail_alias values(?,?,?)',(i,str(p),old[0] if old else None))
        added.append(row)
        if len(added)%500==0:db.commit()
    db.commit()
    pending=db.execute('SELECT '+','.join('f.'+s for s in cols.split(','))+' FROM files f JOIN tail_alias t ON t.id=f.id LEFT JOIN exported e ON e.id=f.id WHERE e.id IS NULL ORDER BY f.id').fetchall()
    for k in range(0,len(pending),3200):
        rows=pending[k:k+3200];n=db.execute('select coalesce(max(id),0)+1 from waves').fetchone()[0]
        gz(CTL/'waves'/f'{n:05d}.json.gz',rows)
        db.execute('insert into waves values(?,?,?,?)',(n,rows[0][0],rows[-1][0],'final_tail'))
        db.executemany('insert into exported values(?)',[(r[0],) for r in rows]);db.commit()
    gz(CTL/'tail-aliases.json.gz',list(db.execute('select id,path,previous_id from tail_alias order by id')))
    stats=db.execute('select count(*) from files').fetchone()[0]
    exp=db.execute('select count(*) from exported').fetchone()[0]
    if stats!=exp:raise ValueError(('Incomplete export',stats,exp))
    save('catalog.done.json',dict(at=now(),first_snapshot=meta['start_time'],finalized_initially_active=active,
                     tail_records=db.execute('select count(*) from tail_alias').fetchone()[0],
                     exported_paths=exp,unavailable_paths=missing,waves=db.execute('select count(*) from waves').fetchone()[0],**BG))
    db.close()
if __name__=='__main__':main()
