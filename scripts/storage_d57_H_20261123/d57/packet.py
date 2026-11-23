"""Binary SSH endpoints for the H archive: manifest waves, content-addressed ZIPs and acknowledged retirements."""
import contextlib,fcntl,gzip,hashlib,io,json,os,sqlite3,stat,subprocess,sys,time,zipfile
from pathlib import Path
from inventory import CTL,BG,now,sid
from hash_inventory import freeze,open_regular,stable_sig
from policy import Policy,live_paths

def db_open():return sqlite3.connect((CTL/'inventory.sqlite').as_uri()+'?mode=ro',uri=True,timeout=90)
def read(p):return json.loads(Path(p).read_text())
def report(x):print(json.dumps(x,separators=(',',':')),flush=True)
def affinity():
    allowed=os.sched_getaffinity(0); want=set(range(4,10))|{14,15,20,21,22,23}
    stage=Path('/root/autodl-tmp/MCP-RL-w58/work/w58/stage.json')
    if stage.exists() and read(stage).get('stage') not in ('waiting_W57_natural_release','finished'):
        p=Path('/root/autodl-tmp/MCP-RL-w58/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w58/pool.json')
        if p.exists():
            for row in read(p).get('active',[]):
                for cores in row.get('cpu_groups',[]):want-=set(cores)
    if allowed & want:os.sched_setaffinity(0,allowed&want)
def status():
    s=os.statvfs('/autodl-fs/data');m=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    press=lambda n:float(Path('/sys/fs/cgroup/'+n+'.pressure').read_text().splitlines()[0].split()[1].split('=')[1])
    used=int(dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])/1e6
    return dict(now=time.time(),cpu_total=used,io_pressure=press('io'),cpu_pressure=press('cpu'),
        physics=recent_physics(),working_GiB=(int(Path('/sys/fs/cgroup/memory.current').read_text())-int(m['inactive_file']))/2**30,
        policy_ready=(CTL/'policy.jsonl.gz').is_file(),waves=len(list((CTL/'waves').glob('*.json.gz'))),complete=(CTL/'catalog.done.json').exists(),inodes_free=s.f_favail,
        scan=read(CTL/'inventory.done.json') if (CTL/'inventory.done.json').exists() else None,**BG)
class Prefix:
    def __init__(self,out,skip,expected):
        self.out=out;self.rem=skip;self.h=hashlib.sha256();self.expected=expected
    def write(self,b):
        n=len(b)
        if self.rem:
            take=min(n,self.rem);self.h.update(b[:take]);self.rem-=take;b=b[take:]
            if not self.rem and self.h.hexdigest()!=self.expected:raise ValueError('Different retained ZIP prefix')
        if b:self.out.write(b)
        return n
    def flush(self):self.out.flush()

def recent_physics():
    b=Path('/root/autodl-tmp/MCP-RL-w57/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
    result={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            a=(p/'cmdline').read_bytes().split(b'\0')
            if b'--attempt' not in a or not any(x.endswith(b'run_s1_go2_r12.py') for x in a):continue
            name=os.fsdecode(a[a.index(b'--attempt')+1])
            f=b/name/'live.json'
            if f.is_file():
                d=read(f);result[name]=[d['physical_time_s'],d['phase']]
        except (OSError,ValueError,KeyError,IndexError):continue
    return result

def waves(req):
    first=int(req['first']);end=min(first+31,int(req['last']))
    with zipfile.ZipFile(sys.stdout.buffer,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for n in range(first,end+1):
            path=CTL/'waves'/f'{n:05d}.json.gz'
            if not path.is_file():break
            z.write(path,path.name)
def pack(req,output=None):
    expected=req['objects'];db=db_open()
    meta=[];omitted=[]
    sink=Prefix(output if output is not None else sys.stdout.buffer,int(req.get('offset',0)),req.get('prefix',''))
    with zipfile.ZipFile(sink,'w',allowZip64=True,compresslevel=1) as archive:
        for ident,target,expected_size in expected:
            current=db.execute("SELECT path,size,mtime,ctime,dev,ino,nlink,digest,capture,state FROM files WHERE id=? AND kind='file'",(int(ident),)).fetchone()
            if not current or current[7]!=target or current[1]!=expected_size:raise ValueError('Unbound source id '+str(ident))
            path,size,mt,ct,dev,ino,nl,digest,cap,origin=current
            try:
                f,st=open_regular(cap or path)
            except FileNotFoundError:
                omitted.append([ident,'vanished_after_scan']);continue
            revised=None
            if not cap and stable_sig(st)!=(size,mt,ct,dev,ino,nl):
                f.close()
                try:
                    digest,state,actual,cap=freeze(ident,path)
                except (FileNotFoundError,OSError) as err:
                    omitted.append([ident,'changed:'+type(err).__name__]);continue
                size=actual[0];revised=[size,digest,list(actual),state]
                f,st=open_regular(cap)
            with f:
                member=f'objects/{ident}-{digest}'
                zi=zipfile.ZipInfo(member);zi.date_time=(2026,11,23,0,0,0)
                ext=Path(path).suffix.lower()
                head=f.read(12);f.seek(0)
                stored=ext in ('.gz','.bz2','.xz','.zst','.zip','.7z','.mp4','.jpg','.png','.whl') or (ext=='.npz' and head[8:10]==bytes((8,0)))
                zi.compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
                zi._compresslevel=1;zi.external_attr=0o100600<<16
                h=hashlib.sha256();n=0
                with archive.open(zi,'w',force_zip64=True) as dst:
                    while b:=f.read(2**20):
                        n+=len(b);h.update(b);dst.write(b)
                if n!=size or h.hexdigest()!=digest:
                    omitted.append([ident,'changed_during_read']);continue
                meta.append([ident,n,digest,member,revised])
        archive.writestr('_verification.json',json.dumps(dict(files=meta,omitted=omitted),separators=(',',':')))
    if sink.rem:raise ValueError('ZIP prefix exceeded current archive size')
    db.close()

def refresh(req):
    db=sqlite3.connect(CTL/'inventory.sqlite',timeout=60);out=[]
    for i in req['ids']:
        row=db.execute("SELECT path,kind,size,digest FROM files WHERE id=?",(i,)).fetchone()
        if not row or row[1]!='file':raise ValueError('Unknown refresh source')
        try: digest,state,st,cap=freeze(i,row[0])
        except (OSError,ValueError) as e:
            out.append([i,None,str(type(e).__name__)]);continue
        db.execute('UPDATE files SET size=?,mtime=?,ctime=?,dev=?,ino=?,nlink=?,allocated=?,mode=?,digest=?,state=?,capture=? WHERE id=?',
                   (*st,digest,state,cap,i));out.append([i,digest,list(st),cap])
    db.commit();db.close();report(out)

def planner():
    db=db_open();policy=Policy(db,fresh=True);p=CTL/'policy.jsonl.gz';tmp=p.with_suffix('.pending')
    with gzip.open(tmp,'wt',compresslevel=1) as f:
        for i,path,root,k,tr,nl,op,st in db.execute('SELECT id,path,root,kind,tracked,nlink,opened,state FROM files ORDER BY id'):
            f.write(json.dumps([i,policy.reason(path,root,k,tr,nl,op,st)],separators=(',',':'))+'\n')
    os.replace(tmp,p);db.close()
    print(now(),'completed policy map',p.stat().st_size,flush=True)

def retire(req):
    if req.get('volume_serial')!='6063F4DA' or req.get('volume_label')!='AI_DATA_8TB':raise ValueError('H identity differs')
    if len(req['files'])>2400:raise ValueError('Oversized receipt')
    db=db_open();policy=Policy(db)
    ack=sqlite3.connect(CTL/'retired.sqlite',timeout=90);ack.execute('pragma journal_mode=WAL');ack.execute('pragma synchronous=FULL')
    ack.execute('create table if not exists actions(id integer primary key,state text,sha text,proof text,at real)');ack.commit()
    result=[]
    authorized=req['archives']
    for a in authorized.values():
        if len(a['sha256'])!=64 or a['size']<=0:raise ValueError('Incomplete H verification')
    # Exactly one unlink coordinator owns this finite archive. No recursive deletion or link traversal.
    with (CTL/'retire.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        tracked={};last_check=time.monotonic()
        verified=[]
        def flush():
            if not verified:return
            ack.executemany('insert or replace into actions values(?,?,?,?,?)',[(i,'intent',h,proof,time.time()) for i,path,s,h,proof in verified]);ack.commit()
            for i,path,signature,h,proof in verified:
                parent=Path(path).parent
                if str(parent.resolve())!=str(parent):
                    result.append([i,'retained:path_changed']);continue
                fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                try:
                    st=os.stat(os.path.basename(path),dir_fd=fd,follow_symlinks=False)
                    if stable_sig(st)!=signature or not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:
                        result.append([i,'retained:changed_before_unlink']);continue
                    os.unlink(os.path.basename(path),dir_fd=fd)
                    ack.execute("update actions set state='removed',at=? where id=?",(time.time(),i));result.append([i,'removed'])
                finally:os.close(fd)
            ack.commit();verified.clear()
        for item in req['files']:
            i,h,ar,member=item
            if ar not in authorized:raise ValueError('Unknown H archive')
            row=db.execute("select path,root,kind,size,mtime,ctime,dev,ino,nlink,tracked,opened,digest,state from files where id=?",(i,)).fetchone()
            if not row or row[11]!=h:raise ValueError('H content does not match inventory row '+str(i))
            path,root,k,n,mt,ct,dev,ino,nl,tr,op,_,state=row
            prev=ack.execute('select state,sha from actions where id=?',(i,)).fetchone()
            if prev and not os.path.lexists(path):
                ack.execute("update actions set state='removed' where id=?",(i,));ack.commit();result.append([i,'removed']);continue
            if prev and prev[0]=='removed':
                result.append([i,'retained:reappeared']);continue
            if time.monotonic()-last_check>6:
                policy.live,policy.busy=live_paths();last_check=time.monotonic()
            reason=policy.reason(path,root,k,tr,nl,op,state)
            if not reason.startswith('remove:'):result.append([i,reason]);continue
            if root in policy.registered and root not in tracked:
                tracked[root]=set(os.fsdecode(s) for s in subprocess.check_output(['git','-C',root,'ls-files','-z']).split(b'\0') if s)
            if path[len(root)+1:] in tracked.get(root,()):result.append([i,'retained:newly_tracked']);continue
            allowed=policy.roots['roots']
            if root not in allowed or not path.startswith(root+'/') or str(Path(path).parent.resolve())!=str(Path(path).parent):
                result.append([i,'retain:path_identity']);continue
            signature=(n,mt,ct,dev,ino,nl)
            try:f,s=open_regular(path)
            except FileNotFoundError:result.append([i,'missing_without_unlink']);continue
            with f:
                if stable_sig(s)!=signature or not stat.S_ISREG(s.st_mode) or nl!=1:result.append([i,'retained:changed']);continue
                if hashlib.file_digest(f,'sha256').hexdigest()!=h or stable_sig(os.fstat(f.fileno()))!=signature:
                    result.append([i,'retained:source_digest_changed']);continue
            proof=json.dumps([ar,authorized[ar]['sha256'],member],separators=(',',':'))
            verified.append((i,path,signature,h,proof))
            if len(verified)>=32:flush()
        flush()
    ack.close();db.close();return result

if __name__=='__main__':
    affinity();act=sys.argv[1]
    if act=='status':report(status())
    elif act=='waves':waves(json.load(sys.stdin))
    elif act=='pack':
        from tmux_archive import stream
        stream(json.load(sys.stdin))
    elif act=='late-policy':
        from late_retention import eligible
        report(eligible(json.load(sys.stdin)))
    elif act=='retirement-closed':
        from inventory import save
        info=json.load(sys.stdin)
        save('retirement-closed.json',dict(at=now(),scope='23 Nov fixed watermark and two completed tails',H_gate=info,**BG));report({'accepted':True})
    elif act=='receiver-closed':
        from inventory import save
        save('receiver-closed.json',dict(at=now(),H= json.load(sys.stdin),**BG));report({'accepted':True})
    elif act=='postcheck':
        from postcheck import run
        while not (CTL/'retirement-closed.json').exists():time.sleep(5)
        outcome=run()
        from inventory import save
        save('postcheck-verified.json',outcome);report(outcome)
    elif act=='git-publications':
        path=CTL/'github-states.json'
        report(read(path) if path.exists() else {})
    elif act=='H-progress':
        from inventory import save
        save('H-progress.json',json.load(sys.stdin));report({'saved_at':now()})
    elif act=='project-meta':
        manifest={k:read(CTL/k) for k in ('roots.json','inventory.done.json','prior-matched.json','git-snapshot.json','hash.done.json','catalog.done.json') if (CTL/k).exists()}
        report(manifest)
    elif act in ('git-pack','tail-map','scan-issues'):
        target=CTL/{'git-pack':'project-history.pack','tail-map':'tail-aliases.json.gz','scan-issues':'scan-issues.json.gz'}[act]
        with target.open('rb') as f:
            while b:=f.read(2**20):sys.stdout.buffer.write(b)
    elif act=='policy-build':planner()
    elif act=='policy':
        with (CTL/'policy.jsonl.gz').open('rb') as f:
            while data:=f.read(1024**2):sys.stdout.buffer.write(data)
    elif act=='retire':
        from tmux_archive import store_retire
        report(store_retire(json.load(sys.stdin)))
    elif act=='refresh':refresh(json.load(sys.stdin))
    else:raise ValueError('Unknown operation')
