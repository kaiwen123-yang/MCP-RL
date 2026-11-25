"""D57 side of the finite, H-verified W58 closed-process archive."""
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid

ROOT=Path('/root/autodl-tmp/MCP-RL-w58')
REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
SMALL=ROOT/'results'/REL/'w58'
RAW=ROOT/'outputs'/REL
PERSONAL=Path('/autodl-fs/data/MCP-RL/W58/originals')
WORK=ROOT/'work/w58'
CTL=WORK/'h-process-20261125'
AUDIT=WORK/'shared-extents-w58-w57-20261125.jsonl'
SH=ZoneInfo('Asia/Shanghai')
BG=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
PAT=re.compile(r'(?:material-[0-9]+\.npz|physical-[0-9]+\.npz|w55-f/(?:drift|mpc|control)-[0-9]+\.npz)$')
PROCESS=re.compile(r'w55-f/(?:mpc|drift)-[0-9]+\.npz$')

def now():return datetime.now(SH).isoformat(timespec='seconds')
def read(p):return json.loads(p.read_text())
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    os.replace(tmp,p)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def free():
    return {k:shutil.disk_usage(p).free for k,p in [('scratch',ROOT),('personal',PERSONAL)]}

def closed():
    rows={}
    for f in (SMALL/'attempts').glob('*.json'):
        d=SMALL/'delivery'/f.name
        if not d.is_file():continue
        a=read(f);pub=read(d);p=(RAW/f.stem).resolve()
        if (a.get('ended_unix_s') and pub.get('status')=='verified' and pub.get('assets') and
                p.parent in (RAW,PERSONAL) and p.is_dir() and p.name==f.stem):
            rows[p]=a
    return rows

def where(p,rows):
    if p.parent.name=='w55-f':root=p.parent.parent
    else:root=p.parent
    rel=p.relative_to(root).as_posix()
    if root not in rows or not PAT.fullmatch(rel) or p.parent.resolve()!=p.parent:
        raise ValueError('Outside closed episode process files: '+str(p))
    return rows[root],rel

def add(groups,h,n,source):
    return groups.setdefault(h,dict(h=h,n=n,source=str(source),links=[],drops=[]))

def prepare(stage):
    dest=CTL/stage; info=dest/'index.json'
    if info.exists():return
    dest.mkdir(parents=True,exist_ok=True)
    rows=closed(); groups={}; protected=set()
    for line in AUDIT.open():
        a,b,h,n=json.loads(line);src=Path(a);dst=Path(b)
        if not(a.startswith(str(RAW)+'/') and b.startswith(str(RAW)+'/')):continue
        if dst.name=='model.mjb':continue
        where(src,rows);where(dst,rows)
        protected.update((src,dst))
        if stage=='duplicates':
            z=add(groups,h,n,src)
            if b not in z['links'] and b!=z['source']:z['links'].append(b)
    previous=set(read(CTL/'process/index.json')['closed_attempts']) if stage=='followup' else set()
    if stage in ('process','followup'):
        for root,a in rows.items():
            if root.name in previous:continue
            pats=['w55-f/mpc-*.npz','w55-f/drift-*.npz']
            if a.get('status')=='technical_interruption':
                pats+=['physical-[0-9]*.npz','material-[0-9]*.npz','w55-f/control-*.npz']
            for pat in pats:
                for p in root.glob(pat):
                    if p in protected or p.is_symlink():continue
                    item,rel=where(p,rows)
                    if not PROCESS.fullmatch(rel) and item.get('status')!='technical_interruption':
                        raise ValueError(str(p))
                    st=p.stat()
                    if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:continue
                    h=sha(p);z=add(groups,h,st.st_size,p)
                    z['drops'].append(str(p))
    items=sorted(groups.values(),key=lambda z:(-len(z['links']),-z['n'],z['h']))
    packs=[];batch=[];size=0
    def flush():
        if not batch:return
        key=f'{stage}-{len(packs)+1:04d}'
        save(dest/(key+'.json'),batch)
        packs.append(dict(id=key,objects=len(batch),bytes=sum(x['n'] for x in batch),
                          link_bytes=sum(len(x['links'])*x['n'] for x in batch),
                          drop_bytes=sum(len(x['drops'])*x['n'] for x in batch)))
    for o in items:
        if batch and (size+o['n']>128*2**20 or len(batch)>=1600):
            flush();batch=[];size=0
        batch.append(o);size+=o['n']
    flush()
    save(info,dict(stage=stage,created=now(),packs=packs,closed_attempts=sorted(p.name for p in rows if p.name not in previous),**BG))
    print(stage,len(packs),sum(x['bytes'] for x in packs),flush=True)

def pack(key):
    if not re.fullmatch(r'(?:duplicates|process|followup)-[0-9]{4}',key):raise ValueError(key)
    return read(CTL/key.split('-')[0]/(key+'.json'))

def open_inodes():
    live=set()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    target=os.readlink(fd)
                    if target.startswith((str(RAW)+'/',str(PERSONAL)+'/')):
                        st=fd.stat();live.add((st.st_dev,st.st_ino))
                except OSError:pass
        except OSError:pass
    return live

def apply(key,proof):
    items=pack(key)
    if (proof['serial']!='6063F4DA' or proof['pack']!=key or not proof['h_readback'] or
            proof['objects']!=[[x['h'],x['n']] for x in items]):raise ValueError('H proof mismatch')
    dest=CTL/'receipts';dest.mkdir(parents=True,exist_ok=True)
    with (dest/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        done=dest/(key+'.json')
        if done.exists():return read(done)
        save(dest/(key+'-H-proof.json'),proof)
        rows=closed();live=open_inodes()
        j=dest/(key+'-actions.jsonl');record={}
        if j.exists():
            for line in j.open():
                if line.endswith('\n'):
                    entry=json.loads(line);record[entry['path']]=entry
        before=free()
        with j.open('a',buffering=1) as events:
            def record_one(path,action,n):
                event=dict(path=str(path),action=action,bytes=n)
                events.write(json.dumps(event,separators=(',',':'))+'\n');record[str(path)]=event
            for o in items:
                src=Path(o['source']);expected=o['h'];size=o['n']
                if o['links']:
                    where(src,rows)
                    if src.is_symlink() or not src.is_file() or sha(src)!=expected:raise ValueError('Anchor changed '+str(src))
                for action,names in [('linked',o['links']),('removed',o['drops'])]:
                    for name in names:
                        if name in record:continue
                        p=Path(name);receipt,rel=where(p,rows)
                        if action=='removed' and not (PROCESS.fullmatch(rel) or receipt['status']=='technical_interruption'):
                            raise ValueError('File is used by future replay '+name)
                        if not p.exists():
                            record_one(p,'already_absent',size);continue
                        if p.is_symlink():raise ValueError('Source changed type '+name)
                        st=p.stat();ino=(st.st_dev,st.st_ino)
                        if ino in live:
                            record_one(p,'retained_open',size);continue
                        if action=='linked' and p.samefile(src):
                            record_one(p,'already_linked',size);continue
                        if st.st_nlink!=1 or st.st_size!=size or sha(p)!=expected:
                            raise ValueError('Version changed '+name)
                        current=p.stat()
                        if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns):raise ValueError('Source changed '+name)
                        if action=='linked':
                            tmp=p.with_name('.w58-H-link-'+uuid.uuid4().hex)
                            os.link(src,tmp)
                            try:os.replace(tmp,p)
                            finally:tmp.unlink(missing_ok=True)
                            if not p.samefile(src):raise RuntimeError('Hard link did not persist')
                        else:p.unlink()
                        record_one(p,action,size)
        counts=defaultdict(lambda:[0,0])
        for x in record.values():
            c=counts[x['action']];c[0]+=1;c[1]+=x['bytes']
        result=dict(pack=key,at=now(),actions=dict(counts),free_before=before,free_after=free(),**BG)
        save(done,result);return result

def main():
    action=sys.argv[1]
    if action=='prepare':
        stage=sys.argv[2]
        if stage not in ('duplicates','process','followup'):raise ValueError(stage)
        prepare(stage);return
    if action=='index':
        if sys.argv[2] not in ('duplicates','process','followup'):raise ValueError(sys.argv[2])
        p=CTL/sys.argv[2]/'index.json'
        print(json.dumps(read(p) if p.is_file() else None));return
    key=sys.argv[2];items=pack(key)
    if action=='manifest':print(json.dumps(items,separators=(',',':')))
    elif action=='stream':
        needed=set(json.load(sys.stdin))
        for o in items:
            if o['h'] in needed:
                with open(o['source'],'rb') as f:shutil.copyfileobj(f,sys.stdout.buffer,1024**2)
        sys.stdout.buffer.flush()
    elif action=='apply':print(json.dumps(apply(key,json.load(sys.stdin))))
    else:raise ValueError(action)
if __name__=='__main__':main()
