"""Receive the finite W58 process archive on the named H volume, then release each verified batch on D57."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
import collections
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile

ROOT=Path('H:/MPC+RL')
CAT=ROOT/'00_catalog/migration-20261125-w58-process'
PAY=ROOT/'40_support_evidence/20261125-w58-process'
INC=ROOT/'90_incoming/migration-20261125-w58-process'
OPS=ROOT/'99_transfer_logs/migration-20261125-w58-process'
OLD=[ROOT/'00_catalog'/n/'archive-index.sqlite' for n in ('migration-20261124-process','migration-20261123','migration-20261122')]
SSH=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','Compression=no','autodl-d57']
TOOL='taskset -c 20,21 ionice -c3 nice -n 10 /root/miniconda3/bin/python -u /root/autodl-tmp/MCP-RL-w58/scripts/storage_d57_H_20261125/d57.py'
SH=timezone(timedelta(hours=8));LOCK=Lock()
BG=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
def now():return datetime.now(SH).isoformat(timespec='seconds')
def media():
    n=ctypes.create_unicode_buffer(261);s=wintypes.DWORD()
    if not ctypes.windll.kernel32.GetVolumeInformationW('H:\\',n,len(n),ctypes.byref(s),None,None,None,0):
        raise OSError('H volume unavailable')
    if (n.value,s.value)!=('AI_DATA_8TB',0x6063F4DA):raise RuntimeError('H volume identity differs')
def atomic(p,v):
    tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    os.replace(tmp,p)
def call(op,key,data=None):
    cmd=[*SSH,TOOL+' '+op+' '+key]
    cp=subprocess.run(cmd,input=json.dumps(data,separators=(',',':')).encode() if data is not None else b'',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if cp.returncode:raise RuntimeError(op+' '+key+' '+cp.stderr.decode(errors='replace')[-700:])
    return json.loads(cp.stdout)
def digest_stream(f):
    return hashlib.file_digest(f,'sha256').hexdigest()
def db():
    return sqlite3.connect(CAT/'archive-index.sqlite',timeout=90)
def init_db():
    with db() as c:
        c.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS objects(h TEXT PRIMARY KEY,n INTEGER,archive TEXT,member TEXT);
        CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY,path TEXT UNIQUE,h TEXT,status TEXT);
        CREATE TABLE IF NOT EXISTS aliases(path TEXT PRIMARY KEY,target TEXT);
        CREATE TABLE IF NOT EXISTS packs(pack TEXT PRIMARY KEY,at TEXT,objects INTEGER,newbytes INTEGER,linked INTEGER,removed INTEGER,linkedbytes INTEGER,removedbytes INTEGER,reused INTEGER);
        CREATE VIEW IF NOT EXISTS archive_lookup AS SELECT f.id AS id,f.path AS server_path,o.n AS bytes,f.h AS sha256,o.archive AS archive_relative_to_H_MPC_RL,o.member AS zip_member,'file' AS kind,NULL AS link,f.status AS server_status,1 AS archive_verified FROM files f JOIN objects o ON f.h=o.h;
        ''')
def stored(items):
    # Verify matching members from the existing H catalogs before reusing their receipts.
    wanted={x['h']:x['n'] for x in items}
    choices={}
    for source in [CAT/'archive-index.sqlite',*OLD]:
        if not source.is_file():continue
        remaining=[h for h in wanted if h not in choices]
        if not remaining:break
        with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=90) as c:
            if source==CAT/'archive-index.sqlite':query='SELECT h,n,archive,member FROM objects WHERE h IN ('
            elif 'migration-20261122' in source.parts:query='SELECT c.sha256,c.bytes,a.path,c.member FROM contents c JOIN archives a ON c.archive_id=a.id WHERE c.sha256 IN ('
            else:query='SELECT o.h,o.n,o.archive,o.member FROM objects o JOIN archives a ON o.archive=a.path WHERE a.ok=1 AND o.h IN ('
            for j in range(0,len(remaining),400):
                keys=remaining[j:j+400]
                for h,n,archive,member in c.execute(query+','.join('?' for _ in keys)+')',keys):
                    if archive and n==wanted[h]:choices[h]=(archive,member)
    groups=collections.defaultdict(list)
    for h,(archive,member) in choices.items():groups[archive].append((h,member))
    checked={}
    for archive,names in groups.items():
        p=ROOT/archive
        with ExitStack() as st:
            z=st.enter_context(zipfile.ZipFile(p)) if any(member is not None for _,member in names) else None
            for h,member in names:
                with (z.open(member) if member is not None else p.open('rb')) as f:
                    if (z.getinfo(member).file_size if member is not None else p.stat().st_size)!=wanted[h] or digest_stream(f)!=h:
                        raise RuntimeError('Existing H member differs '+h)
                checked[h]=(archive,member)
    return checked

def incoming(key,items,old):
    todo=[o for o in items if o['h'] not in old]
    if not todo:return {},0,None
    final=PAY/(key+'.zip');temp=INC/(key+'.zip.part')
    content={x['h']:x['n'] for x in todo}
    if not final.exists():
        cmd=[*SSH,TOOL+' stream '+key]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as child:
            child.stdin.write(json.dumps(list(content)).encode());child.stdin.close()
            with temp.open('wb') as out:
                with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                    for h,n in content.items():
                        remaining=n; dig=hashlib.sha256()
                        with z.open(h,'w',force_zip64=True) as dest:
                            while remaining:
                                b=child.stdout.read(min(1024**2,remaining))
                                if not b:raise IOError('D57 stream ended early '+key+' '+child.stderr.read().decode(errors='replace')[-500:])
                                dest.write(b);dig.update(b);remaining-=len(b)
                        if dig.hexdigest()!=h:raise IOError('Source member differs '+h)
                out.flush();os.fsync(out.fileno())
            if child.stdout.read(1) or child.wait()!=0:raise IOError('Unsuccessful D57 stream '+key)
        media()
        # Keep the file in incoming until the H drive has been read back.
        current=temp
    else:current=final
    with zipfile.ZipFile(current) as z:
        if set(z.namelist())!=set(content):raise IOError('H pack membership changed '+key)
        for h,n in content.items():
            with z.open(h) as f:
                if z.getinfo(h).file_size!=n or digest_stream(f)!=h:raise IOError('H member verification failed '+h)
    with current.open('rb') as f:bundle_hash=digest_stream(f)
    media()
    if current!=final:os.replace(current,final)
    rel=final.relative_to(ROOT).as_posix()
    return {h:(rel,h) for h in content},sum(content.values()),dict(path=rel,size=final.stat().st_size,sha256=bundle_hash)

def progress(c):
    row=c.execute('select count(*),coalesce(sum(newbytes),0),coalesce(sum(linked),0),coalesce(sum(removed),0),coalesce(sum(linkedbytes),0),coalesce(sum(removedbytes),0),coalesce(sum(reused),0) from packs').fetchone()
    return dict(zip(('packs','new_payload_bytes','duplicate_paths_linked','process_files_removed','duplicate_bytes','removed_bytes','previously_archived_objects'),row))
def receive(key):
    media()
    with db() as c:
        if c.execute('select 1 from packs where pack=?',(key,)).fetchone():return
    items=call('manifest',key)
    # A verified local pack may already be indexed after a previous receiver stopped before D57 acknowledged it.
    old=stored(items)
    new,newbytes,bundle=incoming(key,items,old);locations={**old,**new}
    proof=dict(serial='6063F4DA',pack=key,h_readback=True,verified_at=now(),bundle=bundle,objects=[[o['h'],o['n']] for o in items],**BG)
    with LOCK,db() as c:
        for o in items:
            a,m=locations[o['h']]
            c.execute('insert or ignore into objects values(?,?,?,?)',(o['h'],o['n'],a,m))
            paths={o['source'],*o['links'],*o['drops']}
            for p in paths:
                c.execute('insert or ignore into files(path,h,status) values(?,?,?)',(p,o['h'],'H_verified'))
                prefix='/autodl-fs/data/MCP-RL/W58/originals/'
                if p.startswith(prefix):
                    ep=p[len(prefix):].split('/')[0]
                    src='/root/autodl-tmp/MCP-RL-w58/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/'+ep
                    c.execute('insert or ignore into aliases values(?,?)',(src,prefix+ep))
        c.commit();atomic(OPS/(key+'-H.json'),proof)
    media()
    receipt=call('apply',key,proof)
    actions=receipt['actions'];cl,bl=actions.get('linked',[0,0]);cr,br=actions.get('removed',[0,0])
    with LOCK,db() as c:
        for o in items:
            for name in o['links']:c.execute("update files set status='H_verified_D57_link_batch_receipt' where path=?",(name,))
            for name in o['drops']:c.execute("update files set status='H_verified_D57_retirement_batch_receipt' where path=?",(name,))
        c.execute('insert or replace into packs values(?,?,?,?,?,?,?,?,?)',(key,now(),len(items),newbytes,cl,cr,bl,br,len(old)));c.commit()
        atomic(OPS/(key+'-D57.json'),receipt)
        value=dict(at=now(),status='running',last=key,**progress(c),free_D57=receipt['free_after'],**BG)
        atomic(CAT/'current.json',value)
        print(now(),key,'H verified',len(items),'new MiB',round(newbytes/2**20,1),'released logical GiB',round((value['duplicate_bytes']+value['removed_bytes'])/2**30,2),'volumes',receipt['free_after'],flush=True)

def main():
    import msvcrt
    media()
    for path in [CAT,PAY,INC,OPS]:path.mkdir(parents=True,exist_ok=True)
    lock=(OPS/'receiver.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    init_db()
    reader=ROOT/'00_catalog/migration-20261124-process'/('\u8bfb\u53d6\u539f\u4ef6.py')
    target=CAT/reader.name
    if not target.exists():shutil.copyfile(reader,target)
    for stage in ['duplicates','process']:
        index=call('index',stage)
        while index is None:
            time.sleep(4);index=call('index',stage)
        atomic(CAT/(stage+'-plan.json'),index)
        print(now(),stage,len(index['packs']),'packs',flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for _ in pool.map(receive,[x['id'] for x in index['packs']]):pass
    with db() as c:
        final=dict(at=now(),status='completed',**progress(c),H_root=str(ROOT),**BG)
        c.execute('pragma wal_checkpoint(full)')
    atomic(CAT/'current.json',final);atomic(CAT/'final.json',final)
    print(now(),'COMPLETE',json.dumps(final),flush=True)
if __name__=='__main__':main()
