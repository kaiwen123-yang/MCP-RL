"""Snapshot D57 project roots without following aliases; retain live and nonregular objects in the catalog."""
import concurrent.futures as cf
import fcntl
import json
import os
from pathlib import Path
import queue
import sqlite3
import stat
import subprocess
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

BASE=Path('/root/autodl-tmp')
ROOT=BASE/'MCP-RL'
CTL=Path(__file__).resolve().parent
PERSONAL=Path('/autodl-fs/data/MCP-RL')
TZ=ZoneInfo('Asia/Shanghai')
BG=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
def now():return datetime.now(TZ).isoformat(timespec='seconds')
def save(name,obj):
    p=CTL/name;tmp=p.with_name(p.name+f'.{os.getpid()}.part')
    tmp.write_text(json.dumps(obj,ensure_ascii=True,indent=2)+'\n');os.replace(tmp,p)

def current_roots():
    out=subprocess.check_output(['git','-C',str(ROOT),'worktree','list','--porcelain'])
    repos=[]
    for field in out.splitlines():
        if not field.startswith(b'worktree '):continue
        p=Path(os.fsdecode(field[9:]))
        if p.parent!=BASE or (p.name!='MCP-RL' and not p.name.startswith('MCP-RL-')) or p.resolve()!=p:
            raise ValueError('Inspect separately registered worktree '+str(p))
        repos.append(p)
    extras=[p for p in BASE.iterdir() if p.is_dir() and not p.is_symlink() and p.name.startswith('MCP-RL-') and p not in repos]
    return repos,sorted(extras)+[PERSONAL]

def opened():
    out=set()
    for pid in Path('/proc').iterdir():
        if not pid.name.isdecimal():continue
        try:
            for fd in (pid/'fd').iterdir():
                try:target=os.readlink(fd)
                except OSError:continue
                if target.startswith((str(ROOT),str(PERSONAL))):out.add(target)
        except OSError:continue
    return out

def sid(st):
    return (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_dev,st.st_ino,st.st_nlink,st.st_blocks*512,st.st_mode)

def main():
    lock=(CTL/'inventory.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (CTL/'inventory.done.json').exists():return
    dbpath=CTL/'inventory.sqlite'
    if dbpath.exists():raise RuntimeError('Existing unfinished inventory: inspect before resuming')
    repos,extra=current_roots();roots=repos+extra
    start=time.time();live=opened()
    manifest=dict(start_time=now(),start_unix=start,roots=list(map(str,roots)),registered=list(map(str,repos)),
                  excluded_foreign_mounts=[],observed_open_count=len(live),**BG)
    save('roots.json',manifest)
    db=sqlite3.connect(dbpath);db.executescript("""
    PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
    CREATE TABLE files(id INTEGER PRIMARY KEY,path TEXT UNIQUE NOT NULL,root TEXT,kind TEXT,size INTEGER,mtime INTEGER,ctime INTEGER,dev INTEGER,ino INTEGER,nlink INTEGER,allocated INTEGER,mode INTEGER,tracked INTEGER,link TEXT,opened INTEGER,digest TEXT,state TEXT NOT NULL DEFAULT 'snapshot');
    CREATE TABLE issues(path TEXT,reason TEXT);
    """)
    tracked={}
    for p in repos:
        data=subprocess.check_output(['git','-C',str(p),'ls-files','-z'])
        tracked[str(p)]={os.fsdecode(e) for e in data.split(b'\0') if e}
    work=queue.Queue();results=queue.Queue(48);errors=[]
    for p in roots:work.put((str(p),str(p),os.lstat(p).st_dev))
    def worker():
        while True:
            item=work.get()
            if item is None:work.task_done();break
            dr,root,dev=item;rows=[];issues=[]
            try:
                with os.scandir(dr) as es:
                    for e in es:
                        try:st=e.stat(follow_symlinks=False)
                        except (FileNotFoundError,PermissionError) as ex:issues.append((e.path,type(ex).__name__));continue
                        mode=st.st_mode
                        kind=('link' if stat.S_ISLNK(mode) else 'dir' if stat.S_ISDIR(mode) else 'file' if stat.S_ISREG(mode) else 'special')
                        link=None
                        if kind=='link':
                            try:link=os.readlink(e.path)
                            except FileNotFoundError:issues.append((e.path,'link changed during capture'));continue
                        rows.append((e.path,root,kind,*sid(st),int(e.path[len(root)+1:] in tracked.get(root,())),link,int(e.path in live)))
                        if kind=='dir':
                            if st.st_dev==dev:work.put((e.path,root,dev))
                            else:issues.append((e.path,'separate mounted filesystem; no traversal'))
                        if len(rows)>=750:results.put((rows,[]));rows=[]
            except (FileNotFoundError,PermissionError) as ex:issues.append((dr,type(ex).__name__))
            except Exception as ex:errors.append(repr(ex));issues.append((dr,repr(ex)))
            finally:
                if rows or issues:results.put((rows,issues))
                work.task_done()
    threads=[threading.Thread(target=worker,daemon=True) for _ in range(8)]
    for t in threads:t.start()
    def finish():
        work.join()
        for t in threads:work.put(None)
        for t in threads:t.join()
        results.put(None)
    threading.Thread(target=finish,daemon=True).start()
    clock=time.monotonic();count=0
    while True:
        item=results.get()
        if item is None:break
        rows,issues=item
        db.executemany('INSERT INTO files(path,root,kind,size,mtime,ctime,dev,ino,nlink,allocated,mode,tracked,link,opened) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',rows)
        db.executemany('INSERT INTO issues VALUES(?,?)',issues);count+=len(rows)
        if time.monotonic()-clock>10:
            db.commit();save('scan-progress.json',dict(at=now(),entries=count,pending_dirs=work.qsize(),**BG));clock=time.monotonic()
    db.commit()
    if errors:raise RuntimeError(str(errors[:5]))
    db.executescript('CREATE INDEX files_inode ON files(dev,ino); CREATE INDEX files_root ON files(root,kind); CREATE INDEX files_size ON files(size);')
    stats=db.execute("SELECT root,count(*),sum(size),sum(allocated) FROM files WHERE kind='file' GROUP BY root ORDER BY sum(size) DESC").fetchall()
    special=db.execute("SELECT kind,count(*) FROM files GROUP BY kind").fetchall()
    trouble=db.execute("SELECT reason,count(*) FROM issues GROUP BY reason").fetchall()
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    save('inventory.done.json',dict(finished_at=now(),total_entries=count,logical_by_root=stats,kinds=special,issues=trouble,**BG))
    print(now(),count,special,trouble,flush=True)
if __name__=='__main__':main()
