"""Share identical XFS extents of completed and published D57 episode files."""
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

BASE = Path('/root/autodl-tmp')
REL = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
WORK = BASE/'MCP-RL-w58/work/w58'
STATE = BASE/'MCP-RL-w58/results'/REL/'w58'
SH = ZoneInfo('Asia/Shanghai')
IOCTL = 0xc0189436  # Linux FIDEDUPERANGE; each destination retains its inode.
REQUEST = struct.Struct('=QQHHIqQQiI')
CHUNK = 8 * 1024**2
PATTERNS = ('material-[0-9]*.npz','physical-[0-9]*.npz','w55-f/control-*.npz',
            'w55-f/drift-*.npz','trajectory.npz','model.mjb')

def now():
    return datetime.now(SH).isoformat(timespec='seconds')

def free():
    fs=os.statvfs(BASE)
    return fs.f_frsize*fs.f_bavail

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').digest()

def share(a,b,n):
    copied=0
    with a.open('rb') as src, b.open('r+b') as dst:
        offset=0
        while offset<n:
            length=min(CHUNK,n-offset)
            data=bytearray(REQUEST.pack(offset,length,1,0,0,dst.fileno(),offset,0,0,0))
            fcntl.ioctl(src.fileno(),IOCTL,data,True)
            _,_,_,_,_,_,_,count,status,_=REQUEST.unpack(data)
            if status or count!=length:
                raise RuntimeError(f'XFS comparison returned {status}, {count}/{length} for {b}')
            copied+=count
            offset+=count
    return copied

def main():
    periods=sys.argv[1:]
    groups=collections.defaultdict(list)
    device=BASE.stat().st_dev
    seen=set()
    count=0
    for code in periods:
        if code not in ('w58','w57'):
            raise ValueError(code)
        root=BASE/('MCP-RL-'+code); small=root/'results'/REL/code
        raw=root/'outputs'/REL
        allowed=[raw.resolve(),(root/'work/w57-raw-scratch').resolve()]
        for receipt in sorted((small/'delivery').glob('*.json')):
            ap=small/'attempts'/receipt.name
            if not ap.is_file():continue
            a=json.loads(ap.read_text()); d=json.loads(receipt.read_text())
            if not a.get('ended_unix_s') or d.get('status')!='verified' or not d.get('assets'):continue
            episode=(raw/receipt.stem).resolve()
            if episode.parent not in allowed or not episode.is_dir() or episode.stat().st_dev!=device:continue
            for pattern in PATTERNS:
                for f in episode.glob(pattern):
                    if f.is_symlink():continue
                    st=f.stat()
                    if st.st_dev!=device or st.st_nlink!=1 or st.st_size<65536:continue
                    if (st.st_dev,st.st_ino) in seen:continue
                    seen.add((st.st_dev,st.st_ino))
                    groups[(str(f.relative_to(episode)),st.st_size)].append(f)
                    count+=1
    before=free();started=now();matched=0;total=0;last=time.monotonic()
    audit=WORK/('shared-extents-'+ '-'.join(periods)+'-20261125.jsonl')
    with audit.open('a',encoding='utf-8',buffering=1) as out:
        for (_,size),paths in sorted(groups.items(),key=lambda kv:-kv[0][1]*len(kv[1])):
            if len(paths)<2:continue
            hashes={}
            for f in paths:
                h=digest(f)
                if h not in hashes:
                    hashes[h]=f
                    continue
                src=hashes[h]; allocated=share(src,f,size)
                out.write(json.dumps([str(src),str(f),h.hex(),allocated])+'\n')
                matched+=1;total+=allocated
            if time.monotonic()-last>=12:
                print(now(),'verified files',matched,'bytes',total,'scratch free',free(),flush=True);last=time.monotonic()
    end=free()
    rec=dict(status='completed',periods=periods,started_shanghai=started,finished_shanghai=now(),
             scope='completed receipts with verified D57 Release assets on project XFS only; per-file SHA-256 then kernel byte comparison; original paths and inodes retained',
             file_count_examined=count,destination_files=matched,kernel_shared_bytes=total,scratch_free_before=before,scratch_free_after=end,
             files_unlinked=0,member_log=str(audit),deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
    receipt=STATE/('closed-extents-20261125-'+'-'.join(periods)+'.json')
    tmp=receipt.with_suffix('.pending');tmp.write_text(json.dumps(rec,indent=2)+'\n');os.replace(tmp,receipt)
    print(json.dumps(rec),flush=True)
if __name__=='__main__':
    main()
