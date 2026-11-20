"""Read selected W55 published ZIP members by HTTP ranges; no robot simulation.

Published sources SHA/size are checked where present. A E300 acquired a sources
manifest in commit a1501926; an earlier snapshot lacking it uses CRC/size only.
Remote asset digest is recorded, but the full ZIP is not downloaded or hashed.
"""
from pathlib import Path
from urllib.request import Request, urlopen
import argparse, concurrent.futures, hashlib, io, json, re, struct, threading, zipfile, zlib

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--inventory',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--workers',type=int,default=4)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
R=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/delivery'
inventory=json.loads(a.inventory.read_text());assets={x['name']:x for x in inventory['assets']}

class Remote(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.bytes=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,n,w=0):self.pos=n if w==0 else self.pos+n if w==1 else self.size+n;return self.pos
 def read(self,n=-1):
  n=min(self.size-self.pos,n if n>=0 else self.size);start=self.pos
  if n<=0:return b''
  data=self.fetch(start,start+n);self.pos+=n;return data
 def fetch(self,start,end):
  for attempt in range(3):
   try:
    with urlopen(Request(self.url,headers={'Range':f'bytes={start}-{end-1}'}),timeout=45) as r:
     if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end-1}/{self.size}':raise IOError('Range response mismatch')
     self.url=r.geturl();data=r.read()
    if len(data)!=end-start:raise IOError('Incomplete response')
    self.bytes+=len(data);return data
   except Exception:
    if attempt==2:raise

def select(member,batch):
 b=Path(member).name
 if b in ['trajectory.npz','resources.jsonl','stage.json','cap-decisions.json','pann-action-dependence.json','contact-events.jsonl','mpc-solves.jsonl','solver-failure.json','w54-n-execution.jsonl','w54-n-context.jsonl','w54-n-events.jsonl','w54-n-recording.json','progress.json','live.json','video-asset.json','G1-project-go2.xml','mujoco-warnings.json'] or b.endswith('-runner.log'):return True
 if b.startswith('physical-') and 'e300' not in batch:return True
 if re.fullmatch(r'(control|drift)-\d+\.npz',b):
  n=int(b.split('-')[1].split('.')[0])
  return n in [1,2,3,35,36,37,38,39,40,41,42,43,44,45,50,59,60,61] if 'e300' not in batch else n in [1,2,3,40,50,100,150,200,250,299,300,301]
 if re.fullmatch(r'mpc-\d+\.npz',b):
  n=int(b.split('-')[1].split('.')[0]);return n in [1,1750,1800,1850,1900,1950,2000,2020,2040,2060,2080,2100,2120,2138,2150,2200,2250,2500,2950,3000,14950,15000]
 return False

tasks=[]
known_batches=['f-a-r53-e60','f-b-r53-e60','f-a-r55-e60','f-b-r55-e60','n-a-r54-e30']
if (R/'f-a-r55-e300/sources.json').exists():known_batches.append('f-a-r55-e300')
for batch in known_batches:
 rows=json.loads((R/batch/'sources.json').read_text());groups={}
 for row in rows:
  if row.get('member') and select(row['member'],batch):groups.setdefault(row['asset'],[]).append(row)
 for name,rows2 in groups.items():tasks.append((batch,assets[name],rows2))
for name,meta in assets.items():
 if 'f-a-r55-e300' not in known_batches and name.startswith('w55-f-a-r55-e300-originals-') and name.endswith('.zip'):tasks.append(('f-a-r55-e300',meta,None))

lock=threading.Lock();all_rows=[];archives=[]
def work(batch,meta,rows):
 remote=Remote(meta['url'],meta['size']);done=[]
 with zipfile.ZipFile(remote) as z:
  info=sorted(z.infolist(),key=lambda x:x.header_offset);ends={v.filename:info[i+1].header_offset if i+1<len(info) else z.start_dir for i,v in enumerate(info)};by={x.filename:x for x in info}
  if rows is None:rows=[dict(member=x.filename,size=x.file_size) for x in info if select(x.filename,batch)]
  selected=[]
  for row in rows:
   inf=by[row['member']];path=a.output/batch/row['member']
   if path.exists():
    data=path.read_bytes();valid=len(data)==row['size'] and (zlib.crc32(data)&0xffffffff)==inf.CRC and (not row.get('sha256') or hashlib.sha256(data).hexdigest()==row['sha256'])
    if valid:done.append(dict(batch=batch,archive=meta['name'],member=row['member'],path=str(path),size=len(data),sha256=hashlib.sha256(data).hexdigest(),published_member_sha256=row.get('sha256'),zip_crc32_verified=True,source_sha_verified=bool(row.get('sha256'))));continue
   selected.append((inf,row,path))
  groups=[]
  for inf,row,path in selected:
   st,en=inf.header_offset,ends[inf.filename]
   if groups and st-groups[-1][1]<8192 and en-groups[-1][0]<128*1024*1024:groups[-1][1]=en;groups[-1][2].append((inf,row,path))
   else:groups.append([st,en,[(inf,row,path)]])
  for st,en,members in groups:
   buf=remote.fetch(st,en)
   for inf,row,path in members:
    off=inf.header_offset-st;head=struct.unpack_from('<IHHHHHIIIHH',buf,off);assert head[0]==0x04034b50;off+=30+head[9]+head[10];comp=buf[off:off+inf.compress_size]
    data=zlib.decompress(comp,-15) if inf.compress_type==8 else comp if inf.compress_type==0 else None
    if data is None:raise ValueError('Unsupported compression')
    assert len(data)==row['size'];assert (zlib.crc32(data)&0xffffffff)==inf.CRC
    sha=hashlib.sha256(data).hexdigest()
    if row.get('sha256'):assert sha==row['sha256']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    done.append(dict(batch=batch,archive=meta['name'],member=row['member'],path=str(path),size=len(data),sha256=sha,published_member_sha256=row.get('sha256'),zip_crc32_verified=True,source_sha_verified=bool(row.get('sha256'))))
 with lock:
  all_rows.extend(done);archives.append(dict(batch=batch,archive=meta['name'],selected_members=len(done),archive_members=len(info),remote_size=meta['size'],remote_published_sha256=meta.get('sha256'),bytes_downloaded=remote.bytes,full_zip_hash_verified=False))
  (a.output/'original-ledger.json').write_text(json.dumps(dict(files=all_rows,archives=archives),indent=2)+'\n')
  print(json.dumps(dict(batch=batch,archive=meta['name'],selected_members=len(done),bytes_downloaded=remote.bytes)),flush=True)

print(json.dumps(dict(tasks=len(tasks))),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
 futures=[pool.submit(work,*t) for t in tasks]
 for f in concurrent.futures.as_completed(futures):f.result()
print('COMPLETE',flush=True)
