"""Read-only HTTP Range audit of selected published S1 recordings."""
from pathlib import Path
from urllib.request import Request,urlopen
import argparse,concurrent.futures,hashlib,io,json,struct,threading,zipfile,zlib
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,required=True);P.add_argument('--output',type=Path,required=True);P.add_argument('--workers',type=int,default=2);a=P.parse_args();a.output.mkdir(parents=True,exist_ok=True)
R=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
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
    with urlopen(Request(self.url,headers={'Range':f'bytes={start}-{end-1}'}),timeout=50) as r:
     if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end-1}/{self.size}':raise IOError('Invalid range response')
     self.url=r.geturl();data=r.read()
    if len(data)!=end-start:raise IOError('Incomplete range')
    self.bytes+=len(data);return data
   except Exception:
    if attempt==2:raise
lock=threading.Lock();all_rows=[];archives=[]
def run(batch,meta,rows):
 url='https://github.com/kaiwen123-yang/MCP-RL/releases/download/'+meta['release_tag']+'/'+meta['name'];remote=Remote(url,meta['size'])
 with zipfile.ZipFile(remote) as z:
  info=sorted(z.infolist(),key=lambda x:x.header_offset);ends={v.filename:(info[i+1].header_offset if i+1<len(info) else z.start_dir) for i,v in enumerate(info)};by={x.filename:x for x in info};done=[]
  for row in rows:
   rel=Path(batch)/row['member'];p=(a.output/rel).resolve()
   if not p.is_relative_to(a.output.resolve()):raise ValueError('Unsafe zip member')
   if p.exists() and p.stat().st_size==row['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']:done.append(dict(batch=batch,local_path=str(p),**row,verified=True));continue
   inf=by[row['member']];buf=remote.fetch(inf.header_offset,ends[inf.filename]);head=struct.unpack_from('<IHHHHHIIIHH',buf,0)
   if head[0]!=0x04034b50:raise IOError('Bad local header')
   off=30+head[9]+head[10];comp=buf[off:off+inf.compress_size]
   data=zlib.decompress(comp,-15) if inf.compress_type==8 else comp if inf.compress_type==0 else None
   if data is None:raise ValueError('Unsupported method')
   if len(data)!=row['size'] or hashlib.sha256(data).hexdigest()!=row['sha256'] or zlib.crc32(data)!=inf.CRC:raise IOError('Size/hash/CRC mismatch')
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);done.append(dict(batch=batch,local_path=str(p),**row,verified=True));del data,buf,comp
 with lock:
  all_rows.extend(done);archives.append(dict(batch=batch,archive=meta['name'],source_archive_bytes=meta['size'],download_range_bytes=remote.bytes,selected_members=len(done)))
  (a.output/'evidence-original-ledger.json').write_text(json.dumps({'files':all_rows,'archives':archives},indent=2)+'\n')
  print(json.dumps({'batch':batch,'archive':meta['name'],'members':len(done),'bytes_downloaded':remote.bytes}),flush=True)
tasks=[]
for w,batch in [('w55','f-a-r55-e600'),('w55','f-b-r55-e600'),('w56','f-a-r60-e600'),('w56','f-b-r60-e600')]:
 folder=R/w/'delivery'/batch;src=json.loads((folder/'sources.json').read_text());man={x['name']:x for x in json.loads((folder/'manifest.json').read_text())};groups={}
 for row in src:
  member=row.get('member','');name=Path(member).name
  if name not in ['trajectory.npz','resources.jsonl','G1-project-go2.xml','physical-000.npz'] and not (w=='w56' and name in ['physical-001.npz','physical-015.npz']):continue
  groups.setdefault(row['asset'],[]).append(row)
 for name,rows in groups.items():tasks.append((batch,man[name],rows))
tasks.sort(key=lambda x:0 if any(Path(r['member']).name=='trajectory.npz' for r in x[2]) else 1)
print(json.dumps({'archives':len(tasks),'selected_members':sum(len(t[2]) for t in tasks),'selected_source_bytes':sum(r['size'] for t in tasks for r in t[2])}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
 fut=[pool.submit(run,*t) for t in tasks]
 for f in concurrent.futures.as_completed(fut):f.result()
print('COMPLETE',flush=True)
