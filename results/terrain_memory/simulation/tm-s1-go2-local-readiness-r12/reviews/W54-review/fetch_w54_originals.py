"""Fetch selected published W54 ZIP members by merged HTTP ranges; no experiment execution."""
from pathlib import Path
from urllib.request import Request,urlopen
import argparse,concurrent.futures,hashlib,io,json,struct,threading,time,zipfile,zlib
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[6]);P.add_argument('--output',type=Path,required=True);P.add_argument('--workers',type=int,default=4);a=P.parse_args()
R=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w54/delivery';a.output.mkdir(parents=True,exist_ok=True)
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
  info=sorted(z.infolist(),key=lambda x:x.header_offset);ends={v.filename:(info[i+1].header_offset if i+1<len(info) else z.start_dir) for i,v in enumerate(info)};by={x.filename:x for x in info};selected=[];done=[]
  for row in rows:
   rel=row['source_path'].split('/tm-s1-go2-local-readiness-r12/',1)[1] if '/tm-s1-go2-local-readiness-r12/' in row['source_path'] else batch+'/'+Path(row['source_path']).name
   p=a.output/rel
   if p.exists() and p.stat().st_size==row['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']:done.append(dict(batch=batch,path=rel,**row,verified=True));continue
   selected.append((by[row['member']],row,rel))
  groups=[]
  for inf,row,rel in sorted(selected,key=lambda x:x[0].header_offset):
   st,en=inf.header_offset,ends[inf.filename]
   if groups and st-groups[-1][1]<8192 and en-groups[-1][0]<=96*1024*1024:groups[-1][1]=en;groups[-1][2].append((inf,row,rel))
   else:groups.append([st,en,[(inf,row,rel)]])
  for st,en,members in groups:
   buf=remote.fetch(st,en)
   for inf,row,rel in members:
    off=inf.header_offset-st;head=struct.unpack_from('<IHHHHHIIIHH',buf,off);assert head[0]==0x04034b50;off+=30+head[9]+head[10];comp=buf[off:off+inf.compress_size]
    data=zlib.decompress(comp,-15) if inf.compress_type==8 else comp if inf.compress_type==0 else None
    if data is None:raise ValueError('Unsupported zip method')
    assert len(data)==row['size'];assert hashlib.sha256(data).hexdigest()==row['sha256'];p=a.output/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);done.append(dict(batch=batch,path=rel,**row,verified=True))
 with lock:
  all_rows.extend(done);archives.append(dict(batch=batch,archive=meta['name'],source_archive_bytes=meta['size'],download_range_bytes=remote.bytes,selected_members=len(done)))
  (a.output/'original-ledger.json').write_text(json.dumps({'files':all_rows,'archives':archives},indent=2)+'\n')
  print(json.dumps({'batch':batch,'archive':meta['name'],'members':len(done),'bytes_downloaded':remote.bytes}),flush=True)
priority=['r52','r51','r50','r48','r44','r49','r47','r46','r45']
tasks=[]
for folder in R.iterdir():
 if not (folder.name.startswith('job-') or folder.name.startswith('c-')):continue
 src=json.loads((folder/'sources.json').read_text());man={x['name']:x for x in json.loads((folder/'manifest.json').read_text())};groups={}
 for row in src:
  if row['source_path'].endswith(('.bin','model.mjb')):continue
  groups.setdefault(row['archive'],[]).append(row)
 for name,rows in groups.items():tasks.append((folder.name,man[name],rows))
tasks.sort(key=lambda t:next((i for i,x in enumerate(priority) if '-a-'+x+'-' in t[0]),20))
print(json.dumps({'archives':len(tasks),'selected_members':sum(len(t[2]) for t in tasks),'selected_source_bytes':sum(x['size'] for t in tasks for x in t[2])}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
 fut=[pool.submit(run,*t) for t in tasks]
 for f in concurrent.futures.as_completed(fut):f.result()
print('COMPLETE',flush=True)
