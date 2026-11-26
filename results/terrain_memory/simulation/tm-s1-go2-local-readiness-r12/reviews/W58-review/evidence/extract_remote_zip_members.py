"""Read selected public-release ZIP members using HTTP Range; validates ZIP CRC.
Whole-asset SHA is manifest metadata, not claimed independently verified by this script.
"""
import pathlib,io,json,urllib.request,zipfile,concurrent.futures,argparse,hashlib
class RemoteFile(io.RawIOBase):
 def __init__(self,a):self.a=a;self.pos=0;self.cache=[];self.transferred=0;self.readpart(max(0,a['size']-500000),a['size'])
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,p,whence=0):self.pos=p if whence==0 else self.pos+p if whence==1 else self.a['size']+p;return self.pos
 def readpart(self,st,en):
  if en<=st:return b''
  for x,y,b in self.cache:
   if x<=st and en<=y:return b[st-x:en-x]
  req=urllib.request.Request(self.a['url'],headers={'Range':f'bytes={st}-{en-1}'})
  with urllib.request.urlopen(req) as res:
   if res.status!=206:raise RuntimeError('Range unsupported, refusing full asset')
   b=res.read();assert len(b)==en-st,(len(b),en-st)
  self.transferred+=len(b);self.cache.append((st,en,b));return b
 def read(self,size=-1):
  if size<0:size=self.a['size']-self.pos
  b=self.readpart(self.pos,min(self.a['size'],self.pos+size));self.pos+=len(b);return b
p=argparse.ArgumentParser();p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
assets=[a for a in json.loads(args.manifest.read_text())['assets'] if a['name'].endswith('.zip')]
fnames=['episode/material-events.jsonl','episode/w58-target-events.jsonl']; records=[]
def work(a):
 f=RemoteFile(a); z=zipfile.ZipFile(f);wanted=[n for n in z.namelist() if n in fnames or n.startswith('receipt/')and n.endswith('.json')and not n.endswith('-resolved.json')];rows=[]
 for n in wanted:
  b=z.read(n);pp=args.out/n;pp.parent.mkdir(parents=True,exist_ok=True);pp.write_bytes(b);rows.append({'member':n,'size':len(b),'crc32':z.getinfo(n).CRC,'sha256':hashlib.sha256(b).hexdigest()})
 return {'asset':a,'range_bytes_downloaded':f.transferred,'members':rows,'zip_members_total':len(z.namelist())}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for result in pool.map(work,assets):records.append(result);print(result['asset']['name'],result['range_bytes_downloaded'],[x['member'] for x in result['members']],flush=True)
(args.out/'range-source-manifest.json').write_text(json.dumps(records,indent=2))
