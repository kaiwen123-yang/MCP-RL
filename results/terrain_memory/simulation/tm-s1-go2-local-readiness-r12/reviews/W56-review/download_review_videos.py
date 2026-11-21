"""Read-only release-video audit. Uses published receipts; no robot execution."""
import argparse,concurrent.futures,hashlib,json,pathlib,subprocess,urllib.request,threading
p=argparse.ArgumentParser();p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--report',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
assets=json.loads(a.manifest.read_text());rows=[];lock=threading.Lock()
def one(x):
 dest=a.output/pathlib.Path(x['name']).name
 expected=x['remote_digest'].removeprefix('sha256:')
 if not dest.exists() or dest.stat().st_size!=x['size']:
  req=urllib.request.Request(x['url'],headers={'User-Agent':'MCP-RL-W56-review'})
  with urllib.request.urlopen(req,timeout=50) as src,dest.open('wb') as out:
   while True:
    b=src.read(1048576)
    if not b:break
    out.write(b)
 h=hashlib.sha256()
 with dest.open('rb') as src:
  for b in iter(lambda:src.read(1048576),b''):h.update(b)
 assert dest.stat().st_size==x['size'] and h.hexdigest()==expected,x['name']
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size:stream=codec_name,width,height,r_frame_rate,nb_frames','-of','json',str(dest)]))
 dec=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(dest),'-threads','1','-f','null','-'],capture_output=True,text=True)
 row={**x,'local_path':str(dest.resolve()),'download_sha256':h.hexdigest(),'sha256_verified':True,'probe':probe,'decode_returncode':dec.returncode,'decode_stderr':dec.stderr}
 with lock:
  rows.append(row);a.report.write_text(json.dumps(rows,indent=2)+'\n')
  print(json.dumps({'name':x['name'],'size':x['size'],'duration':probe['format']['duration'],'decode':dec.returncode}),flush=True)
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for f in concurrent.futures.as_completed([pool.submit(one,x) for x in assets]): f.result()
print('COMPLETE',len(rows),flush=True)
