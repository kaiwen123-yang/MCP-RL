#!/usr/bin/env python3
"""Read selected published W42 CPU evidence; no simulation or training."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,urllib.request
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
BASE=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
WANTED={'planner-records.jsonl','planner-calls.jsonl','reference-packets.json','trajectory.npz','action-execution.jsonl','accepted-reference-gain-H-plan000221.npz','accepted-reference-gain-H-plan001516.npz'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 selected=[]
 for job in sorted((BASE/'jobs').glob('*w42-*')):
  if not job.is_dir():continue
  evidence=json.loads((job/'diagnostics/evidence.json').read_text());manifest=json.loads((job/'diagnostics/manifest.json').read_text());lookup={x['name']:x for x in manifest}
  for f in evidence['files']:
   if '-eval-' not in (f.get('attempt') or '') or Path(f['source_path']).name not in WANTED:continue
   assert len(f['parts'])==1,f['source_path']
   part=f['parts'][0];m=lookup[part['name']]
   selected.append(dict(stream=evidence['stream_id'],attempt=f['attempt'],source_path=f['source_path'],source_size=f['source_size'],codec=f['codec'],size=part['size'],url='https://github.com/kaiwen123-yang/MCP-RL/releases/download/'+m['release_tag']+'/'+m['name'],destination=str(args.output/evidence['stream_id']/Path(part['path']).name)))
 def get(f):
  p=Path(f['destination']);p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists() or p.stat().st_size!=f['size']:
   with urllib.request.urlopen(f['url'],timeout=60) as r,p.with_suffix(p.suffix+'.part').open('wb') as out:
    while block:=r.read(1024*1024):out.write(block)
   p.with_suffix(p.suffix+'.part').replace(p)
  assert p.stat().st_size==f['size'],(p,p.stat().st_size,f['size'])
  return {**f,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':'read'}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  results=list(pool.map(get,selected))
 (HERE/'cpu-raw-read-coverage.json').write_text(json.dumps({'status':'complete','source_commit':'c4b3f9adc698b15de5c94ef952e2884f3abfec62','files':results,'compressed_bytes':sum(f['size'] for f in results),'source_bytes':sum(f['source_size'] for f in results)},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'files':len(results),'compressed_bytes':sum(f['size'] for f in results)}))
if __name__=='__main__':main()
