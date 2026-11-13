#!/usr/bin/env python3
"""Serialize compact W49 publication, then incrementally deliver generated originals."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
SMALL=ROOT/REL
LARGE=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def compact(paths,message):
 with (SMALL/'compact-finish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with (SMALL/'index.lock').open('a') as idx:
   fcntl.flock(idx,fcntl.LOCK_EX)
   args=[str(p.relative_to(ROOT)) if p.is_absolute() else str(p) for p in paths]
   args.append(str(REL/'index.json'))
   subprocess.run(['git','add','--',*args],cwd=ROOT,check=True)
   changed=subprocess.run(['git','diff','--cached','--quiet','--',*args],cwd=ROOT).returncode
   if changed==1:subprocess.run(['git','commit','--only','-m',message,'--',*args],cwd=ROOT,check=True)
  subprocess.run(['git','fetch','origin','main'],cwd=ROOT,check=True,timeout=120)
  subprocess.run(['git','merge','--no-edit','origin/main'],cwd=ROOT,check=True)
  subprocess.run(['git','-c','pack.window=0','push','--no-thin','origin','HEAD:main'],cwd=ROOT,check=True,timeout=120)
  return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def deliver(name,roots):
 transfer=LARGE/'w49-delivery'/name;transfer.mkdir(parents=True,exist_ok=True)
 destination=SMALL/'w49'/'delivery'/name;destination.mkdir(parents=True,exist_ok=True)
 catalog=[];manifest=[];paths=[]
 for root in roots:
  paths.extend([root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file()))
 # Preserve generated originals, including the explicitly current transport buffers.
 groups=[];group=[];size=0
 for p in paths:
  if group and size+p.stat().st_size>256*1024**2:groups.append(group);group=[];size=0
  group.append(p);size+=p.stat().st_size
 if group:groups.append(group)
 for i,group in enumerate(groups,1):
  asset=f'w49-{name}-originals-{i:04d}.zip';target=transfer/asset
  if target.exists():raise ValueError('Keep existing delivery; use its saved manifest to resume publication')
  with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
   for j,p in enumerate(group):
    member=f'{j:05d}/{p.name}';h=hashlib.sha256()
    with p.open('rb') as source,z.open(member,'w',force_zip64=True) as out:
     while b:=source.read(8*1024*1024):h.update(b);out.write(b)
    catalog.append(dict(source_path=str(p),size=p.stat().st_size,sha256=h.hexdigest(),archive=asset,member=member))
  manifest.append(dict(path=str(target.relative_to(ROOT)),name=asset,category='W49_requested_originals'))
 write(destination/'sources.json',catalog);write(destination/'manifest.json',manifest)
 compact([destination],f'Record W49 {name} generated original manifest')
 return resume_delivery(name)
def resume_delivery(name):
 transfer=LARGE/'w49-delivery'/name;destination=SMALL/'w49'/'delivery'/name
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 result=dict(status='publication_pending',deadline_unix_s=None,server_originals_retained=True,source_commit=commit)
 write(destination/'delivery.json',result)
 for action in ['upload','finish']:
  command=[sys.executable,'scripts/publish_s1_go2_r12.py',action,'--server-auth',
   '--manifest',str(destination/'manifest.json'),'--job-name','w49-'+name,
   '--source-commit',commit,'--upload-workers','1','--no-wall-deadline']
  with (transfer/'publisher.log').open('a') as log:
   code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
  result[action+'_exit_code']=code
  if code:break
 result['status']='published' if result.get('finish_exit_code')==0 else 'publication_pending'
 result['finished_unix_s']=time.time();write(destination/'delivery.json',result)
 compact([destination],f'Record W49 {name} original delivery outcome')
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--source',action='append',type=Path);p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.resume:resume_delivery(a.name)
 elif a.source:deliver(a.name,a.source)
 else:p.error('--source or --resume is required')
if __name__=='__main__':main()
