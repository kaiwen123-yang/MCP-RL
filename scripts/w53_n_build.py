#!/usr/bin/env python3
"""Compile all N project units against retained pinned dependency objects, -j6."""
import concurrent.futures,hashlib,json,resource,shlex,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];UP=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'
CACHE=UP/'build-autodl';OUT=ROOT/'work/w53-n';SOURCE=OUT/'source';DEST=ROOT/'work/r11-frozen/S5-r42'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def link_runtime_assets():
 # Git stores Menagerie adaptation patches, while the prepared dependency tree
 # owns the actual generated XML and meshes. Existing restored R8 files win.
 base=ROOT/'work/r11-frozen/S5-r41/runtime-tasks';target=SOURCE/'mjpc/tasks';linked=[]
 for source in sorted(base.rglob('*')):
  rel=source.relative_to(base);dest=target/rel
  if source.is_file() and source.suffix not in ('.h','.cc','.cpp','.patch','.py','.md'):
   if not dest.exists():
    dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(source.resolve())
   if dest.is_symlink() and dest.resolve()==source.resolve():
    linked.append(dict(path=str(rel),retained_source=str(source.resolve()),size=source.stat().st_size))
 return linked

def main():
 begin=time.time();usage=resource.getrusage(resource.RUSAGE_CHILDREN);DEST.mkdir(exist_ok=True)
 runtime_assets=link_runtime_assets()
 commands=json.loads((CACHE/'compile_commands.json').read_text());jobs=[];replacements={}
 archive=shlex.split((CACHE/'mjpc/CMakeFiles/libmjpc.dir/link.txt').read_text().splitlines()[0])
 service=shlex.split((CACHE/'mjpc/grpc/CMakeFiles/mjpc_agent_service.dir/link.txt').read_text().splitlines()[0])
 link=shlex.split((CACHE/'mjpc/grpc/CMakeFiles/agent_server.dir/link.txt').read_text())
 targets={str((CACHE/'mjpc'/x).resolve()) for x in archive if x.endswith('.o')}
 targets|={str((CACHE/'mjpc/grpc'/x).resolve()) for x in link+service if x.endswith('.o')}
 for entry in commands:
  cmd=shlex.split(entry['command']);object_path=str((Path(entry['directory'])/cmd[cmd.index('-o')+1]).resolve())
  if object_path not in targets:continue
  rel=Path(entry['file']).relative_to(UP)
  if rel.parts[0]!='mjpc':continue
  src=SOURCE/rel
  if not src.is_file():raise FileNotFoundError(src)
  obj=OUT/'objects'/rel.with_suffix('.cc.o');obj.parent.mkdir(parents=True,exist_ok=True)
  cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(src);cmd.insert(1,'-I'+str(SOURCE))
  jobs.append(dict(command=cmd,cwd=entry['directory'],source=str(rel)));replacements[object_path]=str(obj)
 def compile(j):
  print('compile '+j['source'],flush=True);subprocess.run(j['command'],cwd=j['cwd'],check=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(compile,jobs))
 for command,oldcwd,newname in [(archive,CACHE/'mjpc','libmjpc.a'),(service,CACHE/'mjpc/grpc','libmjpc_agent_service.a')]:
  command[2]=str(OUT/newname)
  (OUT/newname).unlink(missing_ok=True)  # ar qc must create a fresh owned archive, not append stale objects.
  command=[replacements.get(str((oldcwd/x).resolve()),x) if x.endswith('.o') else x for x in command]
  subprocess.run(command,cwd=oldcwd,check=True);subprocess.run(['/usr/bin/gcc-ranlib-11',str(OUT/newname)],check=True)
 link=[str(OUT/'libmjpc_agent_service.a') if x=='../../lib/libmjpc_agent_service.a' else str(OUT/'libmjpc.a') if x=='../../lib/libmjpc.a' else '-flto=6' if x=='-flto' else replacements.get(str((CACHE/'mjpc/grpc'/x).resolve()),x) if x.endswith('.o') else x for x in link]
 link[link.index('-o')+1]=str(DEST/'agent_server');link.insert(1,'-Wl,-rpath,'+str(DEST/'lib'))
 subprocess.run(link,cwd=CACHE/'mjpc/grpc',check=True)
 for name,target in [('mjpc',SOURCE/'mjpc'),('runtime-tasks',SOURCE/'mjpc/tasks')]:
  path=DEST/name
  if not path.exists():path.symlink_to(target,target_is_directory=True)
 (DEST/'lib').mkdir(exist_ok=True);shutil.copy2(CACHE/'lib/libmujoco.so.3.2.3',DEST/'lib/libmujoco.so.3.2.3')
 (DEST/'SOURCE_COMMIT').write_text('e00c47a5adb9856af2e0f24231bb3a60d5be23c4\n')
 specpath=ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r42.json';spec=json.loads(specpath.read_text())
 end=resource.getrusage(resource.RUSAGE_CHILDREN)
 ident=dict(spec,binary_sha256=sha(DEST/'agent_server'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  specification_path=str(specpath.relative_to(ROOT)),specification_sha256=sha(specpath),build_source_root=str(SOURCE),build_directory=str(OUT),
  patch_path='experiments/terrain_memory/s1-go2-r11-S5-r42.patch',patch_sha256=sha(ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r42.patch'),
  build_started_unix_s=begin,build_wall_s=time.time()-begin,build_cpu_s=end.ru_utime+end.ru_stime-usage.ru_utime-usage.ru_stime,
  build_concurrency_cap=6,translation_units=len(jobs),runtime_assets_adaptation=runtime_assets,compile_commands=jobs,link_command=link,cached_dependencies_read_only=True,
  mujoco_library_sha256=sha(DEST/'lib/libmujoco.so.3.2.3'),qualification='original_E30_not_yet_run',
  source_files_sha256={str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'mjpc').rglob('*') if p.suffix in ('.h','.cc')})
 (DEST/'identity.json').write_text(json.dumps(ident,indent=2)+'\n')
 small=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53';small.mkdir(parents=True,exist_ok=True)
 (small/'S5-r42-build.json').write_text(json.dumps(ident,indent=2)+'\n')
 print(json.dumps({k:ident[k] for k in ['binary_sha256','translation_units','build_wall_s','build_cpu_s']}),flush=True)
if __name__=='__main__':main()
