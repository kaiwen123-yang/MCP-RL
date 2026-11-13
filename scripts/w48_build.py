#!/usr/bin/env python3
"""Formal r36 isolated incremental build from retained r35; read-only cached dependencies."""
import concurrent.futures,hashlib,json,shlex,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'work/r11-frozen/S5-r35'
OUT=ROOT/'work/w48-implementation'
DEST=ROOT/'work/r11-frozen/S5-r36'
UP=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'
CACHE=UP/'build-autodl'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 start=time.time();OUT.mkdir(exist_ok=False);DEST.mkdir(exist_ok=False)
 source=OUT/'source';source.mkdir()
 old=json.loads((BASE/'identity.json').read_text())
 assert sha(CACHE/'bin/agent_server')==old['binary_sha256']
 for rel,digest in old['source_files_sha256'].items():assert sha(UP/rel)==digest,rel
 shutil.copytree(BASE/'mjpc',source/'mjpc')
 for rel in ['mjpc/utilities.cc','mjpc/planners/ilqg/planner.cc']:
  path=source/rel;path.write_text(path.read_text())
 patch=ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r36.patch'
 subprocess.run(['patch','-p1','--batch','-i',str(patch)],cwd=source,check=True)
 commands=json.loads((CACHE/'compile_commands.json').read_text())
 suffixes=['mjpc/utilities.cc','mjpc/planners/ilqg/planner.cc'];jobs=[]
 for rel in suffixes:
  entry=next(c for c in commands if c['file']==str(UP/rel))
  cmd=shlex.split(entry['command']);obj=OUT/'objects'/Path(rel).with_suffix('.cc.o');obj.parent.mkdir(parents=True,exist_ok=True)
  cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(source/rel);cmd.insert(1,'-I'+str(source))
  jobs.append(dict(command=cmd,cwd=entry['directory']))
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  list(pool.map(lambda j:subprocess.run(j['command'],cwd=j['cwd'],check=True),jobs))
 archive=shlex.split((CACHE/'mjpc/CMakeFiles/libmjpc.dir/link.txt').read_text().splitlines()[0])
 lib=OUT/'libmjpc.a';archive[2]=str(lib)
 for rel in suffixes:
  archive=[str(OUT/'objects'/Path(rel).with_suffix('.cc.o')) if x=='CMakeFiles/libmjpc.dir/'+rel[5:]+'.o' else x for x in archive]
 subprocess.run(archive,cwd=CACHE/'mjpc',check=True)
 subprocess.run(['/usr/bin/gcc-ranlib-11',str(lib)],check=True)
 link=shlex.split((CACHE/'mjpc/grpc/CMakeFiles/agent_server.dir/link.txt').read_text())
 link[link.index('-o')+1]=str(DEST/'agent_server')
 link=[str(lib) if x=='../../lib/libmjpc.a' else '-flto=6' if x=='-flto' else x for x in link]
 subprocess.run(link,cwd=CACHE/'mjpc/grpc',check=True)
 shutil.copytree(source/'mjpc',DEST/'mjpc')
 for d in ['runtime-tasks','lib']:shutil.copytree(BASE/d,DEST/d)
 shutil.copy2(BASE/'SOURCE_COMMIT',DEST/'SOURCE_COMMIT')
 specpath=ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r36.json';spec=json.loads(specpath.read_text())
 ident=dict(old)
 ident.update(build='S5-r36',based_on='S5-r35',controller_contract_build='S5-r36',prior_binary_sha256=old['binary_sha256'],
  binary_sha256=sha(DEST/'agent_server'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  specification_path=str(specpath.relative_to(ROOT)),specification_sha256=sha(specpath),
  patch_path=str(patch.relative_to(ROOT)),patch_sha256=sha(patch),
  build_source_root=str(source),build_directory=str(OUT),based_on_identity=str((BASE/'identity.json').relative_to(ROOT)),
  build_started_unix_s=start,build_wall_s=time.time()-start,build_concurrency_cap=6,
  compile_commands=jobs,link_command=link,cached_inputs_read_only=True,original_runtime_library_reused=True,
  mujoco_library_sha256=sha(DEST/'lib/libmujoco.so.3.2.3'),cached_binary_after_sha256=sha(CACHE/'bin/agent_server'),
  configuration_overrides=dict(old['configuration_overrides'],**spec['configuration_overrides']),
  control_patch_sha256=sha(ROOT/spec['control_patch']),recording_patch_sha256=sha(ROOT/spec['recording_patch']),
  deployment_scope='Formal N/r34-strict/constant r36; finite and original-task qualification reported separately',
  deadline_unix_s=None,wall_deadline_enabled=False)
 ident['source_files_sha256']=dict(old['source_files_sha256'])
 for rel in suffixes:ident['source_files_sha256'][rel]=sha(source/rel)
 SMALL.mkdir(parents=True,exist_ok=True)
 for p in [DEST/'identity.json',SMALL/'build.json']:p.write_text(json.dumps(ident,indent=2)+'\n')
 print(json.dumps(dict(build=ident['build'],binary_sha256=ident['binary_sha256'],wall_s=ident['build_wall_s'])),flush=True)
if __name__=='__main__':main()
