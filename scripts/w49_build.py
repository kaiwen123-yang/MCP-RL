#!/usr/bin/env python3
"""Build W49 from the complete frozen r36 configuration, using isolated objects."""
import concurrent.futures,hashlib,json,shlex,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'work/r11-frozen/S5-r36';OUT=ROOT/'work/w49-implementation';SOURCE=OUT/'source'
UP=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2';CACHE=UP/'build-autodl'
DEST=ROOT/'work/r11-frozen/S5-r37';SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w49'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 start=time.time();DEST.mkdir(exist_ok=True)
 old=json.loads((BASE/'identity.json').read_text())
 assert sha(BASE/'agent_server')==old['binary_sha256']=='b05c8da8ca74551dc73ea0272329234217d7160731e5cea8e33101bba3241338'
 commands=json.loads((CACHE/'compile_commands.json').read_text());jobs=[];replacements={}
 # Recompile every project translation unit linked into libmjpc / agent_server.
 # Header layout changed: rebuilding only policy.cc/planner.cc would be an ABI error.
 archive=shlex.split((CACHE/'mjpc/CMakeFiles/libmjpc.dir/link.txt').read_text().splitlines()[0])
 link=shlex.split((CACHE/'mjpc/grpc/CMakeFiles/agent_server.dir/link.txt').read_text())
 service=shlex.split((CACHE/'mjpc/grpc/CMakeFiles/mjpc_agent_service.dir/link.txt').read_text().splitlines()[0])
 targets={str((CACHE/'mjpc'/x).resolve()) for x in archive if x.endswith('.o')}
 targets|={str((CACHE/'mjpc/grpc'/x).resolve()) for x in link+service if x.endswith('.o')}
 for entry in commands:
  cmd=shlex.split(entry['command']);object_path=str((Path(entry['directory'])/cmd[cmd.index('-o')+1]).resolve())
  if object_path not in targets:continue
  rel=Path(entry['file']).relative_to(UP)
  if rel.parts[0]!='mjpc':continue
  src=SOURCE/rel
  assert src.exists(),src
  obj=OUT/'objects'/rel.with_suffix('.cc.o');obj.parent.mkdir(parents=True,exist_ok=True)
  cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(src);cmd.insert(1,'-I'+str(SOURCE))
  jobs.append(dict(command=cmd,cwd=entry['directory'],source=str(rel)))
  replacements[object_path]=str(obj)
 def compile(j):
  print('compile '+j['source'],flush=True);subprocess.run(j['command'],cwd=j['cwd'],check=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(compile,jobs))
 lib=OUT/'libmjpc.a';archive[2]=str(lib)
 archive=[replacements.get(str((CACHE/'mjpc'/x).resolve()),x) if x.endswith('.o') else x for x in archive]
 subprocess.run(archive,cwd=CACHE/'mjpc',check=True)
 subprocess.run(['/usr/bin/gcc-ranlib-11',str(lib)],check=True)
 servicelib=OUT/'libmjpc_agent_service.a';service[2]=str(servicelib)
 service=[replacements.get(str((CACHE/'mjpc/grpc'/x).resolve()),x) if x.endswith('.o') else x for x in service]
 subprocess.run(service,cwd=CACHE/'mjpc/grpc',check=True)
 subprocess.run(['/usr/bin/gcc-ranlib-11',str(servicelib)],check=True)
 link=[str(servicelib) if x=='../../lib/libmjpc_agent_service.a' else x for x in link]
 link[link.index('-o')+1]=str(DEST/'agent_server')
 link=[str(lib) if x=='../../lib/libmjpc.a' else '-flto=6' if x=='-flto' else replacements.get(str((CACHE/'mjpc/grpc'/x).resolve()),x) if x.endswith('.o') else x for x in link]
 subprocess.run(link,cwd=CACHE/'mjpc/grpc',check=True)
 shutil.copytree(SOURCE/'mjpc',DEST/'mjpc',dirs_exist_ok=True)
 for d in ['runtime-tasks','lib']:shutil.copytree(BASE/d,DEST/d,dirs_exist_ok=True)
 shutil.copy2(BASE/'SOURCE_COMMIT',DEST/'SOURCE_COMMIT')
 specpath=ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r37.json'
 spec=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r36.json').read_text())
 spec.update(build='S5-r37',controller_artifact='S5-r37',based_on='S5-r36',
   w49_delta_policy={'schema':49,'feedback':'frozen baseline K/reference/H/scale','increment':'first valid current N backward k, positive alpha only','curve':'accumulated already-scaled linear union knots, own absolute clock','expected':None,'ratio':None,'online':'replace N positive family with nine k-guided candidates when baseline metric is valid; native bootstrap retained'},
   configuration_overrides=dict(old['configuration_overrides']))
 spec.update(status='implemented_finite_and_short_deployment_completed',task='W49_FIXED_FEEDBACK_K_GUIDED',
   mechanism={'name':'k_guided_existing_P_frozen_feedback','input_reference_branch':'constant','dV_for_new_family':None,'ratio_for_new_family':None},
   control_patch='experiments/terrain_memory/s1-go2-r11-S5-r37.patch',recording_patch='scripts/s1_go2_w49_records.py')
 spec['configuration_overrides']['w49_delta_policy']=spec['w49_delta_policy']
 specpath.write_text(json.dumps(spec,indent=2)+'\n')
 ident=dict(old);ident.update(build='S5-r37',based_on='S5-r36',controller_contract_build='S5-r37',
  prior_binary_sha256=old['binary_sha256'],binary_sha256=sha(DEST/'agent_server'),
  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  specification_path=str(specpath.relative_to(ROOT)),specification_sha256=sha(specpath),
  patch_path='experiments/terrain_memory/s1-go2-r11-S5-r37.patch',
  patch_sha256=sha(ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r37.patch'),
  build_source_root=str(SOURCE),build_directory=str(OUT),build_started_unix_s=start,build_wall_s=time.time()-start,
  build_concurrency_cap=6,compile_commands=jobs,link_command=link,cached_inputs_read_only=True,
  configuration_overrides=dict(spec['configuration_overrides']),mujoco_library_sha256=sha(DEST/'lib/libmujoco.so.3.2.3'),
  deadline=None,deadline_unix_s=None,wall_deadline_enabled=False,qualification='finite_not_yet_run')
 ident['source_files_sha256']=dict(old['source_files_sha256'])
 for rel in ['mjpc/planners/ilqg/'+x for x in ['policy.h','policy.cc','planner.h','planner.cc']]:
  ident['source_files_sha256'][rel]=sha(SOURCE/rel)
 for p in [DEST/'identity.json',SMALL/'build.json']:p.write_text(json.dumps(ident,indent=2)+'\n')
 print(json.dumps({'build':'S5-r37','binary_sha256':ident['binary_sha256'],'translation_units':len(jobs),'wall_s':ident['build_wall_s']}),flush=True)
if __name__=='__main__':main()
