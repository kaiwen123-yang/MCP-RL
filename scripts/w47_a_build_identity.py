#!/usr/bin/env python3
"""Isolated W47 paired diagnostic binaries; frozen r35 assets stay read-only."""
import concurrent.futures,difflib,hashlib,json,shlex,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'work/r11-frozen/S5-r35'
UPSTREAM=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'
CACHE=UPSTREAM/'build-autodl'
OUT=ROOT/'work/w47-implementation'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/task-a'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 started=time.time();OUT.mkdir(parents=True,exist_ok=True)
 identity=json.loads((FROZEN/'identity.json').read_text())
 if sha(CACHE/'bin/agent_server')!=identity['binary_sha256']:raise ValueError('Shared cached binary no longer equals frozen r35')
 for rel,digest in identity['source_files_sha256'].items():
  if sha(UPSTREAM/rel)!=digest:raise ValueError('Shared source drift: '+rel)
 source=OUT/'source';source.mkdir(exist_ok=False)
 old=(FROZEN/'mjpc/utilities.cc').read_text()
 marker='    mj_differentiatePos(m, ds, h, s1, s2);'
 fix='''    mj_differentiatePos(m, ds, h, s1, s2);
    // A rotation compared with itself (or its exact quaternion antipode) is
    // exactly zero. Do not amplify quaternion-product roundoff through K.
    // There is no neighbourhood tolerance: distinct rotations are unchanged.
    for (int j=0; j<m->njnt; ++j) {
      const int type=m->jnt_type[j];
      if (type!=mjJNT_FREE && type!=mjJNT_BALL) continue;
      const int q=m->jnt_qposadr[j]+(type==mjJNT_FREE?3:0);
      const int v=m->jnt_dofadr[j]+(type==mjJNT_FREE?3:0);
      bool identical=true,antipodal=true;
      for (int k=0;k<4;++k) {
        identical=identical && s1[q+k]==s2[q+k];
        antipodal=antipodal && s1[q+k]==-s2[q+k];
      }
      if (identical || antipodal) mju_zero(ds+v,3);
    }'''
 if old.count(marker)!=1:raise ValueError('StateDiff anchor differs')
 corrected=old.replace(marker,fix)
 p=source/'mjpc/utilities.cc';p.parent.mkdir(parents=True);p.write_text(corrected)
 patch=ROOT/'experiments/terrain_memory/w47-state-difference-identity.patch'
 patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),corrected.splitlines(True),fromfile='a/mjpc/utilities.cc',tofile='b/mjpc/utilities.cc')))
 planner=(FROZEN/'mjpc/planners/ilqg/planner.cc').read_text()
 marker='  trajectory[0].Rollout(action,task,model,data_[0].get(),state.data(),time,mocap.data(),userdata.data(),horizon);'
 new='''  // W47 offline-only, fixed saved candidate. Alpha is already in u_ff.
  if (std::getenv("MJPC_W47_INDEXED_SCORE")) {
    auto indexed=[&](double* u,const double* x,int k) {
      bool changed,failed,restored;
      policy.ActionIndex(u,x,k,&changed,&failed,&restored);
      r29_failed[0]+=failed;
    };
    trajectory[0].RolloutDiscrete(indexed,task,model,data_[0].get(),state.data(),time,mocap.data(),userdata.data(),horizon);
  } else {
    trajectory[0].Rollout(action,task,model,data_[0].get(),state.data(),time,mocap.data(),userdata.data(),horizon);
  }
  ++r33_rollouts;'''
 if planner.count(marker)!=1:raise ValueError('ScoreOnly anchor differs')
 diagnostic=planner.replace(marker,new)
 p=source/'mjpc/planners/ilqg/planner.cc';p.parent.mkdir(parents=True);p.write_text(diagnostic)
 p=ROOT/'experiments/terrain_memory/w47-fixed-candidate-diagnostic.patch'
 p.write_text(''.join(difflib.unified_diff(planner.splitlines(True),diagnostic.splitlines(True),fromfile='a/mjpc/planners/ilqg/planner.cc',tofile='b/mjpc/planners/ilqg/planner.cc')))
 commands=json.loads((CACHE/'compile_commands.json').read_text());jobs=[];records=[]
 for suffix in ('mjpc/utilities.cc','mjpc/planners/ilqg/planner.cc'):
  entry=next(c for c in commands if c['file']==str(UPSTREAM/suffix));cmd=shlex.split(entry['command'])
  obj=OUT/'objects'/Path(suffix).with_suffix('.cc.o');obj.parent.mkdir(parents=True,exist_ok=True)
  cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(source/suffix)
  cmd.insert(1,'-I'+str(FROZEN));cmd.insert(1,'-I'+str(source))
  jobs.append((cmd,entry['directory']));records.append(dict(command=cmd,cwd=entry['directory']))
 def execute(job):subprocess.run(job[0],cwd=job[1],check=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(execute,jobs))
 archive=shlex.split((CACHE/'mjpc/CMakeFiles/libmjpc.dir/link.txt').read_text().splitlines()[0])
 link=shlex.split((CACHE/'mjpc/grpc/CMakeFiles/agent_server.dir/link.txt').read_text())
 variants={}
 for mode in ('old','corrected'):
  build=OUT/mode;build.mkdir(exist_ok=False);library=build/'libmjpc.a'
  cmd=archive[:];cmd[2]=str(library)
  for i,v in enumerate(cmd):
   if v=='CMakeFiles/libmjpc.dir/planners/ilqg/planner.cc.o':cmd[i]=str(OUT/'objects/mjpc/planners/ilqg/planner.cc.o')
   if mode=='corrected' and v=='CMakeFiles/libmjpc.dir/utilities.cc.o':cmd[i]=str(OUT/'objects/mjpc/utilities.cc.o')
  subprocess.run(cmd,cwd=CACHE/'mjpc',check=True);subprocess.run(['/usr/bin/gcc-ranlib-11',str(library)],check=True)
  cmd=link[:];cmd[cmd.index('-o')+1]=str(build/'agent_server')
  cmd=[str(library) if x=='../../lib/libmjpc.a' else '-flto=6' if x=='-flto' else x for x in cmd]
  subprocess.run(cmd,cwd=CACHE/'mjpc/grpc',check=True)
  variants[mode]=dict(binary=str((build/'agent_server').relative_to(ROOT)),binary_sha256=sha(build/'agent_server'),
   utilities_source_sha256=sha(source/'mjpc/utilities.cc') if mode=='corrected' else sha(FROZEN/'mjpc/utilities.cc'),
   planner_source_sha256=sha(source/'mjpc/planners/ilqg/planner.cc'),link_command=cmd)
  print(json.dumps(dict(built=mode,binary_sha256=variants[mode]['binary_sha256'],wall_s=time.time()-started)),flush=True)
 result=dict(status='built_diagnostic_only',based_on='S5-r35',frozen_binary_sha256=identity['binary_sha256'],
  deployment_qualified=False,build_concurrency_cap=6,started_unix_s=started,ended_unix_s=time.time(),variants=variants,
  shared_read_only_library=str(FROZEN/'lib/libmujoco.so.3.2.3'),mujoco_sha256=sha(FROZEN/'lib/libmujoco.so.3.2.3'),
  patches=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [patch,ROOT/'experiments/terrain_memory/w47-fixed-candidate-diagnostic.patch']],
  compile_commands=records,shared_cached_binary_after_sha256=sha(CACHE/'bin/agent_server'))
 SMALL.mkdir(parents=True,exist_ok=True);(SMALL/'diagnostic-build.json').write_text(json.dumps(result,indent=2)+'\n')
 (OUT/'identity.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
