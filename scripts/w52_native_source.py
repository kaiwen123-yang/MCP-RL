#!/usr/bin/env python3
"""W52 geometry correction and explicitly isolated optional B/S search."""
from pathlib import Path
import copy,difflib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'work/r11-frozen/S5-r39'
def change(t,a,b):
 assert t.count(a)==1,(a[:100],t.count(a))
 return t.replace(a,b,1)
def main(build='S5-r40'):
 dest=ROOT/'work/w52-implementation'/build/'source';dest.mkdir(parents=True,exist_ok=True)
 shutil.copytree(BASE/'mjpc',dest/'mjpc',dirs_exist_ok=True)
 shutil.copy2(ROOT/'experiments/terrain_memory/w52_geometry.h',dest/'mjpc/w52_geometry.h')
 h=dest/'mjpc/planners/ilqg/planner.h';t=h.read_text()
 t=change(t,'#include <shared_mutex>','#include <shared_mutex>\n#include <memory>')
 t=change(t,'  void W51Seed(int horizon,ThreadPool& pool);','''  void W51Seed(int horizon,ThreadPool& pool);
  struct W52Candidate {iLQGPolicy policy;Trajectory trajectory;int kind,slot;double alpha;std::string source;};
  bool w52_collect=false;
  std::string w52_branch;
  std::vector<W52Candidate> w52_pending;
  std::unique_ptr<iLQGPlanner> w52_workspaces[2];
  void W52Search(int horizon,ThreadPool& pool);''')
 h.write_text(t)
 p=dest/'mjpc/planners/ilqg/planner.cc';t=p.read_text()
 t=change(t,'#include <fstream>','#include <fstream>\n#include "mjpc/w52_geometry.h"')
 a=t.index('void iLQGPlanner::W51Seed');b=t.index('}  // namespace mjpc',a)
 t=t[:a]+(ROOT/'experiments/terrain_memory/w52_seed.inc').read_text()+'\n'+(ROOT/'experiments/terrain_memory/w52_search.inc').read_text()+'\n'+t[b:]
 t=change(t,'  ++r33_slots;', '''  if(w52_collect) {
    w52_pending.push_back({p,tr,kind,slot,alpha,source});return;
  }
  ++r33_slots;''')
 t=change(t,'    W51Seed(horizon,pool);\n    this->Iteration(horizon, pool);',
 '''    const bool dual=std::getenv("MJPC_W52_DUAL");
    w52_pending.clear();w52_collect=dual;
    W51Seed(horizon,pool);
    w52_collect=false;
    if(dual)W52Search(horizon,pool);else this->Iteration(horizon,pool);''')
 # Per-branch raw search record. Candidate ring is written exactly once at merge.
 t=change(t,'std::string(file)+".search"','std::string(file)+".search"+(w52_collect?"-"+w52_branch:"")')
 a=t.index('void iLQGPlanner::W50Transfer');b=t.index('void iLQGPlanner::W51Seed',a)
 block=t[a:b]
 block=change(block,'  backward_pass.regularization=w50_mu;', '  if(!w52_collect) {\n  backward_pass.regularization=w50_mu;')
 block=change(block,'  if(const char* prefix=std::getenv("MJPC_W49_DIRECTION")) {','  }\n  if(const char* prefix=std::getenv("MJPC_W49_DIRECTION")) {')
 t=t[:a]+block+t[b:]
 # The epoch-level scale field is the actual last branch value; branch arrays
 # and metrics remain in their own workspaces and candidate complete objects.
 t=t.replace('S5-r39',build);p.write_text(t)
 files=['mjpc/planners/ilqg/planner.h','mjpc/planners/ilqg/planner.cc','mjpc/w52_geometry.h']
 patch=''
 for name in files:
  old=(BASE/name).read_text() if (BASE/name).exists() else ''
  patch+=''.join(difflib.unified_diff(old.splitlines(True),(dest/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
 (ROOT/f'experiments/terrain_memory/s1-go2-r11-{build}.patch').write_text(patch)
 spec=copy.deepcopy(json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r39.json').read_text()))
 spec.update(build=build,based_on='S5-r39',controller_artifact=build,status='implemented_not_yet_executed',
   task='W52_PAIRED_T0_D8',control_patch=f'experiments/terrain_memory/s1-go2-r11-{build}.patch',
   recording_patch='scripts/s1_go2_w52_records.py')
 spec.pop('actual_result',None)
 geom=dict(original_apex='touchdown_time - .20; no observed-start reset',initial_acceleration_seed=0,
  trajectory='actual position/velocity to fixed original apex/touchdown; piecewise quintic',
  continuous_extrema='recursive derivative-root isolation; original bilinear patch and enabled-boundary partitions',
  geometry_tolerance_m=1e-5,unavailable='normal B search, no S rollout or phase/contact mutation',
  changed_PD_cost_height_horizon_K_PPO=False)
 motion=spec['configuration_overrides']['w51_motion_seed']
 motion.update(continuation=geom['trajectory'],positive_family_count=2 if build=='S5-r41' else 1)
 spec['w51_motion_seed']=motion
 spec['w52_geometry']=geom;spec['configuration_overrides']['w52_geometry']=geom
 search=dict(independent_B_S=build=='S5-r41',workers=10,positive_alphas_per_direction=9,
  branch_storage='independent planner workspaces; frozen same epoch state/context; sequential reuse of one pool',
  merge_order=['retained_B_nominal','B_positive','direct_S','S_positive'],
  regularization='selected legal branch only; final risk adjustment once; no cross-epoch accumulation',
  full_rollout_upper_typical=30 if build=='S5-r41' else 21,expected=None,ratio=None)
 spec['w52_search']=search;spec['configuration_overrides']['w52_search']=search
 (ROOT/f'experiments/terrain_memory/s1-go2-r11-{build}.json').write_text(json.dumps(spec,indent=2)+'\n')
 print('generated',build,flush=True)
if __name__=='__main__':
 import sys
 main(sys.argv[1] if len(sys.argv)>1 else 'S5-r40')
