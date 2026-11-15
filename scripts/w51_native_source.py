#!/usr/bin/env python3
"""Overlay the single W51 motion-seed family on the complete frozen r38."""
from pathlib import Path
import copy,difflib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'work/r11-frozen/S5-r38'
DEST=ROOT/'work/w51-implementation/source'
def change(t,a,b):
    assert t.count(a)==1,(a[:90],t.count(a))
    return t.replace(a,b,1)
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    shutil.copytree(BASE/'mjpc',DEST/'mjpc',dirs_exist_ok=True)
    shutil.copy2(ROOT/'experiments/terrain_memory/w51_motion.h',DEST/'mjpc/w51_motion.h')
    h=DEST/'mjpc/tasks/quadruped/quadruped.h';t=h.read_text()
    t=change(t,'#include "mjpc/task.h"','#include "mjpc/task.h"\n#include "mjpc/w51_motion.h"')
    t=change(t,'  bool R35Moving() const {','''  W51Motion w51_motion;
  void W51Step(double t,double* height,double* velocity) const {
    residual_.FootStep(height,residual_.GetPhase(t),residual_.GetGait(),velocity);
  }
  bool R35Moving() const {''');h.write_text(t)
    p=DEST/'mjpc/tasks/quadruped/quadruped.cc';t=p.read_text()
    t=change(t,'  residual_.last_transition_time_ = data->time;\n}', '''  residual_.last_transition_time_ = data->time;
  if(std::getenv("MJPC_W51_MOTION"))
    w51_motion.Observe(model,data,residual_.foot_geom_id_,residual_.GetPhase(data->time),
      parameters[residual_.cadence_param_id_],parameters[residual_.duty_param_id_],
      R35Moving(),parameters[residual_.amplitude_param_id_]);
}''');p.write_text(t)
    h=DEST/'mjpc/planners/ilqg/planner.h';t=h.read_text()
    t=change(t,'  void W50Capture(int horizon);','''  iLQGPolicy w51_Q;
  Trajectory w51_Qtr;
  bool w51_seed_valid=false;
  int w51_seed_rollouts=0;
  std::string w51_status="not_entered";
  void W51Seed(int horizon,ThreadPool& pool);
  void W50Capture(int horizon);''');h.write_text(t)
    p=DEST/'mjpc/planners/ilqg/planner.cc';t=p.read_text()
    t=change(t,'const bool eligible=kind!=1 ||', 'const bool eligible=(kind!=1 && kind!=4) ||')
    t=change(t,'    this->Iteration(horizon, pool);','    W51Seed(horizon,pool);\n    this->Iteration(horizon, pool);')
    t=change(t,'w49_first_valid=true;w49_base=r33_nominal_policy;w50_backward=candidate_policy[0];',
      'w49_first_valid=true;w49_base=w51_Q;w50_backward=candidate_policy[0];')
    # Only the construction/linearization uses Q. Acceptance and logged deltas
    # continue to refer to untouched original best nominal B everywhere else.
    a=t.index('void iLQGPlanner::W50Capture');b=t.index('void iLQGPlanner::W50Transfer')
    block=t[a:b].replace('r33_nominal_trajectory','w51_Qtr')
    t=t[:a]+block+t[b:]
    t=change(t,'    p.trajectory=r33_nominal_trajectory;p.ClearDelta();','    p.trajectory=w51_Qtr;p.ClearDelta();')
    t=change(t,'  r34_decisions.clear();','  w51_seed_valid=false;w51_seed_rollouts=0;w51_status=\"no_search_same_epoch_or_invalid_nominal\";\n  r34_decisions.clear();')
    t=t.replace('S5-r38','S5-r39')
    marker='       <<",\\\"raw_center_evaluations\\\":"<<w50_raw_center_evaluations'
    addition='''       <<",\\\"w51_seed_status\\\":\\\""<<w51_status<<"\\\""
       <<",\\\"w51_seed_rollouts\\\":"<<w51_seed_rollouts
       <<",\\\"w51_Q_is_S\\\":"<<(w51_seed_valid?"true":"false")
'''
    # The exact all-call record is one occurrence, separate from homotopy file.
    assert t.count(marker)==2
    t=t.replace(marker,addition+marker,1)
    t=change(t,'}  // namespace mjpc',(ROOT/'experiments/terrain_memory/w51_seed.inc').read_text()+'\n}  // namespace mjpc')
    p.write_text(t)
    files=['mjpc/planners/ilqg/planner.h','mjpc/planners/ilqg/planner.cc',
           'mjpc/tasks/quadruped/quadruped.h','mjpc/tasks/quadruped/quadruped.cc','mjpc/w51_motion.h']
    patch=''
    for name in files:
        old=(BASE/name).read_text() if (BASE/name).exists() else ''
        patch+=''.join(difflib.unified_diff(old.splitlines(True),(DEST/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r39.patch').write_text(patch)
    spec=copy.deepcopy(json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r38.json').read_text()))
    spec.update(build='S5-r39',based_on='S5-r38',controller_artifact='S5-r39',
      status='implemented_not_yet_executed',task='W51_PAIRED_T0_D8',
      control_patch='experiments/terrain_memory/s1-go2-r11-S5-r39.patch',recording_patch='scripts/s1_go2_w51_records.py')
    spec.pop('actual_result',None)
    motion=dict(name='one_phase_aware_motion_seed_as_alternative_linearization',
      raw_center='c_B + q_IK - q_B_joint; complete B feedback, PD bias and old delta preserved',
      seed_rollouts_max=1,positive_family_count=1,positive_alphas=9,
      search='valid S may be Q regardless of J(S)-J(B); invalid S uses B',
      acceptance='original R34 risk/J/roundoff strict comparison against original B and archive',
      phase='original absolute 2Hz/.45/C2/.005, native FL/RL/FR/RR',
      anchors='real 500Hz SetState task Transition; per-segment world anchors and fixed current touchdown',
      continuation='quintic position/velocity endpoint correction over original remaining phase; no extra gait parameter',
      ik='continuous previous-node branch, original joint limits, max40 Newton steps, 0.2rad numerical step cap, 1e-5m final error; no target/grid search',
      alpha_zero='original complete B retained; no assertion Q-positive limit equals B',
      expected=None,ratio=None,new_physics_for_ik=0)
    spec['configuration_overrides']['w51_motion_seed']=motion
    spec['w51_motion_seed']=motion
    (ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r39.json').write_text(json.dumps(spec,indent=2)+'\n')
    print('Generated r39 complete source and patch')
if __name__=='__main__':main()
