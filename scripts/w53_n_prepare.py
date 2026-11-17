#!/usr/bin/env python3
"""Restore frozen R8 sources and three patches; apply only named execution fixes."""
import difflib,io,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'
OUT=ROOT/'work/w53-n';SOURCE=OUT/'source'
PIN='e00c47a5adb9856af2e0f24231bb3a60d5be23c4'
FREEZE='fb7f924f2d519582a127d15ff2e1e2cb2cd5bc9e'
def replace(t,a,b):
 assert t.count(a)==1,(a[:90],t.count(a))
 return t.replace(a,b,1)
def main():
 SOURCE.mkdir(parents=True,exist_ok=False)
 payload=subprocess.check_output(['git','archive',PIN,'mjpc'],cwd=UP)
 with tarfile.open(fileobj=io.BytesIO(payload)) as archive:archive.extractall(SOURCE,filter='data')
 for name in ['startup','equilibrium','support']:
  patch=subprocess.check_output(['git','show',f'{FREEZE}:experiments/terrain_memory/s1-go2-wb-{name}-upstream.patch'],cwd=ROOT)
  subprocess.run(['patch','-p1','--batch','--forward'],input=patch,cwd=SOURCE,check=True)
 original={str(p.relative_to(SOURCE)):p.read_text() for p in (SOURCE/'mjpc').rglob('*') if p.suffix in ('.h','.cc')}
 p=SOURCE/'mjpc/planners/model_derivatives.cc';s=p.read_text()
 s=replace(s,'  evaluate_.push_back(T - 2);','  if (evaluate_.back() != T - 2) evaluate_.push_back(T - 2);')
 p.write_text(s)
 p=SOURCE/'mjpc/utilities.cc';s=p.read_text();known=(ROOT/'work/r11-frozen/S5-r36/mjpc/utilities.cc').read_text()
 start=s.index('void StateDiff(');end=s.index('// return global height',start)
 a=known.index('void StateDiff(');b=known.index('// return global height',a)
 s=s[:start]+known[a:b]+s[end:];p.write_text(s)
 p=SOURCE/'mjpc/agent.cc';s=p.read_text()
 s=replace(s,'  // plan\n  if (!allocate_enabled)', '  // W53: no worker-dependent solver warm start in the private predictor.\n  if (GetNumberOrDefault(0, model_, "r11_prediction_coldstart") == 1)\n    model_->opt.disableflags |= mjDSBL_WARMSTART;\n\n  // plan\n  if (!allocate_enabled)')
 p.write_text(s)
 p=SOURCE/'mjpc/planners/ilqg/planner.h';s=p.read_text()
 s=replace(s,' private:\n  int num_trajectory_', ' public:\n  void W53Record(int horizon);\n  unsigned long w53_call=0,w53_policy_call=0,w53_rollouts=0;\n  bool w53_installed=false,w53_moving_recorded=false;\n  double w53_cpu[6]={};\n private:\n  int num_trajectory_')
 p.write_text(s)
 p=SOURCE/'mjpc/planners/ilqg/planner.cc';s=p.read_text()
 s=replace(s,'#include <iostream>','#include <iostream>\n#include <fstream>\n#include <iomanip>\n#include <filesystem>\n#include <ctime>\n#include <cmath>\n#include <cstdint>\n#include <stdexcept>')
 s=replace(s,'  // freeze the GUI value once per optimization step.','  w53_installed=false;w53_rollouts=0;\n  std::fill(w53_cpu,w53_cpu+6,0.);\n  // freeze the GUI value once per optimization step.')
 s=replace(s,'  this->Iteration(horizon, pool);\n}', '  this->Iteration(horizon, pool);\n  W53Record(horizon);\n}')
 s=replace(s,'    policy.CopyFrom(candidate_policy[winner], horizon);','    policy.CopyFrom(candidate_policy[winner], horizon);\n    w53_installed=true;')
 for i,key in enumerate(['nominal','model_derivative','cost_derivative','backward_pass','rollouts','policy_update']):
  a=f'  auto {key}_start = std::chrono::steady_clock::now();'
  s=replace(s,a,a+f'\n  const auto w53_cpu_{i}=std::clock();')
  if key=='nominal':
   a='  nominal_compute_time = GetDuration(nominal_start);'
  else:a=f'  double {key}_time = GetDuration({key}_start);'
  s=replace(s,a,a+f'\n  w53_cpu[{i}]=double(std::clock()-w53_cpu_{i})/CLOCKS_PER_SEC;')
 s=replace(s,'void iLQGPlanner::ActionRollouts(int horizon, ThreadPool& pool) {','void iLQGPlanner::ActionRollouts(int horizon, ThreadPool& pool) {\n  w53_rollouts+=num_trajectory_;')
 s=replace(s,'void iLQGPlanner::FeedbackRollouts(int horizon, ThreadPool& pool) {','void iLQGPlanner::FeedbackRollouts(int horizon, ThreadPool& pool) {\n  w53_rollouts+=num_trajectory_;')
 s=replace(s,'}  // namespace mjpc', (ROOT/'experiments/terrain_memory/w53_n_records.inc').read_text()+'\n}  // namespace mjpc');p.write_text(s)
 patch=''
 for p in sorted((SOURCE/'mjpc').rglob('*')):
  if p.suffix not in ('.h','.cc'):continue
  rel=str(p.relative_to(SOURCE));old=original[rel];new=p.read_text()
  if old!=new:patch+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
 (ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r42.patch').write_text(patch)
 # The restored model adapter is retained as source, with two explicit W53 bridges.
 old=subprocess.check_output(['git','show',f'{FREEZE}:scripts/s1_go2_wb_model.py'],cwd=ROOT,text=True)
 old=replace(old,'    if config[\'stage\'] == \'S1R8\':',"    config = deepcopy(config)\n    config['stage'] = 'S1R8'\n    if config['stage'] == 'S1R8':")
 old=replace(old,"values=dict(s1_stand_equilibrium=0,agent_timestep=prediction['integration_dt_s'],", "values=dict(r11_prediction_coldstart=1,s1_stand_equilibrium=0,agent_timestep=prediction['integration_dt_s'],")
 (ROOT/'scripts/s1_go2_w53_n_model.py').write_text(old)
 cfg=json.loads(subprocess.check_output(['git','show',f'{FREEZE}:experiments/terrain_memory/s1-go2-wb-support.json'],cwd=ROOT))
 overrides={k:cfg[k] for k in ['task','stationary_objective','prediction_settings','support_reference','startup']}
 overrides.update(controller_formulation='R8_NATIVE',local_contact_model={'enabled':False,'causal_force_anchor':False},
  r11_contact_completion=None,r11_body_reference=None,r11_body_twist_reference=None,r26_reference=None,
  r11_prediction_coldstart={'enabled':True,'scope':'private Agent predictor only'})
 spec=dict(build='S5-r42',route='N',status='implemented_not_run',based_on=FREEZE,author_commit=PIN,
  mujoco_commit='088079eff0450e32b98ee743141780ed68307506',menagerie_commit='d82891f8b6c576f9f6a9df308f3a1155ebedbc40',
  configuration_overrides=overrides,adapter_differences=['unique derivative output ownership','private predictor coldstart','StateDiff exact self/antipode zero','current original mission/frozen PPO cap/causal observer shell','complete installed-policy and native stage timing records'],
  prediction_material_model='none: restored R8 material-unaware baseline',historical_qualification_inherited=False,
  record_policy='every installed full policy and actual accepted rollout; all candidate scores and state progression; key-event full candidates; no ordinary rejected full rings',
  deadline=None,wall_deadline_enabled=False)
 (ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r42.json').write_text(json.dumps(spec,indent=2)+'\n')
 print('restored N source',SOURCE,flush=True)
if __name__=='__main__':main()
