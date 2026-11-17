#!/usr/bin/env python3
"""Prepare r44 from exact r42/R8 source and the new memoryless execution map."""
import difflib,io,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'
OUT=ROOT/'work/w54-n-r48';SOURCE=OUT/'source'
PIN='e00c47a5adb9856af2e0f24231bb3a60d5be23c4'
FREEZE='fb7f924f2d519582a127d15ff2e1e2cb2cd5bc9e'
def replace(t,a,b):
    assert t.count(a)==1,(a[:80],t.count(a));return t.replace(a,b,1)
def main():
    SOURCE.mkdir(parents=True,exist_ok=False)
    payload=subprocess.check_output(['git','archive',PIN,'mjpc'],cwd=UP)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:archive.extractall(SOURCE,filter='data')
    for name in ['startup','equilibrium','support']:
        payload=subprocess.check_output(['git','show',f'{FREEZE}:experiments/terrain_memory/s1-go2-wb-{name}-upstream.patch'],cwd=ROOT)
        subprocess.run(['patch','-p1','--batch','--forward'],input=payload,cwd=SOURCE,check=True)
    subprocess.run(['patch','-p1','--batch','--forward'],input=(ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r42.patch').read_bytes(),cwd=SOURCE,check=True)
    old={str(p.relative_to(SOURCE)):p.read_text() for p in (SOURCE/'mjpc').rglob('*') if p.suffix in ('.h','.cc')}
    for name in ['core','native']:
        (SOURCE/f'mjpc/w54_e_{name}.h').write_text((ROOT/f'experiments/terrain_memory/w54_e_{name}_r48.h').read_text())
    p=SOURCE/'mjpc/trajectory.h';s=p.read_text()
    s=replace(s,'  std::vector<double> actions;', '  std::vector<double> w54_applied_actions, w54_execution_diagnostics;\n  std::vector<double> actions;')
    p.write_text(s)
    p=SOURCE/'mjpc/trajectory.cc';s=p.read_text()
    s=replace(s,'#include "mjpc/trajectory.h"','#include "mjpc/trajectory.h"\n#include "mjpc/w54_e_native.h"')
    s=replace(s,'  actions.resize(dim_action * T);','  actions.resize(dim_action * T);\n  w54_applied_actions.resize(dim_action*T);\n  w54_execution_diagnostics.resize(40*T);')
    oldline='    mju_copy(data->ctrl, DataAt(actions, t * nu), nu);'
    newline=oldline+'\n    W54ApplyE(model,data,DataAt(w54_execution_diagnostics,t*40));\n    mju_copy(DataAt(w54_applied_actions,t*nu),data->ctrl,nu);'
    assert s.count(oldline)==2;s=s.replace(oldline,newline)
    a='    mju_copy(DataAt(actions, (horizon - 1) * dim_action),'
    b='    mju_copy(DataAt(w54_applied_actions,(horizon-1)*dim_action),DataAt(w54_applied_actions,(horizon-2)*dim_action),dim_action);\n    mju_copy(DataAt(w54_execution_diagnostics,(horizon-1)*40),DataAt(w54_execution_diagnostics,(horizon-2)*40),40);\n'+a
    assert s.count(a)==2;s=s.replace(a,b)
    p.write_text(s)
    p=SOURCE/'mjpc/planners/model_derivatives.cc';s=p.read_text()
    s=replace(s,'#include "mjpc/planners/model_derivatives.h"','#include "mjpc/planners/model_derivatives.h"\n#include "mjpc/w54_e_native.h"')
    a=s.index('      // Jacobians\n');b=s.index('\n    });',a)
    s=s[:a]+'''      // Direct FD of the SAME E(x,u) plus 2x1ms held-input transition.
      // Every perturbed state/action recomputes E, including shifted nominal B.
      W54TransitionFD(m,d,tol,mode,
          t==T-1?nullptr:DataAt(A,t*dim_state_derivative*dim_state_derivative),
          t==T-1?nullptr:DataAt(B,t*dim_state_derivative*dim_action),
          DataAt(C,t*dim_sensor*dim_state_derivative),
          t==T-1?nullptr:DataAt(D,t*dim_sensor*dim_action),
          dim_sensor,t==T-1);'''+s[b:];p.write_text(s)
    p=SOURCE/'mjpc/planners/ilqg/planner.cc';s=p.read_text()
    s=s.replace('MJPC_W53_N_RECORD_DIR','MJPC_W54_N_RECORD_DIR')
    s=replace(s,'int64_t hdr[10]={53,1,','int64_t hdr[10]={54,2,')
    s=replace(s,'    f.close();if(!f)throw std::runtime_error("W53 N installed-policy write failed");',
      '    a(tr.w54_applied_actions,horizon*dim_action);a(tr.w54_execution_diagnostics,horizon*40);\n    f.close();if(!f)throw std::runtime_error("W54 N installed-policy write failed");')
    s=s.replace('<<(2*(horizon-1)+1)', '<<((horizon-1)*(1+dim_state_derivative+dim_action)+(1+dim_state_derivative))')
    p.write_text(s)
    p=SOURCE/'mjpc/planners/ilqg/planner.cc';s=p.read_text()
    a=s.index('// W53 recording only.');b=s.index('\n}  // namespace mjpc',a)
    s=s[:a]+(ROOT/'experiments/terrain_memory/w54_n_records_r48.inc').read_text()+s[b:];p.write_text(s)
    p=SOURCE/'mjpc/trajectory.h';s=p.read_text()
    s=replace(s,'  bool failure;', '  bool w54_domain_feasible=true; double w54_domain_min_gap=1.e9;\n  bool failure;')
    p.write_text(s)
    p=SOURCE/'mjpc/trajectory.cc';s=p.read_text()
    assert s.count('  failure = false;')==3
    s=s.replace('  failure = false;','  failure = false;w54_domain_feasible=true;w54_domain_min_gap=1.e9;')
    a='    // Residual and trace above belong to the node start.'
    assert s.count(a)==2;s=s.replace(a,'    w54_domain_feasible &= W54PredictedDomainOkay(model,data,&w54_domain_min_gap);\n'+a)
    a='    if (held_control) mj_step(model, data);'
    b='    if (held_control) {mj_step(model, data);w54_domain_feasible &= W54PredictedDomainOkay(model,data,&w54_domain_min_gap);}'
    assert s.count(a)==2;s=s.replace(a,b)
    assert s.count('  UpdateReturn(task);')==2
    s=s.replace('  UpdateReturn(task);','  UpdateReturn(task);\n  if(!w54_domain_feasible){failure=true;total_return=kMaxReturnValue;}')
    p.write_text(s)
    p=SOURCE/'mjpc/planners/ilqg/planner.h';s=p.read_text()
    s=replace(s,'  void W53Record(int horizon);','  bool w54_controller_infeasible=false;\n  void W53Record(int horizon);');p.write_text(s)
    p=SOURCE/'mjpc/planners/ilqg/planner.cc';s=p.read_text()
    s=replace(s,'  w53_installed=false;w53_rollouts=0;','  w53_installed=false;w53_rollouts=0;w54_controller_infeasible=false;')
    a=s.index("    // The policy's traj's first state");b=s.index("    // set feedback scaling",a)
    s=s[:a]+'    // Infeasible shifted plans are linearization sources only, never execution fallback.\n    candidate_policy[0].trajectory = trajectory[0];\n\n'+s[b:]
    s=replace(s,'  if (backward_pass_status == 0) {\n    // set timers','  if (backward_pass_status == 0) {\n    w54_controller_infeasible=true;\n    // set timers')
    s=replace(s,'  if (best_rollout == -1) {\n    return;', '  if (best_rollout == -1) {\n    w54_controller_infeasible=true;\n    return;')
    p.write_text(s)
    from w54_n_nominal_selection import apply as retain_current_nominal
    retain_current_nominal(SOURCE)
    patch=''
    for p in sorted((SOURCE/'mjpc').rglob('*')):
        if p.suffix not in ('.h','.cc'):continue
        rel=str(p.relative_to(SOURCE));before=old.get(rel,'');after=p.read_text()
        if before!=after:patch+=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    (ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r48.patch').write_text(patch)
    spec=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r42.json').read_text())
    spec.update(build='S5-r48',status='implemented_not_built',based_on='df07c4bb690f831745879af3a86800c2c3ec427d',
      baseline_build='S5-r42',execution_source_names=['s1_go2_w54_n_model_r48.py','s1_go2_w54_n_records_r48.py','s1_go2_w54_n_execution_r48.py'],
      execution_operator={'state':'memoryless x/raw action; no latch/filter/anchor states','source':'experiments/terrain_memory/w54_e_core_r48.h',
       'call_sites':['real 500Hz','native continuous/discrete rollout','each composite transition FD perturbation'],
       'control_hold_s':.002,'predictor_substep_s':.001,'predictor_substeps_per_transition':2,
       'plant_substep_s':.0002,'plant_substeps_per_hold':10,
       'surface':'known material map and undeformed upper-surface prior; no hidden material e/q',
       'landing_speed_m_s':.05,'delay_s':.022,'available_acceleration_reserve_fraction':.25,
       'activation':'quintic smooth distance and descending-speed gates; stopping-distance lead with bounded torque capacity',
       'projection_iterations':24,'dynamic_memory':False,'hard_safety_claim':False,
       'prior_model_boundary':'restored r42 rigid contact predictor; no future TM02 history propagation'},
      prediction_material_model='known undeformed surface upper-bound prior in E; full TM02 only in unchanged plant')
    spec['execution_operator'].update(revision='r48_total_acceleration_and_uniform_predicted_domain_check',
       baseline='Jdot*v+J*M^-1*(S*tau-bias+passive-frictionloss*tanh(v/.01)); contact-free current articulated approximation',
       target='total normal deceleration; subtract raw acceleration before requesting torque increment',
       available_acceleration='one quarter of maximum total positive normal acceleration in original torque/input box',
       candidate_check='all E-composed rollouts, including shifted/nominal, reject predicted material gap below -.001m at 1ms; original H=.05,normal strain=.02 proxy; not actual safety',
       failure='current feasible re-evaluated nominal remains in original J/risk ordering when no new direction improves it; infeasible only when no current feasible candidate remains, no unsafe old policy fallback')
    spec['revision_evidence']='results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w54/n-r44-acceleration-evidence.json'
    spec['configuration_overrides'].update(w54_n_model_source='scripts/s1_go2_w54_n_model_r48.py',
       w54_n_records_source='scripts/s1_go2_w54_n_records_r48.py',w54_n_execution_source='scripts/s1_go2_w54_n_execution_r48.py')
    spec['configuration_overrides']['w54_n_execution']=spec['execution_operator']
    spec['adapter_differences']+=['memoryless precontact E in true 500Hz/native forward/every FD','raw and applied actions retained',
      'direct FD of complete 2ms E-composed transition, no analytic chain approximation']
    (ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r48.json').write_text(json.dumps(spec,indent=2)+'\n')
    print('prepared',SOURCE,flush=True)
if __name__=='__main__':main()
