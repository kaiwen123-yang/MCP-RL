"""Retain the current, fully re-evaluated feasible nominal as a real candidate."""
from pathlib import Path
def apply(source):
    source=Path(source)
    def edit(path,changes):
        p=source/path;s=p.read_text()
        for old,new in changes:
            assert s.count(old)==1,(str(path),old[:90],s.count(old));s=s.replace(old,new,1)
        p.write_text(s)
    edit('mjpc/planners/ilqg/planner.h',[
      ('  bool w54_controller_infeasible=false;',
       '''  bool w54_controller_infeasible=false,w54_nominal_selected=false,w54_backward_failed=false;
  int w54_nominal_slot=-1;
  iLQGPolicy w54_nominal_policy;
  Trajectory w54_nominal_rollout;
  void W54AcceptNominal();''')])
    helper='''// This is an explicitly scored current-state candidate, not an old-policy escape.
void iLQGPlanner::W54AcceptNominal() {
  const std::unique_lock<std::shared_mutex> lock(mtx_);
  previous_policy=policy;
  policy=w54_nominal_policy;  // exact reference/K/k/representation/feedback scaling
  winner=w54_nominal_slot;
  w53_installed=true;
  w54_nominal_selected=true;
  w54_controller_infeasible=false;
}

// single iLQG iteration
'''
    edit('mjpc/planners/ilqg/planner.cc',[
      ('  w53_installed=false;w53_rollouts=0;w54_controller_infeasible=false;',
       '  w53_installed=false;w53_rollouts=0;w54_controller_infeasible=false;\n  w54_nominal_slot=-1;w54_nominal_selected=false;w54_backward_failed=false;'),
      ('  int best_nominal = this->BestRollout();',
       '''  int best_nominal = this->BestRollout();
  w54_nominal_slot=best_nominal;
  if(best_nominal>=0){
    // Save before candidate[0]'s reference and the backward pass mutate K/k.
    w54_nominal_policy=candidate_policy[best_nominal];
    w54_nominal_rollout=trajectory[best_nominal];
  }'''),
      ('// single iLQG iteration\n',helper),
      ('  if (backward_pass_status == 0) {\n    w54_controller_infeasible=true;',
       '''  if (backward_pass_status == 0) {
    w54_backward_failed=true;
    if(w54_nominal_slot>=0)W54AcceptNominal();
    else w54_controller_infeasible=true;'''),
      ('''  int best_rollout = this->BestRollout();
  if (best_rollout == -1) {''',
       '''  int best_rollout = this->BestRollout();
  // The already re-evaluated nominal participates under the identical J/risk
  // ordering. An invalid nominal never enters this comparison.
  if(w54_nominal_slot>=0 && (best_rollout<0 ||
      w54_nominal_rollout.total_return<=trajectory[best_rollout].total_return)){
    W54AcceptNominal();
    action_step=0.;expected=1.e-16;improvement=0.;surprise=0.;
    backward_pass.UpdateRegularization(settings.min_regularization,
        settings.max_regularization,surprise,action_step);
    model_derivative_compute_time=model_derivative_time;
    cost_derivative_compute_time=cost_derivative_time;
    backward_pass_compute_time=backward_pass_time;
    rollouts_compute_time=GetDuration(rollouts_start);
    w53_cpu[4]=double(std::clock()-w53_cpu_4)/CLOCKS_PER_SEC;
    policy_update_compute_time=0.;
    return;
  }
  if (best_rollout == -1) {'''),
      ('write(std::string(directory)+name,policy,w53_installed?trajectory[winner]:policy.trajectory);',
       'write(std::string(directory)+name,policy,w54_nominal_selected?w54_nominal_rollout:(w53_installed?trajectory[winner]:policy.trajectory));'),
      ('<<",\\"controller_infeasible\\":"<<(w54_controller_infeasible?',
       '<<",\\"selection_source\\":\\""<<(w54_nominal_selected?"current_reassessed_nominal":"action_rollout")<<"\\",\\"backward_failed\\":"<<(w54_backward_failed?"true":"false")<<",\\"nominal_candidate_slot\\":"<<w54_nominal_slot<<",\\"controller_infeasible\\":"<<(w54_controller_infeasible?')
    ])
