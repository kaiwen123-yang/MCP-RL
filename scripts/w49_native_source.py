#!/usr/bin/env python3
"""W49 native delta-policy representation and bounded k-guided proposals."""
from pathlib import Path
import difflib,shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'work/r11-frozen/S5-r36'
DEST=ROOT/'work/w49-implementation/source'
def change(text,old,new,n=1):
 assert text.count(old)==n,(old[:90],text.count(old),n)
 return text.replace(old,new)
def main():
 DEST.mkdir(parents=True,exist_ok=False)
 shutil.copytree(BASE/'mjpc',DEST/'mjpc')
 h=DEST/'mjpc/planners/ilqg/policy.h';t=h.read_text()
 t=change(t,'#include <vector>','#include <vector>\n#include <iosfwd>\n#include <string>')
 t=change(t,'  // copy policy',"""  // W49 accumulated, already-scaled feedforward curve on its own absolute clock.
  // delta_alpha is provenance only; DeltaAction never multiplies it again.
  std::vector<double> delta_times, delta_values;
  double delta_alpha=0, delta_begin=0, delta_end=0;
  void ClearDelta();
  void DeltaAction(double* u,double time) const;
  void AddDelta(const std::vector<double>& times,const std::vector<double>& values,
                int count,double alpha);
  void WriteDelta(std::ostream& out) const;
  void ReadDelta(std::istream& in);
  void Save(const std::string& path) const;

  // copy policy""")
 h.write_text(t)
 p=DEST/'mjpc/planners/ilqg/policy.cc';t=p.read_text()
 t=change(t,'  metric_valid=false;record_plan_id=0;metric_regularization=0;',
                '  ClearDelta();\n  metric_valid=false;record_plan_id=0;metric_regularization=0;')
 t=change(t,'  std::vector<double> before_clip;',"""  std::vector<double> feedforward_delta(dim_action,0.);
  DeltaAction(feedforward_delta.data(),time);
  mju_addTo(action,feedforward_delta.data(),dim_action);
  std::vector<double> before_clip;""")
 t=change(t,'    array("nominal_unclipped",nominal.data(),dim_action);',
               '    array("nominal_unclipped",nominal.data(),dim_action);\n    array("feedforward_delta",feedforward_delta.data(),dim_action);')
 t=change(t,'  int nu=model->nu,nx=2*model->nv+model->na,ns=model->nq+model->nv+model->na;',
               '  if(!delta_times.empty())mju_error("W49 delta policy requires absolute query time; ActionIndex is invalid");\n  int nu=model->nu,nx=2*model->nv+model->na,ns=model->nq+model->nv+model->na;')
 t=change(t,'  trajectory = policy.trajectory;',"""  trajectory = policy.trajectory;
  delta_times=policy.delta_times;delta_values=policy.delta_values;
  delta_alpha=policy.delta_alpha;delta_begin=policy.delta_begin;delta_end=policy.delta_end;""")
 extra=r'''
void iLQGPolicy::ClearDelta() {
  delta_times.clear();delta_values.clear();delta_alpha=delta_begin=delta_end=0;
}
void iLQGPolicy::DeltaAction(double* out,double t) const {
  mju_zero(out,model->nu);
  if(!delta_times.empty())
    LinearInterpolation(out,t,delta_times,delta_values.data(),model->nu,delta_times.size());
}
void iLQGPolicy::AddDelta(const std::vector<double>& times,const std::vector<double>& values,
                          int count,double alpha) {
  if(alpha==0)return;  // Keep the exact baseline identity without regridding.
  if(representation!=1)mju_error("W49 k-guided family is declared for linear action interpolation");
  const double begin=times[0],end=times[count-1];
  std::vector<double> knots(times.begin(),times.begin()+count);
  for(double t:delta_times)if(t>begin && t<end)knots.push_back(t);
  std::sort(knots.begin(),knots.end());
  knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
  std::vector<double> merged(knots.size()*model->nu),old(model->nu),increment(model->nu);
  for(size_t i=0;i<knots.size();++i) {
    DeltaAction(old.data(),knots[i]);
    LinearInterpolation(increment.data(),knots[i],times,values.data(),model->nu,count);
    mju_addScl(merged.data()+i*model->nu,old.data(),increment.data(),alpha,model->nu);
  }
  delta_times=std::move(knots);delta_values=std::move(merged);
  delta_alpha=alpha;delta_begin=begin;delta_end=end;
}
void iLQGPolicy::WriteDelta(std::ostream& out) const {
  double h[]={49.,double(metric_valid),double(delta_times.size()),delta_alpha,delta_begin,delta_end};
  out.write(reinterpret_cast<const char*>(h),sizeof(h));
  for(const auto* v:{&delta_times,&delta_values})
    out.write(reinterpret_cast<const char*>(v->data()),v->size()*sizeof(double));
}
void iLQGPolicy::ReadDelta(std::istream& in) {
  ClearDelta();double h[6];in.read(reinterpret_cast<char*>(h),sizeof(h));
  if(!in || h[0]!=49. || h[2]<0)mju_error("W49 complete-policy extension missing");
  metric_valid=bool(h[1]);int count=int(h[2]);
  delta_alpha=h[3];delta_begin=h[4];delta_end=h[5];
  delta_times.resize(count);delta_values.resize(count*model->nu);
  for(auto* v:{&delta_times,&delta_values})
    in.read(reinterpret_cast<char*>(v->data()),v->size()*sizeof(double));
  if(!in)mju_error("W49 complete-policy extension truncated");
}
void iLQGPolicy::Save(const std::string& path) const {
  int T=trajectory.horizon,ns=model->nq+model->nv+model->na,nu=model->nu,nx=2*model->nv+model->na;
  std::ofstream out(path,std::ios::binary|std::ios::trunc);
  double h[]={49.,double(record_plan_id),trajectory.times[0],double(T),double(ns),double(nu),double(nx),
    double(trajectory.dim_residual),metric_regularization,double(metric_regularization_type),
    nominal_feedback_scale,action_step_alpha,feedback_scaling,double(representation),
    double(model->nuserdata),trajectory.total_return};
  out.write(reinterpret_cast<const char*>(h),sizeof(h));
  auto w=[&](const std::vector<double>& v,int n){out.write(reinterpret_cast<const char*>(v.data()),n*sizeof(double));};
  w(trajectory.states,T*ns);w(trajectory.actions,(T-1)*nu);w(trajectory.times,T);w(trajectory.residual,T*trajectory.dim_residual);
  w(feedback_gain,T*nu*nx);w(metric,T*nu*nu);w(base_Qu,T*nu);w(base_Qxu,T*nx*nu);w(base_active,T*nu);
  WriteDelta(out);
}
'''
 t=change(t,'}  // namespace mjpc',extra+'\n}  // namespace mjpc');p.write_text(t)
 h=DEST/'mjpc/planners/ilqg/planner.h';t=h.read_text()
 t=change(t,'  void R33Structured(int horizon,ThreadPool& pool);',"""  void R33Structured(int horizon,ThreadPool& pool);
  bool w49_first_valid=false;
  std::vector<double> w49_k,w49_times;
  iLQGPolicy w49_base;
  void W49Capture(int horizon);
  void W49Guided(int horizon,ThreadPool& pool);""");h.write_text(t)
 p=DEST/'mjpc/planners/ilqg/planner.cc';t=p.read_text()
 t=change(t,'  policy.record_plan_id=h[1];policy.metric_valid=true;policy.metric_regularization=h[8];',
 """  policy.ClearDelta();
  policy.record_plan_id=h[1];policy.metric_valid=false;policy.metric_regularization=h[8];""")
 t=change(t,'auto& tr=policy.trajectory;tr.horizon=horizon;','auto& tr=policy.trajectory;tr.horizon=horizon;tr.total_return=h[15];')
 t=change(t,'  policy.action_step_alpha=h[11];policy.feedback_scaling=h[12];policy.representation=h[13];',
 """  policy.action_step_alpha=h[11];policy.feedback_scaling=h[12];policy.representation=h[13];
  if(int(h[0])==49)policy.ReadDelta(in);
  // Legacy schema did not carry metric validity. Its absence is not evidence of a valid metric.
""")
 t=change(t,'r33_mechanism!="P" && r33_mechanism!="R32"', 'r33_mechanism!="P" && r33_mechanism!="K" && r33_mechanism!="R32"')
 t=change(t,'  R33Import(horizon);',"""  R33Import(horizon);
  if(std::getenv("MJPC_W49_QUERY_ONLY")) {
    if(const char* dest=std::getenv("MJPC_W49_EXPORT"))policy.Save(dest);
    r32.deployable=policy.metric_valid;r29_return_reason="w49_imported_query_only";
    return;  // Explicit diagnostic: no R35 rollout, derivatives, solve, or physics.
  }""")
 t=change(t,'  r34_decisions.clear();','  w49_first_valid=false;w49_k.clear();w49_times.clear();\n  r34_decisions.clear();')
 t=change(t,'    this->Iteration(horizon, pool);',"""    this->Iteration(horizon, pool);
    if(std::getenv("MJPC_W49_FINITE_GUIDED") && w49_first_valid)W49Guided(horizon,pool);""")
 t=change(t,'    candidate_policy[0].trajectory = trajectory[best_nominal];',
 """    candidate_policy[0].trajectory = trajectory[best_nominal];
    candidate_policy[0].ClearDelta();  // Actual nominal actions already include the old curve; B above retains it.""")
 t=change(t,'    candidate_policy[0].trajectory=r32_nominal;',
 '    candidate_policy[0].trajectory=r32_nominal;\n    candidate_policy[0].ClearDelta();')
 t=change(t,'    auto ro_start=std::chrono::steady_clock::now();',"""    if(!w49_first_valid)W49Capture(horizon);
    if(r33_mechanism=="K" && w49_base.metric_valid) {
      W49Guided(horizon,pool);
      r32.stopped=true;r32.stop_reason="w49_single_first_direction_finished";
      model_derivative_compute_time=model_derivative_time;
      cost_derivative_compute_time=cost_derivative_time;
      backward_pass_compute_time=backward_pass_time;
      return;
    }
    auto ro_start=std::chrono::steady_clock::now();""")
 # Candidate metadata remains the legacy contiguous block; complete policies use explicit sidecar identities.
 t=change(t,'  ++r33_slots;',"""  ++r33_slots;
  if(const char* path=std::getenv("MJPC_W49_COMPLETE_POLICIES"))
    p.Save(std::string(path)+"-candidate-"+std::to_string(r33_slots)+".bin");""")
 t=change(t,'    part(policy.base_active,horizon*dim_action);',
              '    part(policy.base_active,horizon*dim_action);\n    policy.WriteDelta(out);')
 t=change(t,'    write(policy.base_active,horizon*dim_action);',
              '    write(policy.base_active,horizon*dim_action);\n    policy.WriteDelta(out);')
 t=t.replace('if(r33_mechanism=="N") {','if(r33_mechanism=="N" || r33_mechanism=="K") {')
 t=t.replace('controller_artifact\\":\\"S5-r36','controller_artifact\\":\\"S5-r37')
 extra=r'''
void iLQGPlanner::W49Capture(int horizon) {
  w49_first_valid=true;
  w49_base=r33_nominal_policy;
  w49_times.assign(candidate_policy[0].trajectory.times.begin(),candidate_policy[0].trajectory.times.begin()+horizon-1);
  w49_k.assign(candidate_policy[0].action_improvement.begin(),candidate_policy[0].action_improvement.begin()+(horizon-1)*dim_action);
  if(const char* prefix=std::getenv("MJPC_W49_DIRECTION")) {
    w49_base.Save(std::string(prefix)+"-baseline.bin");
    std::ofstream out(std::string(prefix)+"-direction.bin",std::ios::binary|std::ios::trunc);
    double h[]={49.,double(r29_call_id),time,double(horizon-1),double(dim_action),
      double(r29_backward_attempts),backward_pass.regularization,backward_pass.regularization_rate,
      r33_nominal_trajectory.total_return,r33_nominal_trajectory.trunk_violation,R34Budget(r33_nominal_trajectory)};
    out.write(reinterpret_cast<const char*>(h),sizeof(h));
    for(const auto* v:{&w49_times,&w49_k})out.write(reinterpret_cast<const char*>(v->data()),v->size()*sizeof(double));
  }
}
void iLQGPlanner::W49Guided(int horizon,ThreadPool& pool) {
  if(!w49_first_valid || !w49_base.metric_valid)return;
  const double mu=backward_pass.regularization,rate=backward_pass.regularization_rate;
  int count=num_trajectory_-1,before=pool.GetCount();
  for(int i=0;i<count;++i) {
    candidate_policy[i]=w49_base;
    candidate_policy[i].AddDelta(w49_times,w49_k,horizon-1,linesearch_steps[i]);
    candidate_policy[i].action_step_alpha=linesearch_steps[i];
    r29_projected[i]=r29_failed[i]=r29_restored[i]=0;
    pool.Schedule([&,i](){
      auto f=[&,i](double* u,const double* x,double t) {
        bool changed,failed,restored;
        candidate_policy[i].ActionImpl(u,x,t,false,&changed,&failed,&restored);
        r29_projected[i]+=changed;r29_failed[i]+=failed;r29_restored[i]+=restored;
      };
      trajectory[i].Rollout(f,task,model,data_[ThreadPool::WorkerId()].get(),state.data(),time,mocap.data(),userdata.data(),horizon);
      if(r29_failed[i])trajectory[i].failure=true;
    });
  }
  pool.WaitCount(before+count);pool.ResetCount();r33_rollouts+=count;
  for(int i=0;i<count;++i)
    R33Consider(candidate_policy[i],trajectory[i],horizon,1,i,linesearch_steps[i],"k_guided_"+std::to_string(i));
  // This family has no applicable N dV/ratio. Preserve damping and its rate;
  // retain the original, separately defined violation-progress update online.
  backward_pass.regularization=mu;backward_pass.regularization_rate=rate;
  if(r33_mechanism=="K" && r33_source.rfind("k_guided_",0)==0 &&
      (r33_nominal_trajectory.trunk_violation>0 || r33_archive_trajectory.trunk_violation>0))
    backward_pass.ScaleRegularization(
      r33_nominal_trajectory.trunk_violation>r33_archive_trajectory.trunk_violation?
        1./backward_pass.regularization_factor:backward_pass.regularization_factor,
      settings.min_regularization,settings.max_regularization);
  if(const char* prefix=std::getenv("MJPC_W49_DIRECTION")) {
    std::ofstream out(std::string(prefix)+"-guided.json",std::ios::trunc);out.precision(17);
    out<<"{\"first_valid_backward\":true,\"baseline_metric_valid\":true,\"extra_candidates\":"<<count
       <<",\"baseline_J\":"<<r33_nominal_trajectory.total_return
       <<",\"baseline_risk\":"<<r33_nominal_trajectory.trunk_violation
       <<",\"baseline_budget\":"<<R34Budget(r33_nominal_trajectory)
       <<",\"expected\":null,\"ratio\":null,\"mu_before\":"<<mu<<",\"rate_before\":"<<rate
       <<",\"mu_after\":"<<backward_pass.regularization<<",\"rate_after\":"<<backward_pass.regularization_rate
       <<",\"candidates\":[";
    for(int i=0;i<count;++i) {
      if(i)out<<",";
      out<<"{\"alpha\":"<<linesearch_steps[i]<<",\"J\":"<<trajectory[i].total_return
         <<",\"risk\":"<<trajectory[i].trunk_violation<<",\"budget\":"<<R34Budget(trajectory[i])
         <<",\"failure\":"<<(trajectory[i].failure?"true":"false")
         <<",\"strict_better_than_B\":"<<(R34Better(trajectory[i],r33_nominal_trajectory)?"true":"false")<<"}";
    }
    out<<"]}\n";
  }
}
'''
 t=change(t,'}  // namespace mjpc',extra+'\n}  // namespace mjpc');p.write_text(t)
 diff=[]
 for rel in ['mjpc/planners/ilqg/'+x for x in ['policy.h','policy.cc','planner.h','planner.cc']]:
  diff+=difflib.unified_diff((BASE/rel).read_text().splitlines(True),(DEST/rel).read_text().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)
 (ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r37.patch').write_text(''.join(diff))
if __name__=='__main__':main()
