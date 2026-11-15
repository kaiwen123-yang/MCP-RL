#!/usr/bin/env python3
"""Generate the W50 native single-alpha reference/feedback/metric transfer."""
from pathlib import Path
import difflib,shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'work/r11-frozen/S5-r37'
DEST=ROOT/'work/w50-implementation/source'
def change(t,old,new,n=1):
 assert t.count(old)==n,(old[:100],t.count(old),n)
 return t.replace(old,new)
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 shutil.copytree(BASE/'mjpc',DEST/'mjpc',dirs_exist_ok=True)
 ph=DEST/'mjpc/planners/ilqg/policy.h';t=ph.read_text()
 t=change(t,'bool* restored=nullptr) const;','bool* restored=nullptr, bool raw_only=false) const;')
 t=change(t,'  void Save(const std::string& path) const;', '''  void Save(const std::string& path) const;
  // Fixed-size provenance, independent of complete-policy installation identity.
  // Unknown historical source fields are -1, never fabricated zero-age sources.
  std::vector<double> provenance;
  void ResetProvenance();
  void WriteProvenance(std::ostream& out) const;
  void ReadProvenance(std::istream& in);
  void ProvenanceJSON(std::ostream& out,double query_time) const;
  std::string MetricRegularizationJSON() const;
  void RawCenter(double* action,const double* state,double time) const;
''')
 ph.write_text(t)
 pc=DEST/'mjpc/planners/ilqg/policy.cc';t=pc.read_text()
 t=change(t,'#include <fstream>','#include <fstream>\n#include <sstream>')
 t=change(t,'  ClearDelta();\n  metric_valid=false;', '  ClearDelta();ResetProvenance();\n  metric_valid=false;')
 t=change(t,'bool* failed, bool* restored) const {','bool* failed, bool* restored, bool raw_only) const {')
 t=change(t,'  std::vector<double> before_clip;','  if(raw_only)return;  // Shared exact pre-clamp/pre-QP center, no integration.\n  std::vector<double> before_clip;')
 t=change(t,'       <<",\\"metric_source_plan_id\\":"<<record_plan_id\n       <<",\\"metric_regularization\\":"<<metric_regularization', '       <<",\\"metric_source_plan_id\\":null"\n       <<",\\"metric_regularization\\":"<<MetricRegularizationJSON()')
 t=change(t,'    out<<"}\\n";','    ProvenanceJSON(out,time);\n    out<<"}\\n";')
 t=change(t,'  delta_times=policy.delta_times;delta_values=policy.delta_values;','  provenance=policy.provenance;\n  delta_times=policy.delta_times;delta_values=policy.delta_values;')
 t=change(t,'  double h[]={49.,double(record_plan_id)', '  double h[]={50.,double(record_plan_id)')
 t=change(t,'  WriteDelta(out);\n}', '  WriteDelta(out);WriteProvenance(out);\n}')
 extra=r'''
void iLQGPolicy::RawCenter(double* action,const double* state,double t) const {
  ActionImpl(action,state,t,false,nullptr,nullptr,nullptr,true);
}
void iLQGPolicy::ResetProvenance() {
  provenance.assign(32,-1.);provenance[0]=50;provenance[1]=0;
}
void iLQGPolicy::WriteProvenance(std::ostream& out) const {
  out.write(reinterpret_cast<const char*>(provenance.data()),32*sizeof(double));
}
void iLQGPolicy::ReadProvenance(std::istream& in) {
  provenance.resize(32);in.read(reinterpret_cast<char*>(provenance.data()),32*sizeof(double));
  if(!in || provenance[0]!=50)mju_error("W50 policy provenance truncated or wrong schema");
}
std::string iLQGPolicy::MetricRegularizationJSON() const {
  if(metric_regularization<0)return "null";  // A mixed H has no single generating mu.
  std::ostringstream s;s.precision(17);s<<metric_regularization;return s.str();
}
void iLQGPolicy::ProvenanceJSON(std::ostream& out,double query_time) const {
  out<<",\"reference_start\":"<<trajectory.times[0]
     <<",\"reference_end\":"<<trajectory.times[trajectory.horizon-1]
     <<",\"reference_covers_query\":"<<((query_time>=trajectory.times[0] && query_time<=trajectory.times[trajectory.horizon-1])?"true":"false")
     <<",\"reference_expired_by_s\":"<<std::max(0.,query_time-trajectory.times[trajectory.horizon-1])
     <<",\"feedback_provenance\":[";
  for(int i=0;i<32;++i){if(i)out<<",";out<<provenance[i];}
  out<<"],\"provenance_unknown_value\":-1,\"derivative_fields_role\":\"captured_backward_not_mixed_derivatives\"";
}
'''
 t=change(t,'}  // namespace mjpc',extra+'\n}  // namespace mjpc');pc.write_text(t)
 h=DEST/'mjpc/planners/ilqg/planner.h';t=h.read_text()
 t=change(t,'  void W49Guided(int horizon,ThreadPool& pool);','''  void W49Guided(int horizon,ThreadPool& pool);
  iLQGPolicy w50_backward;
  std::vector<double> w50_centers,w50_jacobian,w50_metric;
  long w50_raw_center_evaluations=0;
  double w50_mu=0,w50_rate=0;
  void W50Capture(int horizon);
  void W50Transfer(int horizon,ThreadPool& pool);
''');h.write_text(t)
 p=DEST/'mjpc/planners/ilqg/planner.cc';t=p.read_text()
 t=change(t,'  policy.ClearDelta();\n  policy.record_plan_id=', '  policy.ClearDelta();policy.ResetProvenance();\n  policy.record_plan_id=')
 t=change(t,'  if(int(h[0])==49)policy.ReadDelta(in);', '''  if(int(h[0])==49 || int(h[0])==50)policy.ReadDelta(in);
  if(int(h[0])==50)policy.ReadProvenance(in);
  else {policy.provenance[5]=tr.times[0];policy.provenance[6]=tr.times[horizon-1];}
''')
 t=change(t,'r33_mechanism!="K" && r33_mechanism!="R32"','r33_mechanism!="K" && r33_mechanism!="H" && r33_mechanism!="R32"')
 t=change(t,'  R33Import(horizon);\n  if(std::getenv("MJPC_W49_QUERY_ONLY"))', '''  R33Import(horizon);
  if(std::getenv("MJPC_W50_INSTALL_ONCE")) {
    if(const char* dest=std::getenv("MJPC_W49_EXPORT"))policy.Save(dest);
    r32.deployable=policy.metric_valid;r29_return_reason="w50_imported_once";
    // Diagnostic import is consumed. The next actual state/time call uses the
    // ongoing task gait and a fresh R35 reference, with no old context restore.
    for(const char* name:{"MJPC_W49_QUERY_ONLY","MJPC_R33_INPUT_POLICY",
       "MJPC_R34_CONTEXT","MJPC_R35_REFERENCE","MJPC_W50_INSTALL_ONCE"})unsetenv(name);
    return;
  }
  if(std::getenv("MJPC_W49_QUERY_ONLY"))''')
 t=change(t,'    policy.WriteDelta(out);','    policy.WriteDelta(out);policy.WriteProvenance(out);',2)
 t=change(t,'  w49_first_valid=false;w49_k.clear();w49_times.clear();','  w49_first_valid=false;w49_k.clear();w49_times.clear();w50_raw_center_evaluations=0;')
 t=change(t,'    if(std::getenv("MJPC_W49_FINITE_GUIDED") && w49_first_valid)W49Guided(horizon,pool);','    if(r33_mechanism!="H" && std::getenv("MJPC_W49_FINITE_GUIDED") && w49_first_valid)W49Guided(horizon,pool);')
 t=change(t,'if(r33_mechanism=="N" || r33_mechanism=="K") {','if(r33_mechanism=="N" || r33_mechanism=="K" || r33_mechanism=="H") {')
 t=change(t,'    if(!w49_first_valid)W49Capture(horizon);','''    if(r33_mechanism=="H") {
      W50Capture(horizon);
      R32RecordAttempt("w50_first_valid_direction",horizon,-1,r32_nominal,
                       regularization_used,rate_used,0.);
      auto transfer_start=std::chrono::steady_clock::now();
      W50Transfer(horizon,pool);
      r32.stopped=true;r32.stop_reason="w50_single_first_direction_finished";
      model_derivative_compute_time=model_derivative_time;
      cost_derivative_compute_time=cost_derivative_time;
      backward_pass_compute_time=backward_pass_time;
      rollouts_compute_time=GetDuration(transfer_start);
      return;
    }
    if(!w49_first_valid)W49Capture(horizon);''')
 t=t.replace('S5-r37','S5-r38')
 t=t.replace('<<policy.metric_regularization','<<policy.MetricRegularizationJSON()')
 t=change(t,'       <<",\\"native_new_rollouts\\":"<<r33_rollouts', '       <<",\\"raw_center_evaluations\\":"<<w50_raw_center_evaluations\n       <<",\\"native_new_rollouts\\":"<<r33_rollouts')
 t=change(t,'    out<<"],\\"r34_enabled\\":"<<(r34_enabled?"true":"false")', '    out<<"]";policy.ProvenanceJSON(out,time);\n    out<<",\\"r34_enabled\\":"<<(r34_enabled?"true":"false")')
 # Export the selected COMPLETE object only after the archive has installed it.
 t=change(t,'    out<<"]}\\n";\n  }\n}\n\n// compute trajectory using nominal policy', '''    out<<"]}\\n";
  }
  if(const char* dest=std::getenv("MJPC_W49_EXPORT"))policy.Save(dest);
}

// compute trajectory using nominal policy''')
 extra=r'''
void iLQGPlanner::W50Capture(int horizon) {
  // One immutable capture before any alpha modifies any candidate array.
  w49_first_valid=true;w49_base=r33_nominal_policy;w50_backward=candidate_policy[0];
  w50_backward.feedback_scaling=1.;
  w50_mu=backward_pass.regularization;w50_rate=backward_pass.regularization_rate;
  w49_times.assign(w50_backward.trajectory.times.begin(),w50_backward.trajectory.times.begin()+horizon-1);
  w49_k.assign(w50_backward.action_improvement.begin(),w50_backward.action_improvement.begin()+(horizon-1)*dim_action);
  w50_backward.ResetProvenance();
  auto& np=w50_backward.provenance;
  np[1]=1;np[2]=np[7]=np[10]=np[27]=r32.epoch;
  np[3]=np[8]=np[11]=np[28]=r29_call_id;
  np[4]=np[9]=np[12]=np[29]=time;
  np[5]=w50_backward.trajectory.times[0];np[6]=w50_backward.trajectory.times[horizon-1];
  np[24]=w50_mu;np[25]=w50_rate;np[26]=r29_backward_attempts;
  w50_centers.resize((horizon-1)*dim_action);
  w50_jacobian.resize((horizon-1)*dim_action*dim_state_derivative);
  w50_metric.resize((horizon-1)*dim_action*dim_action);
  if(w49_base.metric_valid) {
    // Native MuJoCo tangent central differences, at the fixed original FD scale.
    // These evaluate only the control map; no forward/dynamics/physics call.
    std::vector<double> plus(dim_state),minus(dim_state),up(dim_action),um(dim_action),dq(model->nv);
    const double eps=settings.fd_tolerance;
    for(int k=0;k<horizon-1;++k) {
      const double* x=r33_nominal_trajectory.states.data()+k*dim_state;
      const double t=r33_nominal_trajectory.times[k];
      w49_base.RawCenter(w50_centers.data()+k*dim_action,x,t);++w50_raw_center_evaluations;
      LinearInterpolation(w50_metric.data()+k*dim_action*dim_action,t,
        w49_base.trajectory.times,w49_base.metric.data(),dim_action*dim_action,w49_base.trajectory.horizon-1);
      for(int j=0;j<dim_state_derivative;++j) {
        std::copy_n(x,dim_state,plus.data());std::copy_n(x,dim_state,minus.data());
        if(j<model->nv) {
          std::fill(dq.begin(),dq.end(),0.);dq[j]=1.;
          mj_integratePos(model,plus.data(),dq.data(),eps);
          mj_integratePos(model,minus.data(),dq.data(),-eps);
        } else {
          const int index=model->nq+j-model->nv;
          plus[index]+=eps;minus[index]-=eps;
        }
        w49_base.RawCenter(up.data(),plus.data(),t);w49_base.RawCenter(um.data(),minus.data(),t);
        w50_raw_center_evaluations+=2;
        for(int a=0;a<dim_action;++a)
          w50_jacobian[(k*dim_action+a)*dim_state_derivative+j]=(up[a]-um[a])/(2*eps);
      }
    }
  }
  if(const char* prefix=std::getenv("MJPC_W49_DIRECTION")) {
    w49_base.Save(std::string(prefix)+"-baseline.bin");
    w50_backward.Save(std::string(prefix)+"-backward.bin");
    std::ofstream out(std::string(prefix)+"-direction.bin",std::ios::binary|std::ios::trunc);
    double h[]={50.,double(r29_call_id),time,double(horizon-1),double(dim_action),
      double(r29_backward_attempts),w50_mu,w50_rate,
      r33_nominal_trajectory.total_return,r33_nominal_trajectory.trunk_violation,R34Budget(r33_nominal_trajectory)};
    out.write(reinterpret_cast<const char*>(h),sizeof(h));
    for(const auto* v:{&w49_times,&w49_k})out.write(reinterpret_cast<const char*>(v->data()),v->size()*sizeof(double));
    std::ofstream center(std::string(prefix)+"-centers.bin",std::ios::binary|std::ios::trunc);
    double ch[]={50.,double(horizon-1),double(dim_action),double(dim_state_derivative),
      double(w50_raw_center_evaluations),settings.fd_tolerance,double(w49_base.metric_valid)};
    center.write(reinterpret_cast<const char*>(ch),sizeof(ch));
    if(w49_base.metric_valid)for(const auto* v:{&w50_centers,&w50_jacobian,&w50_metric})
      center.write(reinterpret_cast<const char*>(v->data()),v->size()*sizeof(double));
  }
}
void iLQGPlanner::W50Transfer(int horizon,ThreadPool& pool) {
  const bool transfer=w49_base.metric_valid;
  const int count=num_trajectory_-1,before=pool.GetCount();
  // The implicit alpha=0 is exactly w49_base and its cached native score.
  // No rebasing, new rollout, metric construction or alpha-zero installation.
  for(int i=0;i<count;++i) {
    const double alpha=linesearch_steps[i];
    auto& p=candidate_policy[i];p=w50_backward;
    p.trajectory=r33_nominal_trajectory;p.ClearDelta();
    p.feedback_scaling=1.;p.nominal_feedback_scale=1.;p.action_step_alpha=alpha;
    p.provenance[13]=alpha;p.provenance[14]=w49_base.record_plan_id;
    p.provenance[15]=w49_base.provenance[4];
    for(int j=0;j<3;++j) {
      p.provenance[16+j]=w49_base.provenance[7+j];
      p.provenance[19+j]=w49_base.provenance[10+j];
    }
    p.provenance[22]=w49_base.metric_regularization;
    p.provenance[23]=w49_base.provenance[1];
    p.provenance[30]=w49_base.feedback_scaling;
    p.provenance[31]=w50_raw_center_evaluations;
    if(transfer) {
      p.provenance[1]=2;p.metric_regularization=-1.;
      for(int k=0;k<horizon-1;++k) {
        for(int a=0;a<dim_action;++a)
          p.trajectory.actions[k*dim_action+a]=w50_centers[k*dim_action+a]+alpha*w49_k[k*dim_action+a];
        for(int j=0;j<dim_action*dim_state_derivative;++j) {
          int n=k*dim_action*dim_state_derivative+j;
          p.feedback_gain[n]=(1-alpha)*w50_jacobian[n]+alpha*w50_backward.feedback_gain[n];
        }
        for(int j=0;j<dim_action*dim_action;++j) {
          int n=k*dim_action*dim_action+j;
          p.metric[n]=(1-alpha)*w50_metric[n]+alpha*w50_backward.metric[n];
        }
      }
    } else {
      // Initial policy has no valid H: retain the original native bootstrap
      // family, bounded to this same first valid direction and nine positives.
      for(int j=0;j<(horizon-1)*dim_action;++j)p.trajectory.actions[j]+=alpha*w49_k[j];
    }
    for(auto [v,width]:{std::pair{&p.trajectory.actions,dim_action},
       std::pair{&p.feedback_gain,dim_action*dim_state_derivative},std::pair{&p.metric,dim_action*dim_action}})
      std::copy_n(v->data()+(horizon-2)*width,width,v->data()+(horizon-1)*width);
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
    R33Consider(candidate_policy[i],trajectory[i],horizon,1,i,linesearch_steps[i],
      std::string(transfer?"homotopy_":"bootstrap_")+std::to_string(i));
  backward_pass.regularization=w50_mu;backward_pass.regularization_rate=w50_rate;
  if(r33_source.rfind(transfer?"homotopy_":"bootstrap_",0)==0 &&
     (r33_nominal_trajectory.trunk_violation>0 || r33_archive_trajectory.trunk_violation>0))
    backward_pass.ScaleRegularization(
      r33_nominal_trajectory.trunk_violation>r33_archive_trajectory.trunk_violation?
        1./backward_pass.regularization_factor:backward_pass.regularization_factor,
      settings.min_regularization,settings.max_regularization);
  if(const char* prefix=std::getenv("MJPC_W49_DIRECTION")) {
    std::ofstream out(std::string(prefix)+"-homotopy.json",std::ios::trunc);out.precision(17);
    out<<"{\"first_valid_backward\":true,\"baseline_metric_valid\":"<<(transfer?"true":"false")
       <<",\"extra_candidates\":"<<count<<",\"baseline_J\":"<<r33_nominal_trajectory.total_return
       <<",\"baseline_risk\":"<<r33_nominal_trajectory.trunk_violation
       <<",\"baseline_budget\":"<<R34Budget(r33_nominal_trajectory)
       <<",\"expected\":null,\"ratio\":null,\"backward_mu\":"<<w50_mu<<",\"mu_before\":"<<w50_mu
       <<",\"rate_before\":"<<w50_rate<<",\"backward_attempt\":"<<r29_backward_attempts
       <<",\"raw_center_evaluations\":"<<w50_raw_center_evaluations
       <<",\"raw_center_physics_steps\":0,\"fd_scale\":"<<settings.fd_tolerance
       <<",\"reference_start\":"<<r33_nominal_trajectory.times[0]
       <<",\"reference_end\":"<<r33_nominal_trajectory.times[horizon-1]
       <<",\"mu_after\":"<<backward_pass.regularization<<",\"rate_after\":"<<backward_pass.regularization_rate
       <<",\"winner_source\":\""<<r33_source<<"\",\"alpha0_cached_unmodified\":true,\"candidates\":[";
    for(int i=0;i<count;++i) {
      if(i)out<<",";
      out<<"{\"slot\":"<<i<<",\"alpha\":"<<linesearch_steps[i]<<",\"J\":"<<trajectory[i].total_return
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
 (ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r38.patch').write_text(''.join(diff))
 print('W50 source generated; no build or physics')
if __name__=='__main__':main()
