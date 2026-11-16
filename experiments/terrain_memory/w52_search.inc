void iLQGPlanner::W52Search(int horizon,ThreadPool& pool) {
  const double entry_mu=backward_pass.regularization,entry_rate=backward_pass.regularization_rate;
  const auto S=w51_Q;const auto Str=w51_Qtr;
  const int count=w51_seed_valid?2:1;
  struct Result {bool valid=false;double mu=0,rate=0;int attempts=0;};
  Result result[2];std::vector<W52Candidate> candidates[2];
  if(const char* f=std::getenv("MJPC_W51_RECORD")) {
    std::ofstream out(std::string(f)+"-branches.json",std::ios::trunc);
    out<<"{\"call_id\":"<<r29_call_id<<",\"status\":\"computing\"}\n";
  }
  for(int n=0;n<count;++n) {
    if(!w52_workspaces[n]) {
      auto b=std::make_unique<iLQGPlanner>();b->Initialize(model,*task);
      b->num_trajectory_=num_trajectory_;b->derivative_skip_=derivative_skip_;
      // Independent complete branch storage, sized only for this actual horizon.
      b->model_derivative.Allocate(dim_state_derivative,dim_action,dim_sensor,horizon);
      b->cost_derivative.Allocate(dim_state_derivative,dim_action,task->num_residual,horizon,dim_max);
      b->backward_pass.Allocate(dim_state_derivative,dim_action,horizon);
      b->boxqp.Allocate(dim_action);
      for(int i=0;i<num_trajectory_;++i) {
        b->trajectory[i].Initialize(dim_state,dim_action,task->num_residual,task->num_trace,horizon);
        b->trajectory[i].Allocate(horizon);b->candidate_policy[i].Allocate(model,*task,horizon);
      }
      w52_workspaces[n]=std::move(b);
    }
    auto& b=*w52_workspaces[n];
    b.state=state;b.time=time;b.mocap=mocap;b.userdata=userdata;b.settings=settings;
    b.policy=policy;b.previous_policy=previous_policy;b.r32=r32;b.r32.attempts.clear();
    b.r32.stopped=false;b.r32.stop_reason="none";b.r32.iterations=0;
    b.r33_mechanism=r33_mechanism;b.r34_enabled=r34_enabled;b.r35_epoch=r35_epoch;
    b.r33_nominal_policy=r33_nominal_policy;b.r33_nominal_trajectory=r33_nominal_trajectory;
    b.r33_archive_valid=false;b.r33_source="none";b.r33_rollouts=b.r33_derivative_passes=0;
    b.r33_D.clear();b.r33_scale_mean=b.r33_scale_floor=0;
    b.r29_call_id=r29_call_id;b.r29_backward_attempts=0;b.r29_return_reason="entered";
    b.model_derivative.Reset(dim_state_derivative,dim_action,dim_sensor,horizon);
    b.model_derivative.r33_transition_fd_calls=b.model_derivative.r33_risk_fd_calls=0;
    b.cost_derivative.Reset(dim_state_derivative,dim_action,task->num_residual,horizon);
    b.backward_pass.Reset(dim_state_derivative,dim_action,horizon);
    b.backward_pass.regularization=entry_mu;b.backward_pass.regularization_rate=entry_rate;
    b.backward_pass.regularization_factor=backward_pass.regularization_factor;
    b.w49_first_valid=false;b.w49_k.clear();b.w49_times.clear();b.w50_raw_center_evaluations=0;
    b.w52_collect=true;b.w52_branch=n?"S":"B";b.w52_pending.clear();
    b.w51_Q=n?S:r33_nominal_policy;b.w51_Qtr=n?Str:r33_nominal_trajectory;
    b.candidate_policy[0]=b.w51_Q;b.candidate_policy[0].trajectory=b.w51_Qtr;
    b.candidate_policy[0].ClearDelta();b.feedback_scaling=b.w51_Q.feedback_scaling;
    b.expected=b.improvement=b.surprise=0;
    b.Iteration(horizon,pool); // Only this branch's storage and the shared bounded pool.
    candidates[n]=std::move(b.w52_pending);
    result[n]={b.w49_first_valid,b.w50_mu,b.w50_rate,b.r29_backward_attempts};
    r33_rollouts+=b.r33_rollouts;r33_derivative_passes+=b.r33_derivative_passes;
    r29_backward_attempts+=b.r29_backward_attempts;w50_raw_center_evaluations+=b.w50_raw_center_evaluations;
    model_derivative.r33_transition_fd_calls+=b.model_derivative.r33_transition_fd_calls;
    model_derivative.r33_risk_fd_calls+=b.model_derivative.r33_risk_fd_calls;
    model_derivative_compute_time+=b.model_derivative_compute_time;
    cost_derivative_compute_time+=b.cost_derivative_compute_time;
    backward_pass_compute_time+=b.backward_pass_compute_time;rollouts_compute_time+=b.rollouts_compute_time;
    for(const auto& attempt:b.r32.attempts)r32.attempts.push_back(
      "{\"branch\":\""+b.w52_branch+"\","+attempt.substr(1));
    r32.iterations+=b.r32.iterations;
    if(const char* f=std::getenv("MJPC_W51_RECORD")) {
      if(b.w49_first_valid) {
        b.w50_backward.Save(std::string(f)+"-backward-"+b.w52_branch+".bin");
        std::ofstream out(std::string(f)+"-centers-"+b.w52_branch+".bin",std::ios::binary|std::ios::trunc);
        double h[]={52.,double(r29_call_id),double(horizon-1),double(dim_action),double(dim_state_derivative)};
        out.write(reinterpret_cast<char*>(h),sizeof(h));
        for(const auto* v:{&b.w49_times,&b.w49_k,&b.w50_centers,&b.w50_jacobian,&b.w50_metric})
          out.write(reinterpret_cast<const char*>(v->data()),v->size()*sizeof(double));
      }
    }
  }
  // One deterministic global merge: retained B/nominal, B positives, direct S,
  // S positives. Completion order never affects ties.
  for(const auto& c:candidates[0])R33Consider(c.policy,c.trajectory,horizon,c.kind,c.slot,c.alpha,"B_"+c.source);
  for(const auto& c:w52_pending)R33Consider(c.policy,c.trajectory,horizon,c.kind,c.slot,c.alpha,c.source);
  w52_pending.clear();
  if(count==2)for(const auto& c:candidates[1])
    R33Consider(c.policy,c.trajectory,horizon,c.kind,c.slot,c.alpha,"S_"+c.source);
  const int selected=(r33_source=="motion_seed_S" || r33_source.rfind("S_",0)==0)?1:0;
  backward_pass.regularization=result[selected].valid?result[selected].mu:entry_mu;
  backward_pass.regularization_rate=result[selected].valid?result[selected].rate:entry_rate;
  const double committed_mu=backward_pass.regularization,committed_rate=backward_pass.regularization_rate;
  const bool positive=r33_source.rfind("B_",0)==0 || r33_source.rfind("S_",0)==0;
  if(positive && (r33_nominal_trajectory.trunk_violation>0 || r33_archive_trajectory.trunk_violation>0))
    backward_pass.ScaleRegularization(
      r33_nominal_trajectory.trunk_violation>r33_archive_trajectory.trunk_violation?
      1./backward_pass.regularization_factor:backward_pass.regularization_factor,
      settings.min_regularization,settings.max_regularization);
  r32.stopped=true;r32.stop_reason="w52_independent_directions_finished";
  w49_first_valid=result[0].valid || result[1].valid;expected=surprise=0;
  if(const char* f=std::getenv("MJPC_W51_RECORD")) {
    std::ofstream out(std::string(f)+"-branches.json",std::ios::trunc);out.precision(17);
    out<<"{\"call_id\":"<<r29_call_id<<",\"epoch\":"<<r32.epoch<<",\"time_s\":"<<time
       <<",\"entry_mu\":"<<entry_mu<<",\"entry_rate\":"<<entry_rate
       <<",\"selected_source\":\""<<r33_source<<"\",\"committed_branch\":\""<<(selected?"S":"B")
       <<"\",\"committed_valid\":"<<(result[selected].valid?"true":"false")
       <<",\"committed_mu\":"<<committed_mu<<",\"committed_rate\":"<<committed_rate
       <<",\"final_mu\":"<<backward_pass.regularization<<",\"final_rate\":"<<backward_pass.regularization_rate
       <<",\"final_risk_adjustment_count\":"<<(positive && (r33_nominal_trajectory.trunk_violation>0 || r33_archive_trajectory.trunk_violation>0)?1:0)
       <<",\"expected\":null,\"ratio\":null,\"branches\":[";
    for(int n=0;n<count;++n) {
      if(n)out<<",";const auto& b=*w52_workspaces[n];
      out<<"{\"source\":\""<<(n?"S":"B")<<"\",\"first_valid\":"<<(result[n].valid?"true":"false")
         <<",\"mu\":"<<(result[n].valid?result[n].mu:entry_mu)
         <<",\"rate\":"<<(result[n].valid?result[n].rate:entry_rate)
         <<",\"backward_attempts\":"<<result[n].attempts<<",\"positive_rollouts\":"<<b.r33_rollouts
         <<",\"transition_fd\":"<<b.model_derivative.r33_transition_fd_calls
         <<",\"risk_fd\":"<<b.model_derivative.r33_risk_fd_calls<<"}";
    }
    out<<"]}\n";
  }
}
