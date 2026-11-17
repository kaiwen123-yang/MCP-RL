// W53 recording only. Called after the ordinary restored iLQG iteration.
// No extra optimization, rollout or finite difference is performed here.
void iLQGPlanner::W53Record(int horizon) {
  const char* directory=std::getenv("MJPC_W54_N_RECORD_DIR");
  if(!directory)return;
  const auto begin=std::chrono::steady_clock::now();const auto cpu=std::clock();
  ++w53_call;
  bool requested=std::filesystem::exists(std::string(directory)+"/full-ring.request");
  bool first_move=time>=2.0&&!w53_moving_recorded;
  bool full=w53_call==1||first_move||requested||w54_controller_infeasible;
  if(first_move)w53_moving_recorded=true;
  if(requested)std::filesystem::remove(std::string(directory)+"/full-ring.request");
  auto write=[&](const std::string& path,const iLQGPolicy& p,const Trajectory& tr) {
    std::ofstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("W53 N installed-policy file open failed");
    int64_t hdr[10]={54,3,(int64_t)w53_call,horizon,dim_state,dim_state_derivative,dim_action,task->num_residual,task->num_trace,p.representation};
    f.write((const char*)hdr,sizeof(hdr));
    double info[4]={time,p.feedback_scaling,p.trajectory.total_return,tr.total_return};f.write((const char*)info,sizeof(info));
    auto a=[&](const std::vector<double>& x,size_t count){f.write((const char*)x.data(),count*sizeof(double));};
    a(p.trajectory.times,horizon);a(p.trajectory.states,horizon*dim_state);a(p.trajectory.actions,horizon*dim_action);
    a(p.feedback_gain,horizon*dim_action*dim_state_derivative);a(p.action_improvement,horizon*dim_action);
    a(tr.times,horizon);a(tr.states,horizon*dim_state);a(tr.actions,horizon*dim_action);
    a(tr.residual,horizon*task->num_residual);a(tr.costs,horizon);a(tr.trace,horizon*3*task->num_trace);
    a(tr.w54_applied_actions,horizon*dim_action);a(tr.w54_execution_diagnostics,horizon*40);
    f.close();if(!f)throw std::runtime_error("W54 N installed-policy write failed");
  };
  char name[80];
  if(w53_installed||w53_call==1) {
    w53_policy_call=w53_call;
    std::snprintf(name,sizeof(name),"/policy-%06lu.bin",w53_call);
    write(std::string(directory)+name,policy,w53_installed?trajectory[winner]:policy.trajectory);
  }
  if(full)for(int i=0;i<num_trajectory_;++i) {
    std::snprintf(name,sizeof(name),"/call-%06lu-candidate-%02d.bin",w53_call,i);
    write(std::string(directory)+name,candidate_policy[i],trajectory[i]);
  }
  std::ofstream out(std::string(directory)+"/calls.jsonl",std::ios::app);out<<std::setprecision(17);
  out<<"{\"call_id\":"<<w53_call<<",\"time_s\":"<<time<<",\"policy_call_id\":"<<w53_policy_call
     <<",\"controller_infeasible\":"<<(w54_controller_infeasible?"true":"false")<<",\"installed\":"<<(w53_installed?"true":"false")<<",\"winner\":"<<(w53_installed?winner:-1)<<",\"full_candidates\":"<<(full?"true":"false")
     <<",\"num_candidates\":"<<num_trajectory_<<",\"full_rollouts\":"<<w53_rollouts<<",\"transition_fd_calls\":"<<((horizon-1)*(1+dim_state_derivative+dim_action)+(1+dim_state_derivative))
     <<",\"wall_s\":{\"nominal\":"<<nominal_compute_time*1e-6<<",\"derivative\":"<<model_derivative_compute_time*1e-6
     <<",\"cost\":"<<cost_derivative_compute_time*1e-6<<",\"backward\":"<<backward_pass_compute_time*1e-6
     <<",\"rollouts\":"<<rollouts_compute_time*1e-6<<",\"policy_update\":"<<policy_update_compute_time*1e-6<<"},\"cpu_s\":[";
  for(int i=0;i<6;++i){if(i)out<<",";out<<w53_cpu[i];}out<<"],\"candidates\":[";
  for(int i=0;i<num_trajectory_;++i){if(i)out<<",";auto& tr=trajectory[i];
    double min_gap=1.e9,incident=0,complete_tau=0,active_nodes=0;
    for(int k=0;k<horizon-1;k++){
      for(int foot=0;foot<4;foot++){
        const double* e=tr.w54_execution_diagnostics.data()+k*40+foot*10;
        if(e[7]>0){min_gap=std::min(min_gap,e[0]);if(e[0]<=.001)incident=std::max(incident,std::max(0.,-e[1]));}
        if(e[4]>0)active_nodes+=1;
      }
      for(int j=0;j<dim_action;j++){
        const int joint=model->actuator_trnid[2*j];
        const double q=tr.states[k*dim_state+model->jnt_qposadr[joint]];
        const double v=tr.states[k*dim_state+model->nq+model->jnt_dofadr[joint]];
        complete_tau=std::max(complete_tau,std::abs(60*(tr.w54_applied_actions[k*dim_action+j]-q)-5*v));
      }
    }
    out<<"{\"slot\":"<<i<<",\"J\":"<<(std::isfinite(tr.total_return)?tr.total_return:0)
       <<",\"J_finite\":"<<(std::isfinite(tr.total_return)?"true":"false")<<",\"failure\":"<<(tr.failure?"true":"false")
       <<",\"dx_m\":"<<tr.states[(horizon-1)*dim_state]-tr.states[0]<<",\"final_z_m\":"<<tr.states[(horizon-1)*dim_state+2];
    out<<",\"E_predicted_min_material_gap_m\":"<<(min_gap<1.e9?min_gap:0)
       <<",\"E_material_nodes_present\":"<<(min_gap<1.e9?"true":"false")
       <<",\"E_predicted_near_surface_descent_max_m_s\":"<<incident
       <<",\"E_predicted_complete_PD_request_max_Nm\":"<<complete_tau
       <<",\"E_active_foot_nodes\":"<<active_nodes;
    out<<",\"predicted_domain_feasible\":"<<(tr.w54_domain_feasible?"true":"false")<<",\"predicted_domain_min_gap_m\":"<<tr.w54_domain_min_gap;
    out<<"}";
  }
  out<<"],\"record_wall_s\":"<<GetDuration(begin)*1e-6<<",\"record_cpu_s\":"<<double(std::clock()-cpu)/CLOCKS_PER_SEC<<"}\n";
  out.close();if(!out)throw std::runtime_error("W53 N call record write failed");
}
