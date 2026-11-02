  if (record_transition) {
    std::ofstream out(diagnostic_file,std::ios::app);
    out.precision(17);
    out<<"{\"origin\":"<<time<<",\"winner\":"<<winner
       <<",\"accepted_cost\":"<<trajectory[winner].total_return
       <<",\"executed_reference_cost\":"<<policy.trajectory.total_return
       <<",\"feedback_scaling\":"<<policy.feedback_scaling
       <<",\"nx\":"<<dim_state_derivative<<",\"ns\":"<<dim_sensor
       <<",\"nr\":"<<task->num_residual<<",\"nodes\":[";
    auto array=[&out](const char* key,const double* x,int n) {
      out<<",\""<<key<<"\":[";
      for(int j=0;j<n;++j){if(j)out<<",";out<<x[j];}out<<"]";
    };
    for(int k=0;k<record_nodes;++k) {
      if(k)out<<",";
      out<<"{\"t\":"<<policy.trajectory.times[k];
      array("linearization_state",linearization_state.data()+k*dim_state,dim_state);
      array("linearization_residual",linearization_residual.data()+k*task->num_residual,task->num_residual);
      array("accepted_state",trajectory[winner].states.data()+k*dim_state,dim_state);
      array("executed_reference_state",policy.trajectory.states.data()+k*dim_state,dim_state);
      array("accepted_action",trajectory[winner].actions.data()+k*dim_action,dim_action);
      array("executed_nominal_action",policy.trajectory.actions.data()+k*dim_action,dim_action);
      array("K",policy.feedback_gain.data()+k*dim_action*dim_state_derivative,dim_action*dim_state_derivative);
      array("C",model_derivative.C.data()+k*dim_sensor*dim_state_derivative,dim_sensor*dim_state_derivative);
      array("D",model_derivative.D.data()+k*dim_sensor*dim_action,dim_sensor*dim_action);
      out<<"}";
    }
    out<<"]}\n";
  }

