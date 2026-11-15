void iLQGPlanner::W51Seed(int horizon,ThreadPool& pool) {
  w51_Q=r33_nominal_policy;w51_Qtr=r33_nominal_trajectory;
  w51_status="stand_or_terminal";w51_seed_valid=false;w51_seed_rollouts=0;
  const auto* q=dynamic_cast<const QuadrupedFlat*>(task);
  if(!q || !q->w51_motion.active || !r33_nominal_policy.metric_valid)return;
  const auto& motion=q->w51_motion;
  mjData* actual=mj_makeData(model);mjData* ik=mj_makeData(model);
  mju_copy(actual->qpos,state.data(),model->nq);
  mju_copy(actual->qvel,state.data()+model->nq,model->nv);
  mju_copy(actual->userdata,userdata.data(),model->nuserdata);actual->time=time;
  mj_kinematics(model,actual);mj_comPos(model,actual);
  mju_copy(ik->qpos,actual->qpos,model->nq);
  std::vector<double> knots(horizon-1),delta((horizon-1)*dim_action),jac(3*model->nv);
  std::array<std::vector<int>,4> actuators;
  for(int i=0;i<4;++i)for(int a=0;a<dim_action;++a) {
    int body=model->geom_bodyid[motion.geoms[i]],joint=model->actuator_trnid[2*a];
    while(body>0 && body!=model->jnt_bodyid[joint])body=model->body_parentid[body];
    if(body==model->jnt_bodyid[joint])actuators[i].push_back(a);
  }
  std::array<std::array<double,3>,4> start,velocity,endpos;
  std::array<double,4> begin,end,step0,dstep0;
  std::array<bool,4> swing;
  double first_step[4],first_velocity[4];q->W51Step(time,first_step,first_velocity);
  auto local=motion.feet;
  for(int i=0;i<4;++i) {
    if(actuators[i].size()!=3)mju_error("W51 requires original three-joint Go2 legs");
    mj_jacGeom(model,actual,jac.data(),nullptr,motion.geoms[i]);
    for(int j=0;j<3;++j) {
      start[i][j]=actual->geom_xpos[3*motion.geoms[i]+j];
      velocity[i][j]=mju_dot(jac.data()+j*model->nv,actual->qvel,model->nv);
    }
    begin[i]=time;end[i]=local[i].end;swing[i]=local[i].swing;
    endpos[i]=swing[i]?local[i].touchdown:local[i].anchor;
    step0[i]=first_step[i];dstep0[i]=first_velocity[i];
  }
  bool valid=true;int failed_node=-1;double max_error=0;
  std::vector<double> targets((horizon-1)*12),ik_path((horizon-1)*12);
  for(int k=0;k<horizon-1 && valid;++k) {
    const double t=r33_nominal_trajectory.times[k],tau=t-time;
    knots[k]=t;
    for(int j=0;j<2;++j)ik->qpos[j]=actual->qpos[j]+actual->userdata[78+j]*tau;
    const double* height=GetCustomNumericData(model,"r10_material_support_height");
    const double ztarget=height?height[0]:actual->userdata[95];
    ik->qpos[2]=actual->qpos[2]+W51Motion::Smooth(tau/.35)*(ztarget-actual->qpos[2]);
    double yawq[4]={std::cos(actual->userdata[83]*tau/2),0,0,std::sin(actual->userdata[83]*tau/2)};
    mju_mulQuat(ik->qpos+3,yawq,actual->qpos+3);
    double step[4],stepvel[4];q->W51Step(t,step,stepvel);
    for(int i=0;i<4;++i) {
      while(t>end[i]+1e-10) {
        start[i]=endpos[i];velocity[i]={0,0,0};begin[i]=end[i];
        swing[i]=!swing[i];end[i]+= (swing[i]?1-motion.duty:motion.duty)/motion.frequency;
        endpos[i]=swing[i]?motion.Touchdown(model,actual,i,end[i]):start[i];
        step0[i]=dstep0[i]=0;
      }
      const double duration=end[i]-begin[i],u=std::clamp((t-begin[i])/duration,0.,1.);
      std::array<double,3> target;
      for(int j=0;j<3;++j)target[j]=endpos[i][j]+W51Motion::H0(u)*(start[i][j]-endpos[i][j])+
          duration*W51Motion::H1(u)*velocity[i][j];
      // Original absolute-phase C2 height plus a unique endpoint correction.
      // At the current node both actual foot position and velocity are exact.
      target[2]+=step[i]-W51Motion::H0(u)*step0[i]-duration*W51Motion::H1(u)*dstep0[i];
      for(int j=0;j<3;++j)targets[k*12+3*i+j]=target[j];
      double error=0;
      for(int it=0;it<40;++it) {
        mj_kinematics(model,ik);mj_comPos(model,ik);
        double e[3];mju_sub3(e,target.data(),ik->geom_xpos+3*motion.geoms[i]);
        error=mju_norm3(e);
        if(error<1e-7)break;
        mj_jacGeom(model,ik,jac.data(),nullptr,motion.geoms[i]);
        double J[9],A[9],b[3],dq[3];
        for(int a=0;a<3;++a) {
          const int joint=model->actuator_trnid[2*actuators[i][a]],dof=model->jnt_dofadr[joint];
          for(int j=0;j<3;++j)J[3*j+a]=jac[j*model->nv+dof];
        }
        mju_mulMatTMat(A,J,J,3,3,3);mju_mulMatTVec(b,J,e,3,3);
        for(int a=0;a<3;++a)A[4*a]+=1e-10;
        mju_cholFactor(A,3,1e-14);mju_cholSolve(dq,A,b,3);
        const double scale=std::min(1.,.2/std::max(1e-15,std::max({std::abs(dq[0]),std::abs(dq[1]),std::abs(dq[2])})));
        for(int a=0;a<3;++a) {
          const int joint=model->actuator_trnid[2*actuators[i][a]],adr=model->jnt_qposadr[joint];
          ik->qpos[adr]=std::clamp(ik->qpos[adr]+scale*dq[a],model->jnt_range[2*joint],model->jnt_range[2*joint+1]);
        }
      }
      mj_kinematics(model,ik);
      double e[3];mju_sub3(e,target.data(),ik->geom_xpos+3*motion.geoms[i]);error=mju_norm3(e);
      max_error=std::max(max_error,error);
      if(error>1e-5 || !std::isfinite(error)) {valid=false;failed_node=k;break;}
    }
    for(int a=0;a<dim_action;++a) {
      const int adr=model->jnt_qposadr[model->actuator_trnid[2*a]];
      ik_path[k*dim_action+a]=ik->qpos[adr];
      delta[k*dim_action+a]=ik->qpos[adr]-r33_nominal_trajectory.states[k*dim_state+adr];
    }
  }
  w51_status=valid?"native_rollout":"ik_unavailable";
  iLQGPolicy seed=r33_nominal_policy;
  if(valid) {
    seed.AddDelta(knots,delta,horizon-1,1.);
    Trajectory tr=r33_nominal_trajectory;bool projection_failed=false;
    auto f=[&](double* u,const double* x,double t) {
      bool changed,failed,restored;seed.ActionImpl(u,x,t,false,&changed,&failed,&restored);
      projection_failed|=failed;
    };
    tr.Rollout(f,task,model,data_[0].get(),state.data(),time,mocap.data(),userdata.data(),horizon);
    if(projection_failed)tr.failure=true;
    ++r33_rollouts;w51_seed_rollouts=1;
    R33Consider(seed,tr,horizon,4,0,1.,"motion_seed_S");
    w51_seed_valid=seed.metric_valid && !tr.failure && std::isfinite(tr.total_return) && std::isfinite(tr.trunk_violation);
    if(w51_seed_valid) {
      w51_Q=seed;w51_Qtr=tr;candidate_policy[0]=seed;
      candidate_policy[0].trajectory=tr;candidate_policy[0].ClearDelta();
      feedback_scaling=seed.feedback_scaling;w51_status="Q_is_S";
    } else w51_status="native_seed_invalid_Q_is_B";
    if(const char* p=std::getenv("MJPC_W51_RECORD"))seed.Save(std::string(p)+"-S.bin");
  }
  if(const char* path=std::getenv("MJPC_W51_RECORD")) {
    r33_nominal_policy.Save(std::string(path)+"-B.bin");w51_Q.Save(std::string(path)+"-Q.bin");
    std::ofstream out(std::string(path)+".json",std::ios::trunc);out.precision(17);
    out<<"{\"call_id\":"<<r29_call_id<<",\"epoch\":"<<r32.epoch<<",\"time_s\":"<<time
       <<",\"status\":\""<<w51_status<<"\",\"S_valid\":"<<(w51_seed_valid?"true":"false")
       <<",\"native_seed_rollouts\":"<<w51_seed_rollouts<<",\"B_J\":"<<r33_nominal_trajectory.total_return
       <<",\"Q_J\":"<<w51_Qtr.total_return<<",\"Q_is_S\":"<<(w51_seed_valid?"true":"false")
       <<",\"IK_max_error_m\":"<<max_error<<",\"IK_failed_node\":"<<failed_node
       <<",\"foot_order\":[\"FL\",\"RL\",\"FR\",\"RR\"],\"phase\":"<<motion.phase
       <<",\"expected\":null,\"ratio\":null,\"foot_context\":[";
    for(int i=0;i<4;++i) {
      if(i) {out<<",";}
      const auto& f=motion.feet[i];
      out<<"{\"segment\":"<<f.segment<<",\"swing\":"<<(f.swing?"true":"false")
         <<",\"observed_start\":"<<f.begin<<",\"end\":"<<f.end<<",\"anchor\":[";
      for(int j=0;j<3;++j){if(j)out<<",";out<<f.anchor[j];}
      out<<"],\"touchdown\":[";for(int j=0;j<3;++j){if(j)out<<",";out<<f.touchdown[j];}out<<"]}";
    }
    out<<"]}\n";
    std::ofstream raw(std::string(path)+"-kinematics.bin",std::ios::binary|std::ios::trunc);
    double h[]={51.,double(r29_call_id),time,double(horizon-1),double(valid)};
    raw.write(reinterpret_cast<char*>(h),sizeof(h));
    for(const auto* v:{&knots,&targets,&ik_path,&delta})raw.write(reinterpret_cast<const char*>(v->data()),v->size()*sizeof(double));
  }
  mj_deleteData(actual);mj_deleteData(ik);
}
