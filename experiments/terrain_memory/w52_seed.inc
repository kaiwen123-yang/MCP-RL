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
  std::array<std::vector<w52::Segment>,4> segments;
  bool valid=true;int failed_node=-1;double max_error=0;
  std::string geometry_reason="available";
  std::vector<double> targets((horizon-1)*12),ik_path((horizon-1)*12,std::numeric_limits<double>::quiet_NaN());
  const double finish=r33_nominal_trajectory.times[horizon-2];
  for(int k=0;k<horizon-1;++k)knots[k]=r33_nominal_trajectory.times[k];
  for(int i=0;i<4;++i) {
    if(actuators[i].size()!=3)mju_error("W52 requires original three-joint Go2 legs");
    mj_jacGeom(model,actual,jac.data(),nullptr,motion.geoms[i]);
    std::array<double,3> start,velocity;
    for(int j=0;j<3;++j) {
      start[j]=actual->geom_xpos[3*motion.geoms[i]+j];
      velocity[j]=mju_dot(jac.data()+j*model->nv,actual->qvel,model->nv);
    }
    double begin=time,end=motion.feet[i].end;bool swing=motion.feet[i].swing;
    auto endpoint=swing?motion.feet[i].touchdown:motion.feet[i].anchor;
    w52::Surface surface;
    const double* flat=GetCustomNumericData(model,"s1_flat_ground_height");
    surface.flat=flat?flat[0]:0;surface.radius=model->geom_size[3*motion.geoms[i]];
    surface.material=GetCustomNumericData(model,"r10_local_contact");
    surface.anchored=GetCustomNumericData(model,"r10_contact_anchor");
    std::copy_n(actual->userdata+62,16,surface.patches.begin());
    while(begin<=finish+1e-12) {
      if(end-begin<=1e-12) {
        // Consume the original zero-length event, preserving actual p/v.
        swing=!swing;end+=(swing?1-motion.duty:motion.duty)/motion.frequency;
        endpoint=swing?motion.Touchdown(model,actual,i,end):endpoint;
        continue;
      }
      auto seg=w52::Make(begin,end,swing,start,velocity,endpoint,surface);
      if(!seg.valid && valid){valid=false;geometry_reason=seg.reason;}
      segments[i].push_back(seg);
      begin=end;start=endpoint;velocity={0,0,0};
      swing=!swing;end+=(swing?1-motion.duty:motion.duty)/motion.frequency;
      endpoint=swing?motion.Touchdown(model,actual,i,end):start;
    }
    for(int k=0;k<horizon-1;++k) {
      const w52::Segment* seg=&segments[i].back();
      for(const auto& s:segments[i])if(knots[k]<=s.end+1e-12){seg=&s;break;}
      auto target=w52::Position(*seg,knots[k]);
      for(int j=0;j<3;++j)targets[k*12+3*i+j]=target[j];
    }
  }
  const bool geometry_valid=valid;
  for(int k=0;k<horizon-1 && valid;++k) {
    const double t=r33_nominal_trajectory.times[k],tau=t-time;
    knots[k]=t;
    for(int j=0;j<2;++j)ik->qpos[j]=actual->qpos[j]+actual->userdata[78+j]*tau;
    const double* height=GetCustomNumericData(model,"r10_material_support_height");
    const double ztarget=height?height[0]:actual->userdata[95];
    ik->qpos[2]=actual->qpos[2]+W51Motion::Smooth(tau/.35)*(ztarget-actual->qpos[2]);
    double yawq[4]={std::cos(actual->userdata[83]*tau/2),0,0,std::sin(actual->userdata[83]*tau/2)};
    mju_mulQuat(ik->qpos+3,yawq,actual->qpos+3);
    for(int i=0;i<4;++i) {
      std::array<double,3> target;
      for(int j=0;j<3;++j)target[j]=targets[k*12+3*i+j];
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
  w51_status=valid?"native_rollout":(geometry_valid?"ik_unavailable":geometry_reason);
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
    out<<"],\"geometry_valid\":"<<(geometry_valid?"true":"false")
       <<",\"geometry_reason\":\""<<geometry_reason<<"\",\"seed_initial_acceleration\":0"
       <<",\"geometry_tolerance_m\":1e-5,\"continuous_extrema\":true,\"segments\":[";
    bool comma=false;
    for(int i=0;i<4;++i)for(const auto& s:segments[i]) {
      if(comma){out<<",";}
      comma=true;
      out<<"{\"foot\":"<<i<<",\"begin\":"<<s.begin<<",\"end\":"<<s.end<<",\"apex\":"<<s.apex
         <<",\"swing\":"<<(s.swing?"true":"false")<<",\"valid\":"<<(s.valid?"true":"false")
         <<",\"reason\":\""<<s.reason<<"\",\"min_clearance_m\":"<<s.min_clearance
         <<",\"max_clearance_m\":"<<s.max_clearance<<",\"min_world_z_m\":"<<s.min_world_z
         <<",\"max_world_z_m\":"<<s.max_world_z<<",\"max_descent_velocity_m_s\":"<<s.max_descent_velocity;
      for(const auto& pair:{std::pair{"start",&s.start},std::pair{"velocity",&s.velocity},std::pair{"touchdown",&s.touchdown}}) {
        out<<",\""<<pair.first<<"\":[";
        for(int j=0;j<3;++j){if(j)out<<",";out<<(*pair.second)[j];}out<<"]";
      }
      out<<"}";
    }
    out<<"]}\n";
    std::ofstream raw(std::string(path)+"-kinematics.bin",std::ios::binary|std::ios::trunc);
    double h[]={51.,double(r29_call_id),time,double(horizon-1),double(valid)};
    raw.write(reinterpret_cast<char*>(h),sizeof(h));
    for(const auto* v:{&knots,&targets,&ik_path,&delta})raw.write(reinterpret_cast<const char*>(v->data()),v->size()*sizeof(double));
  }
  mj_deleteData(actual);mj_deleteData(ik);
}
