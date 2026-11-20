// The task supplies its original trot phase. Form a raw, representable PD
// position command on this candidate's mjData. Ordinary E, native contact,
// both 1ms steps and original J evaluate the proposed support exchange.
extern "C" int w59_handoff(const mjModel* m,mjData* d,const double* shifted,
                           double phase,double cadence,double duty,double height,double* out){
  mju_copy(out,shifted,U);
  const double* command=d->userdata+W55Motion;
  if(std::hypot(command[0],command[1])<1.e-8 || height<=0. || cadence<=0.)return 0;
  Foot feet[F];geometry(m,d,feet,true);
  Response response_at_state;
  if(!response(m,d,feet,response_at_state))return 1;
  const int trunk=mj_name2id(m,mjOBJ_BODY,"trunk"),home=mj_name2id(m,mjOBJ_KEY,"home");
  const double mass=m->body_subtreemass[trunk],stance=duty/cadence;
  const double* rotation=d->xmat+9*trunk;
  mj_subtreeVel(m,d);const double* com_v=d->subtree_linvel+3*trunk;
  Vec raw_tau{},proposal{},tau{}; QP qp;
  constraints(m,d,response_at_state,qp,raw_tau,shifted);
  const double swing_fraction=1.-duty;
  double share[F]{},sum=0.,angles[F]{};
  for(int i=0;i<F;++i){
    // Trot uses FL/RR together, FR/RL half a cycle later.
    angles[i]=std::remainder(phase-((i==1||i==2)?mjPI:0.),2*mjPI);
    double swing=std::abs(angles[i])/(mjPI*swing_fraction);
    // A complementary smooth handoff crosses over during double support.
    share[i]=feet[i].bearing?ease(std::min(1.,swing)):0.;
    sum+=share[i];
  }
  double horizon=1./(2*cadence),total_z=mass*(9.81+2*(m->key_qpos[home*m->nq+2]-d->qpos[2])/(horizon*horizon)-2*d->qvel[2]/horizon);
  for(int i=0;i<F;++i){
    const Foot& f=feet[i];int hip=mj_name2id(m,mjOBJ_BODY,(std::string(names[i])+"_hip").c_str());
    int first=m->jnt_dofadr[m->actuator_trnid[6*i]];
    double angle=angles[i],ratio=std::clamp(angle/(mjPI*swing_fraction),-1.,1.);
    bool swing=std::abs(angle)<mjPI*swing_fraction;
    double lift=swing?height*std::cos(.5*mjPI*ratio):0.;
    double vz=swing?-height*mjPI*cadence/swing_fraction*std::sin(.5*mjPI*ratio):0.;
    double sweep=swing ? stance*.5*std::sin(.5*mjPI*ratio) :
                  (std::copysign(mjPI,angle)-angle)/(2*mjPI*cadence);
    // Position relative to the actual hip with the original abduction offset.
    // During stance the horizontal foot reference sweeps backward at command v.
    double offset=m->body_pos[3*mj_name2id(m,mjOBJ_BODY,(std::string(names[i])+"_thigh").c_str())+1];
    double fore[2]={d->xpos[3*hip]+rotation[1]*offset+sweep*command[0],
                    d->xpos[3*hip+1]+rotation[4]*offset+sweep*command[1]};
    double error[3]={fore[0]-f.pos[0],fore[1]-f.pos[1],floor_height(m,d)+lift-f.pos[2]};
    double task_velocity[3]={swing?command[0]*(1.+stance*.5*mjPI*cadence/swing_fraction*std::cos(.5*mjPI*ratio)):0.,
                             swing?command[1]*(1.+stance*.5*mjPI*cadence/swing_fraction*std::cos(.5*mjPI*ratio)):0.,vz};
    // 3x3 local inverse kinematics of the spherical foot center, whose
    // vertical boundary is the quantity used by the original gait residual.
    double jac[3*V];mj_jacGeom(m,d,jac,nullptr,f.geom);
    double J[9],G[9]={},rhs[3],dual[3],delta[3];
    for(int a=0;a<3;++a)for(int k=0;k<3;++k)J[3*a+k]=jac[a*V+first+k];
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)G[3*a+b]=mju_dot(J+3*a,J+3*b,3);
    if(mju_cholFactor(G,3,0)!=3)return 1;
    for(int a=0;a<3;++a)rhs[a]=60*error[a]+5*(task_velocity[a]-mju_dot(jac+a*V,d->qvel,V));
    mju_cholSolve(dual,G,rhs,3);mju_mulMatTVec(delta,J,dual,3,3);
    double weight=sum>0.?share[i]/sum:0.;
    double reaction[3]={weight*mass*(command[0]-com_v[0])/horizon,
                        weight*mass*(command[1]-com_v[1])/horizon,weight*std::max(0.,total_z)};
    for(int j=0;j<3;++j){
      int k=first+j;proposal[3*i+j]=delta[j]+d->qfrc_bias[k]-d->qfrc_passive[k];
      for(int a=0;a<3;++a)proposal[3*i+j]-=f.J[a][k]*reaction[a];
    }
  }
  if(!qp.project(proposal,tau))return 2;
  return map_and_check(m,d,qp,tau,out)?0:3;
}
