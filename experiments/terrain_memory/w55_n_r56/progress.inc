// One additional candidate. Keep the shifted policy's airborne-leg torques
// while allocating the supporting legs' effort to the commanded COM impulse.
// Its entire contact evolution goes through the ordinary E and 1ms checks.
extern "C" int w56_progress(const mjModel* m,mjData* d,const double* shifted,double* out){
  mju_copy(out,shifted,U);
  const double* command=d->userdata+W55Motion;
  const double norm=std::hypot(command[0],command[1]);
  // A new plan is installed every 20 ms; inject one candidate impulse only
  // in its executable prefix. The same ordinary policy covers the remaining
  // 330 ms and must still pass all original horizon checks.
  if(norm<1.e-8 || d->time-d->userdata[W55Epoch]>=.02-1.e-9)return 0;
  Foot feet[F];geometry(m,d,feet,true);Response s;
  if(!response(m,d,feet,s))return 1;
  if(!s.contacts)return 0; // no controllable rigid reaction during flight/soft-only phase
  QP support;Vec raw_tau{},baseline{};
  constraints(m,d,s,support,raw_tau,shifted);
  if(!support.project(raw_tau,baseline))return 2;
  mj_subtreeVel(m,d);
  const int root=mj_name2id(m,mjOBJ_BODY,"trunk");
  const double mass=m->body_subtreemass[root],*vcom=d->subtree_linvel+3*root;
  double dir[2]={command[0]/norm,command[1]/norm};
  double bias[3]={};Vec forces[3]{};
  for(const auto& f:feet)if(f.material&&f.bearing)for(int k=0;k<3;++k)bias[k]+=f.load[k];
  for(int leg=0;leg<s.contacts;++leg)for(int k=0;k<3;++k){
    bias[k]+=s.f0[3*leg+k];
    for(int j=0;j<U;++j)forces[k][j]+=s.L[3*leg+k][j];
  }
  // Preserve the shifted gait's airborne joint inputs. This generator makes
  // its forward impulse solely on the genuine shared supporting contacts.
  for(int i=0;i<F;++i)if(!feet[i].bearing)for(int j=3*i;j<3*i+3;++j){
    Vec axis{};axis[j]=1;support.bounds(axis,baseline[j],baseline[j]);
  }
  // During propulsion the body still has to recover its nominal height. Use
  // the achievable baseline when even that vertical demand exceeds capacity.
  const double horizon=.35, home=m->key_qpos[mj_name2id(m,mjOBJ_KEY,"home")*m->nq+2];
  const double target_vertical=mass*(9.81+2*(home-d->qpos[2])/(horizon*horizon)-2*d->qvel[2]/horizon);
  support.lower(forces[2],std::min(dot(forces[2],baseline),target_vertical-bias[2]));
  // Stage one demonstrated that a cheaper joint snapshot can tip the
  // support and lower the trunk. Keep roll/pitch acceleration equal to the
  // feasible shifted response and preserve its vertical acceleration margin.
  for(int k:{3,4}){Vec angular=row_of(s,k);double current=dot(angular,baseline);support.bounds(angular,current,current);}
  Vec up=row_of(s,2);support.lower(up,dot(up,baseline));
  Vec drive{};for(int j=0;j<U;++j)drive[j]=dir[0]*forces[0][j]+dir[1]*forces[1][j];
  double measured=dir[0]*vcom[0]+dir[1]*vcom[1];
  double demanded=mass*(norm-measured)/horizon-dir[0]*bias[0]-dir[1]*bias[1];
  double now=dot(drive,baseline),preferred=std::max(now,demanded);
  // Minimal supporting torque for a physically available fraction of the
  // requested impulse. The fixed shifted swing actions maintain gait timing.
  Vec target=baseline; // minimum additional supporting torque for the impulse
  Vec best=baseline;double low=0.,high=1.;bool solved=false;
  for(int it=0;it<9;++it){
    double frac=it==0?1.:(low+high)*.5;
    QP trial=support;trial.lower(drive,now+frac*(preferred-now));
    Vec tau{};
    if(trial.project(target,tau)){best=tau;solved=true;low=frac;if(frac==1.)break;}
    else high=frac;
  }
  if(!solved){
    // A zero impulse is allowed only when its current supporting baseline
    // actually satisfies the same constraints; this candidate remains subject
    // to the common full-horizon selector.
    for(const auto& r:support.rows)if(dot(r.a,baseline)<r.b-2.e-5)return 3;
    best=baseline;
  }
  for(int j=0;j<U;++j){
    int joint=m->actuator_trnid[2*j],k=m->jnt_dofadr[joint];
    double u=d->qpos[m->jnt_qposadr[joint]]+(best[j]+5*d->qvel[k])/60.;
    out[j]=std::clamp(u,m->actuator_ctrlrange[2*j],m->actuator_ctrlrange[2*j+1]);
  }
  return 0;
}
