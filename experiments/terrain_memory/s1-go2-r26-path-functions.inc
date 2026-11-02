
inline double R26DomainValue(const mjModel* m,const mjData* d,int geom) {
  if (!R10MaterialFoot(m,d,geom)) return 0;
  const double* margin=GetCustomNumericData(m,"r11_domain_margin");
  const double floor=GetCustomNumericData(m,"s1_flat_ground_height")[0];
  const double over=floor+m->geom_size[3*geom]-d->geom_xpos[3*geom+2]-margin[0];
  return .5*(over+mju_sqrt(over*over+1e-12))/margin[1];
}
inline double R26ClearanceValue(const mjModel* m,const mjData* d,int geom) {
  const int floor=mj_name2id(m,mjOBJ_GEOM,"floor");
  const double margin=mju_max(m->geom_margin[floor],m->geom_margin[geom]);
  const double over=margin-mj_geomDistance(m,d,floor,geom,1.,nullptr);
  return .5*(over+mju_sqrt(over*over+1e-12))/
      GetCustomNumericData(m,"r12_trunk_clearance")[0];
}
inline double R26ImpactValue(const mjModel* m,const mjData* d,int geom) {
  const double* p=d->geom_xpos+3*geom;
  constexpr double half=.015811388300841896;
  const double band=GetNumberOrDefault(0.,m,"r11_landing_approach_band");
  if (p[0]<.8+half-band || p[0]>2.2-half+band || mju_abs(p[1])>.6-half+band) return 0;
  const double* stable=GetCustomNumericData(m,"r12_stable_support");
  const double z=p[2]-m->geom_size[3*geom]-GetCustomNumericData(m,"s1_flat_ground_height")[0];
  double velocity[6];mj_objectVelocity(m,d,mjOBJ_GEOM,geom,velocity,0);
  const double down=mju_max(-velocity[5],0.);
  const double hold=m->opt.timestep*GetNumberOrDefault(1,m,"s1_control_hold_substeps");
  const double over=hold*down+down*down/(2*mju_abs(m->opt.gravity[2]))-(z+stable[1]);
  return .5*(over+mju_sqrt(over*over+1e-12))/stable[2];
}
inline void R26PathValues(const mjModel* m,const mjData* d,double* values) {
  int j=0;
  const char* feet[]={"FL","RL","FR","RR"};
  for(const char* foot:feet)values[j++]=R26DomainValue(m,d,mj_name2id(m,mjOBJ_GEOM,foot));
  const int trunk=mj_name2id(m,mjOBJ_BODY,"trunk");
  const int floor=mj_name2id(m,mjOBJ_GEOM,"floor");
  for(int geom=0;geom<m->ngeom;++geom) {
    if(m->geom_bodyid[geom]==trunk &&
       ((m->geom_contype[geom]&m->geom_conaffinity[floor]) ||
        (m->geom_contype[floor]&m->geom_conaffinity[geom])))
      values[j++]=R26ClearanceValue(m,d,geom);
  }
  for(const char* foot:feet)values[j++]=R26ImpactValue(m,d,mj_name2id(m,mjOBJ_GEOM,foot));
}
// Risk rows depend only on q,v. Differentiate these geometric functions without
// recomputing contact dynamics or the other sensor residuals at each substep.
inline void R26PathJacobian(const mjModel* m,mjData* d,double eps,int centered,
                            int nr,int nx,double* jac) {
  std::vector<double> q(d->qpos,d->qpos+m->nq),v(d->qvel,d->qvel+m->nv);
  std::vector<double> delta(m->nv,0.),base(nr),plus(nr),minus(nr);
  mj_kinematics(m,d);mj_comPos(m,d);mj_comVel(m,d);
  R26PathValues(m,d,base.data());mju_zero(jac,nr*nx);
  for(int k=0;k<2*m->nv;++k) {
    const int index=k%m->nv;
    for(int sign=1;sign>=-1;sign-=2) {
      if(sign<0 && !centered)break;
      mju_copy(d->qpos,q.data(),m->nq);mju_copy(d->qvel,v.data(),m->nv);
      if(k<m->nv) {
        delta[index]=sign*eps;mj_integratePos(m,d->qpos,delta.data(),1.);
        delta[index]=0.;
      } else d->qvel[index]+=sign*eps;
      mj_kinematics(m,d);mj_comPos(m,d);mj_comVel(m,d);
      R26PathValues(m,d,sign>0 ? plus.data() : minus.data());
    }
    for(int row=0;row<nr;++row)
      jac[row*nx+k]=(plus[row]-(centered ? minus[row] : base[row]))/
                       ((centered ? 2 : 1)*eps);
  }
  mju_copy(d->qpos,q.data(),m->nq);mju_copy(d->qvel,v.data(),m->nv);
  mj_kinematics(m,d);mj_comPos(m,d);mj_comVel(m,d);
}
