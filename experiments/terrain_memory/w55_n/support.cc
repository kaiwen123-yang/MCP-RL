#include "support.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {
constexpr int V=18,U=12,F=4;
constexpr double hold=.002, dt=.0002, friction=.5, sole=.001, pitch=.02;
constexpr double tangent=1227272.7272727273;
constexpr double cone=friction/1.4142135623730951;
constexpr char const* names[F]={"FL","FR","RL","RR"};
using Vec=std::array<double,U>;
using Dof=std::array<double,V>;
using XYZ=std::array<double,3>;
double dot(const Vec& a,const Vec& b){return mju_dot(a.data(),b.data(),U);}
bool soft(const double* p){
  const double radius=std::sqrt(sole)/2;
  return p[0]>=.8+radius && p[0]<=2.2-radius && std::abs(p[1])<=.6-radius;
}
// The causal tangent is frozen about each observed world cell. The interval
// integral is the loaded overlap area, including partial cells and release.
double normal_force(const mjData* d,const double* p,double gap,double& stiffness){
  stiffness=0.; if(!soft(p))return 0.;
  double f=0.,r=std::sqrt(sole)/2;
  for(int ix=std::max(40,int(std::floor((p[0]-r)/pitch)));ix<std::min(110,1+int(std::floor((p[0]+r)/pitch)));++ix){
    double dx=std::max(0.,std::min(p[0]+r,(ix+1)*pitch)-std::max(p[0]-r,ix*pitch));
    for(int iy=std::max(-30,int(std::floor((p[1]-r)/pitch)));iy<std::min(30,1+int(std::floor((p[1]+r)/pitch)));++iy){
      double dy=std::max(0.,std::min(p[1]+r,(iy+1)*pitch)-std::max(p[1]-r,iy*pitch));
      double penetration=d->userdata[W55Surface+(ix-40)*W55Ny+iy+30]-gap;
      if(penetration>0.){double k=tangent*dx*dy/sole;f+=k*penetration;stiffness+=k;}
    }
  }
  return f;
}
double floor_height(const mjModel* m,const mjData* d){return d->geom_xpos[3*mj_name2id(m,mjOBJ_GEOM,"floor")+2];}
struct Foot{
  int geom=0; XYZ pos{},vel{},drift{},load{};
  Dof J[3]{}; double gap=0.,stiffness=0.;
  bool material=false,bearing=false;
};
void geometry(const mjModel* m,mjData* d,Foot (&feet)[F],bool dynamics){
  mj_kinematics(m,d);mj_comPos(m,d);
  const double floor=floor_height(m,d);
  for(int i=0;i<F;++i){
    auto& f=feet[i]; f.geom=mj_name2id(m,mjOBJ_GEOM,names[i]);
    const double* p=d->geom_xpos+3*f.geom;double rad=m->geom_size[3*f.geom];
    mju_copy3(f.pos.data(),p);f.pos[2]-=rad;f.gap=f.pos[2]-floor;f.material=soft(p);
    // World-fixed downward radius: v_roll = v_c + omega x (-rn).
    double jp[3*V],jr[3*V];mj_jacGeom(m,d,jp,jr,f.geom);
    for(int j=0;j<V;++j){f.J[0][j]=jp[j]-rad*jr[V+j];f.J[1][j]=jp[V+j]+rad*jr[j];f.J[2][j]=jp[2*V+j];}
    for(int a=0;a<3;++a)f.vel[a]=mju_dot(f.J[a].data(),d->qvel,V);
    const double* obs=d->userdata+W55FootObservation+4*i;
    if(f.material){
      f.load[2]=normal_force(d,p,f.gap,f.stiffness);
      double scale=-friction*f.load[2]/std::sqrt(f.vel[0]*f.vel[0]+f.vel[1]*f.vel[1]+.0001);
      f.load[0]=scale*f.vel[0];f.load[1]=scale*f.vel[1];
      f.bearing=f.load[2]>0.;
    }else{
      // At the exact synchronized state take measured contact. For all
      // subsequent predicted/executed states, use their own contact geometry.
      f.bearing=(std::abs(d->time-d->userdata[W55Epoch])<1.e-10 && obs[0]>0)
          || (f.gap<=0. && f.vel[2]<=std::max(0.,-f.gap/hold));
      f.load[2]=obs[3];
    }
  }
  if(!dynamics)return;
  // Directional derivative of the complete rolling Jacobian at moving
  // configurations. This is independent of the old MuJoCo mj_jacDot.
  Dof plus[F][3];double q[19];mju_copy(q,d->qpos,m->nq);
  const double eps=1.e-5;
  for(int side=0;side<2;++side){
    mju_copy(d->qpos,q,m->nq);mj_integratePos(m,d->qpos,d->qvel,side==0?eps:-eps);
    mj_kinematics(m,d);mj_comPos(m,d);
    for(int i=0;i<F;++i){
      double jp[3*V],jr[3*V];const double rad=m->geom_size[3*feet[i].geom];
      mj_jacGeom(m,d,jp,jr,feet[i].geom);
      for(int j=0;j<V;++j){jp[j]-=rad*jr[V+j];jp[V+j]+=rad*jr[j];}
      for(int a=0;a<3;++a){
        if(!side)mju_copy(plus[i][a].data(),jp+a*V,V);
        else feet[i].drift[a]=(mju_dot(plus[i][a].data(),d->qvel,V)-mju_dot(jp+a*V,d->qvel,V))/(2*eps);
      }
    }
  }
  mju_copy(d->qpos,q,m->nq);mj_kinematics(m,d);mj_comPos(m,d);
  mj_crb(m,d);mj_factorM(m,d);mj_comVel(m,d);mj_passive(m,d);mj_rne(m,d,0,d->qfrc_bias);
}
struct Row{Vec a{};double b=0.;};
struct Response{
  Dof a0{}, A[U]{}; double f0[12]{},L[12][U]{};int legs[4]{},contacts=0;
};
// M_eff a = S tau_start + passive - h + J'lambda - S kp*v*mean(t).
// Impact/held-input approximations apply only to this projection. The original
// ten physical substeps and force saturation remain in the actual plant.
bool response(const mjModel* m,mjData* d,const Foot (&feet)[F],Response& s){
  double mass[V*V];mj_fullM(m,mass,d->qM);
  double mean_t=4.5*dt,mean_t2=28.5*dt*dt;
  Dof b{}; for(int k=0;k<V;++k)b[k]=d->qfrc_passive[k]-d->qfrc_bias[k]-m->dof_frictionloss[k]*std::tanh(d->qvel[k]/.01);
  for(int j=0;j<U;++j){int k=m->jnt_dofadr[m->actuator_trnid[2*j]];mass[k*V+k]+=5*mean_t+30*mean_t2;b[k]-=60*d->qvel[k]*mean_t;}
  Dof J[12]{},W[12]{};double target[12]{};
  for(int i=0;i<F;++i)if(feet[i].bearing){
    const auto& f=feet[i];
    if(f.material){
      for(int k=0;k<V;++k)for(int a=0;a<3;++a)b[k]+=f.J[a][k]*f.load[a];
    }else{
      int n=s.contacts++;s.legs[n]=i;
      for(int a=0;a<3;++a){J[3*n+a]=f.J[a];target[3*n+a]=-f.drift[a];}
    }
  }
  if(mju_cholFactor(mass,V,0)!=V)return false;
  mju_cholSolve(s.a0.data(),mass,b.data(),V);
  for(int j=0;j<U;++j){Dof selector{};selector[m->jnt_dofadr[m->actuator_trnid[2*j]]]=m->actuator_gear[6*j];mju_cholSolve(s.A[j].data(),mass,selector.data(),V);}
  int n=3*s.contacts;if(!n)return true;
  double schur[144]={},rhs[12]={},x[12]={};
  for(int k=0;k<n;++k)mju_cholSolve(W[k].data(),mass,J[k].data(),V);
  for(int k=0;k<n;++k)for(int l=0;l<n;++l)schur[n*k+l]=mju_dot(J[k].data(),W[l].data(),V);
  if(mju_cholFactor(schur,n,0)!=n)return false;
  for(int k=0;k<n;++k)rhs[k]=target[k]-mju_dot(J[k].data(),s.a0.data(),V);
  mju_cholSolve(s.f0,schur,rhs,n);
  for(int k=0;k<n;++k)mju_addToScl(s.a0.data(),W[k].data(),s.f0[k],V);
  for(int j=0;j<U;++j){
    for(int k=0;k<n;++k)rhs[k]=-mju_dot(J[k].data(),s.A[j].data(),V);
    mju_cholSolve(x,schur,rhs,n);
    for(int k=0;k<n;++k){s.L[k][j]=x[k];mju_addToScl(s.A[j].data(),W[k].data(),x[k],V);}
  }
  return true;
}
struct QP{
  std::vector<Row> rows;
  void lower(Vec a,double b){double n=std::sqrt(dot(a,a));if(n>1e-13){for(double& v:a)v/=n;b/=n;}rows.push_back({a,b});}
  void bounds(Vec a,double lo,double hi){lower(a,lo);for(double& x:a)x=-x;lower(a,-hi);}
  // Dual coordinate minimization of ||tau - target||^2 on normalized
  // halfspaces; a result is usable only after testing each primal constraint.
  bool project(const Vec& target,Vec& x)const{
    x=target;std::vector<double> dual(rows.size());
    for(int sweep=0;sweep<120;++sweep){
      double step=0;
      for(size_t i=0;i<rows.size();++i){const auto& row=rows[i];double next=std::max(0.,dual[i]+row.b-dot(row.a,x));double delta=next-dual[i];
        if(delta!=0.){for(int j=0;j<U;++j)x[j]+=delta*row.a[j];dual[i]=next;step=std::max(step,std::abs(delta));}}
      if(step<1.e-8)break;
    }
    for(const auto& row:rows)if(dot(row.a,x)<row.b-2.e-5)return false;
    return true;
  }
};
Vec row_of(const Response& s,int k){Vec a{};for(int j=0;j<U;++j)a[j]=s.A[j][k];return a;}
Vec force_row(const Response& s,int k){Vec a{};std::copy_n(s.L[k],U,a.begin());return a;}
void constraints(const mjModel* m,mjData* d,const Response& s,QP& qp,Vec& raw_tau,const double* raw){
  for(int j=0;j<U;++j){
    int jid=m->actuator_trnid[2*j],k=m->jnt_dofadr[jid];double q=d->qpos[m->jnt_qposadr[jid]],v=d->qvel[k];Vec u{};u[j]=1.;
    const double* tr=m->actuator_forcerange+2*j;const double* ur=m->actuator_ctrlrange+2*j;
    raw_tau[j]=std::clamp(60*(raw[j]-q)-5*v,tr[0],tr[1]);qp.bounds(u,std::max(tr[0],60*(ur[0]-q)-5*v),std::min(tr[1],60*(ur[1]-q)-5*v));
    Vec ar=row_of(s,k);
    for(int n=1;n<10;++n){double t=n*dt,c=5*t+30*t*t;Vec h=u;for(int z=0;z<U;++z)h[z]-=c*ar[z];const double off=-60*v*t-c*s.a0[k];
      qp.bounds(h,tr[0]-off,tr[1]-off);}
    double off=q+v*hold+.5*hold*hold*s.a0[k];for(double& z:ar)z*=.5*hold*hold;
    qp.bounds(ar,m->jnt_range[2*jid]-off,m->jnt_range[2*jid+1]-off);
  }
  for(int n=0;n<s.contacts;++n){Vec fz=force_row(s,3*n+2);double z=s.f0[3*n+2];qp.lower(fz,-z);
    for(int axis=0;axis<2;++axis)for(double sign:{-1.,1.}){Vec row=fz;for(int j=0;j<U;++j)row[j]=cone*row[j]+sign*s.L[3*n+axis][j];qp.lower(row,-cone*z-sign*s.f0[3*n+axis]);}}
}
double ease(double x){x=std::clamp(x,0.,1.);return x*x*x*(10.+x*(-15.+6.*x));}
} // namespace
extern "C" int w55_project(const mjModel* m,mjData* d,const double* raw,double* out,double* diagnostic){
  double temp[F*W55Diag]{};double* log=diagnostic?diagnostic:temp;std::fill(log,log+F*W55Diag,0.);
  Foot feet[F];geometry(m,d,feet,false);bool active=false;
  mju_copy(out,raw,U);
  for(int i=0;i<F;++i){
    const auto& f=feet[i];double* e=log+i*W55Diag;e[0]=f.gap;e[1]=f.vel[2];e[7]=f.material;e[10]=f.bearing;e[11]=f.material?f.load[2]:d->userdata[W55FootObservation+4*i+3];
    double speed=std::max(0.,-f.vel[2]),lead=.022*speed+std::max(0.,speed*speed-.05*.05)/(2*9.81);
    // 1g sets the conservative onset distance only. Actual braking comes
    // from the single feasible floating-base solve below, never footwise boxes.
    e[3]=lead;e[4]=f.material&&!f.bearing&&speed>.05&&lead>f.gap?ease((lead-std::max(0.,f.gap))/std::max(lead,1e-12))*ease((speed-.05)/.05):0.;
    active|=e[4]>0.;e[19]=d->userdata[W55Epoch];
  }
  if(!active)return 0;
  geometry(m,d,feet,true);Response s;QP qp;Vec raw_tau{},base{},braking{};
  auto fail=[&](int why){for(int i=0;i<F;++i)log[i*W55Diag+23]=why;return why;};
  if(!response(m,d,feet,s))return fail(1);
  constraints(m,d,s,qp,raw_tau,raw);
  if(!qp.project(raw_tau,base))return fail(2);
  // Support priority: braking cannot reduce any baseline normal reaction or
  // downward body acceleration margin relative to the nearest admissible raw.
  Vec body=row_of(s,2);qp.lower(body,dot(body,base));
  for(int k=0;k<s.contacts;++k){Vec z=force_row(s,3*k+2);qp.lower(z,dot(z,base));}
  Vec rows[F]{};double req[F]{},orig[F]{},lip=1.;
  for(int i=0;i<F;++i){const auto& f=feet[i];double* e=log+i*W55Diag;
    for(int j=0;j<U;++j)rows[i][j]=mju_dot(f.J[2].data(),s.A[j].data(),V);
    orig[i]=f.drift[2]+mju_dot(f.J[2].data(),s.a0.data(),V);
    double a0=orig[i]+dot(rows[i],base),speed=std::max(0.,-f.vel[2]);
    double desired=std::max(0.,speed*speed-.05*.05)/(2*std::max(f.gap-speed*hold,.05*hold));
    req[i]=a0+e[4]*std::max(0.,desired-a0);e[5]=req[i];e[8]=a0;e[9]=req[i]-a0;
    lip+=e[4]*dot(rows[i],rows[i]);
  }
  braking=base;
  // Minimize distance to the original admissible torque and simultaneous
  // swing acceleration error in the same support-preserving polytope.
  for(int it=0;it<12;++it){
    Vec step=braking;
    for(int j=0;j<U;++j){double g=braking[j]-base[j];
      for(int i=0;i<F;++i)g+=log[i*W55Diag+4]*rows[i][j]*(orig[i]+dot(rows[i],braking)-req[i]);
      step[j]-=g/lip;}
    Vec candidate{};
    if(!qp.project(step,candidate))break; // the previous iterate is feasible
    double change=0.;for(int j=0;j<U;++j)change=std::max(change,std::abs(candidate[j]-braking[j]));
    braking=candidate;if(change<1.e-6)break;
  }
  for(int j=0;j<U;++j){int joint=m->actuator_trnid[2*j];out[j]=d->qpos[m->jnt_qposadr[joint]]+(braking[j]+5*d->qvel[m->jnt_dofadr[joint]])/60;}
  for(int i=0;i<F;++i){double* e=log+i*W55Diag;e[6]=orig[i]+dot(rows[i],braking);e[2]=e[6];e[12]=e[1]+hold*e[6];
    e[13]=std::max(0.,req[i]-e[6]);e[14]=s.a0[2]+dot(body,base);e[15]=s.a0[2]+dot(body,braking);
    e[16]=feet[i].material?feet[i].load[2]:0.;e[17]=e[16];
    for(int k=0;k<s.contacts;++k)if(s.legs[k]==i){auto z=force_row(s,3*k+2);e[16]=s.f0[3*k+2]+dot(z,base);e[17]=s.f0[3*k+2]+dot(z,braking);}
    for(int j=3*i;j<3*i+3;++j)e[18]=std::max(e[18],std::abs(braking[j]-raw_tau[j]));
    e[20]=s.contacts;e[21]=std::max(0.,e[5]-e[6]);e[22]=feet[i].stiffness;
  }
  return 0;
}

extern "C" void w55_recovery(const mjModel* m,mjData* d,double* raw){
  Foot feet[F];geometry(m,d,feet,false);
  const int home=mj_name2id(m,mjOBJ_KEY,"home");
  const double* posture=m->key_qpos+home*m->nq;
  double bz=d->qpos[2],vz=d->qvel[2],mass=m->body_subtreemass[mj_name2id(m,mjOBJ_BODY,"trunk")];
  // Allocate a support wrench about the actual base. Pitch/roll acceleration
  // is represented by desired center of pressure. In flight reach the same
  // nominal leg posture; the original mission continues to score the candidate.
  double roll_pitch[3]{};double ref[4]={1,0,0,0};mju_subQuat(roll_pitch,ref,d->qpos+3);
  double total=std::max(0.,mass*(9.81+2*(posture[2]-bz)/(.35*.35)-2*vz/.35));
  double weight[F]{},sum=0.;
  for(int i=0;i<F;++i)if(feet[i].bearing){double lateral=feet[i].pos[1]-d->qpos[1],fore=feet[i].pos[0]-d->qpos[0];
    weight[i]=std::max(0.,1.+(roll_pitch[0]-.35*d->qvel[3])*lateral/.10-(roll_pitch[1]-.35*d->qvel[4])*fore/.20);sum+=weight[i];}
  double tau[U]{};for(int j=0;j<U;++j){int jid=m->actuator_trnid[2*j],q=m->jnt_qposadr[jid],v=m->jnt_dofadr[jid];tau[j]=20*(posture[q]-d->qpos[q])-2*d->qvel[v];}
  for(int i=0;i<F;++i)if(sum>0. && weight[i]>0.)for(int j=0;j<U;++j){
    int k=m->jnt_dofadr[m->actuator_trnid[2*j]];tau[j]-=feet[i].J[2][k]*(total*weight[i]/sum);
  }
  for(int j=0;j<U;++j){int jid=m->actuator_trnid[2*j],q=m->jnt_qposadr[jid],v=m->jnt_dofadr[jid];
    double t=std::clamp(tau[j],m->actuator_forcerange[2*j],m->actuator_forcerange[2*j+1]);
    raw[j]=std::clamp(d->qpos[q]+(t+5*d->qvel[v])/60,m->actuator_ctrlrange[2*j],m->actuator_ctrlrange[2*j+1]);}
}
namespace {
int native_filter(const mjModel* m,mjData* d,int g1,int g2){
  if(!(m->geom_contype[g1]&m->geom_conaffinity[g2]) && !(m->geom_contype[g2]&m->geom_conaffinity[g1]))return 1;
  if(m->nuserdata!=W55Data)return 0;
  int g=m->geom_bodyid[g1]==0?g2:m->geom_bodyid[g2]==0?g1:-1;
  if(g<0 || !soft(d->geom_xpos+3*g))return 0;
  for(auto name:names)if(g==mj_name2id(m,mjOBJ_GEOM,name))return 1;
  return 0;
}
}
extern "C" void w55_native_contact_filter(){mjcb_contactfilter=native_filter;}
extern "C" void w55_native_step(const mjModel* m,mjData* d,bool integrate){
  // Only native model. A tangent contact force is applied once; the filter
  // removes precisely the material-interior foot/floor pair for this mjData.
  mj_kinematics(m,d);
  bool on_material=false;for(auto name:names)on_material|=soft(d->geom_xpos+3*mj_name2id(m,mjOBJ_GEOM,name));
  mju_zero(d->qfrc_applied,V);
  if(on_material){
    Foot feet[F];geometry(m,d,feet,false);mj_crb(m,d);mj_factorM(m,d);
    // Backward-Euler linearized, coupled normal compliance at the 1 ms native
    // step. Geometry and frozen world-cell rest heights are local to this data.
    Dof inverse[F]{};double force[F]{};
    for(int i=0;i<F;++i)if(feet[i].material){
      // An entering contact may first be reached partway through this step.
      double gap=feet[i].gap+m->opt.timestep*feet[i].vel[2];double k;
      force[i]=normal_force(d,feet[i].pos.data(),gap,k);feet[i].stiffness=k;
      if(k>0.)mj_solveM(m,d,inverse[i].data(),feet[i].J[2].data(),1);
    }
    double applied[F]{};
    for(int it=0;it<16;++it){double change=0.;
      for(int i=0;i<F;++i)if(feet[i].stiffness>0.){double h=feet[i].stiffness*m->opt.timestep*m->opt.timestep;
        double rhs=force[i];for(int j=0;j<F;++j)if(i!=j)rhs-=h*mju_dot(feet[i].J[2].data(),inverse[j].data(),V)*applied[j];
        double out=std::max(0.,rhs/(1.+h*mju_dot(feet[i].J[2].data(),inverse[i].data(),V)));change=std::max(change,std::abs(out-applied[i]));applied[i]=out;}
      if(change<1.e-8)break;
    }
    for(int i=0;i<F;++i)if(applied[i]>0.){const auto& f=feet[i];double scale=-friction*applied[i]/std::sqrt(f.vel[0]*f.vel[0]+f.vel[1]*f.vel[1]+.0001);
      for(int k=0;k<V;++k)d->qfrc_applied[k]+=f.J[2][k]*applied[i]+scale*(f.J[0][k]*f.vel[0]+f.J[1][k]*f.vel[1]);}
  }
  if(integrate)mj_step(m,d);else mj_forward(m,d);
}
extern "C" int w55_checks(const mjModel* m,mjData* d,double* min_gap,double* margin){
  mj_kinematics(m,d);mj_comPos(m,d);
  int floor=mj_name2id(m,mjOBJ_GEOM,"floor"),base=mj_name2id(m,mjOBJ_BODY,"trunk");
  double floor_z=d->geom_xpos[3*floor+2],clearance=1.e9,closing=0.;
  if(d->qpos[2]<.12 || d->xmat[9*base+8]<=0.)return 1;
  const double* rot=d->xmat+9*base;
  if(std::abs(std::asin(std::clamp(-rot[6],-1.,1.)))>1.2 || std::abs(std::atan2(rot[7],rot[8]))>1.2)return 1;
  for(int g=m->body_geomadr[base];g<m->body_geomadr[base]+m->body_geomnum[base];++g)if(m->geom_contype[g]||m->geom_conaffinity[g]){
    double seg[6];double dist=mj_geomDistance(m,d,floor,g,1.,seg);
    if(dist<=0.)return 2;
    if(dist<clearance){clearance=dist;double jp[3*V];mj_jac(m,d,jp,nullptr,seg+3,base);closing=std::max(0.,-mju_dot(jp+2*V,d->qvel,V));}
  }
  for(int j=0;j<U;++j){int jid=m->actuator_trnid[2*j];double q=d->qpos[m->jnt_qposadr[jid]];
    if(q<m->jnt_range[2*jid] || q>m->jnt_range[2*jid+1])return 3;
    if(d->ctrl[j]<m->actuator_ctrlrange[2*j]-1.e-10||d->ctrl[j]>m->actuator_ctrlrange[2*j+1]+1.e-10)return 4;
    if(d->actuator_force[j]<m->actuator_forcerange[2*j]-1.e-9||d->actuator_force[j]>m->actuator_forcerange[2*j+1]+1.e-9)return 4;
  }
  for(auto name:names){int g=mj_name2id(m,mjOBJ_GEOM,name);const double* p=d->geom_xpos+3*g;
    if(soft(p)){double gap=p[2]-m->geom_size[3*g]-floor_z;*min_gap=std::min(*min_gap,gap);if(gap<-.001)return 5;}}
  if(closing>0. && clearance<closing*.35){
    Foot feet[F];geometry(m,d,feet,true);bool loaded=false;for(const auto& f:feet)loaded|=f.bearing;
    if(!loaded){
      double touchdown=1.e9;for(const auto& f:feet){double z=std::max(f.gap,0.);
        touchdown=std::min(touchdown,(f.vel[2]+std::sqrt(f.vel[2]*f.vel[2]+2*9.81*z))/9.81);}
      double free=clearance-closing*touchdown-.5*9.81*touchdown*touchdown;*margin=std::min(*margin,free);if(free<=0)return 6;
    }else{
      Response s;if(!response(m,d,feet,s))return 7;QP qp;Vec raw{},origin{},x{};constraints(m,d,s,qp,raw,d->ctrl);
      Vec normal=row_of(s,2);double required=closing*closing/(2*clearance);qp.lower(normal,required-s.a0[2]);
      if(!qp.project(raw,x))return 7;double available=s.a0[2]+dot(normal,x);
      *margin=std::min(*margin,clearance-closing*closing/(2*std::max(available,1.e-10)));
    }
  }
  return 0;
}
