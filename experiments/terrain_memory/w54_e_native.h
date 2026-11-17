#ifndef W54_E_NATIVE_H_
#define W54_E_NATIVE_H_
#include <vector>
#include <stdexcept>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"
#include "mjpc/w54_e_core.h"
namespace mjpc {
inline bool W54Enabled(const mjModel* m){return GetNumberOrDefault(0,m,"w54_precontact_E")==1;}
inline void W54ApplyE(const mjModel* m,mjData* d,double* saved=nullptr) {
  if(!W54Enabled(m))return;
  mj_forward(m,d);
  const int nv=m->nv,nu=m->nu;
  double q[12],v[12],raw[12],H[48],gap[4],vz[4],material[4],diag[32],out[12];
  std::vector<double> J(3*nv),Jr(3*nv),sol(nv);
  for(int j=0;j<nu;j++){
    const int joint=m->actuator_trnid[2*j];
    q[j]=d->qpos[m->jnt_qposadr[joint]];v[j]=d->qvel[m->jnt_dofadr[joint]];raw[j]=d->ctrl[j];
  }
  const double floor=GetNumberOrDefault(0,m,"s1_flat_ground_height");
  const char* names[4]={"FL","FR","RL","RR"};
  for(int i=0;i<4;i++){
    const int geom=mj_name2id(m,mjOBJ_GEOM,names[i]);
    const double* p=d->geom_xpos+3*geom;const double radius=m->geom_size[3*geom];
    mj_jacGeom(m,d,J.data(),Jr.data(),geom);
    mj_solveM(m,d,sol.data(),J.data()+2*nv,1);
    gap[i]=p[2]-radius-floor;vz[i]=mju_dot(J.data()+2*nv,d->qvel,nv);
    // Known material map and undeformed surface upper-bound prior. No evaluator
    // strain/history or future landing/contact is queried.
    material[i]=(p[0]>=.8-radius&&p[0]<=2.2+radius&&std::abs(p[1])<=.6+radius)?1.:0.;
    for(int j=0;j<nu;j++)H[i*nu+j]=sol[m->jnt_dofadr[m->actuator_trnid[2*j]]]*m->actuator_gear[6*j];
  }
  if(!w54_e_core(nu,q,v,raw,m->actuator_ctrlrange,m->actuator_forcerange,H,gap,vz,material,out,diag))throw std::runtime_error("W54 E original torque/input intersection empty");
  mju_copy(d->ctrl,out,nu);if(saved)mju_copy(saved,diag,32);
}
inline void W54Transition(const mjModel* m,mjData* d,bool terminal,double* sensors,int ns) {
  W54ApplyE(m,d);
  if(terminal)mj_forward(m,d);else mj_step(m,d);
  if(sensors)mju_copy(sensors,d->sensordata,ns);
  if(!terminal)mj_step(m,d);  // same applied position command held for exactly 2 ms
}
inline void W54TransitionFD(const mjModel* m,mjData* d,double eps,int centered,
    double* A,double* B,double* C,double* D,int ns,bool terminal) {
  const int nv=m->nv,nu=m->nu,nx=2*nv+m->na;
  auto saved=MakeUniqueMjData(mj_makeData(m)),plus=MakeUniqueMjData(mj_makeData(m)),minus=MakeUniqueMjData(mj_makeData(m));
  mj_copyData(saved.get(),m,d);mj_copyData(plus.get(),m,saved.get());
  std::vector<double> baseline_s(ns),sp(ns),sm(ns),diff(nx),direction(nv,0);
  W54Transition(m,plus.get(),terminal,baseline_s.data(),ns);
  mj_copyData(minus.get(),m,plus.get());
  auto baseline=MakeUniqueMjData(mj_makeData(m));mj_copyData(baseline.get(),m,plus.get());
  for(int col=0;col<nx+nu;col++){
    if(col>=nx&&!B&&!D)continue;
    if(col<nx&&!A&&!C)continue;
    const bool state=col<nx;const int j=state?col:col-nx;
    auto perturb=[&](mjData* dst,double sign){
      mj_copyData(dst,m,saved.get());
      if(!state)dst->ctrl[j]+=sign*eps;
      else if(j<nv){direction[j]=1; mj_integratePos(m,dst->qpos,direction.data(),sign*eps);direction[j]=0;}
      else if(j<2*nv)dst->qvel[j-nv]+=sign*eps;
      else dst->act[j-2*nv]+=sign*eps;
    };
    perturb(plus.get(),1);W54Transition(m,plus.get(),terminal,sp.data(),ns);
    const mjData* left=baseline.get();const double* left_s=baseline_s.data();double den=eps;
    if(centered){perturb(minus.get(),-1);W54Transition(m,minus.get(),terminal,sm.data(),ns);left=minus.get();left_s=sm.data();den*=2;}
    if((state&&A)||(!state&&B)){
      mj_differentiatePos(m,diff.data(),den,left->qpos,plus->qpos);
      for(int k=0;k<nv;k++)diff[nv+k]=(plus->qvel[k]-left->qvel[k])/den;
      for(int k=0;k<m->na;k++)diff[2*nv+k]=(plus->act[k]-left->act[k])/den;
      double* dst=state?A:B;const int width=state?nx:nu;
      for(int row=0;row<nx;row++)dst[row*width+j]=diff[row];
    }
    if((state&&C)||(!state&&D)){
      double* dst=state?C:D;const int width=state?nx:nu;
      for(int row=0;row<ns;row++)dst[row*width+j]=(sp[row]-left_s[row])/den;
    }
  }
  mj_copyData(d,m,saved.get());
}
}  // namespace mjpc
#endif
