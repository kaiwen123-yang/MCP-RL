#include <mujoco/mujoco.h>
#include <string_view>
#include "mjpc/planners/ilqg/r29_projection.h"
#include "mjpc/tasks/quadruped/r10_contact.h"
namespace mjpc {
double* GetCustomNumericData(const mjModel* m,std::string_view name) {
 for(int i=0;i<m->nnumeric;++i)if(std::string_view(m->names+m->name_numericadr[i])==name)return m->numeric_data+m->numeric_adr[i];
 return nullptr;
}
}
struct World {mjModel* m;mjData* d;};
extern "C" {
void* w46_create(const char* path,char* error) {
 mjModel* m=mj_loadXML(path,nullptr,error,2048);if(!m)return nullptr;
 m->opt.timestep=.0002;m->opt.integrator=mjINT_IMPLICITFAST;m->opt.disableflags|=mjDSBL_WARMSTART;
 for(int i=0;i<m->njnt;++i)m->jnt_solimp[mjNIMP*i]=0;
 mjcb_contactfilter=mjpc::R10ContactFilter;mjcb_passive=mjpc::R10ContactForce;
 return new World{m,mj_makeData(m)};
}
void w46_reset(void* p,const double* state,const double* userdata,double time) {
 auto* w=static_cast<World*>(p);mj_resetData(w->m,w->d);
 mju_copy(w->d->qpos,state,19);mju_copy(w->d->qvel,state+19,18);
 mju_copy(w->d->userdata,userdata,w->m->nuserdata);w->d->time=time;
}
void w46_diff(void* p,const double* ref,const double* state,double* dx) {
 auto* w=static_cast<World*>(p);mj_differentiatePos(w->m,dx,1.,ref,state);
 mju_sub(dx+18,state+19,ref+19,18);
}
void w46_project(void* p,const double* state,const double* H,double* u,double* info) {
 auto* w=static_cast<World*>(p);auto r=mjpc::R29Project(w->m,state,H,true,u);
 info[0]=r.converged;info[1]=r.primal;info[2]=r.kkt_scaled;info[3]=r.restoration;
 for(int i=0;i<12;++i)info[4+i]=r.active[i];
}
void w46_hold(void* p,const double* u,double* state,double* record) {
 auto* w=static_cast<World*>(p);mju_copy(w->d->ctrl,u,12);
 int trunk=mj_name2id(w->m,mjOBJ_BODY,"trunk"),floorid=mj_name2id(w->m,mjOBJ_GEOM,"floor");
 for(int k=0;k<10;++k) {
  mj_step(w->m,w->d);mju_copy(record+k*13,w->d->actuator_force,12);
  double worst=0;
  for(int g=0;g<w->m->ngeom;++g)if(w->m->geom_bodyid[g]==trunk &&
      ((w->m->geom_contype[g]&w->m->geom_conaffinity[floorid]) || (w->m->geom_contype[floorid]&w->m->geom_conaffinity[g]))) {
   const double* r=w->d->geom_xmat+9*g;const double* z=w->m->geom_size+3*g;double extent=0;
   if(w->m->geom_type[g]==mjGEOM_BOX)extent=std::abs(r[6])*z[0]+std::abs(r[7])*z[1]+std::abs(r[8])*z[2];
   else if(w->m->geom_type[g]==mjGEOM_CAPSULE)extent=z[0]+std::abs(r[8])*z[1];
   else if(w->m->geom_type[g]==mjGEOM_SPHERE)extent=z[0];
   double ground=mjpc::GetCustomNumericData(w->m,"s1_flat_ground_height")[0];
   worst=std::max(worst,std::max(w->m->geom_margin[g],w->m->geom_margin[floorid])-(w->d->geom_xpos[3*g+2]-extent-ground));
  }
  record[k*13+12]=worst;
 }
 mju_copy(state,w->d->qpos,19);mju_copy(state+19,w->d->qvel,18);
}
void w46_destroy(void* p) {auto* w=static_cast<World*>(p);mj_deleteData(w->d);mj_deleteModel(w->m);delete w;}
}
