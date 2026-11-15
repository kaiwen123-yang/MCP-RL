#include "w49_native_bridge.cc"
extern "C" void w50_userdata(void* p,const double* values) {
 auto* w=static_cast<World*>(p);mju_copy(w->d->userdata,values,w->m->nuserdata);
}
// Read the last solver's forces only; never call forward/step for observation.
extern "C" void w50_support(void* p,double* values) {
 auto* w=static_cast<World*>(p);mju_zero(values,14);values[0]=w->d->time;
 const char* names[4]={"FL","FR","RL","RR"};int feet[4];
 for(int i=0;i<4;++i)feet[i]=mj_name2id(w->m,mjOBJ_GEOM,names[i]);
 const int floor=mj_name2id(w->m,mjOBJ_GEOM,"floor");
 for(int k=0;k<w->d->ncon;++k) {
  const auto& c=w->d->contact[k];if(c.geom[0]!=floor && c.geom[1]!=floor)continue;
  const int other=c.geom[0]==floor?c.geom[1]:c.geom[0];double f[6];mj_contactForce(w->m,w->d,k,f);
  const double sign=c.geom[0]==floor?1.:-1.;
  const double fz=sign*(c.frame[2]*f[0]+c.frame[5]*f[1]+c.frame[8]*f[2]);
  int index=-1;for(int i=0;i<4;++i)if(other==feet[i])index=i;
  if(index<0)values[9]+=fz;else {values[1+index]+=fz;values[10+index]+=1;}
 }
 const double* opt=mjpc::GetCustomNumericData(w->m,"r10_local_contact");
 if(opt)for(int i=0;i<4;++i)if(mjpc::R10MaterialFoot(w->m,w->d,feet[i])) {
  double point[3];mju_copy3(point,w->d->geom_xpos+3*feet[i]);point[2]-=w->m->geom_size[3*feet[i]];
  const double ground=mjpc::GetCustomNumericData(w->m,"s1_flat_ground_height")[0];
  values[5+i]=opt[0]*mju_max(ground+mjpc::R10ContactRest(w->m,w->d,point)-point[2],0.);
 }
}
