#include "s1_go2_w46_native_bridge.cc"
extern "C" {
void w49_mocap(void* p,const double* pos,const double* quat) {
 auto* w=static_cast<World*>(p);
 mju_copy(w->d->mocap_pos,pos,3*w->m->nmocap);
 mju_copy(w->d->mocap_quat,quat,4*w->m->nmocap);
}
int w49_integration_size(void* p) {
 return mj_stateSize(static_cast<World*>(p)->m,mjSTATE_INTEGRATION);
}
void w49_integration(void* p,double* state) {
 auto* w=static_cast<World*>(p);
 mj_getState(w->m,w->d,state,mjSTATE_INTEGRATION);
}
}
extern "C" double w49_time(void* p) {return static_cast<World*>(p)->d->time;}
