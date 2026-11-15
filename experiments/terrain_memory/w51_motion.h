#ifndef MJPC_W51_MOTION_H_
#define MJPC_W51_MOTION_H_
#include <array>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <mujoco/mujoco.h>
#include "mjpc/tasks/quadruped/r10_contact.h"
#include "mjpc/utilities.h"
namespace mjpc {
// Updated only by the real Agent SetState/Transition, never by candidate rollouts.
struct W51Foot {
  long segment=0;
  bool ready=false,swing=false;
  double begin=0,end=0;
  std::array<double,3> anchor{},touchdown{},nominal{};
};
struct W51Motion {
  std::array<W51Foot,4> feet;
  std::array<int,4> geoms{};
  bool geometry_ready=false,active=false;
  double last_time=-1,phase=0,frequency=2,duty=.45,heading=0;
  static constexpr double shifts[4]={0,.5,.5,0}; // native FL,RL,FR,RR
  static double Smooth(double u) {return u*u*u*(10+u*(-15+6*u));}
  static double H0(double u) {return 1-Smooth(u);}
  static double H1(double u) {return u-6*u*u*u+8*u*u*u*u-3*u*u*u*u*u;}
  void Segment(int i,double t,bool& swing,long& id,double& end) const {
    const double x=phase/(2*mjPI)+(t-last_time)*frequency-shifts[i];
    const long cycle=std::floor(x+.5);
    const double w=x-cycle,half=(1-duty)*.5;
    swing=w>-half && w<half;
    id=2*cycle+(swing?0:(w>=half?1:-1));
    end=t+(swing?half-w:(w>=half?1-half-w:-half-w))/frequency;
  }
  std::array<double,3> Touchdown(const mjModel* m,const mjData* d,int i,double t) const {
    const double tau=t-d->time,psi=heading+d->userdata[83]*tau;
    const double co=std::cos(psi),si=std::sin(psi),ts=duty/frequency;
    const auto& r=feet[i].nominal;
    std::array<double,3> p={d->qpos[0]+d->userdata[78]*(tau+.5*ts)+co*r[0]-si*r[1],
      d->qpos[1]+d->userdata[79]*(tau+.5*ts)+si*r[0]+co*r[1],0};
    const double* flat=GetCustomNumericData(m,"s1_flat_ground_height");
    p[2]=(flat?flat[0]:0)+R10ContactRest(m,d,p.data())+m->geom_size[3*geoms[i]];
    return p;
  }
  void Observe(const mjModel* m,const mjData* d,const int* ids,double phase_now,
               double hz,double duty_now,bool moving,double amplitude) {
    if(d->time<last_time) {for(auto& f:feet)f.ready=false;}
    if(!geometry_ready) {
      mjData* home=mj_makeData(m);
      const int key=mj_name2id(m,mjOBJ_KEY,"home");
      if(key<0)mju_error("W51 original home geometry missing");
      mj_resetDataKeyframe(m,home,key);mj_kinematics(m,home);
      for(int i=0;i<4;++i) {
        geoms[i]=ids[i];
        for(int j=0;j<3;++j)feet[i].nominal[j]=home->geom_xpos[3*ids[i]+j]-home->qpos[j];
      }
      mj_deleteData(home);geometry_ready=true;
    }
    // The native gait switch briefly applies its defaults; capture after the
    // original explicit 2Hz/.45/.005 settings are applied at the same timestamp.
    active=moving && hz==2. && duty_now==.45 && amplitude==.005;
    phase=phase_now;frequency=hz;duty=duty_now;last_time=d->time;
    heading=d->userdata[91];
    if(!active) {if(!moving)for(auto& f:feet)f.ready=false;return;}
    for(int i=0;i<4;++i) {
      bool sw;long id;double end;Segment(i,d->time,sw,id,end);
      auto& f=feet[i];
      if(!f.ready || f.segment!=id) {
        f.ready=true;f.segment=id;f.swing=sw;f.begin=d->time;f.end=end;
        for(int j=0;j<3;++j)f.anchor[j]=d->geom_xpos[3*geoms[i]+j];
        if(sw)f.touchdown=Touchdown(m,d,i,end);
      }
    }
  }
};
} // namespace mjpc
#endif
