// R10 C: stateless instantaneous normal stiffness, evaluated per candidate.
// This approximation has no hidden material q/e and no propagated terrain memory.
#ifndef MJPC_R10_CONTACT_H_
#define MJPC_R10_CONTACT_H_
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
inline bool R10MaterialFoot(const mjModel* m,const mjData* d,int geom) {
  const char* name=mj_id2name(m,mjOBJ_GEOM,geom);
  if (!name || (std::string_view(name)!="FL" && std::string_view(name)!="FR" &&
                std::string_view(name)!="RL" && std::string_view(name)!="RR")) return false;
  const double* p=d->geom_xpos+3*geom;
  constexpr double half=0.015811388300841896;
  return p[0]>=.8+half && p[0]<=2.2-half && mju_abs(p[1])<=.6-half;
}

inline int R10ContactFilter(const mjModel* m,mjData* d,int g1,int g2) {
  if (!GetCustomNumericData(m,"r10_local_contact")) return 0;
  return R10MaterialFoot(m,d,g1) || R10MaterialFoot(m,d,g2);
}

inline void R10ContactForce(const mjModel* m,mjData* d) {
  const double* opt=GetCustomNumericData(m,"r10_local_contact");
  if (!opt) return;
  const double floor=GetCustomNumericData(m,"s1_flat_ground_height")[0];
  for (const char* name : {"FL","FR","RL","RR"}) {
    const int geom=mj_name2id(m,mjOBJ_GEOM,name);
    if (!R10MaterialFoot(m,d,geom)) continue;
    const double radius=m->geom_size[3*geom];
    double point[3];mju_copy3(point,d->geom_xpos+3*geom);point[2]-=radius;
    const double depth=mju_max(floor-point[2],0);
    double velocity[6];mj_objectVelocity(m,d,mjOBJ_GEOM,geom,velocity,0);
    const double vx=velocity[3]-radius*velocity[1];
    const double vy=velocity[4]+radius*velocity[0];
    const double fz=opt[0]*depth;
    const double slip=mju_sqrt(vx*vx+vy*vy+.01*.01);
    double force[3]={-.5*fz*vx/slip,-.5*fz*vy/slip,fz};
    const double torque[3]={0,0,0};
    mj_applyFT(m,d,force,torque,point,m->geom_bodyid[geom],d->qfrc_passive);
  }
}
}  // namespace mjpc
#endif
