#ifndef W54_E_CORE_H_
#define W54_E_CORE_H_
#include <algorithm>
#include <cmath>
// Pure deterministic action map. No persistent state, random draws, material truth,
// contact cache, or nominal-trajectory lookup is read here.
inline double w54_quintic(double x) {
  x=std::clamp(x,0.0,1.0);return x*x*x*(10.0+x*(-15.0+6.0*x));
}
inline bool w54_e_core(int nu,const double* q,const double* v,const double* raw,
    const double* urange,const double* trange,const double* H,const double* gap,
    const double* vz,const double* material,double* applied,double* diagnostic) {
  constexpr double hold=.002, delay=.022, landing_speed=.05, reserve=.25;
  double lower[12],upper[12],tau0[12],tau[12],request[4]={},weight[4]={};
  bool active=false,empty=false;double lipschitz=1e-12;
  for(int j=0;j<nu;j++){
    lower[j]=std::max(trange[2*j],60*(urange[2*j]-q[j])-5*v[j]);
    upper[j]=std::min(trange[2*j+1],60*(urange[2*j+1]-q[j])-5*v[j]);
    empty=empty||(lower[j]>upper[j]);
    tau0[j]=std::min(std::max(60*(raw[j]-q[j])-5*v[j],lower[j]),upper[j]);
    tau[j]=tau0[j];applied[j]=raw[j];
  }
  for(int i=0;i<4;i++){
    const double speed=std::max(0.0,-vz[i]);
    double available=0;
    for(int j=0;j<nu;j++){
      const double h=H[i*nu+j];
      available+=h*(h>0?upper[j]-tau0[j]:lower[j]-tau0[j]);
    }
    available=std::max(0.0,reserve*available);
    const double excess=std::max(0.0,speed*speed-landing_speed*landing_speed);
    const double lead=speed*delay+excess/(2*std::max(available,1e-6));
    const double gate=material[i]*w54_quintic((lead-std::max(gap[i],0.0))/std::max(lead,1e-9))*
        w54_quintic((speed-landing_speed)/landing_speed);
    const double stopping=excess/(2*std::max(gap[i]-speed*hold,landing_speed*hold));
    request[i]=gate*std::min(available,stopping);weight[i]=gate;
    active=active||(request[i]>0);
    diagnostic[8*i]=gap[i];diagnostic[8*i+1]=vz[i];diagnostic[8*i+2]=available;
    diagnostic[8*i+3]=lead;diagnostic[8*i+4]=gate;diagnostic[8*i+5]=request[i];
    diagnostic[8*i+6]=0;diagnostic[8*i+7]=material[i];
    for(int j=0;j<nu;j++)lipschitz+=gate*H[i*nu+j]*H[i*nu+j];
  }
  if(!active)return true;
  if(empty)return false;
  // Frozen projected gradient map over the original complete torque/input box.
  // Every candidate and derivative sees these same 24 iterations. This is an
  // execution map inside native optimization, not a post-prediction filter.
  for(int k=0;k<24;k++){
    double grad[12]={};
    for(int i=0;i<4;i++){
      double achieved=0;for(int j=0;j<nu;j++)achieved+=H[i*nu+j]*(tau[j]-tau0[j]);
      const double error=weight[i]*(achieved-request[i]);
      for(int j=0;j<nu;j++)grad[j]+=H[i*nu+j]*error;
    }
    for(int j=0;j<nu;j++)tau[j]=std::clamp(tau[j]-grad[j]/lipschitz,lower[j],upper[j]);
  }
  for(int j=0;j<nu;j++)applied[j]=std::clamp(q[j]+(tau[j]+5*v[j])/60,urange[2*j],urange[2*j+1]);
  for(int i=0;i<4;i++)for(int j=0;j<nu;j++)diagnostic[8*i+6]+=H[i*nu+j]*(tau[j]-tau0[j]);
  return true;
}
#endif
