#ifndef W54_E_CORE_H_
#define W54_E_CORE_H_
#include <algorithm>
#include <cmath>
inline double w54_quintic(double x){x=std::clamp(x,0.0,1.0);return x*x*x*(10+x*(-15+6*x));}
// r48: complete normal acceleration, including the raw-action acceleration,
// is the target. The torque correction alone is never called deceleration.
inline bool w54_e_core(int nu,const double* q,const double* v,const double* raw,
    const double* urange,const double* trange,const double* H,const double* drift,
    const double* gap,const double* vz,const double* material,double* applied,double* diagnostic){
 constexpr double hold=.002,delay=.022,landing_speed=.05,reserve=.25;
 double lo[12],hi[12],tau0[12],tau[12],request[4]={},weight[4]={},a_raw[4]={};
 bool active=false,empty=false;double L=1e-12;
 for(int j=0;j<nu;j++){
  lo[j]=std::max(trange[2*j],60*(urange[2*j]-q[j])-5*v[j]);
  hi[j]=std::min(trange[2*j+1],60*(urange[2*j+1]-q[j])-5*v[j]);
  empty=empty||(lo[j]>hi[j]);tau0[j]=std::min(std::max(60*(raw[j]-q[j])-5*v[j],lo[j]),hi[j]);
  tau[j]=tau0[j];applied[j]=raw[j];
 }
 for(int i=0;i<4;i++){
  a_raw[i]=drift[i];double a_max=drift[i];
  for(int j=0;j<nu;j++){
   const double h=H[i*nu+j];a_raw[i]+=h*tau0[j];a_max+=h*(h>0?hi[j]:lo[j]);
  }
  const double available=reserve*std::max(0.0,a_max);
  const double speed=std::max(0.0,-vz[i]),excess=std::max(0.0,speed*speed-landing_speed*landing_speed);
  const double lead=speed*delay+excess/(2*std::max(available,1e-6));
  const double gate=material[i]*w54_quintic((lead-std::max(gap[i],0.0))/std::max(lead,1e-9))*
      w54_quintic((speed-landing_speed)/landing_speed);
  const double stopping=excess/(2*std::max(gap[i]-speed*hold,landing_speed*hold));
  const double target=std::max(a_raw[i],std::min(available,stopping));
  request[i]=gate*(target-a_raw[i]);weight[i]=gate;active=active||(request[i]>0);
  diagnostic[i*10]=gap[i];diagnostic[i*10+1]=vz[i];diagnostic[i*10+2]=available;
  diagnostic[i*10+3]=lead;diagnostic[i*10+4]=gate;diagnostic[i*10+5]=a_raw[i]+request[i];
  diagnostic[i*10+6]=a_raw[i];diagnostic[i*10+7]=material[i];
  diagnostic[i*10+8]=a_raw[i];diagnostic[i*10+9]=request[i];
  for(int j=0;j<nu;j++)L+=gate*H[i*nu+j]*H[i*nu+j];
 }
 if(!active)return true;
 if(empty)return false;
 for(int k=0;k<24;k++){
  double gradient[12]={};
  for(int i=0;i<4;i++){
   double increment=0;for(int j=0;j<nu;j++)increment+=H[i*nu+j]*(tau[j]-tau0[j]);
   const double error=weight[i]*(increment-request[i]);
   for(int j=0;j<nu;j++)gradient[j]+=H[i*nu+j]*error;
  }
  for(int j=0;j<nu;j++)tau[j]=std::clamp(tau[j]-gradient[j]/L,lo[j],hi[j]);
 }
 for(int j=0;j<nu;j++)applied[j]=std::clamp(q[j]+(tau[j]+5*v[j])/60,urange[2*j],urange[2*j+1]);
 for(int i=0;i<4;i++)for(int j=0;j<nu;j++)diagnostic[i*10+6]+=H[i*nu+j]*(tau[j]-tau0[j]);
 return true;
}
#endif
