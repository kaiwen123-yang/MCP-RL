#ifndef MJPC_W52_GEOMETRY_H_
#define MJPC_W52_GEOMETRY_H_
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
namespace w52 {
using Poly=std::vector<long double>;
inline long double Eval(const Poly& p,long double u) {
  long double y=0;for(auto i=p.rbegin();i!=p.rend();++i)y=y*u+*i;return y;
}
inline Poly Derivative(const Poly& p) {
  Poly q;for(size_t i=1;i<p.size();++i)q.push_back(i*p[i]);return q;
}
inline Poly Add(Poly p,const Poly& q,long double scale=1) {
  p.resize(std::max(p.size(),q.size()),0);
  for(size_t i=0;i<q.size();++i){p[i]+=scale*q[i];}
  return p;
}
inline Poly Multiply(const Poly& p,const Poly& q) {
  Poly r(p.size()+q.size()-1,0);
  for(size_t i=0;i<p.size();++i)for(size_t j=0;j<q.size();++j){r[i+j]+=p[i]*q[j];}
  return r;
}
inline Poly Compose(const Poly& p,long double offset,long double scale) {
  Poly q{0};for(auto i=p.rbegin();i!=p.rend();++i){q=Multiply(q,{offset,scale});q[0]+=*i;}return q;
}
// Derivative-root partitioning isolates every real root on this finite interval.
// Each open partition is monotone; repeated roots occur at partition endpoints.
inline std::vector<long double> Roots(Poly p,long double lo=0,long double hi=1) {
  long double scale=0;for(auto x:p)scale+=std::abs(x);
  const long double eps=std::max(1e-24L,scale*1e-17L);
  while(p.size()>1 && std::abs(p.back())<eps)p.pop_back();
  if(p.size()<2)return {};
  if(p.size()==2) {
    long double x=-p[0]/p[1];return x>=lo && x<=hi?std::vector<long double>{x}:std::vector<long double>{};
  }
  auto cuts=Roots(Derivative(p),lo,hi);cuts.insert(cuts.begin(),lo);cuts.push_back(hi);
  std::vector<long double> roots;
  for(auto x:cuts)if(std::abs(Eval(p,x))<=eps)roots.push_back(x);
  for(size_t i=1;i<cuts.size();++i) {
    long double a=cuts[i-1],b=cuts[i],fa=Eval(p,a),fb=Eval(p,b);
    if(fa*fb>=0 || std::abs(fa)<=eps || std::abs(fb)<=eps)continue;
    for(int k=0;k<80;++k) {
      long double m=(a+b)/2,fm=Eval(p,m);
      if(fa*fm>0){a=m;fa=fm;}else b=m;
    }
    roots.push_back((a+b)/2);
  }
  std::sort(roots.begin(),roots.end());
  roots.erase(std::unique(roots.begin(),roots.end(),[](auto a,auto b){return std::abs(a-b)<1e-14L;}),roots.end());
  return roots;
}
inline std::array<double,2> Range(const Poly& p,long double lo=0,long double hi=1) {
  auto pts=Roots(Derivative(p),lo,hi);pts.push_back(lo);pts.push_back(hi);
  long double mn=std::numeric_limits<long double>::infinity(),mx=-mn;
  for(auto u:pts){auto v=Eval(p,u);mn=std::min(mn,v);mx=std::max(mx,v);}
  return {double(mn),double(mx)};
}
inline Poly Quintic(double start,double velocity,double duration,double finish) {
  const long double d=finish-start,v=duration*velocity;
  return {start,v,0,10*d-6*v,-15*d+8*v,6*d-3*v};
}
struct Surface {
  double flat=0,radius=.022;
  bool material=false,anchored=false;
  std::array<double,16> patches{};
};
struct Piece {
  double begin=0,end=0;
  std::array<Poly,3> xyz;
};
struct Segment {
  double begin=0,end=0,apex=0;
  bool swing=false;
  std::array<double,3> start{},velocity{},touchdown{};
  std::vector<Piece> pieces;
  bool valid=true;
  std::string reason="available";
  double min_clearance=0,max_clearance=0,min_world_z=0,max_world_z=0,max_descent_velocity=0;
};
inline bool InMaterial(double x,double y) {
  constexpr double half=.015811388300841896;
  return x>=.8+half && x<=2.2-half && std::abs(y)<=.6-half;
}
inline Poly Ground(const Piece& p,const Surface& s,long double u) {
  Poly ground{s.flat+s.radius};
  const double x=Eval(p.xyz[0],u),y=Eval(p.xyz[1],u);
  if(!s.material || !s.anchored || !InMaterial(x,y))return ground;
  constexpr long double w=.03162277660168379L;
  for(int i=0;i<4;++i) {
    const double* a=s.patches.data()+4*i;
    if(std::abs(x-a[1])>=w || std::abs(y-a[2])>=w)continue;
    const long double sx=x>=a[1]?1:-1,sy=y>=a[2]?1:-1;
    Poly px=Add({1+sx*a[1]/w},p.xyz[0],-sx/w);
    Poly py=Add({1+sy*a[2]/w},p.xyz[1],-sy/w);
    ground=Add(ground,Multiply(px,py),a[0]*a[3]);
  }
  return ground;
}
inline std::array<double,2> Clearance(const Piece& p,const Surface& s) {
  std::vector<long double> cuts{0,1};
  auto crossing=[&](int axis,double boundary) {
    Poly v=p.xyz[axis];v[0]-=boundary;
    auto r=Roots(v);cuts.insert(cuts.end(),r.begin(),r.end());
  };
  if(s.material) {
    constexpr double half=.015811388300841896,w=.03162277660168379;
    for(double x:{.8+half,2.2-half})crossing(0,x);
    for(double y:{-.6+half,.6-half})crossing(1,y);
    if(s.anchored)for(int i=0;i<4;++i)for(int axis=0;axis<2;++axis)
      for(double shift:{-w,0.,w})crossing(axis,s.patches[4*i+axis+1]+shift);
  }
  std::sort(cuts.begin(),cuts.end());
  cuts.erase(std::unique(cuts.begin(),cuts.end(),[](auto a,auto b){return std::abs(a-b)<1e-14L;}),cuts.end());
  double mn=INFINITY,mx=-INFINITY;
  for(size_t i=1;i<cuts.size();++i) {
    auto c=Add(p.xyz[2],Ground(p,s,(cuts[i-1]+cuts[i])/2),-1);
    auto r=Range(c,cuts[i-1],cuts[i]);mn=std::min(mn,r[0]);mx=std::max(mx,r[1]);
  }
  // Include exact boundary semantics as well as both one-sided interval values.
  for(auto u:cuts){double c=Eval(p.xyz[2],u)-Eval(Ground(p,s,u),u);mn=std::min(mn,c);mx=std::max(mx,c);}
  return {mn,mx};
}
inline Segment Make(double begin,double end,bool swing,const std::array<double,3>& start,
  const std::array<double,3>& velocity,const std::array<double,3>& touchdown,const Surface& surface) {
  Segment s;s.begin=begin;s.end=end;s.swing=swing;s.start=start;s.velocity=velocity;s.touchdown=touchdown;
  s.apex=end-.20;const double T=end-begin,tol=1e-5;
  // A zero-duration old event is consumed by the caller using the original phase.
  if(T<=1e-12){s.valid=false;s.reason="zero_duration_event";return s;}
  std::array<Poly,3> xyz;
  for(int j=0;j<3;++j)xyz[j]=Quintic(start[j],velocity[j],T,touchdown[j]);
  auto add=[&](double a,double b,const Poly& z) {
    Piece p;p.begin=a;p.end=b;
    for(int j=0;j<2;++j)p.xyz[j]=Compose(xyz[j],(a-begin)/T,(b-a)/T);
    p.xyz[2]=z;s.pieces.push_back(p);
  };
  if(swing && begin<s.apex-1e-12) {
    add(begin,s.apex,Quintic(start[2],velocity[2],s.apex-begin,touchdown[2]+.005));
    add(s.apex,end,Quintic(touchdown[2]+.005,0,end-s.apex,touchdown[2]));
  } else add(begin,end,xyz[2]);
  s.min_clearance=s.min_world_z=INFINITY;s.max_clearance=s.max_world_z=-INFINITY;
  for(const auto& p:s.pieces) {
    auto c=Clearance(p,surface),z=Range(p.xyz[2]);
    s.min_clearance=std::min(s.min_clearance,c[0]);s.max_clearance=std::max(s.max_clearance,c[1]);
    s.min_world_z=std::min(s.min_world_z,z[0]);s.max_world_z=std::max(s.max_world_z,z[1]);
  }
  auto ground0=Ground(s.pieces.front(),surface,0);
  const double initial=start[2]-Eval(ground0,0);
  if(s.min_clearance<std::min(0.,initial)-tol || initial< -tol) {
    s.valid=false;s.reason="continuous_surface_clearance";
  } else if(swing && s.max_world_z>touchdown[2]+.005+tol) {
    s.valid=false;s.reason="original_apex_amplitude";
  } else if(swing && begin>=s.apex-1e-12) {
    const auto v=Range(Derivative(s.pieces[0].xyz[2]));
    s.max_descent_velocity=v[1]/T;
    if(initial<=tol){s.valid=false;s.reason="late_swing_still_grounded";}
    else if(v[1]>tol){s.valid=false;s.reason="late_swing_not_descending";}
  }
  return s;
}
inline std::array<double,3> Position(const Segment& s,double t) {
  const Piece* p=&s.pieces.back();
  for(const auto& v:s.pieces)if(t<=v.end+1e-12){p=&v;break;}
  double u=std::clamp((t-p->begin)/(p->end-p->begin),0.,1.);
  return {double(Eval(p->xyz[0],u)),double(Eval(p->xyz[1],u)),double(Eval(p->xyz[2],u))};
}
} // namespace w52
#endif
