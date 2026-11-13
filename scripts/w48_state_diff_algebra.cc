#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
int main(int argc,char** argv) {
 char err[2048];mjModel* m=mj_loadXML(argv[1],nullptr,err,sizeof(err));
 if(!m){std::cerr<<err;return 1;}
 int ns=m->nq+m->nv+m->na,nx=2*m->nv+m->na;
 std::vector<double> a(ns),b,ds(nx),v(m->nv,0.);std::ifstream in(argv[2]);
 for(auto& x:a)in>>x;
 std::cout.precision(17);
 for(int kind=0;kind<5;++kind) {
  b=a;
  if(kind==1)for(int j=0;j<4;++j)b[3+j]=-a[3+j];
  if(kind==2||kind==3){v[3]=(kind==2?1.:-1.)*1e-8;mj_integratePos(m,b.data(),v.data(),1.);}
  if(kind==4){b[0]+=.001;b[m->nq]+=.002;if(m->na)b[m->nq+m->nv]+=.003;}
  mjpc::StateDiff(m,ds.data(),a.data(),b.data(),1.);
  std::cout<<"{\"case\":"<<kind<<",\"ds\":[";
  for(int j=0;j<nx;++j){if(j)std::cout<<",";std::cout<<ds[j];}
  std::cout<<"]}\n";
 }
 mj_deleteModel(m);
}
