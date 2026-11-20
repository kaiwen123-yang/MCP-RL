#include <mujoco/mujoco.h>
#include "support.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>

int main(int argc,char** argv){
  char error[2048];mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m){std::cerr<<error;return 2;}
  mjData* d=mj_makeData(m);m->opt.timestep=.001;m->opt.disableflags|=mjDSBL_WARMSTART;
  for(int i=0;i<m->njnt;++i)m->jnt_solimp[mjNIMP*i]=0;
  for(int i=0;i<m->ngeom;++i)m->geom_solimp[mjNIMP*i]=0;
  for(int i=0;i<m->npair;++i)m->pair_solimp[mjNIMP*i]=0;
  w55_native_contact_filter();
  std::ifstream in(argv[2],std::ios::binary);
  auto read=[&](double* x,int n){in.read(reinterpret_cast<char*>(x),n*sizeof(double));};
  read(&d->time,1);read(d->qpos,m->nq);read(d->qvel,m->nv);read(d->ctrl,m->nu);
  read(d->mocap_pos,3*m->nmocap);read(d->mocap_quat,4*m->nmocap);read(d->userdata,m->nuserdata);
  double gap=1e9,margin=1e9;
  if(argc==4){
    for(int i=0;i<std::atoi(argv[3]);++i)w55_native_step(m,d,true);
    std::cout<<w55_checks(m,d,&gap,&margin)<<"\n";return 0;
  }
  double schedule[176][12];read(&schedule[0][0],176*12);
  std::ofstream out(argv[3]);out<<std::setprecision(17);
  double t0=d->time,phase=std::atof(argv[4]);bool candidate=std::atoi(argv[5]);
  for(int k=0;k<175;++k){
    double u[12],applied[12],dg[96];mju_copy(u,schedule[k],12);
    int generator=candidate?w59_handoff(m,d,schedule[k],phase+(d->time-t0)*4*mjPI,2.,.45,.03,u):0;
    int e=w55_project(m,d,u,applied,dg);
    out<<d->time<<" "<<d->qpos[0]<<" "<<d->qpos[2]<<" "<<generator<<" "<<e;
    for(int j=0;j<4;++j)out<<" "<<dg[24*j+10];
    for(double a:applied)out<<" "<<a;
    mju_copy(d->ctrl,applied,12);
    int code[2];
    for(int sub=0;sub<2;++sub){w55_native_step(m,d,true);code[sub]=w55_checks(m,d,&gap,&margin);}
    out<<" "<<code[0]<<" "<<code[1]<<" "<<d->qpos[0]<<" "<<d->qpos[2]<<"\n";
  }
}
