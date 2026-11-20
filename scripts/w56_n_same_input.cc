#include "mjpc/w54_e_native.h"
#include <fstream>
#include <iostream>
namespace mjpc {
double* GetCustomNumericData(const mjModel* model,std::string_view name) {
  int id=mj_name2id(model,mjOBJ_NUMERIC,std::string(name).c_str());
  return id>=0?model->numeric_data+model->numeric_adr[id]:nullptr;
}
}
int main(int argc,char** argv){
  char error[2048];mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));if(!m){std::cerr<<error;return 2;}
  mjData* d=mj_makeData(m);m->opt.timestep=.001;m->opt.disableflags|=mjDSBL_WARMSTART;
  for(int i=0;i<m->njnt;++i)m->jnt_solimp[mjNIMP*i]=0;
  for(int i=0;i<m->ngeom;++i)m->geom_solimp[mjNIMP*i]=0;
  for(int i=0;i<m->npair;++i)m->pair_solimp[mjNIMP*i]=0;
  w55_native_contact_filter();
  std::ifstream in(argv[2],std::ios::binary);std::ofstream out(argv[3],std::ios::binary);
  auto read=[&](double* x,int n){in.read((char*)x,n*sizeof(double));};
  auto write=[&](const double* x,int n){out.write((char*)x,n*sizeof(double));};
  read(&d->time,1);read(d->qpos,m->nq);read(d->qvel,m->nv);read(d->ctrl,m->nu);
  read(d->mocap_pos,3*m->nmocap);read(d->mocap_quat,4*m->nmocap);read(d->userdata,m->nuserdata);
  double schedule[176][12];read(&schedule[0][0],176*12);
  auto saved=mjpc::MakeUniqueMjData(mj_makeData(m));mj_copyData(saved.get(),m,d);
  double raw[12],command[12],diagnostic[4*W55Diag];mju_copy(raw,d->ctrl,12);
  int status=w55_project(m,d,raw,command,diagnostic);double code=status;write(&code,1);write(command,12);write(diagnostic,4*W55Diag);
  mj_copyData(d,m,saved.get());
  status=w56_progress(m,d,raw,command);code=status;write(&code,1);write(command,12);
  for(int alternative=0;alternative<2;++alternative){
    mj_copyData(d,m,saved.get());double start=d->qpos[0],effort=0,minGap=1e9,margin=1e9;int first=0,changed=0;
    for(int k=0;k<175;++k){
      mju_copy(raw,schedule[k],12);
      if(alternative){int issue=w56_progress(m,d,schedule[k],raw);if(issue&&!first)first=9;
        for(int j=0;j<12;++j)if(std::abs(raw[j]-schedule[k][j])>1e-6){++changed;break;}}
      int issue=w55_project(m,d,raw,command,diagnostic);if(issue&&!first)first=8;
      mju_copy(d->ctrl,command,12);
      for(int sub=0;sub<2;++sub){w55_native_step(m,d,true);int bad=w55_checks(m,d,&minGap,&margin);if(bad&&!first)first=bad;
        effort+=mju_dot(d->actuator_force,d->actuator_force,12);}
    }
    double summary[7]={d->qpos[0]-start,d->qpos[2],effort/350.,double(first),double(changed),minGap,margin};write(summary,7);
  }
  mj_deleteData(d);mj_deleteModel(m);return 0;
}
