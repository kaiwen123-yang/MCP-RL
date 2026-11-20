#include "mjpc/w54_e_native.h"
#include <fstream>
#include <iostream>
namespace mjpc {
double* GetCustomNumericData(const mjModel* m,std::string_view name){
  for(int i=0;i<m->nnumeric;i++)if(std::string_view(m->names+m->name_numericadr[i])==name)return m->numeric_data+m->numeric_adr[i];
  return nullptr;
}
}
int main(int argc,char**argv){
 char error[2048];mjModel*m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
 if(!m){std::cerr<<error;return 2;}mjData*d=mj_makeData(m);
 m->opt.timestep=.001;m->opt.disableflags|=mjDSBL_WARMSTART;
 for(int j=0;j<m->njnt;j++)m->jnt_solimp[mjNIMP*j]=0;
 for(int j=0;j<m->ngeom;j++)m->geom_solimp[mjNIMP*j]=0;
 for(int j=0;j<m->npair;j++)m->pair_solimp[mjNIMP*j]=0;
 std::ifstream input(argv[2],std::ios::binary);auto read=[&](double*p,int n){input.read((char*)p,n*sizeof(double));};
 read(&d->time,1);read(d->qpos,m->nq);read(d->qvel,m->nv);read(d->ctrl,m->nu);
 read(d->mocap_pos,3*m->nmocap);read(d->mocap_quat,4*m->nmocap);read(d->userdata,m->nuserdata);
 auto saved=mjpc::MakeUniqueMjData(mj_makeData(m));mj_copyData(saved.get(),m,d);
 std::ofstream output(argv[3],std::ios::binary);auto write=[&](const double*p,int n){output.write((const char*)p,n*sizeof(double));};
 w55_native_contact_filter();double diag[4*W55Diag];mjpc::W54ApplyE(m,d,diag);write(d->ctrl,m->nu);write(diag,4*W55Diag);
 mj_copyData(d,m,saved.get());mjpc::W54Transition(m,d,false,nullptr,0);write(d->qpos,m->nq);write(d->qvel,m->nv);
 mj_copyData(d,m,saved.get());const int nx=2*m->nv+m->na;
 std::vector<double>A(nx*nx),B(nx*m->nu);
 mjpc::W54TransitionFD(m,d,1e-6,0,A.data(),B.data(),nullptr,nullptr,0,false);write(A.data(),A.size());write(B.data(),B.size());
 mj_deleteData(d);mj_deleteModel(m);
}
