#include <algorithm>
#include <cstring>
#include <vector>
#include "r29_projection_qp1.h"

extern "C" int mju_boxQPoption(double*,double*,int*,const double*,const double*,int,
                               const double*,const double*,int,double,double,double,double,char*,int);

extern "C" int native_projection(int n,const double* H,const double* center,
                                  const double* lower,const double* upper,double* u,
                                  char* log,int log_size) {
  std::vector<int> free;
  bool inside=true;
  for(int i=0;i<n;++i) {
    u[i]=std::clamp(center[i],lower[i],upper[i]);
    if(lower[i]<upper[i]) free.push_back(i);
    inside=inside && center[i]>=lower[i] && center[i]<=upper[i];
  }
  if(inside || free.empty()) return -2;
  const int nf=free.size();mjpc::BoxQP qp;qp.Allocate(nf);
  for(int a=0;a<nf;++a) {
    int i=free[a];qp.lower[a]=lower[i];qp.upper[a]=upper[i];qp.res[a]=u[i];
    for(int j=0;j<n;++j) {
      qp.g[a]-=H[i*n+j]*center[j];
      if(lower[j]==upper[j]) qp.g[a]+=H[i*n+j]*u[j];
    }
    for(int b=0;b<nf;++b)qp.H[a*nf+b]=H[i*n+free[b]];
  }
  int result=mju_boxQPoption(qp.res.data(),qp.R.data(),qp.index.data(),qp.H.data(),qp.g.data(),
      nf,qp.lower.data(),qp.upper.data(),100,1e-16,.5,1e-22,.1,log,log_size);
  for(int a=0;a<nf;++a)u[free[a]]=qp.res[a];
  return result;
}

extern "C" int refine_projection(int n,const double* H,const double* center,
                                  const double* lower,const double* upper,double* u,
                                  int* iterations,char* status,int status_size) {
  mjpc::R29Projection r;
  r.center.assign(center,center+n);r.lower.assign(lower,lower+n);r.upper.assign(upper,upper+n);
  r.fixed.resize(n);
  for(int i=0;i<n;++i)r.fixed[i]=lower[i]==upper[i];
  bool ok=mjpc::R29RefineProjection(H,r,u);
  *iterations=r.refinement_iterations;
  std::strncpy(status,r.refinement_status,status_size-1);status[status_size-1]='\0';
  return ok?r.nfree:-1;
}
