// Keep the frozen constitutive equations, Newton norm and backtracking from W55.
// Each 8-lane subgroup cooperatively evaluates four finite differences. A CPU
// worker computes the same forward differences serially for a single history.
#include "s1_w55_c_column.cu"

HD inline int material_lane() {
#ifdef __CUDA_ARCH__
    return threadIdx.x & 7;
#else
    return 0;
#endif
}
HD inline double material_solve(const Model& m, const double* prev, double n, double dt,
                               bool fixed, double* x, double* stress) {
    for (int j=0;j<4;j++) x[j]=prev[j];
    double r[4], s[2];
    residual(m,x,prev,n,dt,fixed,r,s);
    for (int it=0;it<=30;it++) {
        double err=0.;
        for (int j=0;j<4;j++) err=fmax(err,fabs(r[j]));
        if (err<=1e-8 || it==30) {*stress=s[0];return err;}
        double a[4][5], rp[4]={};
#ifdef __CUDA_ARCH__
        int lane=material_lane(), j=lane%4;
        unsigned group=0xffu << (threadIdx.x & 24);
        if (lane<4) {
            double xp[4], sp[2];
            for (int k=0;k<4;k++) xp[k]=x[k];
            double dx=j<2?1e-8:dt*1e-5;
            xp[j]+=dx;
            residual(m,xp,prev,n,dt,fixed,rp,sp);
            for (int k=0;k<4;k++) rp[k]=(rp[k]-r[k])/dx;
        }
        for (int c=0;c<4;c++) for(int k=0;k<4;k++)
            a[k][c]=__shfl_sync(group,rp[k],c,8);
#else
        for (int j=0;j<4;j++) {
            double xp[4], sp[2];
            for (int k=0;k<4;k++) xp[k]=x[k];
            double dx=j<2?1e-8:dt*1e-5;
            xp[j]+=dx;
            residual(m,xp,prev,n,dt,fixed,rp,sp);
            for(int k=0;k<4;k++) a[k][j]=(rp[k]-r[k])/dx;
        }
#endif
        for (int k=0;k<4;k++) a[k][4]=-r[k];
        double change[4];
        if (!linear4(a,change)) return INFINITY;
        double alpha=1., rr[4], xx[4], old=norm4(r);
        for(int j=0;j<=16;j++) {
            for(int k=0;k<4;k++) xx[k]=x[k]+alpha*change[k];
            residual(m,xx,prev,n,dt,fixed,rr,s);
            if(norm4(rr)<=(1.-1e-4*alpha)*old || j==16) break;
            alpha*=.5;
        }
        for(int k=0;k<4;k++){r[k]=rr[k];x[k]=xx[k];}
    }
    return INFINITY;
}

HD inline void material_history(const Model& m,const double *n,int K,int T,double dt,double *out) {
    double x[4]={};
    for(int t=0;t<T;t++) {
        double free[4]={}, loaded[4], s=0.,sl=0., err=0.;
        if (x[0]!=0 || x[1]!=0 || x[2]!=0 || x[3]!=0)
            err=material_solve(m,x,0.,dt,false,free,&s);
        bool active=false;
        if (n[t*K]<=free[0]) {
            err=fmax(err,material_solve(m,x,n[t*K],dt,true,loaded,&sl));
            active=sl<0;
        }
        for(int k=0;k<4;k++) x[k]=active?loaded[k]:free[k];
        if (material_lane()==0) {
            for(int k=0;k<4;k++) out[t*K*6+k]=x[k];
            out[t*K*6+4]=active?sl:0.;
            out[t*K*6+5]=err;
        }
    }
}

extern "C" void batched_cpu(const Model* m,const double* n,int K,int T,double dt,double* out) {
    #pragma omp parallel for schedule(dynamic)
    for(int k=0;k<K;k++) material_history(*m,n+k,K,T,dt,out+k*6);
}

#ifdef __CUDACC__
__global__ void grouped_history(const Model* m,const double *n,int K,int T,double dt,double *out) {
    int k=(threadIdx.x+blockIdx.x*blockDim.x)/8;
    if(k<K) material_history(*m,n+k,K,T,dt,out+6*k);
}
// Python retains the device model and arrays over both warm-up and timed runs.
extern "C" int batched_gpu(const Model* dm,const double* dn,int K,int T,double dt,
                          double* dout,cudaStream_t stream) {
    grouped_history<<<(K*8+127)/128,128,0,stream>>>(dm,dn,K,T,dt,dout);
    return int(cudaGetLastError());
}
#endif
