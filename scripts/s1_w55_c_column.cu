// Axisymmetric evaluation of the existing FICNN/PICNN derivatives. Each thread
// advances one independent world-column history, with its own e and q.
#include <cmath>
#include <algorithm>
#include <limits>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HD __host__ __device__
#else
#define HD
#endif

struct Elastic { double wi[3][10],b[10],wo[10],skip[3],slope0; };
struct Model {
    Elastic eq,ov;
    double v[3][10],bv[10],g1[3][3],bg1[3],u1[3][10],y1[3][10],b1[10];
    double gu[10][10],bgu[10],gx[10][3],bgx[3],u2[10],x2[3];
    double im[3],is[3],rm[3],rs[3],sm[3],ss[3],es,ds;
};
HD inline double sp(double x){return fmax(x,0.)+log1p(exp(-fabs(x)));}
HD inline double sig(double x){double v=exp(-fabs(x));return x>=0?1./(1.+v):v/(1.+v);}
HD inline void inv(const double *a, const double *mid, const double *scale, bool cube, double *z) {
    const double x=a[0], y=a[1], x2=x*x,y2=y*y;
    z[0]=(x+2.*y-mid[0])/scale[0];z[1]=(x2+2.*y2-mid[1])/scale[1];
    z[2]=((cube?x*x2+2.*y*y2:x2*x2+2.*y2*y2)-mid[2])/scale[2];
}
HD inline void tensor_gradient(const double *a, const double *slope, const double *scales,
                              double slope0, double factor, double *out){
    for(int i=0;i<2;++i) {
        double b=a[i];
        out[i]=factor*(slope[0]/scales[0]-slope0+2.*b*slope[1]/scales[1]+4.*b*b*b*slope[2]/scales[2]);
    }
}
HD inline void elastic(const Model& m,const Elastic& w,const double *a,double *out){
    double x[3],slope[3];inv(a,m.im,m.is,false,x);
    for(int k=0;k<3;++k)slope[k]=w.skip[k];
    for(int i=0;i<10;++i){
        double z=w.b[i];for(int k=0;k<3;++k)z+=x[k]*w.wi[k][i];
        double factor=sig(z)*w.wo[i];
        for(int k=0;k<3;++k)slope[k]+=w.wi[k][i]*factor;
    }
    tensor_gradient(a,slope,m.is,w.slope0,m.es,out);
}
HD inline void dissipation(const Model& m,const double *rate,const double *p,double *out){
    double x[3],y[3],v[10],g1[3],gu[10],gx[3];
    inv(rate,m.rm,m.rs,false,x);inv(p,m.sm,m.ss,true,y);
    for(int j=0;j<10;++j){
        double z=m.bv[j];for(int k=0;k<3;++k)z+=y[k]*m.v[k][j];v[j]=tanh(z);
    }
    for(int i=0;i<3;++i){
        double g=m.bg1[i],h=m.bgx[i];
        for(int k=0;k<3;++k)g+=y[k]*m.g1[k][i];
        for(int k=0;k<10;++k)h+=v[k]*m.gx[k][i];
        g1[i]=sp(g);gx[i]=sp(h);
    }
    for(int i=0;i<10;++i){
        double z=m.bgu[i];for(int k=0;k<10;++k)z+=v[k]*m.gu[k][i];
        gu[i]=sp(z)*m.u2[i];
    }
    double slope[3],slope0=gx[0]*m.x2[0];
    for(int k=0;k<3;++k)slope[k]=gx[k]*m.x2[k];
    for(int i=0;i<10;++i){
        double z=m.b1[i];for(int k=0;k<3;++k)z+=y[k]*m.y1[k][i];
        double zz=z;
        for(int k=0;k<3;++k){
            z+=x[k]*g1[k]*m.u1[k][i];
            zz-=m.rm[k]/m.rs[k]*g1[k]*m.u1[k][i];
        }
        double f=sig(z)*gu[i];slope0+=sig(zz)*gu[i]*g1[0]*m.u1[0][i];
        for(int k=0;k<3;++k)slope[k]+=f*g1[k]*m.u1[k][i];
    }
    tensor_gradient(rate,slope,m.rs,slope0/m.rs[0],m.ds,out);
}
HD inline void residual(const Model& m,const double *x,const double *prev,double n,double dt,bool fixed,double *r,double *s){
    double p[2]={x[0]-x[2],x[1]-x[3]},vel[2]={(x[2]-prev[2])/dt,(x[3]-prev[3])/dt};
    double e[2],tau[2],v[2];elastic(m,m.eq,x,e);elastic(m,m.ov,p,tau);dissipation(m,vel,p,v);
    for(int i=0;i<2;++i)s[i]=e[i]+tau[i];
    r[0]=fixed?3000.*(x[0]-n):s[0]-n;r[1]=s[1];r[2]=v[0]-tau[0];r[3]=v[1]-tau[1];
}
HD inline double norm4(const double* x){
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3];
}
HD inline bool linear4(double a[4][5],double* result){
    for(int k=0;k<4;++k){
        int p=k;for(int j=k+1;j<4;++j) if(fabs(a[j][k])>fabs(a[p][k]))p=j;
        if(!(fabs(a[p][k])>0))return false;
        for(int j=k;j<5;++j){double tmp=a[p][j];a[p][j]=a[k][j];a[k][j]=tmp;}
        double diag=a[k][k];for(int j=k;j<5;++j)a[k][j]/=diag;
        for(int i=0;i<4;++i) if(i!=k){
            double f=a[i][k];for(int j=k;j<5;++j)a[i][j]-=f*a[k][j];
        }
    }
    for(int i=0;i<4;++i)result[i]=a[i][4];return true;
}
HD inline double solve(const Model& m,const double *prev,double n,double dt,bool fixed,double *x,double *stress){
    for(int i=0;i<4;++i)x[i]=prev[i];
    double r[4],s[2];residual(m,x,prev,n,dt,fixed,r,s);
    for(int k=0;k<=30;++k){
        double error=0.;for(int i=0;i<4;++i)error=fmax(error,fabs(r[i]));
        if(error<=1e-8 || k==30){*stress=s[0];return error;}
        double mat[4][5];
        for(int j=0;j<4;++j){
            const double dx=j<2?1e-8:dt*1e-5;
            double xp[4],xm[4],rp[4],rm[4],dummy[2];
            for(int i=0;i<4;++i){xp[i]=x[i];xm[i]=x[i];}
            xp[j]+=dx;xm[j]-=dx;
            residual(m,xp,prev,n,dt,fixed,rp,dummy);residual(m,xm,prev,n,dt,fixed,rm,dummy);
            for(int i=0;i<4;++i)mat[i][j]=(rp[i]-rm[i])/(2.*dx);
        }
        for(int i=0;i<4;++i)mat[i][4]=-r[i];
        double step[4];if(!linear4(mat,step))return INFINITY;
        double alpha=1.,rr[4],xx[4],old=norm4(r);
        for(int j=0;j<=16;++j){
            for(int i=0;i<4;++i)xx[i]=x[i]+alpha*step[i];
            residual(m,xx,prev,n,dt,fixed,rr,s);
            if(norm4(rr)<=(1.-1e-4*alpha)*old || j==16)break;
            alpha*=.5;
        }
        for(int i=0;i<4;++i){r[i]=rr[i];x[i]=xx[i];}
    }
    return INFINITY;
}
HD inline void run_one(const Model& m,const double* normal,int stride,int count,double dt,double* state,double* out){
    for(int t=0;t<count;++t){
        double n=normal[t*stride],free[4],next[4],sf=0,sc=0,err=0;bool active=false;
        bool virgin=state[0]==0 && state[1]==0 && state[2]==0 && state[3]==0;
        if(virgin)for(int j=0;j<4;++j)free[j]=0;
        else err=solve(m,state,0.,dt,false,free,&sf);
        if(n<=free[0]){
            const double contact_error=solve(m,state,n,dt,true,next,&sc);
            err=fmax(err,contact_error);active=sc<0;
        }
        for(int j=0;j<4;++j){state[j]=active?next[j]:free[j];out[t*stride*6+j]=state[j];}
        out[t*stride*6+4]=active?sc:0.;
        out[t*stride*6+5]=err;
    }
}
extern "C" int model_doubles(){return sizeof(Model)/sizeof(double);}
extern "C" void cpu_run(const Model *m,const double *n,int K,int T,double dt,double *out){
    #pragma omp parallel for schedule(dynamic)
    for(int k=0;k<K;++k){double state[4]={};run_one(*m,n+k,K,T,dt,state,out+k*6);}
}
#ifdef __CUDACC__
__global__ void advance(const Model *m,const double *n,int K,int T,double dt,double *state,double *out){
    int k=threadIdx.x+blockDim.x*blockIdx.x;
    if(k<K){double x[4];for(int i=0;i<4;++i)x[i]=state[4*k+i];
        run_one(*m,n+k,K,T,dt,x,out+k*6);for(int i=0;i<4;++i)state[4*k+i]=x[i];}
}
#define TRY(expr) if((status=(expr))!=cudaSuccess)goto done
// Synchronized phase timers include input and output copies separately from hot kernels.
extern "C" int gpu_run(const Model *m,const double *n,int K,int T,double dt,double *out,double *timings){
    Model *dm=nullptr;double *dn=nullptr,*state=nullptr,*dout=nullptr;
    cudaEvent_t a{},b{},c{},d{};cudaError_t status=cudaSuccess;float ms;
    TRY(cudaEventCreate(&a));TRY(cudaEventCreate(&b));TRY(cudaEventCreate(&c));TRY(cudaEventCreate(&d));
    TRY(cudaEventRecord(a));TRY(cudaMalloc(&dm,sizeof(Model)));TRY(cudaMalloc(&dn,sizeof(double)*K*T));
    TRY(cudaMalloc(&state,sizeof(double)*K*4));TRY(cudaMalloc(&dout,sizeof(double)*K*T*6));
    TRY(cudaMemset(state,0,sizeof(double)*K*4));TRY(cudaMemcpy(dm,m,sizeof(Model),cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(dn,n,sizeof(double)*K*T,cudaMemcpyHostToDevice));TRY(cudaEventRecord(b));
    for(int t=0;t<T;t+=64){
        advance<<<(K+3)/4,4>>>(dm,dn+t*K,K,std::min(64,T-t),dt,state,dout+t*K*6);
        TRY(cudaGetLastError());
    }
    TRY(cudaEventRecord(c));TRY(cudaMemcpy(out,dout,sizeof(double)*K*T*6,cudaMemcpyDeviceToHost));
    TRY(cudaEventRecord(d));TRY(cudaEventSynchronize(d));
    TRY(cudaEventElapsedTime(&ms,a,b));timings[0]=ms*.001;
    TRY(cudaEventElapsedTime(&ms,b,c));timings[1]=ms*.001;
    TRY(cudaEventElapsedTime(&ms,c,d));timings[2]=ms*.001;
done:
    cudaFree(dm);cudaFree(dn);cudaFree(state);cudaFree(dout);
    cudaEventDestroy(a);cudaEventDestroy(b);cudaEventDestroy(c);cudaEventDestroy(d);
    return int(status);
}
#endif
