// Independent real measurement histories seeded by a published M0/M1 state.
#include "s1_w56_column.cu"

HD inline void forecast(const Model& m, const double* initial, const double* boundary,
                        const int* length, int paths, int path, double dt, bool free_only,
                        double* result) {
    double x[4]; for (int j=0;j<4;++j) x[j]=initial[path*4+j];
    bool settled=(x[0]==0. && x[1]==0. && x[2]==0. && x[3]==0.);
    double err=0.,peak=0.;
    for (int t=0;t<length[path];++t) {
        const double n=free_only ? INFINITY : boundary[t*paths+path];
        if (settled && n==INFINITY) {
            if (free_only) break;
            continue;
        }
        double next[4],loaded[4],s_free=0.,s_loaded=0.;
        if (settled) for (int j=0;j<4;++j) next[j]=x[j];
        else err=fmax(err,material_solve(m,x,0.,dt,false,next,&s_free));
        bool press=false;
        if (n<=next[0]) {
            err=fmax(err,material_solve(m,x,n,dt,true,loaded,&s_loaded));
            press=s_loaded<0.;
        }
        bool still=!press;
        for (int j=0;j<4;++j) {
            double value=press?loaded[j]:next[j];still=still && value==x[j];x[j]=value;
        }
        peak=fmax(peak,fmax(fabs(x[0]),fabs(x[1])));
        settled=still;
    }
    if (material_lane()==0) {
        for(int j=0;j<4;++j)result[path*6+j]=x[j];
        result[path*6+4]=err;result[path*6+5]=peak;
    }
}

extern "C" void w57_forecast_cpu(const Model*m,const double* initial,const double* boundary,
        const int* length,int paths,double dt,int free_only,double* result) {
    for(int p=0;p<paths;++p) forecast(*m,initial,boundary,length,paths,p,dt,free_only,result);
}

#ifdef __CUDACC__
__global__ void forecast_groups(const Model*m,const double* initial,const double* boundary,
        const int* length,int paths,double dt,int free_only,double* result) {
    int p=(threadIdx.x+blockIdx.x*blockDim.x)/8;
    if(p<paths) forecast(*m,initial,boundary,length,paths,p,dt,free_only,result);
}

extern "C" int w57_forecast_gpu(const Model*m,const double* initial,const double* boundary,
        const int* length,int paths,double dt,int free_only,double* result,cudaStream_t stream) {
    forecast_groups<<<(8*paths+127)/128,128,0,stream>>>(m,initial,boundary,length,paths,dt,free_only,result);
    return int(cudaGetLastError());
}
#endif
