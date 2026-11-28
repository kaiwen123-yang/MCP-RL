// Nominal W59 M1 propagation. CPU force assimilation follows after this synchronous ABI.
// The CUDA paths use the frozen W55 constitutive equations and central FD Newton semantics.
#include "s1_w55_c_column.cu"
#include <climits>
#include <cstdint>
#include <cstring>
#include <new>

namespace {

struct Inputs {
    double *dt, *normal, *area, *initial;
    int32_t *owner;
    uint8_t *settled, *loaded;
};

struct Outputs {
    double *state, *before_last, *force, *stats;
    int *bad;
    uint8_t *settled, *loaded, *before_loaded, *trace;
};

template<class T>
T* take(unsigned char*& bytes, size_t count) {
    auto* begin = reinterpret_cast<T*>(bytes);
    bytes += sizeof(T) * count;
    return begin;
}

Inputs inputs(unsigned char* p, int n, int ticks) {
    const size_t cells = size_t(n) * ticks;
    return {take<double>(p,ticks), take<double>(p,cells), take<double>(p,cells),
            take<double>(p,6*n), take<int32_t>(p,cells), take<uint8_t>(p,n), take<uint8_t>(p,n)};
}

Outputs outputs(unsigned char* p, int n, int ticks) {
    return {take<double>(p,6*n), take<double>(p,6*n), take<double>(p,4*ticks),
            take<double>(p,4*ticks), take<int>(p,1), take<uint8_t>(p,n),
            take<uint8_t>(p,n), take<uint8_t>(p,n), take<uint8_t>(p,size_t(ticks)*n)};
}

size_t input_bytes(size_t n, size_t t) {
    return sizeof(double)*(t+2*n*t+6*n) + sizeof(int32_t)*n*t + 2*n;
}
size_t output_bytes(size_t n, size_t t) {
    return sizeof(double)*(12*n+8*t) + sizeof(int) + (3+t)*n;
}

struct Step {
    double state[6], err, rate;
    uint8_t settled, loaded;
};

// All eight lanes of a column follow the same Newton/line-search decisions.
// One lane evaluates each signed FD sample; coefficients use the original +/- order.
__device__ double centered_solve(const Model& m, const double* prev, double normal,
                                 double dt, bool fixed, double* x) {
    const int lane = threadIdx.x & 7;
    const unsigned group = 0xffu << (threadIdx.x & 24);
    for (int j=0; j<4; ++j) x[j]=prev[j];
    double r[4], stress[2];
    residual(m,x,prev,normal,dt,fixed,r,stress);
    for (int it=0; it<=30; ++it) {
        double error=0.;
        for (int j=0; j<4; ++j) error=fmax(error,fabs(r[j]));
        if (error<=1e-8 || it==30) return error;

        const int component=lane%4;
        const double delta=component<2 ? 1e-8 : dt*1e-5;
        double perturbed[4], shifted[4], unused_stress[2], mat[4][5];
        for (int j=0; j<4; ++j) perturbed[j]=x[j];
        if (lane<4) perturbed[component]+=delta;
        else perturbed[component]-=delta;
        residual(m,perturbed,prev,normal,dt,fixed,shifted,unused_stress);
        for (int j=0; j<4; ++j) {
            const double dx=j<2 ? 1e-8 : dt*1e-5;
            for (int k=0; k<4; ++k) {
                const double plus=__shfl_sync(group,shifted[k],j,8);
                const double minus=__shfl_sync(group,shifted[k],j+4,8);
                mat[k][j]=(plus-minus)/(2.*dx);
            }
        }
        for (int k=0; k<4; ++k) mat[k][4]=-r[k];
        double step[4];
        if (!linear4(mat,step)) return INFINITY;

        double alpha=1., rr[4], xx[4], old=norm4(r);
        for (int j=0; j<=16; ++j) {
            for (int k=0; k<4; ++k) xx[k]=x[k]+alpha*step[k];
            residual(m,xx,prev,normal,dt,fixed,rr,stress);
            if (norm4(rr)<=(1.-1e-4*alpha)*old || j==16) break;
            alpha*=.5;
        }
        for (int k=0; k<4; ++k) { r[k]=rr[k]; x[k]=xx[k]; }
    }
    return INFINITY;
}

__device__ double mixed_gpu(const Model& m,const double* prev,double normal,double dt,
                            bool fixed,double* row) {
    double err=centered_solve(m,prev,normal,dt,fixed,row);
    double eq[2], over[2], p[2]={row[0]-row[2],row[1]-row[3]};
    elastic(m,m.eq,row,eq);
    elastic(m,m.ov,p,over);
    row[4]=eq[0]+over[0];
    row[5]=eq[1]+over[1];
    return err;
}

__device__ bool acceptable(double err,const double* row) {
    if (!isfinite(err) || err>1e-7) return false;
    for (int i=0;i<6;++i) if (!isfinite(row[i])) return false;
    return true;
}

__global__ void propagate_columns(const Model* model, int n, int ticks, Inputs in,
                                  Step* history, int* failed) {
    const int i=(blockIdx.x*blockDim.x+threadIdx.x)/8;
    if (i>=n) return;
    const bool writer=(threadIdx.x&7)==0;
    if (writer) failed[i]=INT_MAX;
    double x[6];
    for (int c=0;c<6;++c) x[c]=in.initial[i*6+c];
    bool settled=in.settled[i]!=0;
    for (int k=0;k<ticks;++k) {
        const int j=k*n+i;
        double free[6], contact[6], err=0.;
        for (int c=0;c<6;++c) free[c]=x[c];
        if (!settled) {
            err=mixed_gpu(*model,x,0.,in.dt[k],false,free);
            if (!acceptable(err,free)) {
                if (writer) { failed[i]=j; history[j].err=err; }
                return;
            }
        }
        bool loaded=false;
        if (in.normal[j]<=free[0]) {
            const double rc=mixed_gpu(*model,x,in.normal[j],in.dt[k],true,contact);
            if (!acceptable(rc,contact)) {
                if (writer) { failed[i]=j; history[j].err=rc; }
                return;
            }
            err=fmax(err,rc);
            loaded=contact[4]<0.;
        }
        const double* next=loaded ? contact : free;
        settled=!loaded;
        for (int c=0;c<4;++c) settled=settled && x[c]==next[c];
        if (writer) {
            Step& step=history[j];
            step.rate=fmax(fabs(next[0]-x[0])/in.dt[k],fabs(next[1]-x[1])/in.dt[k]);
            step.err=err;
            step.settled=settled;
            step.loaded=loaded;
            for (int c=0;c<6;++c) step.state[c]=next[c];
        }
        for (int c=0;c<6;++c) x[c]=next[c];
    }
}

__global__ void finish_packet(int n,int ticks,Inputs in,Outputs out,const Step* history,
                              const int* failures,double gamma) {
    __shared__ int bad;
    if (threadIdx.x==0) {
        int first=INT_MAX;
        for (int i=0;i<n;++i) first=min(first,failures[i]);
        bad=first;
        *out.bad=(first==INT_MAX) ? -1 : first;
    }
    __syncthreads();
    const int failed_tick=(bad==INT_MAX) ? ticks : bad/n;
    // Serial addition in exactly the same world-ID order as the native force accumulator.
    for (int k=threadIdx.x;k<ticks;k+=blockDim.x) {
        double f[4]={}, peak[4]={};
        const int end=(k<failed_tick) ? n : (k==failed_tick ? bad%n : 0);
        for (int i=0;i<end;++i) {
            const int j=k*n+i;
            const Step& step=history[j];
            if (step.loaded) f[in.owner[j]]-=step.state[4]*1e6*gamma*in.area[j];
            peak[0]=fmax(peak[0],fmax(fabs(step.state[0]),fabs(step.state[1])));
            peak[1]=fmax(peak[1],fmax(fabs(step.state[4]),fabs(step.state[5])));
            peak[2]=fmax(peak[2],step.rate);
            peak[3]=fmax(peak[3],step.err);
        }
        if (k==failed_tick) peak[3]=history[bad].err;
        for (int c=0;c<4;++c) {
            out.force[4*k+c]=f[c];
            out.stats[4*k+c]=peak[c];
        }
    }
    for(int i=threadIdx.x;i<n;i+=blockDim.x) {
        const int last_tick=bad==INT_MAX ? ticks-1 : failed_tick-1+(i<bad%n);
        const int previous=min(last_tick,ticks-2);
        const double* final=last_tick<0 ? in.initial+6*i : history[last_tick*n+i].state;
        const double* prior=previous<0 ? in.initial+6*i : history[previous*n+i].state;
        for (int c=0;c<6;++c) {
            out.state[6*i+c]=final[c];
            out.before_last[6*i+c]=prior[c];
        }
        out.loaded[i]=last_tick<0 ? in.loaded[i] : history[last_tick*n+i].loaded;
        out.settled[i]=last_tick<0 ? in.settled[i] : history[last_tick*n+i].settled;
        out.before_loaded[i]=previous<0 ? in.loaded[i] : history[previous*n+i].loaded;
        for (int k=0;k<failed_tick;++k) out.trace[k*n+i]=history[k*n+i].loaded;
    }
}

struct Context {
    cudaStream_t stream{};
    Model* model{};
    int cells{},ticks{};
    unsigned char *din{},*hin{},*dout{},*hout{};
    Step* history{};
    int* failed{};

    void clear() {
        if (din) cudaFree(din);
        if (dout) cudaFree(dout);
        if (hin) cudaFreeHost(hin);
        if (hout) cudaFreeHost(hout);
        if (history) cudaFree(history);
        if (failed) cudaFree(failed);
        din=dout=hin=hout=nullptr;history=nullptr;failed=nullptr;cells=ticks=0;
    }
    ~Context() {
        clear();
        if (model) cudaFree(model);
        if (stream) cudaStreamDestroy(stream);
    }
    cudaError_t reserve(int n,int t) {
        if (cells>=n && ticks>=t) return cudaSuccess;
        const int new_cells=std::max(n,cells*2), new_ticks=std::max(t,ticks);
        clear();
        const size_t ni=input_bytes(new_cells,new_ticks),no=output_bytes(new_cells,new_ticks);
        cudaError_t e=cudaMalloc(&din,ni);
        if (e==cudaSuccess) e=cudaMalloc(&dout,no);
        if (e==cudaSuccess) e=cudaMallocHost(&hin,ni);
        if (e==cudaSuccess) e=cudaMallocHost(&hout,no);
        if (e==cudaSuccess) e=cudaMalloc(&history,sizeof(Step)*size_t(new_cells)*new_ticks);
        if (e==cudaSuccess) e=cudaMalloc(&failed,sizeof(int)*new_cells);
        if (e!=cudaSuccess) { clear(); return e; }
        cells=new_cells;ticks=new_ticks;
        return cudaSuccess;
    }
};

} // namespace

extern "C" int w59_gpu_create(const double* weights,void** handle) {
    *handle=nullptr;
    Context* ctx=new(std::nothrow) Context;
    if (!ctx) return cudaErrorMemoryAllocation;
    cudaError_t e=cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking);
    if (e==cudaSuccess) e=cudaMalloc(&ctx->model,sizeof(Model));
    if (e==cudaSuccess) e=cudaMemcpyAsync(ctx->model,weights,sizeof(Model),cudaMemcpyHostToDevice,ctx->stream);
    if (e==cudaSuccess) e=cudaStreamSynchronize(ctx->stream);
    if (e!=cudaSuccess) { delete ctx; return e; }
    *handle=ctx;
    return cudaSuccess;
}

extern "C" void w59_gpu_destroy(void* ctx) { delete static_cast<Context*>(ctx); }
extern "C" const char* w59_gpu_error(int code) { return cudaGetErrorString(cudaError_t(code)); }

// Contiguous host arrays use the same tick-major layout and registered ID order as w57_packet.
// Returns a CUDA runtime status. bad_index is separately the frozen solver's first failing row.
extern "C" int w59_gpu_propagate(void* handle,int n,int ticks,const double* dt,
        const double* normal,const double* area,const int32_t* owner,double gamma,
        double* state,uint8_t* settled,uint8_t* loaded,double* force,double* stats,
        uint8_t* trace,double* before_last,uint8_t* before_loaded,int* bad_index) {
    Context& ctx=*static_cast<Context*>(handle);
    cudaError_t e=ctx.reserve(n,ticks);
    if (e!=cudaSuccess) return e;
    Inputs h=inputs(ctx.hin,n,ticks),d=inputs(ctx.din,n,ticks);
    Outputs o=outputs(ctx.hout,n,ticks),dest=outputs(ctx.dout,n,ticks);
    const size_t cells=size_t(n)*ticks;
    std::memcpy(h.dt,dt,ticks*sizeof(double));
    std::memcpy(h.normal,normal,cells*sizeof(double));
    std::memcpy(h.area,area,cells*sizeof(double));
    std::memcpy(h.owner,owner,cells*sizeof(int32_t));
    std::memcpy(h.initial,state,6*n*sizeof(double));
    std::memcpy(h.settled,settled,n);std::memcpy(h.loaded,loaded,n);
    e=cudaMemcpyAsync(ctx.din,ctx.hin,input_bytes(n,ticks),cudaMemcpyHostToDevice,ctx.stream);
    if (e==cudaSuccess) {
        // Independent recorded packets have tens to hundreds of active columns.
        // The alternative gives each eight-lane column its own block to expose more SMs.
#ifdef W59_ONE_COLUMN_BLOCK
        constexpr int threads=8;
        const int blocks=n;
#else
        constexpr int threads=128;
        const int blocks=(8*n+(threads-1))/threads;
#endif
        propagate_columns<<<blocks,threads,0,ctx.stream>>>(
            ctx.model,n,ticks,d,ctx.history,ctx.failed);
        e=cudaGetLastError();
    }
    if (e==cudaSuccess) {
        finish_packet<<<1,128,0,ctx.stream>>>(n,ticks,d,dest,ctx.history,ctx.failed,gamma);
        e=cudaGetLastError();
    }
    if (e==cudaSuccess) e=cudaMemcpyAsync(ctx.hout,ctx.dout,output_bytes(n,ticks),cudaMemcpyDeviceToHost,ctx.stream);
    if (e==cudaSuccess) e=cudaStreamSynchronize(ctx.stream);
    if (e!=cudaSuccess) return e;
    *bad_index=*o.bad;
    std::memcpy(state,o.state,6*n*sizeof(double));
    std::memcpy(before_last,o.before_last,6*n*sizeof(double));
    std::memcpy(loaded,o.loaded,n);std::memcpy(settled,o.settled,n);
    std::memcpy(before_loaded,o.before_loaded,n);
    std::memcpy(force,o.force,4*ticks*sizeof(double));
    std::memcpy(stats,o.stats,4*ticks*sizeof(double));
    size_t valid=*bad_index<0 ? cells : size_t(*bad_index/n)*n;
    std::memcpy(trace,o.trace,valid);
    return cudaSuccess;
}
