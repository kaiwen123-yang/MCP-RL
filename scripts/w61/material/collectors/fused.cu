// Appended to the pinned M1 CUDA source by build.py after turning its two
// original kernels into device procedures. grid.y identifies an independent
// collector and never changes a cell's Newton loop, masks, or sum order.
struct BatchItem {
    int n;
    size_t in, out, history, failure;
    double gamma;
};
struct BatchIndex { BatchItem items[6]; };

__global__ void batch_columns(const Model* m, BatchIndex ix, int ticks,
                              unsigned char* input, Step* hist, int* failed) {
    const auto e=ix.items[blockIdx.y];
    propagate_columns(m,e.n,ticks,inputs(input+e.in,e.n,ticks),hist+e.history,failed+e.failure);
}
__global__ void batch_finish(BatchIndex ix,int ticks,unsigned char* input,unsigned char* output,
                             const Step* hist,const int* failed) {
    const auto e=ix.items[blockIdx.y];
    finish_packet(e.n,ticks,inputs(input+e.in,e.n,ticks),outputs(output+e.out,e.n,ticks),
                  hist+e.history,failed+e.failure,e.gamma);
}

struct BatchContext {
    Context* original{};
    unsigned char *hi{},*di{},*ho{},*dout{};
    Step* history{};
    int* failure{};
    int max_env{},max_n{},ticks{};
    ~BatchContext() {
        if(hi)cudaFreeHost(hi);
        if(ho)cudaFreeHost(ho);
        if(di)cudaFree(di);
        if(dout)cudaFree(dout);
        if(history)cudaFree(history);
        if(failure)cudaFree(failure);
        delete original;
    }
};
size_t aligned(size_t n) {return (n+7)/8*8;}

extern "C" int w61_create(const double* mod,int count,int cells,int ticks,void** handle) {
    *handle=nullptr;
    if(count<1||count>6||cells<1||ticks<1)return cudaErrorInvalidValue;
    BatchContext* b=new(std::nothrow) BatchContext;
    if(!b)return cudaErrorMemoryAllocation;
    void* original=nullptr;
    int err=w59_gpu_create(mod,&original);
    b->original=static_cast<Context*>(original);
    b->max_env=count;b->max_n=cells;b->ticks=ticks;
    size_t ni=count*aligned(input_bytes(cells,ticks)),no=count*aligned(output_bytes(cells,ticks));
    if(!err)err=cudaMallocHost(&b->hi,ni);
    if(!err)err=cudaMallocHost(&b->ho,no);
    if(!err)err=cudaMalloc(&b->di,ni);
    if(!err)err=cudaMalloc(&b->dout,no);
    if(!err)err=cudaMalloc(&b->history,size_t(count)*ticks*cells*sizeof(Step));
    if(!err)err=cudaMalloc(&b->failure,size_t(count)*cells*sizeof(int));
    if(err){delete b;return err;}
    *handle=b;return cudaSuccess;
}
extern "C" void w61_destroy(void* handle){delete static_cast<BatchContext*>(handle);}
extern "C" int w61_dispatch(void* opaque,int count,const int* nc,const double* gamma,
                            const uintptr_t* in,const uintptr_t* out) {
    BatchContext& b=*static_cast<BatchContext*>(opaque);
    if(count<1||count>b.max_env)return cudaErrorInvalidValue;
    BatchIndex ix{};int max_n=0;size_t ni=0,no=0,nh=0,nf=0;
    for(int i=0;i<count;++i){
        int n=nc[i];
        if(n<1||n>b.max_n)return cudaErrorInvalidValue;
        ix.items[i]={n,ni,no,nh,nf,gamma[i]};
        std::memcpy(b.hi+ni,reinterpret_cast<const void*>(in[i]),input_bytes(n,b.ticks));
        ni+=aligned(input_bytes(n,b.ticks));no+=aligned(output_bytes(n,b.ticks));
        nh+=size_t(n)*b.ticks;nf+=n;max_n=(n>max_n?n:max_n);
    }
    const auto stream=b.original->stream;
    int err=cudaMemcpyAsync(b.di,b.hi,ni,cudaMemcpyHostToDevice,stream);
    if(!err){
        batch_columns<<<dim3(max_n,count),8,0,stream>>>(b.original->model,ix,b.ticks,b.di,b.history,b.failure);
        err=cudaGetLastError();
    }
    if(!err){
        batch_finish<<<dim3(1,count),128,0,stream>>>(ix,b.ticks,b.di,b.dout,b.history,b.failure);
        err=cudaGetLastError();
    }
    if(!err)err=cudaMemcpyAsync(b.ho,b.dout,no,cudaMemcpyDeviceToHost,stream);
    if(!err)err=cudaStreamSynchronize(stream);
    if(!err)for(int i=0;i<count;++i)
        std::memcpy(reinterpret_cast<void*>(out[i]),b.ho+ix.items[i].out,output_bytes(nc[i],b.ticks));
    return err;
}
