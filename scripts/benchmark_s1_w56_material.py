"""FP64 CPU/GPU free evolution over distinct natural r55 first-contact windows."""
import ctypes as ct
import json
import os
import resource
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from s1_w55_c_data import DT, R12, WEIGHTS, geometry
from s1_w55_c_replay import D, pack, weights

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT.parent/'MCP-RL-w55-f'/'outputs'/R12
OUT=ROOT/'results'/R12/'w56/c'
LIB=ROOT/'work/w56-c'
SAMPLES=600


def intervals():
    paths=[]; sources=[]
    for name in ('a','b'):
        folder,=RAW.glob(f'job-*-w55-f-{name}-r55-eval*-E30-attempt01')
        first={}
        for line in (folder/'material-events.jsonl').open():
            event=json.loads(line)
            key=tuple(event['world_key'])
            if event['type']=='load' and key not in first:
                first[key]=max(0,round(event['time_s']/DT)-11)
        chunks={}
        for key,start in first.items():
            if start+SAMPLES>30/DT:continue
            lo=start//5000;hi=(start+SAMPLES-1)//5000
            for c in range(lo,hi+1):
                if c not in chunks:
                    with np.load(folder/f'material-{c:04}.npz') as z:
                        chunks[c]=z['foot'][:,6:18].reshape(-1,4,3)
            bottom=(chunks[lo] if lo==hi else np.concatenate([chunks[lo],chunks[hi]]))[start%5000:start%5000+SAMPLES]
            if len(bottom)!=SAMPLES:raise ValueError('Incomplete first-contact window')
            norm,_,_=geometry(bottom,np.asarray([key]))
            paths.append(norm[:,0]);sources.append([name,*key,start])
    full=np.stack(paths,axis=1)
    unique,idx=np.unique(full,axis=1,return_index=True)
    return np.ascontiguousarray(unique),[sources[i] for i in idx],len(sources)


def pointer(array):return array.ctypes.data_as(D)


def cpu(model,boundary,version):
    mod=ct.CDLL(str(LIB/'column_cpu.so'))
    fn=getattr(mod,version)
    fn.argtypes=[D,D,ct.c_int,ct.c_int,ct.c_double,D];fn.restype=None
    out=np.empty((*boundary.shape,6))
    start=time.perf_counter();proc=time.process_time()
    fn(pointer(model),pointer(boundary),boundary.shape[1],len(boundary),DT,pointer(out))
    return out,dict(wall_s=time.perf_counter()-start,cpu_s=time.process_time()-proc)


def old_gpu(model,boundary):
    mod=ct.CDLL(str(LIB/'column_gpu.so'))
    fn=mod.gpu_run;fn.argtypes=[D,D,ct.c_int,ct.c_int,ct.c_double,D,D];fn.restype=ct.c_int
    out=np.empty((*boundary.shape,6));phases=np.zeros(3)
    s=time.perf_counter();c=time.process_time()
    status=fn(pointer(model),pointer(boundary),boundary.shape[1],len(boundary),DT,pointer(out),pointer(phases))
    if status:raise RuntimeError(f'W55 CUDA returned {status}')
    return out,dict(wall_s=time.perf_counter()-s,cpu_s=time.process_time()-c,
                    setup_and_h2d_s=phases[0],kernel_s=phases[1],d2h_s=phases[2])


def gpu(model,boundary):
    mod=ct.CDLL(str(LIB/'column_gpu.so'))
    p=ct.c_void_p
    fn=mod.batched_gpu;fn.argtypes=[p,p,ct.c_int,ct.c_int,ct.c_double,p,p];fn.restype=ct.c_int
    start=time.perf_counter();cs=time.process_time()
    events=[torch.cuda.Event(enable_timing=True) for _ in range(6)]
    events[0].record()
    weights_d=torch.from_numpy(model).cuda()
    input_d=torch.from_numpy(boundary).cuda()
    output_d=torch.empty((*boundary.shape,6),dtype=torch.float64,device='cuda')
    events[1].record()
    def launch():
        ret=fn(weights_d.data_ptr(),input_d.data_ptr(),boundary.shape[1],len(boundary),DT,
               output_d.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if ret:raise RuntimeError(f'W56 CUDA returned {ret}')
    launch();events[2].record()
    events[3].record();launch();events[4].record()
    out=output_d.cpu().numpy();events[5].record();events[-1].synchronize()
    elapsed=lambda a,b:events[a].elapsed_time(events[b])*.001
    total=time.perf_counter()-start
    cold=elapsed(1,2);hot=elapsed(3,4)
    return out,dict(wall_s_both_launches=total,cpu_s_both_launches=time.process_time()-cs,
                    h2d_and_allocate_s=elapsed(0,1),first_kernel_s=cold,hot_kernel_s=hot,
                    d2h_s=elapsed(4,5),full_single_launch_wall_s=total-hot,
                    hot_paths_per_s=boundary.shape[1]/hot,peak_torch_bytes=torch.cuda.max_memory_allocated())


def comparison(model,boundary,full):
    reference,old_cpu=cpu(model,boundary,'cpu_run')
    new_cpu,cpu_time=cpu(model,boundary,'batched_cpu')
    gpu_out,new_gpu=gpu(model,boundary)
    legacy_gpu,old_gpu_time=old_gpu(model,boundary)
    errors={}
    for label,out in [('w56_cpu',new_cpu),('w56_gpu',gpu_out),('w55_gpu',legacy_gpu)]:
        diff=np.max(np.abs(out[:,:,:5]-reference[:,:,:5]),axis=(0,1))
        err=float(np.max(out[:,:,5]));errors[label]=dict(state_and_stress_error=diff.tolist(),max_residual_MPa=err)
        if not np.isfinite(out).all() or max(diff[:4])>1e-8 or diff[4]>1e-6 or err>1e-7:
            raise ValueError(f'FP64 history mismatch {label}: {errors[label]}')
    return dict(paths=boundary.shape[1],steps=SAMPLES,old_cpu=old_cpu,new_cpu=cpu_time,
                old_gpu=old_gpu_time,new_gpu=new_gpu,comparisons=errors,
                cpu_speedup=old_cpu['wall_s']/cpu_time['wall_s'],
                gpu_kernel_speedup=old_gpu_time['kernel_s']/new_gpu['hot_kernel_s'],
                cpu_over_gpu_full_wall=cpu_time['wall_s']/new_gpu['full_single_launch_wall_s'])


def main():
    start=time.time();cs=time.process_time()
    proc=int(os.getpid());gpu_mb=[];done=threading.Event()
    def sample():
        while not done.is_set():
            lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
            for line in lines.splitlines():
                k,v=line.split(',')
                if k.strip()==str(proc):gpu_mb.append(int(v))
            done.wait(.3)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    model=pack(weights(WEIGHTS));x,rows,count=intervals()
    prep=time.time()-start
    startup=time.perf_counter();torch.cuda.init();torch.cuda.synchronize();context=time.perf_counter()-startup
    single=comparison(model,np.ascontiguousarray(x[:,[0]]),False)
    batch=comparison(model,x,True)
    done.set();monitor.join()
    summary=dict(source='r55 A/B E30 first loading of each world-cell, ten measured steps before onset followed by complete fixed 120 ms boundary sequence',
                 frozen_model='B_random1000--ideal-potentials',all_records=count,distinct_boundary_histories=len(rows),
                 unique_case_key_start_tick=rows,baseline='W55 central FD, four-thread CUDA blocks; fixed model FP64 with unchanged residual, Newton and backtracking thresholds',
                 new_method='Forward FD, original units; 8-lane CUDA subgroup per real history,128 threads/block; persistent device tensors over cold/hot kernels',
                 single=single,batch=batch,device=torch.cuda.get_device_name(),cuda_context_s=context,
                 cuda_process_peak_sampled_MiB=max(gpu_mb,default=None),sample_period_s=.3,
                 prep_s=prep,max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 start_unix=start,end_unix=time.time(),cpu_s=time.process_time()-cs,
                 math_threads=os.getenv('OMP_NUM_THREADS'),build=(LIB/'build.log').read_text(),
                 scope='Independent prescribed-boundary constitutive histories; excludes force-MPC, WBC and online controller latency.')
    path=OUT/'cpu-gpu-real-histories.json';path.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('distinct_boundary_histories','single','batch','start_unix','end_unix')}),flush=True)


if __name__=='__main__':main()
