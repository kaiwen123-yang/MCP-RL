"""Direct WALL timing of single-invocation material calculations on r55 histories."""
import ctypes as ct
import hashlib
import json
import multiprocessing as mp
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from benchmark_s1_w56_material import DT, LIB, OUT, RAW, WEIGHTS, cpu, intervals, pack, weights


def utc(ns):
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat(timespec='microseconds')


def clocks():
    return time.time_ns(), time.perf_counter_ns(), time.process_time_ns()


def physical_jobs():
    rows = {}
    for name in ('a','b'):
        path, = RAW.glob(f'job-*-w55-f-{name}-r55-eval*-E600-attempt01/live.json')
        live = json.loads(path.read_text())
        pid = live['pid']
        cmd = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        if path.parent.name.encode() not in cmd:
            raise ValueError(f'Original physics PID no longer matches {path.parent.name}')
        stat = (Path('/proc')/str(pid)/'stat').read_text().split(') ')[1].split()
        rows[name] = dict(attempt=path.parent.name,pid=pid,proc_ticks_user_plus_kernel=int(stat[11])+int(stat[12]),
                          physical_s=live['physical_time_s'],last_live_utc=utc(round(live['unix_s']*1e9)))
    return rows


def device_call(model,boundary):
    n,k = boundary.shape
    lib = ct.CDLL(str(LIB/'column_gpu.so'))
    fn = lib.batched_gpu
    fn.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_void_p,ct.c_void_p]
    fn.restype = ct.c_int
    init_before = torch.cuda.is_initialized()
    ctx_t0=time.perf_counter_ns()
    torch.cuda.init()
    stream = torch.cuda.current_stream()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ctx_t1=time.perf_counter_ns()
    events[0].record(stream)
    m = torch.from_numpy(model).to('cuda')
    x = torch.from_numpy(boundary).to('cuda')
    y = torch.empty((n,k,6),device='cuda',dtype=torch.float64)
    events[1].record(stream)
    # Exactly one batched constitutive launch in this invocation.
    ret = fn(m.data_ptr(),x.data_ptr(),k,n,DT,y.data_ptr(),stream.cuda_stream)
    if ret:
        raise RuntimeError(f'CUDA launch error {ret}')
    events[2].record(stream)
    answer = y.cpu().numpy()
    events[3].record(stream)
    sync_t0 = time.perf_counter_ns()
    events[3].synchronize()
    stream.synchronize()
    sync_t1 = time.perf_counter_ns()
    metrics = dict(cuda_initialized_before_call=init_before,material_kernel_launches=1,
                   context_and_event_creation_wall_s=(ctx_t1-ctx_t0)/1e9,
                   h2d_and_output_allocation_event_s=events[0].elapsed_time(events[1])/1000.,
                   kernel_event_s=events[1].elapsed_time(events[2])/1000.,
                   d2h_result_event_s=events[2].elapsed_time(events[3])/1000.,
                   terminal_sync_wall_s=(sync_t1-sync_t0)/1e9,
                   final_tensor_live_bytes=torch.cuda.memory_allocated(),peak_torch_bytes=torch.cuda.max_memory_allocated())
    return answer,metrics


def host_call(model,boundary):
    n,k=boundary.shape
    lib=ct.CDLL(str(LIB/'column_cpu.so'))
    fn=lib.batched_cpu
    dp=ct.POINTER(ct.c_double)
    fn.argtypes=[dp,dp,ct.c_int,ct.c_int,ct.c_double,dp];fn.restype=None
    answer=np.empty((n,k,6))
    fn(model.ctypes.data_as(dp),boundary.ctypes.data_as(dp),k,n,DT,answer.ctypes.data_as(dp))
    return answer,dict(native_calls=1,openmp_threads=os.environ['OMP_NUM_THREADS'])


def worker(device,model,x,pipe):
    outputs={};records={}
    for label in ('cold_first_call','warm_second_call'):
        s,m,c=clocks()
        out,metrics=(device_call if device=='gpu' else host_call)(model,x)
        e,m1,c1=clocks()
        records[label]=dict(start_utc=utc(s),end_utc=utc(e),start_unix_ns=s,end_unix_ns=e,
                            wall_s=(m1-m)/1e9,process_cpu_s=(c1-c)/1e9,**metrics)
        outputs[label]=out
    # Independent legacy CPU reference is evaluated only after both clocks stop.
    reference,_=cpu(model,x,'cpu_run')
    for label,out in outputs.items():
        diff=np.max(np.abs(out[:,:,:5]-reference[:,:,:5]),axis=(0,1))
        res=float(np.max(out[:,:,5]))
        records[label].update(max_error_e11_e22_q11_q22_stress_MPa=diff.tolist(),
                              max_constitutive_residual_MPa=res)
        if not np.isfinite(out).all() or np.max(diff[:4])>1e-8 or diff[4]>1e-6 or res>1e-7:
            raise ValueError(f'{device} {label} diverged {diff}, {res}')
    pipe.send(dict(pid=os.getpid(),max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**records))
    pipe.close()


def main():
    start=time.time_ns();before=physical_jobs()
    model=pack(weights(WEIGHTS));x,rows,total=intervals()
    original=json.loads((OUT/'cpu-gpu-real-histories.json').read_text())
    if rows!=original['unique_case_key_start_tick'] or total!=original['all_records']:
        raise ValueError('Benchmark input selection differs from W56 recorded run')
    result={}
    ctx=mp.get_context('fork')
    for label,boundary in [('single',np.ascontiguousarray(x[:,[0]])),('batch',x)]:
        row=dict(shape=list(boundary.shape),input_sha256=hashlib.sha256(boundary.tobytes()).hexdigest())
        for device in ('cpu','gpu'):
            # Each width/device receives an independent fresh process with no CUDA context.
            rx,tx=ctx.Pipe(duplex=False)
            p=ctx.Process(target=worker,args=(device,model,boundary,tx))
            p.start();tx.close()
            row[device]=rx.recv();rx.close();p.join()
            if p.exitcode!=0:
                raise RuntimeError(f'Worker failed {p.exitcode}')
        for temp in ('cold_first_call','warm_second_call'):
            row[temp+'_cpu_over_gpu']=row['cpu'][temp]['wall_s']/row['gpu'][temp]['wall_s']
        result[label]=row
    after=physical_jobs();end=time.time_ns()
    output=dict(schema='w56_direct_host_wall_single_dispatch_v1',start_utc=utc(start),end_utc=utc(end),
                device='RTX PRO 6000 Blackwell Server Edition',
                input='Matches previous 811 first-contact windows, 174 distinct r55 A/B E30 histories, 600 steps each. Single uses exactly the first selected path.',
                calls='A fresh process for each width/device. First call includes native library binding, CUDA context initialization as applicable, allocation, H2D, one material kernel, D2H and stream synchronization. Second call repeats the whole invocation with the process and context warm. Host input already materialized; Python imports and history assembly excluded. CPU includes native binding and allocation. Each output validated against original FP64 CPU law.',
                physical_jobs_before=before,physical_jobs_after=after,clock_ticks_per_second=os.sysconf('SC_CLK_TCK'),
                cold_worker_start='POSIX fork of host-only process; CUDA not initialized in parent. No material function called before first timed invocation in child.',
                samples_per_temperature_and_workload=1,previous_results='cpu-gpu-real-histories.json',
                lower_library_cpu_sha256=hashlib.sha256((LIB/'column_cpu.so').read_bytes()).hexdigest(),
                lower_library_gpu_sha256=hashlib.sha256((LIB/'column_gpu.so').read_bytes()).hexdigest(),results=result)
    dest=OUT/'cpu-gpu-direct-single-launch.json';dest.write_text(json.dumps(output,indent=2)+'\n')
    for label,row in result.items():
        print(label,{k:(row['cpu'][k]['wall_s'],row['gpu'][k]['wall_s'],row[k+'_cpu_over_gpu']) for k in ('cold_first_call','warm_second_call')},flush=True)


if __name__=='__main__':main()
