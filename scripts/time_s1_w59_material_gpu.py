"""Time the existing CUDA forecast and the actual frozen M1 call on registered development packets."""
import ctypes as ct
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from s1_w57_material import W57MaterialObserver

out=ROOT/'work/w59/gpu-speed-20261127'
out.mkdir(parents=True,exist_ok=True)
reg=json.loads((ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/action/registry-v2.json').read_text())
pars=json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
def getfn(file,symbol,nargs):
    fn=getattr(ct.CDLL(str(ROOT/'work/w57-m'/file)),symbol)
    ptr=ct.c_void_p
    fn.argtypes=[ptr,ptr,ptr,ptr,ct.c_int,ct.c_double,ct.c_int,ptr]+([ptr] if nargs==9 else [])
    fn.restype=ct.c_int if nargs==9 else None
    return fn
host=getfn('forecast_cpu.so','w57_forecast_cpu',8)
dev=getfn('forecast_gpu.so','w57_forecast_gpu',9)
results=[]
for source in reg['states']['development']:
    blob=Path(source['snapshot']).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=source['sha256']:raise ValueError('Registered snapshot changed')
    snap=pickle.loads(gzip.decompress(blob))
    obs=W57MaterialObserver(pars,ROOT/'work/assets/B_random1000--ideal-potentials.npz',mode='M1')
    vars(obs.ground).update(snap['belief']);vars(obs).update(snap['observer'])
    native=obs.native;times=[];captured=[]
    class Capture:
        def w57_packet(self,*args):
            for unused in range(6):
                call_args=list(args)
                for j in (8,9,10,11,12,13,16):
                    call_args[j]=args[j].copy()
                start=time.perf_counter();bad=native.w57_packet(*call_args);times.append(time.perf_counter()-start)
                if bad>=0:raise ValueError('Production solver rejected this saved window')
            captured.append((args[0].copy(),args[1],np.ascontiguousarray(args[8][:,:4]),args[4].copy(),float(np.mean(args[3]))))
            return native.w57_packet(*args)
    obs.native=Capture()
    last=obs.packet[-1]
    obs.observe(float(last[1]),last[2],last[3])
    rec=dict(id=source['snapshot_id'],stratum=source['stratum'],columns=0)
    if captured:
        weights,n,initial,boundary,dt=captured[0]
        length=np.full(n,10,np.int32)
        arrays=[np.ascontiguousarray(x) for x in (weights,initial,boundary,length)]
        cpu_y=np.empty((n,6));gpu_y=np.empty_like(cpu_y)
        tensors=[torch.from_numpy(a) for a in arrays];bufs=[t.cuda() for t in tensors]
        output=torch.empty((n,6),device='cuda',dtype=torch.float64);dest=torch.from_numpy(gpu_y)
        clock=torch.cuda.current_stream().cuda_stream
        cpu=[];gpu=[];kern=[]
        for repeat in range(6):
            tic=time.perf_counter()
            host(*(a.ctypes.data for a in arrays),n,dt,0,cpu_y.ctypes.data)
            cpu.append(time.perf_counter()-tic)
            ev1,ev2=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            tic=time.perf_counter()
            for dst,src in zip(bufs[1:],tensors[1:]):dst.copy_(src)
            ev1.record()
            status=dev(*(a.data_ptr() for a in bufs),n,dt,0,output.data_ptr(),clock)
            if status:raise RuntimeError(status)
            ev2.record();dest.copy_(output);torch.cuda.synchronize()
            gpu.append(time.perf_counter()-tic);kern.append(ev1.elapsed_time(ev2))
        if not np.isfinite(gpu_y).all() or not np.isfinite(cpu_y).all():raise FloatingPointError('Nonfinite forecast')
        rec.update(columns=n,production_M1_ms=1000*median(times[1:]),forecast_CPU_ms=1000*median(cpu[1:]),
                   forecast_GPU_ms=1000*median(gpu[1:]),forecast_GPU_kernel_ms=median(kern[1:]),
                   max_abs_forecast_e_q_error=float(np.max(np.abs(gpu_y[:,:4]-cpu_y[:,:4]))))
    results.append(rec)
report=dict(source='12 registered W59 B3 development legal-request snapshots, one real 10x0.2 ms observer window each. Empty-material windows stay in the denominator. Timings are warm medians over five calls.',
    scope='The existing CPU/CUDA forecast pair gets identical initial e/q, observed position histories and constant mean physical dt. It returns state, peak strain and residual. Production CPU additionally uses per-tick measured dt, load trace, forces and the M1 assimilation correction. The CUDA operator does not implement that full ABI.',
    GPU_timing='Warm context, resident weights and preallocated output. Includes copying packet data to device and results to CPU. Python state construction excluded from both sides.',
    run_device=torch.cuda.get_device_name(),results=results,new_physical_steps=0,active_controller_switched=False)
(out/'material-packets.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(results),flush=True)
