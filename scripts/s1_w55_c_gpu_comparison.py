"""Measure a whole new candidate history workload on both implementations."""
import json
import time
import numpy as np
from s1_w55_c_data import RAW,COMPACT,DT,write
from s1_w55_c_replay import run,weights


def main():
    name='F51'
    w=weights(RAW/'W55-short-load-ideal-potentials.npz')
    host=json.loads((COMPACT/f'candidate-{name}.json').read_text())
    timing=host['execution'][0]
    with np.load(RAW/f'{name}-histories.npz') as data,np.load(RAW/f'candidate-{name}-model.npz') as cpu:
        measured,device=run(w,data['normal'][:,cpu['unique']],DT,'gpu')
        errors=np.max(np.abs(measured[:,:,:5]-cpu['fine'][:,:,:5]),axis=(0,1))
    if max(errors[:4])>1e-9 or errors[4]>1e-6:raise ValueError(f'CPU/CUDA free history mismatch {errors}')
    write(COMPACT/'cpu-gpu-comparison.json',dict(
        episode=name,candidate=True,same_complete_boundaries=True,cpu_single_math_thread=True,
        cpu=timing,gpu=device,host_over_gpu_wall_speedup=timing['wall_s']/device['wall_s'],
        max_state_e11_e22_q11_q22_stress_error=errors.tolist(),
        scope='Offline archived constraints, reset and advance each complete column. CPU and GPU receive identical FP64 model coefficients and full prescribed timelines. GPU wall includes context/transfer and kernel. AutoDL physical simulator and online WBC not timed in this comparison.'))


if __name__=='__main__':main()
