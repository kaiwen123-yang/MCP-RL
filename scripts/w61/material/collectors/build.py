"""Compile the original FP64 M1 algorithms as one batched multi-environment launch."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ORIG=Path('/root/autodl-tmp/MCP-RL-w59/scripts')
OUT=ROOT/'work/w61/material/w61-e-packet-throughput-v1'
OUT.mkdir(exist_ok=True,parents=True)
src=ORIG/'s1_w59_m1_gpu.cu'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(src)=='e59421025cd4f95d87af25e6a550b3da5386304dd4f533598b98821a4c4eb679'
s=src.read_text()
# Keep the native create/cleanup and native material arithmetic. Exclude the
# old single-collector launch wrapper rather than calling it once per consumer.
s=s[:s.index('extern "C" int w59_gpu_propagate')]
for text in ('T* take(','Inputs inputs(','Outputs outputs('):
    assert s.count(text)==1
    s=s.replace(text,'__host__ __device__ '+text)
for name in ('propagate_columns','finish_packet'):
    s=s.replace('__global__ void '+name,'__device__ void '+name)
compilation=OUT/'fused-m1.cu';compilation.write_text(s+'\n'+(HERE/'fused.cu').read_text())
target=OUT/'fused-m1.so'
cmd=['/usr/local/cuda-12.8/bin/nvcc','-O3','-std=c++17','-arch=sm_120','--fmad=false','-DW59_ONE_COLUMN_BLOCK',
     '-I',str(ORIG),'-shared','-Xcompiler=-fPIC',str(compilation),'-o',str(target)]
subprocess.run(cmd,check=True)
(OUT/'build.json').write_text(json.dumps(dict(command=cmd,original_sha256=sha(src),
    constitutive_sha256=sha(ORIG/'s1_w55_c_column.cu'),batch_source_sha256=sha(HERE/'fused.cu'),
    compiled_source_sha256=sha(compilation),binary_sha256=sha(target)),indent=2)+'\n')
print(sha(target),flush=True)
