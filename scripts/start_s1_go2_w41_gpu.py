#!/usr/bin/env python3
"""Start the independent W41 GPU work with no wall-clock deadline."""
import json,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu'
def main():
    session='s1-w41-gpu'
    if subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode==0:
        raise ValueError('W41 GPU already running; inspect its exact job.')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'job.json').exists():raise ValueError('Existing W41 GPU job must be continued explicitly from its saved stage.')
    cpus=sorted(os.sched_getaffinity(0))[-2:]
    command=['taskset','-c',','.join(map(str,cpus)),str(ROOT/'work/envs/d57/bin/python'),'-u','scripts/s1_go2_w41_gpu.py']
    job=dict(task='w41-gpu',session=session,deadline_unix_s=None,started_unix_s=time.time(),host_cpus=cpus,
        memory_budget_GiB=16,command=command,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        status='launched',old_w40_reused_as_reference_only=True,ppo_branch=False)
    (OUT/'job.json').write_text(json.dumps(job,indent=2)+'\n');(OUT/'clock.json').write_text(json.dumps(dict(started_unix_s=job['started_unix_s'],deadline_unix_s=None),indent=2)+'\n')
    script=OUT/'run-gpu.sh'
    script.write_text('\n'.join(['#!/bin/bash','set -e','cd '+shlex.quote(str(ROOT)),
        'source scripts/autodl_d57_env.sh','export PYTHONPATH="$PWD/work/gpu-w40-python:$PYTHONPATH"',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true',
        'export JAX_COMPILATION_CACHE_DIR="$PWD/work/cache/w41-jax"',
        'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=2"',
        'set +e',shlex.join(command)+' > '+shlex.quote(str(OUT/'gpu.log'))+' 2>&1',
        'echo "$?" > '+shlex.quote(str(OUT/'exit-code.txt')),''])+'\n')
    subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(script)],check=True)
    print(json.dumps(job,indent=2))
if __name__=='__main__':main()
