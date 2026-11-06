#!/usr/bin/env python3
"""Launch exactly one bounded independent GPU adaptation task."""
import json,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w40-gpu'
def main():
    OUT.mkdir(parents=True,exist_ok=True);session='s1-w40-gpu'
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:
        raise RuntimeError('Existing GPU task: inspect it rather than duplicate.')
    clockfile=OUT/'clock.json'
    if clockfile.exists():clock=json.loads(clockfile.read_text())
    else:
        now=time.time();clock=dict(started_unix_s=now,deadline_unix_s=now+28*3600)
        clockfile.write_text(json.dumps(clock,indent=2)+'\n')
    remaining=clock['deadline_unix_s']-time.time()
    if remaining<=0:raise RuntimeError('Original GPU deadline elapsed.')
    cpus=sorted(os.sched_getaffinity(0))[-2:]
    stamp=str(int(time.time()))
    prior_job=OUT/'job.json'
    is_resume=prior_job.exists()
    if is_resume:(OUT/('job-before-resume-'+stamp+'.json')).write_bytes(prior_job.read_bytes())
    logfile=OUT/(('gpu-resume-'+stamp+'.log') if is_resume else 'gpu.log')
    script=OUT/(('run-gpu-resume-'+stamp+'.sh') if is_resume else 'run-gpu.sh')
    cmd=['timeout','--signal=TERM','--kill-after=30s',str(int(remaining)),
         'taskset','-c',','.join(map(str,cpus)),str(ROOT/'work/envs/d57/bin/python'),'-u',
         'scripts/s1_go2_w40_gpu.py']
    script.write_text('\n'.join(['#!/bin/bash','set -e','cd '+shlex.quote(str(ROOT)),
      'source scripts/autodl_d57_env.sh',
      'export PYTHONPATH="$PWD/work/gpu-w40-python:$PYTHONPATH"',
      'export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true',
      'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
      'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=2"',
      'set +e',shlex.join(cmd)+' > '+shlex.quote(str(logfile))+' 2>&1',
       'echo "$?" > '+shlex.quote(str(OUT/'exit-code.txt')),
      'source scripts/autodl_network.sh academic',
      'python scripts/finish_s1_go2_w40_gpu.py > '+shlex.quote(str(OUT/('finish-'+stamp+'.log')))+' 2>&1',''])+'\n')
    job=dict(task='w40-gpu',session=session,clock=clock,host_cpus=cpus,log=str(logfile.relative_to(ROOT)),resume=is_resume,
      command=cmd,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      dependencies={'mujoco':'3.2.3','mujoco-mjx':'3.2.3','jax':'0.6.2','jaxlib':'0.6.2','trimesh':'4.5.3'},
      dependency_directory='work/gpu-w40-python',main_environment_changed=False,
      max_compile_jobs=2,ppo_branch=False)
    (OUT/'job.json').write_text(json.dumps(job,indent=2)+'\n')
    subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(script)],check=True)
    print(json.dumps(job,indent=2))
if __name__=='__main__':main()
