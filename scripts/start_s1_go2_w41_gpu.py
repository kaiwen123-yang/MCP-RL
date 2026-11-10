#!/usr/bin/env python3
"""Start or explicitly continue the independent W41 GPU work without a deadline."""
import argparse,json,os,shlex,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu'
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from',choices=['G2','G4','G5'])
    parser.add_argument('--network-mode',choices=['academic','clash','direct'],default='academic')
    parser.add_argument('--g4-backends',nargs='+',choices=['native','mjx'],default=['native','mjx'])
    args=parser.parse_args()
    session='s1-w41-gpu'
    if subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode==0:
        raise ValueError('W41 GPU already running; inspect its exact job.')
    OUT.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    stamp=args.resume_from.lower()+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') if args.resume_from else None
    if args.resume_from:
        report=json.loads((OUT/'result.json').read_text())
        if len(report['stages'].get('G1',[]))!=report['new_events']:
            raise ValueError('The existing G1 work is not complete; do not create this continuation.')
        if args.resume_from=='G4' and len(report['stages'].get('G2',{}).get('node_blocks',[]))!=176:
            raise ValueError('G4 continuation requires all completed G2 nodes; no automatic recomputation')
        if args.resume_from in ('G4','G5') and not (OUT/'native-evaluation-contract.json').is_file():
            raise FileNotFoundError('The original frozen GPU evaluation contract must be retained')
        history=OUT/'resume-history'/stamp;history.mkdir(parents=True,exist_ok=False)
        # Record existing file metadata before any new invocation records.
        baseline={str(p.relative_to(ROOT)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            for p in OUT.rglob('*') if p.is_file() and 'resume-history' not in p.relative_to(OUT).parts}
        (history/'baseline-sources.json').write_text(json.dumps(baseline,indent=2)+'\n')
        for name in ['job.json','clock.json','result.json','gpu.log','publication.log']:
            if (OUT/name).exists():shutil.copy2(OUT/name,history/name)
        if (OUT/'exit-code.txt').exists():(OUT/'exit-code.txt').rename(history/'exit-code.txt')
        if (SMALL/'delivery.json').exists():
            shutil.copy2(SMALL/'delivery.json',history/'delivery.json')
            delivery=json.loads((SMALL/'delivery.json').read_text())
            delivery.update(status='publication_pending',active_continuation=stamp,
                retained_published_delivery_record=str((history/'delivery.json').relative_to(ROOT)),
                continuation_delivery='Pending explicitly selected continuation stages; existing verified assets retained and not reuploaded.')
            (SMALL/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
        job=json.loads((OUT/'job.json').read_text());cpus=job['host_cpus']
        logfile=OUT/('gpu-resume-'+stamp+'.log');script=OUT/('run-gpu-resume-'+stamp+'.sh')
    else:
        if (OUT/'job.json').exists():raise ValueError('Existing W41 GPU job needs an explicit --resume-from stage.')
        cpus=sorted(os.sched_getaffinity(0))[-2:];logfile=OUT/'gpu.log';script=OUT/'run-gpu.sh'
        job=dict(task='w41-gpu',session=session,deadline_unix_s=None,started_unix_s=time.time(),host_cpus=cpus,
            memory_budget_GiB=16,source_commit=source,old_w40_reused_as_reference_only=True,ppo_branch=False)
        (OUT/'clock.json').write_text(json.dumps(dict(started_unix_s=job['started_unix_s'],deadline_unix_s=None),indent=2)+'\n')
    command=['taskset','-c',','.join(map(str,cpus)),str(ROOT/'work/envs/d57/bin/python'),'-u','scripts/s1_go2_w41_gpu.py']
    if args.resume_from:
        command+=['--resume-from',args.resume_from,'--resume-record',str(history/'result.json'),'--invocation',stamp]
        job.setdefault('continuations',[]).append(dict(invocation=stamp,resume_from=args.resume_from,started_unix_s=time.time(),
            source_commit=source,preserved_record=str((history/'result.json').relative_to(ROOT)),G1_recomputed=False,
            G2_recomputed=False if args.resume_from in ('G4','G5') else None))
        job['active_invocation']=stamp
    command+=['--g4-backends',*args.g4_backends]
    job.update(status='launched',command=command,log=str(logfile.relative_to(ROOT)),active_source_commit=source,deadline_unix_s=None,network_mode=args.network_mode)
    (OUT/'job.json').write_text(json.dumps(job,indent=2)+'\n')
    lines=['#!/bin/bash','set -e','cd '+shlex.quote(str(ROOT)),
        'source scripts/autodl_d57_env.sh','export PYTHONPATH="$PWD/work/gpu-w40-python:$PYTHONPATH"',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true',
        'export JAX_COMPILATION_CACHE_DIR="$PWD/work/cache/w41-jax"',
        'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=2"',
        'set +e',shlex.join(command)+' > '+shlex.quote(str(logfile))+' 2>&1',
        'echo "$?" > '+shlex.quote(str(OUT/'exit-code.txt'))]
    if stamp:
        # Publication owns new names and includes only continuation additions.
        finish=[str(ROOT/'work/envs/d57/bin/python'),'-u','scripts/finish_s1_go2_w41_gpu.py','--continuation',stamp]
        lines+=['source scripts/autodl_network.sh '+args.network_mode+' >/dev/null','unset GH_TOKEN GITHUB_TOKEN',
            shlex.join(finish)+' > '+shlex.quote(str(OUT/('publication-'+stamp+'.log')))+' 2>&1']
    script.write_text('\n'.join(lines)+'\n')
    subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(script)],check=True)
    print(json.dumps(dict(session=session,resume_from=args.resume_from,invocation=stamp,log=job['log'],source_commit=source,deadline_unix_s=None),indent=2))
if __name__=='__main__':main()
