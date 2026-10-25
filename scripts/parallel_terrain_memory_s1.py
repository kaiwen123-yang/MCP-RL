#!/usr/bin/env python3
"""Fixed S1 transition matrix: concurrent complete missions and host publication."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
CONFIG='experiments/terrain_memory/s1-contact-transition.json'
C=json.loads((ROOT/CONFIG).read_text())
SMALL=ROOT/C['outputs']['compact_root']
INDEX=SMALL/'index.json'
WORK=ROOT/'work/s1-transition'

@contextmanager
def index_edit():
    with (SMALL/'index.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        index=json.loads(INDEX.read_text())
        yield index
        INDEX.write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')

def adopt(record,development=False):
    row=json.loads(record.read_text())
    name=f"{row['terrain']}-{row['controller']}-seed-{row['seed']}"
    row['actual_execution_record']=str(record.relative_to(ROOT))
    row['resource_context']=dict(mode='parallel',maximum_workers=7,nice=10,
        cpu_affinity='one CPU per worker; BLAS/OMP=1',GPU='CUDA sampling MPC; no preallocation',
        timing_scope='Shared CPU/GPU, not an isolated method benchmark or NX timing.')
    if development:
        row['reused_development_from']=str(record.relative_to(ROOT))
    target=SMALL/'formal-v1'/(name+'.json')
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(row,indent=2)+'\n')

def resources(active):
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    gpu=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    ours={item['process'].pid for item in active.values()}
    external=[int(p) for p in compute.splitlines() if p.strip().isdigit() and int(p) not in ours]
    return dict(available_memory_GiB=memory['MemAvailable']/1024**2,free_GPU_MiB=gpu,
        competing_GPU_compute_processes=len(external),
        admit=memory['MemAvailable']>4*1024**2 and gpu>2048 and not external)

def run_cells(cells,workers):
    queue=list(cells); active={}; errors=[]
    while queue or active:
        for name,item in list(active.items()):
            code=item['process'].poll()
            if code is None:
                continue
            item['log'].close()
            record=SMALL/item['cohort']/(name+'.json')
            if code==0:
                adopt(record)
                row=json.loads(record.read_text())
                print(json.dumps(dict(event='finished',case=name,success=row['success'],termination=row['termination'],duration_s=row['duration_s'])),flush=True)
            else:
                errors.append(dict(case=name,exit_code=code,log=str(item['logpath'].relative_to(ROOT))))
                with index_edit() as index:
                    for a in index['attempts']:
                        if a['cohort']==item['cohort'] and a['status']=='started':
                            a.update(status='execution_error',exit_code=code,log=str(item['logpath'].relative_to(ROOT)))
                    index['execution_errors']=errors
                print(json.dumps(dict(event='execution_error',**errors[-1])),flush=True)
            del active[name]
        if errors:
            queue=[]
        if queue and len(active)<workers:
            resource=resources(active)
            if resource['admit']:
                terrain,method,seed=queue.pop(0)
                name=f'{terrain}-{method}-seed-{seed}'
                cohort='formal-v1-'+name
                cpu=next(c for c in range(1,workers+1) if c not in [x['cpu'] for x in active.values()])
                logpath=WORK/(cohort+'.log'); log=logpath.open('w')
                command=['nice','-n','10','taskset','-c',str(cpu),sys.executable,str(ROOT/'scripts/run_terrain_memory_s1.py'),
                    '--config',CONFIG,'--terrain',terrain,'--controller',method,'--seeds',str(seed),'--cohort',cohort]
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                active[name]=dict(process=process,log=log,logpath=logpath,cohort=cohort,cpu=cpu)
                with index_edit() as index:
                    index['parallel_execution']['last_admission']=dict(utc=datetime.now(timezone.utc).isoformat(),case=name,pid=process.pid,cpu=cpu,**resource)
                print(json.dumps(dict(event='launched',case=name,pid=process.pid,cpu=cpu,active_workers=len(active))),flush=True)
        time.sleep(2)
    if errors:
        with index_edit() as index:
            index['status']='execution_error_requires_inspection'
        raise RuntimeError('Execution errors retained; no retries: '+json.dumps(errors))

def readiness():
    rows=[json.loads(p.read_text()) for p in (SMALL/'formal-v1').glob('*-T1-seed-*.json')]
    completed={t:sum(r['terrain']==t and r['success'] for r in rows) for t in C['phase_A']['terrain_ids']}
    qualifying=[r['seed'] for r in rows if r['terrain']=='tm02_anchor' and r['success'] and r['front_to_hind_revisit']]
    queries=any(r['terrain']=='tm02_anchor' and r['precontact_query_used'] for r in rows)
    ready=(completed['rigid']==5 and completed['kv_moderate']>=4 and completed['tm02_anchor']>=4 and len(qualifying)>=4 and queries)
    result=dict(ready=ready,T1_completions=completed,completed_TM02_front_to_hind_seeds=qualifying,
        precontact_query_used_in_execution=queries,decision='Run all five T2 seeds.' if ready else 'Phase B not run; fixed readiness not met.')
    with index_edit() as index:
        index['phase_A_complete']=True; index['phase_B_readiness']=result
    print(json.dumps(dict(event='phase_B_readiness',**result)),flush=True)
    return ready

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=7)
    parser.add_argument('--adopt-development',default='development-v2-')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args(); workers=min(args.workers,os.cpu_count()-1)
    WORK.mkdir(exist_ok=True)
    paths=[ROOT/'scripts/run_terrain_memory_s1.py',*sorted((ROOT/'src/tm02').glob('s1*.py'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with index_edit() as index:
        index.update(status='running',phase_A_complete=False,phase_B_complete=False,
            implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            frozen_execution_sha256=hashes,common_correction=C['resolved_transition']['common_swing_correction'],
            common_correction_affected_cells='All 15 B1 pairs, three terrains and five seeds.',
            parallel_execution=dict(status='running',maximum_workers=workers,nice=10,CPU_count=os.cpu_count(),
                reserved_CPU=0,BLAS_threads_per_worker=1,GPU_preallocate=False,
                admission='At least 4GiB host and 2GiB GPU free; no competing GPU compute process at admission.',
                timing_scope='Shared CPU/GPU; no isolated speed or NX claim.',command=[sys.executable,*sys.argv]),
            known_unrecoverable_development_data='First three attempts lost in-memory trajectories at serialization. Exact snapshots/logs retained; physical outcome not inferred. Included in development and total budgets.')
        attempts=list(index['attempts'])
    for a in attempts:
        if a['cohort'].startswith(args.adopt_development) and a['status']=='finished':
            record=ROOT/a['record']; row=json.loads(record.read_text())
            if row['source_sha256']==hashes:
                adopt(record,development=True)
    cells=[(t,m,s) for s in C['phase_A']['paired_optimizer_seeds'] for t in C['phase_A']['terrain_ids'] for m in C['phase_A']['controller_ids']
        if not (SMALL/'formal-v1'/f'{t}-{m}-seed-{s}.json').exists()]
    run_cells(cells,workers)
    if readiness():
        run_cells([('tm02_anchor','T2',s) for s in C['phase_B']['paired_optimizer_seeds']],workers)
        with index_edit() as index:
            index['phase_B_complete']=True
    with index_edit() as index:
        index['status']='experiments_finished_postprocessing'; index['parallel_execution']['status']='finished'
    for command in (
        [sys.executable,'scripts/summarize_s1_transition.py'],
        [sys.executable,'scripts/video_terrain_memory_s1.py','--config',CONFIG,'--cohort','formal-v1'],
        [sys.executable,'scripts/package_terrain_memory_s1.py','--config',CONFIG,'--cohort','formal-v1','--release-tag','tm-s1-contact-transition-evidence-20261025']):
        subprocess.run(command,cwd=ROOT,check=True)
    if args.publish:
        subprocess.run([sys.executable,'scripts/publish_s1_transition.py'],cwd=ROOT,check=True)
    print(json.dumps(dict(event='pipeline_finished',published=args.publish)),flush=True)

if __name__=='__main__':
    main()
