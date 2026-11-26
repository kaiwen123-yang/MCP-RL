"""Dispatch registered action continuations through the shared D57 physical lease."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
SMALL=ROOT/'results'/REL/'w59'
LARGE=ROOT/'outputs'/REL


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def immutable(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=record:raise ValueError('Changed branch registration '+str(path))
    else:
        with path.open('x') as out:json.dump(record,out,indent=2);out.write('\n')


def materialize(path,role,freeze):
    reg=json.loads(path.read_text())
    if reg['status']!='locked':raise ValueError('Incomplete action-state registration')
    if role=='sealed_admission' and not freeze:
        raise ValueError('Sealed action admission requires the recipe/checkpoint freeze before any candidate result')
    digest=sha(path)
    for state in reg['states'][role]:
        for signed,value in reg['signed'].items():
            job=f'w59-action-{role.split("_")[0]}-{state["case"].lower()}-r{state["request_id"]:05d}-{signed}-a01'
            record=dict(**state,role=role,signed=signed,replacement=value,duration_s=reg['duration_s'],
                registry_sha256=digest,model_recipe_sha256=sha(freeze) if freeze else None,
                rng_key=state['sha256']+'/stored',actor_digest=None,job=job)
            config=ROOT/'work/w59/interventions'/role/(job+'.json')
            immutable(config,record)
            yield job,state['case'],config


def launch(job,case,config,runtime):
    cmd=[sys.executable,str(ROOT/'scripts/s1_w59_lease.py'),'--runtime',str(runtime),'--job',job,'--slots','1','--',
        sys.executable,'-u','scripts/run_s1_w59_research.py','--case',case,'--arm','B3','--level','E30',
        '--task','branch','--branch',str(config),'--attempt',job]
    path=SMALL/'episodes'/(job+'.json')
    if not path.exists():
        rc=subprocess.run(cmd,cwd=runtime).returncode
    else:rc=0
    if not path.is_file():raise RuntimeError(f'{job} has no terminal record (launcher={rc})')
    row=json.loads(path.read_text())
    print(json.dumps({k:row.get(k) for k in ('attempt','stage_status','physical_samples','reason')}),flush=True)
    if row['stage_status']=='technical_interruption' or not row.get('branch',{}).get('applied'):
        raise RuntimeError(job+' recorded a technical interruption or did not reach its registered request')
    return job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--role',choices=('development','sealed_admission'),required=True)
    p.add_argument('--freeze',type=Path);p.add_argument('--workers',type=int,choices=range(1,7),default=3)
    args=p.parse_args()
    jobs=iter(materialize(args.registry,args.role,args.freeze))
    # A small queue lets the other original qualifications and campaigns share this same pool.
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        active=set()
        def next_job():
            item=next(jobs,None)
            if item:active.add(pool.submit(launch,*item,args.runtime))
        for _ in range(args.workers):next_job()
        while active:
            done,active=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for job in done:job.result()
            for _ in done:next_job()


if __name__=='__main__':main()
