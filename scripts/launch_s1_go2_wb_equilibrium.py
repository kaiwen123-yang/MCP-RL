#!/usr/bin/env python3
"""Launch only the explicit R7 complete-mission batch, with bounded concurrency."""
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG='experiments/terrain_memory/s1-go2-wb-equilibrium.json'
SOURCE=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'
ORIGINAL=ROOT.parent/'terrain-memory-s1-execution'
ROBOT=ORIGINAL/'work/envs/pympc/lib/python3.11/site-packages/gym_quadruped'


def command(attempt,stage,arm,condition,purpose,mission,cohort,workers=2):
    return [sys.executable,'-u',str(ROOT/'scripts/run_s1_go2_wb_baseline.py'),
        '--config',CONFIG,'--attempt',attempt,'--stage',stage,'--arm',arm,'--condition',condition,
        '--purpose',purpose,'--mission',mission,'--cohort',cohort,'--workers',str(workers),
        '--agent-server',str(SOURCE/'build/bin/agent_server'),'--author-source',str(SOURCE),
        '--original-robot-dir',str(ROBOT/'robot_model/go2'),
        '--original-scene',str(ROBOT/'utils/mujoco/assets/scene_flat.xml'),
        '--original-resolved',str(ORIGINAL/'results/terrain_memory/simulation/tm-s1-contact-execution-20261026/development-v1-seed4/rigid-T1R-resolved.json')]


def execute(spec):
    cmd=command(**spec);name=spec['attempt'];log=ROOT/'work'/('r7-'+name+'.log')
    print(json.dumps(dict(start=name,command=cmd)),flush=True)
    with log.open('w') as f:
        process=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    (ROOT/'work'/('r7-'+name+'.exit')).write_text(str(process.returncode)+'\n')
    print(json.dumps(dict(finished=name,exit_code=process.returncode)),flush=True)
    return name,process.returncode


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['g0','development','formal'],required=True)
    p.add_argument('--arm',default='ON');p.add_argument('--attempt');p.add_argument('--cohort',default='formal-v1')
    p.add_argument('--reuse-nominal');p.add_argument('--parallel',type=int,default=3);args=p.parse_args()
    c=json.loads((ROOT/CONFIG).read_text());specs=[]
    if args.batch=='g0':
        specs=[dict(attempt='g0-'+arm.lower()+'-01',stage='G0',arm=arm,condition='nominal',purpose='development',mission='single',cohort='development-v1') for arm in ['O','N','ON']]
    elif args.batch=='development':
        specs=[dict(attempt=args.attempt,stage='G1',arm=args.arm,condition='nominal',purpose='development',mission='single',cohort='development-v1')]
    else:
        for condition in c['single_mission_protocol']['formal_conditions']:
            name=condition['id']
            if name=='nominal' and args.reuse_nominal:continue
            specs.append(dict(attempt=args.cohort+'-'+name,stage='G1',arm=args.arm,condition=name,purpose='formal',mission='single',cohort=args.cohort))
        # Long traces are scheduled after the single-task cohort using the same immutable controller.
        long_specs=[dict(attempt=args.cohort+'-long-'+name,stage='G1',arm=args.arm,condition=name,purpose='formal',mission='long',cohort=args.cohort) for name in c['long_mission_protocol']['conditions']]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for future in as_completed([pool.submit(execute,x) for x in specs]):future.result()
    if args.batch=='formal':
        with ThreadPoolExecutor(max_workers=min(2,args.parallel)) as pool:
            for future in as_completed([pool.submit(execute,x) for x in long_specs]):future.result()
    print('Explicit complete-mission batch finished; inspect all actual results before choosing revisions or publishing.',flush=True)
if __name__=='__main__':main()
