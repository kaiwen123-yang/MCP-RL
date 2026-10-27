#!/usr/bin/env python3
"""Explicit R8 author, development and frozen coverage batches."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG='experiments/terrain_memory/s1-go2-wb-support.json'
SOURCE=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'
ORIGINAL=ROOT.parent/'terrain-memory-s1-execution'
ROBOT=ORIGINAL/'work/envs/pympc/lib/python3.11/site-packages/gym_quadruped'

def execute(spec):
    name=spec['attempt'];stage=spec['stage']
    binary=ROOT/'work/r8-author-agent_server' if stage=='G0' else SOURCE/'build/bin/agent_server'
    cmd=[sys.executable,'-u',str(ROOT/'scripts/run_s1_go2_wb_baseline.py'),'--config',CONFIG,
        '--agent-server',str(binary),'--author-source',str(SOURCE),
        '--original-robot-dir',str(ROBOT/'robot_model/go2'),
        '--original-scene',str(ROBOT/'utils/mujoco/assets/scene_flat.xml'),
        '--original-resolved',str(ORIGINAL/'results/terrain_memory/simulation/tm-s1-contact-execution-20261026/development-v1-seed4/rigid-T1R-resolved.json')]
    for key,value in spec.items():cmd+=['--'+key.replace('_','-'),str(value)]
    with (ROOT/'work'/('r8-'+name+'.log')).open('w') as f:
        p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    (ROOT/'work'/('r8-'+name+'.exit')).write_text(str(p.returncode)+'\n')
    print(json.dumps(dict(attempt=name,exit_code=p.returncode)),flush=True)
    return name,p.returncode

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['author','development','formal'],required=True)
    p.add_argument('--arm',default='C');p.add_argument('--attempt');p.add_argument('--mission',choices=['single','long'],default='single')
    p.add_argument('--cohort',default='development-v1');p.add_argument('--workers',type=int,default=6)
    p.add_argument('--parallel',type=int,default=1);p.add_argument('--reuse-single');p.add_argument('--reuse-long')
    args=p.parse_args();c=json.loads((ROOT/CONFIG).read_text())
    if args.batch=='author':
        specs=[dict(attempt='author-n-long-'+x,stage='G0',arm='AUTHOR_N',condition=x,
            purpose='formal',mission='long',cohort='author-reference',workers=args.workers) for x in c['author_long_reference']['conditions']]
    elif args.batch=='development':
        specs=[dict(attempt=args.attempt,stage='G1',arm=args.arm,condition='nominal',purpose='development',
            mission=args.mission,cohort=args.cohort,workers=args.workers)]
    else:
        from run_s1_go2_wb_baseline import index_file
        root=ROOT/c['outputs']['root']
        reused=[x for x in [args.reuse_single,args.reuse_long] if x]
        if reused:
            for name,mission in [(args.reuse_single,'single'),(args.reuse_long,'long')]:
                r=json.loads((root/(name+'.json')).read_text())
                if not (r['sustained_execution_success'] and r['arm']==args.arm and r['mission']==mission):
                    raise ValueError('Formal reuse requires both identical successful qualification missions')
        with index_file(root,c) as idx:
            idx.setdefault('formal_cohorts',{})[args.cohort]=dict(arm=args.arm,reused_attempts=reused)
        specs=[]
        for mission,conditions,reuse in [('single',[x['id'] for x in c['single_mission_protocol']['formal_conditions']],args.reuse_single),
                ('long',c['long_mission_protocol']['conditions'],args.reuse_long)]:
            for name in conditions:
                if name=='nominal' and reuse:continue
                specs.append(dict(attempt=args.cohort+'-'+mission+'-'+name,stage='G1',arm=args.arm,condition=name,
                    purpose='formal',mission=mission,cohort=args.cohort,workers=args.workers))
    with ThreadPoolExecutor(args.parallel) as pool:
        for future in as_completed([pool.submit(execute,x) for x in specs]):future.result()
if __name__=='__main__':main()
