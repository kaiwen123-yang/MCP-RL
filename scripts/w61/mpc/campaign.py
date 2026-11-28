"""Run all registered W61-C stride cases through the shared physical pool, one slot per trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
COMPACT=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/mpc')
RUN='w61-c-transfer-closed-v1'
LEASE=Path('/root/autodl-tmp/MCP-RL-w61-joint/scripts/w61/lease.py')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--wall',action='store_true')
    parser.add_argument('--weighted',action='store_true')
    args=parser.parse_args()
    if args.weighted:
        prior=ROOT/COMPACT/'w61-c-material-history-v1'
        names=[f'{case}-{arm}.json' for case in ('entry','retrace') for arm in ('virgin','private','shared','posterior')]
        while not all((prior/name).exists() for name in names):time.sleep(15)
    runtime=args.runtime.resolve()
    cfg_rel=Path('experiments/terrain_memory/w61/mpc')/('transfer-wall-v1.json' if args.wall else 'transfer-v1.json')
    if args.weighted:
        cfg_rel=cfg_rel.with_name('mppi-wall-v2.json' if args.wall else 'mppi-candidates-v2.json')
    cfg=json.loads((runtime/cfg_rel).read_text())
    run_id=cfg['id']
    cache=ROOT/'work/w61/mpc'/run_id
    compact=ROOT/COMPACT/run_id
    compact.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ, PYTHONPATH=str(runtime/'scripts'), JAX_COMPILATION_CACHE_DIR=str(cache/'jax-cache'),
             JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0',GIT_DIR=str(runtime/'.git'),GIT_WORK_TREE=str(runtime))
    for case in cfg['registered_physical_cases']:
        methods=cfg['methods'] if args.weighted else (('hybrid','mppi','local','vr_mppi') if args.wall else ('hybrid','qp','mppi','local','vr_mppi'))
        for method in methods:
            name=f'{case}-{method}'
            out=cache/name
            receipt=compact/(name+'.json')
            if receipt.exists():
                continue
            cmd=[sys.executable,'-u',str(runtime/'scripts/w61_gait_rollout.py'),
                 '--contract',str(runtime/cfg_rel),'--case',case,'--out',str(out),'--role',method]
            if method!='qp':
                cmd+=['--mixin','w61.mpc.timed:TimedTopologyMPC' if args.wall else 'w61.mpc.controller:TopologyMPC']
            lease=(run_id+'-'+name) if args.weighted else ('c-wall-v1-' if args.wall else 'c-transfer-v1-')+name
            run=subprocess.run([sys.executable,str(LEASE),'--runtime',str(runtime),'--job',lease,
                                '--slots','1','--cores','1','--ram-gib','2','--gpu-mib','1800' if method!='qp' else '0',
                                '--scratch-gib','.18','--',*cmd],env=env)
            record=dict(run=run_id,case=case,method=method,returncode=run.returncode,pinned_source=args.source_commit,
                        raw_directory=str(out),lease=lease,command=cmd)
            if (out/'result.json').is_file():
                record['result']=json.loads((out/'result.json').read_text())
                record['gpu_used']=method!='qp' and (out/'c-plans.jsonl').stat().st_size>0
                record['gpu_scope']='JAX transfer search only; inherited runner times original 500 Hz feedback and 5 kHz physics'
                if (out/'c-activity.json').is_file():
                    record['search_activity']=json.loads((out/'c-activity.json').read_text())
                if (out/'timed-controller.json').is_file():
                    record['controller_timing']=json.loads((out/'timed-controller.json').read_text())
                record['trajectory_sha256']=hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest()
            receipt.write_text(json.dumps(record,indent=2)+'\n')
            print(name,run.returncode,record.get('result',{}).get('end_reason'),flush=True)
            if run.returncode:
                raise SystemExit(run.returncode)


if __name__=='__main__':
    main()
