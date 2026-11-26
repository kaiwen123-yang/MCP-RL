"""Run an earned planning arm on four new fixed-footprint episodes through the shared lease."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from s1_w59_lease import ROOT
from run_s1_w59_research import SMALL
from publish_s1_go2_w54 import write


def execute(runtime, arm, material, split, case, mode, checkpoints, build, recipe):
    name=f'w59-{arm.lower()}-{material.lower()}-{case.lower()}-{split}-{mode}-a01'
    args=['--case',case,'--arm',arm,'--material',material,'--task',mode,'--split',split,
          '--level','E60','--attempt',name]
    if recipe:
        args+=['--wm-recipe',str(recipe)]
    for ckpt in checkpoints:
        args+=['--wm-checkpoint',str(ckpt)]
    final=SMALL/'episodes'/(name+'.json')
    if not final.is_file():
        cmd=[sys.executable,str(ROOT/'scripts/s1_w59_lease.py'),'--runtime',str(runtime),'--job',name,
             '--slots','1','--',sys.executable,'-u','scripts/run_s1_w59_research.py',*args]
        rc=subprocess.run(cmd,cwd=runtime).returncode
        if not final.exists():
            raise RuntimeError(f'{name}: missing result, launcher exit {rc}')
    actual=json.loads((SMALL/'attempts'/(name+'.json')).read_text())
    if actual['controller']!=build:
        raise ValueError('Mechanism controller differs from the qualified controller')
    return {'attempt':name,'status':actual['status'],'controller':build}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True,type=Path)
    p.add_argument('--arm',required=True,choices=('B3','Fixed-action','P3-full','P3-private'))
    p.add_argument('--material',default='M1',choices=('M1','M3'))
    p.add_argument('--split',required=True,choices=('development','heldout'))
    p.add_argument('--build',required=True)
    p.add_argument('--wm-checkpoint',action='append',default=[],type=Path)
    p.add_argument('--workers',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--mode',choices=('short','recovered'),help='Admit a missing mode alongside an existing four-episode coordinator')
    p.add_argument('--freeze',type=Path)
    p.add_argument('--wm-recipe',type=Path)
    a=p.parse_args()
    if a.split=='heldout' and (not a.freeze or not a.freeze.is_file()):
        p.error('Heldout episodes require the locked method selection')
    # Match original qualification receipts by controller hash, task, stage and case.
    # After every attempt for an identity has closed, only qualifying results open its next stage.
    while True:
        needed={(case,level) for case in 'AB' for level in ('E30','E60')}
        failed=[]
        for source in (SMALL/'attempts').glob('*.json'):
            row=json.loads(source.read_text())
            key=row.get('case'),row.get('level')
            if row.get('task_mode')!='original' or row.get('controller')!=a.build or key not in needed:
                continue
            if row['status']=='passed':
                needed.discard(key)
            elif row.get('ended_unix_s'):
                failed.append((key,row['status']))
        if not needed:
            break
        if any(key in needed for key,_ in failed):
            print('Qualification incomplete for this identity',needed,failed,flush=True)
            return
        time.sleep(20)
    # Verify the full declared case-material-model identity before charging a new episode.
    import os
    check = """from types import SimpleNamespace
from run_s1_w59_research import declaration
import sys
a=SimpleNamespace(case='A',arm=sys.argv[1],material=sys.argv[2],split=sys.argv[3],task='short',level='E60',
wm_checkpoint=sys.argv[6:],learner_config=None,branch=None,wm_recipe=sys.argv[5] or None)
assert declaration(a)[1]['controller_build']==sys.argv[4]"""
    env=dict(os.environ,PYTHONPATH=str(a.runtime/'scripts')+':'+os.environ.get('PYTHONPATH',''))
    subprocess.run([sys.executable,'-c',check,a.arm,a.material,a.split,a.build,str(a.wm_recipe or ''),*map(str,a.wm_checkpoint)],cwd=a.runtime,env=env,check=True)
    with ThreadPoolExecutor(a.workers) as pool:
        pending=[pool.submit(execute,a.runtime,a.arm,a.material,a.split,c,m,a.wm_checkpoint,a.build,a.wm_recipe)
                 for m in ((a.mode,) if a.mode else ('short','recovered')) for c in 'AB']
        result=[f.result() for f in pending]
    if a.mode:
        return
    name=f'{a.arm.lower()}-{a.material.lower()}-{a.split}'
    write(SMALL/'mechanism'/(name+'.json'),dict(arm=a.arm,material=a.material,split=a.split,build=a.build,
          episodes=result,denominator=4,freeze=str(a.freeze) if a.freeze else None))


if __name__=='__main__':
    main()
