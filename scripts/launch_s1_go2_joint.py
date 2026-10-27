#!/usr/bin/env python3
"""The five predeclared remaining R9 full-mission adaptation starts."""
import json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def execute(spec):
    name=spec['attempt']
    cmd=[sys.executable,'-u',str(ROOT/'scripts/run_s1_go2_joint.py'),
        '--attempt',name,'--cap',str(spec['cap']),'--profile',spec['profile']]
    with (ROOT/'work'/('r9-'+name+'.log')).open('w') as f:
        result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(attempt=name,exit_code=result.returncode)),flush=True)
    if result.returncode==0:
        with (ROOT/'work'/('r9-'+name+'-render.log')).open('w') as f:
            subprocess.run(cmd+['--render'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    return result.returncode

def main():
    c=json.loads((ROOT/'experiments/terrain_memory/s1-go2-joint.json').read_text())
    with ThreadPoolExecutor(3) as pool:
        codes=list(pool.map(execute,c['development']['remaining_predeclared_intents']))
    print(json.dumps(dict(finished=True,codes=codes)),flush=True)
if __name__=='__main__':main()

