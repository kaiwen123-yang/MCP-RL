"""Chain the C material-history trials after C's resident-budget physical matrix."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
SITE=Path('/root/autodl-tmp/MCP-RL-w61-mpc')
CFG=ROOT/'experiments/terrain_memory/w61/mpc/history-v1.json'
SPEC=json.loads(CFG.read_text())
POOL=Path('/root/autodl-tmp/MCP-RL-w61-joint/scripts/w61/lease.py')
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/mpc')
BASE=SITE/'work/w61/mpc'/SPEC['id']


def main():
    prior=SITE/REL/'w61-c-transfer-wall-v1'
    while len(list(prior.glob('query*.json')))<8:time.sleep(15)
    dest=SITE/REL/SPEC['id']
    dest.mkdir(parents=True,exist_ok=True)
    for case in SPEC['cases']:
        for arm in SPEC['arms']:
            name=case+'-'+arm
            out=BASE/name
            result=dest/(name+'.json')
            if result.exists():continue
            job='c-history-v1-'+name
            cmd=[sys.executable,str(ROOT/'scripts/w61/mpc/history.py'),str(CFG),case,arm,str(out)]
            proc=subprocess.run([sys.executable,str(POOL),'--runtime',str(ROOT),'--job',job,'--slots','1','--cores','1',
                '--ram-gib','2','--gpu-mib','1800','--scratch-gib','.2','--',*cmd])
            rec=dict(job=job,case=case,arm=arm,source_commit=os.environ['W61_C_SOURCE'],observer_source=SPEC['observer_source'],
                     exit_code=proc.returncode,raw_directory=str(out))
            if (out/'result.json').is_file():rec['result']=json.loads((out/'result.json').read_text())
            if (out/'memory.json').is_file():rec['history_decisions']=json.loads((out/'memory.json').read_text())
            result.write_text(json.dumps(rec,indent=2)+'\n')
            print(name,proc.returncode,rec.get('result',{}).get('reason'),flush=True)
            if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__': main()
