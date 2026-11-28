"""Publish each naturally terminated real observer run; lease one physical slot per trial."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT_BASE=Path('/root/autodl-tmp/MCP-RL-w61-material')
SPEC=ROOT/'experiments/terrain_memory/w61/material/m1-fullchain-v1.json'
POOL=Path('/root/autodl-tmp/MCP-RL-w61-joint/scripts/w61/lease.py')
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/material')
CONF=json.loads(SPEC.read_text())
RAW=OUT_BASE/'work/w61/material'/CONF['id']
DEST=OUT_BASE/REL/CONF['id']


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for case in CONF['cases']:
        for arm in CONF['arms']:
            outcome=DEST/f'{case}-{arm}.json'
            if outcome.exists():continue
            out=RAW/f'{case}-{arm}'
            job=f'w61-m1-live-v1-{case}-{arm}'
            cmd=[sys.executable,str(ROOT/'scripts/w61/material/live_observer.py'),str(SPEC),case,arm,str(out)]
            proc=subprocess.run([sys.executable,str(POOL),'--runtime',str(ROOT),'--job',job,'--slots','1','--cores','1',
                                 '--ram-gib','1.5','--gpu-mib','500' if arm!='cpu' else '0','--scratch-gib','.15','--',*cmd])
            result=dict(job=job,source_commit=os.environ['W61_OBS_SOURCE'],exit_code=proc.returncode,
                        original_directory=str(out),command=cmd)
            if (out/'result.json').exists():result['result']=json.loads((out/'result.json').read_text())
            outcome.write_text(json.dumps(result,indent=2)+'\n')
            print(case,arm,proc.returncode,result.get('result',{}).get('reason'),flush=True)
            if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':
    main()
