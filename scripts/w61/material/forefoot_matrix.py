"""Run the preregistered physical material comparison with one reserved slot per trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from w61.joint import plant

ROOT=Path(__file__).resolve().parents[3]
W61ROOT=Path('/root/autodl-tmp/MCP-RL-w61-material')
POOL=Path('/root/autodl-tmp/MCP-RL-w61-joint/scripts/w61/lease.py')
CFG=ROOT/'experiments/terrain_memory/w61/material/forefoot-controls-v1.json'
PAR=json.loads(CFG.read_text())
RUN=PAR['id']
RAW=W61ROOT/'work/w61/material'/RUN
DEST=W61ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/material'/RUN


def main():
    dest=DEST
    dest.mkdir(parents=True,exist_ok=True)
    for i,case in enumerate(PAR['cases']):
        trained=[(method,seed) for seed in PAR['fit_seeds'] for method in PAR['learned_methods']]
        methods=[(m,0) for m in PAR['analytic_methods']]+trained
        for method,seed in methods:
            name=f"s{case['specimen']}-{method}-{seed}"
            fn=dest/(name+'.json')
            if fn.is_file():
                previous=json.loads(fn.read_text())
                if previous['exit_code']==0:continue
                # The first A attempt stopped at startup with an omitted coupon config.
                if name!='s612241-ordinary_KV-0':raise RuntimeError('Review failed physical attempt '+name)
                name+='-a02'
                fn=dest/(name+'.json')
                if fn.exists():continue
            out=RAW/name
            cmd=[sys.executable,'-u',str(ROOT/'scripts/w61/material/forefoot.py'),
                 str(CFG),str(i),method,str(seed),str(out)]
            lease='a-forefoot-v1-'+name
            proc=subprocess.run([sys.executable,str(POOL),'--runtime',str(ROOT),'--job',lease,
                        '--slots','1','--cores','1','--ram-gib','1.4','--gpu-mib','0','--scratch-gib','.08','--',*cmd])
            record=dict(case=case,method=method,coupon_fit_seed=seed or None,lease=lease,
                        runtime_material=os.environ['W61_A_SOURCE'],runtime_joint=PAR['frozen_joint'],
                        source_contract_sha256=hashlib.sha256(CFG.read_bytes()).hexdigest(),
                        original_directory=str(out),exit_code=proc.returncode)
            if (out/'result.json').is_file():record['result']=json.loads((out/'result.json').read_text())
            fn.write_text(json.dumps(record,indent=2)+'\n')
            print(name,proc.returncode,record.get('result',{}).get('reason'),flush=True)
            if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':
    main()
