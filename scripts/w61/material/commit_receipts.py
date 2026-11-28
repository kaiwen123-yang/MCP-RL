"""Commit closed A/C evidence on each existing branch for the one non-force integrator."""
import json
from pathlib import Path
import subprocess
import time

BASE=Path('/root/autodl-tmp')
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61')
LANES={'material':{'w61-a1-forefoot-controls-v1':24,'w61-a-m1-full-chain-v1':6,'w61-a2-physical-response-v1':3},
       'mpc':{'w61-c-transfer-wall-v1':8,'w61-c-material-history-v1':8,
              'w61-c-mppi-candidates-v2':6,'w61-c-mppi-wall-v2':4}}


def main():
    while True:
        closed=0
        for lane,studies in LANES.items():
            root=BASE/('MCP-RL-w61-'+lane)
            files=[];counts=[]
            for name,n in studies.items():
                folder=root/REL/lane/name
                rows=[]
                for p in sorted(folder.glob('*.json')):
                    try:rows.append((p,json.loads(p.read_text())))
                    except json.JSONDecodeError:continue # individual worker still finishing this output
                files.extend(str(p.relative_to(root)) for p,d in rows)
                fit=name=='w61-a2-physical-response-v1'
                count=sum(('rows' in d and d.get('kind') in ('held','residual')) if fit else
                          (('result' in d) and isinstance(d['result'],dict)) for p,d in rows)
                received=next((d for p,d in rows if p.name=='release.json'),{})
                done=(count==16 and 'model_and_optimizer_rng' in received) if fit else (count==n and received.get('completed_verified')==n)
                closed+=done
                counts.append(f'{name} {count}/{16 if fit else n}')
            if not files:continue
            subprocess.run(['git','add','--sparse','--',*files],cwd=root,check=True)
            status=subprocess.run(['git','diff','--cached','--quiet'],cwd=root)
            if status.returncode==1:
                subprocess.run(['git','commit','-qm','Record completed W61 '+lane+' runs and checked release assets: '+', '.join(counts)],cwd=root,check=True)
                print(lane,*counts,flush=True)
        if closed==sum(map(len,LANES.values())):return
        time.sleep(45)


if __name__=='__main__':main()
