"""Apply W59 physical branch comparisons to the unchanged W59 50k backbone ensembles."""
import gzip
import hashlib
import json
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[3]
FROZEN=ROOT.parent/'MCP-RL-w59'
sys.path.insert(0,str(FROZEN/'scripts'))
from run_s1_w59_interventions import SMALL,LARGE,job_name
from score_s1_w59_interventions import physical_rows,starting_state,summarize
from s1_w59_action_horizons import NODES,horizon_comparisons
from s1_w59_forward_domain import historical
from s1_w59_model import ExecutionEnsemble
from s1_w59_model_bank import write
from s1_w59_model_fields import BG


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/material'
    run='w61-a0-50k-action-v1'
    out=root/run
    raw=ROOT/'work/w61/material'/run
    raw.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    old=SMALL/'action'
    source=old/'registry-v3.json'
    reg=json.loads(source.read_text())
    frozen=json.loads((old/'model-recipe-freeze.json').read_text())
    lim=frozen['observed_normal_max_N']
    cp_dir=FROZEN/'work/w59/model/checkpoints'
    origin_reports=root/'w61-a0-backbones-50k-v1'
    kinds=('structured','blackbox')
    models={}
    allowed={}
    identity={}
    for kind in kinds:
        files=[cp_dir/kind/str(s)/'node-50000.pt' for s in (5901,5902,5903)]
        models[kind]=ExecutionEnsemble.load(files,'cuda').eval()
        identity[kind]=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
        for s in (5901,5902,5903):
            link=raw/f'{kind}-{s}-open-loop.json'
            if not link.exists():
                link.symlink_to(origin_reports/f'{kind}-{s}.json')
        allowed[kind]=historical(raw,kind,NODES,lim)
    runtime=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(out/'inputs.json',dict(runtime=runtime,W59_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=FROZEN,text=True).strip(),
        registry=str(source),registry_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoints=identity,
        forward_gate=allowed,observed_normal_max_N=lim,backbone_step=50000,action_calibration='none; original 50k weights',
        material='M1',nominal_successor='frozen source policy',new_plant_steps=0,new_optimizer_updates=0,**BG))
    for role in ('development','sealed_admission'):
        rows={k:[] for k in kinds}
        t=time.perf_counter()
        for state,records,snapshot in physical_rows(reg,role):
            key=state['snapshot_id']
            valid=all('cost' in r for r in records.values())
            if valid:
                clean=pickle.loads(gzip.decompress(Path(state['snapshot']).read_bytes()))
                x,table=starting_state(LARGE/job_name(state,role,'nominal1'),state,clean)
            for k in kinds:
                row=dict(state_id=key,stratum=state['stratum'],raw_branches_with_cost=sum('cost' in r for r in records.values()))
                if valid:
                    row['horizons']=horizon_comparisons(x,table,records,models[k],reg,'M1',state,allowed=allowed[k],normal_max=lim)
                rows[k].append(row)
            print(role,key,valid,round(time.perf_counter()-t,2),flush=True)
        for k in kinds:
            summary={}
            for n in NODES:
                graded=[dict(r,comparison=r['horizons'][str(n)]) if r.get('horizons',{}).get(str(n),{}).get('usable') else r for r in rows[k]]
                signed=summarize(graded)
                ranks=[dict(r,comparison=dict(r['comparison'],directions=[dict(d,gradient_correct=d['rank_correct']) for d in r['comparison']['directions']])) if 'comparison' in r else r for r in graded]
                signed.update(ordering=summarize(ranks),historical_pass=allowed[k][str(n)]['pass_gate'])
                summary[str(n)]=signed
            detail=raw/f'{role}-{k}.json.gz'
            with gzip.open(detail,'wt') as f:
                json.dump(rows[k],f,allow_nan=False)
            write(out/f'{role}-{k}.json',dict(runtime=runtime,role=role,kind=k,registered_states=len(reg['states'][role]),
                results_by_ticks=summary,wall_s_role_both_ensembles=time.perf_counter()-t,
                peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                details=dict(path=str(detail),bytes=detail.stat().st_size,sha256=hashlib.sha256(detail.read_bytes()).hexdigest()),**BG))


if __name__=='__main__':
    main()
