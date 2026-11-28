"""Fit W61 physically acting execution residuals only on unique complete historical A transitions."""
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from s1_w59_model_fields import STATE,INTERVAL
from response_model import Response,features

R='w61-a2-physical-response-v1'
BANK=ROOT.parent/'MCP-RL-w59/work/w59/model'
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
INDEX=ROOT.parent/'MCP-RL-w59'/REL/'w59/model/bank.json'
RAW=ROOT/'work/w61/material'/R
OUT=ROOT/REL/'w61/material'/R
SPEC=ROOT/'experiments/terrain_memory/w61/material/response-v1.json'


def write(p,d):
    tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def data():
    blocks=[[] for _ in range(5)]
    seen=set()
    inventory=[]
    for row in json.loads(INDEX.read_text())['episodes']:
        if row['split']!='fit' or 'intervals' not in row:continue
        ep=row['episode']
        target=BANK/'aligned'/(ep+'.npz')
        rigid=BANK/'rigid'/(ep+'.npy')
        if not rigid.is_file():continue
        with np.load(target) as z:
            valid=z['valid'][:,INTERVAL['load_mean']].all(1) & (z['state'][:-1,STATE['control_valid']].ravel()>.5) & (z['state'][1:,STATE['control_valid']].ravel()>.5)
            ids=[]
            for i in np.flatnonzero(valid):
                h=int(z['fingerprint'][i])
                if h not in seen:seen.add(h);ids.append(i)
            ids=np.array(ids,dtype=int)
            if len(ids):
                st=z['state'];act=z['action'];normal=z['interval'][:,INTERVAL['load_mean']]
                for dst,arr in zip(blocks,(st[ids],st[ids+1],act[ids],normal[ids],np.load(rigid,mmap_mode='r')[ids])):
                    dst.append(arr)
            inventory.append(dict(episode=ep,unique_valid=len(ids),source=str(target),
                                  sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    return [torch.tensor(np.concatenate(b),device='cuda',dtype=torch.float32) for b in blocks],inventory


def main():
    cfg=json.loads(SPEC.read_text());torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    arrays,inputs=data();x,nxt,a,normal,rigid=arrays
    feat=features(x,a);m,s=feat.mean(0),feat.std(0).clamp_min(.05)
    vscale=nxt[:,STATE['qvel']].std(0).clamp_min(.25)
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(OUT/'training-inputs.json',dict(runtime=rev,bank=str(INDEX),sources=inputs,
          unique_valid_fit_transitions=len(x),config_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(),
          normalizer='fit only; serialised inside every model',used_validation_for_selection=False))
    for seed in cfg['seeds']:
        rng=torch.Generator('cuda').manual_seed(seed);torch.manual_seed(seed)
        model=Response(m,s).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1.e-4)
        wall=time.perf_counter();cpu=time.process_time();history=[]
        torch.cuda.reset_peak_memory_stats()
        for n in range(1,cfg['updates']+1):
            idx=torch.randint(len(x),(cfg['batch'],),generator=rng,device='cuda')
            estimate,force,tau=model(x[idx],a[idx],rigid[idx])
            nexts=nxt[idx]
            losses=[F.smooth_l1_loss(estimate[:,STATE['qvel']]/vscale,nexts[:,STATE['qvel']]/vscale),
                    F.smooth_l1_loss(estimate[:,STATE['held_command']]/.08,nexts[:,STATE['held_command']]/.08),
                    F.smooth_l1_loss(estimate[:,STATE['measured_force']]/35,nexts[:,STATE['measured_force']]/35),
                    F.smooth_l1_loss(force[:,:,2]/35,normal[idx]/35)]
            loss=sum(losses)
            opt.zero_grad();loss.backward();opt.step()
            if n%250==0:history.append(dict(update=n,components=[float(y.detach().item()) for y in losses]))
        target=RAW/f'physical-response-{seed}.pt'
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=rng.get_state(),
            seed=seed,updates=n,spec=cfg,source=rev),target)
        write(OUT/f'fit-{seed}.json',dict(runtime=rev,seed=seed,updates=n,batch=cfg['batch'],presentations=n*cfg['batch'],
            distinct_source=len(x),sample_unit='unique physical 20ms window preselected by archived fingerprint; reused across optimizer draws',
            curves=history,wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu,
            GPU_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,RSS_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            checkpoint=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print('fit',seed,n,len(x),flush=True)


if __name__=='__main__': main()
