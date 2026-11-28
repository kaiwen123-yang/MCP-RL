"""W61 frozen-origin execution-model evaluation, with physical-time and event masks."""
import argparse
import gzip
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from s1_w59_model import ExecutionMember, advance_world, query_world
from s1_w59_model_fields import BG, DT, INTERVAL, PHASES, STATE
from s1_w59_model_open_loop import TIMES, rollpitch
from s1_w59_model_bank import write

REL = Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
FROZEN = ROOT.parent / 'MCP-RL-w59'
BANK = FROZEN / 'work/w59/model'
REGISTER = FROZEN / REL / 'w59/model/historical-origins.json'
WORK = ROOT / 'work/w61/material'
EVENT_FIELDS = ('work_J','slip_m','task_reward','touch','rebound','bearing_on','request')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_episode(member, source, step, device):
    path = BANK / 'historical-origins' / (source['episode'] + '.npz')
    with np.load(path) as d:
        x, a, world, rigid = [torch.as_tensor(d[key], device=device, dtype=torch.float32)
                              for key in ('state','action','world','rigid')]
        truth, intervals, origins = (d[key] for key in ('physical','interval','index'))
    phase = x[:, STATE['phase']].argmax(-1).cpu().numpy()
    active = torch.ones(len(x), dtype=torch.bool, device=device)
    sequence, events, masks = [x], [], [active]
    nominal = x.new_tensor([0., .05, 0.])
    for _ in range(round(max(TIMES)/DT)):
        nxt, y = step(member, query_world(x, world), a, rigid, active)
        advanced = advance_world(nxt, world, y[:, INTERVAL['load_mean']])
        world = torch.where(active[:,None,None], advanced, world)
        x = torch.where(active[:,None], query_world(nxt,world), x)
        sequence.append(x)
        events.append(torch.where(active[:,None],y,0.))
        a = torch.where(y[:,INTERVAL['request']] >= .5,nominal,a)
        active = active & (y[:,INTERVAL['hard_failure']].squeeze(-1)<.5) & (y[:,INTERVAL['complete']].squeeze(-1)<.5)
        masks.append(active)
    trajectory=torch.stack(sequence).cpu().numpy()
    intervals_hat=torch.stack(events).cpu().numpy()
    live=torch.stack(masks).cpu().numpy()
    rows=[]
    for j, sec in enumerate(TIMES):
        time_index=sec/DT
        lo,hi=int(np.floor(time_index+1e-7)),int(np.ceil(time_index-1e-7))
        fraction=time_index-lo
        for i, origin in enumerate(origins):
            rec=dict(episode=source['episode'], controller=source['controller'], origin_index=int(origin),
                     phase=PHASES[phase[i]], horizon_s=sec, has_observation=bool(np.isfinite(truth[i,j,0])),
                     prediction_active=bool(live[lo,i]))
            with np.errstate(invalid='ignore',over='ignore'):
                pred=(1-fraction)*trajectory[lo,i]+fraction*trajectory[hi,i]
            rec['finite']=bool(np.isfinite(pred).all())
            if rec['has_observation'] and rec['finite']:
                target=truth[i,j]
                mae=lambda a,b:float(np.mean(np.abs(a.astype('float64')-b)))
                rec.update(foot_xyz_MAE_m=mae(pred[STATE['feet']],target[:12]),
                           foot_fz_MAE_N=mae(pred[STATE['measured_force']].reshape(4,3)[:,2],target[12:16]),
                           roll_pitch_MAE_rad=mae(rollpitch(pred[STATE['qpos']][3:7].astype('float64')),target[16:18]),
                           height_abs_error_m=float(abs(pred[STATE['qpos']][2]-target[18])))
                if lo==hi:
                    values=intervals_hat[:lo,i].sum(0,dtype='float64')
                    for k in EVENT_FIELDS:
                        sl=INTERVAL[k]
                        if np.isfinite(values[sl]).all() and np.isfinite(intervals[i,j,sl]).all():
                            rec[k+'_integral_abs_error']=mae(values[sl],intervals[i,j,sl])
            rows.append(rec)
    return rows


def summary(rows):
    result=[]
    for h in TIMES:
        selected=[r for r in rows if r['horizon_s']==h]
        metrics={}
        for k in sorted(set().union(*(r.keys() for r in selected))):
            if not ('error' in k or 'MAE' in k):
                continue
            arr=[r[k] for r in selected if k in r]
            metrics[k]=dict(n=len(arr),mean=float(np.mean(arr)))
        result.append(dict(horizon_s=h, origins=len(selected), observed=sum(r['has_observation'] for r in selected),
                           predicted_active=sum(r['prediction_active'] for r in selected),
                           finite=sum(r['finite'] for r in selected),metrics=metrics))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['frozen50k','implicit','configuration','kinematics'],default='frozen50k')
    args=parser.parse_args()
    run='w61-a0-backbones-50k-v1' if args.variant=='frozen50k' else 'w61-a2-'+args.variant+'-v1'
    out=ROOT / REL / 'w61/material' / run
    raw=WORK/run
    raw.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    origins=json.loads(REGISTER.read_text())['episodes']
    source_info=[dict(episode=r['episode'], origins=r['n'], sha256=sha(BANK/'historical-origins'/(r['episode']+'.npz'))) for r in origins]
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    binary_index=FROZEN/REL/'w59/model/bank.json'
    roots={r['episode']:Path(r['root']) for r in json.loads(binary_index.read_text())['episodes']}
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(1)
    kinds=('blackbox','structured') if args.variant=='frozen50k' else ('structured',)
    prereg=dict(run=run,runtime=rev,variant=args.variant,frozen_registration=str(REGISTER),
        registration_sha256=sha(REGISTER),inputs=source_info,physics_dt_s=.0002,forecast_dt_s=DT,
        horizons_s=TIMES,seeds=[5901,5902,5903],kinds=kinds,origins=sum(r['n'] for r in origins),
        new_physical_transitions=0,optimizer_updates=0,torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),tf32=False,
        common_control='historical input and held macro; nominal successor after a predicted request; freeze state at predicted terminal',
        finite='all predicted state coordinates after the original interpolation; include frozen terminal states as in W59',
        by_phase_and_episode='each detail row has frozen controller, phase, whole episode, 20ms origin index and observed/active/finite flags',
        **BG)
    write(out/'inputs.json',prereg)
    for kind in kinds:
        for seed in prereg['seeds']:
            name=f'{kind}-{seed}'
            cp=BANK/'checkpoints'/kind/str(seed)/'node-50000.pt'
            start=time.perf_counter()
            cpu=time.process_time()
            member=ExecutionMember.load(cp,'cuda').eval()
            torch.cuda.reset_peak_memory_stats()
            rows=[]
            omitted=[]
            max_defect=0.
            with torch.inference_mode():
                for source in origins:
                    if args.variant in ('configuration','kinematics') and not (roots[source['episode']]/'model.mjb').is_file():
                        omitted.append(dict(episode=source['episode'], origins=source['n'], reason='original per-episode model binary unavailable on D57'))
                        continue
                    if args.variant=='frozen50k':
                        step=lambda m,x,a,d,active:m(x,a,d)
                    else:
                        from discrete import DiscreteExecution
                        step=DiscreteExecution(args.variant,roots[source['episode']]/'model.mjb')
                    rows.extend(score_episode(member,source,step,'cuda'))
                    max_defect=max(max_defect,getattr(step,'torque_defect_max',0.))
            detail=raw/(name+'.jsonl.gz')
            with gzip.open(detail,'wt') as f:
                for r in rows:
                    f.write(json.dumps(r,allow_nan=False)+'\n')
            write(out/(name+'.json'),dict(run=run,kind=kind,seed=seed,backbone_step=50000,
                checkpoint=str(cp),checkpoint_sha256=sha(cp),runtime=rev,per_horizon=summary(rows),
                registered_origins=sum(r['n'] for r in origins),omitted_sources=omitted,
                max_projected_torque_equation_defect_Nm=max_defect if args.variant!='frozen50k' else None,
                differentiation='inference only; configuration and FK use detached CPU MuJoCo, no path-gradient claim',
                wall_s=time.perf_counter()-start,cpu_core_s=time.process_time()-cpu,
                peak_allocated_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,
                peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                raw=dict(path=str(detail),bytes=detail.stat().st_size,sha256=sha(detail)),**BG))
            print(name,len(rows),time.perf_counter()-start,flush=True)


if __name__=='__main__':
    main()
