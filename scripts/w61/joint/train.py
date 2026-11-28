"""Paired real plant queries, block controls and simultaneous adaptation-control training."""
import argparse, concurrent.futures, hashlib, json, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
from w61.joint.heads import NA,NC
from w61.joint.plant import CONTRACT,ROOT,rollout,write


def result(path,theta,case,rng,arm):
    completed=path/'result.json'
    if completed.exists():
        obj=json.loads(completed.read_text())
        if obj['theta']!=theta.tolist() or obj['arm']!=arm or obj['noise_seed']!=rng or obj['case']!=case:
            raise ValueError('Completed observation belongs to a different candidate')
        return obj
    return rollout(path,theta,case,rng,arm)


def work(args):
    cfg=json.loads(CONTRACT.read_text())
    job=cfg['study']
    dest=args.output/args.arm/str(args.seed)
    dest.mkdir(parents=True,exist_ok=True)
    dim=NA+NC
    rng=np.random.default_rng(np.random.SeedSequence([args.seed,13,args.round]))
    if args.round==0:
        theta=np.random.default_rng(np.random.SeedSequence([args.seed,7])).normal(0,.08,dim)
    else:
        prev=dest/f'round-{args.round-1:02d}.json'
        theta=np.array(json.loads(prev.read_text())['theta_after'])
    if args.round==job['rounds']:
        scores=[]
        for j,case in enumerate(cfg['unseen']):
            outcome=result(dest/f'eval-{j}',theta,case,715620+j,'joint')
            scores.append({k:outcome[k] for k in ('joint_cost','physical_cost','completed','reason','progress_m',
                  'query_steps','calibration_rmse_N','mechanical_abs_work_J','slip_m','vertical_foot_force_p95_N')})
        write(dest/'evaluation.json',dict(seed=args.seed,arm=args.arm,frozen_theta=theta.tolist(),cases=scores))
        return
    if args.arm=='simultaneous':block='AC'
    elif args.arm=='A_then_C':block='A' if args.round<job['rounds']//2 else 'C'
    else:block='A' if args.round%2==0 else 'C'
    case=cfg['train'][args.round%len(cfg['train'])]
    d=rng.normal(size=dim)
    if block=='A':d[NA:]=0.
    if block=='C':d[:NA]=0.
    initial=theta.copy()
    ref=np.random.default_rng(np.random.SeedSequence([args.seed,7])).normal(0,.08,dim)
    if args.arm=='independent':
        if block=='C':initial[:NA]=ref[:NA]
        else:initial[NA:]=ref[NA:]
    query_rng=int(rng.integers(2**31))
    signed=[]
    for sign,name in ((1,'plus'),(-1,'minus')):
        # Gaussian smoothing is over the unconstrained head; tanh inside the heads
        # defines the physical trust envelope. Both signs execute their own plant.
        trial=initial+sign*job['sigma']*d
        outcome=result(dest/f'r{args.round:02d}-{name}',trial,case,query_rng,'joint')
        if outcome['calibration_rmse_N'] is None:raise RuntimeError('Fixed support failed before any candidate could act')
        if args.arm=='independent':
            value=(outcome['calibration_rmse_N']/40.)**2 if block=='A' else outcome['physical_cost']
        elif args.arm=='A_then_C' and block=='A':
            value=(outcome['calibration_rmse_N']/40.)**2
        else:value=outcome['joint_cost']
        signed.append(dict(sign=sign,objective=value,result=str((dest/f'r{args.round:02d}-{name}'/'result.json').relative_to(args.output)),
            support_digest=digest_support(dest/f'r{args.round:02d}-{name}'/'support.npz'),
            physical_cost=outcome['physical_cost'],real_query_steps=outcome['query_steps'],completed=outcome['completed'],
            mp_s=outcome['physical_s'],rmse_N=outcome['calibration_rmse_N'],reason=outcome['reason'],wall_s=outcome['wall_s']))
    if signed[0]['support_digest']!=signed[1]['support_digest']:
        raise ValueError('Paired candidates did not observe identical fixed-policy support')
    g=(signed[0]['objective']-signed[1]['objective'])*d/(2*job['sigma'])
    delta=-job['step_size']*g
    delta*=min(1.,job['max_block_delta']/max(1.e-15,np.linalg.norm(delta)))
    updated=np.clip(theta+delta,-job['theta_bound'],job['theta_bound'])
    write(dest/f'round-{args.round:02d}.json',dict(arm=args.arm,seed=args.seed,round=args.round,
          scenario=case,query_seed=query_rng,block=block,direction=d.tolist(),theta_before=theta.tolist(),
          theta_after=updated.tolist(),signed=signed,update_norm_A=float(np.linalg.norm((updated-theta)[:NA])),
          update_norm_C=float(np.linalg.norm((updated-theta)[NA:])),real_queries=len(signed),
          source=os.environ['W61_F_RUNTIME_COMMIT']))


def digest_support(path):
    # Compare the physical arrays, independent of ZIP headers.
    h=hashlib.sha256()
    with np.load(path) as a:
        for k in sorted(a.files):h.update(k.encode());h.update(a[k].tobytes())
    return h.hexdigest()


def pair_controls(args):
    spec=json.loads(CONTRACT.read_text())
    for j,case in enumerate(spec['unseen']):
        path=args.output/'factorial'/str(j)/args.arm
        path.parent.mkdir(parents=True,exist_ok=True)
        outcome=result(path,np.zeros(NA+NC),case,715620+j,args.arm)
        print(args.arm,j,outcome['joint_cost'],flush=True)


def schedule(args):
    spec=json.loads(CONTRACT.read_text());j=spec['study']
    jobs=[]
    # Finite rounds are interleaved between arms. Each worker waits for the
    # previous round of its own head before claiming another physical slot.
    for s in j['training_seeds']:
        for a in j['arms']:jobs.append((a,s))
    def command(arm,seed,round_id,verb):
        name=f'f-ac2-{arm}-{seed}-{round_id}-{verb}'
        cmd=[sys.executable,str(args.lease),'--runtime',str(ROOT),'--job',name,'--slots','1','--cores','1',
             '--ram-gib','1.15','--scratch-gib','.10','--',sys.executable,'-m','w61.joint.train',
             verb,'--output',str(args.output),'--seed',str(seed),'--arm',arm,'--round',str(round_id)]
        rc=subprocess.call(cmd)
        if rc:raise RuntimeError(f'{name} exited {rc}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        # Two genuine single-collector tasks, each independently leased; plus/minus
        # always run sequentially in its own lease and never use another arm's data.
        for rid in range(j['rounds']+1):
            pending={ex.submit(command,arm,seed,rid,'work'):(arm,seed) for arm,seed in sorted(jobs,key=lambda x:(j['arms'].index(x[0]),x[1]))}
            errors=[]
            for future in concurrent.futures.as_completed(pending):
                try:future.result()
                except RuntimeError as e:
                    errors.append(str(e));jobs.remove(pending[future])
            if errors:
                write(args.output/f'interruptions-{rid}.json',errors)
    for a in j['fixed_pair_controls']:command(a,0,0,'factorial')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('verb',choices=('work','schedule','factorial'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lease',type=Path);p.add_argument('--arm');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--round',type=int,default=0)
    args=p.parse_args()
    {'work':work,'schedule':schedule,'factorial':pair_controls}[args.verb](args)
