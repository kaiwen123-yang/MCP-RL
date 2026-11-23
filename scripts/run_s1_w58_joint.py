"""One W58 P2 + timing head learns from its two actual, atomically leased plant streams."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
from inspect_s1_w58_predecessor import REL, BG
R=ROOT/'results'/REL/'w58'
WORK=ROOT/'work/w58/campaigns'


def read(path):
    return json.loads(path.read_text())


def paths(seed,head):
    run=WORK/f'seed-{seed}'/head
    return run,R/'learning'/f'seed-{seed}'/head


def collect(a):
    from s1_w57_learning import batch_path,read_rows,jsonline,STEPS,close_interrupted_sequence
    run,_=paths(a.seed,a.head)
    scope=read(run/'run.json')
    case=a.case
    fds=tuple(map(int,os.environ['S1_W58_LEASE_FDS'].split(',')))
    ledger=run/f'episodes-{case}.jsonl'
    ends=run/f'episode-ends-{case}.jsonl'
    existing=read_rows(ledger)
    if len(existing)!=len(read_rows(ends)):
        raise RuntimeError(f'An unclosed real collector attempt requires reconciliation at {ledger}')
    episode=len(existing)
    while True:
        state=read(run/'published.json')
        if state['finished']:return
        v=state['version']
        if (run/f'ready-{v:03}-{case}.json').exists():
            time.sleep(1)
            continue
        before=STEPS*v+len(read_rows(batch_path(run,v,case)))
        episode+=1
        name=f'w58-j{a.seed}-{a.head}-{case.lower()}-e60-{episode:04d}'
        cmd=[sys.executable,'-u','scripts/run_s1_w58.py','--attempt',name,'--case',case,'--arm',scope['arm'],
             '--gait',a.gait,'--material',a.material,'--split','training','--task','training','--level','E60',
             '--learning-seed',str(a.seed),'--timing-config',str(run/f'collector-{case}.json')]
        jsonline(ledger,dict(attempt=name,case=case,version_at_start=v,unix_s=time.time(),command=cmd,**BG))
        code=subprocess.call(cmd,cwd=ROOT,pass_fds=fds)
        path=R/'attempts'/(name+'.json')
        row=read(path) if path.is_file() else {}
        jsonline(ends,dict(attempt=name,status=row.get('status'),code=code,simulation_s=row.get('simulation_s'),
                           reason=row.get('reason'),unix_s=time.time()))
        if code or row.get('status')=='technical_interruption':
            close_interrupted_sequence(run,case,name,row.get('reason') or 'physical_process_interrupted')
            raise RuntimeError(f'Collector {case} stopped at {name}: {row.get("reason")}')
        new=read(run/'published.json')
        if not new['finished'] and STEPS*new['version']+len(read_rows(batch_path(run,new['version'],case)))==before:
            raise RuntimeError('Natural episode had no new distinct decisions: '+name)


def run_head(a):
    import torch
    from run_s1_w55_f import digest
    from publish_s1_go2_w54 import write
    from s1_w57_learning import (SPEC, SPEC_HASH, STEPS, common_initialization, checkpoint_path,
                                 train_batch, read_rows, batch_path)
    from s1_w57_training_evidence import node_evidence
    run,compact=paths(a.seed,a.head)
    run.mkdir(parents=True,exist_ok=True)
    compact.mkdir(parents=True,exist_ok=True)
    if not (R/'development-decision.json').is_file() or not read(R/'development-decision.json')['joint_eligible']:
        raise RuntimeError('The physics development comparison has not earned joint acquisition')
    arm='P2-full' if a.head=='full' else 'P2-private'
    common=run.parent/'common-initialization.pt'
    if not common.exists():
        if a.head!='full':raise FileNotFoundError(common)
        common_initialization(common)
    cap=(8192 if a.head=='full' else read(paths(a.seed,'full')[1]/'final.json')['distinct'])
    identity=dict(seed=a.seed,head=a.head,gait=a.gait,material=a.material,arm=arm,
        code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        spec_sha256=SPEC_HASH,development_decision_sha256=digest(R/'development-decision.json'),
        common_initialization_sha256=digest(common),declared_ceiling=cap)
    if (run/'run.json').exists():
        if read(run/'run.json')['identity']!=identity:raise ValueError('Seed/head identity changed')
    else:
        model=torch.load(common,map_location='cpu',weights_only=False)
        if (model['learning_seed'],model['samples'],model['spec_sha256'])!=(a.seed,0,SPEC_HASH):
            raise ValueError('Seed initialization provenance differs')
        model.update(head=a.head,identity=identity)
        torch.save(model,checkpoint_path(run,0))
        for case in 'AB':
            write(run/f'collector-{case}.json',dict(mode='collect',case=case,head=a.head,
                                                    learning_seed=a.seed,run_dir=str(run)))
        write(run/'run.json',dict(arm=arm,head=a.head,identity=identity,rollout=[128,2],**BG))
        write(run/'published.json',dict(version=0,finished=False,cumulative_distinct=0))
        write(compact/'identity.json',identity)
    leases=json.loads(os.environ['S1_W58_PAIR_CPUS'])
    if len(leases)!=2 or set(leases[0])&set(leases[1]):raise ValueError('Two disjoint collector leases required')
    children={}
    frozen=read(run/'published.json')
    if frozen.get('incomplete'):raise RuntimeError('Previous physical or learner interruption requires diagnosis')
    failure=None
    try:
        if not frozen['finished']:
            for case,cores in zip('AB',leases):
                cmd=['taskset','-c',','.join(map(str,cores)),sys.executable,'-u',__file__,
                     'collector','--seed',str(a.seed),'--head',a.head,'--case',case,
                     '--gait',a.gait,'--material',a.material]
                with (run/f'collector-{case}.log').open('a') as out:
                    children[case]=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
                        pass_fds=tuple(map(int,os.environ['S1_W58_LEASE_FDS'].split(','))))
        state=read(run/'published.json')
        while not state['finished']:
            version=state['version']
            while not all((run/f'ready-{version:03}-{c}.json').is_file() for c in 'AB'):
                if any(p.poll() is not None for p in children.values()):
                    raise RuntimeError('A collector ended before its current batch closed')
                time.sleep(1)
            weight=checkpoint_path(run,version+1)
            update=run/f'update-{version+1:03}.json'
            metrics=read(update) if weight.is_file() and update.is_file() else train_batch(
                run,version,a.head,identity,'cuda')
            write(compact/update.name,metrics)
            v=version+1
            n=v*2*STEPS
            finished=n>=cap
            if n in SPEC['nodes']:
                evidence,matches=node_evidence(run,v,True)
                if a.head=='full':
                    finished=finished or not evidence['eligible']
                evidence.update(checkpoint_sha256=digest(weight),source_checkpoint=str(weight),
                                version=v,seed=a.seed,continue_same_identity=not finished,
                                selection_uses='training episodes only')
                write(compact/f'node-{n:05d}.json',evidence)
                with gzip.open(run/f'node-{n:05d}-matched-loads.jsonl.gz','wt') as out:
                    for row in matches:out.write(json.dumps(row)+'\n')
            state=dict(version=v,finished=finished,cumulative_distinct=n,checkpoint=str(weight))
            write(run/'published.json',state)
    except Exception as error:
        failure=repr(error)
        state=read(run/'published.json')
        # Let the partner finish its existing episode on the last legal policy.
        # A failed pair is retained with its own partial ledger and cannot qualify a head.
        write(run/'published.json',dict(state,finished=True,incomplete=True,reason=failure))
    finally:
        exits={case:p.wait() for case,p in children.items()}
    state=read(run/'published.json')
    n=state['cumulative_distinct']
    total_partial={c:len(read_rows(batch_path(run,state['version'],c))) for c in 'AB'}
    final=dict(**identity,distinct=n,partial=total_partial,collector_exit=exits,
               after_natural_tail_unix_s=time.time(),failure=failure,
               complete=not failure and not any(exits.values()) and n in SPEC['nodes'],**BG)
    if final['complete']:
        weight=checkpoint_path(run,state['version'])
        final.update(checkpoint=str(weight),checkpoint_sha256=digest(weight))
        write(run/'frozen-evaluation.json',dict(mode='evaluate',head=a.head,learning_seed=a.seed,checkpoint=str(weight)))
    write(compact/'final.json',final)
    if not final['complete']:raise RuntimeError(str(final))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('head','collector'))
    p.add_argument('--seed',type=int,choices=(5801,5802,5803),required=True)
    p.add_argument('--head',choices=('full','blind'),required=True)
    p.add_argument('--case',choices=('A','B'))
    p.add_argument('--gait',required=True)
    p.add_argument('--material',required=True)
    a=p.parse_args()
    os.environ['S1_W58_JOINT_SEED']=str(a.seed)
    (collect if a.action=='collector' else run_head)(a)

