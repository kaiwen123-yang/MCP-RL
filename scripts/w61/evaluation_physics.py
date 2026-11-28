"""Physical-only entry points for the frozen W61 native and curve evaluators."""
import argparse
from pathlib import Path
import time

import evaluate_learning as native
from campaign import RUN, SMALL, STUDY, head_name
import campaign
from s1_w59_learner_critic import atomic_json, file_digest

QUEUE=RUN/'render-queue'
INTENTS=QUEUE/'items'
POOL=Path('/root/autodl-tmp/MCP-RL-w61-joint/work/w61/pool/jobs')


def write_new(path,data):
    if path.exists():
        old=native.read(path)
        for field in ('id','lease','source_file','actor_sha256'):
            if field in data and old.get(field)!=data[field]:raise ValueError(f'Changed {field}: {path}')
        return
    atomic_json(path,data)


def put(kind,name,lease,raw,result,receipt,actor,**metadata):
    write_new(INTENTS/(name+'.json'),dict(id=name,kind=kind,lease=lease,raw=str(raw),source_file=str(raw/'trajectory.npz'),
       physical_result=str(result),physical_receipt=str(receipt),actor_sha256=actor,registered_unix=time.time(),**metadata))


def native_closure(method,seed,version,case,kind,condition):
    name=native.ident(method,seed,version,case,kind,condition);head=head_name(method,seed)
    folder=native.LEDGER/head;out=native.DATA/name
    record=folder/(name+'.json');resolved=folder/(name+'-resolved.json');receipt=folder/(name+'-physical.json')
    r=native.read(record);c=native.read(resolved)['r12_configuration'];freeze=c['w59']['w61_final']
    if (freeze['method'],freeze['learning_seed'],freeze['version'],freeze['case'])!=(method,seed,version,case):
        raise ValueError(f'Frozen case identity changed: {name}')
    if r['case_key']!=c['case_key']:raise ValueError(f'Original evaluator case key mismatch: {name}')
    if r.get('physical_samples',0) and not (out/'trajectory.npz').is_file():raise ValueError('Missing closed native trajectory')
    rec=dict(id=name,head=head,case=case,stage=kind,condition=condition,actor_sha256=freeze['actor_file_sha256'],
        stage_status=r['stage_status'],qualification_usable=r.get('qualification_usable',condition=='nominal'),
        case_key=c['case_key'],physical_s=r['simulation_s'],return_credit=(r.get('w61_frozen_actor') or {}).get('original_task_credit'),
        task_success=r.get('success'),qualification=r.get('qualification'),mechanism=r.get('mechanism'),raw=str(out),
        physical_samples=r.get('physical_samples',0),used_training_events=0,
        predecessor_attempt=c.get('predecessor_attempt'),physics_wall_s=r.get('wall_s'),physical_started_unix=r.get('physical_started_unix_s'),
        final_integration_unix=r.get('physical_ended_unix_s'),physical_closed_unix=record.stat().st_mtime,
        native_evaluator_sha256=file_digest(native.SELF),native_backend_sha256=file_digest(native.BACKEND))
    write_new(receipt,rec)
    if r.get('physical_samples',0) and not (folder/(name+'-summary.json')).is_file():
        put('native',name,name,out,record,receipt,rec['actor_sha256'],head=head,stage=kind,case=case,condition=condition)
    elif not r.get('physical_samples',0):
        atomic_json(folder/(name+'-technical.json'),dict(attempt=name,status='technical_interruption',used=0,
           reason='Closed native evaluator recorded zero physical force samples',raw=str(out)))
    return rec


class FinishedPhysics(Exception):
    pass


def physical_native(method,seed,version,case,kind,condition):
    name=native.ident(method,seed,version,case,kind,condition)
    root=native.LEDGER/head_name(method,seed)
    if (root/(name+'.json')).is_file():
        return native_closure(method,seed,version,case,kind,condition)
    # Capture the boundary after the unchanged native run, its finally block closes
    # the actor and learning streams. No renderer imports or draws inside this lease.
    original=native.run
    def finished(*args,**kwargs):
        original(*args,**kwargs)
        raise FinishedPhysics
    native.run=finished
    try:
        try:native.native(method,seed,version,case,kind,condition)
        except FinishedPhysics:pass
    finally:native.run=original
    return native_closure(method,seed,version,case,kind,condition)


def all_trained_curves_closed(suffix):
    lifecycle=SMALL/'training-lifecycle.json'
    if not lifecycle.is_file():return False
    heads=native.read(lifecycle)['heads']
    return all((SMALL/head/f'evaluate-{v:03d}{suffix}.json').is_file()
               for head,state in heads.items() for v in (8,16,32) if v<=state['version'])


def curve_id(method,seed,version):
    return f'w61-{STUDY}-{head_name(method,seed)}-evaluate-v{version:03d}'


def curve_closure(method,seed,version):
    head=head_name(method,seed);lease=curve_id(method,seed,version)
    receipt=SMALL/head/f'evaluate-{version:03d}.json'
    r=native.read(receipt);actor=r['checkpoint_sha']
    for case in r['cases']:
        for ep in case.get('episodes',[]):
            if not ep.get('physical_steps'):continue
            n=ep['run'];raw=RUN/head/f'evaluate-{version:03d}-{case["case"]}'/n
            put('curve',n,lease,raw,raw/'result.json',receipt,actor,head=head,method=method,seed=seed,
                version=version,case=case['case'])
        if case.get('status')=='technical' and case.get('raw'):
            raw=Path(case['raw']);result=raw/'result.json'
            if result.is_file() and (raw/'trajectory.npz').is_file() and native.read(result).get('physical_steps'):
                put('curve',raw.name,lease,raw,result,receipt,actor,head=head,method=method,seed=seed,
                    version=version,case=case['case'])
    return r


def physical_curve(method,seed,version):
    receipt=SMALL/head_name(method,seed)/f'evaluate-{version:03d}.json'
    if receipt.is_file():
        curve_closure(method,seed,version);return 0
    import render as frozen_render
    previous=frozen_render.render
    # This hook is only used by campaign.case_rollouts after collect has closed
    # its physical files, outside the generation of any observations or actions.
    frozen_render.render=lambda out:None
    try:
        rc=campaign.paired(method,seed,version,'evaluate')
    finally:frozen_render.render=previous
    if receipt.exists():curve_closure(method,seed,version)
    return rc


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['native','curve','native-recover'])
    p.add_argument('method');p.add_argument('seed',type=int);p.add_argument('version',type=int);p.add_argument('parameters',nargs='*')
    a=p.parse_args()
    if a.mode=='curve':raise SystemExit(physical_curve(a.method,a.seed,a.version))
    (physical_native if a.mode=='native' else native_closure)(a.method,a.seed,a.version,*a.parameters)
