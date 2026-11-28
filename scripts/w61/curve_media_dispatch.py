"""Consume frozen checkpoint notices and publish completed curve videos without training dependencies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import evaluate_learning as native
from evaluation_physics import POOL, QUEUE, INTENTS, curve_closure, curve_id, all_trained_curves_closed
from campaign import RUN, SMALL, STUDY, head_name
from develop import admitted_call
from hybrid import ROOT, SPEC
from publish_learning import setup, publisher, verified, DELIVERY
from s1_w59_learner_critic import atomic_json, file_digest

HERE=Path(__file__).resolve().parent
DONE=QUEUE/'rendered'
READY=QUEUE/'curve-ready'


def assignment(node):
    m,s,v=node['method'],int(node['seed']),int(node['version']);h=head_name(m,s)
    if m not in [*SPEC['methods'],SPEC['geometry_ablation']] or s not in SPEC['seeds'] or v not in (8,16,32):
        raise ValueError('Unregistered curve head or node')
    if node['study']!=STUDY or node['used']!=v*256:
        raise ValueError('Curve notice differs from registered distinct event target')
    ck=RUN/h/f'policy-{v:03d}.pt';trained=native.read(SMALL/h/f'train-{v-1:03d}.json')
    expected=node.get('checkpoint_sha256',node.get('updated_checkpoint_sha256'))
    if trained.get('status')!='updated' or trained.get('used')!=node['used'] or file_digest(ck)!=expected or trained['updated_checkpoint_sha256']!=expected:
        raise ValueError('Exact immutable checkpoint and training receipt required')
    return m,s,v,h


def physical(job):
    method,seed,version,head=job
    cmd=[sys.executable,'-u',str(HERE/'evaluation_physics.py'),'curve',method,str(seed),str(version)]
    return admitted_call(curve_id(method,seed,version),2,cmd,ram=5.5,gpu=1500,disk=1.0)


def run_scheduler():
    READY.mkdir(parents=True,exist_ok=True)
    with (QUEUE/'curve-manager.lock').open('a') as lock, ThreadPoolExecutor(max_workers=1) as pool:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        future=None;current=None;observed=set()
        while True:
            if future is not None and future.done():
                rc=future.result()
                m,s,v,h=current;receipt=SMALL/h/f'evaluate-{v:03d}.json'
                if rc and not receipt.exists():
                    atomic_json(QUEUE/'curve-technical'/(curve_id(m,s,v)+'.json'),dict(job=curve_id(m,s,v),exit_code=rc))
                future=None
            for n in sorted(READY.glob('*.json'),key=lambda p:p.stat().st_mtime):
                if n.name in observed:continue
                item=assignment(native.read(n));m,s,v,h=item
                rec=SMALL/h/f'evaluate-{v:03d}.json';name=curve_id(m,s,v)
                if rec.exists():
                    curve_closure(m,s,v);observed.add(n.name);continue
                lease=POOL/(name+'.json')
                if lease.exists():
                    old=native.read(lease)
                    if old['state']=='ended' and not rec.exists():
                        # A real collection failure stays visible; never create a replacement episode.
                        atomic_json(QUEUE/'curve-technical'/(name+'.json'),dict(job=name,exit_code=old['exit_code'],
                            physical_receipt=None,reason='inspect existing original collector and logs'))
                        observed.add(n.name)
                    continue
                if future is None:
                    current=item;future=pool.submit(physical,item)
            if future is None and all_trained_curves_closed(''):return
            time.sleep(5)


def published_for_case(h,v,case,id,media,receipt):
    label=f'w61-E-{STUDY}-{h}-frozen-{v*256:05d}-{case}'
    raw=Path(media['video']).parent
    if not verified(label):
        setup()
        publisher.publish(label,[('episode',raw),('evaluation.json',receipt),('media-clock.json',DONE/(id+'.json'))],Path(media['video']))
    r=native.read(DELIVERY/(label+'.json'))
    video_asset=next(a for a in r['assets'] if a['name'].endswith('.mp4'))
    if (video_asset['size'],video_asset['digest'])!=(media['bytes'],'sha256:'+media['video_sha256']):
        raise ValueError('Remote curve movie and off-pool render differ')
    return dict(label=label,receipt=str(DELIVERY/(label+'.json')),video=video_asset,verified=r['status']=='verified')


def close_media(node):
    m,s,v,h=assignment(node)
    folder=SMALL/h;physics=folder/f'evaluate-{v:03d}.json';aggregate=folder/f'evaluate-{v:03d}-media.json'
    if aggregate.is_file():return
    if not physics.is_file():
        issue=QUEUE/'curve-technical'/(curve_id(m,s,v)+'.json')
        if not issue.exists():return
        roots=[RUN/h/f'evaluate-{v:03d}-{case}' for case in SPEC['scenarios']]
        original=[p for d in roots for p in d.glob('*/trajectory.npz')]
        if original:
            atomic_json(QUEUE/'curve-review'/(curve_id(m,s,v)+'.json'),dict(issue=str(issue),
               preserved_trajectories=[str(p) for p in original],reason='no canonical physical receipt; inspect original, no new simulation'))
            return
        atomic_json(physics,dict(study=STUDY,method=m,seed=s,version=v,role='evaluate',used=0,
            cases=[],status='technical_or_starved',checkpoint_sha=node.get('checkpoint_sha256',node.get('updated_checkpoint_sha256')),
            reason='closed lease without a renderable physical trajectory',lease_failure=native.read(issue)))
    result=curve_closure(m,s,v);rows=[]
    for c in result['cases']:
        names=[e['run'] for e in c.get('episodes',[]) if e.get('physical_steps')]
        if c.get('status')=='technical' and c.get('raw'):
            raw=Path(c['raw'])
            if (raw/'result.json').is_file() and native.read(raw/'result.json').get('physical_steps'):names.append(raw.name)
        for name in dict.fromkeys(names):
            p=DONE/(name+'.json')
            if not p.is_file():return
            rec=native.read(p);delivery=published_for_case(h,v,c['case'],name,rec,physics)
            rows.append(dict(case=c['case'],attempt=name,render_receipt=str(p),render_wall_s=rec['render_wall_s'],
                  render_started_unix=rec['render_started_unix'],render_closed_unix=rec['render_closed_unix'],
                  physical_closed_unix=rec['physical_closed_unix'],physics_wall_s=rec['physics_wall_s'],
                  child_exited_unix=rec['child_exited_unix'],slots_released_unix=rec['slots_released_unix'],
                  lease_released_observed_unix=rec['lease_released_observed_unix'],
                  lease_ended=rec['lease_ended'],pts=rec['pts'],video_sha256=rec['video_sha256'],verified=delivery))
    status='verified' if rows else 'technical_no_renderable_trajectory'
    atomic_json(aggregate,dict(head=h,version=v,used=0,checkpoint_sha256=result['checkpoint_sha'],physics_status=result['status'],
              status=status,media=rows,closed_unix=time.time()))
    git=['git','-C',str(ROOT)]
    names=[str(p.relative_to(ROOT)) for p in [physics,aggregate,*[Path(r['verified']['receipt']) for r in rows]]]
    subprocess.run([*git,'add','--sparse','--',*names],check=True)
    subprocess.run([*git,'commit','--only','-qm','Deliver closed frozen curve physics and independently verified media','--',*names],check=True)


def publish_loop():
    QUEUE.mkdir(exist_ok=True,parents=True)
    with (QUEUE/'curve-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        retry={}
        while True:
            for n in sorted(READY.glob('*.json')):
                if retry.get(n,0)>time.time():continue
                try:close_media(native.read(n))
                except (OSError,subprocess.SubprocessError,RuntimeError) as err:
                    atomic_json(QUEUE/'publish-pending'/n.name,dict(when=time.time(),error=str(err)))
                    retry[n]=time.time()+90
            if all_trained_curves_closed('-media'):return
            time.sleep(6)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['physics','publish']);a=p.parse_args()
    (run_scheduler if a.action=='physics' else publish_loop)()
