"""Continue the one registered W59 action lane through a frozen recipe and independent admission."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from publish_s1_go2_w54 import write
from run_s1_w59_interventions import ROOT, SMALL, LARGE, immutable, sha
from register_s1_w59_independent_requests import register as independent_sources

OUT=SMALL/'action'
REG=OUT/'registry-v2.json'
DEV=OUT/'development-scored.json'
P3=OUT/'p3-development-recipe.json'
FREEZE=OUT/'model-recipe-freeze.json'
V3=OUT/'registry-v3.json'
SEALED=OUT/'sealed-scored.json'
PREFIX='w59-p3-full-m1-'
SEEDS=(5901,5902,5903)
BG=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)


def read(p):return json.loads(p.read_text())


def status(state,**detail):
    row=dict(state=state,**detail,**BG)
    path=OUT/'stage.json'
    if path.is_file() and {k:v for k,v in read(path).items() if k!='changed_unix_s'}==row:return
    write(path,dict(**row,changed_unix_s=time.time()))
    print(json.dumps(row),flush=True)


def run(script,*args,cwd=ROOT):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=cwd,check=True)


def checkpoints():
    return [ROOT/'work/w59/model/action-checkpoints/structured'/f'{s}-node-10000.pt' for s in SEEDS]


def closed(role):
    paths=list((SMALL/'episodes').glob('w59-action-'+role+'-*-a01.json'))
    return len(paths),sum(read(p).get('stage_status')=='technical_interruption' for p in paths)


def score(reg,role,paths,target,recipe=None):
    if target.exists():return
    args=['--registry',reg,'--role',role,'--material','M1','--output',target]
    for p in paths:args+=['--wm-checkpoint',p]
    if recipe:args+=['--wm-recipe',recipe]
    run('score_s1_w59_interventions.py',*args)


def original(runtime,case,paths):
    for level in ('E30','E60'):
        name=PREFIX+case.lower()+'-original-'+level.lower()+'-a01'
        output=SMALL/'episodes'/(name+'.json')
        if not output.exists():
            args=[sys.executable,'-u','scripts/run_s1_w59_research.py','--case',case,'--arm','P3-full',
                '--material','M1','--level',level,'--task','original','--wm-recipe',str(P3),'--attempt',name]
            for path in paths:args+=['--wm-checkpoint',str(path)]
            run('s1_w59_lease.py','--runtime',runtime,'--job',name,'--slots',1,'--',*args)
        if not output.is_file() or read(output).get('stage_status')!='passed':break


def available_sources():
    # Register the first complete set of naturally acquired, excluded-from-fit, independent sources.
    ids=[PREFIX+c+'-original-e30-a01' for c in ('b','a')]
    ids += [f'w59-{arm}-m1-{case}-development-short-a01' for arm in ('fixed-action','b3') for case in ('b','a')]
    result=[]
    for name in ids:
        receipt=SMALL/'attempts'/(name+'.json')
        raw=LARGE/name
        if (receipt.is_file() and read(receipt).get('ended_unix_s') and
            (raw/'w59-legal-requests/manifest.jsonl').exists() and (raw/'trajectory.npz').exists() and
            (raw/'physical-000.npz').exists()):
            if 'original' not in name or read(receipt).get('stage_status')=='passed':result.append(raw)
    return result,ids


def main(runtime):
    paths=checkpoints();OUT.mkdir(parents=True,exist_ok=True)
    status('waiting_development_and_matched_models',required_development_branches=96,
           model_checkpoints=list(map(str,paths)),runtime=str(runtime))
    while closed('development')[0]<96 or not all(p.is_file() for p in paths):
        if closed('development')[1]:
            status('technical_block',reason='A charged development branch has a technical interruption.')
            return
        time.sleep(20)
    score(REG,'development',paths,DEV)
    development=read(DEV)
    if development['summary']['measured_branches']!=96:
        raise ValueError('The development comparison lacks a complete physical denominator')
    result=development['fixed_horizon_comparisons']
    eligible=[int(k) for k,v in result.items() if v['ordering_eligible']]
    # Model fitting uses these actual development branches. This choice only specifies the
    # prospective method for independent confirmation; it is never reported as held-out benefit.
    horizon=max(eligible) if eligible else 32
    base=dict(schema='W59_frozen_candidate_recipe_v1',model_paths=list(map(str,paths)),
        model_sha256=[sha(p) for p in paths],material='M1',planning_substeps=horizon,
        ranking_development_eligible=bool(eligible),eligible_development_substeps=sorted(eligible),
        development_score_sha256=sha(DEV),source_registry_sha256=sha(REG),
        fixed_day_recipe='64 candidates twice; first request held to its next model-predicted legal event; later nominal requests.',
        selector='longest registered horizon with real ranking above the replay floor and no excess hard failure; 0.64s prospective negative comparator otherwise')
    immutable(P3,base)
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(original,runtime,case,paths) for case in ('A','B')] if eligible else []
        status('waiting_independent_sources_and_gradient_recipe',p3_originals_open=bool(eligible),
            p3_recipe=str(P3),development_score=str(DEV),required_new_states=36)
        last=None
        while not V3.is_file():
            for fut in futures:
                if fut.done():fut.result()
            sources,all_ids=available_sources()
            signatures=tuple(p.name for p in sources)
            if signatures and signatures!=last:
                last=signatures
                r=independent_sources(REG,sources,V3)
                if r['status']!='locked':status('waiting_independent_sources_and_gradient_recipe',
                    p3_originals_open=bool(eligible),source_screen=r['source_screen'],
                    sealed_count=r['sealed_count'],first_update_counts=r['first_update_counts'])
            if V3.is_file():break
            wanted=all_ids if eligible else all_ids[2:]
            # A failed original qualifier precludes its matching source only. Other sources continue.
            if all((SMALL/'attempts'/(k+'.json')).is_file() and read(SMALL/'attempts'/(k+'.json')).get('ended_unix_s') for k in wanted):
                status('blocked_independent_sources',observed_sources=list(signatures),
                    reason='All registered source tasks ended without the required new legal states after exclusion.')
                return
            time.sleep(20)
        horizons=OUT/'r3-development-horizons.json'
        while not horizons.exists():time.sleep(10)
        grad=read(horizons);critic=ROOT/'work/w59/learner/historical-task-critic.pt'
        freeze=dict(**base,source_registry_v3_sha256=sha(V3),r3_substeps=grad['chosen_substeps'],
            critic_sha256=sha(critic),learner_rule_sha256=sha(ROOT/'experiments/terrain_memory/s1-w59-gradient-admission.json'),
            online_recipe_sha256=sha(ROOT/'experiments/terrain_memory/s1-w59-online-wm.json'),
            r3_development_horizons_sha256=sha(horizons),frozen_before_sealed=True,**BG)
        immutable(FREEZE,freeze)
        status('sealed_admission_running',registry=str(V3),freeze=str(FREEZE),sealed_budget=96)
        run('run_s1_w59_interventions.py','--registry',V3,'--runtime',runtime,'--role','sealed_admission',
            '--freeze',FREEZE,'--workers',3)
        score(V3,'sealed_admission',paths,SEALED,FREEZE)
        sealed=read(SEALED)
        status('sealed_scored',role='independent',sealed_score=str(SEALED),registry=str(V3),
            physics=sealed['summary']['observed_mj_steps'],horizon=horizon,
            ranking=sealed['fixed_horizon_comparisons'][str(horizon)]['ordering_eligible'],
            gradient=sealed['summary']['admitted'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'work/w59/action-stages.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:sys.exit(0)
    try:main(a.runtime.resolve())
    except Exception as error:
        status('technical_block',reason=repr(error))
        raise
