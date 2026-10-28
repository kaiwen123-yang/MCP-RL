#!/usr/bin/env python3
"""Advance only qualified R11 cases; keep the campaign's one physical clock."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from s1_go2_r11_protocol import ROOT,CONTRACT,LEVELS,CONDITIONS,RUN
from s1_go2_r11_runtime import index_file,dump
SMALL=ROOT/CONTRACT['outputs']['compact_root']
LARGE=ROOT/CONTRACT['outputs']['large_root']
TRAIN_PY=ROOT/'work/envs/joint-py312/bin/python'
GROUPS=['PPO_RAW_S5','PANN_PPO_S5']
def current():return json.loads((SMALL/'index.json').read_text())
def launch(attempt,build,level,condition='nominal',group='FIXED_S5',seed=None,checkpoint=None,purpose='formal',terrain='tm02_anchor'):
    py=TRAIN_PY if group!='FIXED_S5' else Path(sys.executable)
    workers=current().get('frozen_controller',{}).get('workers',8 if build=='S5-r2' else 2)
    cmd=[str(py),'-u',str(ROOT/'scripts/run_s1_go2_r11.py'),'--attempt',attempt,'--build',build,
         '--level',level,'--condition',condition,'--group',group,'--purpose',purpose,'--terrain',terrain,'--workers',str(workers)]
    if seed is not None:cmd+=['--seed',str(seed),'--checkpoint',str(checkpoint)]
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')+':'+env.get('PYTHONPATH','')
    env['LD_LIBRARY_PATH']=str(ROOT/'work/r11-frozen'/build/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
        stdout=(ROOT/'work'/f'r11-{attempt}.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    return p
def result(attempt):
    p=SMALL/(attempt+'.json')
    return json.loads(p.read_text()) if p.exists() else None
def wait_case(attempt):
    while True:
        r=result(attempt)
        if r is not None:return r
        a=next(x for x in current()['attempts'] if x['attempt']==attempt)
        if a['status'] in ('initialization_failed','interrupted'):return a
        time.sleep(10)
def chain(build,prefix,condition,group='FIXED_S5',seed=None,checkpoint=None,levels=LEVELS):
    outcomes=[]
    for level in levels:
        if time.time()>=current()['campaign_deadline_unix_s']:break
        attempt=f'{prefix}-{level}'
        job=launch(attempt,build,level,condition,group,seed,checkpoint)
        job.wait()
        r=result(attempt)
        outcomes.append(dict(attempt=attempt,returncode=job.returncode,result=r))
        if r is None or not r['success']:break
    return outcomes
def train_directory(group,seed):return LARGE/'training'/f'{group}-seed{seed}'
def training_progress():
    values=[]
    for group in GROUPS:
        for seed in range(3):
            d=train_directory(group,seed);p=d/'progress.json'
            j=json.loads(p.read_text()) if p.exists() else dict(samples=0,physics_s=0,wall_s=0)
            stopped=d/'stopped.json'
            tail=json.loads(stopped.read_text()) if stopped.exists() else {}
            j.update(group=group,seed=seed,actual_samples=tail.get('actual_transitions',j['samples']),
                stopped=tail)
            values.append(j)
    return values
def reserve():
    idx=current()
    measured=[a['wall_s']/a['simulation_s'] for a in idx['attempts']
        if a.get('success') and a.get('simulation_s',0)>=30]
    fixed_ratio=max(measured,default=CONTRACT['budget']['actual_cost_reference']['slowdown'])
    progress=training_progress()
    ratios=[v['wall_s']/v['physics_s'] for v in progress if v['physics_s']>=30]
    learned_ratio=max(ratios,default=fixed_ratio)
    # Two seed0 learned chains can run together after sampling; fixed nominal runs now.
    # All four dependencies of both learned chains remain protected until actual predecessor failure.
    protected_learning=990*learned_ratio+4*45
    fixed_remaining=900
    fixed_rows=[a for a in idx['attempts'] if (a.get('case_identity') or a.get('case') or {}).get('group')=='FIXED_S5'
        and a.get('condition')=='nominal' and a.get('purpose')=='formal']
    for a in fixed_rows:
        level=a.get('evaluation_stage',{}).get('level')
        if a.get('stage_status') in ('physical_material_failure','task_criteria_unmet'):
            fixed_remaining=0
        elif level=='E300' and a.get('success'):fixed_remaining=600
        elif level=='E600' and a.get('success'):fixed_remaining=0
    seconds=max(6*3600,1.2*max(protected_learning,fixed_remaining*fixed_ratio))
    with index_file(SMALL,CONTRACT) as j:
        j['evaluation_reserve']=dict(wall_s=seconds,fixed_ratio=fixed_ratio,learned_ratio=learned_ratio,
            fixed_remaining_sim_s=fixed_remaining,protected_learning_sim_s_per_case=990,
            basis='Actual complete R11 stages and >=30s per-stream sampling; two protected learning ladders execute in parallel. Includes not-yet-qualified E30/E60/E300/E600. No assumption of sixfold speedup.',
            updated_unix_s=time.time())
    return seconds
def main():
    p=argparse.ArgumentParser();p.add_argument('--follow',required=True);a=p.parse_args()
    first=wait_case(a.follow)
    if not first.get('success'):
        print('Development E30 did not qualify; evidence review required.',flush=True);return
    build=(first.get('case_identity') or {})['common_configuration_sha256']
    # The actual build identity is carried in the resolved configuration.
    resolved=json.loads((SMALL/(a.follow+'-resolved.json')).read_text())
    build=resolved['r11_configuration']['controller_build']
    second=f"dev{len(current()['attempts'])+1:02d}-{build}-E60"
    job=launch(second,build,'E60',purpose='development');job.wait()
    second_result=result(second)
    if not second_result or not second_result['success']:
        print('Same-version E60 did not qualify; S5 remains unfrozen.',flush=True);return
    frozen=current()['frozen_controller']
    # One nominal rigid regression, fixed nominal long ladder and training overlap.
    rigid=launch('rigid-'+build+'-E60',build,'E60',purpose='rigid_regression',terrain='rigid')
    fixed_cmd=[sys.executable,str(Path(__file__)),'--fixed-chain',build]
    # A dedicated process runs the fixed chain; this process coordinates training.
    fixed=subprocess.Popen(fixed_cmd,cwd=ROOT,stdin=subprocess.DEVNULL,
        stdout=(ROOT/'work/r11-fixed-chain.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    deadline=current()['campaign_deadline_unix_s']-reserve()
    dump(SMALL/'training-deadline.json',dict(deadline_unix_s=deadline,basis='protected_nominal_ladders'))
    streams=[]
    for group in GROUPS:
        for seed in range(3):
            cmd=[str(TRAIN_PY),'-u',str(ROOT/'scripts/train_s1_go2_r11.py'),'--group',group,'--seed',str(seed)]
            env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')+':'+env.get('PYTHONPATH','')
            env['LD_LIBRARY_PATH']=str(ROOT/'work/r11-frozen'/build/'lib')+':'+env.get('LD_LIBRARY_PATH','')
            streams.append(subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=(ROOT/'work'/f'r11-train-{group}-s{seed}.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True))
    while any(p.poll() is None for p in streams):
        seconds=reserve();deadline=current()['campaign_deadline_unix_s']-seconds
        if any(p.poll() not in (None,0) for p in streams):deadline=min(deadline,time.time())
        dump(SMALL/'training-deadline.json',dict(deadline_unix_s=deadline,basis='measured_protected_nominal_ladders'))
        time.sleep(30)
    progress=training_progress();common=min(v['samples']//256*256 for v in progress)
    with index_file(SMALL,CONTRACT) as idx:
        idx.update(training_streams=progress,training_transitions=sum(v['actual_samples'] for v in progress),
            primary_checkpoint_transitions=common,training_under_target=common<32768)
    if common==0:
        print('No common updated checkpoint; learned evaluation remains unexecuted.',flush=True)
        fixed.wait();rigid.wait();return
    primary={f'{g}-seed{s}':str(train_directory(g,s)/f'checkpoint-{common:06d}.pt') for g in GROUPS for s in range(3)}
    with index_file(SMALL,CONTRACT) as idx:idx['primary_checkpoints']=primary
    # The two declared seed0 nominal cases are independent and may advance in parallel.
    jobs=[]
    for g in GROUPS:
        cmd=[sys.executable,str(Path(__file__)),'--learned-chain',build,g,'0','nominal',primary[f'{g}-seed0']]
        jobs.append(subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=(ROOT/'work'/f'r11-{g}-seed0-ladder.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True))
    for p in jobs:p.wait()
    fixed.wait();rigid.wait()
    for condition in CONDITIONS[1:]:
        chain(build,'fixed-'+condition,condition)
    for seed in range(3):
        for condition in (CONDITIONS[1:] if seed==0 else CONDITIONS):
            jobs=[]
            for group in GROUPS:
                cmd=[sys.executable,str(Path(__file__)),'--learned-chain',build,group,str(seed),condition,primary[f'{group}-seed{seed}']]
                jobs.append(subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=(ROOT/'work'/f'r11-{group}-s{seed}-{condition}-ladder.log').open('w'),
                    stderr=subprocess.STDOUT,start_new_session=True))
            for p in jobs:p.wait()
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixed-chain':
        chain(sys.argv[2],'fixed-nominal','nominal',levels=['E300','E600'])
    elif len(sys.argv)>1 and sys.argv[1]=='--learned-chain':
        _,_,build,group,seed,condition,checkpoint=sys.argv
        chain(build,f'{group}-s{seed}-{condition}',condition,group,int(seed),checkpoint)
    else:main()
