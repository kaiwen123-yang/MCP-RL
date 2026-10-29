#!/usr/bin/env python3
"""Advance only qualified R11 cases; keep the campaign's one physical clock."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from s1_go2_r11_protocol import ROOT,CONTRACT,LEVELS,CONDITIONS,RUN
from s1_go2_r11_runtime import index_file,dump
SMALL=ROOT/CONTRACT['outputs']['compact_root']
LARGE=ROOT/CONTRACT['outputs']['large_root']
TRAIN_PY=ROOT/'work/envs/joint-py312/bin/python'
GROUPS=['PPO_RAW_S5','PANN_PPO_S5']
def current():
    with (SMALL/'index.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        return json.loads((SMALL/'index.json').read_text())
def deadline_file(value):
    path=SMALL/'training-deadline.json'
    temporary=path.with_suffix('.tmp')
    dump(temporary,value)
    temporary.replace(path)
def launch(attempt,build,level,condition='nominal',group='FIXED_S5',seed=None,checkpoint=None,purpose='formal',terrain='tm02_anchor',workers=None):
    py=TRAIN_PY if group!='FIXED_S5' else Path(sys.executable)
    if workers is None:workers=current().get('frozen_controller',{}).get('workers',2 if build=='S5-r1' else 8)
    cmd=[str(py),'-u',str(ROOT/'scripts/run_s1_go2_r11.py'),'--attempt',attempt,'--build',build,
         '--level',level,'--condition',condition,'--group',group,'--purpose',purpose,'--terrain',terrain,'--workers',str(workers)]
    if seed is not None:cmd+=['--seed',str(seed),'--checkpoint',str(checkpoint)]
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')+':'+env.get('PYTHONPATH','')
    env['LD_LIBRARY_PATH']=str(ROOT/'work/r11-frozen'/build/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
        stdout=(ROOT/'work'/f'r11-{attempt}.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    return p
def result(attempt):
    row=next((a for a in current()['attempts'] if a['attempt']==attempt),None)
    if row is None or not row.get('result'):return None
    return json.loads((SMALL/row['result']).read_text())
def wait_case(attempt):
    while True:
        r=result(attempt)
        if r is not None:return r
        a=next((x for x in current()['attempts'] if x['attempt']==attempt),None)
        # The launched runner imports its environment before publishing registration.
        # Wait for that registration; never launch another physical task here.
        if a is None:
            time.sleep(1)
            continue
        if a['status'] in ('initialization_failed','interrupted'):return a
        time.sleep(10)
def chain(build,prefix,condition,group='FIXED_S5',seed=None,checkpoint=None,levels=LEVELS):
    outcomes=[]
    for level in levels:
        if time.time()>=current()['campaign_deadline_unix_s']:break
        attempt=f'{prefix}-{level}'
        idx=current()
        completed=[a for a in idx['attempts'] if a.get('success') and a.get('simulation_s',0)>=30]
        ratio=max((a['wall_s']/a['simulation_s'] for a in completed),default=CONTRACT['budget']['actual_cost_reference']['slowdown'])
        estimate=1.2*int(level[1:])*ratio
        if time.time()+estimate>idx['campaign_deadline_unix_s']:
            with index_file(SMALL,CONTRACT) as state:
                state.setdefault('budget_unexecuted',[]).append(dict(group=group,seed=seed,condition=condition,
                    level=level,build=build,estimated_wall_s=estimate,remaining_wall_s=idx['campaign_deadline_unix_s']-time.time(),
                    reason='Measured full-stage cost does not fit remaining campaign clock; no physical launch'))
            break
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
            collection=d/'collection.json'
            collected=json.loads(collection.read_text()) if collection.exists() else {}
            j.update(group=group,seed=seed,actual_samples=tail.get('actual_transitions',collected.get('collected_transitions',j['samples'])),
                collection=collected,stopped=tail)
            values.append(j)
    return values
def reserve():
    idx=current()
    frozen=idx.get('frozen_controller',{})
    measured=[a['wall_s']/a['simulation_s'] for a in idx['attempts']
        if a.get('success') and a.get('simulation_s',0)>=30 and
        (not frozen or a.get('case_key')==frozen['case_key'])]
    workers=frozen.get('workers',8)
    cpu_samples=[x['aggregate_average_cores'] for x in idx.get('actual_resource_samples',[]) if x['workers']==workers]
    measured_cpu_cores=max(cpu_samples) if cpu_samples else workers+1
    parallel_contention=max(1.,2*(measured_cpu_cores+1)/25.)
    fixed_ratio=max(measured,default=CONTRACT['budget']['actual_cost_reference']['slowdown'])
    # Formal evaluation uses the frozen8-worker execution setting. Do not
    # confuse the2-worker six-stream acquisition cost with that formal cost.
    observer_sources=[]
    replay=idx.get('observer_preparation')
    if replay:observer_sources.append((replay,replay['observed_through_s']))
    else:
        previous=json.loads((ROOT/'results/terrain_memory/simulation/tm-s1-go2-soft-landing-r10/index.json').read_text())['observer_replay']
        observer_sources.append((previous,previous['observed_through_s']))
    for attempt in idx['attempts']:
        if attempt.get('purpose')=='training' and attempt.get('result'):
            actual=json.loads((SMALL/attempt['result']).read_text())
            observer=(actual.get('policy') or {}).get('observer')
            if observer and observer['world_cells']>0 and actual['simulation_s']>0:
                observer_sources.append((observer,actual['simulation_s']))
    observer_ratio=0.
    for observation,duration in observer_sources:
        compilation=observation['first_compilation_s'] or 0.
        observation_rate=max(0.,observation['observation_wall_s']-compilation)/duration
        query_rate=observation['query_wall_s']/duration
        # Original fixed strip/grid has about4200 world cells. This is an
        # explicit upper-geometry extrapolation of the observed component cost.
        scale=max(1.,4200/observation['world_cells'])
        observer_ratio=max(observer_ratio,observation_rate*scale+query_rate+compilation/30)
    learned_ratio=(fixed_ratio+observer_ratio)*parallel_contention
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
        j['evaluation_reserve']=dict(wall_s=seconds,fixed_ratio=fixed_ratio,learned_ratio=learned_ratio,observer_component_ratio=observer_ratio,
            formal_workers=workers,measured_average_cpu_cores=measured_cpu_cores,parallel_contention_factor=parallel_contention,allocated_cpu_cores=25,
            fixed_remaining_sim_s=fixed_remaining,protected_learning_sim_s_per_case=990,
            basis='Actual complete stages of the same frozen controller and worker setting plus retained/real-training PANN component costs, conservatively scaled to the fixed strip grid. Before measured concurrent-stage data exists, use process CPU counters from actual tasks, plus one conservative observer core per job, to estimate two-job contention on25vCPU. Thread count is not continuous CPU utilization. All not-yet-qualified stages remain included. Acquisition slowdown is not substituted for formal evaluation throughput.',
            updated_unix_s=time.time())
    return seconds
def training_cutoff(required_reserve):
    idx=current();now=time.time();end=idx['campaign_deadline_unix_s']
    full_cutoff=end-required_reserve
    allocation=idx['evaluation_reserve']
    priority_reserve=max(6*3600,1.2*allocation['fixed_remaining_sim_s']*allocation['fixed_ratio'])
    priority_cutoff=end-priority_reserve
    progress=training_progress()
    common=min(x['samples']//256*256 for x in progress)
    cutoff=full_cutoff;basis='protected_full_nominal_ladders'
    remaining_update_cost=None
    if common==0:
        if all(x['actual_samples']>0 and x.get('collection') for x in progress):
            remaining_update_cost=max(max(0,256-x['actual_samples'])*
                (now-x['collection']['collection_started_unix_s'])/x['actual_samples'] for x in progress)
            candidate=min(priority_cutoff,now+1.2*remaining_update_cost)
            if now+remaining_update_cost<=priority_cutoff and candidate>cutoff:
                cutoff=candidate;basis='infeasible_full_scope_fair_first_update'
        elif full_cutoff<=now:
            workers=allocation['formal_workers']
            cores=allocation['measured_average_cpu_cores']
            fraction=min(1.,max(0.,(cores-1)/max(1,workers-1)))
            two_worker_ratio=allocation['fixed_ratio']*((1-fraction)+fraction*workers/2)
            initial_session_cost=1.2*60*(two_worker_ratio+3*allocation['observer_component_ratio'])
            cutoff=min(priority_cutoff,now+initial_session_cost)
            basis='infeasible_full_scope_initial_fair_acquisition'
    if basis!='protected_full_nominal_ladders':
        with index_file(SMALL,CONTRACT) as state:
            state['partial_scope_budget_allocation']=dict(
                full_conditional_nominal_reserve_s=required_reserve,
                full_reserve_is_retained_in_cost_accounting=True,
                priority_P1_and_minimum_evaluation_reserve_s=priority_reserve,
                actual_common_updated_transitions=common,
                estimated_remaining_wall_to_first_common_update_s=remaining_update_cost,
                effective_training_cutoff_unix_s=cutoff,updated_unix_s=now,
                rule='Contract exception for infeasible full nominal coverage: preserve P1 priority and at least6h evaluation, allow a fair first common256 update only when empirical collection cost fits, then freeze by budget. Full conditional E30/E60/E300/E600 costs remain recorded; no stage gate or curriculum rule is skipped. Unfunded higher stages remain missing, not failed or passed.',
                basis=basis)
    return cutoff,basis

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
    job=launch(second,build,'E60',purpose='development',workers=resolved['workers']);job.wait()
    second_result=result(second)
    if not second_result or not second_result['success']:
        print('Same-version E60 did not qualify; S5 remains unfrozen.',flush=True);return
    frozen=current()['frozen_controller']
    # One nominal rigid regression, fixed nominal long ladder and training overlap.
    rigid=launch('rigid-'+build+'-E60',build,'E60',purpose='rigid_regression',terrain='rigid')
    rigid.wait()
    fixed_cmd=[sys.executable,str(Path(__file__)),'--fixed-chain',build]
    # A dedicated process runs the fixed chain; this process coordinates training.
    fixed=subprocess.Popen(fixed_cmd,cwd=ROOT,stdin=subprocess.DEVNULL,
        stdout=(ROOT/'work/r11-fixed-chain.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    deadline,basis=training_cutoff(reserve())
    deadline_file(dict(deadline_unix_s=deadline,basis=basis))
    streams=[]
    for group in GROUPS:
        for seed in range(3):
            cmd=[str(TRAIN_PY),'-u',str(ROOT/'scripts/train_s1_go2_r11.py'),'--group',group,'--seed',str(seed)]
            env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')+':'+env.get('PYTHONPATH','')
            env['LD_LIBRARY_PATH']=str(ROOT/'work/r11-frozen'/build/'lib')+':'+env.get('LD_LIBRARY_PATH','')
            streams.append(subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=(ROOT/'work'/f'r11-train-{group}-s{seed}.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True))
    while any(p.poll() is None for p in streams):
        seconds=reserve();deadline,basis=training_cutoff(seconds)
        if any(p.poll() not in (None,0) for p in streams):deadline=min(deadline,time.time())
        deadline_file(dict(deadline_unix_s=deadline,basis=basis))
        time.sleep(30)
    progress=training_progress()
    errors=[p.returncode for p in streams if p.returncode!=0]
    if errors:
        with index_file(SMALL,CONTRACT) as idx:
            idx.update(training_streams=progress,training_transitions=sum(v['actual_samples'] for v in progress),
                training_status='technical_interruption_requires_repair',training_process_exit_codes=errors)
        print('Training infrastructure interrupted; no primary checkpoint selected from this technical stop.',flush=True)
        return
    common=min(v['samples']//256*256 for v in progress)
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
