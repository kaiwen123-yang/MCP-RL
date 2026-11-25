"""Continue only earned W58 physical stages, after every W57 queue and natural tail is terminal."""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from inspect_s1_w58_predecessor import BG, REL, W57, SH, read, w57_state
from publish_s1_go2_w54 import write
from s1_w58_pool import Pool, WORK, STATE as R, RAW
from compare_s1_w58 import ARMS, matched_matrix, metrics, decide

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(5801,5802,5803)
STAGE=WORK/'stage.json'


def stage(name,detail,**extra):
    state=dict(stage=name,detail=detail,updated_at=datetime.now(SH).isoformat(),tmux=os.environ['W58_TMUX'],
               dispatcher_pid=os.getpid(),**extra,**BG)
    write(STAGE,state)
    print(state['updated_at'],name,detail,flush=True)


def small_w57(d):
    return dict(time_shanghai=d['observed_at'],primary=d['primary'],private_short_A=d['private_short_A'],
        pair=d['pair_dispatch'],final=d['final'],panes={n:i['pid'] if i else None for n,i in d['panes'].items()},
        attempts=[a['attempt'] for a in d['active_physics']],clocks={
            a['attempt']:(a['last_physics'] or {}).get('physical_time_s') for a in d['active_physics']},
        committed={h:v['cumulative_distinct'] for h,v in d['heads'].items()},
        updates={h:v['version'] for h,v in d['heads'].items()},
        uncommitted={h:v['uncommitted_rows'] for h,v in d['heads'].items()})


def old_pool_free():
    # Check the queue and both flock owners under the same W56 dispatch mutex.
    queue=W57.parent/'MCP-RL/work/w56-physical-slots'
    held=[]
    with (queue/'dispatch.lock').open('a') as mux:
        fcntl.flock(mux,fcntl.LOCK_EX)
        if any(Path('/proc',t.stem.rsplit('-',1)[-1]).exists() for t in queue.glob('*.waiting')):return False
        try:
            for i in range(6):
                f=(queue/f'slot-{i}.lock').open('a')
                held.append(f)
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        finally:
            for f in held:f.close()
    return True


def wait_w57():
    last=None
    while True:
        d=w57_state()
        view=small_w57(d)
        passed=(d['final'] and d['primary']=='queue_ended' and d['private_short_A'] in ('ended','not_run')
                and d['pair_dispatch']=='series_finished_after_natural_episodes' and not d['active_physics']
                and d['status']=='finished' and not d['panes'])
        if passed and old_pool_free():
            write(R/'w57-natural-release.json',w57_state(verify=True))
            return
        key=(tuple(view['attempts']),view['primary'],view['private_short_A'],tuple(view['committed'].values()))
        if key!=last:
            stage('waiting_W57_natural_release','等待 W57 的后继与交付调度自然结束。',
                  w57_live=view,w58_physics_admitted=False,slots=0,w57_active_physics=len(view['attempts']))
            last=key
        else:
            current=read(STAGE)
            current.update(w57_live=view,w57_active_physics=len(view['attempts']),updated_at=datetime.now(SH).isoformat())
            write(STAGE,current)
        time.sleep(30)


def job(arm,case,gait,material,task='original',level='E30',split='original',seed=None):
    key=f'w58-{gait.lower()}-{material.lower()}-{arm.lower()}'
    if seed:key+=f'-s{seed}'
    key+=f'-{case.lower()}-{split}-{level.lower() if task=="original" else task}-a01'
    # Only a documented technical interruption can nominate another attempt for this cell.
    receipt=R/'attempts'/(key+'.json')
    while receipt.is_file():
        prior=read(receipt)
        if prior.get('status')!='technical_interruption' or not prior.get('replacement_attempt'):
            break
        key=prior['replacement_attempt']
        receipt=R/'attempts'/(key+'.json')
    args=['--attempt',key,'--arm',arm,'--case',case,'--gait',gait,'--material',material,'--task',task,'--split',split,'--level',level]
    if seed:
        head='full' if arm=='P2-full' else 'blind'
        config=WORK/'campaigns'/f'seed-{seed}'/head/'frozen-evaluation.json'
        args+=['--learning-seed',str(seed),'--timing-config',str(config)]
    physical_s=int(level[1:]) if task=='original' else 168 if task=='short' else 288
    return dict(id=key,command=[sys.executable,'-u','scripts/run_s1_w58.py',*args],
        slots_needed=1,disk_GiB=max(2,physical_s*.03),kind=('gait-' if arm=='G1' else 'p2-')+task+'-'+level,
        data=dict(arm=arm,case=case,gait=gait,material=material,task=task,level=level,split=split,learning_seed=seed))


def head_job(seed,head,gait,material):
    key=f'w58-joint-{seed}-{head}'
    return dict(id=key,command=[sys.executable,'-u','scripts/run_s1_w58_joint.py','head','--seed',str(seed),
        '--head',head,'--gait',gait,'--material',material],slots_needed=2,disk_GiB=4,kind='joint_original_E60',seed=seed,head=head)


def passed(j):
    record=R/'attempts'/(j['id']+'.json')
    return record.is_file() and read(record)['status']=='passed'


def qual_index(rows):
    return {(j['data']['arm'],j['data']['case'],j['data']['learning_seed']):j['id'] if passed(j) else None for j in rows}


def score_gait(jobs):
    from score_s1_w58_gait import score_episode,select_gait
    g=R/'g'
    for case in 'ab':
        for level in ('e30','e60'):
            name=f'w57-m1-b-{case}-original-{level}-a01'
            result=W57/'results'/REL/'w57/episodes'/(name+'.json')
            target=g/'g0'/(name+'.json')
            if result.is_file() and not target.exists():score_episode(W57/'outputs'/REL/name,result,target)
    for j in jobs:
        name=j['id']; result=R/'episodes'/(name+'.json')
        target=g/'g1'/(name+'.json')
        if (RAW/name/'trajectory.npz').is_file() and result.is_file() and not target.exists():
            score_episode(RAW/name,result,target)
    return select_gait(g/'g0',g/'g1',g/'selection.json')


def ready_table(jobs, eligible):
    return [j for j in jobs if eligible.get((j['data']['arm'],j['data']['case'],j['data']['learning_seed']))]


def mechanism(arms,gait,material,split):
    return [job(a,c,gait,material,mode,'E60',split,seed) for mode in ('short','recovered')
            for a,seed in arms for c in 'AB']


def run():
    runtime=read(WORK/'runtime.json')
    physics_root=Path(runtime['path']).resolve()
    if (physics_root/'work/w58/runtime.json').resolve()!=(WORK/'runtime.json').resolve():
        raise RuntimeError('Dispatcher and frozen physics must use the same D57 campaign')
    wait_w57()
    pool=Pool(physics_root)
    stage('G1_original_qualification','W57 两份物理 dispatcher、自然 collector 尾段和发布后继已结束，旧槽已核空闲。',slots=3)
    pulse=lambda:None  # The independent integration pane consumes atomic job and stage receipts.
    g30=[job('G1',c,'G1','M1') for c in 'AB']
    pool.batch(g30,pulse)
    g60=[job('G1',j['data']['case'],'G1','M1',level='E60') for j in g30 if passed(j)]
    pool.batch(g60,pulse)
    g_result=score_gait(g30+g60)
    gait=g_result['selected']
    stage('freeze_common','按原资格和全摆动接管、代价核对选用共同步态。等候已登记整段 M2 保留评估。',selected_gait=gait)
    while not (R/'m/selection.json').is_file():time.sleep(30)
    material=read(R/'m/selection.json')['selected']
    shared=dict(selected_gait=gait,selected_material=material)
    write(R/'common-freeze.json',dict(gait=gait,material=material,source=runtime,development_route='s1-w58-revisit-v2',
        arms=list(ARMS),fixed_wait=read(ROOT/'experiments/terrain_memory/s1-w58-p2.json')['arms'] if
        'arms' in read(ROOT/'experiments/terrain_memory/s1-w58-p2.json') else read(ROOT/'experiments/terrain_memory/s1-w58-p2.json'),
        registered_before_physical_matrix=True,**BG))
    stage('original_qualifications','四个开发身份逐 case 取得原资格，另有预选 A 的时长修正开关配对。',**shared)
    q30=[job(a,c,gait,material) for a in ARMS for c in 'AB']
    off=job('availability-off','A',gait,material)
    pool.batch(q30+[off],pulse)
    qual=qual_index(q30)
    controls=[(a,None) for a in ARMS]
    dev=mechanism(controls,gait,material,'development')
    q60=[job(j['data']['arm'],j['data']['case'],gait,material,level='E60') for j in q30 if passed(j)]
    stage('development_matrix','合格 case 执行预登记开发任务及各自原 E60。',**shared)
    pool.batch(ready_table(dev,qual)+q60,pulse)
    q60idx=qual_index(q60)
    avail_on=next(j for j in q30 if j['data']['arm']=='B2' and j['data']['case']=='A')
    write(R/'availability-original-A.json',dict(condition='A original E30, same gait/material/zero event wait',on=metrics(avail_on['id']),off=metrics(off['id'])))
    results=matched_matrix(dev,qual,'development')
    decision=decide(results,{a+'-'+c:bool(q60idx.get((a,c,None))) for a in ('P2-full','P2-private') for c in 'AB'})
    trained={}
    if decision['joint_eligible']:
        stage('matched_joint_learning','按 5801/5802/5803 逐个 head 原子预留两槽。full 的已冻结节点决定各自 blind 预算。',**shared)
        full=[head_job(s,'full',gait,material) for s in SEEDS]
        # Independent B2/fixed validation uses the spare slot. No validation score enters
        # the learner's initialization, reward, node selection or matched budget.
        singleton_validation=mechanism([('B2',None),('Fixed-wait',None)],gait,material,'validation')
        pool.batch(full+ready_table(singleton_validation,qual),pulse)
        def complete(s,h):
            path=R/'learning'/f'seed-{s}'/h/'final.json'
            return read(path) if path.is_file() else dict(complete=False,distinct=0)
        pool.batch([head_job(s,'blind',gait,material) for s in SEEDS if complete(s,'full')['complete']],pulse)
        arms=[(a,s) for s in SEEDS for a in ('P2-full','P2-private')]
        available=[]
        for s in SEEDS:
            f,b=complete(s,'full'),complete(s,'blind')
            trained[str(s)]=dict(full=f,private_blind=b,matched=f['complete'] and b['complete'] and f['distinct']==b['distinct'])
            if f['complete']:available.append(('P2-full',s))
            if b['complete']:available.append(('P2-private',s))
        q30new=[job(a,c,gait,material,seed=s) for a,s in available for c in 'AB']
        stage('frozen_joint_qualifications','各个最终冻结 seed/head/case 从原 E30 取得自身资格。',**shared)
        pool.batch(q30new,pulse)
        qual.update(qual_index(q30new))
        q60new=[job(j['data']['arm'],j['data']['case'],gait,material,level='E60',seed=j['data']['learning_seed']) for j in q30new if passed(j)]
        pool.batch(q60new,pulse)
        q60idx.update(qual_index(q60new))
        controls=[('B2',None),('Fixed-wait',None),*arms]
    stage('independent_validation','按预先冻结的独立路线对全部身份和未运行格逐项入账。',joint_eligible=decision['joint_eligible'],**shared)
    hold=mechanism(controls,gait,material,'validation')
    pool.batch(ready_table(hold,qual),pulse)
    rows=matched_matrix(hold,qual,'validation')
    long=[]
    if decision['joint_eligible'] and trained['5801']['matched']:
        for c in 'AB':
            own=all(q60idx.get((a,c,5801)) for a in ('P2-full','P2-private'))
            relevant=[r for r in rows if r['seed']==5801 and r['case']==c]
            # Same two frozen heads, both history burdens observed, actual target exposures:
            # now sustained original-route performance is an unanswered engineering endpoint.
            if own and len(relevant)==4 and all(r['full_window'] and r['targets_realized'] for r in relevant):
                long.extend(job(a,c,gait,material,level='E300',seed=5801) for a in ('P2-full','P2-private'))
    stage('conditional_long_qualification','主展示 seed 5801 逐 case 按既有原资格和实际机制记录进入长资格。',**shared)
    pool.batch(long,pulse)
    last=[job(j['data']['arm'],j['data']['case'],gait,material,level='E600',seed=5801) for j in long if passed(j)]
    pool.batch(last,pulse)
    write(R/'outcome.json',dict(**shared,joint_eligible=decision['joint_eligible'],seed_outcomes=trained,
        qualifications=[dict(attempt=p.stem,status=v['status'],level=v['level'],case=v['case'],controller=v['controller'])
            for p in sorted((R/'attempts').glob('*.json')) for v in [read(p)] if v['task_mode']=='original'],
        development='development-matrix.json',validation='validation-matrix.json',
        long_attempts=[j['id'] for j in long+last],ended_at=datetime.now(SH).isoformat(),**BG))
    stage('finished','全部已具条件的物理后继已自然结束，交付进程逐件比对远端 size/digest。',**shared)


if __name__=='__main__':
    lock=(WORK/'dispatcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run()

