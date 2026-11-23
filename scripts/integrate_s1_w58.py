"""Integrate compact, completed records from the three D57 lanes into the existing daily worklog."""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from inspect_s1_w58_predecessor import BG, REL, SH, read
from publish_s1_go2_w54 import write

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'/REL/'w58'
WORK=ROOT/'work/w58'
DOC=ROOT/'docs/S1_DAILY_WORKLOG.md'
LOGKEYS=('stage','selected_gait','selected_material','joint_eligible','slots','running','attempts','completed','joint_distinct')


def append_event(date,event,title,body):
    text=DOC.read_text()
    eid=date.replace('-','')+'/w58/'+event
    if eid in text:return
    block=f'\n### {eid}｜{title}\n\n{body.strip()}\n\n'
    import re
    header=re.search(r'^## '+date+r'[^\n]*\n',text,re.M)
    if header:
        text=text[:header.end()]+block+text[header.end():]
    else:
        point=text.index('## 以后每天如何追加')
        text=text[:point]+f'## {date}｜W58 执行\n'+block+text[point:]
    DOC.write_text(text)


def harvest_material():
    lane=ROOT.parent/'MCP-RL-w58-m2/results'/REL/'w58'
    # Only small finished episode reports and compressed event rows cross worktrees on D57.
    # The raw state volumes are published directly at their original paths.
    for sub in ('m2','m'):
        origin=lane/sub
        if not origin.is_dir():continue
        for src in sorted(origin.iterdir()):
            if not (src.name.endswith('.json') or src.name.endswith('.json.gz')):continue
            if src.suffix=='.json':read(src)
            elif not (origin/'errors-and-coverage.json').is_file() and not (origin/(src.name.split('-forecast')[0]+'-forecast-cost.json')).is_file():
                continue
            dst=R/sub/src.name
            if dst.is_file() and (dst.stat().st_size==src.stat().st_size and dst.stat().st_mtime_ns>=src.stat().st_mtime_ns):continue
            dst.parent.mkdir(exist_ok=True,parents=True)
            temp=dst.with_name(dst.name+'.pending')
            shutil.copy2(src,temp)
            os.replace(temp,dst)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def publish_git():
    index=str(R.relative_to(ROOT))
    includes=['docs/S1_DAILY_WORKLOG.md',index,':(exclude)**/*.lock',':(exclude)**/*.pending',':(exclude)**/*.tmp*',
              f':(exclude){index}/episodes/**']
    git('add','--',*includes)
    if (R/'episodes/index.json').is_file():git('add','--',index+'/episodes/index.json')
    if subprocess.call(['git','diff','--cached','--quiet'],cwd=ROOT):
        git('commit','-m','Publish W58 measured progress and Shanghai event log')
    git('fetch','origin','main')
    git('merge','--no-edit','origin/main')
    git('push','origin','HEAD:main')
    head=git('rev-parse','HEAD')
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    if remote!=head:
        git('fetch','origin','main')
        if subprocess.call(['git','merge-base','--is-ancestor',head,'origin/main'],cwd=ROOT):
            raise RuntimeError('W58 compact publication is not yet visible on main')
    write(WORK/'last-push.json',dict(commit=head,remote_head=remote,verified_unix_s=time.time()))


def episode_events():
    for p in sorted((R/'attempts').glob('*.json')):
        row=read(p)
        if 'ended_unix_s' not in row:continue
        name=row['attempt']
        date=datetime.fromtimestamp(row['ended_unix_s'],SH).date().isoformat()
        episode=R/'episodes'/p.name
        outcome=read(episode) if episode.is_file() else {}
        mech=outcome.get('mechanism',{})
        ev=outcome.get('w57_timing') or {}
        append_event(date,'attempt/'+name,'物理回合结束',
            f"{name}，{row['w58']['split']} / {row['task_mode']} / {row['case']}，控制器 {row['controller']}，源码 {row['source_commit']}。"
            f"本次 {row.get('physical_samples',0)} 个物理子步，模拟 {row.get('simulation_s')} s，结论 {row['status']}，终止原因 {row.get('reason')}。"
            f"机制曝光 {mech.get('confirmed_cycle_count','—')}/{mech.get('cycles_requested','—')}，"
            f"本回合进入采样账本 {ev.get('new_training_transitions',0)} 条，CPU 用户/系统秒 {row.get('user_cpu_s')}/{row.get('system_cpu_s')}。"
            f"物理槽 {row.get('slot')}。[执行记录](../{R.relative_to(ROOT)}/attempts/{p.name})。")


def summary(s):
    return '，'.join(f'{k}={s[k]}' for k in LOGKEYS if k in s)


def refresh():
    harvest_material()
    stage=WORK/'stage.json'
    if not stage.is_file():return
    s=read(stage)
    attempts=[read(p) for p in (R/'attempts').glob('*.json')]
    learned={str(p.parent.relative_to(R/'learning')):read(p)['cumulative_distinct']
             for p in sorted((R/'learning').glob('seed-*/*/update-*.json'))}
    observed=s.get('w57_live',{})
    s.update(attempts=len(attempts),completed=sum('ended_unix_s' in r for r in attempts),
             running=[r['attempt'] for r in attempts if 'ended_unix_s' not in r],joint_distinct=learned)
    s.update(geometry='experiments/terrain_memory/s1-w58-revisit-v2.json',worktrees={
        k:str(ROOT.parent/f'MCP-RL-w58-{d}') for k,d in [('G','g'),('M2','m2'),('P2','p2')]},
        v1_baseline_confirmed=[6,7],v1_denominator_each=12,
        development_and_validation='separate preregistered routes and episode sets',
        runtime=read(WORK/'runtime.json'),physical_pool=str(WORK/'pool'),**BG)
    s['science']={'gait':str(R.relative_to(ROOT)/'g'),'material':str(R.relative_to(ROOT)/'m2'),
                  'planner':str(R.relative_to(ROOT)/'p2'),'joint_contract':'experiments/terrain_memory/s1-w58-joint.json'}
    if (R/'m/selection.json').exists():s['selected_material']=read(R/'m/selection.json')['selected']
    if (R/'publication.json').is_file():s['publication']=read(R/'publication.json')
    if (R/'development-decision.json').is_file():s['joint_eligible']=read(R/'development-decision.json')['joint_eligible']
    now=datetime.now(SH)
    s['observed_at']=now.isoformat()
    previous=WORK/'daily-last.json'
    old=read(previous) if previous.is_file() else None
    if old and old['observed_at'][:10]!=now.date().isoformat():
        append_event(old['observed_at'][:10],'day-end','跨日实测续接',summary(old)+'。最后观察 '+old['observed_at']+
                     '。W57 同时段现场 '+json.dumps(old.get('w57_live',{}),ensure_ascii=False)+'。')
    write(previous,s)
    fingerprint=dict(stage=s['stage'],attempts=s['attempts'],completed=s['completed'],joint=s['joint_distinct'],
        m2=s.get('selected_material'),bundles=s.get('publication',{}).get('verified_bundles'),day=now.date().isoformat(),
        material_replays=sorted(p.name for p in (R/'m2').glob('*-replay.json')),
        material_forecasts=sorted(p.name for p in (R/'m2').glob('*-forecast-cost.json')),
        geometry=(R/'m2/source-footprint-strata.json.gz').is_file(),
        w57_heads=observed.get('committed'),w57_active=observed.get('attempts'),w57_phase=observed.get('primary'))
    last=WORK/'git-fingerprint.json'
    if not last.is_file() or read(last)!=fingerprint:
        if not old or s['stage']!=old['stage'] or old['observed_at'][:10]!=now.date().isoformat():
            append_event(now.date().isoformat(),'stage/'+s['stage'],'调度与实测进度',summary(s)+'。'+s.get('detail','')+
                '。运行快照 '+json.dumps(observed,ensure_ascii=False)+'。')
        if s.get('selected_material'):
            append_event(now.date().isoformat(),'material-selected','保留轨迹与估计器选择',
                f"选择 {s['selected_material']}。各保留轨迹单列释放→跨足重载、短预测误差、区间覆盖与宽度，见"
                f"[选择](../{R.relative_to(ROOT)}/m/selection.json)和[分族指标](../{R.relative_to(ROOT)}/m2/errors-and-coverage.json)。")
        episode_events()
        write(R/'index.json',s)
        publish_git()
        write(last,fingerprint)
        print(now.isoformat(),summary(s),flush=True)


def main():
    lock=(WORK/'integration.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            refresh()
            if (R/'delivery-ended.json').is_file() and (WORK/'stage.json').is_file() and read(WORK/'stage.json')['stage']=='finished':
                today=datetime.now(SH).date().isoformat()
                append_event(today,'verified-delivery','GitHub 完成交付',json.dumps(read(R/'delivery-ended.json'),ensure_ascii=False))
                publish_git()
                return
        except (subprocess.CalledProcessError,RuntimeError) as error:
            print(datetime.now(SH).isoformat(),'GitHub integration pending',repr(error),flush=True)
        time.sleep(45)


if __name__=='__main__':main()

