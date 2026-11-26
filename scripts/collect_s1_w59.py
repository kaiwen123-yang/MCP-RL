"""Collect original and counterfactual physical costs once, by terminal attempt and Shanghai date."""
from collections import Counter
from datetime import datetime
import fcntl
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from publish_s1_go2_w54 import write
from s1_w59_lease import ROOT

ZONE = ZoneInfo('Asia/Shanghai')
R = ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59'


def read(p):
    return json.loads(p.read_text())


def sh(stamp):
    return datetime.fromtimestamp(stamp,ZONE)


def event(section, stamp, identifier, heading, body):
    key=sh(stamp).strftime('%Y%m%d')+'/w59/'+identifier
    if key in section:
        return section
    date=sh(stamp).date().isoformat()
    if '## '+date not in section:
        section+='\n## '+date+'｜W59 执行\n'
    anchor='### '+key+'｜'+heading
    return section.rstrip()+'\n\n'+anchor+'\n\n'+body+'\n\n'


def collect():
    terminal=[]
    current=[]
    for p in sorted((R/'attempts').glob('*.json')):
        a=read(p)
        if not a.get('ended_unix_s'):
            current.append({k:a.get(k) for k in ('attempt','status','pid','controller','slot','tmux')})
            continue
        name=p.stem
        final=R/'episodes'/(name+'.json')
        f=read(final) if final.is_file() else {}
        raw_n=f.get('physical_samples',a.get('physical_samples'))
        row=dict(attempt=name,arm=a.get('w59',{}).get('arm',a.get('arm')),case=a['case'],task=a.get('task_mode'),
            level=a.get('level'),source_commit=a['source_commit'],controller=a['controller'],stage_status=a['status'],
            reason=a.get('reason'),physical_rows=raw_n,executed_s=None if raw_n is None else raw_n*.0002,
            terminal_clock_s=f.get('simulation_s'),wall_s=f.get('wall_s'),end=a['ended_unix_s'],
            predecessor=(f.get('qualification') or {}).get('predecessor_attempt'),
            full_stable_s=f.get('final_stable_s'),min_height_m=f.get('min_body_height_m'),
            task_result=f.get('stage_progress'),branch=f.get('branch',{}).get('applied'))
        terminal.append(row)
    dest=R/'index.json'
    summary=read(dest)
    delivery=[read(p) for p in sorted((R/'delivery').glob('*.json'))]
    assets={a['name']:a for d in delivery if d.get('status')=='verified' for a in d['assets']}
    done=[v for v in terminal if v['stage_status']=='passed' and v['task']=='original']
    summary.update(physical=dict(attempts=len(terminal)+len(current),active=current,
        terminal=terminal,terminal_cost_sim_s=sum(x['executed_s'] or 0 for x in terminal),
        missing_costs=[x['attempt'] for x in terminal if x['executed_s'] is None],
        reasons=dict(Counter(str(x['reason']) for x in terminal)),
        qualified={arm:{c:[v['level'] for v in done if v['arm']==arm and v['case']==c] for c in 'AB'} for arm in sorted({v['arm'] for v in done})}),
        publication=dict(verified_bundles=sum(v.get('status')=='verified' for v in delivery),
        verified_assets=len(assets),verified_bytes=sum(v['size'] for v in assets.values())))
    write(dest,summary)
    diary=ROOT/'docs/S1_DAILY_WORKLOG.md'
    text=diary.read_text()
    head,template=text.split('## 以后每天如何追加',1)
    for a in sorted(terminal,key=lambda v:v['end']):
        name=a['attempt']
        if name in summary.get('initial_terminal_attempts',{}):
            continue
        if a['task']=='branch':
            continue  # The state-level eight-way comparison is logged when its complete denominator seals.
        ts=sh(a['end']).strftime('%H:%M:%S')
        detail=(f"{ts}。{a['arm']} / {a['case']} / {a['task']} / {a['level']}，控制器 {a['controller']}，源码 {a['source_commit'][:12]}。"
                f"实际 {a['physical_rows']} 个 0.2 ms 子步，{a['executed_s']} 模拟秒。判定 {a['stage_status']}，原因 {a['reason']}。"
                f"原前置回合 {a['predecessor']}，最低机身高度 {a['min_height_m']} m。"
                f"完整任务、步态和发布回执分别在 w59/episodes、w59/gait、w59/delivery，训练成本按独立 head 学习账本计。")
        head=event(head,a['end'],'attempt/'+name,'自然终局或技术中断',detail)
    diary.write_text(head+'## 以后每天如何追加'+template)


if __name__=='__main__':
    w=ROOT/'work/w59'
    with (w/'collect.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        collect()
