"""Close one Shanghai calendar day from distinct W57 attempt and PPO-update receipts."""
import argparse
from collections import defaultdict
from datetime import date,datetime,timedelta
from zoneinfo import ZoneInfo
import time
import subprocess

from summarize_s1_w57 import R, ROOT, read


LOCAL=ZoneInfo('Asia/Shanghai')


def day(t):
    return datetime.fromtimestamp(t,tz=LOCAL).date().isoformat()


def generate(stamp):
    rows=[read(p) for p in sorted((R/'attempts').glob('*.json'))]
    ended=[r for r in rows if r.get('ended_unix_s') and day(r['ended_unix_s'])==stamp]
    starts=[r['attempt'] for r in rows if day(r['started_unix_s'])==stamp]
    active=[r for r in rows if day(r['started_unix_s'])<=stamp and
            (not r.get('ended_unix_s') or day(r['ended_unix_s'])>stamp)]
    nsteps=sum(r.get('physical_samples') or 0 for r in ended)
    cpu=sum((r.get('user_cpu_s') or 0)+(r.get('system_cpu_s') or 0) for r in ended)
    by_head=defaultdict(list)
    for p in sorted((R/'r').glob('*/*/update-*.json')):
        # Written once immediately after the actual optimizer advance. Cross-midnight
        # partial buffers remain on their own serials until the paired batch is accepted.
        if day(p.stat().st_mtime)==stamp:by_head[p.parent.name].append(read(p))
    pub=[read(p) for p in (R/'delivery').glob('*.json')]
    assets=[a for repo in pub for a in repo['assets'] if day(a['verified_unix_s'])==stamp]
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    lines=[f'### W57 当日实验入账（{stamp}，Asia/Shanghai）','',
           f'累计账本使用独立 attempt 和 head/case/version/serial。本日启动 {len(starts)} 个自然任务，结束 {len(ended)} 个。结束任务共有 {nsteps:,} 个已记录物理子步、{cpu:.2f} CPU 核秒。这两个量按结束日归集，同一跨日任务计一次。']
    if ended:
        lines+=['','| attempt | 控制 / 环境 case | 任务 | 实际物理时间（s） | 结果 |',
                '| --- | --- | --- | ---: | --- |']
        for r in sorted(ended,key=lambda v:v['ended_unix_s']):
            mode=r['task_mode'] if r['task_mode']!='original' else '原 '+r['level']
            lines.append(f"| `{r['attempt']}` | {r['arm']}/{r['material']} · {r['case']} | {mode} | {r.get('simulation_s',0) or 0:.4f} | {r['status']} · {r.get('reason','')} |")
    samples=sum(u['distinct'] for group in by_head.values() for u in group)
    lines+=['',f'本日闭合的 on-policy 批次含 {samples:,} 条 distinct 转移，旧 W44 速度策略更新 0。样本以同版本两个 128 段交齐并产生优化更新的日期入账。跨日未闭合段保留原序列编号。']
    for head,group in sorted(by_head.items()):
        lines.append(f"{head} 入账 {sum(x['distinct'] for x in group):,} 条、{len(group)} 次更新，实际改变交接时刻 {sum(x.get('effective_liftoff',0) for x in group)} 次，只改变载荷交接斜坡 {sum(x.get('only_ramp',0) for x in group)} 次。")
    if active:
        lines+=['','跨日继续的唯一任务为 '+ '、'.join(f"`{r['attempt']}`" for r in active)+'。']
    lines+=['',f'截至入账，按 GitHub size/digest 得到的本日新增资产回执共 {len(assets)} 个，{sum(a["size"] for a in assets):,} B。源码和紧凑记录基线 `{rev[:12]}`。物理槽、前序资格、仍在执行或等待的模块与发布资产详见 [W57 索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57/index.json)。','']
    return '\n'.join(lines)


def append_day(stamp):
    path=ROOT/'docs/S1_DAILY_WORKLOG.md'
    text=path.read_text()
    if f'### W57 当日实验入账（{stamp}，Asia/Shanghai）' in text:return
    marker='## 以后每天如何追加'
    block=generate(stamp)+'\n'
    if f'## {stamp}' not in text:
        block=f'## {stamp}：W57 材料、控制与学习执行\n\n'+block
    if marker in text:
        text=text.replace(marker,block+marker,1)
    else:text=text.rstrip()+'\n\n'+block
    path.write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('day',nargs='?',type=date.fromisoformat)
    p.add_argument('--follow',action='store_true')
    a=p.parse_args()
    if a.follow:
        active=datetime.now(LOCAL).date()
        while True:
            today=datetime.now(LOCAL).date()
            while active<today:
                append_day(active.isoformat())
                active+=timedelta(days=1)
            if read(R/'index.json').get('status')=='finished':
                append_day(active.isoformat())
                break
            time.sleep(45)
    elif a.day:
        append_day(a.day.isoformat())
    else:
        p.error('Use one Shanghai date or --follow')
