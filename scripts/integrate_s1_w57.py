"""Integrate W57 compact results and verify GitHub delivery from D57 through the natural campaign end."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from append_s1_w57_day import append_day
from publish_s1_w57 import originals, release_assets, REPO, TAG
from publish_s1_go2_w54 import write
from summarize_s1_w57 import ROOT, REL, R, RAW, combine, read

WORK=ROOT/'work/w57'
CAMPAIGN=ROOT/'work/w57-r/pilot02'
LOCAL=ZoneInfo('Asia/Shanghai')
BG=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def stage_and_push(message):
    small=str(R.relative_to(ROOT))
    git('add','--','AGENTS.md','README.md','docs/S1_DAILY_WORKLOG.md',str(REL/'current.json'),small,
        f':(exclude){small}/episodes/**',':(exclude)**/*.lock')
    git('add','--',str(REL/'w57/episodes/index.json'))
    if subprocess.call(['git','diff','--cached','--quiet'],cwd=ROOT):
        git('commit','-m',message)
    git('fetch','origin','main')
    git('merge','--no-edit','origin/main')
    git('push','origin','HEAD:main')
    local=git('rev-parse','HEAD')
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    if local!=remote:raise RuntimeError('Remote main no longer equals the pushed commit')
    print('Verified main',remote,flush=True)


def refresh(finished=False):
    idx=combine(finished=finished)
    p=ROOT/REL/'current.json';current=read(p)
    heads=idx['observed_on_policy']
    ranks={'E30':30,'E60':60,'E300':300,'E600':600}
    highest=max((a['level'] for a in idx['original_qualifications']),key=ranks.get,default=None)
    current.update(date=datetime.now(LOCAL).date().isoformat(),execution_snapshot=idx['as_of_shanghai'],
                   status='W57_ended' if finished else 'W57_executing')
    current['W57'].update(status='ended' if finished else 'executing',
        new_PPO_samples=sum(h['total_distinct'] for h in heads.values()),
        new_PPO_updates=sum(h['version'] for h in heads.values()),
        counts_as_of_shanghai=idx['as_of_shanghai'],highest_W57_original_at_snapshot=highest,
        actual_physics_starts=idx['distinct_physics_attempts'],release_assets_verified=idx['verified_release_assets'])
    write(p,current)
    return idx


def run_script(name,*args):
    subprocess.run([sys.executable,'-u','scripts/'+name,*map(str,args)],cwd=ROOT,check=True)


def all_terminal():
    q=R/'dispatch/frozen-evaluations.json';pr=R/'dispatch/private-short-A.json'
    if not q.exists() or read(q)['stage']!='queue_ended':return False
    if not pr.exists() or read(pr)['status'] not in ('ended','not_run'):return False
    pairs=[R/'episodes'/f'w57-m1-b-{c}-short-a01.json' for c in ('a','b')]
    if all(p.exists() for p in pairs) and not (R/'m/closed-loop-short.json').exists():return False
    return (CAMPAIGN/'final.json').is_file()


def checkpoint_fingerprint():
    ended=0
    for p in (R/'attempts').glob('*.json'):
        ended+=int('ended_unix_s' in read(p))
    accepted=len(list((R/'r').glob('*/*/update-*.json')))
    verified=sum(read(p).get('status')=='verified' for p in (R/'delivery').glob('*.json'))
    date=datetime.now(LOCAL).date().isoformat()
    return [ended,accepted//4,verified,date]


def active_publisher():
    out=subprocess.check_output(['tmux','list-panes','-a','-F','#{session_name} #{pane_dead}'],text=True)
    return any(s.startswith('w57-direct-delivery') and s.endswith(' 0') for s in out.splitlines())


def recover_publication():
    # A verified D57-native alternate is used only after the observed API/transport failure.
    if active_publisher() or read(R/'dispatch/frozen-evaluations.json')['stage']=='queue_ended':return
    log=WORK/'direct-delivery.log'
    tail=log.read_bytes()[-5500:].decode(errors='replace') if log.exists() else ''
    if tail.rstrip().lower().endswith(': eof') or any(k in tail.lower() for k in (
            'tls handshake timeout','http 422','http 502','connection reset','i/o timeout')):
        command=('source scripts/autodl_d57_env.sh && source scripts/autodl_network.sh clash >/dev/null'
                 ' && exec taskset -c 16-19 python -u scripts/publish_s1_w57.py follow-episodes >>work/w57/direct-delivery.log 2>&1')
        subprocess.run(['tmux','new-session','-d','-s','w57-direct-delivery-d57-route','-c',str(ROOT),command],check=True)


def final_delivery():
    run_script('collect_s1_w57_planner_physics.py')
    run_script('summarize_s1_w57_timing.py','--campaign',CAMPAIGN,'--training')
    frozen=[read(p) for p in (R/'attempts').glob('*.json')]
    if any(r['arm'] in ('R','PR','blind') and r['task_mode']!='training' and
           (RAW/r['attempt']/'timing-summary.json').is_file() for r in frozen):
        run_script('summarize_s1_w57_timing.py','--campaign',CAMPAIGN)
    run_script('analyze_s1_w57_episodes.py','--material','M1')
    run_script('publish_s1_w57.py','ready-episodes')
    run_script('publish_s1_w57_learning.py')
    rows=[read(p) for p in sorted((R/'attempts').glob('*.json'))]
    for row in rows:
        attempt=row['attempt']
        if 'ended_unix_s' not in row:raise RuntimeError('Attempt still live: '+attempt)
        if (R/'episodes'/(attempt+'.json')).exists():continue
        files=[p for p in (RAW/'jobs'/attempt,RAW/attempt) if p.exists()]
        if files:run_script('publish_s1_w57.py','bundle',attempt+'-retained-prefix',*files)
    active=RAW/'w57/resource-trace.jsonl';captured=WORK/'resource-trace-through-last-physics.jsonl'
    if not captured.exists():
        b=active.read_bytes()
        captured.write_bytes(b[:b.rfind(b'\n')+1])
    metadata=sorted((R/'episodes').glob('*.json'))
    credit=[p for head in ('full','blind') for p in sorted((CAMPAIGN/head).glob('credit-*.npz'))]
    run_script('publish_s1_w57.py','bundle','w57-final-episode-metadata-and-credit',*metadata,captured,*credit)
    rel,remote=release_assets()
    assets=[a for f in sorted((R/'delivery').glob('*.json')) for a in read(f)['assets']]
    for a in assets:
        actual=remote[a['name']]
        if (actual['size'],actual.get('digest'),actual['state'])!=(a['size'],a['digest'],'uploaded'):
            raise ValueError('Unverified final Release asset: '+a['name'])
    return rows,dict(release=rel['url'],release_id=rel['databaseId'],asset_count=len(assets),
                     bytes=sum(a['size'] for a in assets),verified_unix_s=time.time())


def final_summary(rows,receipt):
    heads=read(CAMPAIGN/'final.json')
    mechanism=[r for r in rows if r['task_mode'] in ('short','recovered') and r['material']=='M1'
               and r['settings'].get('planner_history','full')=='full']
    primary=[r for r in mechanism if r['arm']!='blind']
    blind=[r for r in mechanism if r['arm']=='blind']
    checks=dict(primary_mechanisms=dict(attempted=len(primary),declared=16,passed=sum(r.get('success') is True for r in primary)),
                blind_mechanisms=dict(attempted=len(blind),declared=4 if heads['blind'] else 0,
                                      passed=sum(r.get('success') is True for r in blind)),
                distinct_transitions={'full':heads['full'],'blind':heads['blind']},
                paired_results=str(REL/'w57/paired-mechanisms.json'),private_intervention=read(R/'dispatch/private-short-A.json'),
                technical_interruptions=[r['attempt'] for r in rows if r['status']=='technical_interruption'],
                original_long_decision=read(R/'dispatch/final-qualification.json'),publication=receipt)
    metric=lambda r: {k:r.get(k) for k in ('phase_status','observed_physical_s','original_slip_integral_s','max_principal_strain')}
    for h,stem in [('full','w57-m1-p-a-short-a01'),('private_reset','w57-m1-p-a-private-short-a01')]:
        path=R/'mechanics'/(stem+'.json')
        if path.exists():checks.setdefault('matched_A_short_history',{})[h]=metric(read(path))
    stage_and_push('Publish final W57 measurements, qualification decisions and verified original receipts')
    today=datetime.now(LOCAL).date().isoformat()
    append_day(today)
    idx=refresh(True)
    idx['completion']=checks; write(R/'index.json',idx)
    readme=ROOT/'README.md';body=readme.read_text()
    lead=f"## W57 实验结果\n\nW57 的正式机制任务已执行 {len(primary)}/16，达到全部严格判据 {checks['primary_mechanisms']['passed']} 个。新 timing head full/blind 分别累计 {heads['full']}/{heads['blind']} 条不同的 on-policy 转移。各任务的失败、原任务资格、交互及 Release 验证见[最终索引]({REL}/w57/index.json)。"
    readme.write_text(body.replace('## W57 实验运行中',lead,1))
    stage_and_push('Close W57 with actual independent, joint and original-task results')


def main():
    lock=(WORK/'github-integrator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    last=None
    while True:
        try:
            if all_terminal():
                rows,receipt=final_delivery()
                final_summary(rows,receipt)
                return
            recover_publication()
            fingerprint=checkpoint_fingerprint()
            if fingerprint!=last:
                refresh()
                stage_and_push('Update actual W57 physics, learning counts and publication receipts')
                last=fingerprint
        except (subprocess.CalledProcessError,RuntimeError) as exc:
            print(datetime.now(LOCAL).isoformat(),'Integration pending',str(exc),flush=True)
        time.sleep(60)


if __name__=='__main__':main()
