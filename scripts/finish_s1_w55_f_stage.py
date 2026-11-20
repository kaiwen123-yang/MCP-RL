"""Deliver each recorded stage of a frozen F case from D57 as it concludes."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from deliver_s1_w55_stage import deliver_stage
from publish_s1_go2_w54 import write
from record_s1_w55 import ROOT, REL, WINDOW, LEVELS, read


def git(*args, output=False):
    return subprocess.run(['git', *map(str,args)], cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE if output else None).stdout


def push():
    git('fetch', 'origin', 'main')
    git('merge', '--no-edit', 'origin/main')
    revision = git('rev-parse','HEAD',output=True).strip()
    git('push','origin','HEAD:main')
    git('fetch','-q','origin','main')
    git('merge-base','--is-ancestor',revision,'origin/main')
    print('Verified on main',revision,flush=True)


def navigation():
    index = read(WINDOW/'index.json')
    phase = read(WINDOW/'phase.json')
    current_path = ROOT/REL/'current.json'
    current = read(current_path)
    current.update(date=time.strftime('%Y-%m-%d Asia/Shanghai'),status='W55_'+index['status'])
    info = current['W55']
    info.pop('active_F_stages_at_snapshot',None)
    info.update(status=index['status'],E30_passes=index['E30_passes'],
                highest_passed_original_stage=index['highest_passed_original_stage'],
                completed_original_attempts=index['completed_original_attempts'],
                recorded_legal_round_trips=index['legal_round_trips'],
                active_stages_at_snapshot=[dict(attempt=r['attempt'],pid=r.get('pid'),
                    physical_s=r.get('live',{}).get('physical_time_s')) for r in phase['cases']
                    if r['status']=='running' and r.get('pid_running_at_snapshot')],
                snapshot_unix_s=phase['updated_unix_s'])
    current['publication']['verified_assets'] = index.get('publication',{}).get('asset_count',0)
    write(current_path,current)
    config_path = ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json'
    config=read(config_path)
    config['next_authorized_plan'].update(status=index['status'], E30_passes=index['E30_passes'],
        highest_passed_original_stage=index['highest_passed_original_stage'],
        last_actual_index=str(WINDOW.relative_to(ROOT)/'index.json'))
    # Keep the protocol's existing readable Chinese JSON.
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    return [current_path,config_path,WINDOW/'index.json',WINDOW/'phase.json']


def compact(paths, message):
    relative = [p.relative_to(ROOT) for p in paths]
    git('add','--',*relative)
    if subprocess.run(['git','diff','--cached','--quiet','--',*relative],cwd=ROOT).returncode:
        git('commit','--only','-m',message,'--',*relative)


def record():
    subprocess.run([sys.executable, str(ROOT/'scripts/record_s1_w55.py'),
        '--f',str(ROOT.parent/'MCP-RL-w55-f'),'--n',str(ROOT.parent/'MCP-RL-w55-n'),
        '--c',str(ROOT.parent/'MCP-RL-w55-c'),'--status','original_stage_results_updated'],check=True)


def watch_stage(f, label, level):
    stage=f/REL/'w55/f'/label/(level+'.json')
    # The existing F collector writes its analysis commit after the full video.
    while True:
        if stage.is_file():
            row=read(stage)
            if row.get('reason')=='process_exited_without_completed_result':
                raise RuntimeError(f'Original process stopped without a recorded result: {stage}')
            if row.get('video'):
                analysis=REL/'w55/f'/(row['attempt']+'-analysis.json')
                revision=git('log','-1','--format=%H','codex/w55-f','--',analysis,output=True).strip()
                if revision:
                    break
        time.sleep(20)
    receipt=WINDOW/'delivery'/(label+'-'+level).lower()/'delivery.json'
    with (ROOT/'work/w55-integration.lock').open('a') as integration:
        fcntl.flock(integration,fcntl.LOCK_EX)
        git('merge','--no-edit',revision)
        record()
        compact(navigation(),f'Record W55 {label} {level} original outcome and qualification')
        push()
    if not receipt.is_file():
        deliver_stage(f,label,level)
    with (ROOT/'work/w55-integration.lock').open('a') as integration:
        fcntl.flock(integration,fcntl.LOCK_EX)
        record()
        compact(navigation()+[receipt.parent],f'Verify W55 {label} {level} complete Release assets')
        push()
    return row['status']=='passed'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=('A','B'))
    parser.add_argument('--build',required=True)
    parser.add_argument('--levels',nargs='+',choices=LEVELS,required=True)
    args=parser.parse_args()
    lane=ROOT.parent/'MCP-RL-w55-f'
    label=f'F-{args.case}-{args.build.removeprefix("S5-")}'
    for level in args.levels:
        if not watch_stage(lane,label,level):
            break
