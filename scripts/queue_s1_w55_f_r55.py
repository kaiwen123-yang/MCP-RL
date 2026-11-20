"""Dispatch r55 in released original slots, then advance only its own evidence."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from run_s1_w55_f import ROOT, SMALL, launch
from publish_s1_go2_w54 import write
from queue_s1_w55_f_b import N_CASE, B_CASE, read, session
from qualify_s1_w55_f_r55 import review

STATUS=SMALL/'r55-dispatch.json'
TARGET=SMALL/'F-A-r55'
N56_MARKER=ROOT.parent/'MCP-RL-w55-n-r56/work/w55-n-r56/slot-released.json'


def child_ended(stage):
    if not stage.is_file():
        return None
    record=read(stage)
    if record.get('status') in ('declared','running') or record.get('exit_code') is None or not record.get('pid'):
        return None
    try:
        argv=(Path('/proc')/str(record['pid'])/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        argv=[]
    return record if record['attempt'].encode() not in argv else None


def released_b_slot():
    for level in ('E60','E300','E600'):
        record=child_ended(B_CASE/(level+'.json'))
        if not record:
            return None
        if record['status']!='passed' or level=='E600':
            return record


def n56_released():
    if not N56_MARKER.is_file():
        return None
    release=read(N56_MARKER)
    if release.get('target')!='N-A-r56/E30' or not release.get('released_unix_s'):
        raise RuntimeError('Unexpected N/r56 release record')
    if release['status']=='explicitly_released' and release.get('reason'):
        return release
    if release['status']!='naturally_ended':
        raise RuntimeError('Unrecognized N/r56 slot disposition')
    stage=Path(release['stage_file'])
    row=child_ended(stage)
    if not row:
        return None
    expected=release['job']+'-eval-PANN_PPO_S5-s1-E30-attempt01'
    if (row['attempt'],row['pid'],row['exit_code'])!=(expected,release['child_pid'],release['child_exit_code']):
        raise RuntimeError('N/r56 release does not match its original exited child')
    return release


def commit_scientific_review(evidence):
    comparison='r55-selection.json' if evidence['normal_sustained_motion'] else 'r55-motion-comparison.json'
    paths=[str((SMALL/name).relative_to(ROOT)) for name in (comparison,'r55-dispatch.json')]
    with (SMALL/'index.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=ROOT).returncode==1:
            subprocess.run(['git','commit','--only','-m','Record r55 independent original E30 motion review','--',*paths],cwd=ROOT,check=True)


def main():
    n_stage=N_CASE/'E30.json'
    n_initial=read(n_stage)
    b_initial=read(B_CASE/'E60.json')
    current=dict(status='waiting_for_a_naturally_released_physical_slot',tmux='s1-w55-f-a-r55-slots',pid=os.getpid(),
                 deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False,
                 source_N=dict(attempt=n_initial['attempt'],pid=n_initial['pid']),
                 source_B=dict(attempt=b_initial['attempt'],pid=b_initial['pid']),
                 rule='r55 A takes the slot released by N-A-r54 E30. F-B-r53 retains its separate ladder slot, next reserved for N-A-r56, then eligible r55 B.',
                 second_slot_N56_release_marker=str(N56_MARKER),
                 created_unix_s=time.time())
    prior={}
    if STATUS.is_file():
        prior=read(STATUS)
        if prior.get('source_N')==current['source_N'] and prior.get('source_B')==current['source_B']:
            current['created_unix_s']=prior['created_unix_s']
    write(STATUS,current)
    while True:
        n=read(n_stage)
        if (n['attempt'],n['pid'])!=(n_initial['attempt'],n_initial['pid']):
            raise RuntimeError('Reserved N stage changed identity')
        b=read(B_CASE/'E60.json')
        if (b['attempt'],b['pid'])!=(b_initial['attempt'],b_initial['pid']):
            raise RuntimeError('Reserved first B stage changed identity')
        release=child_ended(n_stage)
        if release:
            break
        time.sleep(5)
    current.update(status='launching_r55_A_original_E30',slot_source_attempt=release['attempt'],
                   released_evaluation_child_pid=release['pid'],release_stage_status=release['status'],
                   slot_released_unix_s=prior.get('slot_released_unix_s',time.time()))
    write(STATUS,current)
    if (TARGET/'E30.json').is_file():
        existing=read(TARGET/'job.json')
        if (existing['controller_build'],existing['case'],existing['seed'])!=('S5-r55','A',1):
            raise ValueError('Existing A E30 has a different frozen identity')
        current.update(status='following_existing_r55_A_original_E30',job=existing['job_id'])
        write(STATUS,current)
    else:
        launch('S5-r55','A','E30')
    session('s1-w55-f-a-r55-e30-collect','13','collect_s1_w55_f.py','--case-dir',TARGET,'--levels','E30')
    while True:
        stage=read(TARGET/'E30.json')
        if stage['status'] not in ('running','declared') and stage['status']!='passed':
            current.update(status='r55_A_E30_did_not_qualify',scientific_status=stage['status'],reason=stage.get('reason'))
            write(STATUS,current)
            return
        if stage.get('video') and (SMALL/(stage['attempt']+'-analysis.json')).is_file():
            break
        time.sleep(5)
    selection=SMALL/'r55-selection.json'
    if selection.is_file():
        evidence=read(selection)
        if evidence.get('attempt')!=stage['attempt'] or evidence.get('build')!='S5-r55' or evidence.get('normal_sustained_motion') is not True:
            raise ValueError('Existing scientific selection does not identify this passed original A/r55 E30')
    else:
        evidence=review()
    current.update(attempt=stage['attempt'],status=evidence['status'],review_unix_s=time.time())
    write(STATUS,current);commit_scientific_review(evidence)
    if not evidence['normal_sustained_motion']:
        return
    selection=SMALL/'r55-selection.json'
    session('s1-w55-f-a-r55-ladder','10-13','continue_s1_w55_f.py','--case','A','--build','S5-r55','--selection',selection)
    session('s1-w55-f-a-r55-collect','13','collect_s1_w55_f.py','--case-dir',TARGET,'--levels','E60','E300','E600')
    current.update(status='r55_A_ladder_dispatched_waiting_for_second_slot',a_ladder_unix_s=time.time())
    write(STATUS,current)
    while True:
        release_b=released_b_slot()
        release_n56=n56_released()
        if release_b and release_n56:
            break
        time.sleep(5)
    current.update(second_slot_previous_F_attempt=release_b['attempt'],second_slot_N56_release=release_n56)
    write(STATUS,current)
    bcase=SMALL/'F-B-r55'
    session('s1-w55-f-b-r55-ladder','0-3','continue_s1_w55_f.py','--case','B','--build','S5-r55','--selection',selection)
    session('s1-w55-f-b-r55-collect','3','collect_s1_w55_f.py','--case-dir',bcase,'--levels','E30','E60','E300','E600')
    current.update(status='r55_both_case_ladders_dispatched',b_ladder_unix_s=time.time())
    write(STATUS,current)


if __name__=='__main__':
    main()
