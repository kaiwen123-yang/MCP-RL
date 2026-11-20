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
                 rule='N-A-r54 E30 exits, or F-B-r53 exhausts its current original ladder. At most two heavy robot jobs.',
                 created_unix_s=time.time())
    write(STATUS,current)
    while True:
        n=read(n_stage)
        if (n['attempt'],n['pid'])!=(n_initial['attempt'],n_initial['pid']):
            raise RuntimeError('Reserved N stage changed identity')
        b=read(B_CASE/'E60.json')
        if (b['attempt'],b['pid'])!=(b_initial['attempt'],b_initial['pid']):
            raise RuntimeError('Reserved first B stage changed identity')
        release=child_ended(n_stage) or released_b_slot()
        if release:
            break
        time.sleep(5)
    current.update(status='launching_r55_A_original_E30',slot_source_attempt=release['attempt'],
                   released_evaluation_child_pid=release['pid'],release_stage_status=release['status'],
                   slot_released_unix_s=time.time())
    write(STATUS,current)
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
    while not (child_ended(n_stage) and released_b_slot()):
        time.sleep(5)
    bcase=SMALL/'F-B-r55'
    session('s1-w55-f-b-r55-ladder','0-3','continue_s1_w55_f.py','--case','B','--build','S5-r55','--selection',selection)
    session('s1-w55-f-b-r55-collect','3','collect_s1_w55_f.py','--case-dir',bcase,'--levels','E30','E60','E300','E600')
    current.update(status='r55_both_case_ladders_dispatched',b_ladder_unix_s=time.time())
    write(STATUS,current)


if __name__=='__main__':
    main()
