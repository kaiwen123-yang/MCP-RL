"""Transfer the physical slot from the exact N/A r54 E30 child to frozen F/B r53."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from publish_s1_go2_w54 import write
from run_s1_w55_f import ROOT, SMALL

RUN = 'tm-s1-go2-local-readiness-r12'
N_CASE = (ROOT.parent/'MCP-RL-w55-n/results/terrain_memory/simulation'/RUN/'w55/n/N-A-r54')
B_CASE = SMALL/'F-B-r53'
RECORD = SMALL/'slot-N-A-r54-E30-to-F-B-r53.json'


def read(path):
    return json.loads(path.read_text())


def session(name, cpus, script, *args):
    if subprocess.run(['tmux','has-session','-t','='+name],capture_output=True).returncode == 0:
        return
    command = [sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,args)]
    log = ROOT/'work/w55-f'/(name+'.log')
    shell = (f'source scripts/autodl_d57_env.sh && exec taskset -c {cpus} '+shlex.join(command)
             +' > '+shlex.quote(str(log))+' 2>&1')
    subprocess.run(['tmux','new-session','-d','-s',name,'-c',str(ROOT),shell],check=True)
    subprocess.run(['tmux','set-option','-t',name,'remain-on-exit','on'],check=True)


def main():
    b_stage = B_CASE/'E30.json'
    b = read(b_stage)
    n_stage = N_CASE/'E30.json'
    record = dict(status='waiting_for_F_B_E30_and_natural_N_A_r54_E30_exit',pid=os.getpid(),
                  tmux='s1-w55-f-b-after-n-r54',source_stage=str(n_stage),target='F-B-r53/E60→E300→E600',
                  own_predecessor_stage=str(b_stage),own_predecessor_attempt=b['attempt'],
                  created_unix_s=time.time(),deadline=None,wall_deadline_enabled=False,
                  kill_on_conversation_end=False,maximum_heavy_physical_jobs=2)
    if RECORD.exists():
        earlier=read(RECORD)
        if earlier.get('own_predecessor_attempt')==b['attempt']:
            record['created_unix_s']=earlier['created_unix_s']
    write(RECORD,record)
    while True:
        b = read(b_stage)
        if b.get('attempt') != record['own_predecessor_attempt']:
            raise RuntimeError('Reserved F/B predecessor changed')
        if b['status'] not in ('running','declared','passed'):
            record.update(status='own_E30_did_not_qualify',own_predecessor_status=b['status'])
            write(RECORD,record)
            return
        if b['status']=='passed' and n_stage.exists():
            n = read(n_stage)
            job = read(N_CASE/'job.json')
            if (job['route'],job['case'],job['seed'],job['controller_build']) != ('N','A',1,'S5-r54'):
                raise ValueError('N stage identity no longer matches its slot reservation')
            expected = job['job_id']+'-eval-PANN_PPO_S5-s1-E30-attempt01'
            if n.get('attempt') and (n['attempt']!=expected or n.get('job')!=job['job_id'] or n['level']!='E30'):
                raise ValueError('Unexpected attempt in the reserved original N stage')
            if n.get('pid') and 'source_pid' not in record:
                record.update(source_attempt=expected,source_pid=n['pid'],source_job=job['job_id'],
                              status='following_exact_N_E30_child')
                write(RECORD,record)
            if (n.get('attempt')==expected and n.get('pid') and n['status'] not in ('running','declared')
                    and n.get('ended_unix_s') is not None and n.get('exit_code') is not None):
                if n['pid'] != record['source_pid']:
                    raise ValueError('The reserved N simulation child changed')
                proc=Path('/proc')/str(n['pid'])/'cmdline'
                try:
                    commands=proc.read_bytes().split(b'\0')
                except FileNotFoundError:
                    commands=[]
                if expected.encode() not in commands:
                    record.update(source_status=n['status'],source_child_exit_code=n['exit_code'],
                                  source_child_ended_unix_s=n['ended_unix_s'],source_physical_s=n.get('physical_s'),
                                  source_physical_steps=n.get('physical_samples'),slot_released_unix_s=time.time(),
                                  status='starting_frozen_F_B_ladder')
                    write(RECORD,record)
                    break
        time.sleep(5)
    # The coordinator can release A's slot earlier while N is still running.
    b_successor=B_CASE/'E60.json'
    if b_successor.exists():
        actual=read(b_successor)
        original_A=read(SMALL/'F-A-r53/E60.json')
        if actual.get('job')!=b['job'] or original_A.get('ended_unix_s') is None:
            raise RuntimeError('Existing F/B successor does not match its original case and slot')
        record.update(status='F_B_already_dispatched_from_released_F_A_slot',
                      actual_source_attempt=original_A['attempt'],actual_source_ended_unix_s=original_A['ended_unix_s'],
                      actual_B_successor=actual.get('attempt'),actual_B_started_unix_s=actual.get('started_unix_s'),
                      original_N_reservation_reassigned_to='F-A-r55/E30')
        write(RECORD,record)
        return
    session('s1-w55-f-b-r53-ladder','0-3','continue_s1_w55_f.py','--case','B','--build','S5-r53',
            '--selection',SMALL/'r53-selection.json')
    session('s1-w55-f-b-collect','3','collect_s1_w55_f.py','--case-dir',B_CASE,'--levels','E60','E300','E600')
    record.update(status='F_B_ladder_and_collector_dispatched',ladder_tmux='s1-w55-f-b-r53-ladder',
                  collector_tmux='s1-w55-f-b-collect',dispatched_unix_s=time.time())
    write(RECORD,record)


if __name__ == '__main__':
    main()
