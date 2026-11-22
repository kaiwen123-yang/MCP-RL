"""One preselected A short closed-loop private-history comparison with its own original qualification."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from publish_s1_go2_w54 import write
from queue_s1_w57_evaluations import BACKGROUND, R, WORK, read, source_for


def physical(task,level):
    src=source_for('P')
    stem='w57-m1-p-a-private-'+('original-'+level.lower() if task=='original' else 'short')+'-a01'
    receipt=R/'attempts'/(stem+'.json')
    if receipt.exists():
        while 'ended_unix_s' not in read(receipt):time.sleep(15)
        return read(receipt)
    cmd=[sys.executable,'-u','scripts/s1_w56_slot.py','--priority','F','--label',stem,'--',
         sys.executable,'-u','scripts/run_s1_w57.py','episode','--arm','P','--case','A','--material','M1',
         '--history','private_reset','--task',task,'--level',level,'--attempt',stem]
    env=dict(os.environ,W57_TMUX='w57-private-short-A',PYTHONPATH=str(src/'src')+':'+str(src/'work/upstreams/mujoco_mpc/python'))
    with (WORK/(stem+'.log')).open('a') as log:
        result=subprocess.run(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not receipt.is_file():raise RuntimeError(f'No receipt for {stem}, exit {result.returncode}')
    return read(receipt)


def run():
    gate=R/'p/physical/private-short-A-eligibility.json'
    path=R/'dispatch/private-short-A.json'
    state=dict(case='A',mode='short',observer='M1',planning='P1',timing='R0',
               alternative='private_reset',paired_full_attempt='w57-m1-p-a-short-a01',
               required_evidence=str(gate.relative_to(R)),scope='Secondary causal intervention, outside 16 and outside matched blind.',
               origin='Preselected A before any official P/A short outcome. Require matched real subsequent loads and mapped differences exceeding same-input repeatability in original A and short A.',
               status='waiting_for_full_short_A_evidence',**BACKGROUND)
    write(path,state)
    while not gate.exists():
        main=R/'dispatch/frozen-evaluations.json'
        if main.exists() and read(main)['stage']=='queue_ended':
            full=R/'attempts/w57-m1-p-a-short-a01.json'
            if not full.exists() or read(full)['status']=='technical_interruption':
                state.update(status='not_run',reason='no completed qualified full A short observation')
                write(path,state)
                return
        time.sleep(30)
    evidence=read(gate)
    state['eligibility']=evidence
    if not evidence['eligible_for_separate_physics']:
        state.update(status='not_run',reason='matched realized load or mapped authority missing')
        write(path,state)
        return
    state['status']='executing';write(path,state)
    a=physical('original','E30')
    state['original_E30']={k:a.get(k) for k in ('attempt','status','reason')}
    if a['status']=='passed':
        short=physical('short','E60')
        state['short']={k:short.get(k) for k in ('attempt','status','reason','mechanism')}
        longer=physical('original','E60')
        state['original_E60']={k:longer.get(k) for k in ('attempt','status','reason')}
    else:
        state['reason']='own E30 did not pass; retain original outcome and matched-state intervention'
    state['status']='ended'
    write(path,state)


if __name__=='__main__':
    with (WORK/'private-short-A.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
