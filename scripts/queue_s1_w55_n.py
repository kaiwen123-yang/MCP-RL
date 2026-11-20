"""Release the second physical slot from the exact F/B E30 to N/A E30."""
import json, os, subprocess, sys, time
from pathlib import Path
from run_s1_w55_n import ROOT, SMALL, launch, write
def main():
    root=Path('/root/autodl-tmp/MCP-RL-w55-f')
    stage=root/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/f/F-B-r53/E30.json'
    dest=SMALL/'slot-F-B-r53-E30-to-N-A-r54.json'
    recorded=json.loads(stage.read_text());expected=recorded['attempt']
    state=dict(status='waiting_for_natural_stage_end',pid=os.getpid(),tmux='s1-w55-n-r54-slot',
        source_stage=str(stage),source_attempt=expected,source_pid=recorded['pid'],
        target='N-A-r54/E30',deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False,
        created_unix_s=time.time(),physical_slots=2)
    write(dest,state)
    while True:
        # The state writer uses direct JSON writes. A partially written sample
        # is an incomplete observation of the same running process.
        try:row=json.loads(stage.read_text())
        except json.JSONDecodeError:
            time.sleep(1);continue
        if row['attempt']!=expected:raise RuntimeError('The reserved source attempt changed')
        if row['status'] not in ('declared','running'):
            if 'ended_unix_s' not in row:raise RuntimeError('Source terminal decision lacks the child exit receipt')
            break
        time.sleep(3)
    state.update(source_end_unix_s=row['ended_unix_s'],source_status=row['status'],status='launching',slot_released_unix_s=time.time())
    write(dest,state)
    launch('S5-r54','A','E30')
    launched=json.loads((SMALL/'N-A-r54/E30.json').read_text())
    state.update(status='launched',target_tmux=launched['tmux'],target_job=launched['job'],launched_unix_s=time.time())
    write(dest,state)
if __name__=='__main__':main()
