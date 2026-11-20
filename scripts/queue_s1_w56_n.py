"""Admit r56 only after the exact prior N and B/r53 qualified ladder release."""
import json,os,time
from pathlib import Path
from run_s1_w56_n import SMALL,write,launch
ROOT=Path(__file__).resolve().parents[1];RUN='tm-s1-go2-local-readiness-r12'
F=Path('/root/autodl-tmp/MCP-RL-w55-f/results/terrain_memory/simulation')/RUN/'w55/f/F-B-r53'
OLD=Path('/root/autodl-tmp/MCP-RL-w55-n/results/terrain_memory/simulation')/RUN/'w55/n/N-A-r54/E30.json'
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
def finished(path,job=None):
    while True:
        row=read(path)
        if row and row.get('attempt') and row.get('ended_unix_s') and row.get('status') not in ('declared','running'):
            if job and row['job']!=job:raise ValueError('Frozen source case identity changed')
            return row
        time.sleep(3)
def main():
    receipt=SMALL/'queue-r56.json'
    job=json.loads((F/'job.json').read_text())['job_id']
    r54=finished(OLD)
    verdict=read(SMALL/'same-input-r56.json')
    if not verdict or verdict['rows'][0]['supported_progress']['first_constraint_code']!=0:raise RuntimeError('Supported counterexample has not passed')
    state=dict(status='waiting_for_B_r53_ladder',pid=os.getpid(),tmux='s1-w55-n-r56-slot',N_r54_predecessor=r54['attempt'],source_job=job,
        target='N-A-r56/E30',deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False,started_unix_s=time.time())
    write(receipt,state)
    for level in ('E60','E300','E600'):
        state.update(source_stage=str(F/(level+'.json')),source_level=level);write(receipt,state)
        result=finished(F/(level+'.json'),job)
        if not result.get('success') or level=='E600':break
    state.update(source_attempt=result['attempt'],source_end_unix_s=result['ended_unix_s'],status='launching');write(receipt,state)
    launch('S5-r56','A','E30')
    row=json.loads((SMALL/'N-A-r56/E30.json').read_text());state.update(status='launched',target_job=row['job'],target_tmux=row['tmux'],launched_unix_s=time.time())
    write(receipt,state)
if __name__=='__main__':main()
