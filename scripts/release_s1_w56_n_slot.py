"""Observe N/A E30's physical child exit independently of artifact rendering."""
import time
from run_s1_w56_n import ROOT,SMALL,write
from queue_s1_w56_n import finished
stage=SMALL/'N-A-r56/E30.json'
row=finished(stage)
target=ROOT/'work/w55-n-r56/slot-released.json'
if not target.exists():
    write(target,dict(status='naturally_ended',target='N-A-r56/E30',released_unix_s=row['ended_unix_s'],
        job=row['job'],attempt=row['attempt'],child_pid=row['pid'],child_exit_code=row['exit_code'],
        physical_started=bool(row.get('physical_samples')),stage_file=str(stage),reason=row.get('reason'),
        deadline=None,wall_deadline_enabled=False))
