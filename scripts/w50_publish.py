#!/usr/bin/env python3
"""Publish the completed finite W50 result; never launch further physics."""
import fcntl,json,os,sys,time,traceback
from pathlib import Path
from publish_s1_go2_w50 import ROOT,SMALL,LARGE,compact,deliver,resume_delivery,write

def main():
 p=SMALL/'w50/publication.json'
 r=dict(status='running',pid=os.getpid(),started_unix_s=time.time(),deadline=None,wall_deadline_enabled=False,
  source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  new_physics=0,new_PPO=0,server_originals_retained=True,computer_download=False)
 write(p,r)
 try:
  if '--resume' in sys.argv:
   delivery=resume_delivery('finite')
  else:
   delivery=deliver('finite',[LARGE/'w50',ROOT/'work/r11-frozen/S5-r38/agent_server',
    ROOT/'work/r11-frozen/S5-r38/identity.json',ROOT/'work/w50-implementation/native-bridge.so',
    ROOT/'work/w50-implementation/bridge-build.json',ROOT/'work/w50-implementation/build.log'])
  r.update(status='completed' if delivery['status']=='published' else 'publication_pending',delivery=delivery)
 except Exception:
  r.update(status='publication_pending',traceback=traceback.format_exc());raise
 finally:
  r['ended_unix_s']=time.time();write(p,r)
  with (SMALL/'index.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   phase=json.loads((SMALL/'w50/phase.json').read_text())
   phase.update(status='completed_finite_negative_published' if r['status']=='completed' else 'finite_negative_publication_pending',
    publication=dict(status=r['status'],receipt=str(p.relative_to(ROOT)),finished_unix_s=r['ended_unix_s']))
   if r['status']=='completed':phase['continuation']='W50 complete. No W50 simulation or training remains active. Do not rerun used finite/deployment budgets. Any new experiment requires a new user instruction.'
   write(SMALL/'w50/phase.json',phase)
   idx=json.loads((SMALL/'index.json').read_text());idx['w50']=phase;write(SMALL/'index.json',idx)
  compact([p,SMALL/'w50/phase.json'], 'Complete W50 finite-negative publication state')

if __name__=='__main__':main()
