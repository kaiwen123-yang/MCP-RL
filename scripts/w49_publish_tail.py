#!/usr/bin/env python3
"""D57-only incremental W48 technical tail, then generated W49 finite originals."""
import json,os,time,traceback
from pathlib import Path
from publish_s1_go2_w48 import resume_delivery as old_resume
from publish_s1_go2_w49 import ROOT,SMALL,LARGE,write,deliver,compact
def main():
 p=SMALL/'w49/publication.json'
 r=dict(status='running',pid=os.getpid(),started_unix_s=time.time(),deadline=None,wall_deadline_enabled=False,
  affinity=sorted(os.sched_getaffinity(0)),old_technical={},server_originals_retained=True,computer_download=False)
 write(p,r)
 try:
  for name in ['w48-a','w48-b']:
   r['active']='old_technical_'+name;write(p,r)
   r['old_technical'][name]=old_resume(name);write(p,r)
  r['active']='w49_finite_originals';write(p,r)
  roots=[LARGE/'w49',ROOT/'work/r11-frozen/S5-r37/agent_server',ROOT/'work/r11-frozen/S5-r37/lib/libmujoco.so.3.2.3',
   ROOT/'work/w49-implementation/native-bridge.so',ROOT/'work/w49-implementation/finite-before-service-relink-identity.json',
   ROOT/'work/w49-implementation/finite-before-service-relink-agent_server']
  r['finite']=deliver('finite',roots)
  r['status']='completed' if r['finite']['status']=='published' and all(x['status']=='published' for x in r['old_technical'].values()) else 'publication_pending'
 except Exception:r.update(status='technical_failure',traceback=traceback.format_exc());raise
 finally:
  r['ended_unix_s']=time.time();write(p,r)
  compact([p],'Record W49 incremental original publication state')
if __name__=='__main__':main()
