#!/usr/bin/env python3
"""Close the already-decided W52 after its exact original publisher has exited."""
import argparse,json,time
from pathlib import Path
from publish_s1_go2_w52 import ROOT,SMALL,LARGE,write,compact,deliver
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-pid',type=int,required=True);a=p.parse_args()
 W=SMALL/'w52'
 identity=Path('/proc')/str(a.parent_pid)
 if identity.exists():
  fields=(identity/'stat').read_text().rsplit(') ',1)[1].split()
  if fields[0]!='Z':
   args=(identity/'cmdline').read_bytes()
   if b'run_s1_go2_w52.py' not in args or b'--pair' not in args:
    raise RuntimeError('The supplied PID is not the original W52 paired publisher')
   start=fields[19]
   print('Waiting for the existing W52 original publisher; no new physical task.',flush=True)
   while identity.exists():
    try:
     fields=(identity/'stat').read_text().rsplit(') ',1)[1].split()
     if fields[0]=='Z' or fields[19]!=start:break
    except FileNotFoundError:break
    time.sleep(15)
 receipt=json.loads((W/'delivery/paired-D8/delivery.json').read_text())
 if receipt['status']!='published':
  raise RuntimeError('Original publication remains pending; inspect existing receipts before resuming uploads')
 evidence=deliver('analysis',[LARGE/'w52-analysis'])
 phase=json.loads((W/'phase.json').read_text());paired=json.loads((W/'paired-results.json').read_text())
 if paired['selection_status']!='completed_negative_no_controller_selected':
  raise RuntimeError('Scientific selection changed; do not overwrite it')
 phase.update(status='completed_negative_published',selection=None,
  scientific_selection=dict(status=paired['selection_status'],reason=paired['decision']['reason'],
    source='paired-results.json:full_trajectory_analysis',original_E30_started=0,
    formal_chain='unrun_after_failed_D8_scientific_selection'),
  delivery=receipt,analysis_delivery=evidence,completed_unix_s=time.time(),
  new_physical_starts_after_original_pair=0,deadline=None,wall_deadline_enabled=False)
 write(W/'phase.json',phase)
 import fcntl
 with (SMALL/'index.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  index=json.loads((SMALL/'index.json').read_text())
  index['w52_execution'].update(status='completed_negative_published',selection=None,
   publication_complete=True,updated_unix_s=time.time(),
   next='W52 is complete. No controller selected and no A/B E30 authorized by these results. Preserve all original evidence; any new mechanism requires a new scoped round.')
  write(SMALL/'index.json',index)
 for name in ['AGENTS.md','docs/TERRAIN_MEMORY_PLAN.md','docs/S1_W52_STABILITY_PLAN.md']:
  p=ROOT/name;t=p.read_text()
  t=t.replace('原件/视频直传的完成状态读取现有phase和delivery回执。','原件/视频已由D57直传并核验，见现有phase和delivery回执。')
  t=t.replace('原件与两个全时段视频由原发布进程D57直传；读取phase/delivery确认完成，不重复启动。','原件与两个全时段视频已由D57直传并核验；见phase/delivery，不重复启动。')
  t=t.replace('直接视频和原件由原发布进程从D57继续上传，发布完成以phase和delivery回执为准。','两个直接视频、全部原件及本次分析图帧已从D57发布并核验，见phase和delivery回执。')
  p.write_text(t)
 commit=compact([W/'phase.json',ROOT/'AGENTS.md',ROOT/'docs/TERRAIN_MEMORY_PLAN.md',
  ROOT/'docs/S1_W52_STABILITY_PLAN.md'],'Complete W52 negative result and verified direct original video delivery')
 print(json.dumps(dict(status=phase['status'],commit=commit,original_assets=len(receipt['verified_assets']),
  analysis_assets=len(evidence['verified_assets']),new_physics=0)),flush=True)
if __name__=='__main__':main()
