#!/usr/bin/env python3
"""Restore only the one authorized R8 scientific input on D57, then inspect it."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np
from w48_build import ROOT,sha
def main():
 out=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/task-c';out.mkdir(parents=True,exist_ok=True)
 name='project-s3-formal-v1-single-yaw_plus5-trajectory.npz';digest='079f57e1ebad260c9cdac842dc089012e29cb46992ab64d2c31cd9d380b61625'
 result=dict(status='checking_existing_source',asset_id=681937433,expected_bytes=28100505,expected_sha256=digest,
  deadline_unix_s=None,wall_deadline_enabled=False,new_physics_steps=0,donor_solves=0,downloaded_bytes=0,network_route='authorized D57 academic',started_unix_s=time.time())
 try:
  matches=subprocess.check_output(['find','/root/autodl-tmp','-type','f','-name',name],text=True).splitlines()
  if matches:p=Path(matches[0])
  else:
   dest=ROOT/'work/w48-inputs/r8';dest.mkdir(parents=True,exist_ok=True);p=dest/name
   result['command']=['gh','release','download','tm-s1-go2-wb-support-r8','--repo','kaiwen123-yang/MCP-RL','--pattern',name,'--dir',str(dest)]
   process=subprocess.run(result['command'],capture_output=True,text=True,timeout=120)
   if process.returncode:raise RuntimeError('Authorized single-source download failed: '+process.stderr[:600])
   result['downloaded_bytes']=p.stat().st_size
  result.update(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
  assert result['bytes']==28100505 and result['sha256']==digest
  p.chmod(0o444)
  z=np.load(p);schema={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
  result.update(status='restored_verified_schema_inspected',schema=schema)
  for k in ['time','time_s','phase','plan_origin','q_target']:
   if k in z:
    a=z[k]
    if k=='phase':result['phase_values']=np.unique(a).tolist()
    if k in ('time','time_s'):result['clock']=dict(key=k,first=float(a[0]),last=float(a[-1]),dt_min=float(np.diff(a).min()),dt_max=float(np.diff(a).max()),nodes=len(a))
  # R8 phase means mission stage; it is not the private gait accumulator.
  result['gait_phase_reconstruction']='Not established from trajectory phase labels alone; requires source cadence and GetPhase/private accumulator convention. No donor solve performed.'
 except Exception as e:result.update(status='source_unavailable_or_inspection_error',error=repr(e))
 result['ended_unix_s']=time.time()
 (out/'r8-source.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='schema'}),flush=True)
if __name__=='__main__':main()
