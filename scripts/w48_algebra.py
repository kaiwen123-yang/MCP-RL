#!/usr/bin/env python3
"""One non-integrating algebra inspection using the formal r36 compiled StateDiff."""
import json,shlex,subprocess,os
from pathlib import Path
import numpy as np
from w48_build import ROOT,OUT,CACHE,UP,DEST,sha
def main():
 entries=json.loads((CACHE/'compile_commands.json').read_text())
 e=next(c for c in entries if c['file']==str(UP/'mjpc/utilities.cc'));cmd=shlex.split(e['command'])
 obj=OUT/'algebra.o';cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(ROOT/'scripts/w48_state_diff_algebra.cc');cmd.insert(1,'-I'+str(DEST))
 subprocess.run(cmd,cwd=e['directory'],check=True)
 link=json.loads((DEST/'identity.json').read_text())['link_command']
 link=[str(obj) if x=='CMakeFiles/agent_server.dir/agent_server.cc.o' else x for x in link]
 exe=OUT/'state-diff-algebra';link[link.index('-o')+1]=str(exe)
 subprocess.run(link,cwd=CACHE/'mjpc/grpc',check=True)
 source=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46-mechanisms'
 rows=[]
 for label in ['A148','A153','B155','B158']:
  s=np.load(source/label/'input-state.npz');state=np.r_[s['qpos'],s['qvel'],s['act']]
  p=OUT/(label+'-state.txt');np.savetxt(p,state)
  output=subprocess.check_output([str(exe),str(source/label/'model.xml'),str(p)],text=True,
   env=dict(os.environ,LD_LIBRARY_PATH=str(DEST/'lib')))
  results=[json.loads(x) for x in output.splitlines()]
  assert not np.any(results[0]['ds']) and not np.any(results[1]['ds'])
  assert results[2]['ds'][3]>0 and results[3]['ds'][3]<0
  assert abs(results[4]['ds'][0]-.001)<1e-14 and abs(results[4]['ds'][18]-.002)<1e-14
  rows.append(dict(label=label,results=results,rotation_tangent_rad=1e-8))
 # Actual Go2 has na=0; a non-integrating free+ball+activation model checks act preservation.
 xml=OUT/'algebra-act.xml';xml.write_text('<mujoco><worldbody><body><freejoint/><geom size=".1" mass="1"/><body pos="0 0 .3"><joint name="ball" type="ball"/><geom size=".1" mass="1"/></body></body></worldbody><actuator><general joint="ball" dyntype="integrator" gear="1 0 0"/></actuator></mujoco>')
 import mujoco
 m=mujoco.MjModel.from_xml_path(str(xml));state=np.r_[m.qpos0,np.zeros(m.nv),.2]
 p=OUT/'act-state.txt';np.savetxt(p,state)
 output=subprocess.check_output([str(exe),str(xml),str(p)],text=True,env=dict(os.environ,LD_LIBRARY_PATH=str(DEST/'lib')))
 act=[json.loads(x) for x in output.splitlines()];assert abs(act[4]['ds'][-1]-.003)<1e-14
 report=dict(status='completed',rows=rows,activation_model=act,native_library_sha256=sha(DEST/'lib/libmujoco.so.3.2.3'),
  utilities_sha256=sha(DEST/'mjpc/utilities.cc'),compiled_archive_sha256=sha(OUT/'libmjpc.a'),
  integration_steps=0,optimizer_solves=0,act_note='Go2 na=0; separate free/ball integrator-act model used without mj_step')
 p=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/task-a/algebra.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n')
 print('StateDiff algebra completed; zero integration',flush=True)
if __name__=='__main__':main()
