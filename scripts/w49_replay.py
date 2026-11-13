#!/usr/bin/env python3
"""A1 saved held and A2 real RPC on one explicit shared predictor coldstart; A3 unavailable."""
import json,os,time,traceback,xml.etree.ElementTree as ET
import numpy as np,mujoco
from mujoco_mpc.agent import Agent
from compare_s1_go2_w45_saved import initialize_agent
from w49_common import *
from w49_deployment import Bridge
def main():
 start=time.time();out=LARGE/'task-a-model';out.mkdir(parents=True,exist_ok=False)
 r=dict(status='running',pid=os.getpid(),started_unix_s=start,deadline=None,wall_deadline_enabled=False,
   physics_substeps=0,native_solves=0,FD=0,arms=[],new_PPO=0,A3='unavailable: historical external material and complete integration state were not recorded; no plant arm',
   scope='shared explicit coldstart native predictor, not exact historical MjData replay')
 write(SMALL/'task-a-model.json',r)
 try:
  bridge=Bridge()
  for label in ['A213','B176','A363','B328']:
   dest=out/label;dest.mkdir();inp=LARGE/'inputs'/label;f=json.loads((inp/'input.json').read_text());st=arrays(inp/'state.npz');st['time']=float(st['time'])
   q=json.loads((LARGE/'task-a-relinked'/label/'queries.json').read_text())
   model=mujoco.MjModel.from_xml_path(f['model_path'])
   tree=ET.parse(f['model_path']);tree.getroot().remove(tree.find('sensor'));xml=dest/'predictor.xml';tree.write(xml,encoding='unicode')
   settings=env(dest,label,'installed.bin',MJPC_W49_QUERY_ONLY='1')
   trajectories={};initials={}
   with (dest/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
      server_binary_path=BUILD/'agent_server',extra_flags=['--mjpc_workers=10'],subprocess_kwargs=dict(stdout=log,stderr=log,env=settings),run_init=False) as agent:
    initialize_agent(agent,f,st);agent.planner_step()
    for arm in ['A1','A2']:
     pointer,x=bridge.create(xml,st);initial=np.empty(bridge.lib.w49_integration_size(pointer));bridge.lib.w49_integration(pointer,initial)
     initials[arm]=initial;states=[x.copy()];actions=[];times=[float(st['time'])]
     for k in range(10):
      t=bridge.lib.w49_time(pointer)
      if arm=='A1':u=np.asarray(q[k]['queries'][0]['rpc'],dtype=np.float64)
      else:
       agent.set_state(time=t,qpos=x[:19],qvel=x[19:],act=np.zeros(0),mocap_pos=st['mocap_pos'],mocap_quat=st['mocap_quat'],userdata=st['userdata'])
       u=np.asarray(agent.get_action(time=t,nominal_action=False,averaging_duration=0),dtype=np.float64)
      rec=np.empty((10,13));bridge.lib.w46_hold(pointer,np.ascontiguousarray(u),x,rec);r['physics_substeps']+=10
      states.append(x.copy());actions.append(u.copy());times.append(bridge.lib.w49_time(pointer))
      write(SMALL/'task-a-model.json',r)
     bridge.lib.w46_destroy(pointer);trajectories[arm]=dict(states=np.asarray(states),actions=np.asarray(actions),times=np.asarray(times),integration=initial)
     np.savez_compressed(dest/(arm+'.npz'),**trajectories[arm])
   assert np.array_equal(initials['A1'],initials['A2'])
   old=np.load(Path(f['source'])/'trajectory.npz');idx=[int(np.argmin(abs(old['time']-t))) for t in trajectories['A1']['times']]
   actual=np.c_[old['qpos'][idx],old['qvel'][idx]]
   native_diff=trajectories['A1']['states']-actual;execution_diff=trajectories['A2']['states']-trajectories['A1']['states']
   def first(x):
    a=np.flatnonzero(np.any(x!=0,axis=1));return int(a[0]) if len(a) else None
   e=dict(label=label,physics_substeps=200,common_integration_bitwise_equal=True,
    first_A1_vs_historical_state_node=first(native_diff),first_A2_vs_A1_state_node=first(execution_diff),
    A1_vs_historical=dict(max_translation_m=float(abs(native_diff[:,:3]).max()),max_joint_rad=float(abs(native_diff[:,7:19]).max()),max_velocity=float(abs(native_diff[:,19:]).max())),
    A2_vs_A1=dict(max_translation_m=float(abs(execution_diff[:,:3]).max()),max_joint_rad=float(abs(execution_diff[:,7:19]).max()),max_velocity=float(abs(execution_diff[:,19:]).max()),
     max_held_rad=float(abs(trajectories['A2']['actions']-trajectories['A1']['actions']).max())),
    attribution='Same-input predictor versus historical plant is confounded by missing historical integration/external state; closed-loop feedback contribution measured on the shared reconstructed predictor.',
    source_model_sha256=sha(Path(f['model_path'])))
   r['arms'].append(e);write(SMALL/'task-a-model.json',r);print(json.dumps(e),flush=True)
  r.update(status='completed',ended_unix_s=time.time(),rollouts=8,classification='same-input model/coldstart difference followed by feedback divergence; historical model versus hidden-state effects not uniquely separated')
 except Exception:r.update(status='technical_failure',traceback=traceback.format_exc());raise
 finally:r['wall_s']=time.time()-start;write(SMALL/'task-a-model.json',r)
if __name__=='__main__':main()
