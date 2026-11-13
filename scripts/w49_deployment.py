#!/usr/bin/env python3
"""Four bounded real-RPC held deployments on the common native predictor cold-start."""
import ctypes,json,os,time,traceback,xml.etree.ElementTree as ET
import numpy as np,mujoco
from mujoco_mpc.agent import Agent
from compare_s1_go2_w45_saved import initialize_agent
from w49_common import *
class Bridge:
 def __init__(self):
  self.lib=ctypes.CDLL(str(ROOT/'work/w49-implementation/native-bridge.so'),mode=os.RTLD_LOCAL|os.RTLD_DEEPBIND)
  P=ctypes.c_void_p;D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
  for name,args in [('w46_create',[ctypes.c_char_p,ctypes.c_char_p]),('w46_reset',[P,D,D,ctypes.c_double]),('w46_hold',[P,D,D,D]),
   ('w49_mocap',[P,D,D]),('w49_integration',[P,D]),('w49_integration_size',[P]),('w49_time',[P]),('w46_destroy',[P])]:
   getattr(self.lib,name).argtypes=args
  self.lib.w46_create.restype=P;self.lib.w49_time.restype=ctypes.c_double;self.lib.w49_integration_size.restype=ctypes.c_int
 def create(self,model,state):
  error=ctypes.create_string_buffer(2048);p=self.lib.w46_create(str(model).encode(),error)
  if not p:raise RuntimeError(error.value)
  x=np.r_[state['qpos'],state['qvel']]
  self.lib.w46_reset(p,x,np.ascontiguousarray(state['userdata']),float(state['time']))
  self.lib.w49_mocap(p,np.ascontiguousarray(state['mocap_pos'].ravel()),np.ascontiguousarray(state['mocap_quat'].ravel()))
  return p,x
def main():
 begin=time.time();out=LARGE/'deployment2';out.mkdir(parents=True,exist_ok=False)
 evidence=json.loads((SMALL/'task-b.json').read_text())
 r=dict(status='running',pid=os.getpid(),started_unix_s=begin,deadline=None,wall_deadline_enabled=False,
  checks=[],physics_substeps=0,native_solves=0,FD=0,original_plant_substeps=0,new_PPO=0,
  binary_sha256=sha(BUILD/'agent_server'),bridge_sha256=sha(ROOT/'work/w49-implementation/native-bridge.so'),
  scope='Real GetAction RPC output advances native3.2.3 predictor with original callbacks, same explicit coldstart; not historical plant replay')
 write(SMALL/'deployment.json',r)
 try:
  bridge=Bridge()
  for event in evidence['events']:
   if not event['qualified_proposals']:continue
   chosen=min(event['qualified_proposals'],key=lambda x:(x['risk'],x['J']))
   label=event['label'];dest=out/label;dest.mkdir();inp=LARGE/'inputs'/label
   f=json.loads((inp/'input.json').read_text());st=arrays(inp/'state.npz');st['time']=float(st['time'])
   model=mujoco.MjModel.from_xml_path(f['model_path'])
   tree=ET.parse(f['model_path']);tree.getroot().remove(tree.find('sensor'));xml=dest/'predictor-physics.xml';tree.write(xml,encoding='unicode')
   settings=env(dest,label,MJPC_R33_INPUT_POLICY=chosen['policy_path'],MJPC_W49_QUERY_ONLY='1',MJPC_W49_EXPORT=str(dest/'imported.bin'))
   held=[];states=[];native=[];records=[];queries=[];times=[]
   with (dest/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
     server_binary_path=BUILD/'agent_server',extra_flags=['--mjpc_workers=10'],
     subprocess_kwargs=dict(stdout=log,stderr=log,env=settings),run_init=False) as agent:
    initialize_agent(agent,f,st);agent.planner_step()
    # Init serializes the original sensor-bearing model before loading the separate physics XML.
    pointer,x=bridge.create(xml,st);states.append(x.copy())
    integration=np.empty(bridge.lib.w49_integration_size(pointer));bridge.lib.w49_integration(pointer,integration)
    assert sha(dest/'imported.bin')==sha(chosen['policy_path']),'complete delta/metric import-export changed bytes'
    offset=0
    for k in range(10):
     t=bridge.lib.w49_time(pointer);times.append(t)
     agent.set_state(time=t,qpos=x[:19],qvel=x[19:],act=np.zeros(0),mocap_pos=st['mocap_pos'],mocap_quat=st['mocap_quat'],userdata=st['userdata'])
     u=np.asarray(agent.get_action(time=t,nominal_action=False,averaging_duration=0),dtype=np.float64)
     with (dest/'feedback-actions.jsonl').open() as log:
      log.seek(offset);row=[json.loads(line) for line in log][-1];offset=log.tell()
     assert row['metric_valid'] and row['qp_converged']
     assert np.array_equal(u,np.asarray(row['u_star']).astype(np.float32).astype(np.float64))
     held.append(u.copy());native.append(row['u_star']);queries.append(row)
     rec=np.empty((10,13));bridge.lib.w46_hold(pointer,np.ascontiguousarray(u),x,rec)
     r['physics_substeps']+=10;records.append(rec);states.append(x.copy())
     write(SMALL/'deployment.json',r)
   bridge.lib.w46_destroy(pointer)
   np.savez_compressed(dest/'real-interface-20ms.npz',states=states,held=held,native_double=native,substeps=records,
     query_times=times,initial_integration=integration)
   write(dest/'queries.json',queries)
   check=dict(label=label,selected=chosen,physics_substeps=100,logical_s=.020,
     import_export_byte_exact=True,metric_and_QP_valid=True,RPC_equals_single_FP32_native_cast=True,
     response_roundoff_rad=float(np.max(abs(np.asarray(native)-held))),
     final_xyz=states[-1][:3].tolist(),initial_xyz=states[0][:3].tolist(),
     max_recorded_trunk_violation_m=float(np.max(np.asarray(records)[:,:,12])),
     artifact=dict(path=str(dest/'real-interface-20ms.npz'),sha256=sha(dest/'real-interface-20ms.npz'),
     bytes=(dest/'real-interface-20ms.npz').stat().st_size))
   r['checks'].append(check);write(SMALL/'deployment.json',r);print(json.dumps(check),flush=True)
  r.update(status='completed',ended_unix_s=time.time(),formal_mechanism='K',
    ready_for_original=bool(r['checks']) and all(c['metric_and_QP_valid'] and c['import_export_byte_exact'] for c in r['checks']))
 except Exception:
  r.update(status='technical_failure',traceback=traceback.format_exc());raise
 finally:
  r['wall_s']=time.time()-begin;write(SMALL/'deployment.json',r)
if __name__=='__main__':main()
