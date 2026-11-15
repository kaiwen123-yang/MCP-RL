#!/usr/bin/env python3
"""At most two real-RPC 40 ms native predictor deployments, one replan each."""
import ctypes,json,os,time,traceback,xml.etree.ElementTree as ET
import numpy as np,mujoco
from mujoco_mpc.agent import Agent
from compare_s1_go2_w45_saved import initialize_agent
from w50_common import *
from w50_policy_io import read_policy
from w50_task_b import WORK_KEYS

class Bridge:
 def __init__(self):
  # Python and frozen native libraries share SONAME 3.2.3 but differ in ABI.
  # A new ELF namespace prevents the already-loaded Python wheel from winning.
  loader=ctypes.CDLL('libdl.so.2')
  loader.dlmopen.argtypes=[ctypes.c_long,ctypes.c_char_p,ctypes.c_int];loader.dlmopen.restype=ctypes.c_void_p
  library=ROOT/'work/w50-implementation/native-bridge.so'
  handle=loader.dlmopen(-1,str(library).encode(),os.RTLD_NOW|os.RTLD_LOCAL)
  if not handle:raise RuntimeError('Cannot load frozen native bridge in its own ELF namespace')
  self.lib=ctypes.CDLL(str(library),handle=handle)

  P=ctypes.c_void_p;D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
  for name,args in [('w46_create',[ctypes.c_char_p,ctypes.c_char_p]),('w46_reset',[P,D,D,ctypes.c_double]),('w46_hold',[P,D,D,D]),
   ('w49_mocap',[P,D,D]),('w49_integration',[P,D]),('w49_integration_size',[P]),('w49_time',[P]),('w46_destroy',[P]),
   ('w50_userdata',[P,D]),('w50_support',[P,D])]:getattr(self.lib,name).argtypes=args
  self.lib.w46_create.restype=P;self.lib.w49_time.restype=ctypes.c_double;self.lib.w49_integration_size.restype=ctypes.c_int
 def create(self,model,state):
  error=ctypes.create_string_buffer(2048);pointer=self.lib.w46_create(str(model).encode(),error)
  if not pointer:raise RuntimeError(error.value)
  x=np.r_[state['qpos'],state['qvel']]
  self.lib.w46_reset(pointer,x,np.ascontiguousarray(state['userdata']),float(state['time']))
  self.lib.w49_mocap(pointer,np.ascontiguousarray(state['mocap_pos'].ravel()),np.ascontiguousarray(state['mocap_quat'].ravel()))
  return pointer,x

def main():
 start=time.time();out=LARGE/'deployment';out.mkdir(parents=True,exist_ok=False)
 evidence=json.loads((SMALL/'task-b.json').read_text());chosen=[]
 for stream in ['a','b']:
  event=next((e for e in evidence['events'] if e['stream']==stream and e['selected'] is not None),None)
  if event:chosen.append(event)
 r=dict(status='running',pid=os.getpid(),started_unix_s=start,deadline=None,wall_deadline_enabled=False,
  checks=[],physics_substeps=0,native_solves=0,work={k:0 for k in WORK_KEYS},original_plant_substeps=0,new_PPO_samples=0,new_PPO_updates=0,
  ready_for_original=False,readiness='Requires substantive early-problem evidence and completed interface/source review.',
  binary_sha256=sha(BUILD/'agent_server'),bridge_sha256=sha(ROOT/'work/w50-implementation/native-bridge.so'),
  scope='Native predictor with original callbacks; actual RPC/query/hold and one new-epoch solve. Not exact historical plant replay.',
  context_scope='40ms held mission command/cap and frozen material proxy, causal native gait. At 20ms normal r26 anchor refresh and fresh r35 freeze.')
 write(SMALL/'deployment.json',r)
 try:
  bridge=Bridge()
  for event in chosen:
   selected=event['selected'];label=event['label'];inp=LARGE/'inputs'/label;dest=out/label;dest.mkdir()
   f=json.loads((inp/'input.json').read_text());st=arrays(inp/'state.npz');st['time']=float(st['time'])
   model=mujoco.MjModel.from_xml_path(f['model_path'])
   tree=ET.parse(f['model_path']);tree.getroot().remove(tree.find('sensor'));xml=dest/'predictor-physics.xml';tree.write(xml,encoding='unicode')
   settings=env(dest,label,MJPC_R33_INPUT_POLICY=selected['policy_path'],MJPC_W50_INSTALL_ONCE='1',
    MJPC_W49_EXPORT=str(dest/'imported.bin'),MJPC_W49_DIRECTION=str(dest/'replan'),MJPC_W49_COMPLETE_POLICIES=str(dest/'complete'))
   held=[];states=[];native=[];records=[];queries=[];times=[];supports=[]
   r.update(active_event=label,active_origin_s=f['origin_s']);write(SMALL/'deployment.json',r)
   with (dest/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
    server_binary_path=BUILD/'agent_server',extra_flags=['--mjpc_workers=10'],
    subprocess_kwargs=dict(stdout=log,stderr=log,env=settings),run_init=False) as agent:
    initialize_agent(agent,f,st);agent.planner_step()
    # Serialize original sensor model in Agent.Init before sensor-free bridge XML.
    pointer,x=bridge.create(xml,st);states.append(x.copy())
    integration=np.empty(bridge.lib.w49_integration_size(pointer));bridge.lib.w49_integration(pointer,integration)
    import_exact=sha(dest/'imported.bin')==sha(selected['policy_path'])
    if not import_exact:raise RuntimeError('Winner complete policy did not survive actual initial installation exactly')
    initial_policy=(dest/'imported.bin').read_bytes();(dest/'initial-installed.bin').write_bytes(initial_policy)
    offset=0;midpoint=None
    for k in range(20):
     t=bridge.lib.w49_time(pointer);times.append(t)
     if k==10:
      packet=st['userdata'][84:96];packet[1]=t;packet[2]=x[0]
      packet[4]=np.sign(packet[3]-x[0]);packet[5]=np.clip(packet[4]*st['userdata'][78],0.,.25)
      bridge.lib.w50_userdata(pointer,np.ascontiguousarray(st['userdata']))
     agent.set_state(time=t,qpos=x[:19],qvel=x[19:],act=np.zeros(0),mocap_pos=st['mocap_pos'],mocap_quat=st['mocap_quat'],userdata=st['userdata'])
     if k==10:
      agent.planner_step();r['native_solves']+=1
      call=[json.loads(line) for line in (dest/'planner-calls.jsonl').open()][-1]
      h,pol=read_policy(dest/'imported.bin');(dest/'midpoint-installed.bin').write_bytes((dest/'imported.bin').read_bytes())
      midpoint=dict(time_s=t,call=call,reference_start_s=float(pol['reference_times'][0]),reference_end_s=float(pol['reference_times'][-1]),
       provenance=pol['provenance'].tolist(),r26_packet=st['userdata'][84:96].tolist(),full_policy_path=str(dest/'midpoint-installed.bin'))
      if abs(call['origin']-t)>1e-9:raise RuntimeError('Midpoint planner did not use actual new epoch time')
      for key in WORK_KEYS:r['work'][key]+=call[key]
     u=np.asarray(agent.get_action(time=t,nominal_action=False,averaging_duration=0),dtype=np.float64)
     with (dest/'feedback-actions.jsonl').open() as log:
      log.seek(offset);row=[json.loads(line) for line in log][-1];offset=log.tell()
     if not row['metric_valid'] or not row['qp_converged']:raise RuntimeError('Actual RPC projection invalid')
     if not np.array_equal(u,np.asarray(row['u_star']).astype(np.float32).astype(np.float64)):raise RuntimeError('Actual RPC differs from native action cast')
     held.append(u.copy());native.append(row['u_star']);queries.append(row)
     rec=np.empty((10,13));bridge.lib.w46_hold(pointer,np.ascontiguousarray(u),x,rec)
     r['physics_substeps']+=10;records.append(rec);states.append(x.copy())
     support=np.empty(14);bridge.lib.w50_support(pointer,support);supports.append(support)
     write(SMALL/'deployment.json',r)
   bridge.lib.w46_destroy(pointer)
   np.savez_compressed(dest/'real-interface-40ms.npz',states=states,held=held,native_double=native,substeps=records,
    query_times=times,initial_integration=integration,support_after_hold=supports)
   write(dest/'queries.json',queries)
   check=dict(label=label,selected=selected,physics_substeps=200,logical_s=.040,native_solves=1,midpoint=midpoint,
    initial_import_export_byte_exact=import_exact,metric_and_QP_valid=True,RPC_equals_single_FP32_native_cast=True,
    response_roundoff_rad=float(np.max(abs(np.asarray(native)-held))),initial_xyz=states[0][:3].tolist(),final_xyz=states[-1][:3].tolist(),
    minimum_z_m=float(np.asarray(states)[:,2].min()),max_recorded_trunk_violation_m=float(np.max(np.asarray(records)[:,:,12])),
    support=dict(foot_order=['FL','FR','RL','RR'],native_Fz_mean_N=np.asarray(supports)[:,1:5].mean(axis=0).tolist(),
     proxy_Fz_mean_N=np.asarray(supports)[:,5:9].mean(axis=0).tolist(),nonfoot_Fz_max_N=float(np.asarray(supports)[:,9].max()),
     sample_scope='Last solver forces after each 2ms held interval; native predictor, not historical plant. No extra forward or integration calls.'),
    artifact=dict(path=str(dest/'real-interface-40ms.npz'),sha256=sha(dest/'real-interface-40ms.npz'),bytes=(dest/'real-interface-40ms.npz').stat().st_size))
   r['checks'].append(check);write(SMALL/'deployment.json',r)
   print(json.dumps(dict(label=label,physics_substeps=200,midpoint_source=midpoint['call']['candidate_source'],final_xyz=check['final_xyz'])),flush=True)
  r.update(status='completed',ended_unix_s=time.time(),interface_complete=bool(r['checks']))
 except Exception:r.update(status='technical_failure',traceback=traceback.format_exc());raise
 finally:r['wall_s']=time.time()-start;write(SMALL/'deployment.json',r)

if __name__=='__main__':main()
