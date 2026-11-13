#!/usr/bin/env python3
"""W49 native/RPC/held queries on actual W48 state nodes; no integration."""
import json,time,os,traceback,hashlib,importlib
import numpy as np,mujoco
from mujoco_mpc.agent import Agent
from compare_s1_go2_w45_saved import initialize_agent
from w49_common import *
def main():
 begin=time.time();out=LARGE/('task-a'+os.environ.get('W49_QUERY_ATTEMPT',''));out.mkdir(parents=True,exist_ok=False)
 r=dict(status='running',pid=os.getpid(),started_unix_s=begin,deadline=None,wall_deadline_enabled=False,
  affinity=sorted(os.sched_getaffinity(0)),native_optimizer_solves=0,transition_FD=0,physics_substeps=0,
  original_plant_substeps=0,new_PPO=0,events=[])
 write(SMALL/'task-a.json',r)
 try:
  pb=importlib.import_module('mujoco_mpc.proto.agent_pb2')
  r['protobuf_descriptor_sha256']=hashlib.sha256(pb.DESCRIPTOR.serialized_pb).hexdigest()
  r['binary_sha256']=sha(BUILD/'agent_server')
  for label in ['A213','B176','A363','A364','B328']:
   inp=LARGE/'inputs'/label;f=json.loads((inp/'input.json').read_text());d=Path(f['source'])
   dest=out/label;dest.mkdir();st=arrays(inp/'state.npz');st['time']=float(st['time'])
   traj=np.load(d/'trajectory.npz')
   links=[json.loads(x) for x in (d/'action-execution.jsonl').open()]
   width=.00001 if label=='A364' else .020
   links=[x for x in links if x['query_kind']=='applied_hold' and f['origin_s']-1e-9<=x['time_s']<f['origin_s']+width-1e-9]
   native={x['query_id']:x for x in map(json.loads,(d/'feedback-actions.jsonl').open()) if any(l['query_id']==x['query_id'] for l in links)}
   model=mujoco.MjModel.from_xml_path(f['model_path'])
   settings=env(dest,label,'installed.bin',MJPC_W49_QUERY_ONLY='1',MJPC_W49_EXPORT=str(dest/'imported.bin'))
   rows=[]
   with (dest/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
    server_binary_path=Path(os.environ.get('W49_QUERY_BINARY',str(BUILD/'agent_server'))),extra_flags=['--mjpc_workers=10'],
    subprocess_kwargs=dict(stdout=log,stderr=log,env=settings),run_init=False) as agent:
    initialize_agent(agent,f,st);agent.planner_step()
    assert sha(dest/'imported.bin')==sha(inp/'installed.bin'),'complete import/export changed policy'
    offset=0
    for link in links:
     t=link['time_s'];i=int(np.argmin(abs(traj['time']-t)));assert abs(traj['time'][i]-t)<1e-9
     q=traj['qpos'][i];v=traj['qvel'][i];query=[]
     for mode in ['explicit_float','implicit_state_double']:
      agent.set_state(time=t,qpos=q,qvel=v,act=np.zeros(0),mocap_pos=st['mocap_pos'],mocap_quat=st['mocap_quat'],userdata=st['userdata'])
      rpc=np.asarray(agent.get_action(time=t if mode=='explicit_float' else None,nominal_action=False,averaging_duration=0))
      with (dest/'feedback-actions.jsonl').open() as log:
       log.seek(offset);got=[json.loads(x) for x in log];offset=log.tell()
      raw=got[-1]
      query.append(dict(mode=mode,SetState_time_double=t,native_querytime=raw['time'],rpc=rpc.tolist(),native=raw))
     original=native[link['query_id']];e,im=query
     before=np.asarray(e['native']['action_before_clip']);after=np.asarray(im['native']['action_before_clip'])
     u=np.asarray(e['native']['u_star']);u2=np.asarray(im['native']['u_star'])
     torque=np.clip(60*(np.asarray(e['rpc'])-q[7:])-5*v[6:],model.actuator_forcerange[:,0],model.actuator_forcerange[:,1])
     torque2=np.clip(60*(np.asarray(im['rpc'])-q[7:])-5*v[6:],model.actuator_forcerange[:,0],model.actuator_forcerange[:,1])
     row=dict(time_s=t,query_id=link['query_id'],old_plan_id=link['plan_id'],queries=query,
      same_state_historical_native_max_error_rad=float(np.max(abs(u-np.asarray(original['u_star'])))),
      actual_held_vs_explicit_rpc_max_error_rad=float(np.max(abs(np.asarray(link['held_target'])-e['rpc']))),
      time_quantization_s=e['native']['time']-t,
      time_only_raw_delta_rad=float(np.max(abs(before-after))),time_only_projected_delta_rad=float(np.max(abs(u-u2))),
      time_only_RPC_delta_rad=float(np.max(abs(np.asarray(e['rpc'])-im['rpc']))),
      time_only_limited_PD_delta_Nm=float(np.max(abs(torque-torque2))),
      response_roundoff_rad=float(np.max(abs(u-e['rpc']))),
      unbounded_PD_equivalent_delta_Nm=60*float(np.max(abs(u-u2))),
      active_set_equal=e['native']['active_bounds']==im['native']['active_bounds'],
      native_reference_interpolation_nodes=e['native']['interpolation_nodes'])
     rows.append(row)
   write(dest/'queries.json',rows)
   metrics={k:max(x[k] if not k.endswith('_s') else abs(x[k]) for x in rows) for k in [
    'same_state_historical_native_max_error_rad','actual_held_vs_explicit_rpc_max_error_rad','time_quantization_s',
    'time_only_raw_delta_rad','time_only_projected_delta_rad','time_only_RPC_delta_rad','time_only_limited_PD_delta_Nm','response_roundoff_rad']}
   e=dict(label=label,origin_s=f['origin_s'],queries=len(rows)*2,state_nodes=len(rows),maxima=metrics,
    native_exact_historical_reproduction=metrics['same_state_historical_native_max_error_rad']==0,
    actual_held_matches_RPC=metrics['actual_held_vs_explicit_rpc_max_error_rad']==0,
    baseline_policy_sha256=sha(inp/'installed.bin'),import_export_exact=True,
    artifact=dict(path=str(dest/'queries.json'),bytes=(dest/'queries.json').stat().st_size,sha256=sha(dest/'queries.json')),
    physical_scope='zero-integration actual state queries; no claim of hidden-state model replay')
   r['events'].append(e);write(SMALL/'task-a.json',r);print(json.dumps(e),flush=True)
  r.update(status='completed_zero_integration',ended_unix_s=time.time(),
   diagnostic_rollouts=0,diagnostic_replay_status='not_run_pending_same_input_model_reconstruction; original external callback/integration state unavailable for historical A3',
   mechanism='Known time/RPC precision quantified at actual new events; native and held comparisons separate. Model/coldstart contribution remains unidentified by zero-integration queries.',
   query_count=sum(x['queries'] for x in r['events']))
 except Exception:
  r.update(status='technical_failure',traceback=traceback.format_exc());raise
 finally:
  r['wall_s']=time.time()-begin;write(SMALL/'task-a.json',r)
if __name__=='__main__':main()
