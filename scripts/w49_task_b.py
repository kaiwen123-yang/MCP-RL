#!/usr/bin/env python3
"""Four new N problems, first-valid-k guided frozen-feedback candidates only."""
import json,time,os,traceback
import numpy as np,mujoco
from mujoco_mpc.agent import Agent
from compare_s1_go2_w45_saved import initialize_agent
from compare_s1_go2_w44_saved import cost_terms
from w49_common import *
def main():
 begin=time.time();out=LARGE/'task-b';out.mkdir(parents=True,exist_ok=False)
 r=dict(status='running',pid=os.getpid(),started_unix_s=begin,deadline=None,wall_deadline_enabled=False,
  affinity=sorted(os.sched_getaffinity(0)),events=[],new_PPO=0,original_plant_substeps=0,
  direction_rule='first valid backward before alpha modifies actions; +alpha*k, one direction per new problem',
  candidate_expected=None,candidate_ratio=None,binary_sha256=sha(BUILD/'agent_server'))
 write(SMALL/'task-b.json',r)
 try:
  for label in ['A213','B176','A364','B328']:
   inp=LARGE/'inputs'/label;f=json.loads((inp/'input.json').read_text());dest=out/label;dest.mkdir()
   st=arrays(inp/'state.npz');st['time']=float(st['time'])
   a=arrays(Path(f['source_objects']['accepted']['path']));x=a['accepted_states'];t=a['accepted_times']
   old=dict(source='saved actual installed accepted rollout',horizon_s=float(t[-1]-t[0]),initial_xyz=x[0,:3].tolist(),
    final_xyz=x[-1,:3].tolist(),minimum_z_m=float(x[:,2].min()),forward_displacement_m=float(x[-1,0]-x[0,0]),
    max_adjacent_action_delta_rad=float(np.max(abs(np.diff(a['accepted_actions'],axis=0)))),
    J=f['boundary']['current_solve'],predicted_contact='not retained in these W48 accepted objects; do not infer support from gait',
    actual_comparison_limit_s=.02,actual_after_replanning_is_same_policy=False)
   model=mujoco.MjModel.from_xml_path(f['model_path'])
   settings=env(dest,label,MJPC_W49_FINITE_GUIDED='1',MJPC_W49_DIRECTION=str(dest/'first'),
    MJPC_W49_COMPLETE_POLICIES=str(dest/'complete'))
   write(dest/'input-identity.json',f)
   r['active_event']=label;r['active_origin_s']=f['origin_s'];r['active_started_unix_s']=time.time();write(SMALL/'task-b.json',r)
   with (dest/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
    server_binary_path=BUILD/'agent_server',extra_flags=['--mjpc_workers=10'],
    subprocess_kwargs=dict(stdout=log,stderr=log,env=settings),run_init=False) as agent:
    initialize_agent(agent,f,st)
    r['native_pid']=agent.server_process.pid
    r['actual_exe_sha256']=sha(Path(f'/proc/{agent.server_process.pid}/exe'))
    write(SMALL/'task-b.json',r)
    agent.planner_step()
   calls=[json.loads(x) for x in (dest/'planner-calls.jsonl').open()]
   call=calls[-1];candidates=parse_candidates(dest/'snapshot.bin.r33-candidates')
   guided_path=dest/'first-guided.json';guided=json.loads(guided_path.read_text()) if guided_path.exists() else None
   event=dict(label=label,origin_s=f['origin_s'],saved_plan=old,call=call,guided=guided,
    native_solves=1,additional_guided_rollouts=guided['extra_candidates'] if guided else 0,
    native_N_rollouts=call['native_new_rollouts']-(guided['extra_candidates'] if guided else 0),
    work={k:call[k] for k in ['native_new_rollouts','transition_fd_calls','risk_geometry_fd_calls','derivative_passes','backward_attempts','candidate_records']},
    original_plant_substeps=0,additional_baseline_rollouts=0,first_valid_direction=(dest/'first-direction.bin').exists(),
    historical_exact_solve=False,qualified_proposals=[])
   if guided:
    bh=sha(dest/'first-baseline.bin')
    baseindex=next(i for i,c in enumerate(candidates) if c['metadata']['kind']==0 and sha(dest/f'complete-candidate-{i+1}.bin')==bh)
    b=candidates[baseindex];_,bp=read_policy(dest/'first-baseline.bin')
    for j,g in enumerate(guided['candidates']):
     idx=len(candidates)-len(guided['candidates'])+j;c=candidates[idx];path=dest/f'complete-candidate-{idx+1}.bin';h,p=read_policy(path)
     frozen=['reference_states','reference_actions','reference_times','K','H']
     unchanged=all(np.array_equal(p[k],bp[k]) for k in frozen) and p['semantics'][5]==bp['semantics'][5]
     assert unchanged and bool(p['semantics'][0])
     change=float(np.max(abs(c['arrays']['actions']-b['arrays']['actions'])))
     g.update(policy_path=str(path),policy_sha256=sha(path),action_change_vs_baseline_rad=change,
      frozen_feedback_equal=True,delta_knots=len(p['delta_times']),expected=None,ratio=None,
      J_parts=cost_terms(c['arrays']['residual'],f['cost_definitions']))
     if g['strict_better_than_B'] and change>0:
      event['qualified_proposals'].append(dict(label=label,**g))
    event['baseline_policy_sha256']=bh
    event['baseline_native_J']=b['metadata']['native_J']
   write(dest/'result.json',event);r['events'].append(event);write(SMALL/'task-b.json',r)
   print(json.dumps(dict(label=label,work=event['work'],qualified=len(event['qualified_proposals']))),flush=True)
  candidates=[c for e in r['events'] for c in e['qualified_proposals']]
  r.update(status='completed',ended_unix_s=time.time(),finite_candidate_exists=bool(candidates),
   native_solves=len(r['events']),additional_guided_rollouts=sum(e['additional_guided_rollouts'] for e in r['events']),
   work={k:sum(e['work'][k] for e in r['events']) for k in r['events'][0]['work']},
   proposed_formal_mechanism='K' if candidates else None,
   deployment_status='pending_bounded_real_interface_check' if candidates else 'not_run_no_strict_behavior_changing_candidate')
 except Exception:
  r.update(status='technical_failure',traceback=traceback.format_exc());raise
 finally:
  r['wall_s']=time.time()-begin;write(SMALL/'task-b.json',r)
if __name__=='__main__':main()
