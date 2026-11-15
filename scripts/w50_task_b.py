#!/usr/bin/env python3
"""Exactly four W49 saved problems, one current-horizon homotopy solve each."""
import argparse,json,os,time,traceback
import numpy as np,mujoco
from mujoco_mpc.agent import Agent
from compare_s1_go2_w45_saved import initialize_agent
from w50_common import *
from w50_policy_io import read_policy
WORK_KEYS=['native_new_rollouts','transition_fd_calls','risk_geometry_fd_calls','derivative_passes','backward_attempts','candidate_records']

def motion(c,f):
 a=c['arrays'];x=a['states'];terms=cost_terms(a['residual'],f['cost_definitions'])
 row=dict(J=c['metadata']['native_J'],risk=c['metadata']['trunk_violation'],
  minimum_z_m=float(x[:,2].min()),final_z_m=float(x[-1,2]),
  forward_displacement_m=float(x[-1,0]-x[0,0]),
  vertical_velocity_range_m_s=[float(x[:,21].min()),float(x[:,21].max())],J_parts=terms)
 feet=a['feet'].reshape(-1,29)
 if len(feet):
  force=feet[:,1:].reshape(-1,4,7)
  row['predicted_support']=dict(foot_order=['FL','FR','RL','RR'],samples=len(feet),
   source='Trajectory::R33RecordFeet: FL FR RL RR; differs from residual Gait order FL HL FR HR',
   sample_clock='First native substep force/contact at each rollout node, labelled by its pre-integration state time; not a 5000Hz force series',
   time_range_s=[float(feet[0,0]),float(feet[-1,0])],
   per_foot_clearance_range_m=np.stack([force[:,:,3].min(axis=0),force[:,:,3].max(axis=0)]).tolist(),
   per_foot_total_Fz_quantiles_N=np.quantile(force[:,:,4]+force[:,:,5],[0,.1,.5,.9,1],axis=0).tolist(),
   per_foot_load_over_1N_fraction=((force[:,:,4]+force[:,:,5])>1).mean(axis=0).tolist(),
   diagnostic_load_threshold_N=1,
   nonfoot_support='not retained in candidate feet rows; cannot infer zero',
   native_Fz_per_foot_mean_N=force[:,:,4].mean(axis=0).tolist(),
   proxy_Fz_per_foot_mean_N=force[:,:,5].mean(axis=0).tolist(),
   total_Fz_quantiles_N=np.quantile((force[:,:,4]+force[:,:,5]).sum(axis=1),[0,.1,.5,.9,1]).tolist(),
   native_contact_transitions_per_foot=np.count_nonzero(np.diff(force[:,:,6]>0,axis=0),axis=0).tolist(),
   scope='native predictor contact and material proxy, not original physical task qualification')
 return row

def analyze(label):
 dest=LARGE/'task-b'/label;f=json.loads((LARGE/'inputs'/label/'input.json').read_text())
 call=[json.loads(x) for x in (dest/'planner-calls.jsonl').open()][-1]
 candidates=parse_candidates(dest/'snapshot.bin.r33-candidates')
 p=dest/'first-homotopy.json';guide=json.loads(p.read_text()) if p.exists() else None
 event=dict(label=label,stream=LABELS[label][0],origin_s=f['origin_s'],call=call,
  native_solves=1,work={k:call[k] for k in WORK_KEYS},same_epoch_r35_reference=True,
  historical_exact_replay=False,original_plant_substeps=0,additional_baseline_rollouts=0,
  first_valid_direction=(dest/'first-direction.bin').exists(),homotopy=guide,
  qualified_proposals=[],selected=None,selection_source=call['candidate_source'])
 if guide:
  _,bp=read_policy(dest/'first-baseline.bin')
  baseline=next(c for c in candidates if c['metadata']['kind']==0 and
   c['metadata']['feedback_scale']==bp['semantics'][5] and
   all(np.array_equal(c['arrays'][k],bp[k]) for k in ['reference_states','reference_actions','reference_times','K','H']))
  event['baseline']=motion(baseline,f);event['baseline_policy_path']=str(dest/'first-baseline.bin')
  event['baseline_policy_sha256']=sha(dest/'first-baseline.bin')
  event['raw_center_evaluations']=guide.get('raw_center_evaluations')
  positive=[(i,c) for i,c in enumerate(candidates) if c['metadata']['kind']==1]
  for j,((idx,c),g) in enumerate(zip(positive,guide['candidates'])):
   path=dest/f'complete-candidate-{idx+1}.bin';h,pol=read_policy(path)
   item=dict(g,slot=c['metadata']['slot'],policy_path=str(path),policy_sha256=sha(path),
    action_change_vs_baseline_rad=float(np.max(abs(c['arrays']['actions']-baseline['arrays']['actions']))),
    reference_start_s=float(pol['reference_times'][0]),reference_end_s=float(pol['reference_times'][-1]),
    feedback_scale=float(pol['semantics'][5]),delta_knots=len(pol['delta_times']),
    provenance=pol['provenance'].tolist(),expected=None,ratio=None,
    motion=motion(c,f) if not c['metadata']['failure'] else None)
   if g['strict_better_than_B'] and not g['failure']:
    event['qualified_proposals'].append(item)
   if call['candidate_source']==f'homotopy_{c["metadata"]["slot"]}':event['selected']=item
  event['alpha_zero']=dict(policy='complete baseline retained exactly',scoring='reuse original B score',extra_rollouts=0,installed=False)
  event['same_backward']=dict(path=str(dest/'first-backward.bin'),sha256=sha(dest/'first-backward.bin'))
 write(dest/'result.json',event)
 return event

def run(stream):
 begin=time.time();path=SMALL/f'task-b-{stream}.json'
 r=dict(status='running',pid=os.getpid(),started_unix_s=begin,deadline=None,wall_deadline_enabled=False,
  affinity=sorted(os.sched_getaffinity(0)),events=[],new_PPO_samples=0,new_PPO_updates=0,
  binary_sha256=sha(BUILD/'agent_server'),original_plant_substeps=0)
 write(path,r)
 try:
  for label in [x for x in LABELS if LABELS[x][0]==stream]:
   inp=LARGE/'inputs'/label;f=json.loads((inp/'input.json').read_text())
   dest=LARGE/'task-b'/label;dest.mkdir(parents=True,exist_ok=False)
   st=arrays(inp/'state.npz');st['time']=float(st['time'])
   model=mujoco.MjModel.from_xml_path(f['model_path'])
   settings=env(dest,label,MJPC_W49_DIRECTION=str(dest/'first'),MJPC_W49_COMPLETE_POLICIES=str(dest/'complete'),MJPC_W49_EXPORT=str(dest/'installed.bin'))
   write(dest/'input-identity.json',f)
   r.update(active_event=label,active_origin_s=f['origin_s'],active_started_unix_s=time.time());write(path,r)
   with (dest/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
    server_binary_path=BUILD/'agent_server',extra_flags=['--mjpc_workers=10'],
    subprocess_kwargs=dict(stdout=log,stderr=log,env=settings),run_init=False) as agent:
    initialize_agent(agent,f,st);r['native_pid']=agent.server_process.pid;write(path,r)
    agent.planner_step()
   event=analyze(label);r['events'].append(event);write(path,r)
   print(json.dumps(dict(label=label,work=event['work'],selected=event['selection_source'],qualified=len(event['qualified_proposals']))),flush=True)
  r.update(status='completed',ended_unix_s=time.time())
 except Exception:
  r.update(status='technical_failure',traceback=traceback.format_exc());raise
 finally:
  r['wall_s']=time.time()-begin;write(path,r)

def aggregate():
 records=[json.loads((SMALL/f'task-b-{s}.json').read_text()) for s in ['a','b']]
 events=[e for r in records for e in r['events']]
 result=dict(status='completed' if all(r['status']=='completed' for r in records) else 'incomplete',
  events=events,streams=records,deadline=None,wall_deadline_enabled=False,new_PPO_samples=0,new_PPO_updates=0,
  native_solves=len(events),work={k:sum(e['work'][k] for e in events) for k in WORK_KEYS},
  raw_center_evaluations=sum(e.get('raw_center_evaluations') or 0 for e in events),
  original_plant_substeps=0,qualification='finite diagnostics only; original stages not run')
 write(SMALL/'task-b.json',result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stream',choices=['a','b']);p.add_argument('--analyze',choices=list(LABELS));p.add_argument('--aggregate',action='store_true');a=p.parse_args()
 if a.aggregate:print(json.dumps(aggregate(),indent=2))
 elif a.analyze:print(json.dumps(analyze(a.analyze),indent=2))
 else:run(a.stream)
