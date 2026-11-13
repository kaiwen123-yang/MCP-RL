#!/usr/bin/env python3
"""Exactly four old/corrected alpha0 pairs, native indexed path, no optimize."""
import json,os,sys,time,traceback
from pathlib import Path
import mujoco
import numpy as np
from mujoco_mpc.agent import Agent
from compare_s1_go2_w45_saved import initialize_agent
from w47_a_saved_arrays import ROOT,REL,SOURCE,OUT as SMALL,candidates,ident,terms,prefix,ma,sha,write
LARGE=ROOT/REL/'w47/task-a/pairs'
BUILD=ROOT/'work/w47-implementation'
FROZEN=ROOT/'work/r11-frozen/S5-r35'

def candidate_file(path,m,a):
 h=np.array([32,m['input_policy_id'],m['origin'],len(a['states']),37,12,36,a['residual'].shape[1],
             m['mu'],0,m['scale'],m['alpha'],m['scale'],m['representation'],96,m['J']],dtype='<f8')
 seq=[h]+[a[k].ravel() for k in ('reference_states','reference_actions','reference_times','reference_residual','K','H','Qu','Qxu','active')]
 np.concatenate(seq).astype('<f8').tofile(path)

def run_one(label,mode,policy,meta,nom,expected,build,record):
 dest=LARGE/label/mode;dest.mkdir(parents=True,exist_ok=False)
 f=json.loads((SOURCE/label/'input.json').read_text());state=dict(np.load(SOURCE/label/'input-state.npz'));state['time']=float(state['time'])
 model=mujoco.MjModel.from_xml_path(str(SOURCE/label/'model.xml'))
 binary=ROOT/build['variants'][mode]['binary'];reference=SOURCE/label/'constant/r29-current-plan.bin.r35-reference-1.bin'
 env=dict(os.environ,MJPC_TASKS_DIR=str(FROZEN/'runtime-tasks'),LD_LIBRARY_PATH=str(FROZEN/'lib'),
   MJPC_R33_INPUT_POLICY=str(policy),MJPC_R33_MECHANISM='N',MJPC_R34_ACCEPTANCE='strict',
   MJPC_R34_CONTEXT=str(SOURCE/label/'context.bin'),MJPC_R35_REFERENCE=str(reference),MJPC_R35_BRANCH='constant',
   MJPC_R34_SCORE_ONLY='1',MJPC_W47_INDEXED_SCORE='1',MJPC_R35_DIAGNOSTIC='1',
   MJPC_R29_SNAPSHOT=str(dest/'candidate.bin'),MJPC_R29_CALL_RECORD=str(dest/'planner-calls.jsonl'),
   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 begin=time.perf_counter();row=dict(label=label,mode=mode,status='starting_native_process',started_unix_s=time.time(),
   input_policy_sha256=sha(policy),context_sha256=sha(SOURCE/label/'context.bin'),reference_sha256=sha(reference),
   model_sha256=sha(SOURCE/label/'model.xml'),state_sha256=sha(SOURCE/label/'input-state.npz'),
   binary_sha256=sha(binary),origin_s=state['time'],deadline_unix_s=None,wall_deadline_enabled=False,
   source_candidate=meta,source_policy_identity=ident(meta,expected),path='ActionIndex/RolloutDiscrete',
   alpha_already_in_feedforward=True,new_optimizer_solves=0,new_PPO_updates=0)
 record['rows'].append(row);write(SMALL/'paired-evidence.json',record)
 with (dest/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,server_binary_path=binary,
     extra_flags=['--mjpc_workers=10'],subprocess_kwargs=dict(stdout=log,stderr=log,env=env),run_init=False) as agent:
  initialize_agent(agent,f,state)
  pid=agent.server_process.pid;row.update(pid=pid,status='running_fixed_rollout',actual_binary=str(Path(f'/proc/{pid}/exe').resolve()),
    actual_binary_sha256=sha(Path(f'/proc/{pid}/exe')),affinity=sorted(os.sched_getaffinity(pid)),
    loaded_mujoco=sorted(set(line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines() if 'libmujoco' in line)),
    task_parameters=agent.get_task_parameters(),cost_weights=agent.get_cost_weights(),mode_readback=agent.get_mode())
  write(SMALL/'paired-evidence.json',record)
  record['started_rollouts']+=1;row['integration_started_unix_s']=time.time();write(SMALL/'paired-evidence.json',record)
  rpc=time.perf_counter();agent.planner_step();row['native_rpc_wall_s']=time.perf_counter()-rpc
  cs=candidates(dest/'candidate.bin.r33-candidates')
  if len(cs)!=1:raise ValueError('Fixed score produced non-single candidate archive')
  sm,a=cs[0];sub=prefix(a['prefix'],a['residual'].shape[1])
  row.update(status='native_rollout_recorded',actual_rollouts=1,physics_substeps=len(sub),held_intervals=len(a['actions']),
    first_physical_clock_s=float(sub[0]['before']),last_physical_clock_s=float(sub[-1]['after']))
  record['actual_rollouts']=sum(r.get('actual_rollouts',0) for r in record['rows'])
  record['physics_substeps']=sum(r.get('physics_substeps',0) for r in record['rows'])
  write(SMALL/'paired-evidence.json',record)
  cost=terms(a['residual'],f['cost_definitions'])
  np.savez_compressed(dest/'trajectory.npz',**a)
  row.update(status='completed',wall_s=time.perf_counter()-begin,ended_unix_s=time.time(),
    actual_rollouts=1,physics_substeps=len(sub),held_intervals=len(a['actions']),physical_duration_s=float(a['times'][-1]-a['times'][0]),
    first_physical_clock_s=float(sub[0]['before']),last_physical_clock_s=float(sub[-1]['after']),
    recorded_state_nodes=len(a['states']),J=sm['J'],J_parts=cost,trunk_violation=sm['violation'],rollout_failure=sm['failed'],
    delta_J_vs_saved_nominal=sm['J']-meta['nominal_J'],
    error_vs_saved_nominal={k:ma(a[k]-nom[k]) for k in ('states','actions','times','residual')},
    error_vs_saved_alpha0={k:ma(a[k]-expected[k]) for k in ('states','actions','times','residual')},
    trajectory_dx_m=float(a['states'][-1,0]-a['states'][0,0]),
    candidate_sha256=sha(dest/'candidate.bin.r33-candidates'),trajectory_sha256=sha(dest/'trajectory.npz'))
  if len(sub)!=1750 or len(a['actions'])!=175:raise ValueError('Unexpected finite numerical work count')
 record['actual_rollouts']=sum(r.get('actual_rollouts',0) for r in record['rows'])
 record['physics_substeps']=sum(r.get('physics_substeps',0) for r in record['rows'])
 write(dest/'result.json',row);write(SMALL/'paired-evidence.json',record)
 print(json.dumps(dict(label=label,mode=mode,status=row['status'],physical_clock_s=row['last_physical_clock_s'],
    native_rpc_wall_s=row['native_rpc_wall_s'],error_vs_nominal=row['error_vs_saved_nominal'],delta_J=row['delta_J_vs_saved_nominal'])),flush=True)
 return row

def main():
 if LARGE.exists():raise ValueError('W47 paired attempt already exists; inspect recorded state, never repeat automatically')
 LARGE.mkdir(parents=True)
 build=json.loads((BUILD/'identity.json').read_text());start=time.time()
 record=dict(schema='w47_task_a_identity_pair_v1',status='running',started_unix_s=start,pid=os.getpid(),
   deadline_unix_s=None,wall_deadline_enabled=False,budget=dict(post_repair_fixed_rollouts=8,solves=0),
   started_rollouts=0,actual_rollouts=0,physics_substeps=0,plant_episodes=0,new_PPO_samples=0,new_PPO_updates=0,rows=[],pairs=[],
   source_execution='Stored first backward attempt alpha0; same input/context/reference. Each old/corrected path invokes actual ActionIndex and RolloutDiscrete once.',
   zero_alpha_admission_unchanged=True,r34_strict_unchanged=True,diagnostic_build='W47 exact-quaternion-identity diagnostic; not S5-r36 qualification')
 write(SMALL/'paired-evidence.json',record)
 try:
  for label in ('A148','A153','B155','B158'):
   p=SOURCE/label/'constant';cs=candidates(p/'r29-current-plan.bin.r33-candidates')
   nm,nom=min((x for x in cs if x[0]['kind']==0),key=lambda x:(x[0]['violation'],x[0]['J']))
   meta,expected=next(x for x in cs if x[0]['kind']==1 and x[0]['alpha']==0)
   meta=dict(meta,nominal_J=nm['J']);dest=LARGE/label;dest.mkdir();policy=dest/'saved-alpha0-policy.bin';candidate_file(policy,meta,expected)
   pair=[]
   for mode in ('old','corrected'):pair.append(run_one(label,mode,policy,meta,nom,expected,build,record))
   record['pairs'].append(dict(label=label,old_delta_J=pair[0]['delta_J_vs_saved_nominal'],corrected_delta_J=pair[1]['delta_J_vs_saved_nominal'],
      old_reproduces_archived_alpha0=all(v==0 for v in pair[0]['error_vs_saved_alpha0'].values()),
      corrected_reproduces_nominal=all(v==0 for v in pair[1]['error_vs_saved_nominal'].values()),
      old_dx_m=pair[0]['trajectory_dx_m'],corrected_dx_m=pair[1]['trajectory_dx_m']))
   write(SMALL/'paired-evidence.json',record)
  record['status']='completed_finite_identity_pairs'
  record['conclusion']='Exact quaternion identity repair evaluated only at zero endpoint. Positive directions were not reoptimized; no latest W46 A/B moving/recovery admission evidence is created.'
 except Exception as exc:
  record.update(status='technical_failure',error=repr(exc),traceback=traceback.format_exc(),
    unresolved_started_rollouts=record['started_rollouts']-record['actual_rollouts'],
    unresolved_physics_substeps=None if record['started_rollouts']>record['actual_rollouts'] else 0)
  raise
 finally:
  record.update(ended_unix_s=time.time(),wall_s=time.time()-start)
  files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(LARGE.rglob('*')) if p.is_file()]
  write(SMALL/'paired-originals.json',dict(server_originals_retained=True,files=files,actual_rollouts=record['actual_rollouts'],physics_substeps=record['physics_substeps']))
  write(SMALL/'paired-evidence.json',record)
if __name__=='__main__':main()
