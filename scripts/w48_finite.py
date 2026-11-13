#!/usr/bin/env python3
"""W48 four paired fixed-input N solves and four corrected ActionImpl deployment checks."""
import argparse,json,os,time,traceback
from pathlib import Path
import mujoco,numpy as np
from mujoco_mpc.agent import Agent
from compare_s1_go2_w44_saved import parse_candidates,take_policy_arrays,policy_hash,cost_terms
from compare_s1_go2_w45_saved import initialize_agent,policy_bin,candidate_policy
from compare_s1_go2_w46_saved import summary,prefix
from w48_build import sha,ROOT
REL=Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
SOURCE=ROOT/REL/'w46-mechanisms'
LARGE=ROOT/REL/'w48/task-a'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/task-a'
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def run(label,mode,solve=True,policy=None):
 source=SOURCE/label;out=LARGE/label/mode;out.mkdir(parents=True,exist_ok=False)
 build='S5-r35' if mode=='old' else 'S5-r36';frozen=ROOT/'work/r11-frozen'/build
 f=json.loads((source/'input.json').read_text());state=dict(np.load(source/'input-state.npz'));state['time']=float(state['time'])
 model=mujoco.MjModel.from_xml_path(str(source/'model.xml'))
 reference=source/'constant/r29-current-plan.bin.r35-reference-1.bin'
 inp=policy or source/'input-policy.bin'
 env=dict(os.environ,MJPC_TASKS_DIR=str(frozen/'runtime-tasks'),LD_LIBRARY_PATH=str(frozen/'lib'),
  MJPC_R33_INPUT_POLICY=str(inp),MJPC_R33_MECHANISM='N',MJPC_R34_ACCEPTANCE='strict',
  MJPC_R34_CONTEXT=str(source/'context.bin'),MJPC_R35_REFERENCE=str(reference),MJPC_R35_BRANCH='constant',
  MJPC_R29_SNAPSHOT=str(out/'r29-current-plan.bin'),MJPC_R29_CALL_RECORD=str(out/'planner-calls.jsonl'),
  MJPC_R27_PLAN_RECORD=str(out/'planner-records.jsonl'),MJPC_R27_ACTION_RECORD=str(out/'feedback-actions.jsonl'))
 for k in ['MJPC_R34_SCORE_ONLY','MJPC_W47_INDEXED_SCORE','MJPC_R35_DIAGNOSTIC']:env.pop(k,None)
 if build=='S5-r36':env['MJPC_W48_PRECALL']=str(out/'precall.bin')
 if not solve:env.update(MJPC_R34_SCORE_ONLY='1',MJPC_R35_DIAGNOSTIC='1')
 identities={k:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for k,p in dict(
  input_policy=inp,state=source/'input-state.npz',context=source/'context.bin',
  reference=reference,model=source/'model.xml',fixture=source/'input.json').items()}
 r=dict(label=label,mode=mode,build=build,status='starting',started_unix_s=time.time(),identity=identities,
  binary_sha256=sha(frozen/'agent_server'),library_sha256=sha(frozen/'lib/libmujoco.so.3.2.3'),
  native_optimizer_solves=int(solve),fixed_policy_rollouts=int(not solve),plant_episodes=0,new_PPO_samples=0,new_PPO_updates=0,
  deadline_unix_s=None,wall_deadline_enabled=False,entry='normal OptimizePolicy' if solve else 'offline score-only ActionImpl/Rollout',
  old_repeat_reason='W46 record did not retain per-process exe/env and used generated rather than identical imported frozen reference.' if mode=='old' else None)
 write(out/'result.json',r);write(SMALL/(label+'-'+mode+'.json'),r)
 start=time.perf_counter()
 try:
  with (out/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
   server_binary_path=frozen/'agent_server',extra_flags=['--mjpc_workers=10'],
   subprocess_kwargs=dict(stdout=log,stderr=log,env=env),run_init=False) as agent:
   initialize_agent(agent,f,state)
   pid=agent.server_process.pid;r.update(status='running',pid=pid,actual_exe=str(Path(f'/proc/{pid}/exe').resolve()),
    actual_exe_sha256=sha(Path(f'/proc/{pid}/exe')),affinity=sorted(os.sched_getaffinity(pid)),
    actual_cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(),
    actual_mjpc_environment={k:v for k,v in env.items() if k.startswith('MJPC_')},
    task_parameters=agent.get_task_parameters(),cost_weights=agent.get_cost_weights(),mode_readback=agent.get_mode())
   write(out/'result.json',r);write(SMALL/(label+'-'+mode+'.json'),r)
   agent.planner_step()
   candidates=parse_candidates(out/'r29-current-plan.bin.r33-candidates')
   raw=np.fromfile(out/'r29-current-plan.bin',dtype='<f8');h=raw[:16]
   arrays,_=take_policy_arrays(raw,16,*map(int,h[3:8]))
   ident=policy_hash(arrays,float(h[12]),int(h[13]))
   chosen=next(c for c in candidates if policy_hash(c['arrays'],c['metadata']['feedback_scale'],c['metadata']['representation'])==ident and abs(c['metadata']['native_J']-h[15])<1e-12)
   pol,head=candidate_policy(chosen);policy_bin(out/'selected-policy.bin',pol,head)
   np.savez_compressed(out/'selected.npz',**chosen['arrays'])
   selected=summary(chosen,f)
   selected['recorded_prefix_substeps']=selected.pop('native_substeps');selected.update(rollout_held_intervals=175,rollout_physics_substeps=1750)
   r.update(status='completed',selected=selected,selected_policy_sha256=sha(out/'selected-policy.bin'),
    installed_policy_identity=ident,candidate_count=len(candidates),actual_scale=float(h[12]),header=h.tolist(),
    candidates=[dict(**c['metadata'],J_parts=cost_terms(c['arrays']['residual'],f['cost_definitions']),
     policy_identity=policy_hash(c['arrays'],c['metadata']['feedback_scale'],c['metadata']['representation'])) for c in candidates])
   if solve:
    call=json.loads((out/'planner-calls.jsonl').read_text().splitlines()[-1]);r['call']=call
    r['work']={k:call[k] for k in ['native_new_rollouts','transition_fd_calls','risk_geometry_fd_calls','derivative_passes','backward_attempts','candidate_records']}
   else:
    # Inspect raw/clip/projected outputs on this already-generated trajectory, without integration.
    a=chosen['arrays']
    for x,t in zip(a['states'][:-1],a['times'][:-1]):
     agent.set_state(time=float(t),qpos=x[:model.nq],qvel=x[model.nq:model.nq+model.nv],act=x[model.nq+model.nv:])
     agent.get_action(time=float(t),nominal_action=False,averaging_duration=0)
    r['work']=dict(native_new_rollouts=1,transition_fd_calls=0,risk_geometry_fd_calls=0,derivative_passes=0,
     physics_substeps=selected['rollout_physics_substeps'],held_intervals=len(a['actions']),action_queries_no_integration=len(a['actions']))
    expected=np.load(LARGE/label/'corrected/selected.npz')
    differences={k:float(np.max(abs(a[k]-expected[k]))) for k in ['states','actions','times','residual']}
    idx=np.flatnonzero(np.any(a['actions']!=expected['actions'],axis=1))
    si=np.flatnonzero(np.any(a['states']!=expected['states'],axis=1))
    r['deployment_comparison']=dict(score_source=str(LARGE/label/'corrected/selected.npz'),
     score_path='ActionIndex/RolloutDiscrete at scale1' if json.loads((LARGE/label/'corrected/result.json').read_text())['call']['candidate_source'].startswith('alpha_') else 'ActionImpl/Rollout at actual recorded scale',
     deployment_path='ActionImpl/Rollout at actual recorded scale',max_abs_errors=differences,
     first_action_difference_node=int(idx[0]) if len(idx) else None,first_state_difference_node=int(si[0]) if len(si) else None,
     action_time_s=float(a['times'][idx[0]]) if len(idx) else None,alpha_applied_again=False,
     delta_J=selected['native_J']-json.loads((LARGE/label/'corrected/result.json').read_text())['selected']['native_J'])
    assert selected['rollout_physics_substeps']==1750
 except Exception:
  r.update(status='technical_failure',traceback=traceback.format_exc());raise
 finally:
  r.update(ended_unix_s=time.time(),wall_s=time.perf_counter()-start)
  write(out/'result.json',r);write(SMALL/(label+'-'+mode+'.json'),r)
 return r
def one(label):
 old=run(label,'old');new=run(label,'corrected')
 for k in old['identity']:assert old['identity'][k]==new['identity'][k],k
 for k in ['task_parameters','cost_weights','mode_readback']:assert old[k]==new[k],k
 for k in ['task_parameters','task_weights','gait_context']:assert old['call'][k]==new['call'][k],k
 deployed=run(label,'deployment',False,LARGE/label/'corrected/selected-policy.bin')
 result=dict(label=label,status='completed',solves=2,fixed_policy_rollouts=1,
  same_input_identity=True,old_source=old['call']['candidate_source'],corrected_source=new['call']['candidate_source'],
  old=old['selected'],corrected=new['selected'],deployment=deployed['deployment_comparison'],
  work={k:old['work'][k]+new['work'][k]+deployed['work'].get(k,0) for k in old['work']},
  deployment_physics_substeps=1750,plant_episodes=0,new_PPO_samples=0,new_PPO_updates=0)
 write(SMALL/(label+'.json'),result);print(json.dumps(result),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--labels',nargs='+',required=True);a=p.parse_args()
 for label in a.labels:one(label)
if __name__=='__main__':main()
