#!/usr/bin/env python3
"""Four W44 output problems, shared recovered context, finite old/new acceptance."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import mujoco
import numpy as np
from mujoco_mpc.agent import Agent
from compare_s1_go2_w44_saved import cost_definitions,cost_terms,parse_candidates,policy_hash
from s1_go2_r29_records import R29Records
ROOT=Path(__file__).resolve().parents[1]
REL=Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
OUT=ROOT/REL/'w45-mechanisms'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45'
CASES=[('A158','a',1,'143358',158,151),('A163','a',1,'143358',163,156),
       ('B153','b',2,'143400',153,143),('B173','b',2,'143400',173,161)]
MISSING=['Original plant qacc_warmstart/contact solver memory was not saved.',
 'Original pre-call incumbent and preceding optimizer lifecycle are not reconstructed: saved call output is the common imported policy.',
 'Only recorded task/gait/filter fields are restored; absent earlier private transition history is not invented.']
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def relative(p):return str(Path(p).relative_to(ROOT))
def policy_bin(p,arrays,h):
 values=[h]
 for key in ['reference_states','reference_actions','reference_times','reference_residual','K','H','base_Qu','base_Qxu','base_active']:
  values.append(arrays[key].ravel())
 np.concatenate(values).astype('<f8').tofile(p)
def candidate_policy(c):
 a=c['arrays'];m=c['metadata']
 return {'reference_'+k:a['reference_'+k] for k in ['states','actions','times','residual']}|dict(
 K=a['K'],H=a['H'],base_Qu=a['Qu'],base_Qxu=a['Qxu'],base_active=a['active']),np.array([
 32,m['input_policy_id'],m['origin_s'],m['horizon'],37,12,36,a['reference_residual'].shape[1],
 m['mu'],0,m['feedback_scale'],m['alpha'],m['feedback_scale'],m['representation'],96,m['native_J']],float)
def budget(residual,defs):
 magnitude=sum(abs(v) for v in cost_terms(residual,defs).values())
 magnitude+=sum(2*abs(d['weight']*d['norm_parameter'])*(d['end']-d['start'] if d['norm']==6 else 1)
                for d in defs if d['norm'] in (2,6))
 n=len(residual)*(8*residual.shape[1]+4*len(defs)+2)+1;u=np.finfo(float).eps/2
 return float(n*u/(1-n*u)*magnitude)
def prepare(case):
 label,suffix,seed,stamp,call,pid=case
 source=ROOT/REL/f'job-20261110T{stamp}Z-w44-{suffix}-eval-PANN_PPO_S5-s{seed}-E30'
 dest=OUT/label;dest.mkdir(parents=True,exist_ok=False)
 plan=source/f'accepted-reference-gain-H-plan{pid:06d}.npz'
 with np.load(plan) as z:policy={k:z[k].copy() for k in z.files}
 row=next(json.loads(s) for s in (source/'planner-calls.jsonl').open() if json.loads(s)['call_id']==call)
 origin=row['origin']
 with np.load(source/'trajectory.npz') as z:
  i=int(np.argmin(abs(z['time']-origin)));assert abs(z['time'][i]-origin)<1e-8
  state=dict(time=z['time'][i],qpos=z['qpos'][i],qvel=z['qvel'][i],act=np.zeros(0),
    mocap_pos=z['mocap_pos'][i],mocap_quat=z['mocap_quat'][i],userdata=z['support_reference_userdata'][i])
 resolved=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/f'{source.name}-resolved.json'
 settings=[s for s in json.loads(resolved.read_text())['applied_settings'] if s['time_s']<=origin+1e-10][-1]
 np.savez_compressed(dest/'input-state.npz',**state)
 np.asarray(row['gait_context'],dtype='<f8').tofile(dest/'context.bin')
 policy_bin(dest/'input-policy.bin',policy,policy['header'])
 model=source/'model/G1-project-go2.xml'
 fixture=dict(label=label,source_call=call,source_plan=pid,origin_s=origin,settings=settings,
  model_path=relative(model),source=relative(source),policy_sha256=sha(dest/'input-policy.bin'),
  state_sha256=sha(dest/'input-state.npz'),context_sha256=sha(dest/'context.bin'),
  source_plan_sha256=sha(plan),source_resolved_sha256=sha(resolved),gait_context=row['gait_context'],
  missing=MISSING,historical_replay=False,cost_definitions=cost_definitions(model,settings['cost_weights']))
 # Two actual source identities; unavailable original archive -> two explicitly duplicate slots.
 candidates=source/f'r33-candidates-call{call:06d}.npz'
 baseline=policy;baseline_h=policy['header'];baseline_source='saved_output_duplicate_no_archive'
 if candidates.exists():
  with np.load(candidates) as z:
   names=[k[:-6] for k in z.files if k.endswith('header')]
   nominal=[n for n in names if int(z[n+'header'][2])==0 and not z[n+'header'][13]]
   n=min(nominal,key=lambda n:(z[n+'header'][12],z[n+'header'][11]))
   baseline={k:z[n+k].copy() for k in ['reference_states','reference_actions','reference_times','reference_residual','K','H','base_Qu','base_Qxu','base_active']}
   h=z[n+'header'];baseline_h=policy['header'].copy();baseline_h[[10,12]]=h[10];baseline_h[11]=h[4];baseline_h[15]=h[11]
   baseline_source=relative(candidates)+':'+n
   if call in (163,173):
    selected='candidate019_';hsel=z[selected+'header']
    fixture['historical_rule']=dict(nominal_candidate=n,selected_candidate=selected,
     nominal_J=float(h[11]),selected_J=float(hsel[11]),decrease=float(h[11]-hsel[11]),
     tolerance=budget(z[n+'accepted_residual'],fixture['cost_definitions'])+budget(z[selected+'accepted_residual'],fixture['cost_definitions']),
     nominal_scale=float(h[10]),selected_scale=float(hsel[10]),alpha=float(hsel[4]),
     old_accepts=True,new_accepts=False,new_reason='alpha0_no_step',
     K_max_abs_change=float(np.max(abs(z[n+'K']-z[selected+'K']))))
 policy_bin(dest/'baseline-policy.bin',baseline,baseline_h)
 fixture['gpu_baseline_source']=baseline_source
 write(dest/'input.json',fixture);return fixture
def initialize_agent(agent,fixture,state):
 agent.init(task_id='Quadruped Flat',model=agent.model,send_as='xml')
 agent.set_mode('Quadruped')
 agent.set_task_parameters({'Gait switch':'Manual','Gait':fixture['settings']['task_parameters']['Gait']})
 agent.set_state(**state)
 numeric={k:v for k,v in fixture['settings']['task_parameters'].items() if not isinstance(v,str)}
 agent.set_task_parameters(numeric);agent.set_cost_weights(fixture['settings']['cost_weights']);agent.set_state(**state)
def execute(label,rule,workers,score_only=False,policy_path=None,tag=None):
 dest=OUT/label;fixture=json.loads((dest/'input.json').read_text())
 run=dest/(tag or rule);run.mkdir(exist_ok=False)
 frozen=ROOT/'work/r11-frozen/S5-r34'
 model=mujoco.MjModel.from_xml_path(str(ROOT/fixture['model_path']))
 with np.load(dest/'input-state.npz') as z:state={k:z[k].copy() for k in z.files}
 state['time']=float(state['time'])
 env=dict(os.environ,MJPC_TASKS_DIR=str(frozen/'runtime-tasks'),LD_LIBRARY_PATH=str(frozen/'lib'),
  MJPC_R33_INPUT_POLICY=str(policy_path or dest/'input-policy.bin'),MJPC_R33_MECHANISM='N',
  MJPC_R34_CONTEXT=str(dest/'context.bin'),MJPC_R34_ACCEPTANCE=rule,
  MJPC_R29_SNAPSHOT=str(run/'r29-current-plan.bin'),MJPC_R29_CALL_RECORD=str(run/'planner-calls.jsonl'),
  MJPC_R27_PLAN_RECORD=str(run/'planner-records.jsonl'),MJPC_R27_ACTION_RECORD=str(run/'feedback-actions.jsonl'))
 if score_only:env['MJPC_R34_SCORE_ONLY']='1'
 begin=time.perf_counter()
 with (run/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
   server_binary_path=frozen/'agent_server',extra_flags=[f'--mjpc_workers={workers}'],
   subprocess_kwargs=dict(stdout=log,stderr=log,env=env),run_init=False) as agent:
  initialize_agent(agent,fixture,state)
  rpc=time.perf_counter();agent.planner_step();wall=time.perf_counter()-rpc
  arrays=parse_candidates(run/'r29-current-plan.bin.r33-candidates')
  if score_only:
   first=arrays[0]
   agent.set_state(**state)
   repeat=time.perf_counter();agent.planner_step();repeat_wall=time.perf_counter()-repeat
   second=parse_candidates(run/'r29-current-plan.bin.r33-candidates')[0]
   result=dict(label=label,rule=rule,status='completed',native_new_rollouts=2,derivative_passes=0,
    first_J=first['metadata']['native_J'],repeat_J=second['metadata']['native_J'],
    repeated_J_error=abs(first['metadata']['native_J']-second['metadata']['native_J']),
    repeated_states_max_abs=float(np.max(abs(first['arrays']['states']-second['arrays']['states']))),
    cost_budget=budget(first['arrays']['residual'],fixture['cost_definitions']),rpc_wall_s=wall,
    repeat_wall_s=repeat_wall)
  else:
   records=R29Records(run,model);records.controller='S5-r34';records.plan(state['time'])
   records.event(state['time'],'four_saved_problem_comparison');records.finish(state['time'],'finite_solver_finished')
   call=json.loads((run/'planner-calls.jsonl').read_text().splitlines()[-1])
   result=dict(label=label,rule=rule,status='completed',rpc_wall_s=wall,call=call,
     native_new_rollouts=call['native_new_rollouts'],derivative_passes=call['derivative_passes'],
     candidates=[dict(c['metadata'],policy_sha256=policy_hash(c['arrays'],c['metadata']['feedback_scale'],c['metadata']['representation']),
       cost_budget=budget(c['arrays']['residual'],fixture['cost_definitions'])) for c in arrays])
  result['wall_s']=time.perf_counter()-begin
 write(run/'result.json',result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=10);a=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);fixtures=[];results=[]
 for case in CASES:
  f=prepare(case);fixtures.append(f)
  for rule in ['legacy','strict']:
   r=execute(case[0],rule,a.workers);results.append(r)
   print(json.dumps(dict(label=case[0],rule=rule,source=r['call']['candidate_source'],
      cost=r['call']['deployed_current_cost'],wall_s=r['wall_s'])),flush=True)
  results.append(execute(case[0],'strict',a.workers,score_only=True,tag='repeat'))
  write(SMALL/'mechanism-evidence.json',dict(status='running',fixtures=fixtures,results=results,historical_replay=False))
 result=dict(status='completed',fixtures=fixtures,results=results,historical_replay=False,missing=MISSING,
  physical_episodes=0,perturbed_rollouts=0,new_PPO_samples=0,new_PPO_updates=0,
  comparison='One legacy and one strict finite solve per saved output problem; same binary/state/context. Two identical full policy scores per problem quantify repetition error.',
  total_native_rollouts=sum(r['native_new_rollouts'] for r in results))
 write(SMALL/'mechanism-evidence.json',result)
 print(json.dumps(dict(status='completed',total_native_rollouts=result['total_native_rollouts'])),flush=True)
if __name__=='__main__':main()
