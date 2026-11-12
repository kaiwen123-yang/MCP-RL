#!/usr/bin/env python3
"""W46 fixed four problems, three solves each, three native fixed-policy diagnostics each."""
import argparse,copy,hashlib,json,os,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from mujoco_mpc.agent import Agent
from compare_s1_go2_w44_saved import cost_definitions,cost_terms,parse_candidates,policy_hash,take_policy_arrays
from compare_s1_go2_w45_saved import policy_bin,candidate_policy,initialize_agent,MISSING
ROOT=Path(__file__).resolve().parents[1]
REL=Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
OUT=ROOT/REL/'w46-mechanisms'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46'
CASES=[('A148','a',1,148,136),('A153','a',1,153,140),('B155','b',2,155,141),('B158','b',2,158,144)]
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def ma(v):return float(np.max(np.abs(v)))
def prepare():
 for label,suffix,seed,call,pid in CASES:
  source=ROOT/REL/f'job-20261112T103825Z-w45-{suffix}-eval-PANN_PPO_S5-s{seed}-E30-attempt01'
  dest=OUT/label;dest.mkdir(parents=True,exist_ok=False)
  plan=source/f'accepted-reference-gain-H-plan{pid:06d}.npz'
  policy=dict(np.load(plan))
  row=next(json.loads(s) for s in (source/'planner-calls.jsonl').open() if json.loads(s)['call_id']==call)
  origin=row['origin']
  with np.load(source/'trajectory.npz') as z:
   i=int(np.argmin(abs(z['time']-origin)));assert abs(z['time'][i]-origin)<1e-8
   state=dict(time=z['time'][i],qpos=z['qpos'][i],qvel=z['qvel'][i],act=np.zeros(0),
       mocap_pos=z['mocap_pos'][i],mocap_quat=z['mocap_quat'][i],userdata=z['support_reference_userdata'][i])
  resolved=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/f'{source.name}-resolved.json'
  settings=[s for s in json.loads(resolved.read_text())['applied_settings'] if s['time_s']<=origin+1e-10][-1]
  np.savez_compressed(dest/'input-state.npz',**state)
  np.asarray(row['gait_context'],dtype='<f8').tofile(dest/'context-r34.bin')
  np.asarray(row['gait_context']+[1.],dtype='<f8').tofile(dest/'context.bin')
  policy_bin(dest/'original-policy.bin',policy,policy['header'])
  z=np.load(source/f'r33-candidates-call{call:06d}.npz')
  nd=next(x for x in reversed(row['r34_decisions']) if x['kind']==0 and x['replaced'])
  n=next(k[:-6] for k in z.files if k.endswith('header') and int(z[k][2])==0 and int(z[k][3])==nd['slot'])
  baseline={k:z[n+k].copy() for k in ['reference_states','reference_actions','reference_times','reference_residual','K','H','base_Qu','base_Qxu','base_active']}
  bh=policy['header'].copy();h=z[n+'header'];bh[[10,12]]=h[10];bh[11]=h[4];bh[15]=h[11]
  policy_bin(dest/'original-baseline.bin',baseline,bh)
  for name,pol,head in [('input-policy.bin',policy,policy['header']),('baseline-policy.bin',baseline,bh)]:
   pol=dict(pol);pol['reference_residual']=np.pad(pol['reference_residual'],((0,0),(0,12)))
   head=head.copy();head[7]+=12;policy_bin(dest/name,pol,head)
  old_model=source/'model/G1-project-go2.xml';tree=ET.parse(old_model)
  sensors=tree.find('sensor');index=next(i for i,s in enumerate(sensors) if s.tag!='user')
  sensors.insert(index,ET.Element('user',name='InputReference',dim='12',user='0 1 0 1'))
  ET.SubElement(tree.find('custom'),'numeric',name='r35_input_reference',data='1')
  modelpath=dest/'model.xml';tree.write(modelpath,encoding='unicode')
  model=mujoco.MjModel.from_xml_path(str(modelpath))
  prev=None
  for p in sorted(source.glob('accepted-reference-gain-H-plan*.npz'),reverse=True):
   with np.load(p) as q:
    if float(q['header'][2])>=origin-1e-8:continue
    ti=int(np.argmin(abs(q['accepted_times']-origin)))
    if abs(q['accepted_times'][ti]-origin)>1e-8:continue
    pred=q['accepted_states'][ti].copy();prev=dict(path=str(p.relative_to(ROOT)),origin_s=float(q['header'][2]),prediction_node=ti,observed_at_s=origin);break
  if prev is None:raise ValueError('No preceding saved prediction for measured perturbation '+label)
  dq=np.empty(model.nv);mujoco.mj_differentiatePos(model,dq,1.,pred[:model.nq],state['qpos'])
  delta=np.r_[dq,state['qvel']-pred[model.nq:]]
  np.savez_compressed(dest/'perturbation.npz',delta=delta,predicted_state=pred)
  original_settings=copy.deepcopy(settings);settings['cost_weights']['InputReference']=1.
  f=dict(label=label,source=str(source.relative_to(ROOT)),source_call=call,source_plan=pid,origin_s=origin,
    model_path=str(modelpath.relative_to(ROOT)),original_model_path=str(old_model.relative_to(ROOT)),settings=settings,
    original_settings=original_settings,missing=MISSING,historical_replay=False,
    common_reference_source='same saved deployed output policy, rerolled at current state with shared reconstructed context',
    baseline_source=f'{source}/r33-candidates-call{call:06d}.npz:{n}',perturbation_source=prev,
    perturbation_abs_max_tangent_position=ma(delta[:18]),perturbation_abs_max_velocity=ma(delta[18:]),
    cost_definitions=cost_definitions(modelpath,settings['cost_weights']),
    input_policy_sha256=sha(dest/'input-policy.bin'),original_policy_sha256=sha(dest/'original-policy.bin'),
    original_model_sha256=sha(old_model),context_sha256=sha(dest/'context.bin'))
  write(dest/'input.json',f)
 write(SMALL/'fixtures.json',dict(cases=[json.loads((OUT/c[0]/'input.json').read_text()) for c in CASES]))
def state_at(dest,sign=0):
 s=dict(np.load(dest/'input-state.npz'));s['time']=float(s['time'])
 if sign:
  delta=np.load(dest/'perturbation.npz')['delta']
  m=mujoco.MjModel.from_xml_path(str(dest/'model.xml'))
  mujoco.mj_integratePos(m,s['qpos'],delta[:m.nv],float(sign));s['qvel']+=sign*delta[m.nv:]
 return s
def prefix(raw,nr):
 rows=[];i=0
 while i<len(raw):
  h=raw[i:i+5];i+=5;nc=int(h[4]);r=dict(node=int(h[0]),sub=int(h[1]),before=float(h[2]),after=float(h[3]))
  for name,n in [('pre',37),('post',37),('raw_pd',12),('actuator',12),('passive',18),('residual',nr)]:
   r[name]=raw[i:i+n];i+=n
  r['contacts']=raw[i:i+8*nc].reshape(nc,8);i+=8*nc;rows.append(r)
 assert i==len(raw)
 return rows
def summary(c,f):
 a=c['arrays'];m=c['metadata'];s=a['states'];rpy=Rotation.from_quat(s[:,[4,5,6,3]]).as_euler('xyz')
 parts=cost_terms(a['residual'],f['cost_definitions'])
 rows=prefix(a['prefix'],a['residual'].shape[1]);torque=np.array([r['actuator'] for r in rows])
 limits=np.tile([23.7,23.7,45.43],4)
 feet=a['feet'].reshape(-1,29)[:,1:].reshape(-1,4,7)
 return dict(**m,J0=sum(v for k,v in parts.items() if k!='InputReference'),Jref=parts['InputReference'],
  parts=parts,cost_recompute_error=abs(sum(parts.values())-m['native_J']),
  dx_m=float(s[-1,0]-s[0,0]),dy_m=float(s[-1,1]-s[0,1]),z_min_m=float(s[:,2].min()),
  terminal_z_vz=[float(s[-1,2]),float(s[-1,21])],max_abs_roll_pitch_rad=ma(rpy[:,:2]),
  body_fall=bool((s[:,2]<.12).any() or (abs(rpy[:,:2])>1.2).any()),
  target_jump_max_rad=ma(np.diff(a['actions'],axis=0)),target_jump_p95_rad=float(np.percentile(abs(np.diff(a['actions'],axis=0)),95)),
  torque_adjacent_substep_max_Nm=ma(np.diff(torque,axis=0)),torque_abs_max_Nm=ma(torque),
  actuator_saturation_fraction=float(np.mean(abs(torque)>=limits-1e-9)),native_substeps=len(rows),
  nofoot_nodes=int(np.sum(feet[:,:,4:6].sum(axis=(1,2))<=0)),
  policy_sha256=policy_hash(a,m['feedback_scale'],m['representation']))
def run_native(label,branch,directory,policy=None,sign=0,reference=None,legacy=False,solve=False):
 dest=OUT/label;f=json.loads((dest/'input.json').read_text());directory.mkdir(parents=True,exist_ok=False)
 frozen=ROOT/'work/r11-frozen'/('S5-r34' if legacy else 'S5-r35')
 model=mujoco.MjModel.from_xml_path(str(ROOT/f['original_model_path'] if legacy else dest/'model.xml'))
 state=state_at(dest,sign)
 env=dict(os.environ,MJPC_TASKS_DIR=str(frozen/'runtime-tasks'),LD_LIBRARY_PATH=str(frozen/'lib'),
  MJPC_R33_INPUT_POLICY=str(policy or dest/('original-policy.bin' if legacy else 'input-policy.bin')),
  MJPC_R33_MECHANISM='N',MJPC_R34_ACCEPTANCE='strict',MJPC_R34_CONTEXT=str(dest/('context-r34.bin' if legacy else 'context.bin')),
  MJPC_R29_SNAPSHOT=str(directory/'r29-current-plan.bin'),MJPC_R29_CALL_RECORD=str(directory/'planner-calls.jsonl'),
  MJPC_R27_PLAN_RECORD=str(directory/'planner-records.jsonl'),MJPC_R27_ACTION_RECORD=str(directory/'feedback-actions.jsonl'),
  MJPC_R35_BRANCH=branch)
 if not solve:env.update(MJPC_R34_SCORE_ONLY='1',MJPC_R35_DIAGNOSTIC='1')
 if reference:env['MJPC_R35_REFERENCE']=str(reference)
 begin=time.perf_counter()
 with (directory/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
   server_binary_path=frozen/'agent_server',extra_flags=['--mjpc_workers=10'],
   subprocess_kwargs=dict(stdout=log,stderr=log,env=env),run_init=False) as agent:
  ff=dict(f)
  if legacy:ff['settings']=f['original_settings']
  initialize_agent(agent,ff,state);agent.planner_step()
  candidates=parse_candidates(directory/'r29-current-plan.bin.r33-candidates')
  raw=np.fromfile(directory/'r29-current-plan.bin',dtype='<f8');h=raw[:16]
  arrays,_=take_policy_arrays(raw,16,*map(int,h[3:8]))
  ident=policy_hash(arrays,float(h[12]),int(h[13]))
  chosen=next(c for c in candidates if policy_hash(c['arrays'],c['metadata']['feedback_scale'],c['metadata']['representation'])==ident and abs(c['metadata']['native_J']-h[15])<1e-12)
  if solve:
   p,head=candidate_policy(chosen);policy_bin(directory/'selected-policy.bin',p,head)
   call=json.loads((directory/'planner-calls.jsonl').read_text().splitlines()[-1])
   report=dict(label=label,branch=branch,status='completed',call=call,wall_s=time.perf_counter()-begin,
      selected=summary(chosen,f),candidate_count=len(candidates),installed_policy_identity=ident,
      derivative_evidence=[json.loads(x) for x in (directory/'r29-current-plan.bin.r35-derivatives.jsonl').read_text().splitlines()])
  else:
   if legacy:
    report=dict(label=label,branch='legacy_equivalence',status='completed',wall_s=time.perf_counter()-begin,
      native_J=chosen['metadata']['native_J'])
   else:report=dict(label=label,branch=branch,sign=sign,status='completed',wall_s=time.perf_counter()-begin,summary=summary(chosen,f))
  np.savez_compressed(directory/'selected.npz',**chosen['arrays'])
  write(directory/'result.json',report)
 return report
def one(label):
 dest=OUT/label;results=[]
 for branch in ['none','constant','load']:
  directory=dest/branch;start=time.perf_counter()
  r=run_native(label,branch,directory,solve=True)
  reference=directory/'r29-current-plan.bin.r35-reference-1.bin'
  r['reference_sha256']=sha(reference);r['diagnostics']=[]
  for sign,name in [(0,'nominal'),(1,'plus'),(-1,'minus')]:
   d=run_native(label,branch,directory/name,policy=directory/'selected-policy.bin',sign=sign,reference=reference)
   r['diagnostics'].append(d)
  write(directory/'result.json',r);results.append(r)
  print(json.dumps(dict(label=label,branch=branch,wall_s=time.perf_counter()-start,selected=r['selected'],
                       diagnostic=[d['summary'] for d in r['diagnostics']])),flush=True)
 # Same old policy/context with zero new residual: score-only equivalence, no extra solve.
 old=run_native(label,'none',dest/'old-r34-equivalence',legacy=True)
 new=run_native(label,'none',dest/'r35-off-equivalence',reference=dest/'none/r29-current-plan.bin.r35-reference-1.bin')
 x=np.load(dest/'old-r34-equivalence/selected.npz');y=np.load(dest/'r35-off-equivalence/selected.npz')
 equivalence=dict(J_error=abs(old['native_J']-new['summary']['native_J']),
  states_max_error=ma(x['states']-y['states']),actions_max_error=ma(x['actions']-y['actions']),
  old_residual_max_error=ma(x['residual']-y['residual'][:,:101]))
 refs=[np.fromfile(dest/b/'r29-current-plan.bin.r35-reference-1.bin',dtype='<f8') for b in ['none','constant','load']]
 shared=ma(refs[0][8:8+175*12+176*4]-refs[1][8:8+175*12+176*4])
 shared=max(shared,ma(refs[0][8:8+175*12+176*4]-refs[2][8:8+175*12+176*4]))
 const=refs[1][-175*12:].reshape(175,12);load=refs[2][-175*12:].reshape(175,12)
 result=dict(label=label,status='completed',results=results,shared_ubar_force_max_error=shared,
  trace_match_max_error=ma(const.sum(1)-load.sum(1)),disabled_equivalence=equivalence,
  finite_solves=3,fixed_policy_diagnostics=9,ancillary_equivalence_rollouts=2,
  physical_episodes=0,new_PPO_samples=0,new_PPO_updates=0)
 write(SMALL/(label+'.json'),result)
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--labels',nargs='+');a=p.parse_args()
 if a.prepare:prepare()
 else:
  for label in a.labels:one(label)
if __name__=='__main__':main()
