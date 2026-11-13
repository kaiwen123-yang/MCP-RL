#!/usr/bin/env python3
"""W47 Task C: read existing D57 originals, without physics, native calls or GPU."""
import csv, hashlib, json, os, struct, sys, time, xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
OUT=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/task-c'
GPU=Path('/root/autodl-tmp/MCP-RL')/BASE/'gpu/w46'
POL=ROOT/BASE/'w46-mechanisms'
sys.path.insert(0,str(ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review'))
from policy_arrays import raw_policy, state_diff
SOURCES={}
def source(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 d={'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()};SOURCES[str(p)]=d;return d

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def read(p):source(p);return json.loads(p.read_text())
def stats(x):
 a=np.asarray([v for v in x if v is not None and np.isfinite(v)],float)
 if not a.size:return {'n':0}
 return dict(n=int(a.size),min=float(a.min()),max=float(a.max()),mean=float(a.mean()),p50=float(np.median(a)),p95=float(np.quantile(a,.95)))
def counts(rows,key):return dict(sorted(Counter(str(r[key]) for r in rows).items()))
def gait(row):return int(struct.unpack('<q',struct.pack('<d',row['task_parameters'][0]))[0])
def group(rows):
 attempts=[(r,s) for r in rows for s in r['search_attempts']]
 cand=[(r,s,c) for r,s in attempts for c in s.get('native_candidates',[])]
 pos=[(r,s,c) for r,s,c in cand if c['alpha']>0]
 zero=[(r,s,c) for r,s,c in cand if c['alpha']==0]
 decisions=[(r,d) for r in rows for d in r['r34_decisions']]
 finalpos=[r for r in rows if r['policy_updated'] and r['candidate_source'].startswith('alpha_') and r['chosen_alpha']>0]
 out=dict(calls=len(rows),physical_origin_s=stats([r['origin'] for r in rows]),candidate_source=counts(rows,'candidate_source'),
  stop_reason=counts(rows,'stop_reason'),return_reason=counts(rows,'return_reason'),
  r35_enabled=counts(rows,'r35_enabled'),gait_enum=dict(Counter(str(gait(r)) for r in rows)),
  final_positive_step_installs=len(finalpos),
  final_zero_step_installs=sum(r['policy_updated'] and r['candidate_source'].startswith('alpha_') and r['chosen_alpha']==0 for r in rows),
  nominal_installs=sum(r['policy_updated'] and r['candidate_source'].startswith('nominal_') for r in rows),
  incumbent_retained=sum(r['return_reason']=='incumbent_retained' for r in rows),
  search_attempt_status=dict(Counter(s['status'] for _,s in attempts)),
  r34_tournament_decision_reason=dict(Counter(d['reason'] for _,d in decisions)),
  tournament_replacements=sum(d['replaced'] for _,d in decisions),
  positive_candidates=len(pos),zero_candidates=len(zero),
  positive_candidate_failed=sum(c['failed'] for _,_,c in pos),
  positive_candidate_trunk_positive=sum(c['trunk_violation']>0 for _,_,c in pos),
  positive_candidate_projection_failed_nodes=sum(c['projection_failed_count'] for _,_,c in pos),
  positive_candidate_restoration_nodes=sum(c['actuator_restoration_count'] for _,_,c in pos),
  positive_deltaJ=stats([c['cost']-s['nominal_current_cost'] for _,s,c in pos if c['cost'] is not None and s['nominal_current_cost'] is not None]),
  zero_deltaJ=stats([c['cost']-s['nominal_current_cost'] for _,s,c in zero if c['cost'] is not None and s['nominal_current_cost'] is not None]),
  installed_nominal_scale=dict(Counter(str(d['feedback_scale']) for r,d in decisions if r['policy_updated'] and r['candidate_source']==d['source'] and d['kind']==0)),
  planning_wall_s=stats([r['wall_s'] for r in rows]),planning_wall_total_s=sum(r['wall_s'] for r in rows),
  chosen_actual_decrease=stats([r['chosen_actual'] for r in finalpos]),
  deployed_minus_current_incumbent_J=stats([r['deployed_current_cost']-r['incumbent_current_cost'] for r in rows if r['deployed_current_cost'] is not None and r['incumbent_current_cost'] is not None]),
  work={k:sum(r[k] for r in rows) for k in ['native_new_rollouts','candidate_records','cache_hits','derivative_passes','transition_fd_calls','risk_geometry_fd_calls','backward_attempts']})
 out['positive_strict_cost_descent_candidates']=sum(c['cost'] is not None and s['nominal_current_cost'] is not None and c['cost']<s['nominal_current_cost'] for _,s,c in pos)
 return out

def calls_summary():
 allrows=[];out={};flat=[]
 for case,seed in [('a',1),('b',2)]:
  d=ROOT/BASE/f'job-20261112T133729Z-w46-{case}-eval-PANN_PPO_S5-s{seed}-E30-attempt01'
  f=d/'planner-calls.jsonl';source(f);rows=[json.loads(x) for x in f.read_text().splitlines()]
  allrows.extend(rows)
  records=d/'planner-records.jsonl';source(records);recs=[json.loads(x) for x in records.read_text().splitlines()]
  result=read(ROOT/'results'/BASE.relative_to('outputs')/(d.name+'.json'))
  out[case]=dict(all=group(rows),Stand=group([r for r in rows if gait(r)==0]),moving=group([r for r in rows if gait(r)!=0]),
   r35_enabled=group([r for r in rows if r['r35_enabled']]),r35_disabled=group([r for r in rows if not r['r35_enabled']]),
   installed_record_count=len(recs),all_N=all(r['r33_mechanism']=='N' for r in rows),all_r34=all(r['r34_enabled'] for r in rows))
  source(d/'trajectory.npz');z=np.load(d/'trajectory.npz');times=z['time'];dtar=np.max(abs(np.diff(z['q_target'],axis=0)),axis=1)
  rev=np.argwhere(abs(np.diff(z['torque'],axis=0))>=90.86-1e-7)
  events=[]
  for label,idx in [('first_knee_full_reversal',int(rev[0,0]+1)),('first_target_jump_over_1rad',int(np.flatnonzero(dtar>1)[0]+1))]:
   t=float(times[idx]);r=max((r for r in rows if r['origin']<=t+1e-10),key=lambda r:(r['origin'],r['call_id']))
   p=d/f"accepted-reference-gain-H-plan{r['installed_plan_id']:06d}.npz"
   events.append(dict(event=label,event_time_s=t,call_id=r['call_id'],origin_s=r['origin'],installed_plan_id=r['installed_plan_id'],candidate_source=r['candidate_source'],policy_updated=r['policy_updated'],return_reason=r['return_reason'],policy_generation_call=r['policy_generation_call'],policy_generation_time=r['policy_generation_time'],matching_full_policy=source(p) if p.exists() else {'path':str(p),'status':'not_present_in_retained_episode_directory'},context_complete_from_scalar=False))
  out[case]['events']=events
  posrecords=[r for r in recs if r['candidate_source'].startswith('alpha_')]
  out[case]['installed_positive_acceptance']=dict(acceptance_mode=dict(Counter(r['acceptance_mode'] for r in posrecords)),J_descent=sum(r['cost_improvement']>0 for r in posrecords),J_increase=sum(r['cost_improvement']<0 for r in posrecords),J_increase_rows=[{k:r[k] for k in ['source_call_id','origin','acceptance_mode','nominal_cost','accepted_cost','nominal_trunk_violation_m','accepted_trunk_violation_m']} for r in posrecords if r['cost_improvement']<0])
  out[case]['time_bins']={str(i):group([r for r in rows if i<=r['origin']<i+2]) for i in range(0,int(rows[-1]['origin'])+1,2)}
  for r in rows:
   flat.append({k:r[k] for k in ['call_id','origin','wall_s','problem_epoch','candidate_source','stop_reason','return_reason','policy_updated','installed_plan_id','policy_generation_call','r35_enabled','chosen_alpha','chosen_actual','incumbent_current_cost','deployed_current_cost','native_new_rollouts','candidate_records','derivative_passes','transition_fd_calls','risk_geometry_fd_calls']}|{'case':case,'gait_enum':gait(r)})
  out[case]['original_result_top_fields']={k:v for k,v in result.items() if k in ['status','failure','reason','termination_reason','success','duration_s','simulation_s','error','exit_code']}
 with (OUT/'planner-calls.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 out['combined']=group(allrows)
 out['cost_summary_scope']='DeltaJ rows compare each candidate with its own same-call nominal/current incumbent. Cross-epoch distributions are descriptive, never summed as task benefit. Strict J decrease alone is not r34 acceptance; r34 prioritizes trunk violation. Of 1601 positive installations,1598 lower nominal J and3 A infeasible-restoration installs raise J while reducing v.'
 out['objects_distinguished']=dict(original_physical_attempts=2,original_physical_failures=2,original_technical_failures=0,original_blocked_later_stages=6,
  new_W47_physics=0,new_W47_solves=0,new_W47_PPO=0,
  candidate_rejection='Search and r34 decision rows are solver-internal alternatives, not physical episodes. Tournament replacement is provisional, distinct from final installation.',
  counting='Includes startup calls. Work counters sum reported per-call values, not independently reconstructed hidden substeps. Warnings are cumulative and are not summed.')
 return out

def readpolicy(p):
 source(p);z=np.fromfile(p,dtype='<f8');h=z[:16];T,ns,nu,nx,nr=map(int,h[3:8]);i=16;d={}
 for name,shape in [('states',(T,ns)),('actions',(T-1,nu)),('times',(T,)),('residual',(T,nr)),('K',(T,nu,nx)),('H',(T,nu,nu))]:
  n=int(np.prod(shape));d[name]=z[i:i+n].reshape(shape);i+=n
 d['scale']=float(h[12]);d['representation']=int(h[13]);return d

def vector_formula(x,pol,t):
 u,fb,_,_=raw_policy(t,x,pol['times'],pol['states'],pol['actions'],pol['K'],pol['scale']);return u+fb

def describe_max(a,tags,kind='state'):
 idx=tuple(int(i) for i in np.unravel_index(np.argmax(abs(a)),a.shape))
 out={'abs_max':float(abs(a[idx])),'array_index':list(idx),'scenario':tags[idx[-2]],'component':idx[-1]}
 if len(idx)==3:
  if kind=='action':out.update(decision_after_origin_s=.002*idx[0],hold_end_after_origin_s=.002*(idx[0]+1))
  else:out['post_state_after_origin_s']=.002*idx[0]
 return out

def gpu_summary():
 tags=read(GPU/'scenarios.json');source(GPU/'paired-states-actions.npz');z=np.load(GPU/'paired-states-actions.npz')
 source(GPU/'action-algebra.npz');alg=np.load(GPU/'action-algebra.npz');old=read(GPU/'result.json')
 policies={};inputs={};t0={};model_ids={}
 for label in ['A148','A153','B155','B158']:
  inputs[label]=read(POL/label/'input.json');st=POL/label/'input-state.npz';source(st);t0[label]=float(np.load(st)['time']);source(POL/label/'perturbation.npz')
  model_ids[label]=source(GPU/(label+'-model.xml'))['sha256']
  for pol in ['original-baseline.bin','original-policy.bin']:policies[label,pol]=readpolicy(POL/label/pol)
 # The actual GPU batch used compiled models[0], and native models were checked equal on declared physics arrays.
 root=ET.parse(GPU/'A148-model.xml').getroot();acts=list(root.find('actuator'))
 bounds=np.array([list(map(float,a.attrib['ctrlrange'].split())) for a in acts])
 raw=[];clip=[];margins=[];identities=[]
 for i,tag in enumerate(tags):
  p=policies[tag['label'],tag['policy']];x=alg['states'][i];r=vector_formula(x,p,t0[tag['label']]);raw.append(r);clip.append(np.clip(r,bounds[:,0],bounds[:,1]))
  force=np.array([23.7,23.7,45.43]*4);lo=np.maximum(bounds[:,0],x[7:19]+(5*x[25:37]-force)/60);hi=np.minimum(bounds[:,1],x[7:19]+(5*x[25:37]+force)/60)
  margins.append(float(min((clip[-1]-lo).min(),(hi-clip[-1]).min())))
  h=hashlib.sha256()
  for key in ['states','actions','times','K','H']:h.update(np.asarray(p[key],dtype='<f8').tobytes())
  h.update(np.asarray([p['scale'],p['representation']],dtype='<f8').tobytes());ph=h.hexdigest()
  h.update(x.astype('<f8').tobytes());h.update(np.asarray([t0[tag['label']]],dtype='<f8').tobytes());h.update(model_ids[tag['label']].encode())
  identities.append(dict(scenario_index=i,**tag,policy_execution_sha256=ph,scenario_execution_sha256=h.hexdigest(),model_xml_sha256=model_ids[tag['label']],origin_s=t0[tag['label']],raw_recomputed=r.tolist(),clipped_recomputed=clip[-1].tolist(),held_gpu=alg['gpu_action'][i].tolist(),held_native=alg['native_action'][i].tolist(),PD_box_margin_rad=margins[-1]))
 raw=np.asarray(raw);clip=np.asarray(clip);n=z['native_states'];g=z['gpu_states'];un=z['native_actions'];ug=z['gpu_actions']
 tan=np.empty(n.shape[:-1]+(36,))
 for k in range(len(n)):
  for i in range(len(tags)):tan[k,i]=state_diff(n[k,i],g[k,i])
 groups={'base_position_m':slice(0,3),'orientation_tangent_rad':slice(3,6),'joint_position_rad':slice(6,18),'base_linear_velocity_m_s':slice(18,21),'base_angular_velocity_rad_s':slice(21,24),'joint_velocity_rad_s':slice(24,36)}
 def metrics(k):
  return {name:describe_max(tan[k,:,sl],tags) for name,sl in groups.items()}
 short=[]
 for k in range(1,11):
  i,j=np.unravel_index(np.argmax(abs(ug[k-1]-un[k-1])),ug[k-1].shape)
  short.append(dict(decision_after_origin_s=.002*(k-1),hold_start_after_origin_s=.002*(k-1),hold_end_after_origin_s=.002*k,post_state_after_origin_s=.002*k,
    action_max_error_rad=float(abs(ug[k-1,i,j]-un[k-1,i,j])),action_scenario_index=int(i),action_scenario=tags[i],action_joint_index=int(j),state_errors=metrics(k)))
 full={name:describe_max(tan[:,:,sl],tags) for name,sl in groups.items()}
 per_scenario=[]
 for i,tag in enumerate(tags):
  row=dict(index=i,**tag)
  for key,x in [('native',n[:,i]),('gpu',g[:,i])]:
   qw,qx,qy,qz=x[:,3:7].T;norm=np.sqrt(qw*qw+qx*qx+qy*qy+qz*qz);qw,qx,qy,qz=qw/norm,qx/norm,qy/norm,qz/norm
   roll=np.arctan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy));pitch=np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1))
   row[key]=dict(minimum_z_m=float(x[:,2].min()),maximum_abs_roll_rad=float(abs(roll).max()),maximum_abs_pitch_rad=float(abs(pitch).max()),body_fall_sampled=bool((x[:,2]<.12).any() or (abs(roll)>1.2).any() or (abs(pitch)>1.2).any()),final_cumulative_trunk_violation_m=float(z[key+'_trunk'][i]))
  per_scenario.append(row)
 # Restore raw values algebraically from each side's saved state. This is not the unrecorded device output.
 rawg=np.array([[vector_formula(g[k,i],policies[tag['label'],tag['policy']],t0[tag['label']]+.002*k) for i,tag in enumerate(tags)] for k in range(len(ug))])
 rawn=np.array([[vector_formula(n[k,i],policies[tag['label'],tag['policy']],t0[tag['label']]+.002*k) for i,tag in enumerate(tags)] for k in range(len(un))])
 clipg=np.clip(rawg,bounds[:,0],bounds[:,1]);clipn=np.clip(rawn,bounds[:,0],bounds[:,1])
 return dict(status='completed_existing_originals_only',source_round='W45 full baseline/selected policies executed in W46 GPU diagnostic; not r35 new selected',
  scenario_count=len(tags),distinct_policy_execution_count=len(set(x['policy_execution_sha256'] for x in identities)),distinct_scenario_execution_count=len(set(x['scenario_execution_sha256'] for x in identities)),
  joint_order=[a.attrib.get('name',a.attrib.get('joint')) for a in acts],scenarios=identities,
  original_raw_center_field_valid=False,original_raw_device_output_saved=False,raw_provenance='Recomputed offline from saved states and full policy FP64 formula; not historic device raw output.',
  initial=dict(recomputed_clipped_vs_native_held_max_rad=float(abs(clip-alg['native_action']).max()),recomputed_clipped_vs_gpu_held_max_rad=float(abs(clip-alg['gpu_action']).max()),saved_held_difference_max_rad=float(abs(alg['gpu_action']-alg['native_action']).max()),minimum_PD_box_margin_rad=min(margins),all_24_initial_centers_inside_PD_box=all(v>=0 for v in margins),saved_projection_info_equal=bool(np.array_equal(alg['gpu_info'],alg['native_info']))),
  short_holds=short,full350ms_state_errors=full,full350ms_held_action_error=describe_max(ug-un,tags,'action'),
  full350ms_recomputed_raw_error=describe_max(rawg-rawn,tags,'action'),
  full350ms_held_minus_recomputed_clipped={'gpu':describe_max(ug-clipg,tags,'action'),'native':describe_max(un-clipn,tags,'action')},
  per_scenario_sampled_hard_conditions=per_scenario,
  work_correction=dict(original_recorded_warmup_world_steps=old['warmup_world_steps'],executed_warmup_world_steps=24,capture_recorded_world_steps=24,main_capture_launch_count=1750,main_gpu_world_steps=42000,main_native_world_steps=42000,host_HQP_calls=16896,feedback_kernel_launches=351,native_hold_calls=4200,basis='Original arrays dimensions plus source/API semantics. No new execution.'),
  historical_window_interpretation='350ms retained as a finite negative diagnostic. Zero short-window trunk disagreement did not establish explained dynamics or all original hard-condition agreement.',
  limits=['No saved .2ms GPU contact/normal-force/KKT trace; first engine-state divergence only localizable to initial 2ms hold.','Body checks above are sampled 2ms states; raw source did not save per-substep body conditions.','Model version, compilation, backend and precision are not isolated causes.','No full native J on GPU; no equivalent ranking or online acceleration evidence.'],
  optional_single_hold='not_run: existing native bridge lacks matching every-.2ms compiled-model/contact/solver readbacks for three engines within 30-substep budget; no new physics required for record corrections.',
  original_wall_s=old['wall_s'],original_timing=old['timing'])

def main():
 start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
 c=calls_summary();write(OUT/'planner-summary.json',c)
 g=gpu_summary();write(OUT/'gpu-corrected.json',g)
 source(Path(__file__))
 for p in ['scripts/s1_go2_w46_gpu.py','scripts/analyze_s1_go2_w46.py','scripts/s1_go2_w46_native_bridge.cc','results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/policy_arrays.py']:source(ROOT/p)
 run=dict(status='completed',deadline_unix_s=None,wall_deadline_enabled=False,pid=os.getpid(),host_affinity=sorted(os.sched_getaffinity(0)),library_threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},elapsed_wall_s=time.perf_counter()-start,new_physics_substeps=0,new_native_solves=0,new_fixed_policy_rollouts=0,new_PPO_samples=0,new_PPO_updates=0,new_GPU_allocations=0,source_count=len(SOURCES),source_bytes=sum(x['bytes'] for x in SOURCES.values()),sources=list(SOURCES.values()))
 write(OUT/'run.json',run)
 print(json.dumps({'wall_s':run['elapsed_wall_s'],'calls':c['combined']['calls'],'positive_installs':c['combined']['final_positive_step_installs'],'stop_reason':c['combined']['stop_reason'],'GPU_scenarios':g['distinct_scenario_execution_count'],'outputs':str(OUT)}))
if __name__=='__main__':main()
