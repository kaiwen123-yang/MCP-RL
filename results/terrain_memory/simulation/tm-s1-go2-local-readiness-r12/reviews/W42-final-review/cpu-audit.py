#!/usr/bin/env python3
"""Recompute W42 CPU results and selected published original diagnostics.
No simulation, training, controller modification, or GPU execution.
"""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
BASE=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
BASELINE='c4b3f9adc698b15de5c94ef952e2884f3abfec62'
JOBS={'w42-a':'job-20261108T154719Z-w42-a','w42-b':'job-20261108T154727Z-w42-b'}
READ={}
def read(path):
 path=Path(path); raw=path.read_bytes()
 READ[str(path.relative_to(ROOT))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 if path.suffix=='.jsonl':return [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
 return json.loads(raw)
def stats(v):
 a=np.array(list(v),dtype=float)
 return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),'mean':float(a.mean()),'max':float(a.max())} if len(a) else {'n':0}
def fam(s):return str(s).split(':')[0]
def finite(o):
 if isinstance(o,dict):return all(finite(x) for x in o.values())
 if isinstance(o,list):return all(finite(x) for x in o)
 return not isinstance(o,float) or math.isfinite(o)
def summarized_episode(r):
 return {k:r.get(k) for k in ['attempt','reason','error','success','stage_status','simulation_s','wall_s','max_x_m','final_qpos','completed_round_trips','legal_route','stage_progress','policy','timing','mujoco_warning_counts']}
def compact(stream,job):
 d=BASE/'jobs'/job
 for path in sorted(d.rglob('*')):
  if path.suffix in {'.json','.jsonl'}:read(path)
 ep=[read(p) for p in sorted((d/'training').glob('episode-*.json'))]
 results=[read(BASE/(r['attempt']+'.json')) for r in ep]
 for r in ep:
  p=BASE/(r['attempt']+'-resolved.json')
  if p.exists():read(p)
 jobdata=read(d/'job.json');p=read(d/'training/progress.json');l=read(d/'training/learning.json');e=read(d/'evaluation.json')
 evalattempt=e['attempts'][0]
 if not isinstance(evalattempt,str):evalattempt=evalattempt.get('attempt')
 er=read(BASE/(evalattempt+'.json'))
 resolved=read(BASE/(evalattempt+'-resolved.json'))
 transitions=[x for f in sorted((d/'numeric-evidence/training').glob('*-transitions.jsonl')) for x in read(f)]
 values=[x for f in sorted((d/'numeric-evidence/training').glob('*-pann-values.jsonl')) for x in read(f)]
 serials=[x['transition_serial'] for x in transitions]
 pann_keys=[k for k in (values[0] if values else {}) if 'pann' in k.lower() or 'recover' in k.lower()]
 nonzero={k:sum(bool(np.any(np.asarray(x[k],dtype=float)!=0)) for x in values if k in x and isinstance(x[k],list)) for k in pann_keys}
 warnings=[read(f) for f in sorted((d/'episodes').glob('*/mujoco-warnings.json'))]
 warningtotals=collections.Counter()
 for w in warnings:
  for k,v in w.get('plant',{}).items():warningtotals[k]+=int(v.get('number',0))
 eventmanifests=[read(f) for f in sorted((d/'episodes').glob('*/r29-event-manifest.json'))]
 return {'job':job,'stream':stream,'controller':p['controller_artifact'],'parent':jobdata['resume_checkpoint'],
 'initial_samples':jobdata['initial_transitions'],'final_samples':p['samples'],'initial_updates':jobdata['initial_updates'],'final_updates':p['updates'],
 'tail':p['unupdated_fragment'],'excluded':p['excluded_old_version_transitions'],'last_episode':p['episodes_started'],'checkpoint':p['checkpoint'],
 'training_episodes':len(ep),'training_reason_counts':dict(collections.Counter(fam(x['reason']) for x in results)),
 'training_status_counts':dict(collections.Counter(x['stage_status'] for x in results)),
 'training_errors':[x['error'] for x in results if x.get('error')],
 'training_max_x_m':stats(x['max_x_m'] for x in results),'training_final_x_m':stats(x['final_qpos'][0] for x in results),
 'training_simulation_s':stats(x['simulation_s'] for x in results),'training_wall_s':sum(x['wall_s'] for x in results),
 'training_material_loaded':sum(bool(x['stage_progress']['actual_material_loaded']) for x in results),
 'training_legal_outbound':sum(x['legal_route']['legal_outbound_traversals'] for x in results),
 'training_legal_return':sum(x['legal_route']['legal_return_traversals'] for x in results),
 'training_legal_roundtrips':sum(x['legal_route']['completed_roundtrips'] for x in results),
 'new_transitions':len(transitions),'unique_serials':len(set(serials)),'serial_range':[min(serials),max(serials)],
 'serials_complete':sorted(serials)==list(range(jobdata['initial_transitions']+1,p['samples']+1)),
 'terminated_rows':sum(x['terminated'] for x in transitions),'truncated_rows':sum(x['truncated'] for x in transitions),
 'behavior_update_counts':dict(collections.Counter(x['behavior_updates'] for x in transitions)),
 'phase_counts':dict(collections.Counter(x.get('phase') for x in transitions)),
 'cap_m_s':stats(x['cap_m_s'] for x in transitions),'reward':stats(x['reward'] for x in transitions),
 'total_reward':sum(x['reward'] for x in transitions),'values_rows':len(values),'pann_feature_fields':pann_keys,'pann_nonzero_rows':nonzero,
 'all_training_values_finite':finite(transitions) and finite(values),'new_update_count':len(l['updates']),
 'new_updates':l['updates'],'warning_episodes':len(warnings),'plant_warning_totals':dict(warningtotals),
 'service_warning_lines':dict(collections.Counter(line for w in warnings for line in w.get('service_warning_lines',[]))),
 'command_nonprogress_events':[x for m in eventmanifests for x in m['events'] if x['kind']=='command_nonprogress_2s'],
 'evaluation':summarized_episode(er),'evaluation_checkpoint':e['checkpoint'],'evaluation_checkpoint_transitions':e['checkpoint_transitions'],
 'evaluation_checkpoint_updates':e['checkpoint_updates'],'evaluation_stages':e['stages'],
 'deadline':jobdata['deadline_unix_s'],'workers':jobdata['mpc_workers'],
 'episodes':[summarized_episode(r) for r in results]}
def gzrows(path):return [json.loads(x) for x in gzip.open(path,'rt') if x.strip()]
def raw_analysis(rawroot,stream):
 d=rawroot/stream
 plans=gzrows(next(d.glob('*planner-records*')));calls=gzrows(next(d.glob('*planner-calls*')));ex=gzrows(next(d.glob('*action-execution*')))
 with gzip.open(next(d.glob('*reference-packets*')),'rt') as f:ref=json.load(f)
 with np.load(next(d.glob('*trajectory.npz'))) as z:
  t=z['time']; moving=z['phase']==1; first=int(np.flatnonzero(moving)[0]); post=t>=2.002-1e-8
  trajectory={'rows':len(t),'terminal_time_s':float(t[-1]),'first_outbound_s':float(t[first]),'x_max_m':float(z['qpos'][:,0].max()),'x_min_m':float(z['qpos'][:,0].min()),
  'final_xyz_m':z['qpos'][-1,:3].tolist(),'outbound_net_dx_m':float(z['qpos'][-1,0]-z['qpos'][first,0]),
  'outbound_mean_command_vx':float(z['command_velocity'][post,0].mean()),'outbound_mean_actual_vx':float(z['qvel'][post,0].mean()),
  'outbound_all_four_contact_fraction':float(z['contact'][post].all(axis=1).mean()),
  'outbound_max_abs_q_target_change_from_switch_rad':float(np.abs(z['q_target'][post]-z['q_target'][first]).max()),
  'outbound_final_q_target_change_from_switch_rad':(z['q_target'][-1]-z['q_target'][first]).tolist(),
  'max_nonfoot_contact':int(z['nonfoot_ground_contacts'].max()),'max_nonfoot_force_N':float(np.abs(z['nonfoot_ground_force']).max()),
  'max_abs_roll_pitch_rad':np.abs(z['rpy'][:,:2]).max(axis=0).tolist(),'array_finite':all(np.isfinite(z[k]).all() for k in z.files)}
 after=[p for p in plans if p['origin']>=trajectory['first_outbound_s']-1e-8];physical=[p for p in plans if p['origin']>0];startup=[p for p in plans if p['origin']==0]
 def row_summary(p):
  return {**{k:p[k] for k in ['plan_id','origin','nominal_cost','accepted_cost','cost_improvement','proxy_expected_decrease','cost_surprise','action_step_alpha','regularization_used','regularization_after','no_improvement_reason']},
  'K_max_two_saved_nodes':max(abs(x) for n in p['nodes'] for x in n['K']),
  'candidate_projection_count':sum(c['projection_count'] for c in p['candidates']),
  'candidate_costs':[c['cost'] for c in p['candidates']]}
 return {'trajectory':trajectory,'planner_rows':len(plans),'planner_call_rows':len(calls),'return_reasons':dict(collections.Counter(x['return_reason'] for x in calls)),
 'all_policy_updated':all(x['policy_updated'] for x in calls),'backward_attempt_counts':dict(collections.Counter(x['backward_attempts'] for x in calls)),
 'all_metric_valid':all(x['metric_valid'] for x in calls),'max_recorded_mujoco_warning':max(v for c in calls for v in c['mujoco_warning_counts_native_cumulative_then_soft_call']),
 'startup_calls':len(startup),'physical_calls':len(physical),'physical_calls_at_max_regularization':sum(p['regularization_used']>=1e6 for p in physical),
 'outbound_calls':len(after),'outbound_cost_improvement':stats(p['cost_improvement'] for p in after),'outbound_proxy_expected_decrease':stats(p['proxy_expected_decrease'] for p in after),
 'outbound_cost_surprise':stats(p['cost_surprise'] for p in after),'outbound_K_max_saved_nodes':stats(max(abs(x) for n in p['nodes'] for x in n['K']) for p in after),
 'outbound_alpha':dict(collections.Counter(p['action_step_alpha'] for p in after)),
 'candidate_count':sum(len(p['candidates']) for p in plans),'candidate_failed':sum(c['failed'] for p in plans for c in p['candidates']),
 'candidate_projection_count':sum(c['projection_count'] for p in plans for c in p['candidates']),
 'candidate_projection_failed':sum(c['projection_failed_count'] for p in plans for c in p['candidates']),
 'candidate_actuator_restoration_count':sum(c['actuator_restoration_count'] for p in plans for c in p['candidates']),
 'no_improvement_reason_counts':dict(collections.Counter(p['no_improvement_reason'] for p in plans)),
 'startup_plans':[row_summary(p) for p in startup],'switch_plans':[row_summary(p) for p in plans if 1.97<=p['origin']<=2.05],
 'action_execution_rows':len(ex),'applied_hold_queries':sum(x.get('applied_action_id') is not None for x in ex),
 'actually_integrated_hold_queries':sum(x.get('applied_action_id') is not None and x['time_s']<trajectory['terminal_time_s']-1e-8 for x in ex),
 'action_query_kinds':dict(collections.Counter(x['query_kind'] for x in ex)),
 'reference_packets':len(ref['packets']),'reference_phase_counts':dict(collections.Counter(x['phase'] for x in ref['packets'])),
 'scope':'Complete E30 planner calls/records, trajectory, reference packets and execution links; K statistics use the two logged nodes, not whole horizon. physical_calls includes the terminal replan. applied_hold_queries includes a terminal query without a later physics integration. No new physics.'}
def startup_analysis(rawroot):
 output={}
 for stream in JOBS:
  d=rawroot/stream; plans=gzrows(next(d.glob('*planner-records*')))
  selected=[]
  for p in plans:
   if p['plan_id']>22 and not 119<=p['plan_id']<=123:continue
   selected.append({**{k:p[k] for k in ['origin','plan_id','nominal_cost','accepted_cost','proxy_expected_decrease','cost_surprise','cost_improvement','action_step_alpha','regularization_before','regularization_used','regularization_after','no_improvement_reason']},
    'projection_total':sum(c['projection_count'] for c in p['candidates']),
    'max_abs_K_saved_nodes':max(abs(x) for n in p['nodes'] for x in n['K']),
    'candidate_costs':[c['cost'] for c in p['candidates']]})
  points=[]
  with np.load(next(d.glob('*trajectory.npz'))) as z:
   for t in [.02,1.98,2.0,2.002,2.02,2.04,5.0,29.9]:
    i=int(np.argmin(np.abs(z['time']-t)))
    points.append({'time':float(z['time'][i]),'phase':int(z['phase'][i]),'base':z['qpos'][i,:3].tolist(),
     'command_velocity':z['command_velocity'][i].tolist(),'body_target':z['body_target'][i].tolist(),
     'contacts':z['contact'][i].tolist(),'q_target':z['q_target'][i].tolist()})
  output[stream]={'selected_plans':selected,'trajectory_points':points,'node_fields':list(plans[0]['nodes'][0]),
   'limitation':'Two saved node slices contain K, not raw Q_u/k/dV. proxy_expected_decrease=1e-16 on alpha0 is floored accepted-step value, not proof that the unscaled gradient is zero. Later full-horizon matrices are analyzed separately.'}
 (HERE/'cpu-planner-startup.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
def matrix_analysis(rawroot):
 output=[]
 for stream in JOBS:
  for path in sorted((rawroot/stream).glob('*accepted-reference-gain-H-plan*.npz')):
   with np.load(path) as z:
    # Last node is terminal and has no control. All 175 controlled nodes are read.
    H=z['H'][:-1];Qu=z['base_Qu'][:-1];K=z['K'][:-1];Qux=np.swapaxes(z['base_Qxu'][:-1],1,2)
    reg=float(z['header'][8]);k=np.linalg.solve(H,-Qu[...,None])[...,0]
    diag=np.diagonal(H,axis1=1,axis2=2)-reg
    output.append({'stream':stream,'plan':int(z['header'][1]),'regularization':reg,
     'all_active_zero':bool(np.all(z['base_active'][:-1]==0)),
     'max_abs_base_Qu':float(np.abs(Qu).max()),'base_Qu_l2':float(np.linalg.norm(Qu)),
     'max_abs_reconstructed_k':float(np.abs(k).max()),'max_abs_K':float(np.abs(K).max()),
     'H_diag_minus_lambda':{'min':float(diag.min()),'max':float(diag.max())},
     'max_abs_HK_plus_Qux':float(np.abs(H@K+Qux).max()),
     'max_accepted_minus_reference_action':float(np.abs(z['accepted_actions']-z['reference_actions']).max()),
     'first_node_k':k[0].tolist(),'accepted_reference_first_float32_equal':bool(np.array_equal(z['accepted_actions'][0].astype('float32'),z['reference_actions'][0].astype('float32'))),
     'base_dx_accepted_horizon_m':float(z['accepted_states'][-1,0]-z['accepted_states'][0,0])})
 (HERE/'cpu-full-plan-matrices.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
def plot(rawroot,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
 for stream,color in [('w42-a','#2369bd'),('w42-b','#cb5a35')]:
  d=rawroot/stream; p=gzrows(next(d.glob('*planner-records*')))
  s=[x for x in p if x['origin']==0];q=[x for x in p if x['origin']>0]
  axs[0,0].semilogy([x['plan_id'] for x in s],[x['regularization_after'] for x in s],'.-',label=stream,color=color)
  axs[0,1].semilogy([x['origin'] for x in q],[x['regularization_used'] for x in q],label=stream,color=color)
  with np.load(next(d.glob('*trajectory.npz'))) as z:
   axs[1,0].plot(z['time'],z['qpos'][:,0]*1000,label=stream,color=color)
   axs[1,1].plot(z['time'],z['command_velocity'][:,0],label=stream+' command',color=color)
   axs[1,1].plot(z['time'],z['qvel'][:,0],'--',label=stream+' actual',color=color,alpha=.7)
 axs[0,0].set(title='Stand preplanning at t = 0: damping reaches maximum',xlabel='Startup planning iteration',ylabel='Regularization after iteration')
 axs[0,1].set(title='All physical replans remain at maximum damping',xlabel='Simulation time (s)',ylabel='Regularization used',ylim=(1e-6,2e6))
 axs[1,0].set(title='Base remains near the starting point',xlabel='Simulation time (s)',ylabel='Base x (mm)')
 axs[1,1].set(title='Outbound command exists; realized velocity is near zero',xlabel='Simulation time (s)',ylabel='Velocity x (m/s)')
 for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
 for ax in [axs[0,1],axs[1,0],axs[1,1]]:ax.axvline(2.002,color='#555',ls=':',lw=1)
 fig.savefig(HERE/'cpu-damping-and-motion.png',dpi=170);plt.close(fig)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw-root',type=Path,required=True);a=ap.parse_args()
 output={'source_commit':BASELINE,'method':'Complete current-phase compact records plus selected complete original E30 members. No inherited session_records counted as new episodes.','streams':{}}
 for s,j in JOBS.items():output['streams'][s]={'compact':compact(s,j),'raw':raw_analysis(a.raw_root,s)}
 ss=list(output['streams'].values());output['total']={'training_episodes':sum(s['compact']['training_episodes'] for s in ss),'new_transitions':sum(s['compact']['new_transitions'] for s in ss),'new_updates':sum(s['compact']['new_update_count'] for s in ss),'training_errors':sum(len(s['compact']['training_errors']) for s in ss),'planner_calls_e30':sum(s['raw']['planner_call_rows'] for s in ss),'physical_calls_at_max_regularization':sum(s['raw']['physical_calls_at_max_regularization'] for s in ss),'warning_reports':sum(s['compact']['warning_episodes'] for s in ss)}
 output['read_coverage']={'files':len(READ),'bytes':sum(x['bytes'] for x in READ.values()),'inventory':READ}
 (HERE/'cpu-summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');startup_analysis(a.raw_root);matrix_analysis(a.raw_root);plot(a.raw_root,output)
 print(json.dumps(output['total']));print('coverage',len(READ),sum(x['bytes'] for x in READ.values()))
if __name__=='__main__':main()
