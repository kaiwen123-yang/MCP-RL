"""Independent ledger/paired arithmetic from committed per-attempt JSON, no experiments."""
from pathlib import Path
import argparse,json,collections,itertools,gzip,csv,numpy as np
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,required=True);P.add_argument('--output',type=Path,required=True);a=P.parse_args();R=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12';p=R/'w56';phase=json.loads((p/'phase.json').read_text());checks=[];records=[]
for row in phase['cases']:
 d={k:row.get(k)for k in ['label','level','window','status','attempt','source','stage_record','case_key','simulation_s','physical_samples','first_completion_s','max_x_m','min_body_height_m','legal_route','physical_started_unix_s','physical_ended_unix_s','last_mpc_write_unix_s','last_confirmed_mpc_s']}
 if row.get('source'):
  rr=json.loads((a.repo/row['source']).read_text());d['source_status']=rr['stage_status'];d['evaluator_sha256']=rr['qualification']['evaluator_sha256'];d['workers']=rr['qualification']['solver_workers'];d['case_identity']=rr['case_identity'];d['wall_s']=rr['wall_s'];d['source_material']={k:rr['material'].get(k)for k in ['truth','domain_exit','max_abs_principal_strain','integral_slip_sq_s','wall_s','loaded_ever_cells']};d['timing']=rr['timing'];d['observer']=rr['policy']['observer'];d['source_reason']=rr['reason'];d['source_success']=rr['success'];d['source_original_success']=rr.get('original_mission_success');d['first_completion_s']=rr['first_completion_s'];d['last_30s_hold']=rr.get('last_30s_hold');d['final_stable_s']=rr.get('final_stable_s')
  d['predecessor_attempt']=rr['qualification']['predecessor_attempt']
  for k in ['attempt','case_key','simulation_s','physical_samples','first_completion_s','max_x_m','min_body_height_m']:
   checks.append({'source':row['source'],'field':k,'ok':row.get(k)==rr.get(k)})
  checks.append({'source':row['source'],'field':'status','ok':(row['status']=='passed')==rr['success']})
 if row.get('stage_record'):
  sr=json.loads((a.repo/row['stage_record']).read_text());d['process_cpu_s']=sr.get('user_cpu_s',0)+sr.get('system_cpu_s',0);d['process_wall_s']=sr['ended_unix_s']-sr['started_unix_s']if sr.get('ended_unix_s')else None;d['average_process_cores']=d['process_cpu_s']/d['process_wall_s']if d['process_wall_s']else None;d['command']=sr.get('command');d['background']={k:sr.get(k)for k in ['deadline','wall_deadline_enabled','kill_on_conversation_end']}
 records.append(d)
by_attempt={r['attempt']:r for r in records if r.get('source')};prior_level={'E60':'E30','E300':'E60','E600':'E300'};predecessors=[]
for r in by_attempt.values():
 prev=by_attempt.get(r['predecessor_attempt'])
 okay=(r['predecessor_attempt'] is None if r['level']=='E30' else bool(prev and prev['status']=='passed' and prev['level']==prior_level[r['level']] and prev['label']==r['label'] and prev['case_identity']==r['case_identity'] and prev['workers']==r['workers']))
 predecessors.append(dict(attempt=r['attempt'],level=r['level'],predecessor_attempt=r['predecessor_attempt'],same_identity_prior_level_passed=okay))
counts={}
for w in ['w55','w56']:
 xs=[r for r in records if r['window']==w];counts[w]=dict(status_counts=dict(collections.Counter(r['status']for r in xs)),unique_ended_attempts=len({r['attempt']for r in xs if r['attempt']}),full_result_simulation_s=sum(r['simulation_s']or 0 for r in xs),full_result_physics_steps=sum(r['physical_samples']or 0 for r in xs),process_cpu_s=sum(r.get('process_cpu_s',0)for r in xs),full_result_legal_roundtrips=sum((r['legal_route']or {}).get('completed_roundtrips',0)for r in xs))
# Recompute physical-activity intervals, truncating interrupted jobs at last durable MPC write.
intervals=[]
for r in records:
 begin=r.get('physical_started_unix_s');end=r.get('physical_ended_unix_s')or r.get('last_mpc_write_unix_s')
 if begin and end:intervals.append((begin,end,r['attempt'],r['window']))
def overlaps(scope):
 xs=[x for x in intervals if scope=='all' or x[3]==scope];ev=sorted([(x[0],1)for x in xs]+[(x[1],-1)for x in xs]);prev=None;n=0;dur=collections.defaultdict(float);mx=0
 for t,v in ev:
  if prev is not None:dur[n]+=t-prev
  n+=v;mx=max(mx,n);prev=t
 pairs=[]
 for x,y in itertools.combinations(xs,2):
  b=max(x[0],y[0]);e=min(x[1],y[1]);
  if e>b:pairs.append(dict(attempts=[x[2],y[2]],begin_unix_s=b,end_unix_s=e,wall_s=e-b))
 return dict(max_simultaneous_physics=mx,active_wall_s=sum(v for k,v in dur.items()if k),dual_physics_wall_s=sum(v for k,v in dur.items()if k>=2),pairs=pairs)
identity=[]
for label in sorted({x['label']for x in records}):
 xs=[r for r in records if r['label']==label and r.get('case_identity')];identical=all(r['case_identity']==xs[0]['case_identity']for r in xs);identity.append(dict(label=label,stages=[x['level']for x in xs],full_identity_identical=identical,case_keys=sorted({x['case_key']for x in xs})))
pairs=[]
for f in sorted((p/'f').glob('F-*/*-paired.json')):
 j=json.loads(f.read_text());o=j['baseline'];n=j['new'];r=dict(path=str(f.relative_to(a.repo)),case=j['case'],stage=j['stage'],intervention=j['intervention'],same_actor=o['checkpoint_sha256']==n['checkpoint_sha256'],same_normalizer=o['normalizer_sha256']==n['normalizer_sha256'],same_workers=o['workers']==n['workers'])
 for k in ['slip_integral','peak_material_strain']:
  r[k]={'baseline':o[k],'new':n[k],'delta':n[k]-o[k],'relative_change_pct':100*(n[k]/o[k]-1)};checks.append({'source':str(f.relative_to(a.repo)),'field':k+'_delta','ok':abs(r[k]['delta']-j['difference_new_minus_baseline'][k])<1e-14})
 for side in ['baseline','new']:
  x=j[side];r[side+'_max_roll_pitch_deg']=max(max(v['max_abs_roll_pitch_rad'])for v in x['by_phase'].values())*180/np.pi;r[side+'_actor']=x['actor'];r[side+'_completion_s']=next((x['time_s']for x in x['phase_transitions']if x['phase']=='complete'),None)
 comp=j.get('r58_material_backend_comparison')
 if comp and comp.get('status')=='paired_originals':
  r['r58_r60_physics_wall']={x:comp[x+'_timing']['physical_wall_s']for x in ['r58','r60']};r['r60_physics_speedup']=r['r58_r60_physics_wall']['r58']/r['r58_r60_physics_wall']['r60']
 pairs.append(r)
assets={};receipt_checks=[]
for folder in sorted((p/'delivery').iterdir()):
 if not (folder/'delivery.json').exists():continue
 d=json.loads((folder/'delivery.json').read_text());m={x['name']:x for x in json.loads((folder/'manifest.json').read_text())}
 for x in d['assets']:
  receipt_checks.append(dict(name=x['name'],size_agrees=x['size']==m[x['name']]['size'],digest_agrees=x['remote_digest']=='sha256:'+m[x['name']]['sha256'],state=x['remote_state']));assets[x['name']]=x
expo=[]
for f in sorted((p/'exposure').glob('*.json')):
 j=json.loads(f.read_text());q=sorted(set(['outside','never_loaded','same_lap_same_foot','same_lap_cross_foot','earlier_lap_same_foot','earlier_lap_cross_foot']));groups=collections.defaultdict(list)
 with gzip.open(f.with_name(f.stem+'-query-cells.csv.gz'),'rt')as stream:
  for r in csv.DictReader(stream):groups[r['same_world_cell_history']].append(r)
 g={}
 for k,rs in groups.items():
  ages=[float(x['previous_event_age_s'])for x in rs if x['previous_event_age_s']];qs=[abs(float(x['observer_q11']))for x in rs if x['observer_q11']];over=[float(x['exact_query_next_touch_overlap_m2'])for x in rs];g[k]=dict(rows=len(rs),foot_queries=len({(x['time_s'],x['leg'])for x in rs}),exact_next_touch_overlap_area_m2=sum(over),prior_event_age_s_percentiles=np.percentile(ages,[0,50,95,100]).tolist()if ages else None,max_abs_observer_q11=max(qs)if qs else None)
 expo.append(dict(source=str(f.relative_to(a.repo)),summary=j,recomputed_grouped_cells=g,counts_agree=all(j['foot_queries_touching_each_history_type'].get(k)==x['foot_queries']for k,x in g.items()),area_agrees=all(abs(j['exact_next_touch_overlap_area_m2_by_history'].get(k,0)-x['exact_next_touch_overlap_area_m2'])<1e-10 for k,x in g.items())))
e600_intervals={}
for rev in ['r55','r60']:
 xs=[r for r in records if r['label'].endswith(rev)and r['level']=='E600']
 e600_intervals[rev]=dict(cases=[dict(label=r['label'],attempt=r['attempt'],physical_start_unix_s=r['physical_started_unix_s'],physical_end_unix_s=r['physical_ended_unix_s'],physical_elapsed_wall_s=r['physical_ended_unix_s']-r['physical_started_unix_s'],process_wall_s=r['process_wall_s'],process_cpu_s=r['process_cpu_s'],mean_process_cores=r['average_process_cores'])for r in xs],dual_physics_wall_s=max(0,min(r['physical_ended_unix_s']for r in xs)-max(r['physical_started_unix_s']for r in xs)))
out=dict(source_commit='3590646bc15a59903c9561381368d985963dfe9e',scope='Committed original result/runner/phase/paired/exposure records; raw E600 arrays audited separately',counts=counts,attempts=records,identity_checks=identity,predecessor_checks=predecessors,source_checks=checks,pair_arithmetic=pairs,overlap={w:overlaps(w)for w in ['all','w55','w56']},e600_actual_intervals=e600_intervals,two_physics_plus_material_measurements={k:phase[k]for k in ['two_physics_plus_material_benchmark_overlap_s','direct_CPU_GPU_measurement_with_two_original_robots_s']},delivery=dict(unique_receipt_assets=len(assets),bytes=sum(x['size']for x in assets.values()),video_count=sum(k.endswith('.mp4')for k in assets),manifest_checks=receipt_checks),exposure=expo)
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(counts=counts,all_source_checks=all(x['ok']for x in checks),identity=identity,overlap={k:{kk:vv for kk,vv in v.items()if kk!='pairs'}for k,v in out['overlap'].items()},delivery={k:v for k,v in out['delivery'].items()if k!='manifest_checks'},exposure=[{k:x[k]for k in ['source','counts_agree','area_agrees']}for x in expo]),indent=2))
