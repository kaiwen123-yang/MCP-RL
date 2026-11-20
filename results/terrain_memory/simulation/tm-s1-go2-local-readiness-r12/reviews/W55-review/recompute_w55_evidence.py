"""Reconcile W55 stage records and released evidence without executing physics."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,copy,hashlib,json,subprocess
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--originals',type=Path,required=True);p.add_argument('--reviewed-commit');a=p.parse_args()
W=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55'
phase=json.loads((W/'phase.json').read_text());index=json.loads((W/'index.json').read_text());rows=[]
checkout_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()
commit=a.reviewed_commit or checkout_commit
if a.reviewed_commit:
 changed=subprocess.check_output(['git','diff','--name-only',commit,'HEAD','--',str(W.relative_to(a.repo))],cwd=a.repo,text=True).strip()
 if changed:raise ValueError('W55 evidence differs from declared reviewed commit: '+changed)
arrays=json.loads((a.audit/'arrays-recomputed.json').read_text());ledger=json.loads((a.originals/'original-ledger.json').read_text());inventory=json.loads((a.audit/'release-inventory.json').read_text())
needed=set('simulation_s physical_s physical_samples max_x_m min_body_height_m completed_round_trips case_key qualification material legal_route stage_progress samples_500Hz timing error reason source_commit physical_started_unix_s physical_ended_unix_s started_unix_s ended_unix_s user_cpu_s system_cpu_s exit_code success stage_status wall_s first_completion_s final_qpos final_qvel hold max_joint_limit_excess_rad mujoco_warning_counts'.split())
for r0 in phase['cases']:
 r=copy.deepcopy(r0);record=a.repo/r['record'];dedicated=next(iter(W.rglob(r['attempt']+'.json')),None)
 runner=json.loads(record.read_text()) if record.exists() else {}
 result=json.loads(dedicated.read_text()) if dedicated else {}
 for source in [runner,result]:
  for k,v in source.items():
   if k not in needed:continue
   if k not in r or r[k] is None:r[k]=v
 if 'simulation_s' not in r and 'physical_s' in r:r['simulation_s']=r['physical_s']
 label=r['job'].split('-w55-')[-1].upper()
 batch=label.lower()+'-'+r['level'].lower()
 r['_audit']=dict(case=label,record_exists=record.exists(),dedicated_result_exists=dedicated is not None,
  actual_result_path=str(dedicated.relative_to(a.repo)) if dedicated else None,
  raw_array_independently_checked=batch in arrays['cases'],
  video_release_assets=[x['name'] for x in inventory['assets'] if x['name'].startswith('w55-'+batch+'-') and x['name'].endswith('.mp4')])
 if result:r['_audit']['reported_roundtrip_completion_times_s']=[x['time_s'] for x in result.get('transitions',[]) if x['phase'] in ['turn_home','complete']]
 rows.append(r)
finished=[x for x in rows if x.get('ended_unix_s') is not None and x['status']!='running']
running=[x for x in rows if x['status']=='running']
counts=dict(finished=len(finished),passed=sum(x['status']=='passed' for x in finished),nonpass=sum(x['status']!='passed' for x in finished),
 simulation_s=sum(x.get('simulation_s',0) for x in finished),physical_steps=sum(x.get('physical_samples',0) for x in finished),
 process_cpu_s=sum(x.get('user_cpu_s',0)+x.get('system_cpu_s',0) for x in finished),active_snapshot=len(running))
intervals=[(x['physical_started_unix_s'],x['physical_ended_unix_s'],x['attempt']) for x in finished]
overlaps=[]
for i,(s1,e1,n1) in enumerate(intervals):
 for s2,e2,n2 in intervals[i+1:]:
  if min(e1,e2)>max(s1,s2):overlaps.append(dict(a=n1,b=n2,wall_s=min(e1,e2)-max(s1,s2)))
times=sorted({x for s,e,n in intervals for x in [s,e]});hist={}
for s,e in zip(times[:-1],times[1:]):
 n=sum(s0<e and e0>s for s0,e0,_ in intervals);hist[n]=hist.get(n,0.)+e-s
full=dict(reviewed_commit=commit,snapshot_unix_s=phase['updated_unix_s'],cases=rows,counts=counts,
 physical_overlap=overlaps,wall_duration_by_concurrent_finished_physics=hist,
 physical_overlap_pairwise_total_s=sum(x['wall_s'] for x in overlaps),max_finished_physics_concurrency=max(hist))
(a.audit/'phase-recomputed.json').write_text(json.dumps(full,indent=2)+'\n')
matrix=[]
for r in rows:
 row={k:r.get(k) for k in ['attempt','level','status','reason','simulation_s','physical_samples','max_x_m','min_body_height_m','completed_round_trips','first_completion_s','hold','max_joint_limit_excess_rad','mujoco_warning_counts','wall_s','case_key','qualification','material','legal_route','started_unix_s','ended_unix_s','physical_started_unix_s','physical_ended_unix_s','user_cpu_s','system_cpu_s']}
 row['case']=r['_audit']['case'];row['evidence']=r['_audit'];row['live']=r.get('live')
 if row['ended_unix_s']:
  row['runner_wall_s']=row['ended_unix_s']-row['started_unix_s'];row['process_cpu_s']=r.get('user_cpu_s',0)+r.get('system_cpu_s',0);row['process_cpu_per_wall']=row['process_cpu_s']/row['runner_wall_s']
 matrix.append(row)
stages={lev:dict(completed=sum(r['level']==lev and r in finished for r in rows),passed=sum(r['level']==lev and r['status']=='passed' for r in rows),nonpass=sum(r['level']==lev and r in finished and r['status']!='passed' for r in rows),running_snapshot=sum(r['level']==lev and r['status']=='running' for r in rows),not_run=6-sum(r['level']==lev for r in rows)) for lev in ['E30','E60','E300','E600']}
sampled_resources={}
for batch in ['f-a-r55-e60','f-b-r55-e60','f-a-r55-e300','n-a-r54-e30']:
 samples=[json.loads(x) for x in (a.originals/batch/'job-snapshot/resources.jsonl').read_text().splitlines()]
 rates=[]
 for r0,r1 in zip(samples[:-1],samples[1:]):
  delta=r1['user_cpu_s']+r1['system_cpu_s']-r0['user_cpu_s']-r0['system_cpu_s']
  if r0['pids']==r1['pids'] and delta>=0:rates.append(delta/(r1['unix_s']-r0['unix_s']))
 sampled_resources[batch]=dict(samples=len(samples),max_pids=max(len(r['pids']) for r in samples),peak_RSS_GiB=max(r['rss_bytes'] for r in samples)/2**30,
  same_PID_intervals_CPUcore_p50_p95=np.percentile(rates,[50,95]).tolist(),
  caveat='Interval quantiles only compare identical observed PID sets; final parent-only counter is not an aggregate of exited children.')
summary=dict(reviewed_commit=commit,analysis_updated_utc=datetime.now(timezone.utc).isoformat(),phase_snapshot_utc=datetime.fromtimestamp(phase['updated_unix_s'],timezone.utc).isoformat(),
 counts=counts,stage_counts=stages,case_matrix=matrix,
 index_corrections={k:dict(published=index.get(src),recomputed=counts[k]) for k,src in [('finished','completed_original_attempts'),('simulation_s','completed_simulation_s'),('physical_steps','completed_physical_steps'),('process_cpu_s','completed_attempt_cpu_s')]},
 resource=dict(finished_physics_max_concurrency=max(hist),finished_physics_pairwise_overlap_wall_s=full['physical_overlap_pairwise_total_s'],published_overlap_wall_s=index['effective_physical_overlap_wall_s'],
 process_cpu_core_hours=counts['process_cpu_s']/3600,sampled_resources=sampled_resources,wall_deadline_enabled=phase['wall_deadline_enabled'],deadline=phase['deadline'],kill_on_conversation_end=phase['kill_on_conversation_end'],
 caveat='Per-attempt process CPU accounting and observed interval overlap; not complete machine utilization. Gaps do not establish idle time. Logical 2/20 ms deadline counters are performance diagnostics, not a background wall cutoff.'),
 selected_originals=dict(files=len(ledger['files']),batches=len({x['batch'] for x in ledger['files']}),uncompressed_bytes=sum(x['size'] for x in ledger['files']),http_range_bytes=sum(x['bytes_downloaded'] for x in ledger['archives']),all_members_zip_crc_verified=all(x['zip_crc32_verified'] for x in ledger['files']),all_members_published_sha_size_verified=all(x['source_sha_verified'] for x in ledger['files']),full_zip_sha_verified=False,verification_updated_utc=ledger.get('verification_updated_utc')),
 release=dict(queried_at_utc=inventory['query_time_utc'],asset_count=len(inventory['assets']),asset_bytes=sum(x['size'] for x in inventory['assets']),video_assets=[x['name'] for x in inventory['assets'] if x['name'].endswith('.mp4')]),
 array_recomputation=arrays,
 limitations=['B r55 E300 now has dedicated stage runner/result/analysis/resolved at reviewed commit e5ff1478. A direct Release query at the recorded time also finds its full video and 20 originals ZIPs. This audit read the JSON but did not download/recompute its raw arrays; its detailed hold metrics are execution-side analysis.',
 'N r56 has a dedicated result under w55/n/r56 and E30-actual-analysis.json; raw arrays and full video were not public in this release inventory.',
 'E600 rows are running at the repository snapshot, not an observation of current live processes; no final results available.',
 'E300 500 Hz trajectory and selected control/drift members were read. Its entire 5 kHz physical stream was not downloaded.',
 'This evidence sub-agent did not visually inspect any video; the parent reviewer performs and records the visual review.',
 'Stage counts combine versions and deterministic prefixes and are not independent-trial estimates of current reliability.',
 'All examined qualification cases are nominal; this does not establish all formal perturbation conditions or material-memory causal benefit.'])
(a.audit/'evidence-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(commit=commit,counts=counts,stages=stages,phase_snapshot_utc=summary['phase_snapshot_utc'],verified_original_members=len(ledger['files'])),indent=2))
