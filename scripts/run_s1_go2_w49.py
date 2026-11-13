#!/usr/bin/env python3
"""Evaluation-only W49 ladder. No training branch and no wall deadline."""
import argparse,copy,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from publish_s1_go2_w49 import compact,deliver,write,ROOT,SMALL,LARGE
LEVELS=['E30','E60','E300','E600']
def run(jobdir):
 job=json.loads((jobdir/'job.json').read_text());contract=jobdir/'execution-contract.json'
 state=dict(job=job['job_id'],wall_deadline_enabled=False,status='running',started_unix_s=time.time(),pid=os.getpid(),
  deadline_unix_s=None,new_PPO_samples=0,new_PPO_updates=0,samples=14737,updates=57,attempts=[],stages=[])
 path=SMALL/'w49'/(job['stream_id']+'.json')
 write(jobdir/'runtime.json',state);write(path,state)
 failed=None
 for level in LEVELS:
  if failed:
   state['stages'].append(dict(level=level,status='blocked',blocked_by=failed));continue
  attempt=job['job_id']+'-eval-PANN_PPO_S5-s'+str(job['seed'])+'-'+level+'-attempt01'
  command=[sys.executable,'-u','scripts/run_s1_go2_r12.py','--contract',str(contract),
   '--attempt',attempt,'--evaluation-start-unix',str(job['started_unix_s']),
   '--checkpoint',str(ROOT/job['checkpoint']),'--level',level,'--seed',str(job['seed']),
   '--workers','10','--build','S5-r37','--condition','nominal']
  row=dict(level=level,attempt=attempt,status='running',command=command,started_unix_s=time.time())
  state['attempts'].append(attempt);state['stages'].append(row)
  write(path,state);write(jobdir/'runtime.json',state)
  with (jobdir/(level+'.log')).open('w') as log:
   child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   row['pid']=child.pid;write(path,state);write(jobdir/'runtime.json',state)
   code=child.wait()
  resultpath=SMALL/(attempt+'.json')
  result=json.loads(resultpath.read_text()) if resultpath.exists() else None
  row.update(exit_code=code,ended_unix_s=time.time(),result=str(resultpath.relative_to(ROOT)) if result else None)
  if result:
   row.update(status=result['stage_status'],success=result['success'],reason=result['reason'],
    simulation_s=result['simulation_s'],wall_s=result['wall_s'],qualification=result.get('qualification'),
    legal_route=result.get('legal_route'),material_progress=result.get('stage_progress'))
  else:row.update(status='interrupted',success=False,reason='process_exited_without_completed_result')
  if not row['success']:failed=level
  write(path,state);write(jobdir/'runtime.json',state)
  from w49_run_summary import summarize_attempt
  summary_path=summarize_attempt(attempt)
  compact([summary_path,path,resultpath,SMALL/(attempt+'-resolved.json')] if result else [summary_path,path],f'Record W49 {job["stream_id"]} {level} actual outcome')
 state['status']=('technical_interruption' if any(x.get('status') in ('interrupted','technical_interruption') for x in state['stages']) else 'completed_with_failed_qualification') if failed else 'completed_all_qualified'
 state['ended_unix_s']=time.time();write(path,state);write(jobdir/'runtime.json',state)
 with (SMALL/'index.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  phase=json.loads((SMALL/'w49/phase.json').read_text())
  phase.setdefault('stream_results',{})[job['stream_id']]={'status':state['status'],'stages':state['stages']}
  if len(phase['stream_results'])==2:phase['status']='all_original_ladders_naturally_ended'
  write(SMALL/'w49/phase.json',phase)
  idx=json.loads((SMALL/'index.json').read_text());idx['w49']=phase;write(SMALL/'index.json',idx)

 compact([path,SMALL/'w49/phase.json'],f'Complete W49 {job["stream_id"]} fixed-policy ladder')
 # Render actual recorded tasks only; no physics integration.
 for attempt in state['attempts']:
  if (LARGE/attempt/'trajectory.npz').exists():
   with (jobdir/'render.log').open('a') as log:
    code=subprocess.run([sys.executable,'scripts/render_s1_go2_r12.py','--attempt',attempt],
      cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
   state.setdefault('render',[]).append(dict(attempt=attempt,exit_code=code))
 write(path,state)
 delivery=deliver(job['job_id'],[LARGE/name for name in state['attempts'] if (LARGE/name).exists()]+[jobdir])
 state['publication']=delivery;write(path,state);compact([path],f'Record W49 {job["stream_id"]} final delivery state')
def declare():
 gate=json.loads((SMALL/'w49/deployment.json').read_text())
 if not gate.get('ready_for_original'):raise ValueError('No bounded interface evidence for an original task')
 import torch
 now=time.time();stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now))
 base=json.loads((ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json').read_text())
 identity=json.loads((ROOT/'work/r11-frozen/S5-r37/identity.json').read_text())
 phase=json.loads((SMALL/'w49/phase.json').read_text())
 phase.update(status='original_evaluations_declared_not_started',original_evaluation_declared_unix_s=now,deadline_unix_s=None,
  wall_deadline_enabled=False,new_PPO_samples=0,new_PPO_updates=0,binary_sha256=identity['binary_sha256'],jobs=[])
 for suffix,seed,ep,cpus in [('a',1,428,'0-9'),('b',2,417,'10-19')]:
  name=f'job-{stamp}-w49-{suffix}';stream='w49-'+suffix;directory=LARGE/'jobs'/name
  directory.mkdir(parents=True,exist_ok=False)
  parent=LARGE/f'training/w44-{suffix}/PANN_PPO_S5-seed{seed}'
  checkpoint=parent/'checkpoint-014737.pt';resume=parent/f'resume-state-ep{ep:05d}.pt'
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
  complete=torch.load(resume,map_location='cpu',weights_only=False)
  assert (complete['samples'],complete['updates'],complete['episode'],complete['seed'])==(14737,57,ep,seed)
  assert not complete['buffer'] and not saved['buffer'] and complete['excluded_old_version_transitions']==saved['excluded_old_version_transitions']==145
  for role in ['actor','critic']:
   assert all(torch.equal(value,complete[role][key]) for key,value in saved[role].items())
  assert saved['normalization']==complete['normalization'] and saved['fork_identity']==complete['fork_identity'] and saved['phase_lineage']==complete['phase_lineage']
  assert all(k in complete for k in ['optimizer','torch_rng','numpy_rng','cuda_rng'])
  assert (saved['samples'],saved['updates'],saved['seed'])==(14737,57,seed)
  def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
  job=dict(job_id=name,stream_id=stream,phase_id=name,group='PANN_PPO_S5',seed=seed,
    mode='evaluation_only',started_unix_s=now,deadline_unix_s=None,workers=10,affinity=cpus,
    checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest(checkpoint),
    normalization_sha256=hashlib.sha256(json.dumps(saved['normalization'],sort_keys=True).encode()).hexdigest(),
    actor_evaluation='deterministic mean, frozen own update57 and saved normalization',
    resume_checkpoint=str(resume.relative_to(ROOT)),resume_sha256=digest(resume),
    parent_stream=saved['stream_id'],parent_phase=saved['phase_id'],parent_episode=ep,
    samples=14737,updates=57,tail=0,excluded=145,new_PPO_samples=0,new_PPO_updates=0,
    input_reference_branch='constant',controller_build='S5-r37',controller_binary_sha256=identity['binary_sha256'],
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    resume_content_verified=['actor','critic','optimizer','normalization','RNG','fork_identity','phase_lineage'],
    evaluation_restore='Original deterministic evaluation semantics: checkpoint actor/critic/normalization and lineage loaded; complete optimizer/RNG resume preserved and verified, no PPO step or mid-plant restoration.',
    wall_deadline_enabled=False,
    resume_rule='Live PID may resume; exited process without full simulation state remains interrupted. A new explicit attempt starts from original initial state, never concatenated.')
  c=copy.deepcopy(base);c.update(current_task='W49_FIXED_POLICY_STABILITY',protocol_revision='W49_fixed_policy_r37_K_original_ladder')
  c['reuse'].update(active_controller_build='S5-r37',controller_artifact='S5-r37',selected_mechanism='K',input_reference_branch='constant')
  c['learning'].update(phase_id=name,stream_id=stream,phase_initial_transitions=14737,phase_initial_updates=57,
    target_cumulative_transitions_per_stream=14737,target_cumulative_updates_per_stream=57,
    additional_transitions_from_reviewed_checkpoint=0,additional_updates_from_reviewed_checkpoint=0,
    training_enabled=False,active_streams=[],evaluation_only=True)
  c['execution']['background_job'].update(wall_deadline_enabled=False,default_maximum_wall_s=None)
  c['resources'].update(mpc_workers=10,torch_threads=1,torch_device='cuda')
  c['control_and_evaluation']['post_training_evaluation'].update(maximum_wall_s=None,wall_deadline_enabled=False)
  c['w49_fixed_policy']=job
  write(directory/'job.json',job);write(directory/'execution-contract.json',c)
  write(SMALL/'w49'/'jobs'/name/'job.json',job);write(SMALL/'w49'/'jobs'/name/'execution-contract.json',c)
  phase['jobs'].append(dict(job_id=name,stream_id=stream,directory=str(directory.relative_to(ROOT)),
    session='s1-'+stream,affinity=cpus,checkpoint_sha256=job['checkpoint_sha256']))
 write(SMALL/'w49'/'phase.json',phase);write(LARGE/'w49-clock.json',dict(started_unix_s=now,deadline_unix_s=None))
 with (SMALL/'index.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);idx=json.loads((SMALL/'index.json').read_text());idx['w49']=phase;write(SMALL/'index.json',idx)
 print(json.dumps(phase,indent=2))
def start():
 import shlex
 phase=json.loads((SMALL/'w49/phase.json').read_text())
 for row in phase['jobs']:
  jobdir=ROOT/row['directory'];command=[sys.executable,'-u','scripts/run_s1_go2_w49.py','--job-dir',str(jobdir)]
  shell='source scripts/autodl_d57_env.sh && source scripts/autodl_network.sh academic && exec taskset -c '+row['affinity']+' '+shlex.join(command)+' >> '+shlex.quote(str(jobdir/'runner.log'))+' 2>&1'
  subprocess.run(['tmux','new-session','-d','-s',row['session'],'-c',str(ROOT),shell],check=True)
  subprocess.run(['tmux','set-option','-t',row['session'],'remain-on-exit','on'],check=True)
 phase.update(status='original_evaluations_running',launched_unix_s=time.time(),wall_deadline_enabled=False)
 write(SMALL/'w49/phase.json',phase)
def main():
 p=argparse.ArgumentParser();p.add_argument('--declare',action='store_true');p.add_argument('--start',action='store_true');p.add_argument('--job-dir',type=Path);a=p.parse_args()
 if a.declare:declare()
 elif a.start:start()
 else:run(a.job_dir)
if __name__=='__main__':main()
