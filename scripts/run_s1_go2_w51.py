#!/usr/bin/env python3
"""W51 original-path paired D8, finite evaluation ladders, and direct delivery."""
import argparse,copy,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
from publish_s1_go2_w51 import ROOT,SMALL,LARGE,write,compact,deliver
W=SMALL/'w51'
def declare():
 import torch
 now=time.time();stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now))
 phase=dict(status='declared_not_started',started_unix_s=now,deadline=None,wall_deadline_enabled=False,
  plan_commit='354b14bf5313332ae167f02c84ce6b68ef6bae90',new_PPO_samples=0,new_PPO_updates=0,donor=0,
  original_evaluations_started=0,qualification_usable=False,jobs=[],selection=None,
  resources=dict(cpu_quota=25,D0=10,D1=10,C=2,system_other_projects=3,math_threads=1,build_max_jobs=6))
 base=json.loads((ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json').read_text())
 parent=LARGE/'training/w44-a/PANN_PPO_S5-seed1';checkpoint=parent/'checkpoint-014737.pt';resume=parent/'resume-state-ep00428.pt'
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 complete=torch.load(resume,map_location='cpu',weights_only=False)
 # Identity checks of the exact authorized scientific inputs, not a test suite.
 assert (saved['samples'],saved['updates'],saved['seed'])==(14737,57,1)
 assert (complete['samples'],complete['updates'],complete['episode'])==(14737,57,428)
 assert not complete['buffer'] and not saved['buffer']
 assert saved['normalization']==complete['normalization'] and complete['excluded_old_version_transitions']==145
 def sha(p):
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 for arm,build,cpus in [('D0','S5-r38','0-9'),('D1','S5-r39','10-19')]:
  name=f'job-{stamp}-w51-{arm.lower()}';jobdir=LARGE/'jobs'/name;jobdir.mkdir(parents=True,exist_ok=False)
  identity=json.loads((ROOT/'work/r11-frozen'/build/'identity.json').read_text())
  job=dict(job_id=name,stream_id='w51-'+arm.lower(),arm=arm,seed=1,workers=10,affinity=cpus,
   started_unix_s=now,deadline=None,wall_deadline_enabled=False,controller_build=build,
   controller_binary_sha256=identity['binary_sha256'],
   checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),
   resume_checkpoint=str(resume.relative_to(ROOT)),resume_sha256=sha(resume),
   normalization_sha256=hashlib.sha256(json.dumps(saved['normalization'],sort_keys=True).encode()).hexdigest(),
   samples=14737,updates=57,tail=0,excluded=145,new_PPO_samples=0,new_PPO_updates=0,
   initial_state='normal complete original evaluation t=0; no historical plant restoration',
   actor='deterministic mean; training=False; frozen W44 A weights and normalization',
   controller_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
  c=copy.deepcopy(base);c.update(current_task='W51_PAIRED_DIAGNOSTIC',protocol_revision='W51_paired_D8_no_qualification')
  c['reuse'].update(active_controller_build=build,controller_artifact=build,selected_mechanism='H',input_reference_branch='constant')
  c['learning'].update(phase_id=name,stream_id=job['stream_id'],phase_initial_transitions=14737,phase_initial_updates=57,
   target_cumulative_transitions_per_stream=14737,target_cumulative_updates_per_stream=57,training_enabled=False,
   additional_transitions_from_reviewed_checkpoint=0,additional_updates_from_reviewed_checkpoint=0,
   active_streams=[],evaluation_only=True)
  c['resources'].update(mpc_workers=10,torch_threads=1,torch_device='cuda')
  c['execution']['background_job'].update(wall_deadline_enabled=False,default_maximum_wall_s=None)
  c['control_and_evaluation']['post_training_evaluation'].update(maximum_wall_s=None,wall_deadline_enabled=False)
  c['w51_fixed_policy']=job
  for dest in (jobdir,W/'jobs'/name):
   write(dest/'job.json',job);write(dest/'execution-contract.json',c)
  phase['jobs'].append(dict(arm=arm,job_id=name,directory=str(jobdir.relative_to(ROOT)),affinity=cpus))
 write(W/'phase.json',phase)
 return phase
def run_job(jobdir,levels):
 job=json.loads((jobdir/'job.json').read_text());records=[]
 state=dict(status='running',job=job['job_id'],arm=job['arm'],pid=os.getpid(),started_unix_s=time.time(),
  deadline=None,wall_deadline_enabled=False,stages=records,new_PPO_samples=0,new_PPO_updates=0)
 status=W/(job['arm']+'.json');write(status,state);failed=None
 for level in levels:
  if failed:
   records.append(dict(level=level,status='blocked',blocked_by=failed));continue
  attempt=job['job_id']+'-eval-PANN_PPO_S5-s'+str(job['seed'])+'-'+level+'-attempt01'
  command=[sys.executable,'-u','scripts/run_s1_go2_r12.py','--contract',str(jobdir/'execution-contract.json'),
   '--attempt',attempt,'--evaluation-start-unix',str(job['started_unix_s']),'--checkpoint',str(ROOT/job['checkpoint']),
   '--level',level,'--seed',str(job['seed']),'--workers','10','--build',job['controller_build'],'--condition','nominal']
  row=dict(attempt=attempt,level=level,status='running',command=command,started_unix_s=time.time(),
   qualification_usable=level!='D8')
  records.append(row);write(status,state);write(jobdir/'runtime.json',state)
  with (jobdir/(level+'.log')).open('w') as log:
   child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   row['pid']=child.pid;write(status,state);write(jobdir/'runtime.json',state);code=child.wait()
  p=SMALL/(attempt+'.json');result=json.loads(p.read_text()) if p.exists() else None
  row.update(exit_code=code,ended_unix_s=time.time())
  if result:
   row.update(status=result['stage_status'],reason=result['reason'],success=result['success'],
    simulation_s=result['simulation_s'],wall_s=result['wall_s'],result=str(p.relative_to(ROOT)))
  else:row.update(status='technical_interruption',reason='process_exited_without_completed_result',success=False)
  if level!='D8' and not row['success']:failed=level
  write(status,state);write(jobdir/'runtime.json',state)
 state.update(status='completed' if all(r['status']!='technical_interruption' for r in records) else 'technical_interruption',
  ended_unix_s=time.time())
 write(status,state);write(jobdir/'runtime.json',state)
 return state
def finalize(phase):
 from w51_run_summary import summarize_attempt
 paths=[W/'phase.json'];roots=[]
 comparisons={}
 for job in phase['jobs']:
  status=W/(job['arm']+'.json');paths.append(status)
  state=json.loads(status.read_text())
  for row in state['stages']:
   attempt=row.get('attempt')
   if not attempt:continue
   out=LARGE/attempt
   if out.exists():roots.append(out)
   if (SMALL/(attempt+'.json')).exists():
    p=summarize_attempt(attempt);paths.extend([p,SMALL/(attempt+'.json'),SMALL/(attempt+'-resolved.json')])
    comparisons[job['arm']]=json.loads(p.read_text())
    with (ROOT/job['directory']/'render.log').open('a') as log:
     row['render_exit_code']=subprocess.run([sys.executable,'scripts/render_s1_go2_r12.py','--attempt',attempt],
       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
   roots.append(ROOT/job['directory'])
  write(status,state)
 write(W/'paired-results.json',dict(arms=comparisons,qualification_usable=False,new_PPO_samples=0,new_PPO_updates=0,
  selection=None,selection_status='requires_reading_actual_support_windows_and_full_videos_under_W51_plan',
  original_stages='unrun; D8 cannot qualify E30 or E60'))
 paths.append(W/'paired-results.json')
 phase.update(status='paired_D8_ended_pending_scientific_selection',ended_unix_s=time.time())
 write(W/'phase.json',phase);compact(paths,'Record W51 paired D8 actual outcomes and support windows')
 delivery=deliver('paired-D8',roots);phase['delivery']=delivery;write(W/'phase.json',phase)
 compact([W/'phase.json'],'Record W51 paired D8 direct server delivery')
def pair():
 phase=json.loads((W/'phase.json').read_text())
 if phase['status']!='declared_not_started':raise ValueError('Use existing live jobs; pair already launched')
 phase.update(status='running',pid=os.getpid(),launched_unix_s=time.time(),tmux='s1-w51-pair')
 write(W/'phase.json',phase);children=[]
 for job in phase['jobs']:
  jobdir=ROOT/job['directory']
  log=(jobdir/'runner.log').open('a')
  child=subprocess.Popen(['taskset','-c',job['affinity'],sys.executable,'-u',__file__,'--job-dir',str(jobdir)],
   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  job['runner_pid']=child.pid;children.append((child,log));write(W/'phase.json',phase)
 for child,log in children:child.wait();log.close()
 os.sched_setaffinity(0,{22,23,24})
 finalize(phase)
def select(arm):
 # Invoke only after applying the plan's qualitative scientific selection to the
 # complete paired evidence. No invented automatic gait threshold.
 phase=json.loads((W/'phase.json').read_text())
 source=next(x for x in phase['jobs'] if x['arm']==arm)
 basejob=json.loads((ROOT/source['directory']/'job.json').read_text())
 build=basejob['controller_build'];now=time.time();stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now))
 phase['selection']=dict(arm=arm,controller_build=build,selected_unix_s=now)
 phase['formal_jobs']=[]
 for case,seed,episode,cpus in [('A',1,428,'0-9'),('B',2,417,'10-19')]:
  job=copy.deepcopy(basejob);name=f'job-{stamp}-w51-{case.lower()}';directory=LARGE/'jobs'/name
  directory.mkdir(parents=True,exist_ok=False)
  parent=LARGE/f'training/w44-{case.lower()}/PANN_PPO_S5-seed{seed}'
  checkpoint=parent/'checkpoint-014737.pt';resume=parent/f'resume-state-ep{episode:05d}.pt'
  job.update(job_id=name,stream_id='w51-'+case.lower(),arm=case,seed=seed,affinity=cpus,started_unix_s=now,
   checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),
   resume_checkpoint=str(resume.relative_to(ROOT)),resume_sha256=hashlib.file_digest(resume.open('rb'),'sha256').hexdigest())
  import torch
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
  job['normalization_sha256']=hashlib.sha256(json.dumps(saved['normalization'],sort_keys=True).encode()).hexdigest()
  c=json.loads((ROOT/source['directory']/'execution-contract.json').read_text())
  c.update(current_task='W51_ORIGINAL_FIXED_POLICY_LADDER',protocol_revision='W51_selected_frozen_original_ladder')
  c['learning'].update(phase_id=name,stream_id=job['stream_id']);c['w51_fixed_policy']=job
  for dest in (directory,W/'jobs'/name):write(dest/'job.json',job);write(dest/'execution-contract.json',c)
  session='s1-w51-'+case.lower()
  shell='source scripts/autodl_d57_env.sh && source scripts/autodl_network.sh direct && exec taskset -c '+cpus+' '+shlex.join([sys.executable,'-u',__file__,'--job-dir',str(directory),'--formal'])+' > '+shlex.quote(str(directory/'runner.log'))+' 2>&1'
  subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),shell],check=True)
  phase['formal_jobs'].append(dict(arm=case,directory=str(directory.relative_to(ROOT)),session=session))
 phase.update(status='original_E30_running',original_evaluations_started=2);write(W/'phase.json',phase)
 compact([W/'phase.json',W/'jobs'],'Start W51 selected frozen mechanism original A B ladders')
def main():
 p=argparse.ArgumentParser();p.add_argument('--declare',action='store_true');p.add_argument('--pair',action='store_true')
 p.add_argument('--job-dir',type=Path);p.add_argument('--formal',action='store_true');p.add_argument('--select',choices=['D0','D1'])
 a=p.parse_args()
 if a.declare:declare()
 elif a.pair:pair()
 elif a.select:select(a.select)
 else:
  result=run_job(a.job_dir,['E30','E60','E300','E600'] if a.formal else ['D8'])
  if a.formal:
   from w51_run_summary import summarize_attempt
   job=json.loads((a.job_dir/'job.json').read_text());paths=[W/(job['arm']+'.json')];roots=[a.job_dir]
   for row in result['stages']:
    if not row.get('attempt'):continue
    attempt=row['attempt'];out=LARGE/attempt
    if (SMALL/(attempt+'.json')).exists():
     paths.extend([summarize_attempt(attempt),SMALL/(attempt+'.json'),SMALL/(attempt+'-resolved.json')])
     with (a.job_dir/'render.log').open('a') as log:
      row['render_exit_code']=subprocess.run([sys.executable,'scripts/render_s1_go2_r12.py','--attempt',attempt],
       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    if out.exists():roots.append(out)
   write(W/(job['arm']+'.json'),result)
   compact(paths,'Record W51 original '+job['arm']+' actual qualification')
   result['delivery']=deliver(job['job_id'],roots);write(W/(job['arm']+'.json'),result)
   compact([W/(job['arm']+'.json')],'Record W51 original '+job['arm']+' delivery')
if __name__=='__main__':main()
