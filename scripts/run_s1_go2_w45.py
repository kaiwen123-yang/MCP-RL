#!/usr/bin/env python3
"""Evaluation-only W45 ladder. No training branch and no wall deadline."""
import argparse,copy,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from publish_s1_go2_w45 import compact,deliver,write,ROOT,SMALL,LARGE
LEVELS=['E30','E60','E300','E600']
def run(jobdir):
 job=json.loads((jobdir/'job.json').read_text());contract=jobdir/'execution-contract.json'
 state=dict(job=job['job_id'],status='running',started_unix_s=time.time(),pid=os.getpid(),
  deadline_unix_s=None,new_PPO_samples=0,new_PPO_updates=0,samples=14737,updates=57,attempts=[],stages=[])
 path=SMALL/'w45'/(job['stream_id']+'.json')
 write(jobdir/'runtime.json',state);write(path,state)
 failed=None
 for level in LEVELS:
  if failed:
   state['stages'].append(dict(level=level,status='blocked',blocked_by=failed));continue
  attempt=job['job_id']+'-eval-PANN_PPO_S5-s'+str(job['seed'])+'-'+level+'-attempt01'
  command=[sys.executable,'-u','scripts/run_s1_go2_r12.py','--contract',str(contract),
   '--attempt',attempt,'--evaluation-start-unix',str(job['started_unix_s']),
   '--checkpoint',str(ROOT/job['checkpoint']),'--level',level,'--seed',str(job['seed']),
   '--workers','10','--build','S5-r34','--condition','nominal']
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
  compact([path,resultpath,SMALL/(attempt+'-resolved.json')] if result else [path],f'Record W45 {job["stream_id"]} {level} actual outcome')
 state['status']='completed_with_failed_qualification' if failed else 'completed_all_qualified'
 state['ended_unix_s']=time.time();write(path,state);write(jobdir/'runtime.json',state)
 compact([path],f'Complete W45 {job["stream_id"]} fixed-policy ladder')
 # Render actual recorded tasks only; no physics integration.
 for attempt in state['attempts']:
  if (LARGE/attempt/'trajectory.npz').exists():
   with (jobdir/'render.log').open('a') as log:
    code=subprocess.run([sys.executable,'scripts/render_s1_go2_r12.py','--attempt',attempt],
      cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
   state.setdefault('render',[]).append(dict(attempt=attempt,exit_code=code))
 write(path,state)
 deliver(job['stream_id'],[LARGE/name for name in state['attempts'] if (LARGE/name).exists()]+[jobdir])
def declare():
 import torch
 now=time.time();stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now))
 base=json.loads((ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json').read_text())
 identity=json.loads((ROOT/'work/r11-frozen/S5-r34/identity.json').read_text())
 phase=dict(phase='W45',status='declared_not_started',started_unix_s=now,deadline_unix_s=None,
  new_PPO_samples=0,new_PPO_updates=0,binary_sha256=identity['binary_sha256'],jobs=[],gpu='four_saved_problems_32_candidates_350ms')
 for suffix,seed,ep,cpus in [('a',1,428,'0-9'),('b',2,417,'10-19')]:
  name=f'job-{stamp}-w45-{suffix}';stream='w45-'+suffix;directory=LARGE/'jobs'/name
  directory.mkdir(parents=True,exist_ok=False)
  parent=LARGE/f'training/w44-{suffix}/PANN_PPO_S5-seed{seed}'
  checkpoint=parent/'checkpoint-014737.pt';resume=parent/f'resume-state-ep{ep:05d}.pt'
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
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
    controller_build='S5-r34',controller_binary_sha256=identity['binary_sha256'],
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    resume_rule='Live PID may resume; exited process without full simulation state remains interrupted. A new explicit attempt starts from original initial state, never concatenated.')
  c=copy.deepcopy(base);c.update(current_task='W45_FIXED_POLICY_STABILITY',protocol_revision='W45_fixed_policy_r34_original_ladder')
  c['reuse'].update(active_controller_build='S5-r34',controller_artifact='S5-r34',selected_mechanism='N')
  c['learning'].update(phase_id=name,stream_id=stream,phase_initial_transitions=14737,phase_initial_updates=57,
    target_cumulative_transitions_per_stream=14737,target_cumulative_updates_per_stream=57,
    additional_transitions_from_reviewed_checkpoint=0,additional_updates_from_reviewed_checkpoint=0,
    training_enabled=False,active_streams=[],evaluation_only=True)
  c['execution']['background_job'].update(wall_deadline_enabled=False,default_maximum_wall_s=None)
  c['resources'].update(mpc_workers=10,torch_threads=1,torch_device='cuda')
  c['w45_fixed_policy']=job
  write(directory/'job.json',job);write(directory/'execution-contract.json',c)
  write(SMALL/'w45'/'jobs'/name/'job.json',job);write(SMALL/'w45'/'jobs'/name/'execution-contract.json',c)
  phase['jobs'].append(dict(job_id=name,stream_id=stream,directory=str(directory.relative_to(ROOT)),
    session='s1-'+stream,affinity=cpus,checkpoint_sha256=job['checkpoint_sha256']))
 write(SMALL/'w45'/'phase.json',phase);write(LARGE/'w45-clock.json',dict(started_unix_s=now,deadline_unix_s=None))
 with (SMALL/'index.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);idx=json.loads((SMALL/'index.json').read_text());idx['w45']=phase;write(SMALL/'index.json',idx)
 print(json.dumps(phase,indent=2))
def start():
 import shlex
 phase=json.loads((SMALL/'w45/phase.json').read_text())
 for row in phase['jobs']:
  jobdir=ROOT/row['directory'];command=[sys.executable,'-u','scripts/run_s1_go2_w45.py','--job-dir',str(jobdir)]
  shell='source scripts/autodl_d57_env.sh && source scripts/autodl_network.sh academic && exec taskset -c '+row['affinity']+' '+shlex.join(command)+' >> '+shlex.quote(str(jobdir/'runner.log'))+' 2>&1'
  subprocess.run(['tmux','new-session','-d','-s',row['session'],'-c',str(ROOT),shell],check=True)
 phase.update(status='launched',launched_unix_s=time.time())
 write(SMALL/'w45/phase.json',phase)
def main():
 p=argparse.ArgumentParser();p.add_argument('--declare',action='store_true');p.add_argument('--start',action='store_true');p.add_argument('--job-dir',type=Path);a=p.parse_args()
 if a.declare:declare()
 elif a.start:start()
 else:run(a.job_dir)
if __name__=='__main__':main()
