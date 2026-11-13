#!/usr/bin/env python3
"""Run the finite W47 identity pairs, publish their evidence, and stop.

Task B has already finished with the prescribed source unavailable and both latest
W46 context/perturbation pairs missing. This driver has no original-E30 branch.
"""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
from publish_s1_go2_w46 import compact
ROOT=Path(__file__).resolve().parents[1]
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
SMALL=ROOT/REL
W=SMALL/'w47'
LARGE=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47'
DOCS={'AGENTS.md':'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/phase.json',
      'README.md':'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/phase.json',
      'docs/TERRAIN_MEMORY_PLAN.md':'../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/phase.json',
      'docs/S1_W47_STABILITY_PLAN.md':'../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/phase.json'}
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def phase_write(phase):
 with (SMALL/'index.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  write(W/'phase.json',phase)
  idx=json.loads((SMALL/'index.json').read_text());idx['w47']=phase;write(SMALL/'index.json',idx)
def update_docs(phase):
 status=phase['status']
 note=('W47 原位分析、B 来源核验与 C 全量复算已完成；隔离诊断构建已完成，A 的 8 条恒等修复配对验证待启动。' if status=='ready_finite_identity_pairs' else
       'W47 原位分析、B 来源核验与 C 全量复算已完成；A 的 8 条恒等修复配对验证正在运行。' if status=='finite_identity_pairs_running' else
       'W47 有限计算已结束；配对验证、首分叉和工作量见运行索引。')
 if status=='completed_finite_work_and_delivery': note='W47 有限工作与新增原件交付已完成，未获得原任务准入。'
 note+=' 唯一 R8 原件在 D57 不可用，最新 W46 A/B 所需完整策略与同时间预测缺失；B 未启动 donor solve，未把历史 fallback 当最新双流覆盖。1944 次原调用中有 1601 次正步安装（1598 次 J 下降、3 次风险优先恢复）与 343 次内部停滞。N/r34 strict 保留，新 PPO 与新原 plant 回合均为 0；本轮诊断构建不冒称已部署 S5-r36。以下冻结计划和 W46 审查按其形成时点阅读，当前执行状态以索引为准。'
 for rel,link in DOCS.items():
  p=ROOT/rel;text=p.read_text();start='<!-- W47_EXECUTION_BEGIN -->';end='<!-- W47_EXECUTION_END -->'
  block=start+'\n> 执行更新（2026-11-13）：'+note+' [W47 实际运行索引]('+link+')。\n'+end
  if start in text:
   i=text.index(start);j=text.index(end,i)+len(end);text=text[:i]+block+text[j:]
  else:
   i=text.index('\n')+1;text=text[:i]+'\n'+block+'\n'+text[i:]
  p.write_text(text)
def package_originals():
 dest=LARGE/'delivery';dest.mkdir(parents=True,exist_ok=True)
 asset=dest/'w47-identity-pairs-originals-0001.zip'
 paths=sorted(p for p in (LARGE/'task-a/pairs').rglob('*') if p.is_file())
 paths += [ROOT/'work/w47-implementation'/name for name in ('old/agent_server','corrected/agent_server','identity.json')]
 sources=[]
 with zipfile.ZipFile(asset,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for p in paths:
   name=p.relative_to(ROOT).as_posix();h=hashlib.sha256()
   with p.open('rb') as src,z.open(name,'w',force_zip64=True) as out:
    while chunk:=src.read(8*1024*1024): h.update(chunk);out.write(chunk)
   sources.append({'path':name,'size':p.stat().st_size,'sha256':h.hexdigest(),'member':name})
 with asset.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 manifest=[{'path':asset.relative_to(ROOT).as_posix(),'name':asset.name,'category':'W47_identity_pair_originals','size':asset.stat().st_size,'sha256':digest}]
 write(W/'delivery/sources.json',sources);write(W/'delivery/manifest.json',manifest)
 return manifest

def finish(code):
 phase=json.loads((W/'phase.json').read_text());path=W/'task-a/paired-evidence.json'
 pairs=json.loads(path.read_text()) if path.exists() else None
 phase['task_A'].update(status=pairs['status'] if pairs else 'technical_failure_without_paired_summary',paired_evidence=str(path.relative_to(ROOT)) if pairs else None,paired_runner_exit_code=code)
 if pairs:
  phase['task_A']['paired_summary']=pairs
  phase['task_A']['new_rollouts_so_far']=pairs['actual_rollouts']
  phase['task_A']['new_physics_substeps']=pairs['physics_substeps']
  phase['task_A']['new_optimizer_solves']=0
 phase['task_B']=json.loads((W/'task-b.json').read_text())
 c=json.loads((W/'task-c/planner-summary.json').read_text())['combined']
 phase['task_C']={'status':'completed','call_count':c['calls'],'positive_step_installs':c['final_positive_step_installs'],'stop_reason':c['stop_reason'],'nominal_installs':c['nominal_installs'],'incumbent_retained':c['incumbent_retained'],'new_physics_substeps':0,'new_GPU_allocations':0,'result_directory':str((W/'task-c').relative_to(ROOT))}
 phase['status']='finite_work_completed_delivery_pending' if code==0 else 'finite_pairs_technical_failure_delivery_pending'
 phase['computation_ended_unix_s']=time.time()
 phase['background'].update(status='publishing_finite_results',computation_exit_code=code)
 phase['original_evaluation']={'status':'blocked_evidence_gate','attempts':[],'levels':{s:{level:'blocked_evidence_gate' for level in ('E30','E60','E300','E600')} for s in ('A','B')},'reason':'Both latest W46 representative full policies and time-matched perturbations unavailable; no demonstrated useful motion/recovery candidate. Historical fixture zero-endpoint repair is insufficient.'}
 phase['selected_controller_build']=None
 phase['material_memory_benefit']='not_established'
 phase['publication']={'status':'preparing_incremental_server_originals','server_originals_retained':True,'extra_PC_download_archive':False}
 phase_write(phase);update_docs(phase)
 commit=compact([W,*[ROOT/p for p in DOCS]],'Record W47 finite results without original-task admission')
 manifest=package_originals()
 phase['publication'].update(status='upload_pending',source_commit=commit,manifest=str((W/'delivery/manifest.json').relative_to(ROOT)),asset_count=len(manifest),asset_bytes=sum(x['size'] for x in manifest))
 phase_write(phase)
 source_commit=compact([W],'Record W47 generated original manifest')
 result={'source_commit':source_commit,'deadline_unix_s':None,'wall_deadline_enabled':False,'status':'publication_pending','started_unix_s':time.time()}
 for action in ('upload','finish'):
  cmd=[sys.executable,'scripts/publish_s1_go2_r12.py',action,'--server-auth','--manifest',str(W/'delivery/manifest.json'),'--job-name','w47-identity','--source-commit',source_commit,'--upload-workers','1','--no-wall-deadline']
  with (LARGE/'delivery/publisher.log').open('a') as log: rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
  result[action+'_exit_code']=rc
  if rc: break
 idx=json.loads((SMALL/'index.json').read_text());registered={a['name']:a for a in idx['publication']['assets']}
 checked=[]
 for row in manifest:
  actual=registered.get(row['name'],{})
  ok=actual.get('status')=='uploaded' and actual.get('size')==row['size'] and actual.get('remote_digest')=='sha256:'+row['sha256'] and actual.get('remote_state')=='uploaded'
  checked.append({'name':row['name'],'size':row['size'],'sha256':row['sha256'],'remote_digest':actual.get('remote_digest'),'verified':ok,'url':actual.get('url'),'release_tag':actual.get('release_tag')})
 verified=all(a['verified'] for a in checked)
 result.update(status='published_size_digest_verified' if verified else 'publication_pending',assets=checked,ended_unix_s=time.time(),payload_downloaded=False)
 write(W/'delivery/delivery.json',result)
 phase['publication']=result
 phase['status']='completed_finite_work_and_delivery' if code==0 and verified else 'finite_result_retained_publication_or_technical_issue'
 phase['finished_unix_s']=time.time();phase['active_compute_processes']=0
 phase['background'].update(status='finished',ended_unix_s=phase['finished_unix_s'])
 phase_write(phase);update_docs(phase)
 final_commit=compact([W,*[ROOT/p for p in DOCS]],'Publish W47 finite identity evidence and verified delivery')
 print(json.dumps({'status':phase['status'],'commit':final_commit,'assets':checked,'new_original_plant_episodes':0,'new_PPO_updates':0}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--finish-exit-code',type=int);args=p.parse_args()
 if args.finish_exit_code is not None: finish(args.finish_exit_code);return
 phase=json.loads((W/'phase.json').read_text())
 b=json.loads((W/'task-b.json').read_text())
 if b['latest_W46_A_B_both_covered'] or b['donor_status']!='donor_unavailable': raise ValueError('This bounded driver is for the frozen unavailable-source result only')
 phase.update(status='finite_identity_pairs_running',implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),runner_pid=os.getpid(),launched_unix_s=time.time(),deadline_unix_s=None,wall_deadline_enabled=False)
 phase['task_B']=b
 phase['background'].update(status='running',pid=os.getpid(),launched_unix_s=phase['launched_unix_s'])
 phase_write(phase);update_docs(phase)
 LARGE.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,'-u','scripts/w47_a_paired_identity.py']
 with (LARGE/'paired-runner.log').open('w') as log:
  child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  phase['task_A'].update(status='paired_native_running',paired_pid=child.pid,command=cmd);phase_write(phase)
  print(json.dumps({'event':'paired_native_started','pid':child.pid,'command':cmd,'deadline_unix_s':None}),flush=True)
  code=child.wait()
 print(json.dumps({'event':'paired_native_finished','exit_code':code}),flush=True)
 finish(code)
if __name__=='__main__':main()
