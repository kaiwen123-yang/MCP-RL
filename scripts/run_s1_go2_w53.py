#!/usr/bin/env python3
"""Frozen W53 original-task execution; no wall deadline or implicit restart."""
import argparse,copy,hashlib,json,os,resource,shlex,subprocess,sys,time
from pathlib import Path
from publish_s1_go2_w53 import ROOT,SMALL,LARGE,write,compact,deliver
W=SMALL/'w53'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def prepare_f():
    config=ROOT/'experiments/terrain_memory/s1-w53-f.json'
    backend=ROOT/'scripts/s1_w53_f_backend.py'
    dest=ROOT/'work/r11-frozen/S5-r43';dest.mkdir(exist_ok=True)
    identity=dict(build='S5-r43',route='F',backend='independent_CPU_force_MPC_WBC',
        binary_executed=False,backend_source='scripts/s1_w53_f_backend.py',
        backend_sha256=sha(backend),configuration_sha256=sha(config),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        physical_model_source='work/r11-frozen/S5-r41',historical_qualification_inherited=False,
        new_PPO_samples=0,new_PPO_updates=0,deadline=None,wall_deadline_enabled=False)
    write(dest/'identity.json',identity)
    return identity
def declare(route,case='A'):
    import torch
    label=route+'-'+case;status=W/(label+'.json')
    if status.exists():raise ValueError('Existing W53 identity: inspect status/PID and continue it; no duplicate declaration')
    seed=1 if case=='A' else 2;episode=428 if case=='A' else 417
    now=time.time();stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now))
    parent=LARGE/f'training/w44-{case.lower()}/PANN_PPO_S5-seed{seed}'
    checkpoint=parent/'checkpoint-014737.pt';resume=parent/f'resume-state-ep{episode:05d}.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    complete=torch.load(resume,map_location='cpu',weights_only=False)
    assert (saved['samples'],saved['updates'],saved['seed'])==(14737,57,seed)
    assert (complete['samples'],complete['updates'],complete['episode'])==(14737,57,episode)
    assert not saved['buffer'] and not complete['buffer']
    assert saved['normalization']==complete['normalization']
    build='S5-r42' if route=='N' else 'S5-r43'
    artifact=json.loads((ROOT/'work/r11-frozen'/build/'identity.json').read_text())
    name=f'job-{stamp}-w53-{route.lower()}-{case.lower()}'
    jobdir=LARGE/'jobs'/name;jobdir.mkdir(parents=True,exist_ok=False)
    cpus='0-9' if label=='N-A' or case=='A' and route=='N' else '10-19'
    if case=='B':cpus='10-19' if route=='N' else '0-9'
    job=dict(job_id=name,label=label,route=route,case=case,seed=seed,
        workers=10 if route=='N' else 1,affinity=cpus,
        started_unix_s=now,deadline=None,wall_deadline_enabled=False,
        controller_build=build,controller_binary_sha256=artifact.get('binary_sha256'),
        backend_sha256=artifact.get('backend_sha256'),controller_artifact_identity=artifact,
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),
        resume_checkpoint=str(resume.relative_to(ROOT)),resume_sha256=sha(resume),
        normalization_sha256=hashlib.sha256(json.dumps(saved['normalization'],sort_keys=True).encode()).hexdigest(),
        samples=14737,updates=57,tail=0,excluded=145,new_PPO_samples=0,new_PPO_updates=0,donor=0,
        actor='deterministic mean, training=False, frozen W44 case weights and normalization',
        initial_state='complete original evaluation t=0; virgin material, no donor',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan_commit='1832ba0018a5e69b40719f65e6166b2e1160e241')
    c=json.loads((ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json').read_text())
    c.update(current_task='W53_COMPLETE_CONTROLLER_ORIGINAL_TASK',protocol_revision='W53_original_frozen_ladder')
    c['reuse'].update(active_controller_build=build,controller_artifact=build)
    c['learning'].update(phase_id=name,stream_id='w53-'+label.lower(),phase_initial_transitions=14737,
        phase_initial_updates=57,target_cumulative_transitions_per_stream=14737,
        target_cumulative_updates_per_stream=57,training_enabled=False,
        additional_transitions_from_reviewed_checkpoint=0,additional_updates_from_reviewed_checkpoint=0,
        active_streams=[],evaluation_only=True)
    c['resources'].update(mpc_workers=job['workers'],torch_threads=1,torch_device='cuda')
    c['execution']['background_job'].update(wall_deadline_enabled=False,default_maximum_wall_s=None)
    c['control_and_evaluation']['post_training_evaluation'].update(maximum_wall_s=None,wall_deadline_enabled=False)
    c['w53_fixed_policy']=job
    for dest in (jobdir,W/'jobs'/name):
        write(dest/'job.json',job);write(dest/'execution-contract.json',c)
    state=dict(status='declared_not_started',job=job,stages=[],deadline=None,wall_deadline_enabled=False,
        directory=str(jobdir.relative_to(ROOT)),tmux='s1-w53-'+label.lower())
    write(status,state);return state
def launch(state,levels):
    job=state['job'];jobdir=ROOT/state['directory'];session=state['tmux']
    cmd=[sys.executable,'-u',__file__,'--job-dir',str(jobdir),'--levels',*levels]
    shell='source scripts/autodl_d57_env.sh && exec taskset -c '+job['affinity']+' '+shlex.join(cmd)+' > '+shlex.quote(str(jobdir/'runner.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),shell],check=True)
    subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
    return state
def resource_sample(pid,destination):
    rows={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            f=(p/'stat').read_text().rsplit(') ',1)[1].split()
            io={k.strip():int(v.strip()) for k,v in (line.split(':') for line in (p/'io').read_text().splitlines())}
            rows[int(p.name)]=(int(f[1]),int(f[11]),int(f[12]),int(f[21]),io.get('read_bytes',0),io.get('write_bytes',0))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    members={pid}
    while True:
        more={k for k,v in rows.items() if v[0] in members}
        if more<=members:break
        members|=more
    active=[rows[k] for k in members if k in rows];hz=os.sysconf('SC_CLK_TCK')
    sample=dict(unix_s=time.time(),root_pid=pid,pids=sorted(members & rows.keys()),
        user_cpu_s=sum(r[1] for r in active)/hz,system_cpu_s=sum(r[2] for r in active)/hz,
        rss_bytes=sum(r[3] for r in active)*os.sysconf('SC_PAGE_SIZE'),
        read_bytes=sum(r[4] for r in active),write_bytes=sum(r[5] for r in active))
    with destination.open('a') as f:f.write(json.dumps(sample)+'\n')
def summarize(attempt):
    import numpy as np
    p=SMALL/(attempt+'.json');r=json.loads(p.read_text());out=LARGE/attempt
    with np.load(out/'trajectory.npz') as z:
        t=z['time'];q=z['qpos'];f=z['actual_grf'];contact=z['contact']
        dt=np.diff(t);moving=(z['phase'][:-1]!=0)
        weights=dt*moving
        periods=[]
        for a,b in ((0.,2.),(2.,min(float(t[-1]),5.)),(5.,float(t[-1]))):
            m=(t[:-1]>=a)&(t[:-1]<b)
            if not m.any():continue
            ix=np.flatnonzero(m);w=dt[m];fz=f[:-1,:,2][m]
            periods.append(dict(start_s=float(t[ix[0]]),end_s=float(t[ix[-1]+1]),
                dx_m=float(q[ix[-1]+1,0]-q[ix[0],0]),
                min_z_m=float(q[ix,2].min()),
                foot_loaded_fraction=(np.sum((fz>1)*w[:,None],axis=0)/w.sum()).tolist(),
                four_foot_loaded_fraction=float(np.sum((fz>1).all(axis=1)*w)/w.sum())))
        summary=dict(attempt=attempt,route=r.get('w53_route'),original_result={k:r.get(k) for k in
            ('stage_status','reason','success','error','simulation_s','wall_s','physical_samples','max_x_m',
             'min_body_height_m','stage_progress','qualification','legal_route','completed_round_trips')},
            windows=periods,contact_transitions=np.count_nonzero(np.diff(contact.astype(int),axis=0),axis=0).tolist(),
            physical_failure_stage=('material_contact_or_later' if r['stage_progress']['actual_material_loaded'] else 'before_material'),
            new_PPO_samples=0,new_PPO_updates=0,donor=0,
            normal_motion_review='Use complete recorded video and these same-task trajectories; no automatic gait qualification threshold.',
            timing=r['timing'],record_timing=r.get('record_timing'))
    for filename in ('w53-n-recording.json','w53-f/manifest.json'):
        if (out/filename).exists():summary[filename]=json.loads((out/filename).read_text())
    dest=W/'original-evidence'/(attempt+'.json');write(dest,summary);return dest
def run_job(jobdir,levels):
    job=json.loads((jobdir/'job.json').read_text());status=W/(job['label']+'.json')
    state=json.loads(status.read_text())
    state.update(status='running',runner_pid=os.getpid(),launched_unix_s=time.time())
    write(status,state);failed=None
    for level in levels:
        if failed:
            state['stages'].append(dict(level=level,status='blocked_by_predecessor',blocked_by=failed));continue
        prior=[x for x in state['stages'] if x.get('level')==level]
        if prior and not state.pop('authorized_unstarted_technical_resume',False):
            raise ValueError('This identity already has an attempt at '+level+'; inspect exact technical record')
        number=len(prior)+1
        attempt=job['job_id']+f'-eval-PANN_PPO_S5-s{job["seed"]}-{level}-attempt{number:02d}'
        command=[sys.executable,'-u','scripts/run_s1_go2_r12.py','--contract',str(jobdir/'execution-contract.json'),
            '--attempt',attempt,'--evaluation-start-unix',str(job['started_unix_s']),
            '--checkpoint',str(ROOT/job['checkpoint']),'--level',level,'--seed',str(job['seed']),
            '--workers',str(job['workers']),'--build',job['controller_build'],'--condition','nominal']
        row=dict(attempt=attempt,level=level,status='running',command=command,started_unix_s=time.time())
        state['stages'].append(row);write(status,state)
        usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
        with (jobdir/(attempt+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=child.pid;write(status,state)
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:resource_sample(child.pid,jobdir/'resources.jsonl')
        usage1=resource.getrusage(resource.RUSAGE_CHILDREN)
        row.update(exit_code=code,ended_unix_s=time.time(),
            resource_usage=dict(user_cpu_s=usage1.ru_utime-usage0.ru_utime,
            system_cpu_s=usage1.ru_stime-usage0.ru_stime,children_maxrss_KiB=usage1.ru_maxrss))
        rp=SMALL/(attempt+'.json');paths=[status];roots=[jobdir];result=None
        if rp.exists():
            result=json.loads(rp.read_text())
            row.update(status=result['stage_status'],reason=result['reason'],success=result['success'],
                simulation_s=result['simulation_s'],wall_s=result['wall_s'],result=str(rp.relative_to(ROOT)))
            paths += [rp,SMALL/(attempt+'-resolved.json'),summarize(attempt)]
            with (jobdir/(attempt+'-render.log')).open('w') as log:
                row['render_exit_code']=subprocess.run([sys.executable,'scripts/render_s1_go2_r12.py','--attempt',attempt],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        else:
            row.update(status='technical_interruption',reason='process_exited_without_completed_result',success=False)
        if (LARGE/attempt).exists():roots.append(LARGE/attempt)
        if not row.get('success'):failed=level
        write(status,state);write(jobdir/'runtime.json',state)
        compact(paths,'Record W53 '+job['label']+' '+level+' actual original-task outcome')
        row['delivery']=deliver(job['label'].lower()+'-'+level.lower()+f'-attempt{number:02d}',roots)
        write(status,state)
        compact([status],'Verify W53 '+job['label']+' '+level+' direct original delivery')
    state.update(status='completed',ended_unix_s=time.time())
    write(status,state);write(jobdir/'runtime.json',state)
    compact([status],'Record W53 '+job['label']+' stage completion')
def select(route,review):
    chosen=json.loads((W/(route+'-A.json')).read_text())
    first=next(x for x in chosen['stages'] if x['level']=='E30')
    if first.get('status')!='passed':raise ValueError('Selected seed1 must pass original E30')
    evidence=json.loads(Path(review).read_text())
    if evidence.get('selected')!=route or evidence.get('normal_sustained_motion') is not True:
        raise ValueError('Selection needs completed trajectory/video scientific decision')
    write(W/'selection.json',evidence)
    jobdir=ROOT/chosen['directory'];chosen['tmux']='s1-w53-selected-'+route.lower()+'-a'
    write(W/(route+'-A.json'),chosen)
    launch(chosen,['E60','E300','E600'])
    launch(declare(route,'B'),['E30','E60','E300','E600'])
def resume_unstarted(route):
    status=W/(route+'-A.json');state=json.loads(status.read_text());jobdir=ROOT/state['directory']
    previous=state['stages'][-1]
    idx=json.loads((SMALL/'index.json').read_text())
    record=next(x for x in idx['attempts'] if x['attempt']==previous['attempt'])
    if previous['status']!='technical_interruption' or record['physical_started']:
        raise ValueError('This command is only for the inspected zero-plant technical startup correction')
    for original in ('job.json','execution-contract.json'):
        saved=jobdir/(previous['attempt']+'-'+original)
        if not saved.exists():saved.write_bytes((jobdir/original).read_bytes())
    job=state['job'];artifact=json.loads((ROOT/'work/r11-frozen'/job['controller_build']/'identity.json').read_text())
    job['controller_artifact_identity']=artifact
    job['controller_binary_sha256']=artifact.get('binary_sha256');job['backend_sha256']=artifact.get('backend_sha256')
    job['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    c=json.loads((jobdir/'execution-contract.json').read_text());c['w53_fixed_policy']=job
    for dest in (jobdir,W/'jobs'/job['job_id']):
        write(dest/'job.json',job);write(dest/'execution-contract.json',c)
    state.update(authorized_unstarted_technical_resume=True,status='technical_correction_ready',
        tmux='s1-w53-'+route.lower()+'-a-technical-correction')
    write(status,state);launch(state,['E30'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',choices=['N','F']);p.add_argument('--prepare-f',action='store_true')
    p.add_argument('--job-dir',type=Path);p.add_argument('--levels',nargs='+',choices=['E30','E60','E300','E600'])
    p.add_argument('--resume-unstarted',choices=['N','F']);p.add_argument('--select',choices=['N','F']);p.add_argument('--review',type=Path);a=p.parse_args()
    if a.prepare_f:prepare_f()
    elif a.launch:launch(declare(a.launch),['E30'])
    elif a.resume_unstarted:resume_unstarted(a.resume_unstarted)
    elif a.select:select(a.select,a.review)
    else:run_job(a.job_dir,a.levels)
if __name__=='__main__':main()
