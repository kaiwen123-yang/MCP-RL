"""Run one W57 simulation through the frozen W44 actor and the original R12 plant."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from publish_s1_go2_w54 import write
from run_s1_w55_f import digest
from s1_go2_r11_protocol import stable_digest

ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-local-readiness-r12'
REL=Path('results/terrain_memory/simulation')/RUN
SMALL=ROOT/REL/'w57'
LARGE=ROOT/'outputs/terrain_memory/simulation'/RUN
BASE='experiments/terrain_memory/s1-w56-f-r60.json'
MECHANISM='experiments/terrain_memory/s1-w57-mechanism.json'
PLAN='d321d61fd13ba301b477deaf473101debebb88e5'
BACKGROUND=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)


def read(path):
    return json.loads(Path(path).read_text())


def configure_episode(c, contract):
    w=contract['w57']
    c.update(w57=w, implementation_status='W57_force_MPC_WBC_online',material_feature_mode='learned_history')
    cfg=c['w54_f']
    cfg['w57']={'planner':w['planner'],'material':w['material'],'timing':w['timing_mode'],
                'candidate_substep_s':w['candidate_substep_s'],'planner_history':w['planner_history'],
                'planner_material_backend':w['planner_material_backend'],
                'planner_material_workers':w['planner_material_workers']}
    cfg['wbc']['material']=cfg['material_reference']
    cfg.update(identity=c['controller_build'],backend_source=c['w54_backend_source'],
               material_execution_backend='CPU FP64 original frozen potentials; '+w['observer_implementation'])
    sources=contract['w57_execution_sources']
    c['w54_execution_sources']=list(sources)
    c['w54_execution_source_names']=[p.removeprefix('scripts/') for p in sources
                                     if p.startswith('scripts/') and p.endswith('.py')]
    if 'w57_timing' in contract:
        c['w57_timing']=contract['w57_timing']
    mode=w['task_mode']
    if mode in ('short','recovered'):
        spec=w['mechanism']
        duration=spec['observation_s'][mode]
        c['r11_stage']=dict(level='mechanism_'+mode, duration_s=duration, task_kind='world_cell_revisit',
                            target_cycles=spec['cycles'], qualification_usable=False)
        c['task']['maximum_episode_s']=duration
    elif mode=='training':
        c['r11_stage']['qualification_usable']=False


def declaration(a):
    parent_dir=ROOT/REL/'w56/f'/f'F-{a.case}-r60'
    parent=read(parent_dir/'job.json')
    base=read(ROOT/BASE)
    opts=read(a.timing_config) if a.timing_config else None
    uses_timing=a.arm in ('R','PR','blind')
    if uses_timing != (opts is not None):
        raise ValueError('R, PR and blind require an explicit timing configuration')
    if a.task=='training' and (a.arm!='R' or a.level!='E60' or opts['mode']!='collect'):
        raise ValueError('Acquisition uses two W44 cases on M/P0/original E60')
    if uses_timing and a.task!='training' and opts['mode']!='evaluate':
        raise ValueError('Evaluation requires a frozen timing checkpoint')
    if opts and (opts['head']=='blind') != (a.arm=='blind' or a.task=='training' and opts['head']=='blind'):
        raise ValueError('Mismatched declared timing input')
    if opts and opts.get('case',a.case)!=a.case:
        raise ValueError('Collector case mismatch')
    planner='P1' if a.arm in ('P','PR') else 'P0'
    if a.history!='full' and planner!='P1':
        raise ValueError('Private history intervention is a P1 candidate condition')
    unchanged=a.arm=='B' and a.material=='M0'
    backend=base['backend_source'] if a.arm=='B' else 'scripts/s1_w57_f_backend.py'
    observer='r60_cpu_window' if unchanged else 'w57_measured_window'
    w=dict(arm=a.arm,planner=planner,planner_history=a.history,candidate_substep_s=.0002,
           planner_material_backend='equal_columns' if planner=='P1' else 'original',
           planner_material_workers=4 if planner=='P1' else 1,
           material=a.material,observer_implementation=observer,
           timing_mode=opts['mode'] if opts else 'zero',timing_head=opts['head'] if opts else None,
           task_mode=a.task,force_model='centroidal force QP and articulated inverse-dynamics WBC, original PD60/5',
           training_distribution='original virgin E60' if a.task=='training' else None)
    if a.task in ('short','recovered'):
        w['mechanism']=read(ROOT/MECHANISM)
    sources=[BASE,'scripts/run_s1_w57.py','scripts/run_s1_go2_r12.py',
             'scripts/s1_go2_w54_config.py','scripts/s1_go2_r11_protocol.py','scripts/s1_go2_r11_runtime.py',
             'scripts/s1_go2_joint_material.py','scripts/s1_go2_r11_policy.py','scripts/s1_w53_f_backend.py',
             base['backend_source'],*('scripts/'+n for n in base['inherited_execution_sources']),
             *base['material_execution_sources'],*base['material_compiled_artifacts'],
             'work/assets/B_random1000--ideal-potentials.npz']
    if not unchanged:
        sources += [backend,'scripts/s1_w57_material.py','scripts/s1_w57_material_columns.cc','work/w57-m/material.so']
    if planner=='P1':
        sources += ['experiments/terrain_memory/s1-w57-planning.json','scripts/s1_w57_planning.py',
                    'scripts/s1_w57_candidate.cc','scripts/build_s1_w57_equal_columns.sh','work/w57-p-equal/candidate.so']
    if uses_timing:
        sources += ['experiments/terrain_memory/s1-w57-timing.json','scripts/s1_w57_timing.py',
                    'scripts/s1_w57_learning.py','scripts/run_s1_w57_learning.py',
                    'scripts/s1_w57_training_evidence.py','src/response_ppo/ppo_backend.py']
    if a.task in ('short','recovered'):
        sources += [MECHANISM,'scripts/s1_w57_mechanism.py']
    hashes={s:digest(ROOT/s) for s in sorted(set(sources))}
    timing_hash=digest(Path(opts['checkpoint'])) if opts and opts['mode']=='evaluate' else None
    kernel=dict(arm=a.arm,material=a.material,planner=planner,planner_history=a.history,
                observer=observer,backend=backend,
                timing_checkpoint=timing_hash,timing_head=w['timing_head'],executables=hashes)
    build='S5-w57-'+a.arm.lower()+'-'+stable_digest(kernel)[:12]
    artifact=dict(route='F',experiment_window='W57',configuration_source=BASE,backend_source=backend,
                  physical_model_source='work/r11-frozen/S5-r41',controller_build=build,**kernel)
    # Immutable content address. Simultaneous A/B declarations write identical bytes atomically.
    artifact_file=ROOT/'work/r11-frozen'/build/'identity.json'
    if not artifact_file.exists():write(artifact_file,artifact)
    elif read(artifact_file)!=artifact:raise ValueError(f'Artifact identity collision: {build}')
    checkpoint=ROOT/parent['checkpoint']
    if digest(checkpoint)!=parent['checkpoint_sha256'] or digest(ROOT/parent['resume_checkpoint'])!=parent['resume_sha256']:
        raise ValueError('Frozen W44 parent files do not match its original published case')
    job=f'w57-{a.case.lower()}-{a.arm.lower()}-{build.rsplit("-",1)[1]}'
    if opts and a.task=='training':
        job+='-'+opts['head']
    spec=dict(run_id=RUN,current_task='W57_'+a.task,protocol_revision='W57_original_or_separate_mechanism_v1',
              formal_evaluation_protocol='staged_evaluation_30_60_300_600_v3_bidirectional_material_route',
              w54_fixed_policy={'route':'F'},material_feature_mode='learned_history',w57=w,
              reuse=dict(active_controller_build=build,controller_artifact=build,
                         runtime_physics_ppo_and_task_settings='experiments/terrain_memory/s1-go2-material-continuation-r11.json'),
              resources=dict(mpc_workers=1,torch_threads=1,torch_device='cuda',
                             pann_weights_path='work/assets/B_random1000--ideal-potentials.npz'),
              outputs=dict(large_root=str(LARGE.relative_to(ROOT)),compact_root=str((SMALL/'episodes').relative_to(ROOT))),
              learning=dict(phase_id=job,stream_id=job,phase_initial_transitions=14737,phase_initial_updates=57,
                            target_cumulative_transitions_per_stream=14737,target_cumulative_updates_per_stream=57,
                            reward_version='W38_legal_route_terminal_settlement',training_enabled=False,evaluation_only=True),
              execution=dict(background_job=BACKGROUND,**BACKGROUND),
              control_and_evaluation={'post_training_evaluation':dict(maximum_wall_s=None,wall_deadline_enabled=False)},
              w57_execution_sources=hashes,**BACKGROUND)
    if opts:spec['w57_timing']={**opts, 'case':a.case}
    # Allocate the original once, on D57, and keep its canonical path usable.
    # Full collectors use local scratch. Matched blind, recovered and long jobs use
    # the server's personal persistent volume to accommodate full-rate recordings.
    store=os.environ.get('W57_RAW_STORE')
    if not store and (a.task=='recovered' or a.level in ('E300','E600') or
                      a.task=='training' and opts['head']=='blind'):
        store='/autodl-fs/data/MCP-RL/W57/originals'
    if store:spec['w57_raw_root']=store
    return spec,artifact,parent,checkpoint,job


def episode(a, *, declare=declaration, small=SMALL):
    if not a.attempt or Path(a.attempt).name!=a.attempt:
        raise ValueError('Give each natural attempt a unique basename')
    spec, artifact, parent, checkpoint, job=declare(a)
    raw_job=LARGE/'jobs'/a.attempt
    raw_job.mkdir(parents=True,exist_ok=False)
    config=raw_job/'execution-contract.json'
    write(config,spec)
    now=time.time()
    command=[sys.executable,'-u','scripts/run_s1_go2_r12.py','--contract',str(config),
             '--attempt',a.attempt,'--evaluation-start-unix',str(now),'--level',a.level,
             '--checkpoint',str(checkpoint),'--seed',str(parent['seed']),'--workers','1',
             '--build',artifact['controller_build'],'--condition','nominal']
    receipt=dict(attempt=a.attempt,job_id=job,case=a.case,seed=parent['seed'],level=a.level,
                 task_mode=a.task,arm=a.arm,material=a.material,controller=artifact['controller_build'],
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 plan_commit=spec.get('plan_commit',PLAN),execution_contract_sha256=digest(config),active_sources=spec['w57_execution_sources'],
                 checkpoint_sha256=parent['checkpoint_sha256'],normalization_sha256=parent['normalization_sha256'],
                 retained_resume=parent['resume_checkpoint'],retained_resume_sha256=parent['resume_sha256'],
                 timing_checkpoint_sha256=artifact['timing_checkpoint'],settings=spec['w57'],old_actor_updates=0,
                 command=command,launcher_command=sys.argv,tmux=os.environ.get('W58_TMUX') or os.environ.get('W57_TMUX'),
                 slot=os.environ.get('S1_W58_SLOT') or os.environ.get('S1_W56_SLOT'),runner_pid=os.getpid(),started_unix_s=now,
                 cpu_affinity=sorted(os.sched_getaffinity(0)),status='declared',**BACKGROUND)
    if 'w58' in spec:
        receipt.update(w58=spec['w58'])
    if 'w59' in spec:
        receipt.update(w59=spec['w59'],branch=spec.get('w59_branch'),
            slot=os.environ.get('S1_W59_SLOT') or receipt['slot'],tmux=os.environ.get('W59_TMUX') or receipt['tmux'])
    target=small/'attempts'/(a.attempt+'.json')
    write(target,receipt)
    from run_s1_go2_w54 import resource_sample
    usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (raw_job/'physics.log').open('w') as log:
        lease_fds=tuple(int(n) for n in os.environ.get('S1_W58_LEASE_FDS','').split(',') if n)
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=lease_fds)
        receipt.update(pid=proc.pid,status='running')
        write(target,receipt)
        while proc.poll() is None:
            resource_sample(proc.pid,raw_job/'resources.jsonl')
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt.update(ended_unix_s=time.time(),exit_code=proc.returncode,
                   user_cpu_s=usage.ru_utime-usage0.ru_utime,system_cpu_s=usage.ru_stime-usage0.ru_stime)
    path=small/'episodes'/(a.attempt+'.json')
    if path.is_file():
        result=read(path)
        receipt.update({k:result.get(k) for k in ('stage_status','success','reason','simulation_s','physical_samples',
                                                  'physical_started_unix_s','physical_ended_unix_s','case_key',
                                                  'mechanism','qualification','error')})
        receipt['status']=result['stage_status']
    else:
        receipt.update(status='technical_interruption',reason='no_final_physics_result')
    write(target,receipt)
    print(json.dumps({k:receipt.get(k) for k in ('attempt','status','simulation_s','reason','exit_code')}),flush=True)
    return proc.returncode or (1 if receipt['status']=='technical_interruption' else 0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    e=sub.add_parser('episode')
    e.add_argument('--case',choices=('A','B'),required=True)
    e.add_argument('--arm',choices=('B','P','R','PR','blind'),required=True)
    e.add_argument('--level',choices=('E30','E60','E300','E600'),default='E60')
    e.add_argument('--task',choices=('original','short','recovered','training'),required=True)
    e.add_argument('--material',choices=('M0','M1'),default='M0')
    e.add_argument('--history',choices=('full','private_reset'),default='full')
    e.add_argument('--attempt',required=True)
    e.add_argument('--timing-config',type=Path)
    sys.exit(episode(p.parse_args()))
