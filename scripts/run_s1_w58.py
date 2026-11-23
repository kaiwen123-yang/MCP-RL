"""Declare a frozen W58 control identity and run one original or v2 physical episode on D57."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from publish_s1_go2_w54 import write
from run_s1_w55_f import digest
from run_s1_w57 import (ROOT, RUN, REL, LARGE, BASE, BACKGROUND, read,
                        episode as execute_physics, configure_episode as configure_base)
from s1_go2_r11_protocol import stable_digest

SMALL=ROOT/REL/'w58'
SPEC='experiments/terrain_memory/s1-w58-revisit-v2.json'
JOINT='experiments/terrain_memory/s1-w58-joint.json'
ARMS=('G1','B2','P2-full','P2-private','Fixed-wait')
PLAN='fe320346'
P2_CONFIG='experiments/terrain_memory/s1-w58-p2.json'


def configure_episode(c, contract):
    configure_base(c,contract)
    setting=contract['w58']
    c.update(w58=setting,implementation_status='W58 original plant, frozen force MPC/WBC and causal feedback')
    c['w54_f']['w58']=dict(setting)
    if setting['split'] in ('development','validation'):
        c['r11_stage']['target_cycles']=12


def declaration(a):
    parent=read(ROOT/REL/'w56/f'/f'F-{a.case}-r60/job.json')
    base=read(ROOT/BASE)
    p2=a.arm!='G1'
    opts=read(a.timing_config) if a.timing_config else None
    if opts:
        if opts['learning_seed']!=a.learning_seed or opts['head']!=('blind' if a.arm=='P2-private' else 'full'):
            raise ValueError('Head, seed and planning condition disagree')
        if a.arm not in ('P2-full','P2-private') or (opts['mode']=='collect')!=(a.task=='training'):
            raise ValueError('Joint collection requires the selected P2 control and the training distribution')
    if (a.task in ('short','recovered')) != (a.split in ('development','validation')):
        raise ValueError('A v2 physical trial needs its own declared route split')
    base_paths=[BASE,SPEC,'scripts/run_s1_w58.py','scripts/run_s1_w57.py','scripts/run_s1_go2_r12.py',
        'scripts/s1_go2_r11_runtime.py','scripts/s1_go2_r11_protocol.py','scripts/s1_go2_w54_config.py',
        'scripts/s1_go2_joint_material.py','scripts/s1_go2_r11_policy.py','scripts/s1_go2_r11_model.py',
        'scripts/s1_go2_wb_equilibrium.py','scripts/s1_go2_support_reference.py','scripts/s1_go2_legal_route.py',
        'scripts/s1_w53_f_backend.py','scripts/s1_w57_f_backend.py','scripts/s1_w57_material.py',
        'scripts/s1_w57_material_columns.cc','scripts/s1_w57_mechanism.py','scripts/s1_w58_revisit.py',
        'work/assets/B_random1000--ideal-potentials.npz','work/w57-m/material.so',
        base['backend_source'],*('scripts/'+n for n in base['inherited_execution_sources']),
        *base['material_execution_sources'],*base['material_compiled_artifacts']]
    backend=('scripts/s1_w58_gait.py' if not p2 else
             'scripts/s1_w58_p2_gait_backend.py' if a.gait=='G1' else 'scripts/s1_w58_p2_backend.py')
    if a.gait=='G1':base_paths+=['scripts/s1_w58_gait.py']
    if p2:
        base_paths += [backend,'scripts/s1_w58_p2.py',P2_CONFIG,
                       'scripts/s1_w57_planning.py','scripts/s1_w57_candidate.cc',
                       'scripts/build_s1_w57_equal_columns.sh','work/w57-p-equal/candidate.so']
    if a.material=='M2':base_paths+=['scripts/s1_w58_material.py']
    if opts:
        base_paths += [JOINT,'experiments/terrain_memory/s1-w57-timing.json','scripts/run_s1_w58_joint.py',
            'scripts/s1_w57_learning.py','scripts/s1_w57_timing.py','scripts/s1_w57_training_evidence.py',
            'src/response_ppo/ppo_backend.py']
    hashes={s:digest(ROOT/s) for s in sorted(set(base_paths))}
    timing_hash=digest(Path(opts['checkpoint'])) if opts and opts['mode']=='evaluate' else None
    workers=read(ROOT/P2_CONFIG)['native_workers'] if p2 else 1
    kernel=dict(arm=a.arm,material=a.material,gait=a.gait,backend=backend,
                timing_checkpoint=timing_hash,timing_head=opts['head'] if opts else None,
                learning_seed=a.learning_seed,material_workers=workers,executables=hashes)
    build='S5-w58-'+stable_digest(kernel)[:18]
    artifact=dict(route='F',experiment_window='W58',configuration_source=BASE,backend_source=backend,
                  physical_model_source='work/r11-frozen/S5-r41',controller_build=build,**kernel)
    identity=ROOT/'work/r11-frozen'/build/'identity.json'
    if not identity.exists():write(identity,artifact)
    elif read(identity)!=artifact:raise ValueError('Different source content for an existing W58 controller')
    checkpoint=ROOT/parent['checkpoint']
    if digest(checkpoint)!=parent['checkpoint_sha256'] or digest(ROOT/parent['resume_checkpoint'])!=parent['resume_sha256']:
        raise ValueError('W44 case actor or retained state differs from its qualified parent')
    job=f'w58-{a.case}-{build}'
    w=dict(arm=a.arm,planner='P1' if p2 else 'P0',planner_history='private_reset' if a.arm=='P2-private' else 'full',
        planner_material_workers=workers,planner_material_backend='equal_columns',candidate_substep_s=.0002,
        material=a.material,observer_implementation='w57_measured_window',
        timing_mode=opts['mode'] if opts else 'zero',timing_head=opts['head'] if opts else None,
        task_mode=a.task,force_model='centroidal force QP, articulated WBC and original PD60/5')
    if a.task in ('short','recovered'):w['mechanism']=read(ROOT/SPEC)['splits'][a.split]
    setting=dict(arm=a.arm,gait=a.gait,material=a.material,split=a.split,learning_seed=a.learning_seed,
                 planner='P2' if p2 else 'P0',native_workers=workers)
    spec=dict(run_id=RUN,current_task='W58_'+a.task,protocol_revision='W58_original_or_forward_revisit_v2',
        formal_evaluation_protocol='staged_evaluation_30_60_300_600_v3_bidirectional_material_route',
        plan_commit=PLAN,w54_fixed_policy={'route':'F'},w58=setting,w57=w,material_feature_mode='learned_history',
        reuse=dict(active_controller_build=build,controller_artifact=build,
                   runtime_physics_ppo_and_task_settings='experiments/terrain_memory/s1-go2-material-continuation-r11.json'),
        resources=dict(mpc_workers=1,torch_threads=1,torch_device='cuda',pann_weights_path='work/assets/B_random1000--ideal-potentials.npz'),
        outputs=dict(large_root=str(LARGE.relative_to(ROOT)),compact_root=str((SMALL/'episodes').relative_to(ROOT))),
        learning=dict(phase_id=job,stream_id=job,phase_initial_transitions=14737,phase_initial_updates=57,
                      target_cumulative_transitions_per_stream=14737,target_cumulative_updates_per_stream=57,
                      reward_version='W38_legal_route_terminal_settlement',training_enabled=False,evaluation_only=True),
        execution=dict(background_job=BACKGROUND,**BACKGROUND),
        control_and_evaluation={'post_training_evaluation':dict(maximum_wall_s=None,wall_deadline_enabled=False)},
        w57_execution_sources=hashes,**BACKGROUND)
    if opts:spec['w57_timing']={**opts,'case':a.case}
    if os.environ.get('W58_RAW_STORE'):spec['w57_raw_root']=os.environ['W58_RAW_STORE']
    return spec,artifact,parent,checkpoint,job


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('A','B'),required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--gait',choices=('G0','G1'),required=True)
    p.add_argument('--material',choices=('M0','M1','M2'),required=True)
    p.add_argument('--task',choices=('original','short','recovered','training'),required=True)
    p.add_argument('--split',choices=('original','development','validation','training'),required=True)
    p.add_argument('--level',choices=('E30','E60','E300','E600'),required=True)
    p.add_argument('--attempt',required=True)
    p.add_argument('--timing-config',type=Path)
    p.add_argument('--learning-seed',type=int,choices=(5801,5802,5803))
    return p.parse_args()


if __name__=='__main__':
    args=arguments()
    if args.learning_seed:
        os.environ['S1_W58_JOINT_SEED']=str(args.learning_seed)
    if bool(args.learning_seed)!=bool(args.timing_config):
        raise ValueError('Joint evaluation and collection require both a real learning seed and its timing policy')
    sys.exit(execute_physics(args,declare=declaration,small=SMALL))

