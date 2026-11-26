"""Declare one W59 Go2 identity and run its original, mechanism or registered branch."""
import argparse
import os
from pathlib import Path
import sys

from publish_s1_go2_w54 import write
from run_s1_w55_f import digest
from run_s1_w57 import ROOT, RUN, REL, LARGE, BASE, BACKGROUND, read, episode, configure_episode as configure_base
from s1_go2_r11_protocol import stable_digest

SMALL = ROOT/REL/'w59'
CONTRACT = 'experiments/terrain_memory/s1-w59-research-contract.json'
BACKEND = 'scripts/s1_w59_f_backend.py'
ARMS = ('B3','Fixed-action','P3-full','P3-private')


def configure_episode(c,contract):
    configure_base(c,contract)
    c.update(w59=contract['w59'], implementation_status='W59 original plant, held macro request, constrained force MPC/WBC')
    c['w54_f']['w59'] = contract['w59']
    if contract.get('w59_branch'):
        c['r11_stage']['qualification_usable']=False


def declaration(a):
    parent = read(ROOT/REL/'w56/f'/f'F-{a.case}-r60/job.json')
    base = read(ROOT/BASE)
    planner = a.arm.startswith('P3')
    models = [Path(p).resolve() for p in a.wm_checkpoint]
    if planner and len(models)!=3:
        raise ValueError('P3 needs the frozen calibrated three-seed WM ensemble')
    if a.material=='M3' and not planner:
        raise ValueError('M3 material comparison uses the matched P3 execution predictor')
    planning_recipe=read(a.wm_recipe) if a.wm_recipe else {'planning_substeps':32,'successor':'nominal_at_next_legal_request'}
    learner = read(a.learner_config) if a.learner_config else None
    nominal_ppo=bool(learner and a.arm=='B3' and learner.get('head')=='PPO-full' and
        learner.get('science_role')=='fallback-ppo' and learner.get('nominal_fallback'))
    if learner and not planner and not nominal_ppo:
        raise ValueError('R3 receives the per-event P3 base; only explicitly marked real PPO can use B3')
    sources = {BASE, CONTRACT, BACKEND,'scripts/run_s1_w59_research.py','scripts/s1_w59_snapshots.py',
        'scripts/run_s1_w57.py','scripts/run_s1_go2_r12.py','scripts/s1_go2_w54_config.py',
        'scripts/s1_go2_r11_protocol.py','scripts/s1_go2_r11_runtime.py','scripts/s1_w57_f_backend.py',
        'scripts/s1_w57_material.py','scripts/s1_w57_material_columns.cc','scripts/s1_go2_r11_model.py',
        'scripts/s1_go2_wb_equilibrium.py','scripts/s1_go2_support_reference.py','scripts/s1_go2_legal_route.py',
        'scripts/s1_go2_joint_material.py','scripts/s1_w59_model_cells.py','scripts/s1_w59_model_fields.py',
        'work/w57-m/material.so','work/assets/B_random1000--ideal-potentials.npz',
        base['backend_source'],*('scripts/'+f for f in base['inherited_execution_sources']),
        *base['material_execution_sources'],*base['material_compiled_artifacts']}
    # Post-run contact metrics do not change controller identity or trajectories.
    if planner:
        sources.add('scripts/s1_w59_planning.py')
        sources.update('scripts/s1_w59_model_'+key+'.py' for key in (
            'cells','fields','bank','rigid','calibration','reload','recovery'))
        sources.add('scripts/s1_w59_forward_domain.py')
        sources.add('scripts/s1_w59_model.py')
    if a.material=='M3':
        sources.update(['scripts/s1_w59_model_material.py','experiments/terrain_memory/w59-recovery-parameters.json'])
    if nominal_ppo:sources.add('scripts/s1_w59_planning.py')
    if learner:
        sources.update(['scripts/s1_w59_learning.py','scripts/s1_w59_learner_critic.py',
                        'experiments/terrain_memory/s1-w59-learner.json'])
    if a.task in ('short','recovered'):
        sources.add('scripts/s1_w59_geometry.py')
    hashes={p:digest(ROOT/p) for p in sorted(sources)}
    wm_hashes=[digest(p) for p in models]
    actor_sha=digest(Path(learner['checkpoint'])) if learner and learner.get('checkpoint') else None
    kernel=dict(arm=a.arm,material=a.material,gait='G0',backend=BACKEND,executables=hashes,
                wm_sha256=wm_hashes,actor_sha256=actor_sha,learner=learner,material_workers=1,
                planning_recipe=planning_recipe if planner else None)
    # Mechanism selection changes the task, not the controller.
    kernel['executables'] = {p:v for p,v in hashes.items() if p!='scripts/s1_w59_geometry.py'}
    build='S5-w59-research-'+stable_digest(kernel)[:18]
    artifact=dict(route='F',experiment_window='W59-research',configuration_source=BASE,
        physical_model_source='work/r11-frozen/S5-r41',backend_source=BACKEND,controller_build=build,
        timing_checkpoint=None,**kernel)
    identity=ROOT/'work/r11-frozen'/build/'identity.json'
    if not identity.exists():write(identity,artifact)
    elif read(identity)!=artifact:raise ValueError('Controller source does not match registered build')
    checkpoint=ROOT/parent['checkpoint']
    if digest(checkpoint)!=parent['checkpoint_sha256'] or digest(ROOT/parent['resume_checkpoint'])!=parent['resume_sha256']:
        raise ValueError('Original W44 actor or resume differs from its A/B parent')
    setting=dict(arm=a.arm,split=a.split,gait='G0',material=a.material,learner_config=learner,
        model_paths=list(map(str,models)),model_sha256=wm_hashes,legacy_timing_head=False,
        initial_search_candidates=64,search_iterations=2,native_workers=1,
        planning_substeps=int(planning_recipe['planning_substeps']) if planner else None,
        normal_envelope_N=planning_recipe.get('observed_normal_max_N') if planner else None)
    w=dict(arm=a.arm,planner='P0',planner_history='private_reset' if a.arm=='P3-private' else 'full',
        candidate_substep_s=.02,planner_material_backend='original',planner_material_workers=1,
        material='M1',observer_implementation='w57_measured_window',timing_mode='zero',timing_head=None,
        task_mode=a.task,force_model='original force MPC and constrained WBC, event-layer W59 request')
    if a.task in ('short','recovered'):
        from s1_w59_geometry import mechanism_spec
        w['mechanism']=mechanism_spec(a.split)
    job=f'w59-{a.case.lower()}-{build}'
    spec=dict(run_id=RUN,current_task='W59_'+a.task,protocol_revision='W59_original_mechanism_and_branches_v1',
        plan_commit='f11cd7c89654ff74fddc9d70dcb18169c2634094',
        formal_evaluation_protocol='staged_evaluation_30_60_300_600_v3_bidirectional_material_route',
        w54_fixed_policy={'route':'F'},w59=setting,w57=w,material_feature_mode='learned_history',
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
    if a.branch:
        registration=read(a.branch)
        spec['w59_branch']=registration
        if registration.get('source_task') in ('short','recovered'):
            w['mechanism']=registration['mechanism']
    if os.environ.get('W59_RAW_STORE'):spec['w57_raw_root']=os.environ['W59_RAW_STORE']
    return spec,artifact,parent,checkpoint,job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,choices=('A','B'))
    p.add_argument('--arm',required=True,choices=ARMS)
    p.add_argument('--material',choices=('M1','M3'),default='M1')
    p.add_argument('--level',choices=('E30','E60','E300','E600'),default='E30')
    p.add_argument('--task',choices=('original','training','short','recovered','branch'),default='original')
    p.add_argument('--split',choices=('original','training','development','heldout'),default='original')
    p.add_argument('--attempt',required=True)
    p.add_argument('--wm-recipe',type=Path)
    p.add_argument('--wm-checkpoint',action='append',default=[])
    p.add_argument('--learner-config',type=Path)
    p.add_argument('--branch',type=Path)
    a=p.parse_args()
    if (a.task=='branch') != bool(a.branch):
        p.error('branch continuations require their locked registration')
    return episode(a,declare=declaration,small=SMALL)


if __name__=='__main__':
    sys.exit(main())
