#!/usr/bin/env python3
"""Finish the authorized W40 A boundary, declare real W41 parents, then launch.
No retry of failed physics and no wall-clock deadline. At most two main MPC
streams run: W41 B can overlap W40 A final evaluation; W41 A waits for that
original evaluation process to finish.
"""
import copy,fcntl,json,subprocess,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json'
LARGE=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
A=LARGE/'job-20261106T103028Z-w40-a'
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def run(*args):return subprocess.run(args,cwd=ROOT,check=True)
def main():
    dest=LARGE/'w41-launch.json';record=dict(status='waiting_W40_A_goal',deadline_unix_s=None,publication_network_mode='clash',launched_streams=[])
    write(dest,record)
    while not (A/'training-exit-code.txt').exists():time.sleep(20)
    if int((A/'training-exit-code.txt').read_text())!=0:
        record.update(status='blocked_by_W40_A_technical_failure',reason='Original A did not complete normally; preserve its newest checkpoint and first failure input.');write(dest,record);return
    c=read(CONFIG)
    if c['current_task'].startswith('W41_'):raise RuntimeError('W41 already declared; inspect its own clock/jobs instead of declaring twice')
    parents={};active=[]
    for suffix,seed in [('a',1),('b',2)]:
        old=f'w40-{suffix}';new=f'w41-{suffix}';directory=LARGE/'training'/old/f'PANN_PPO_S5-seed{seed}'
        progress=read(directory/'progress.json');checkpoint=Path(progress['checkpoint']);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['buffer'] or saved['samples']<9617 or saved['updates']<37:raise ValueError('W41 needs each own actual completed zero-tail parent')
        newdir=LARGE/'training'/new/f'PANN_PPO_S5-seed{seed}'
        parents[new]=dict(parent_stream_id=old,parent_job=saved['phase_id'],parent_phase_id=saved['phase_id'],
            group=saved['group'],seed=seed,parent_controller_build=saved['controller_build'],parent_controller_artifact=saved['controller_artifact'],
            parent_reward_version=saved['reward_version'],resume_checkpoint=str(checkpoint.relative_to(ROOT)),resume_release_url=None,
            availability='retained_on_D57',parent_training_directory=str(directory.relative_to(ROOT)),training_directory=str(newdir.relative_to(ROOT)),
            parent_episode=saved['episode'],parent_transitions=saved['samples'],parent_updates=saved['updates'],
            parent_completed_update_transitions=saved['updates']*256,parent_tail=0,excluded_old_version_transitions=saved['excluded_old_version_transitions'],
            new_exclusions=0,ancestor_fork_identity=saved['fork_identity'],restore=['actor','critic','optimizer','RNG','normalization','historical counts'],
            new_fork=False,reseed_after_restore=False,parent_directory_read_only=True,
            phase_lineage=dict(status='explicit_W41_phase',parent_stream_id=old,parent_phase_id=saved['phase_id'],
                parent_phase_lineage=saved['phase_lineage'],current_stream_id=new,current_phase_id=None,
                controller_build='S5-r30',reward_version=saved['reward_version']))
        active.append(dict(stream_id=new,group=saved['group'],seed=seed,parent_stream_id=old,training_directory=str(newdir.relative_to(ROOT)),
            parent_map_key=new,target_transitions=saved['samples']+2048,target_updates=saved['updates']+8))
    c.update(current_task='W41_MATCHED_BOUND_RICCATI_JOINT',protocol_revision='W41_r30_matched_bound_riccati_v18')
    c['reuse'].update(active_controller_build='S5-r30',active_controller_specification='experiments/terrain_memory/s1-go2-r11-S5-r30.json',
        active_controller_status='normally_built_ready_for_joint_training',controller_artifact='S5-r30',
        controller_artifact_specification='experiments/terrain_memory/s1-go2-r11-S5-r30.json',controller_artifact_status='normally_built',
        implementation_base_controller_build='S5-r29-qp2',parent_controller_build='S5-r29',
        rule='Matched instantaneous-bound k/K/value recursion; actual saturated ten-substep physics retained. Own W40 parents only.')
    c['learning'].update(parent_map=parents,active_streams=active,
        candidate_requires='Normally built r30 source/spec and actual own parent map pushed; direct joint training without admission tests.',
        target_cumulative_transitions_per_stream=active[0]['target_transitions'],target_cumulative_updates_per_stream=active[0]['target_updates'],
        additional_transitions_from_reviewed_checkpoint=2048,additional_updates_from_reviewed_checkpoint=8,
        phase_initial_transitions=parents['w41-a']['parent_transitions'],phase_initial_updates=parents['w41-a']['parent_updates'],
        schedule='Each own latest W40 zero-tail parent +2048 transitions/+8 updates; preserve separate optimizer/RNG/fork ancestry.',
        unfinished_buffer_rule='New W41 starts at own tail0; ordinary resume retains own valid tail.',
        controller_change_during_current_window='Frozen S5-r30; no hot swap.',controller_boundary_rule='Own W40 r29 artifact -> r30 at tail0; no new fork, exclusions or reseed.',
        phase_count_rule='Per-stream own parent +2048/+8; do not sum branch updates into one policy.')
    c['resources'].update(mpc_workers=12,cpu_budget_rule='Two main12-worker streams, Torch/BLAS1, GPUhost2; native build6 under current allowance, XLAcompile2.')
    c['control_and_evaluation']['post_training_evaluation']['solver_workers']=12
    c['execution']['background_job'].update(session_default='s1-w41',training_clock_path=str((LARGE/'w41-training-clock.json').relative_to(ROOT)),
        parallel_layout=active,current_continuation='W41 own-parent +2048/+8, no wall deadline',default_maximum_wall_s=None,wall_deadline_enabled=False)
    with (SMALL/'compact-finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write(CONFIG,c)
        run('git','add',str(CONFIG.relative_to(ROOT)))
        run('git','commit','-m','Declare W41 own zero-tail parents and 2048-transition continuations','--',str(CONFIG.relative_to(ROOT)))
        run('git','fetch','origin','main');run('git','merge','--no-edit','origin/main');run('git','push','origin','main')
    record.update(status='parents_published',parent_map=parents);write(dest,record)
    run(str(ROOT/'work/envs/d57/bin/python'),'scripts/start_s1_go2_r12_tmux.py','--new-phase','--no-wall-deadline','--stream','w41-b','--session','s1-w41-b','--network-mode','clash')
    record['launched_streams'].append('w41-b');record['status']='W41_B_running_waiting_A_evaluation';write(dest,record)
    while not (A/'evaluation-exit-code.txt').exists():time.sleep(20)
    # A's failed physical E30 blocks only its E60, not the authorized new phase.
    run(str(ROOT/'work/envs/d57/bin/python'),'scripts/start_s1_go2_r12_tmux.py','--new-phase','--no-wall-deadline','--stream','w41-a','--session','s1-w41-a','--network-mode','clash')
    record['launched_streams'].append('w41-a');record['status']='both_W41_streams_launched';write(dest,record)
if __name__=='__main__':main()
