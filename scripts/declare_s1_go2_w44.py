#!/usr/bin/env python3
"""Declare W44 from exact completed W43 parents after the selected r33 evidence is published."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/root/autodl-tmp/MCP-RL')
CONFIG=ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json'
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-mechanism',required=True,choices=('S','N','P'))
    parser.add_argument('--evidence',type=Path,required=True,help='Repository-relative four-problem comparison result to publish with this declaration before launch.')
    args=parser.parse_args()
    evidence=ROOT/args.evidence
    json.loads(evidence.read_text())
    identity=json.loads((ROOT/'work/r11-frozen/S5-r33/identity.json').read_text())
    specification=json.loads((ROOT/'experiments/terrain_memory/s1-go2-r11-S5-r33.json').read_text())
    if identity['build']!='S5-r33' or specification['configuration_overrides']['r33_optimizer_mechanism']['selected_mechanism']!=args.selected_mechanism:
        raise ValueError('Declaration must match the built r33 artifact and selected specification.')
    c=json.loads(CONFIG.read_text())
    if c['current_task']=='W44_MOTION_OPTIMIZATION_JOINT':
        raise RuntimeError('W44 already declared; use its own job/clock rather than declaring again.')
    large=Path(c['outputs']['large_root'])
    parents={};active=[]
    for suffix,seed in [('a',1),('b',2)]:
        old=f'w43-{suffix}';new=f'w44-{suffix}'
        directory=large/'training'/old/f'PANN_PPO_S5-seed{seed}'
        episode=410 if suffix=='a' else 398
        relative=directory/f'resume-state-ep{episode:05d}.pt'
        source=ROOT/relative
        saved=torch.load(ROOT/relative,map_location='cpu',weights_only=False)
        if (saved['stream_id']!=old or saved['buffer'] or saved['controller_build']!='S5-r32'
                or saved['samples']!=14225 or saved['updates']!=55 or saved['episode']!=episode
                or saved['seed']!=seed or saved['excluded_old_version_transitions']!=145
                or saved['phase_id']!=('job-20261110T035658Z-w43-a' if suffix=='a' else 'job-20261110T035700Z-w43-b')):
            raise ValueError('Expected own completed W43 zero-tail parent; inspect actual checkpoint.')
        newdir=large/'training'/new/f'PANN_PPO_S5-seed{seed}'
        parents[new]=dict(parent_stream_id=old,parent_job=saved['phase_id'],parent_phase_id=saved['phase_id'],
            group=saved['group'],seed=seed,parent_controller_build=saved['controller_build'],
            parent_controller_artifact=saved['controller_artifact'],parent_reward_version=saved['reward_version'],
            resume_checkpoint=str(relative),resume_release_url=None,original_source=str(source),
            availability='retained_on_D57_read_only_reference',parent_training_directory=str(directory),
            training_directory=str(newdir),parent_episode=saved['episode'],parent_transitions=saved['samples'],
            parent_updates=saved['updates'],parent_completed_update_transitions=saved['updates']*256,
            parent_tail=0,excluded_old_version_transitions=saved['excluded_old_version_transitions'],
            new_exclusions=0,ancestor_fork_identity=saved['fork_identity'],
            restore=['actor','critic','optimizer','RNG','normalization','historical_counts'],
            new_fork=False,reseed_after_restore=False,parent_directory_read_only=True,
            phase_lineage=dict(status='explicit_W44_phase',parent_stream_id=old,parent_phase_id=saved['phase_id'],
                parent_phase_lineage=saved['phase_lineage'],current_stream_id=new,current_phase_id=None,
                controller_build='S5-r33',reward_version=saved['reward_version']))
        active.append(dict(stream_id=new,group=saved['group'],seed=seed,parent_stream_id=old,
            training_directory=str(newdir),parent_map_key=new,target_transitions=saved['samples']+512,
            target_updates=saved['updates']+2))
    c.update(current_task='W44_MOTION_OPTIMIZATION_JOINT',protocol_revision='W44_r33_motion_optimization_v21')
    c['reuse'].update(active_controller_build='S5-r33',controller_artifact='S5-r33',
        active_controller_specification='experiments/terrain_memory/s1-go2-r11-S5-r33.json',
        controller_artifact_specification='experiments/terrain_memory/s1-go2-r11-S5-r33.json',
        active_controller_status='normally_built_four_problem_comparison_completed',controller_artifact_status='normally_built_four_problem_comparison_completed',
        selected_mechanism=args.selected_mechanism,mechanism_evidence=str(args.evidence),
        controller_binary_sha256=identity['binary_sha256'],
        implementation_base_controller_build='S5-r32',parent_controller_build='S5-r32',
        checkpoint_parent_controller_build='S5-r32',latest_executed_controller_build='S5-r32',
        latest_executed_controller_status='W43_A_B_completed_14225_55_tail0_excluded145',
        parent_controller_specification='experiments/terrain_memory/s1-go2-r11-S5-r32.json',
        implementation_base_controller_specification='experiments/terrain_memory/s1-go2-r11-S5-r32.json',
        rule='W44 selected one motion-optimization mechanism from four saved native problems; own exact W43 parents, no new fork/reseed.',
        source_reference_commit='abd009238304518158edc816e9a9530e2519dcac')
    from start_s1_go2_r12_tmux import phase_metadata
    boundaries={a['stream_id']:phase_metadata(parents[a['stream_id']],a,'S5-r33','S5-r33') for a in active}
    c['learning'].update(parent_map=parents,active_streams=active,
        candidate_requires='Normal r33 build and common-context R32/S/N/P four-problem evidence, committed identity and declaration, then direct joint training. Entire phase512/2.',
        phase_count_rule='Per-stream own W43 parent +512/+2 to14737/57; do not sum branch updates into one policy.',
        target_cumulative_transitions_per_stream=active[0]['target_transitions'],
        target_cumulative_updates_per_stream=active[0]['target_updates'],
        additional_transitions_from_reviewed_checkpoint=512,additional_updates_from_reviewed_checkpoint=2,
        phase_initial_transitions=parents['w44-a']['parent_transitions'],phase_initial_updates=parents['w44-a']['parent_updates'],
        schedule='Each own W43 zero-tail parent +512/+2; separate optimizer/RNG/history.',
        unfinished_buffer_rule='New W44 own tail0; ordinary resume preserves its own valid tail.',
        controller_change_during_current_window='Frozen S5-r33; no hot swap.',
        controller_boundary_rule='Own W43 r32 -> W44 r33 at tail0; no new exclusions, fork or reseed.',
        controller_version_boundary=dict(parent_map_source='learning.parent_map',by_stream=boundaries),
        same_controller_continuation=dict(enabled=True,scope='Ordinary W44 resume preserves own frozen job, clock, RNG and tail.',
            ancestor_fork_identity_unchanged=True,current_identity_source='phase_lineage plus current job/stream/phase'),
        next_runtime_work=['Push implementation, evidence and this exact-parent declaration, then launch W44 streams.'])
    c['resources'].update(mpc_workers=10,project_root_recommendation=str(ROOT),
        cpu_budget_rule='Two 10-worker CPU streams, Torch/BLAS1; preserve all existing GPU jobs. Native build6 before training.',
        workers='New W44 defaults10, explicitly allowed10/11/12; ordinary resume retains frozen workers.',
        environment_policy='Reuse D57 venv and immutable assets; independent mutable upstream/build in W44.',
        hardware_source='2026-11-10 live cgroup25CPU/120GiB; other GPU training PIDs722104,722241 preserved; G5 owns2hostCPUs.',
        throughput_rule='Per-stream new transitions/invocation wall; separate historical parent counts; no realtime claim.')
    c['resources']['future_worker_policy'].update(default_per_stream=10,authorized_per_stream=[10,11,12],
        authorized_on='2026-11-10',scope='New W44 only; no hot change to existing jobs.',
        launcher_option='--workers10|11|12 for declared new W44 jobs',basis='User W44 request; preserve GPU host capacity.')
    settings=c['control_and_evaluation']['post_training_evaluation']
    settings.update(purpose='Own final update57 E30 then same-case eligible E60/E300/E600.',
        levels=['E30','E60','E300','E600'],automatically_run_E300_E600=True,require_declared_target_update=True,
        later_long_evaluation='Same fixed update57 and same case qualify sequentially30->60->300->600; legal outbound,return,roundtrip each>=1/5/10 for E60/300/600.',solver_workers=10,streams=active,
        checkpoint_selection='Generated per stream from its actual parent and target; own final new r33 update57 only.',
        result_reporting='Report original stage criteria and bidirectional_material_roundtrips_v1; raw laps and survival do not qualify.')
    settings['no_new_update']['detail']='No own update newer than actual parent; do not evaluate parent.'
    c['execution'].update(includes=['Isolated r33 implementation/build and four common-context saved problems',
        'Two own-parent W44 continuations; final update evaluation; compact and requested raw publication'],
        allowed_chain=['r33_build_four_problem_evidence_push','declared_W44_joint_training','own_final_update57_E30_eligible_E60_E300_E600','compact_then_raw'],
        unfinished_task_rule='Continue exact own W44 job; preserve active GPU root and all old CPU results.',
        default_reads=['AGENTS.md','docs/TERRAIN_MEMORY_PLAN.md','docs/S1_W44_CPU_PLAN.md','docs/S1_W44_START_PROMPT.md'])
    c['execution']['background_job'].update(session_default='s1-w44',
        training_clock_path=str(large/'w44-training-clock.json'),parallel_layout=active,
        current_continuation='W44 own-parent+512/+2',default_maximum_wall_s=None,wall_deadline_enabled=False,
        supervision='Existing episode/result/checkpoint records; no extra monitoring service or new stop predicate.',
        stop_conditions=['Own declared sample/update target','technical interruption','explicit user stop'],
        ordinary_physical_failure='Save the original physical terminal and continue the next natural episode; no generic failure-count cap.',
        post_training='Own update57 fixed E30 then same-case eligible E60/E300/E600, followed by compact and requested raw publication.',
        training_clock='New W44 shared clock established once by launcher; ordinary resume retains it.',
        launch_identity=dict(declared_unix_s=time.time(),root=str(ROOT),streams=[a['stream_id'] for a in active],
            workers_per_stream=10,network_mode='clash',deadline_unix_s=None,
            controller_binary_sha256=identity['binary_sha256'],selected_mechanism=args.selected_mechanism,
            clock_created_once_by_launcher=True,existing_gpu_preserved=True))
    c['outputs']['automatic_publication']['default_postprocess_wall_s']=None
    c['outputs']['isolation']=dict(root=str(ROOT),protected_gpu_root=str(OLD),
        shared_read_only=['work/envs/d57','work/assets','work/upstreams/mujoco_mpc','work/r11-frozen/S5-r32',
            str(large/'training/w43-a'),str(large/'training/w43-b')],
        private=['work/terrain-memory/upstreams/mujoco_mpc-go2','work/r11-frozen/S5-r33','new jobs/clocks/outputs'],
        network_mode='clash',cleanup=False)
    c['control_and_evaluation'].update(current_window='W44 frozen r33 joint continuation; both own parents +512/+2.',
        next_control_question='Native motion directions and actual legal traversal under the one selected r33 mechanism.',
        comparison_after_migration='Same controller on two own W43 branches; shared ancestry, not independent-from-scratch causal arms.')
    c['outputs']['automatic_publication'].update(diagnostic_implementation_status='Existing complete compact-then-requested-raw pipeline; r33 event snapshots included.',
        compact_export_implementation_status='Existing finisher publishes every new transition/value/PANN and rollout before requested raw queue.')
    c['execution']['mechanism_selection']=dict(selected_mechanism=args.selected_mechanism,evidence=str(args.evidence),
        declaration_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        controller_artifact='S5-r33',binary_sha256=identity['binary_sha256'])
    CONFIG.write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps({a['stream_id']:dict(parent=parents[a['stream_id']]['resume_checkpoint'],
         initial=parents[a['stream_id']]['parent_transitions'],target=a['target_transitions'],
         updates=a['target_updates']) for a in active},indent=2))
if __name__=='__main__':main()
