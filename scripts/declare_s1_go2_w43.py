#!/usr/bin/env python3
"""Declare W43 from the retained own W42 parents; does not launch physics."""
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/root/autodl-tmp/MCP-RL')
CONFIG=ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json'
def main():
    c=json.loads(CONFIG.read_text())
    if c['current_task']=='W43_OPTIMIZER_LIFECYCLE_JOINT':
        raise RuntimeError('W43 already declared; use its own job/clock rather than declaring again.')
    large=Path(c['outputs']['large_root'])
    parents={};active=[]
    for suffix,seed in [('a',1),('b',2)]:
        old=f'w42-{suffix}';new=f'w43-{suffix}'
        directory=large/'training'/old/f'PANN_PPO_S5-seed{seed}'
        progress=json.loads((ROOT/directory/'progress.json').read_text())
        source=Path(progress['checkpoint'])
        relative=source.relative_to(ROOT)
        saved=torch.load(ROOT/relative,map_location='cpu',weights_only=False)
        if saved['stream_id']!=old or saved['buffer'] or saved['controller_build']!='S5-r31':
            raise ValueError('Expected own completed W42 zero-tail parent; inspect actual checkpoint.')
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
            phase_lineage=dict(status='explicit_W43_phase',parent_stream_id=old,parent_phase_id=saved['phase_id'],
                parent_phase_lineage=saved['phase_lineage'],current_stream_id=new,current_phase_id=None,
                controller_build='S5-r32',reward_version=saved['reward_version']))
        active.append(dict(stream_id=new,group=saved['group'],seed=seed,parent_stream_id=old,
            training_directory=str(newdir),parent_map_key=new,target_transitions=saved['samples']+512,
            target_updates=saved['updates']+2))
    c.update(current_task='W43_OPTIMIZER_LIFECYCLE_JOINT',protocol_revision='W43_r32_problem_epoch_native_search_v20')
    c['reuse'].update(active_controller_build='S5-r32',controller_artifact='S5-r32',
        active_controller_specification='experiments/terrain_memory/s1-go2-r11-S5-r32.json',
        controller_artifact_specification='experiments/terrain_memory/s1-go2-r11-S5-r32.json',
        active_controller_status='normally_built_saved_inputs_pending',controller_artifact_status='normally_built_saved_inputs_pending',
        implementation_base_controller_build='S5-r31',parent_controller_build='S5-r31',
        checkpoint_parent_controller_build='S5-r31',latest_executed_controller_build='S5-r31',
        latest_executed_controller_status='W42_A_B_completed_13713_53_tail0',
        parent_controller_specification='experiments/terrain_memory/s1-go2-r11-S5-r31.json',
        implementation_base_controller_specification='experiments/terrain_memory/s1-go2-r11-S5-r31.json',
        rule='W43 problem identity, damping/rate and stopped-epoch lifecycle on r31 forward/derivatives; own W42 parents, no new fork/reseed.',
        source_reference_commit='8ba7fd0')
    from start_s1_go2_r12_tmux import phase_metadata
    boundaries={a['stream_id']:phase_metadata(parents[a['stream_id']],a,'S5-r32','S5-r32') for a in active}
    c['learning'].update(parent_map=parents,active_streams=active,
        candidate_requires='Normal r32 build and W42 saved t0/outbound/4s input evidence, then push and direct joint training. Entire phase512/2, no automatic expansion.',
        target_cumulative_transitions_per_stream=active[0]['target_transitions'],
        target_cumulative_updates_per_stream=active[0]['target_updates'],
        additional_transitions_from_reviewed_checkpoint=512,additional_updates_from_reviewed_checkpoint=2,
        phase_initial_transitions=parents['w43-a']['parent_transitions'],phase_initial_updates=parents['w43-a']['parent_updates'],
        schedule='Each own W42 zero-tail parent +512/+2; separate optimizer/RNG/history.',
        unfinished_buffer_rule='New W43 own tail0; ordinary resume preserves its own valid tail.',
        controller_change_during_current_window='Frozen S5-r32; no hot swap.',
        controller_boundary_rule='Own W42 r31 -> W43 r32 at tail0; no new exclusions, fork or reseed.',
        controller_version_boundary=dict(parent_map_source='learning.parent_map',by_stream=boundaries),
        same_controller_continuation=dict(enabled=True,scope='Ordinary W43 resume preserves own frozen job, clock, RNG and tail.',
            ancestor_fork_identity_unchanged=True,current_identity_source='phase_lineage plus current job/stream/phase'),
        next_runtime_work=['Complete normal build and saved-input evidence, push, launch declared W43 streams.'])
    c['resources'].update(mpc_workers=10,project_root_recommendation=str(ROOT),
        cpu_budget_rule='Two 10-worker CPU streams, Torch/BLAS1; preserve all existing GPU jobs. Native build6 before training.',
        workers='New W43 defaults10, explicitly allowed10/11/12; ordinary resume retains frozen workers.',
        environment_policy='Reuse D57 venv and immutable assets; independent mutable upstream/build in W43.',
        hardware_source='2026-11-10 live cgroup25CPU/120GiB; other GPU training PIDs722104,722241 preserved; G5 owns2hostCPUs.',
        throughput_rule='Per-stream new transitions/invocation wall; separate historical parent counts; no realtime claim.')
    c['resources']['future_worker_policy'].update(default_per_stream=10,authorized_per_stream=[10,11,12],
        authorized_on='2026-11-10',scope='New W43 only; no hot change to existing jobs.',
        launcher_option='--workers10|11|12 for declared new W43 jobs',basis='User W43 request; preserve GPU host capacity.')
    settings=c['control_and_evaluation']['post_training_evaluation']
    settings.update(purpose='Own final new W43 update E30 then same-case eligible E60.',solver_workers=10,streams=active,
        checkpoint_selection='Generated per stream from its actual parent and target; own final new r32 update only.',
        result_reporting='Report original stage criteria and bidirectional_material_roundtrips_v1; raw laps and survival do not qualify.')
    settings['no_new_update']['detail']='No own update newer than actual parent; do not evaluate parent.'
    c['execution'].update(includes=['Isolated r32 implementation/build and specified real saved inputs',
        'Two own-parent W43 continuations; final update evaluation; compact and requested raw publication'],
        allowed_chain=['r31_build_evidence_push','declared_W43_joint_training','own_final_E30_eligible_E60','compact_then_raw'],
        unfinished_task_rule='Continue exact own W43 job; preserve active GPU root and all old CPU results.',
        default_reads=['AGENTS.md','docs/TERRAIN_MEMORY_PLAN.md','docs/S1_W43_CPU_PLAN.md','docs/S1_W43_START_PROMPT.md'])
    c['execution']['background_job'].update(session_default='s1-w43',
        training_clock_path=str(large/'w43-training-clock.json'),parallel_layout=active,
        current_continuation='W43 own-parent+512/+2',default_maximum_wall_s=None,wall_deadline_enabled=False)
    c['outputs']['automatic_publication']['default_postprocess_wall_s']=None
    c['outputs']['isolation']=dict(root=str(ROOT),protected_gpu_root=str(OLD),
        shared_read_only=['work/envs/d57','work/assets','work/upstreams/mujoco_mpc','work/r11-frozen/S5-r31',
            str(large/'training/w42-a'),str(large/'training/w42-b')],
        private=['work/terrain-memory/upstreams/mujoco_mpc-go2','work/r11-frozen/S5-r32','new jobs/clocks/outputs'],
        network_mode='academic',cleanup=False)
    CONFIG.write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps({a['stream_id']:dict(parent=parents[a['stream_id']]['resume_checkpoint'],
         initial=parents[a['stream_id']]['parent_transitions'],target=a['target_transitions'],
         updates=a['target_updates']) for a in active},indent=2))
if __name__=='__main__':main()
