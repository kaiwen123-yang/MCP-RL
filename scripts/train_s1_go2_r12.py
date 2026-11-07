#!/usr/bin/env python3
"""R12 full-loop sampling on the recorded controller build and configured host."""
import argparse, copy, json, math, os, time
from run_s1_go2_r11 import configuration, arguments
from s1_go2_r11_protocol import ROOT
from s1_go2_r11_runtime import run, dump, index_file
from s1_go2_r11_policy import PolicyStream, PolicyMaterial
from mujoco_mpc.agent import Agent
from s1_go2_job_limits import goal_based, expired
CONFIG = "experiments/terrain_memory/s1-go2-local-readiness-r12.json"
def known_r25_startup_regression(result,tolerance):
    """One natural-terminal predicate on saved result JSON; no physics changes."""
    evidence=dict(attempt=result.get('attempt'),goal_position_radius_m=tolerance,
        threshold_source='Resolved original task.goal_position_radius_m',
        original_physical_reason=result.get('reason'))
    try:
        reason=result['reason']
        if not reason.startswith(('body_fall','invalid_contact')):
            return dict(evidence,status='not_matched',detail='Not a natural fall/nonfoot terminal')
        transitions=result['transitions'];progress=result['stage_progress']
        phases=[x['phase'] for x in transitions]
        initial=transitions[0]['position_m'][0]
        maximum=result['max_x_m'];final=result['final_qpos'][0]
        loaded=result['material']['loaded_ever_cells'];actual=progress['actual_material_loaded']
        far=progress['far_arrival_s'];returning=progress['return_start_s']
    except (KeyError,IndexError,TypeError,AttributeError) as error:
        return dict(evidence,status='not_evaluable',detail='Missing required result field: '+str(error))
    evidence.update(initial_x_m=initial,max_x_m=maximum,final_x_m=final,
        loaded_ever_cells=loaded,actual_material_loaded=actual,observed_phases=phases,
        far_arrival_s=far,return_start_s=returning)
    numeric=[initial,maximum,final,loaded]+[v for v in (far,returning) if v is not None]
    if (any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in numeric)
            or not isinstance(actual,bool) or any(not isinstance(p,str) for p in phases)):
        return dict(evidence,status='not_evaluable',detail='Nonfinite or unspecified required values')
    later={'far_hold','turn','return','turn_home','home_hold','terminal_stand','complete'}
    matched=('outbound' in phases and not later.intersection(phases)
        and far is None and returning is None and loaded==0 and actual is False
        and maximum-initial<=tolerance and final-initial < -tolerance)
    return dict(evidence,status='matched' if matched else 'not_matched')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", required=True, choices=["PPO_RAW_S5","PANN_PPO_S5"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window-start-unix", "--job-start-unix", dest="window_start_unix", type=float, required=True)
    p.add_argument("--background-job", action="store_true", help="Use the declared background completion policy independently of the interactive window.")
    p.add_argument("--deadline-unix", type=float, help="Legacy explicit wall limit; omitted for goal-based background jobs.")
    p.add_argument("--resume")
    p.add_argument("--contract", default=CONFIG, help="Recorded job contract for an authorized budget continuation.")
    a=p.parse_args()
    invocation_started_unix_s=time.time()
    contract=json.loads((ROOT/a.contract).read_text())
    active=next(x for x in contract["learning"]["active_streams"] if x["group"]==a.group and x["seed"]==a.seed)
    contract["learning"].update(stream_id=active.get("stream_id"),fork_exploration_seed=active.get("fork_exploration_seed"))
    old=json.loads((ROOT/contract["reuse"]["runtime_physics_ppo_and_task_settings"]).read_text())
    # Retain numerical PPO/shared-observation settings, not the old R11 campaign.
    shared=old["joint_learning"]
    contract["joint_learning"]={key:copy.deepcopy(shared[key]) for key in (
        "ppo","actor_action","shared_observation","pann","pann_sha256","pann_features",
        "on_policy_and_independence","bootstrap_rule")}
    contract["joint_learning"].update(
        standard_target_transitions_per_stream=contract["learning"]["target_cumulative_transitions_per_stream"],
        standard_target_total_transitions=contract["learning"]["target_cumulative_transitions_per_stream"],
        cross_stream_wait=False,streams=1,groups=[a.group],seeds=[a.seed],
        active_streams=[dict(group=a.group,seed=a.seed)],actor_dt_s=contract["learning"]["actor_dt_s"],
        shared_controller=f'Engineering candidate {contract["reuse"]["active_controller_build"]}; not formally qualified.',
        start_condition=contract["learning"]["candidate_requires"],curriculum_enabled=False,
        schedule=contract["learning"]["schedule"])
    build=contract["reuse"]["active_controller_build"]
    artifact=contract["reuse"].get("controller_artifact",build)
    workers=contract["resources"]["mpc_workers"]
    pann_weights=ROOT/contract["resources"]["pann_weights_path"]
    execution_mode="background_job" if a.background_job else "interactive_window"
    if goal_based(contract):
        if a.deadline_unix is not None:
            p.error('Goal-based jobs must not be given --deadline-unix')
        hard=deadline=None
    else:
        if a.deadline_unix is None:
            p.error('Legacy/interactive execution requires its explicit --deadline-unix')
        hard=a.deadline_unix if a.background_job else a.window_start_unix+contract["execution"]["maximum_window_wall_s"]
        deadline=min(a.deadline_unix,hard)
    small=ROOT/contract["outputs"]["compact_root"]
    large=ROOT/contract["outputs"]["large_root"]
    output=ROOT/contract["learning"]["training_directory"] if contract["learning"].get("training_directory") else large/"training"/f"{a.group}-seed{a.seed}"
    output.mkdir(parents=True,exist_ok=True)
    deadline_path=ROOT/contract["learning"]["training_deadline_path"]
    dump(deadline_path,dict(execution_mode=execution_mode,window_start_unix_s=a.window_start_unix,deadline_unix_s=deadline,hard_deadline_unix_s=hard))
    dump(output/"training-contract.json",contract)
    stream=PolicyStream(ROOT,a.group,a.seed,contract,output,checkpoint=a.resume)
    initial_transitions,initial_updates=stream.samples,stream.backend.update_count
    # Persist this new phase in its own directory; ordinary resume retains its own RNG and legal tail.
    stream.checkpoint(resume_only=True)
    # A failed initialization may retain an attempt without a newer checkpoint.
    # Continue checkpoint counts, but never reuse that retained attempt's ID.
    prefix=contract["learning"].get("episode_prefix",f"train-{a.group}-s{a.seed}-ep")
    stream.episode=max([stream.episode]+[int(x.name[len(prefix):]) for x in large.glob(prefix+"*") if x.is_dir()])
    binary=ROOT/"work/r11-frozen"/artifact/"agent_server"
    os.environ["LD_LIBRARY_PATH"]=str(binary.parent/"lib")
    os.environ["MJPC_TASKS_DIR"]=str(binary.parent/"runtime-tasks")
    dump(output/"stream-identity.json",dict(group=a.group,seed=a.seed,engineering_candidate=build,controller_artifact=artifact,
        stream_id=stream.stream_id,fork_identity=stream.fork_identity,phase_lineage=stream.phase_lineage,
        training_deadline_path=str(deadline_path.relative_to(ROOT)),
        formal_qualification=False,normalization=stream.normalization,ppo=stream.backend.provenance(),
        workers=workers,pann_weights_path=str(pann_weights),
        protocol_revision=contract["protocol_revision"],initial_condition="nominal",curriculum_enabled=False,
        resume_checkpoint=a.resume,continuous_environment_restored=False,
        controller_version_boundary=contract["learning"]["controller_version_boundary"],
        reward_version=stream.reward_version,initial_tail_buffer=len(stream.buffer),phase_id=stream.phase_id,
        excluded_old_version_transitions=stream.excluded_old_version_transitions,
        controller_migration=stream.controller_migration,
        phase_initial_transitions=contract["learning"]["phase_initial_transitions"],
        phase_initial_updates=contract["learning"]["phase_initial_updates"],
        execution_mode=execution_mode,window_start_unix_s=a.window_start_unix,deadline_unix_s=deadline,hard_deadline_unix_s=hard))
    last_result=None;phase_stop=None
    while stream.samples<stream.target and not expired(deadline):
        session=contract["learning"]["initial_acquisition_session_s"]
        stream.episode+=1
        name=f"{prefix}{stream.episode:05d}"
        c,r9=configuration(build,"E60",.215,training_duration=session,controller_artifact=artifact)
        r9["pann"]["weights_path"]=str(pann_weights)
        c.update(run_id=contract["run_id"],stage="S1R12",r12_contract=contract,
            campaign_budget=contract["execution"],r11_training=True,r11_policy=True,
            case_key=None,case_identity=None,predecessor_attempt=None,
            window=dict(task=contract["current_task"],execution_mode=execution_mode,started_unix_s=a.window_start_unix,run_deadline_unix_s=deadline,hard_deadline_unix_s=hard))
        c["outputs"]["root"]=contract["outputs"]["compact_root"]
        args=arguments(name,build,"nominal","training","continuing_acquisition",workers=workers,controller_artifact=artifact)
        args.config=a.contract
        args.cohort=contract["protocol_revision"]
        before=stream.samples
        # The retained AgentService permits one Init. A natural new physical
        # session owns a new Agent; ordinary PPO updates keep the live Agent.
        diagnostic_env=dict(os.environ)
        episode_output=large/name
        diagnostic_env.update(MJPC_R27_PLAN_RECORD=str(episode_output/"planner-records.jsonl"),
                              MJPC_R27_ACTION_RECORD=str(episode_output/"feedback-actions.jsonl"))
        if build=='S5-r29':
            diagnostic_env.update(MJPC_R29_SNAPSHOT=str(episode_output/"r29-current-plan.bin"),
                                  MJPC_R29_CALL_RECORD=str(episode_output/"planner-calls.jsonl"))
        with (output/"persistent-mpc.log").open("a") as log, Agent(task_id="Quadruped Flat",
                server_binary_path=str(binary),extra_flags=[f"--mjpc_workers={workers}"],
                subprocess_kwargs=dict(stdout=log,stderr=log,env=diagnostic_env),run_init=False) as agent:
            joint=PolicyMaterial(ROOT,r9,stream,0,session,agent=agent,full_recording=True)
            joint.training_deadline_path=deadline_path
            result=run(args,c,joint)
            last_result=result
        record=dict(episode=stream.episode,attempt=name,controller_build=build,controller_artifact=artifact,condition="nominal",stage=0,
            stream_id=stream.stream_id,phase_id=stream.phase_id,
            session_limit_s=session,transitions=stream.samples-before,reason=result["reason"],
            stage_status=result["stage_status"],simulation_s=result["simulation_s"],wall_s=result["wall_s"],
            completed_round_trips=result["completed_round_trips"],
            legal_route=result.get("legal_route"),
            complete_session=result["simulation_s"]>=session-1e-7 and result["stage_status"]=="acquisition_boundary",
            promoted=False,curriculum_enabled=False)
        stream.session_records.append(record)
        dump(output/f"episode-{stream.episode:05d}.json",record)
        resume_checkpoint=stream.checkpoint(resume_only=True)
        if contract['learning'].get('known_r25_startup_regression',{}).get('enabled_for_planned_phase'):
            saved_result=json.loads((small/f'{name}.json').read_text())
            check=known_r25_startup_regression(saved_result,c['task']['goal_position_radius_m'])
            record['startup_regression_check']=check
            dump(output/f'episode-{stream.episode:05d}.json',record)
            if check['status']=='matched':phase_stop=dict(reason='known_r25_startup_regression',evidence=check)
        with index_file(small,c) as idx:
            streams=idx.setdefault("training_streams",{})
            streams[stream.stream_id]=dict(group=a.group,seed=a.seed,actual_transitions=stream.samples,
                stream_id=stream.stream_id,fork_identity=stream.fork_identity,phase_lineage=stream.phase_lineage,
                phase_initial_transitions=stream.phase_initial_samples,
                phase_new_transitions=stream.samples-stream.phase_initial_samples,
                phase_new_updates=stream.backend.update_count-stream.phase_initial_updates,
                updates=stream.backend.update_count,tail_buffer=len(stream.buffer),episodes=stream.episode,
                excluded_old_version_transitions=stream.excluded_old_version_transitions,
                phase_id=stream.phase_id,controller_migration=stream.controller_migration,phase_stop=phase_stop,
                physics_s=stream.physics_s,controller_build=build,output=str(output.relative_to(ROOT)),
                checkpoint=str(resume_checkpoint.relative_to(ROOT)))
            idx.update(current_task=contract["current_task"],protocol_revision=contract["protocol_revision"],status=f"{a.group}_training",
                training_runs=len(streams),training_transitions=sum(x["phase_new_transitions"] if x.get("fork_identity") else x["actual_transitions"] for x in streams.values()),
                training_updates=None,training_updates_by_stream={k:v["updates"] for k,v in streams.items()},
                training_transitions_by_group={group:sum(
                    row["phase_new_transitions"] if row.get("fork_identity") else row["actual_transitions"]
                    for row in streams.values() if row["group"]==group)
                    for group in {row["group"] for row in streams.values()}},
                training_count_semantics="All-group total includes RAW; per-group totals count shared parent history once and each phase's new data once. Branch updates remain separate policies.",
                window=c["window"])
        print(json.dumps(record),flush=True)
        if phase_stop or result["stage_status"] in ("technical_interruption","budget_truncated"):break
    stream.checkpoint(resume_only=True)
    technical_stop=last_result is not None and last_result['stage_status']=='technical_interruption'
    stop_reason=(phase_stop['reason'] if phase_stop else 'technical_interruption' if technical_stop
        else 'target_reached' if stream.samples>=stream.target else 'deadline_reached' if expired(deadline)
        else last_result['reason'] if last_result else 'stopped_without_episode')
    dump(output/"stopped.json",dict(actual_transitions=stream.samples,updates=stream.backend.update_count,
        stream_id=stream.stream_id,fork_identity=stream.fork_identity,phase_lineage=stream.phase_lineage,
        unupdated_fragment=len(stream.buffer),physics_s=stream.physics_s,session_records=stream.session_records,
        target=stream.target,continuous_environment_snapshot=False,
        invocation_started_unix_s=invocation_started_unix_s,invocation_wall_s=time.time()-invocation_started_unix_s,
        controller_build=build,controller_artifact=artifact,execution_mode=execution_mode,phase_id=stream.phase_id,
        reward_version=stream.reward_version,
        excluded_old_version_transitions=stream.excluded_old_version_transitions,
        controller_migration=stream.controller_migration,last_completed_update=stream.last_completed_update,
        phase_stop=phase_stop,
        last_stage_status=last_result["stage_status"] if last_result else None,
        last_episode_reason=last_result["reason"] if last_result else None,
        last_error=last_result.get('error') if last_result else None,
        last_r29_failure=last_result.get('r29_failure') if last_result else None,
        phase_initial_transitions=contract["learning"]["phase_initial_transitions"],
        phase_new_transitions=stream.samples-contract["learning"]["phase_initial_transitions"],
        phase_initial_updates=contract["learning"]["phase_initial_updates"],
        phase_new_updates=stream.backend.update_count-contract["learning"]["phase_initial_updates"],
        initial_transitions=initial_transitions,new_transitions=stream.samples-initial_transitions,
        initial_updates=initial_updates,new_updates=stream.backend.update_count-initial_updates,
        reason=stop_reason,
        deadline_unix_s=deadline,stopped_unix_s=time.time()))
    stream.log.close()
    stream.learning.close()
    if technical_stop:raise SystemExit(2)
if __name__=="__main__":main()
