#!/usr/bin/env python3
"""R12 full-loop sampling on the recorded controller build and configured host."""
import argparse, copy, json, os, time
from run_s1_go2_r11 import configuration, arguments
from s1_go2_r11_protocol import ROOT
from s1_go2_r11_runtime import run, dump, index_file
from s1_go2_r11_policy import PolicyStream, PolicyMaterial
from mujoco_mpc.agent import Agent
CONFIG = "experiments/terrain_memory/s1-go2-local-readiness-r12.json"
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", required=True, choices=["PPO_RAW_S5","PANN_PPO_S5"])
    p.add_argument("--seed", type=int, default=0, choices=[0])
    p.add_argument("--window-start-unix", "--job-start-unix", dest="window_start_unix", type=float, required=True)
    p.add_argument("--background-job", action="store_true", help="Use the declared job deadline independently of the interactive four-hour window.")
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--resume")
    a=p.parse_args()
    invocation_started_unix_s=time.time()
    contract=json.loads((ROOT/CONFIG).read_text())
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
    workers=contract["resources"]["mpc_workers"]
    pann_weights=ROOT/contract["resources"]["pann_weights_path"]
    execution_mode="background_job" if a.background_job else "interactive_window"
    hard=a.deadline_unix if a.background_job else a.window_start_unix+contract["execution"]["maximum_window_wall_s"]
    deadline=min(a.deadline_unix,hard)
    small=ROOT/contract["outputs"]["compact_root"]
    output=ROOT/contract["outputs"]["large_root"]/"training"/f"{a.group}-seed{a.seed}"
    output.mkdir(parents=True,exist_ok=True)
    dump(small/"training-deadline.json",dict(execution_mode=execution_mode,window_start_unix_s=a.window_start_unix,deadline_unix_s=deadline,hard_deadline_unix_s=hard))
    dump(output/"training-contract.json",contract)
    stream=PolicyStream(ROOT,a.group,a.seed,contract,output,checkpoint=a.resume)
    initial_transitions,initial_updates=stream.samples,stream.backend.update_count
    # A failed initialization may retain an attempt without a newer checkpoint.
    # Continue checkpoint counts, but never reuse that retained attempt's ID.
    prefix=f"train-{a.group}-s{a.seed}-ep"
    stream.episode=max([stream.episode]+[int(x.name[len(prefix):]) for x in output.parent.parent.glob(prefix+"*") if x.is_dir()])
    binary=ROOT/"work/r11-frozen"/build/"agent_server"
    os.environ["LD_LIBRARY_PATH"]=str(binary.parent/"lib")
    os.environ["MJPC_TASKS_DIR"]=str(binary.parent/"runtime-tasks")
    dump(output/"stream-identity.json",dict(group=a.group,seed=a.seed,engineering_candidate=build,
        formal_qualification=False,normalization=stream.normalization,ppo=stream.backend.provenance(),
        workers=workers,pann_weights_path=str(pann_weights),
        protocol_revision=contract["protocol_revision"],initial_condition="nominal",curriculum_enabled=False,
        resume_checkpoint=a.resume,continuous_environment_restored=False,
        execution_mode=execution_mode,window_start_unix_s=a.window_start_unix,deadline_unix_s=deadline,hard_deadline_unix_s=hard))
    last_result=None
    while stream.samples<stream.target and time.time()<deadline:
        session=contract["learning"]["initial_acquisition_session_s"]
        stream.episode+=1
        name=f"train-{a.group}-s{a.seed}-ep{stream.episode:05d}"
        c,r9=configuration(build,"E60",.215,training_duration=session)
        r9["pann"]["weights_path"]=str(pann_weights)
        c.update(run_id=contract["run_id"],stage="S1R12",r12_contract=contract,
            campaign_budget=contract["execution"],r11_training=True,r11_policy=True,
            case_key=None,case_identity=None,predecessor_attempt=None,
            window=dict(task=contract["current_task"],execution_mode=execution_mode,started_unix_s=a.window_start_unix,run_deadline_unix_s=deadline,hard_deadline_unix_s=hard))
        c["outputs"]["root"]=contract["outputs"]["compact_root"]
        args=arguments(name,build,"nominal","training","continuing_acquisition",workers=workers)
        args.config=CONFIG
        args.cohort=contract["protocol_revision"]
        before=stream.samples
        # The retained AgentService permits one Init. A natural new physical
        # session owns a new Agent; ordinary PPO updates keep the live Agent.
        with (output/"persistent-mpc.log").open("a") as log, Agent(task_id="Quadruped Flat",
                server_binary_path=str(binary),extra_flags=[f"--mjpc_workers={workers}"],
                subprocess_kwargs=dict(stdout=log,stderr=log),run_init=False) as agent:
            joint=PolicyMaterial(ROOT,r9,stream,0,session,agent=agent,full_recording=True)
            joint.training_deadline_path=small/"training-deadline.json"
            result=run(args,c,joint)
            last_result=result
        record=dict(episode=stream.episode,attempt=name,controller_build=build,condition="nominal",stage=0,
            session_limit_s=session,transitions=stream.samples-before,reason=result["reason"],
            stage_status=result["stage_status"],simulation_s=result["simulation_s"],wall_s=result["wall_s"],
            completed_round_trips=result["completed_round_trips"],
            complete_session=result["simulation_s"]>=session-1e-7 and result["stage_status"]=="acquisition_boundary",
            promoted=False,curriculum_enabled=False)
        stream.session_records.append(record)
        dump(output/f"episode-{stream.episode:05d}.json",record)
        stream.checkpoint(resume_only=True)
        with index_file(small,c) as idx:
            streams=idx.setdefault("training_streams",{})
            streams[f"{a.group}-seed{a.seed}"]=dict(group=a.group,seed=a.seed,actual_transitions=stream.samples,
                updates=stream.backend.update_count,tail_buffer=len(stream.buffer),episodes=stream.episode,
                physics_s=stream.physics_s,controller_build=build,output=str(output.relative_to(ROOT)),
                checkpoint=str((output/f"resume-state-ep{stream.episode:05d}.pt").relative_to(ROOT)))
            idx.update(current_task=contract["current_task"],protocol_revision=contract["protocol_revision"],status=f"{a.group}_training",
                training_runs=len(streams),training_transitions=sum(x["actual_transitions"] for x in streams.values()),
                training_updates=sum(x["updates"] for x in streams.values()),window=c["window"])
        print(json.dumps(record),flush=True)
        if result["stage_status"] in ("technical_interruption","budget_truncated"):break
    stream.checkpoint(resume_only=True)
    dump(output/"stopped.json",dict(actual_transitions=stream.samples,updates=stream.backend.update_count,
        unupdated_fragment=len(stream.buffer),physics_s=stream.physics_s,session_records=stream.session_records,
        target=stream.target,continuous_environment_snapshot=False,
        invocation_started_unix_s=invocation_started_unix_s,invocation_wall_s=time.time()-invocation_started_unix_s,
        controller_build=build,execution_mode=execution_mode,
        last_stage_status=last_result["stage_status"] if last_result else None,
        last_episode_reason=last_result["reason"] if last_result else None,
        phase_initial_transitions=contract["learning"]["phase_initial_transitions"],
        phase_new_transitions=stream.samples-contract["learning"]["phase_initial_transitions"],
        phase_initial_updates=contract["learning"]["phase_initial_updates"],
        phase_new_updates=stream.backend.update_count-contract["learning"]["phase_initial_updates"],
        initial_transitions=initial_transitions,new_transitions=stream.samples-initial_transitions,
        initial_updates=initial_updates,new_updates=stream.backend.update_count-initial_updates,
        reason="target_reached" if stream.samples>=stream.target else "deadline_or_technical_stop",
        deadline_unix_s=deadline,stopped_unix_s=time.time()))
    stream.log.close()
    stream.learning.close()
if __name__=="__main__":main()
