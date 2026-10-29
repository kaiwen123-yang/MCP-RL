#!/usr/bin/env python3
"""R12 single-stream engineering sampling on the retained W1 executor."""
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
    p.add_argument("--window-start-unix", type=float, required=True)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--resume")
    a=p.parse_args()
    contract=json.loads((ROOT/CONFIG).read_text())
    old=json.loads((ROOT/contract["reuse"]["runtime_physics_ppo_and_task_settings"]).read_text())
    contract["joint_learning"]=copy.deepcopy(old["joint_learning"])
    contract["joint_learning"].update(standard_target_transitions_per_stream=contract["learning"]["first_update_target_per_assigned_stream"],
        cross_stream_wait=False,streams=1,active_streams=[dict(group=a.group,seed=a.seed)],actor_dt_s=contract["learning"]["actor_dt_s"])
    contract["resources"].update(torch_device="cuda",torch_threads=2)
    hard=a.window_start_unix+contract["execution"]["maximum_window_wall_s"]
    deadline=min(a.deadline_unix,hard)
    small=ROOT/contract["outputs"]["compact_root"]
    output=ROOT/contract["outputs"]["large_root"]/"training"/f"{a.group}-seed{a.seed}"
    output.mkdir(parents=True,exist_ok=True)
    dump(small/"training-deadline.json",dict(window_start_unix_s=a.window_start_unix,deadline_unix_s=deadline,hard_deadline_unix_s=hard))
    dump(output/"training-contract.json",contract)
    stream=PolicyStream(ROOT,a.group,a.seed,contract,output,checkpoint=a.resume)
    if a.resume is None:
        prefix=f"train-{a.group}-s{a.seed}-ep"
        stream.episode=max([0]+[int(x.name[len(prefix):]) for x in output.parent.parent.glob(prefix+"*") if x.is_dir()])
    binary=ROOT/"work/r11-frozen/S5-r21/agent_server"
    os.environ["LD_LIBRARY_PATH"]=str(binary.parent/"lib")
    os.environ["MJPC_TASKS_DIR"]=str(binary.parent/"runtime-tasks")
    dump(output/"stream-identity.json",dict(group=a.group,seed=a.seed,engineering_candidate="S5-r21",
        formal_qualification=False,normalization=stream.normalization,ppo=stream.backend.provenance(),
        protocol_revision=contract["protocol_revision"],initial_condition="nominal",curriculum_enabled=False,
        resume_checkpoint=a.resume,continuous_environment_restored=False,
        window_start_unix_s=a.window_start_unix,deadline_unix_s=deadline,hard_deadline_unix_s=hard))
    while stream.samples<stream.target and time.time()<deadline:
        session=contract["learning"]["initial_acquisition_session_s"]
        stream.episode+=1
        name=f"train-{a.group}-s{a.seed}-ep{stream.episode:05d}"
        c,r9=configuration("S5-r21","E60",.215,training_duration=session)
        r9["pann"]["weights_path"]="/home/fanyuan/projects/kaiwen-learning/terrain-memory-study/outputs/terrain_memory/reproduction/material/tm02-memory-study-20261023/selected/B_random1000--ideal-potentials.npz"
        c.update(run_id=contract["run_id"],stage="S1R12",r12_contract=contract,
            campaign_budget=contract["execution"],r11_training=True,r11_policy=True,
            case_key=None,case_identity=None,predecessor_attempt=None,
            window=dict(task="W123",started_unix_s=a.window_start_unix,run_deadline_unix_s=deadline,hard_deadline_unix_s=hard))
        c["outputs"]["root"]=contract["outputs"]["compact_root"]
        args=arguments(name,"S5-r21","nominal","training","continuing_acquisition",workers=5)
        args.config=CONFIG
        args.cohort=contract["protocol_revision"]
        before=stream.samples
        # This r21 AgentService permits one Init. A natural new physical
        # session owns a new Agent; ordinary PPO updates keep the live Agent.
        with (output/"persistent-mpc.log").open("a") as log, Agent(task_id="Quadruped Flat",
                server_binary_path=str(binary),extra_flags=["--mjpc_workers=5"],
                subprocess_kwargs=dict(stdout=log,stderr=log),run_init=False) as agent:
            joint=PolicyMaterial(ROOT,r9,stream,0,session,agent=agent,full_recording=True)
            joint.training_deadline_path=small/"training-deadline.json"
            result=run(args,c,joint)
        record=dict(episode=stream.episode,attempt=name,condition="nominal",stage=0,
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
                physics_s=stream.physics_s,output=str(output.relative_to(ROOT)),
                checkpoint=str((output/f"resume-state-ep{stream.episode:05d}.pt").relative_to(ROOT)))
            idx.update(current_task="W123",protocol_revision=contract["protocol_revision"],status="W123_training",
                training_runs=len(streams),training_transitions=sum(x["actual_transitions"] for x in streams.values()),
                training_updates=sum(x["updates"] for x in streams.values()),window=c["window"])
        print(json.dumps(record),flush=True)
        if result["stage_status"] in ("technical_interruption","budget_truncated"):break
    stream.checkpoint(resume_only=True)
    dump(output/"stopped.json",dict(actual_transitions=stream.samples,updates=stream.backend.update_count,
        unupdated_fragment=len(stream.buffer),physics_s=stream.physics_s,session_records=stream.session_records,
        target=stream.target,continuous_environment_snapshot=False,
        reason="target_reached" if stream.samples>=stream.target else "deadline_or_technical_stop",
        deadline_unix_s=deadline,stopped_unix_s=time.time()))
    stream.log.close()
    stream.learning.close()
if __name__=="__main__":main()
