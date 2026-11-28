"""Frozen native qualification and held-out material/route/disturbance comparisons for W61 E."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"learning"))
from campaign import RUN,SMALL,head_name
from collect import BASE,BASE_REL,SOURCE_REL
from develop import admitted_call
from hybrid import ROOT,SPEC
from s1_go2_r11_policy import PolicyMaterial,PolicyStream
from s1_go2_r11_protocol import case_identity,prerequisite,stable_digest,stage
from s1_go2_r11_runtime import index_file,run
from s1_w59_geometry import mechanism_spec
from s1_w59_learner_critic import atomic_json,file_digest

CONFIG=ROOT/"experiments/terrain_memory/w61/learning/final-evaluator.json"
PLAN=json.loads(CONFIG.read_text())
DATA=RUN/"final-originals"
LEDGER=SMALL/"final-evaluation"
SELF=Path(__file__).resolve()
BACKEND=ROOT/"scripts/s1_w61_final_backend.py"


def read(p):
    return json.loads(p.read_text())


def ident(method,seed,version,case,kind,condition):
    return f"w61-e-final-{head_name(method,seed)}-v{version:03d}-{case.lower()}-{kind.lower()}-{condition.lower()}"


def native(method,seed,version,case,kind,condition):
    name=ident(method,seed,version,case,kind,condition)
    root=LEDGER/head_name(method,seed)
    ck=RUN/head_name(method,seed)/f"policy-{version:03d}.pt"
    source_ref=f"w59-b3-{case.lower()}-original-e30-a01"
    original=read(BASE/BASE_REL/"w59/episodes"/(source_ref+"-resolved.json"))
    source=read(BASE/SOURCE_REL/"jobs"/source_ref/"execution-contract.json")
    parent=read(BASE/BASE_REL/"w56/f"/f"F-{case}-r60/job.json")
    c=copy.deepcopy(original["r12_configuration"])
    for key in ("case_key","case_identity","predecessor_attempt","control_identity"):
        c.pop(key,None)
    c["task"]["command_speed_m_s"]=.215
    c["r11_stage"]=stage(kind) if kind in ("E30","E60") else dict(level="mechanism_"+kind,
        duration_s=PLAN["mechanism"][kind+"_horizon_s"],task_kind="world_cell_revisit",target_cycles=23,qualification_usable=False)
    c["task"]["maximum_episode_s"]=c["r11_stage"]["duration_s"]
    scenario=next((s for s in PLAN["conditions"] if s["id"]==condition),None)
    mode="w61_own_qualification" if scenario is None else kind
    condition_id="nominal" if scenario is None else "w61-"+condition
    if scenario is not None:
        c["single_mission_protocol"]["formal_conditions"].append(dict(id=condition_id,initial_yaw_offset_deg=scenario["yaw_deg"],
            initial_base_linear_velocity_m_s=[0.,scenario["initial_vy_m_s"],0.],initial_base_angular_velocity_rad_s=[0.,0.,0.]))
        task=mechanism_spec("development")
        cell=scenario["target"]["world_key"]
        xy=[(n+.5)*.02 for n in cell]
        task.update(split="W61-independent-"+condition,world_key=cell,fixed_center_m=xy,
            target_strip_x_m=[cell[0]*.02,(cell[0]+1)*.02],target_strip_y_m=[cell[1]*.02,(cell[1]+1)*.02],
            center_park_x_m=xy[0],body_reversal_x_m=[xy[0]-.4,xy[0]+.4])
        c["w57"]["mechanism"]=task
    c["w57"]["task_mode"]=mode
    frozen=dict(study=PLAN["id"],method=method,learning_seed=seed,version=version,case=case,
        checkpoint=str(ck),actor_file_sha256=file_digest(ck),condition=scenario)
    c["w59"].update(learner_config=None,arm="B3",w61_final=frozen)
    c["w54_f"]["w59"]=c["w59"]
    build="W61-E-frozen-"+stable_digest([frozen,file_digest(CONFIG),file_digest(BACKEND)])[:20]
    c.update(controller_build=build,controller_artifact=build,w54_backend_source=str(BACKEND.relative_to(ROOT)),
        evaluation_workers=1,run_id="tm-s1-go2-local-readiness-r12",stage="S1R12")
    c["w54_f"]["identity"]=build
    c["w54_f"]["backend_source"]=c["w54_backend_source"]
    needed=[str(p.relative_to(ROOT)) for p in (BACKEND,SELF,CONFIG,HERE/"learning/collect.py",HERE/"learning/hybrid.py",HERE/"learning/orbits.py")]
    c["w54_execution_sources"]=sorted(set(c["w54_execution_sources"]+needed))
    c["w54_execution_source_names"]=sorted(set(c["w54_execution_source_names"]+[n.removeprefix("scripts/") for n in needed if n.endswith(".py")]))
    c["outputs"]["root"]=str(root.relative_to(ROOT))
    source["w57"]=c["w57"]
    source["w59"]=c["w59"]
    source["w57_raw_root"]=str(DATA)
    source["reuse"].update(active_controller_build=build,controller_artifact=build)
    source["outputs"].update(compact_root=str(root.relative_to(ROOT)),large_root=str(RUN))
    source["learning"].update(phase_id="W61-E-frozen",stream_id="W61-E-frozen")
    source["joint_learning"]=dict(read(ROOT/source["reuse"]["runtime_physics_ppo_and_task_settings"])["joint_learning"],
        standard_target_transitions_per_stream=14737,cross_stream_wait=False)
    c["r12_contract"]=source
    c["window"]=dict(task="W61-E-frozen",execution_mode="frozen_policy_evaluation",started_unix_s=time.time(),
        run_deadline_unix_s=None,hard_deadline_unix_s=None)
    speed=PolicyStream(ROOT,"PANN_PPO_S5",parent["seed"],source,RUN/"final-cap-records"/name,training=False,checkpoint=BASE/parent["checkpoint"])
    r9=read(ROOT/"experiments/terrain_memory/s1-go2-joint.json")
    r9["pann"]["weights_path"]=str(BASE/source["resources"]["pann_weights_path"])
    material=PolicyMaterial(ROOT,r9,speed,0,c["r11_stage"]["duration_s"],full_recording=True)
    case_config=copy.deepcopy(c)
    case_config.pop("window")
    case_config.pop("r12_contract")
    identity,_=case_identity(case_config,"PANN_PPO_S5",parent["seed"],condition_id,BASE/parent["checkpoint"],speed.normalization)
    identity.update(w61_study=PLAN["id"],frozen_actor_sha256=frozen["actor_file_sha256"],
        scenario=condition,kind="prerequisite_original" if scenario is None else kind,workers=1,
        material_source_parameters="original TM02" if scenario is None else scenario["parameters"])
    c["case_identity"]=identity
    c["case_key"]=stable_digest(identity)
    root.mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    if scenario is None:
        with index_file(root,c) as ix: c["predecessor_attempt"]=prerequisite(c,ix,"formal","tm02_anchor")
    assets=BASE/"work/assets"
    args=SimpleNamespace(config=str(CONFIG.relative_to(ROOT)),stage="G1",purpose="formal",condition=condition_id,arm="F",
        mission="single",cohort=PLAN["id"],attempt=name,workers=1,author_source=str(BASE/"work/r11-frozen/S5-r41"),
        original_robot_dir=str(assets/"go2"),original_scene=str(assets/"scene_flat.xml"),
        original_resolved=str(assets/"rigid-T1R-resolved.json"),agent_server=None)
    try:
        result=run(args,c,material)
    finally:
        speed.log.close()
        speed.learning.close()
    import render_s1_go2_r12 as video
    video.SMALL,video.LARGE=root,DATA
    video.render(name)
    movie=DATA/name/"full-mission-material-overlay.mp4"
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries",
        "stream=nb_frames,r_frame_rate,duration,time_base","-of","json",str(movie)],text=True))["streams"][0]
    rec=dict(head=head_name(method,seed),case=case,stage=kind,condition=condition,actor_sha256=frozen["actor_file_sha256"],
        stage_status=result["stage_status"],qualification_usable=result.get("qualification_usable",scenario is None),
        case_key=c["case_key"],physical_s=result["simulation_s"],return_credit=result["w61_frozen_actor"]["original_task_credit"],
        task_success=result.get("success"),qualification=result.get("qualification"),mechanism=result.get("mechanism"),
        true_fps_pts=probe,video_sha256=file_digest(movie),raw=str(DATA/name),used_training_events=0)
    atomic_json(root/(name+"-summary.json"),rec)
    return result


def deliver(name,method,seed):
    from publish_learning import setup,publisher,verified
    receipt=LEDGER/head_name(method,seed)/(name+"-summary.json")
    if not receipt.exists() or verified(name):
        return
    raw=DATA/name
    root=receipt.parent
    setup()
    publisher.publish(name,[("episode",raw),("summary.json",receipt),("result.json",root/(name+".json")),
        ("resolved.json",root/(name+"-resolved.json")),("final-protocol.json",CONFIG)],raw/"full-mission-material-overlay.mp4")


def execute(method,seed,version,case,kind,condition):
    name=ident(method,seed,version,case,kind,condition)
    root=LEDGER/head_name(method,seed)
    compact=root/(name+"-summary.json")
    if compact.is_file():
        deliver(name,method,seed)
        return read(compact)["stage_status"]
    fatal=root/(name+"-technical.json")
    if fatal.exists():
        return "technical_interruption"
    cmd=[sys.executable,"-u",SELF,"native",method,str(seed),str(version),case,kind,condition]
    disk={"E30":2.5,"E60":4.5,"short":12.,"recovered":20.}[kind]
    rc=admitted_call(name,1,cmd,ram=5.,gpu=800,disk=disk)
    if rc or not compact.is_file():
        atomic_json(fatal,dict(attempt=name,exit_code=rc,status="technical_interruption",raw=str(DATA/name),used=0))
        return "technical_interruption"
    deliver(name,method,seed)
    return read(compact)["stage_status"]


def watch():
    LEDGER.mkdir(parents=True,exist_ok=True)
    with (RUN/"final-evaluator.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case in ("A","B"):
            passed=execute("PPO",6001,0,case,"E30","nominal")
            if passed=="passed": execute("PPO",6001,0,case,"E60","nominal")
        seen=set()
        while True:
            for method in [*SPEC["methods"],SPEC["geometry_ablation"]]:
                for seed in SPEC["seeds"]:
                    head=head_name(method,seed)
                    endpoint=SMALL/head/"train-031.json"
                    if head in seen or not endpoint.exists() or read(endpoint).get("used")!=8192:
                        continue
                    for case in ("A","B"):
                        if execute(method,seed,32,case,"E30","nominal")=="passed":
                            execute(method,seed,32,case,"E60","nominal")
                    for kind in ("short","recovered"):
                        for case in ("A","B"):
                            for cond in PLAN["conditions"]: execute(method,seed,32,case,kind,cond["id"])
                    seen.add(head)
            if (SMALL/"lifecycle.json").exists():
                ready={p.parent.name for p in SMALL.glob("*/train-031.json") if read(p).get("used")==8192}
                if ready<=seen:return
            time.sleep(45)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    sub.add_parser("watch")
    n=sub.add_parser("native")
    n.add_argument("method");n.add_argument("seed",type=int);n.add_argument("version",type=int)
    n.add_argument("case");n.add_argument("kind");n.add_argument("condition")
    a=p.parse_args()
    if a.action=="watch":watch()
    else:native(a.method,a.seed,a.version,a.case,a.kind,a.condition)
