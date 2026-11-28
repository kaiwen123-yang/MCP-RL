"""One real Go2 M1 trajectory with a W44 benchmark cap and W61 contact-event policy."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hybrid import ACTIVE, CACHE, CONFIG, RESULT, ROOT, SIGMA, SPEC, action_map, joint_logp, key, new_heads, stream, weight_digest
from orbits import FController, response_context
from s1_go2_r11_policy import PolicyMaterial, PolicyStream
from s1_go2_r11_runtime import LEGS, Mission, contact_state
from s1_w59_learner_critic import atomic_json, file_digest
from s1_w59_learning import original_reward
from s1_w59_planning import controller_state

BASE = Path(os.environ.get("W61_W59_ORIGINAL", "/root/autodl-tmp/MCP-RL-w59"))
BASE_REL = Path("results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
SOURCE_REL = Path("outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
DT = .002


def read(path):
    return json.loads(path.read_text())


def append(path, record):
    with path.open("a") as file:
        file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


class EventSession:
    def __init__(self, args, out):
        self.args, self.out = args, out
        self.actor, self.critic = new_heads(args.seed)
        self.actor.eval()
        self.critic.eval()
        if args.weights:
            ck = torch.load(args.weights,map_location="cpu",weights_only=True)
            if (ck["seed"],ck["method"],ck["version"]) != (args.seed,args.method,args.version):
                raise ValueError("Collector identity disagrees with checkpoint")
            self.actor.load_state_dict(ck["actor"])
            self.critic.load_state_dict(ck["critic"])
        self.actor_sha = weight_digest(self.actor)
        self.pending = None
        self.rows = []
        self.requests = 0
        self.cut = 0
        self.finished = False
        self.last_start = None
        self.last_mode = -1
        self.obs_prior = None

    def state(self, ctl,t,mission,mq,grf):
        core = controller_state(ctl,t,mission,mq,grf)
        eligible = [1., float(self.last_mode != 1)]
        prev = [float(self.last_mode == 0), float(self.last_mode == 1)]
        dt = 0. if self.last_start is None else t - self.last_start
        return torch.from_numpy(np.r_[core, prev, dt, eligible].astype(np.float32))

    def settle(self, end, obs, kind):
        if self.pending is None:
            return
        row, start = self.pending
        self.pending = None
        elapsed = float(end[0]-start[0])
        if elapsed <= 0 or kind == "technical":
            self.cut += 1
            return
        terminal = kind in ("success","physical_failure")
        reward,terms = original_reward(start,end,kind=="physical_failure")
        row.update(end_s=float(end[0]), dt_s=elapsed,reward=reward,reward_terms=terms,
            next_value=0. if terminal else float(self.critic(obs).detach()),terminated=terminal,
            truncated=kind=="truncated",end_kind=kind)
        self.rows.append(row)
        append(self.out/"events.jsonl",row)

    @torch.no_grad()
    def decide(self, ctl,t,mission,mq,grf):
        obs = self.state(ctl,t,mission,mq,grf)
        now = ctl.w59_joint.snapshot(t,mission)
        self.settle(now,obs,"ongoing")
        if len(self.rows) >= self.args.quota:
            self.finished = True
            return None,None
        self.obs_prior = obs
        number = self.requests
        source = (self.args.role, "explore", self.args.case, self.args.version, self.args.episode, number)
        rng = stream(self.args.seed,*source)
        rng_hash = hashlib.sha256(rng.get_state().numpy().tobytes()).hexdigest()
        logp,means = self.actor(obs[None])
        mode = int(torch.multinomial(logp[0].exp(),1,generator=rng))
        if self.args.role == "development":
            mode = int(self.args.variant.startswith("hold") and number%2==0 and obs[-1]>.5)
            polarity = 1. if self.args.variant.endswith("plus") else -1.
            raw = torch.ones(4)*polarity
        elif self.args.role == "evaluate":
            mode = int(logp[0].argmax())
            raw = means[0,mode].clone()
        else:
            raw = means[0,mode] + SIGMA*torch.randn(4, generator=rng)
        raw = raw * ACTIVE[mode]
        physical = action_map(mode,raw.numpy())
        self.requests += 1
        row = dict(event_id=f"{self.out.name}/{number:05d}", attempt=self.out.name,case=self.args.case,
            method=self.args.method,seed=self.args.seed,version=self.args.version,role=self.args.role,
            episode=self.args.episode,actor_sha=self.actor_sha,position=self.args.offset+len(self.rows),
            obs=obs.tolist(),mode=mode,raw=raw.tolist(),issued_action=physical.tolist(),
            logp=float(joint_logp(logp,means,torch.tensor([mode]),raw[None])[0]),
            oldlog=logp[0].tolist(),oldmean=means[0].tolist(),value=float(self.critic(obs)),
            rng_key=key(self.args.seed,*source),rng_state_sha256=rng_hash,start_s=float(t),
            response_context=response_context(ctl,mq))
        self.pending = row,now.copy()
        append(self.out/"requests.jsonl",{k:row[k] for k in ("event_id","start_s","mode","issued_action","rng_key","actor_sha")})
        self.last_start, self.last_mode = t, mode
        return mode, physical


def collect(a):
    out = a.out
    out.mkdir(parents=True,exist_ok=False)
    session = EventSession(a,out)
    ref = f"w59-b3-{a.case.lower()}-original-e30-a01"
    original = BASE / SOURCE_REL / ref
    xml = original / "model/G1-project-go2.xml"
    resolved = read(BASE/BASE_REL/"w59/episodes"/(ref+"-resolved.json"))
    c = resolved["r12_configuration"]
    task = dict(c["task"],command_speed_m_s=.215,maximum_episode_s=a.duration)
    source = read(BASE/SOURCE_REL/"jobs"/ref/"execution-contract.json")
    parent = read(BASE/BASE_REL/"w56/f"/f"F-{a.case}-r60/job.json")
    ck = BASE / parent["checkpoint"]
    r9 = read(ROOT/"experiments/terrain_memory/s1-go2-joint.json")
    r9["pann"]["weights_path"] = str(BASE/source["resources"]["pann_weights_path"])
    original_settings = read(ROOT/source["reuse"]["runtime_physics_ppo_and_task_settings"])
    source["joint_learning"] = dict(original_settings["joint_learning"],standard_target_transitions_per_stream=14737,cross_stream_wait=False)
    source["outputs"] = dict(large_root=str(out))
    speed = PolicyStream(ROOT,"PANN_PPO_S5",parent["seed"],source,out/"w44",training=False,checkpoint=ck)
    joint = PolicyMaterial(ROOT,r9,speed,0,a.duration,full_recording=False)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key("home").id)
    if a.role == "evaluate":
        init = SPEC["evaluation_scenarios"][a.case]
        yaw = math.radians(init["initial_yaw_deg"])
        data.qpos[3:7] = [math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
        data.qvel[1] = init["initial_vy_m_s"]
    mujoco.mj_forward(model,data)
    meta = resolved["model"]
    feet = np.array([model.geom(n).id for n in meta["foot_geom_names"]])
    base = model.body(meta["base_body_name"]).id
    allowed = {0} | {model.body(n).id for n in meta["allowed_ground_body_names"]}
    joint.attach(model,data,feet,allowed,out,mujoco.MjModel.from_xml_path(str(xml)))
    ctl = FController(model,data,feet,base,out,c["w54_f"])
    ctl.w59_joint,ctl.session = joint,session
    mission = Mission(task)
    physics = float(model.opt.timestep)
    assert abs(physics-.0002)<1.e-12
    ids = model.actuator_trnid[:,0]
    rate = model.jnt_dofadr[ids]
    start,cpu = time.monotonic(),time.process_time()
    timing,trace,phase,grf = [],[],None,np.zeros((4,3))
    work = 0.
    max_nonfoot = 0.
    peak_impact = []
    reason = "horizon"
    error = None
    where = dict(role=a.role,method=a.method,seed=a.seed,case=a.case,version=a.version,
        offset=a.offset,quota=a.quota,episode=a.episode,variant=a.variant,run=out.name,actor_sha=session.actor_sha,
        source_commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        source_files={p.name:file_digest(p) for p in Path(__file__).parent.glob("*.py")},
        contract_sha256=file_digest(CONFIG),baseline_contract_sha=file_digest(BASE/SOURCE_REL/"jobs"/ref/"execution-contract.json"),
        original=ref,plant_xml_sha=file_digest(xml),w44_sha=file_digest(ck),normalizer=parent["normalization_sha256"],
        referenced_dependencies={p:file_digest(ROOT/p) for p in source["w57_execution_sources"]},
        worker_cpus=sorted(os.sched_getaffinity(0)),torch=torch.__version__,mujoco=mujoco.__version__,
        physical_dt=physics,feedback_dt=DT,deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
    atomic_json(out/"identity.json",where)
    try:
        for n in range(round(a.duration/DT)+1):
            wall = time.perf_counter()
            t = float(data.time)
            mujoco.mj_forward(model,data)
            orientation=Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler("xyz")
            contact,grf,_ = joint.contacts(contact_state(model,data,feet,allowed,True))
            body_goal,heading=mission.update(t,DT,data.qpos[:3],data.qvel[:3].copy(),orientation,data.qvel[3:6].copy(),bool(contact.all()))
            joint.control(t,mission)
            if mission.phase == "complete" or n == round(a.duration/DT):
                reason = "success" if mission.phase=="complete" else "horizon"
                break
            replan = n%10==0 or mission.phase!=phase
            data.mocap_pos[model.body_mocapid[model.body("goal").id]]=body_goal+Rotation.from_euler("z",heading).as_matrix()@model.site("head").pos
            data.mocap_quat[model.body_mocapid[model.body("goal").id]]=[math.cos(heading/2),0,0,math.sin(heading/2)]
            phase=mission.phase
            info=dict(force_boundary_bottom=joint.force_boundary_bottom.copy(),force_N=grf.copy(),velocity_m_s=joint.velocity.copy(),
                instantaneous_stiffness_N_m=ctl.cfg["material_stiffness_N_m"],material_observer=joint.observer)
            wall_ctl=time.perf_counter()
            command=ctl.update(t,data,mission,contact,grf,joint.soft.copy(),info,replan)
            control_ms=(time.perf_counter()-wall_ctl)*1000
            if session.finished:
                reason="batch_cut"
                break
            if replan:
                joint.latest_plan=ctl.latest_plan
            data.ctrl[:]=command
            force_peak=0.
            for _ in range(10):
                old=float(data.time)
                if not joint.before_physics(physics,phase):
                    reason=joint.reason
                    break
                mujoco.mj_step(model,data)
                real,actual,invalid,total,nonfoot,_=joint.contacts(contact_state(model,data,feet,allowed,True),True)
                joint.after_physics(physics,actual)
                joint.observer.completed(old,float(data.time),joint.force_boundary_bottom,joint.velocity,actual[:,2])
                work+=float(np.abs(data.actuator_force*data.qvel[rate]).sum())*physics
                force_peak=max(force_peak,float(actual[:,2].max()))
                max_nonfoot=max(max_nonfoot,float(np.linalg.norm(nonfoot)))
                if invalid:
                    reason="invalid_contact"
                    break
            peak_impact.append(force_peak)
            timing.append([control_ms,(time.perf_counter()-wall)*1000])
            trace.append(np.r_[t,data.time,data.qpos.copy(),data.qvel.copy(),data.ctrl.copy(),data.actuator_force.copy(),
                data.geom_xpos[feet].ravel(),grf.ravel(),ctl.last_force.ravel(),contact,ctl.last_plan["contact"][0],joint.soft,force_peak])
            if n%250==0:
                atomic_json(out/"live.json",dict(pid=os.getpid(),physical_s=float(data.time),requested=session.requests,observed=len(session.rows),
                    requested_quota=a.quota,mode=session.last_mode,phase=phase,wall_s=time.monotonic()-start))
            if max(abs(Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler("xyz")[:2]))>1.2 or data.qpos[2]<.12:
                reason="body_fall"
            if reason!="horizon":
                break
    except Exception:
        error=traceback.format_exc()
        reason="technical"
    t=float(data.time)
    terminal=reason in ("material_domain_exit","body_fall","invalid_contact")
    if terminal:
        joint.legal_route.update(t,mission,data.qpos[:2],joint.force_boundary_bottom,grf)
        joint.legal_route.settle_physical_terminal(t)
    if session.pending is not None:
        if reason == "technical":
            session.cut+=1
        else:
            # Refresh final measured state and legal footprint before bootstrapping a natural cut.
            mq=ctl.model_quantities(data)
            ctl.plan_publication=ctl.timing_query_plan(t,mq,mission)
            successor=session.state(ctl,t,mission,mq,joint.measurements()[3])
            session.settle(joint.snapshot(t,mission),successor,"success" if reason=="success" else "physical_failure" if terminal else "truncated")
    np.savez_compressed(out/"trajectory.npz",control=np.asarray(trace,np.float32),
        fields="t,post_t,qpos19,qvel18,u12,torque12,foot_xyz12,grf12,planned_force12,contact4,planned_contact4,material_mask4,peak_Fz")
    atomic_json(out/"swing.json",ctl.applied_trajectories)
    result=dict(**where,reason=reason,error=error,physical_s=t,physical_steps=joint.physics_steps,requested=session.requests,
        observed=len(session.rows),used=0,open_cut=session.cut,progress=joint.legal_route.progress,
        physical_failure=terminal,complete=reason=="success",absolute_mechanical_work_J=work,max_nonfoot_force_N=max_nonfoot,
        force_peak_percentiles=np.percentile(peak_impact,[50,95,99]).tolist() if peak_impact else [],
        control_full_ms_p50_p95_p99=np.percentile(timing,[50,95,99],axis=0).tolist() if timing else [],
        wall_s=time.monotonic()-start,cpu_h=(time.process_time()-cpu)/3600,max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),stance_holds=sum(row["mode"]==1 for row in session.rows),
        issued_swings=len(ctl.applied_trajectories),terminal_phase=mission.phase,physical_go2=True)
    joint.flush()
    joint.events_file.close()
    joint.failures_file.close()
    speed.log.close()
    speed.learning.close()
    atomic_json(out/"result.json",result)
    print(json.dumps({k:result[k] for k in ("run","reason","physical_s","requested","observed","wall_s","error")}),flush=True)
    if error:
        raise RuntimeError(error)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    p.add_argument("--method",required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--case",choices=["A","B"],required=True)
    p.add_argument("--role",choices=["development","train","evaluate"],required=True)
    p.add_argument("--variant",default="")
    p.add_argument("--version",type=int,default=0)
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--offset",type=int,default=0)
    p.add_argument("--quota",type=int,default=128)
    p.add_argument("--duration",type=float,default=60.)
    collect(p.parse_args())
