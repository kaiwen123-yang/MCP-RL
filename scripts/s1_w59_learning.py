"""Three-dimensional event actor, causal on-policy ledger and original task settlement."""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from s1_w59_learner_critic import (PHASES, SPEC, TrueTaskCritic, atomic_json, atomic_weights,
                                   file_digest, json_rows, task_observation)
from s1_w59_model_fields import MASS, NS, STATE

LOW = np.asarray([-.05, .04, -.10], dtype=np.float64)
HIGH = np.asarray([.05, .08, .10], dtype=np.float64)
NOMINAL = np.asarray([0., .05, 0.], dtype=np.float64)
SCALE = np.asarray([.05, .03, .10], dtype=np.float64)
BATCH_PER_CASE = 128
TERMINAL_FAILURES = ("material_domain_exit", "body_fall", "invalid_contact")


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def checkpoint(run, version):
    return run / f"policy-{version:03d}.pt"


def batch_file(run, version, case):
    return run / f"events-{version:03d}-{case}.jsonl"


def read_json(path):
    return json.loads(path.read_text())


def project_tensor(base, raw):
    low, high, scale = (torch.as_tensor(a, dtype=base.dtype, device=base.device) for a in (LOW, HIGH, SCALE))
    return (base + scale * raw.tanh()).clamp(low, high)


class RequestActor(nn.Module):
    def __init__(self):
        super().__init__()
        # Signed asinh preserves the sign of SI quantities and compresses large force/clock channels.
        self.net = nn.Sequential(nn.Linear(NS + 9, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        for key, array in (("lower", LOW), ("upper", HIGH), ("nominal", NOMINAL), ("scale", SCALE),
                           ("std", SPEC["actor"]["exploration_raw_std"])):
            self.register_buffer(key, torch.tensor(array, dtype=torch.float32))

    def forward(self, state, base):
        features = torch.cat((state.asinh(), (base-self.nominal)/self.scale,
                              (base-self.lower)/self.scale, (self.upper-base)/self.scale), dim=-1)
        return self.net(features)

    def logp(self, mean, raw):
        return torch.distributions.Normal(mean, self.std).log_prob(raw).sum(-1)


def original_reward(start, end, failed):
    d = end - start
    terms = dict(legal_progress=5*float(d[1]), elapsed=-float(d[0])/60,
                 indentation=-float(d[2])/60, loaded_slip=-.1*float(d[3])/60,
                 old_cap_change=-.01*float(d[4]), legal_laps=10*float(d[5]), physical_failure=-10.*bool(failed))
    return sum(terms.values()), terms


def real_critic_state(joint, t, mission, forces):
    in_hold = mission.phase in ("terminal_stand", "home_hold", "far_hold", "complete")
    since = mission.stable_since if in_hold else None
    hold_s = max(0., t-since) if since is not None else 0.
    return task_observation(t, joint.legal_route.progress, mission.phase,
                            np.asarray(forces).reshape(4,3)[:,2] / (MASS*9.81/4), hold_s)


def init_head(run, head, seed, critic_path, manifest):
    if head not in SPEC["arms"] or seed not in SPEC["seeds"]:
        raise ValueError("Unknown learning arm/seed")
    run.mkdir(parents=True, exist_ok=True)
    path = run / "campaign.json"
    scope = dict(**manifest, arm=head, seed=seed, common_critic_sha256=file_digest(critic_path),
                 initial_critic_source=str(critic_path), background=SPEC["background"])
    if path.exists():
        if read_json(path) != scope:
            raise ValueError(f"Existing campaign identity differs at {path}")
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = RequestActor()
        critic = TrueTaskCritic()
        critic.load_state_dict(torch.load(critic_path, map_location="cpu", weights_only=True)["critic"])
        actor_opt = torch.optim.Adam(actor.parameters(), lr=SPEC["actor"]["lr"])
        critic_opt = torch.optim.Adam(critic.parameters(), lr=SPEC["critic"]["standalone_fit"]["lr"])
        payload = dict(schema=SPEC["schema"], arm=head, seed=seed, version=0, actor=actor.state_dict(),
                       critic=critic.state_dict(), actor_optimizer=actor_opt.state_dict(),
                       critic_optimizer=critic_opt.state_dict(), used_events=0, accepted_actor_steps=0)
        atomic_weights(checkpoint(run,0), payload)
    atomic_json(run/"published.json", dict(version=0, mode="acquire", used_events=0,
                wm_checkpoints=scope["initial_wm"], wm_sha256=scope["initial_wm_sha256"]))
    atomic_json(path, scope)


class LiveEventPolicy:
    def __init__(self, controller, joint, config, out, attempt):
        self.controller, self.joint = controller, joint
        self.config = config
        self.out, self.attempt = Path(out), attempt
        self.head, self.seed, self.case = config["head"], config["seed"], config.get("case")
        self.is_collector = config["mode"] == "collect"
        self.run = Path(config["run_dir"]) if self.is_collector else None
        self.pending = None
        self.last_file_offset = 0
        self.absorbed = False
        self.branch_start = None
        self.branch_end = None
        self.count = dict(requested=0, completed=0, used_barrier_rows=0, tail_excluded=0,
                          evaluation_only=0, zero_duration=0, data_cut=0)
        self.mode = "evaluate"
        if self.is_collector:
            self.publication = read_json(self.run/"published.json")
            self.load_publication()
        else:
            self.install(Path(config["checkpoint"]))
        self.physics_version = dict(controller=controller.cfg.get("identity"), action=SPEC["schema"],
                                    model_sha256=controller.settings["model_sha256"], actor_sha256=self.weight_hash)

    def install(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["arm"],payload["seed"],payload["schema"]) != (self.head,self.seed,SPEC["schema"]):
            raise ValueError("Actor checkpoint and controller identity differ")
        self.version = payload["version"]
        with torch.random.fork_rng(devices=[]):
            self.actor, self.critic = RequestActor().eval(), TrueTaskCritic().eval()
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.weight_hash = file_digest(path)

    def load_publication(self):
        version = self.publication["version"]
        self.install(checkpoint(self.run, version))
        self.mode = self.publication["mode"]
        path = batch_file(self.run, version, self.case)
        self.position = len(json_rows(path)) if path.exists() else 0

    def barrier(self):
        if not self.is_collector or self.mode != "acquire" or self.position < BATCH_PER_CASE:
            return
        while True:
            state = read_json(self.run/"published.json")
            if state["version"] != self.version or state["mode"] != "acquire":
                self.publication = state
                self.load_publication()
                return
            time.sleep(.4)

    def observe_execution(self):
        if self.pending is None:
            return
        path = self.controller.record_dir / "handover-requests.jsonl"
        if not path.is_file():
            return
        with path.open() as stream:
            stream.seek(self.last_file_offset)
            for line in stream:
                item = json.loads(line)
                if item.get("decision_id") != self.pending["request_no"] or item.get("stage") not in ("executed","censored"):
                    continue
                self.pending["execution"] = {k:item.get(k) for k in ("stage","accepted_s","actual_liftoff_s",
                    "first_zero_eligible_s","gate_blocks","reason","transfer_fraction_changed","accepted_gate")}
            self.last_file_offset = stream.tell()

    @torch.no_grad()
    def request(self, controller, t, mission, mq, material_info, grf, base_request, model_state):
        if self.absorbed:
            return base_request
        current = self.joint.snapshot(t, mission)
        critic_inputs = real_critic_state(self.joint,t,mission,grf)
        self.close(current, critic_inputs, "ongoing")
        if t >= self.joint.session_s - 1e-8:
            return base_request
        self.barrier()
        if model_state is None and self.config.get("nominal_fallback"):
            from s1_w59_planning import controller_state
            model_state = controller_state(controller,t,mission,mq,grf)
        state = torch.as_tensor(model_state, dtype=torch.float32).detach().cpu().reshape(NS)
        table = controller.causal_cells.table
        if self.head == "Diff-private":
            from s1_w59_model import private_state, private_world
            world_tensor = torch.as_tensor(table, dtype=state.dtype).unsqueeze(0)
            state_batch = state.unsqueeze(0)
            state = private_state(state_batch, world_tensor)[0]
            table = private_world(state_batch, world_tensor)[0].cpu().numpy()
        base64 = np.asarray(base_request, dtype=np.float64)
        base = torch.as_tensor(base64, dtype=torch.float32)
        mean = self.actor(state,base)
        value = float(self.critic(torch.from_numpy(critic_inputs)))
        serial = self.version*BATCH_PER_CASE+self.position if self.mode=="acquire" else None
        seed = self.seed*2**32 + 2*serial + ("AB".index(self.case)) if serial is not None else None
        if seed is None:
            raw = mean
        else:
            rng = torch.Generator().manual_seed(seed)
            raw = mean + self.actor.std * torch.randn(3, generator=rng)
        unconstrained = base64 + SCALE * np.tanh(raw.cpu().numpy().astype(np.float64))
        action = np.clip(unconstrained, LOW, HIGH)
        sampling = ("on_policy" if self.mode=="acquire" else "natural_tail"
                    if self.is_collector else "frozen_evaluation")
        if self.branch_start is None:
            self.branch_start = current.copy()
        observed = np.flatnonzero(table[:,4] > 0)
        world = dict(keys=observed.tolist(), measured=table[observed].tolist())
        self.pending = dict(event_id=f"{self.attempt}/request-{controller.request_count}",request_no=controller.request_count,
            case=self.case, attempt=self.attempt,source_dir=str(self.out),arm=self.head, seed=self.seed, version=self.version,
            versions=dict(self.physics_version, actor_sha256=self.weight_hash),
            sampling=sampling,serial=serial,rng_seed=seed, position=self.position if serial is not None else None,
            state=state.tolist(),world=world,critic_state=critic_inputs.tolist(),base=base64.tolist(),mean_raw=mean.tolist(),
            raw=raw.tolist(),logp_raw=float(self.actor.logp(mean,raw)),unprojected=unconstrained.tolist(),
            action=action.tolist(),saturated=(unconstrained!=action).tolist(),value=value,execution=None,
            start=current.copy())
        self.count["requested"] += 1
        return action

    @torch.no_grad()
    def close(self, end, successor, end_kind):
        item = self.pending
        if item is None:
            return
        self.observe_execution()
        self.pending = None
        start = item.pop("start")
        if end[0] <= start[0]:
            self.count["zero_duration"] += 1
            return
        absorbing = end_kind in ("success","physical_failure")
        reward, terms = original_reward(start,end,end_kind=="physical_failure")
        item.update(start_s=float(start[0]), end_s=float(end[0]),dt_s=float(end[0]-start[0]),
                    next_critic_state=successor.tolist(), next_value=0. if absorbing else float(self.critic(torch.from_numpy(successor))),
                    reward=reward,reward_terms=terms,terminated=absorbing,task_complete=end_kind=="success",
                    physical_failure=end_kind=="physical_failure",truncated=end_kind=="truncated",end_kind=end_kind)
        append(self.out/"w59-actor-events.jsonl", item)
        self.count["completed"] += 1
        if item["sampling"]=="on_policy":
            if self.position >= BATCH_PER_CASE or self.version != item["version"] or self.position!=item["position"]:
                raise ValueError("Head/position changed during a pending real physical action")
            append(batch_file(self.run,self.version,self.case),item)
            self.position += 1
            self.count["used_barrier_rows"] += 1
            if self.position==BATCH_PER_CASE:
                atomic_json(self.run/f"ready-{self.version:03d}-{self.case}.json",
                            dict(attempt=self.attempt,event_id=item["event_id"],end_s=item["end_s"],
                                 version=self.version,count=self.position))
        elif item["sampling"]=="natural_tail":
            self.count["tail_excluded"] += 1
        else:
            self.count["evaluation_only"] += 1

    def complete(self,t,mission,grf):
        settled = self.joint.snapshot(t,mission)
        self.close(settled,real_critic_state(self.joint,t,mission,grf),"success")
        self.absorbed = True
        self.branch_end = settled.copy()

    def finish(self,result):
        t, mission = float(self.joint.data.time), self.joint.mission
        reason = result["reason"]
        failed = reason.startswith(TERMINAL_FAILURES)
        technical = bool(result.get("error")) or reason=="execution_error"
        now = self.joint.snapshot(t,mission)
        if self.pending is not None and technical:
            append(self.out/"w59-interrupted-requests.jsonl",dict(event_id=self.pending["event_id"],reason=reason,time_s=t))
            self.pending = None
            self.count["data_cut"] += 1
        elif self.pending is not None:
            forces = self.joint.measurements()[3]
            self.close(now,real_critic_state(self.joint,t,mission,forces),"physical_failure" if failed else "truncated")
        row = dict(**self.count, task_absorbed=self.absorbed, physical_end_s=t, technical=technical,
                   qualification_continuation_s=(t-float(self.branch_end[0]) if self.absorbed else 0.),
                   head=self.head, seed=self.seed, terminal_reason=reason, whole_episode_hold=result.get("hold"))
        if self.branch_start is not None:
            last = self.branch_end if self.absorbed else now
            reward, terms = original_reward(self.branch_start,last,failed and not self.absorbed)
            row["continuation"] = dict(start_s=float(self.branch_start[0]), end_s=t, reward_end_s=float(last[0]),
                return_w38=reward if not technical else None, terms=terms,physical_failure=failed,
                actor_digest=self.weight_hash, technical=technical)
        result["w59_learning"] = row
        atomic_json(self.out/"w59-learner-result.json",row)


def attach(controller,joint,config,out,attempt):
    if getattr(controller,"timing_agent",None) is not None:
        raise ValueError("W59 three-dimensional actor cannot share a request with the old timing head")
    adapter = LiveEventPolicy(controller,joint,config,out,attempt)
    controller.w59_learner = adapter
    joint.w59_learner = adapter

