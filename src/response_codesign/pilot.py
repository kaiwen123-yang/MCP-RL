"""Bounded learning pilot using the original SPlaTES planner and Hierarchy.

This is an independent algorithm on a small nonlinear servo, not a reproduction
of the SPlaTES paper or evidence of robot-level superiority.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from .dynamics import PhysicalConfig, physical_step, response_step, nominal_input, teacher_action
from .learning import Actor, JointLearner, UpdateConfig, execution_features, parameter_vector
from .splates_bridge import build_splates_planner, build_hierarchy


ROOT = Path(__file__).resolve().parents[2]


def as_response(x, cfg):
    p, v, a = x.unbind(-1)
    return torch.stack((p, v, a-cfg.k*torch.sin(p)-cfg.d*v-cfg.drag*v*v.abs()), -1)


class ServoBatch(gym.vector.VectorEnv):
    def __init__(self, design, seed, steps=100, command_steps=10):
        self.cfg = PhysicalConfig()
        self.design = design.detach().clone()
        self.num_envs = len(design)
        self.steps = steps
        self.command_steps = command_steps
        self.step_index = 0
        generator = torch.Generator().manual_seed(seed)
        self.x = 0.04*torch.randn(self.num_envs, 3, generator=generator)
        self.x[:, 2] = 0
        self.first_goal = 0.18+0.12*torch.rand(self.num_envs, generator=generator)
        self.first_goal *= (2*torch.randint(0, 2, (self.num_envs,), generator=generator)-1)
        self.second_goal = -self.first_goal*(0.5+0.4*torch.rand(self.num_envs, generator=generator))
        self.switch_time = 0.9+0.4*torch.rand(self.num_envs, generator=generator)
        self.disturbance_phase = 6.28*torch.rand(self.num_envs, generator=generator)
        self.command = torch.zeros(self.num_envs, 1)
        self.reference = as_response(self.x, self.cfg)
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.single_action_space = gym.spaces.Box(-self.cfg.U, self.cfg.U, shape=(1,), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.last_cost = None
        self.last_disturbance = None

    def observation(self):
        return torch.cat((self.x, self.design), -1)

    def target(self, t):
        t = torch.as_tensor(t, dtype=self.x.dtype)
        if t.ndim == 0:
            return torch.where(t < self.switch_time, self.first_goal, self.second_goal)
        return torch.where(t[None, :] < self.switch_time[:, None],
                           self.first_goal[:, None], self.second_goal[:, None])

    def future_targets(self, horizon):
        now = self.step_index*self.cfg.dt
        return self.target(now+(torch.arange(horizon)+1)*self.command_steps*self.cfg.dt)

    def step(self, action):
        action = torch.as_tensor(action).reshape(self.num_envs)
        if self.step_index % self.command_steps == 0:
            self.reference = as_response(self.x, self.cfg)
        request = nominal_input(self.reference, self.command, self.design, self.cfg)
        t = self.step_index*self.cfg.dt
        w = self.cfg.W*(0.7*torch.sin(3.1*t+self.disturbance_phase)
                        +0.3*torch.cos(1.3*t+self.disturbance_phase))
        self.x = physical_step(self.x, action, w, self.cfg)
        self.reference = response_step(self.reference, self.command, self.design, self.cfg.dt)
        self.step_index += 1
        target = self.target(self.step_index*self.cfg.dt)
        task = (self.x[:, 0]-target).square()+0.005*action.square()+0.01*self.x[:, 1].square()
        realization = ((self.x[:, 0]-self.reference[:, 0]).square()
                       +0.05*(self.x[:, 1]-self.reference[:, 1]).square())
        input_error = (action-request).square()
        self.last_cost = torch.stack((task, realization, input_error), -1)
        self.last_disturbance = w
        assert torch.isfinite(self.x).all(), "Non-finite physical rollout"
        done = np.zeros(self.num_envs, dtype=bool)
        return self.observation(), -task, done, done, {}


def planning_objective(state, commands, targets):
    cfg = PhysicalConfig()
    batch, samples, horizon, _ = commands.shape
    z = as_response(state[:, :3], cfg)[:, None].expand(-1, samples, -1).clone()
    design = state[:, None, 3:7].expand(-1, samples, -1)
    cost = torch.zeros(batch, samples)
    margin = torch.full_like(cost, -torch.inf)
    macro_step = 0.2
    for index in range(horizon):
        c = commands[:, :, index, :]
        # Check four points in each command segment, not just a terminal state.
        for _ in range(4):
            request = nominal_input(z, c, design, cfg)
            margin = torch.maximum(margin, request.abs()-(cfg.U-0.20))
            margin = torch.maximum(margin, z[..., 0].abs()-0.65)
            margin = torch.maximum(margin, z[..., 1].abs()-1.0)
            z = response_step(z, c, design, dt=macro_step/4, substeps=2)
            cost += ((z[..., 0]-targets[:, None, index]).square()
                     +0.005*request.square()+0.01*z[..., 1].square())/4
    return cost/horizon, margin


class PilotPlanner:
    """Explicit, unverified zero-command fallback; infeasibility remains counted."""
    def __init__(self, planner, excitation_generator=None):
        self.inner = planner
        self.device = planner.device
        self.command_bound = planner.command_bound
        self.command_dim = planner.command_dim
        self.infeasible = 0
        self.planned_instances = 0
        self.last_info = None
        self.excitation_generator = excitation_generator

    def plan(self, state, task):
        result = self.inner.plan(state, task)
        self.infeasible += int((~result.feasible).sum())
        self.planned_instances += len(result.feasible)
        result.action = torch.where(result.feasible[:, None], result.action, torch.zeros_like(result.action))
        if self.excitation_generator is not None:
            # Common identification phase: explicit excitation replaces the
            # proposed command, including the command seen by Hierarchy logs.
            result.action = (torch.rand(len(state), 1, generator=self.excitation_generator)-0.5)*0.65
        self.last_info = result.info
        return result


def rollout(learner, seed, *, deterministic=False, batch=8, steps=100,
            fixed_design=None, random_commands=False, update=False, trace=False):
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    design_generator = torch.Generator().manual_seed(seed+7000000)
    excitation_generator = torch.Generator().manual_seed(seed+8000000)
    if fixed_design is not None:
        design = fixed_design[None].expand(batch, -1).clone()
    elif deterministic:
        design = learner.mean[None].expand(batch, -1).clone()
    else:
        design = learner.sample_design(batch, design_generator)
    env = ServoBatch(design, seed+19, steps)
    initial_state = env.x.clone()
    counters = {"candidate_sequences": 0, "objective_calls": 0, "response_transitions": 0}
    def counted_objective(state, commands, targets):
        count = commands.shape[0]*commands.shape[1]
        counters["candidate_sequences"] += count
        counters["objective_calls"] += 1
        counters["response_transitions"] += count*commands.shape[2]*4
        return planning_objective(state, commands, targets)
    inner = build_splates_planner(counted_objective, horizon=5, command_dim=1,
                                   command_bound=0.45, n_samples=24, n_iterations=2,
                                   elite_ratio=0.2, constraint_penalty=100.0)
    planner = PilotPlanner(inner, excitation_generator if random_commands else None)
    features, raw_actions, costs = [], [], []
    xs, actions, refs, commands, disturbances = [], [], [], [], []
    segments = []
    segment_start = None
    segment_targets = []
    current_seed = seed
    max_input, saturated_steps = 0.0, 0

    def policy(observation, command):
        nonlocal segment_start, segment_targets
        env.command = command.detach().clone()
        feature = execution_features(observation[:, :3], command)
        with torch.no_grad():
            mean = learner.actor(feature)
            noise = torch.zeros_like(mean) if deterministic else torch.randn(batch, generator=generator)*learner.config.action_std
            raw = mean+noise
            action = env.cfg.U*torch.tanh(raw)
        features.append(feature)
        raw_actions.append(raw)
        if env.step_index % env.command_steps == 0:
            segment_start = as_response(env.x, env.cfg).clone()
            segment_targets = []
        return action[:, None]

    def task_provider(observation, active):
        torch.manual_seed(current_seed+1009*(env.step_index//env.command_steps))
        return env.future_targets(5)

    def record(observation, command, action, next_observation, terminated, truncated):
        nonlocal max_input, saturated_steps
        costs.append(env.last_cost.clone())
        max_input = max(max_input, float(action.abs().max()))
        saturated_steps += int((action.abs() >= 0.99*env.cfg.U).sum())
        segment_targets.append(as_response(env.x, env.cfg).clone())
        if env.step_index % env.command_steps == 0:
            segments.append((segment_start, env.command.clone(), torch.stack(segment_targets)))
        if trace:
            xs.append(env.x.clone()); actions.append(action.clone()); refs.append(env.reference.clone())
            commands.append(env.command.clone()); disturbances.append(env.last_disturbance.clone())

    payload = {}
    def joint_update():
        if update:
            return learner.update(payload)
        return None

    hierarchy = build_hierarchy(planner, policy, env, task_provider, env.command_steps,
                                 transition_callback=record, joint_update=joint_update)
    obs = env.observation()
    with torch.no_grad():
        for _ in range(steps):
            action = hierarchy.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            hierarchy.process_transition(next_obs, reward, terminated, truncated)
            obs = next_obs
    payload.update(features=torch.stack(features), raw_actions=torch.stack(raw_actions),
                   costs=torch.stack(costs), design=design)
    # Leave no_grad before the real Hierarchy.learn -> independent update callback.
    hierarchy.learn()
    mean_costs = payload["costs"].mean(dim=(0, 1)).tolist()
    episode_costs = payload["costs"].mean(dim=0)
    metrics = {"task_cost": mean_costs[0], "realization_mse": mean_costs[1],
               "input_response_mse": mean_costs[2],
               "tracking_episode_violation_fraction": float((episode_costs[:, 1] > learner.config.tracking_budget).float().mean()),
               "input_episode_violation_fraction": float((episode_costs[:, 2] > learner.config.input_residual_budget).float().mean()),
               "max_actual_input": max_input, "near_saturation_fraction": saturated_steps/(steps*batch),
               "nominal_infeasible_fraction": planner.infeasible/max(1, planner.planned_instances),
               "physical_steps": hierarchy.n_total_env_steps,
               "planner_batch_calls": inner.n_plans,
               "planner_instance_calls": planner.planned_instances,
               **counters,
               "wall_seconds": time.perf_counter()-started,
               "actor_parameter_count": sum(p.numel() for p in learner.actor.parameters()),
               "response_parameter_count": 4}
    if trace:
        payload["trace"] = {"state": torch.stack(xs), "action": torch.stack(actions),
                            "reference": torch.stack(refs), "command": torch.stack(commands),
                            "disturbance": torch.stack(disturbances), "initial_state": initial_state}
    payload["segments"] = segments
    payload["metrics"] = metrics
    payload["hierarchy_provenance"] = hierarchy.integration_provenance
    payload["planner_provenance"] = planner.last_info
    return payload


def warm_start(seed, steps=512, fit_steps=600):
    """4096 actual teacher transitions shared by all arms of a seed."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed+1)
    cfg = PhysicalConfig()
    batch = 8
    x = torch.zeros(batch, 3)
    command = torch.zeros(batch, 1)
    data, labels = [], []
    with torch.no_grad():
        for index in range(steps):
            if index % 100 == 0:
                x = 0.06*torch.randn(batch, 3, generator=generator)
                x[:, 2] = 0
            if index % 20 == 0:
                command = (torch.rand(batch, 1, generator=generator)-0.5)*0.7
            static_reference = torch.cat((command, torch.zeros(batch, 2)), -1)
            action = teacher_action(x, static_reference, command, torch.zeros(batch, 4), cfg)
            data.append(execution_features(x, command))
            labels.append(action)
            w = cfg.W*(2*torch.rand(batch, generator=generator)-1)
            x = physical_step(x, action, w, cfg)
    data = torch.cat(data); labels = torch.cat(labels)
    actor = Actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.002)
    for _ in range(fit_steps):
        indices = torch.randint(0, len(data), (256,), generator=generator)
        loss = (cfg.U*torch.tanh(actor(data[indices]))-labels[indices]).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        mse = (cfg.U*torch.tanh(actor(data))-labels).square().mean().item()
    return actor, {"teacher_physical_steps": len(data), "teacher_labels": len(data),
                   "supervised_updates": fit_steps, "supervised_samples_reused": fit_steps*256,
                   "teacher_fit_mse": mse}


def identify(segments, initial, updates=100):
    starts = torch.cat([s[0] for s in segments])
    commands = torch.cat([s[1] for s in segments])
    targets = torch.cat([s[2] for s in segments], dim=1)
    eta = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([eta], lr=0.03)
    generator = torch.Generator().manual_seed(144)
    for _ in range(updates):
        chosen = torch.randint(0, len(starts), (64,), generator=generator)
        z = starts[chosen]
        loss = torch.zeros(())
        for t in range(targets.shape[0]):
            z = response_step(z, commands[chosen], eta[None].expand(len(chosen), -1))
            errors = (z-targets[t, chosen]).square()
            loss = loss+(errors[:, 0]+0.05*errors[:, 1]+0.001*errors[:, 2]).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return eta.detach(), {"identification_updates": updates, "identification_segments": len(starts),
                          "last_identification_loss": float(loss.detach())}


def average_metrics(rollouts):
    metrics = [r["metrics"] for r in rollouts]
    sums = {"physical_steps", "planner_batch_calls", "planner_instance_calls", "candidate_sequences",
            "objective_calls", "response_transitions", "wall_seconds"}
    return {key: (sum(m[key] for m in metrics) if key in sums else
                  max(m[key] for m in metrics) if key == "max_actual_input" else
                  float(np.mean([m[key] for m in metrics])))
            for key in metrics[0]}


OFFSETS = torch.tensor([[0., 0., 0., 0.], [-0.6, 0., 0., -0.4],
                        [0.4, 0.4, -0.4, 0.4], [0., -0.4, 0.4, -0.4]])


def select_feasible(results, cfg):
    """Validation screening only, not a robust realization certificate."""
    metrics = [r["metrics"] for r in results]
    violations = [max(m["realization_mse"]/cfg.tracking_budget-1, 0)
                  +max(m["input_response_mse"]/cfg.input_residual_budget-1, 0)
                  +m["nominal_infeasible_fraction"] for m in metrics]
    feasible = [index for index, value in enumerate(violations) if value <= 1e-8]
    if feasible:
        index = min(feasible, key=lambda i: metrics[i]["task_cost"])
    else:
        index = min(range(len(metrics)), key=lambda i: (violations[i], metrics[i]["task_cost"]))
    return index, {"validation_mean_feasible": bool(feasible), "feasible_candidates": feasible,
                   "candidate_violation_scores": violations, "candidate_metrics": metrics}


def choose_response(learner, seed):
    candidates = learner.mean[None]+OFFSETS
    results = [rollout(learner, seed, deterministic=True, fixed_design=candidate) for candidate in candidates]
    values = [r["metrics"]["task_cost"] for r in results]
    index, screening = select_feasible(results, learner.config)
    learner.mean = candidates[index].clone()
    return {"candidate_costs": values, "selected": index, "metrics": average_metrics(results), **screening}


def analytic_response_initialization():
    # Static backstepping teacher has polynomial s^3+11s^2+29s+32.
    # Include its coefficients in the shared response family before experiments.
    omega = np.sqrt(32/11)
    bounded = torch.tensor([omega, 29/(22*omega), 11., 0.02], dtype=torch.float32)
    probability = (bounded-torch.tensor([1.2, .7, 3., 0.]))/torch.tensor([1.8, .7, 11., .5])
    return torch.logit(probability)


def run_seed(seed, output, updates=40, common_updates=16, test_batches=8):
    cfg = UpdateConfig()
    actor, warm_info = warm_start(seed)
    common = JointLearner(actor, analytic_response_initialization(), "independent", cfg)
    initial_selection = choose_response(common, 40000+seed)
    common_metrics = []
    for index in range(common_updates):
        common_metrics.append(rollout(common, seed*1000+index, update=True))
    results = {}
    for mode in ["joint", "block", "fixed", "independent", "tuning_only"]:
        learner = common.clone(mode)
        initial_actor = parameter_vector(learner.actor)
        history, training_metrics = [], []
        for index in range(updates):
            r = rollout(learner, 100000+seed*1000+index, update=True)
            training_metrics.append({"metrics": r["metrics"]})
            history.append(learner.last_update)
        calibrations = [rollout(learner, 200000+seed*1000+index,
                                deterministic=True, random_commands=True) for index in range(5)]
        all_segments = [segment for item in calibrations for segment in item["segments"]]
        fitted, id_info = identify(all_segments, learner.mean)
        if mode == "independent":
            learner.mean = fitted
        selection = choose_response(learner, 300000+seed)
        evaluations = [rollout(learner, 500000+seed*1000+index, deterministic=True, trace=index == 0)
                       for index in range(test_batches)]
        result = {"seed": seed, "mode": mode, "mean": learner.mean.tolist(),
                  "training": average_metrics(training_metrics),
                  "calibration": average_metrics(calibrations), "identification": id_info,
                  "selection": selection, "evaluation": average_metrics(evaluations),
                  "actor_change_norm": float(torch.linalg.vector_norm(parameter_vector(learner.actor)-initial_actor)),
                  "history": history}
        results[mode] = result
        folder = output/f"seed-{seed}"/mode
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": learner.actor.state_dict(), "mean": learner.mean,
                    "config": asdict(cfg)}, folder/"checkpoint.pt")
        trace = evaluations[0]["trace"]
        np.savez(folder/"trace.npz", **{name: value.numpy() for name, value in trace.items()})
        (folder/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"seed": seed, "mode": mode, **result["evaluation"]}), flush=True)
    # Deliberately stronger fixed-response comparator: each of four designs gets
    # a complete actor-training budget. Selection chooses a trained pair, never
    # changes its response after training. Its extra cost is separately charged.
    fixed_candidates, fixed_training, fixed_history = [], [], []
    for candidate_index, candidate in enumerate(common.mean[None]+OFFSETS):
        learner = common.clone("fixed")
        learner.mean = candidate.clone()
        candidate_history = []
        for index in range(updates):
            r = rollout(learner, 100000+seed*1000+index, update=True)
            fixed_training.append({"metrics": r["metrics"]})
            candidate_history.append(learner.last_update)
        fixed_candidates.append(learner)
        fixed_history.append(candidate_history)
    validations = [rollout(item, 300000+seed, deterministic=True) for item in fixed_candidates]
    selected, screening = select_feasible(validations, cfg)
    learner = fixed_candidates[selected]
    calibrations = [rollout(learner, 200000+seed*1000+index, deterministic=True, random_commands=True)
                    for index in range(5)]
    _, id_info = identify([segment for item in calibrations for segment in item["segments"]], learner.mean)
    evaluations = [rollout(learner, 500000+seed*1000+index, deterministic=True, trace=index == 0)
                   for index in range(test_batches)]
    result = {"seed": seed, "mode": "fixed_search", "mean": learner.mean.tolist(),
              "training": average_metrics(fixed_training), "calibration": average_metrics(calibrations),
              "identification": id_info,
              "selection": {"selected": selected, "metrics": average_metrics(validations), **screening},
              "evaluation": average_metrics(evaluations), "history": fixed_history,
              "actor_change_norm": float(torch.linalg.vector_norm(parameter_vector(learner.actor)-parameter_vector(common.actor))),
              "budget_note": "Four separately trained fixed-response actors, each with full main update budget"}
    results["fixed_search"] = result
    folder = output/f"seed-{seed}"/"fixed_search"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": learner.actor.state_dict(), "mean": learner.mean, "config": asdict(cfg)}, folder/"checkpoint.pt")
    np.savez(folder/"trace.npz", **{name: value.numpy() for name, value in evaluations[0]["trace"].items()})
    (folder/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"seed": seed, "mode": "fixed_search", **result["evaluation"]}), flush=True)
    seed_result = {"seed": seed, "warmstart": warm_info, "initial_selection": initial_selection,
                   "common_training": average_metrics(common_metrics), "arms": results}
    (output/f"seed-{seed}.json").write_text(json.dumps(seed_result, indent=2), encoding="utf-8")
    return seed_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"work/learning-pilot-2026-10-15/main")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--common-updates", type=int, default=16)
    parser.add_argument("--test-batches", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    specification = {"seeds": args.seeds, "updates": args.updates, "common_updates": args.common_updates,
                     "test_batches": args.test_batches, "batch": 8, "episode_steps": 100,
                     "actor": "12-32-32-1 tanh full action; no response design/reference/target observation",
                     "physical": asdict(PhysicalConfig()), "update": asdict(UpdateConfig()),
                     "response": {"ranges": [[1.2,3.0],[.7,1.4],[3.0,14.0],[0.0,.5]],
                                  "jerk_limit": 8.0, "initial_mean": analytic_response_initialization().tolist(),
                                  "candidate_offsets": OFFSETS.tolist()},
                     "fixed_search": "four fixed responses, separate actor training with full updates each; charged separately",
                     "planner": {"upstream": "SPlaTES CEMPlanner + Hierarchy", "horizon": 5,
                                 "samples": 24, "iterations": 2, "command_steps": 10},
                     "scope": "small nonlinear learning pilot; not original SPlaTES reproduction; empirical constraints",
                     "source_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                       for file in Path(__file__).parent.glob("*.py")}}
    (args.output/"specification.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    start = time.perf_counter()
    results = [run_seed(seed, args.output, args.updates, args.common_updates, args.test_batches) for seed in args.seeds]
    summary = {"specification": specification, "wall_seconds": time.perf_counter()-start,
               "seeds": results}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
