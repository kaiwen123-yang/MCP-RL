"""Actual finite-task collection with frozen policy/model during each batch."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor

from response_ppo.ppo_backend import PPOBackend, Rollout
from .environment import (Config, advance_context, initial_state, make_context,
                          project_action, step, task_cost, task_metrics, terminal_cost)
from .learning import Arm, LearningConfig, ResponseBatch, metric_quadratic, predict_macro, task_metric
from .models import CompactResponse, actor_observation
from .planner import CEMPlanner, GradientPlanner, shift_warm_start


@dataclass
class Collection:
    ppo: Rollout
    response: ResponseBatch
    trace: Tensor
    actions: Tensor
    raw_task_reward: Tensor
    response_penalty: Tensor
    metrics: dict[str, Tensor]
    work: dict[str, int | float]


def episode_conditions(environments: int, seed: int, config: Config,
                       device: str, dtype: torch.dtype = torch.float32,
                       evaluation: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """Paired go/stop conditions and hidden bounded pushes.

    Go and stop halves receive identical physical initial conditions and pushes.
    Seeds used for held-out evaluation are disjoint from development/training.
    Future pushes are only consumed by the plant, never by an observation.
    """
    if environments < 2 or environments % 2:
        raise ValueError("Use an even number of environments for paired go/stop tasks")
    half = environments // 2
    rng = torch.Generator(device=device).manual_seed(seed)
    physical = initial_state(half, device, dtype)
    signs = torch.where(torch.arange(half, device=device) % 2 == 0, 1., -1.).to(dtype)
    physical[:, 3] *= signs
    physical[:, 2] += .05 * (2*torch.rand(half, generator=rng, device=device, dtype=dtype)-1)
    state = torch.cat((physical, physical.clone()))
    context = torch.cat((make_context("go", half, device, dtype, config),
                         make_context("stop", half, device, dtype, config)))
    disturbances = torch.zeros(config.horizon, half, 2, device=device, dtype=dtype)
    # Same declared bounded amplitude domain; unseen seeds, not tougher test
    # physics introduced after seeing candidate outcomes.
    amplitude = .25 * (2*torch.rand(half, generator=rng, device=device, dtype=dtype)-1)
    centre = config.horizon // 2
    disturbances[centre:centre+2, :, 1] = amplitude
    return state, context, torch.cat((disturbances, disturbances.clone()), dim=1)


def collect(policy: PPOBackend, model: CompactResponse,
            planner: CEMPlanner | GradientPlanner, arm: Arm,
            learning: LearningConfig, environments: int, seed: int,
            deterministic: bool = False) -> Collection:
    """Collect one complete configured horizon. This never changes weights.

    Evaluation still reports REAL task cost, not the shaped PPO reward.  W,
    targets, command selection and model parameters are frozen throughout
    every PPO batch; updated models always require a fresh actor collection.
    """
    cfg = learning.environment
    if cfg.horizon % cfg.macro_steps:
        raise ValueError("Pilot horizon must contain complete macro intervals")
    reference = next(model.parameters())
    state, context, disturbance = episode_conditions(environments, seed, cfg,
                                                    str(reference.device), reference.dtype)
    state0, context0 = state.clone(), context.clone()
    generator = torch.Generator(device=state.device).manual_seed(seed + 100_003)
    obs, raw_actions, log_probs, values, rewards = [], [], [], [], []
    true_rewards, penalties, actions, states = [], [], [], [state]
    starts, contexts, commands, response_observations, weight_records = [], [], [], [], []
    work = {"physical_steps": 0, "actor_forwards": 0, "critic_forwards": 0,
            "planner_calls": 0, "planner_model_steps": 0, "planner_reverse_model_steps": 0,
            "planner_trajectory_evaluations": 0, "target_model_steps": 0,
            "metric_forward_model_steps": 0, "metric_reverse_model_steps": 0,
            "policy_optimizer_steps": 0, "model_optimizer_steps": 0,
            "planner_wall_seconds": 0.0}
    planner_latencies = []
    warm = None
    for macro in range(cfg.horizon // cfg.macro_steps):
        if state.device.type == "cuda":
            torch.cuda.synchronize(state.device)
        plan_started = time.perf_counter()
        plan = planner.plan(model, state, context, warm)
        if state.device.type == "cuda":
            torch.cuda.synchronize(state.device)
        latency = time.perf_counter() - plan_started
        planner_latencies.append(latency)
        work["planner_wall_seconds"] += latency
        warm = shift_warm_start(plan.commands)
        command = plan.command
        starts.append(state.detach().clone())
        contexts.append(context.detach().clone())
        commands.append(command.detach().clone())
        work["planner_calls"] += environments
        work["planner_model_steps"] += int(plan.evaluations["model_steps"])
        work["planner_reverse_model_steps"] += int(plan.evaluations["reverse_model_steps"])
        work["planner_trajectory_evaluations"] += int(plan.evaluations["trajectory_evaluations"])
        with torch.no_grad():
            targets = predict_macro(model, state, command, context)
        work["target_model_steps"] += cfg.macro_steps * environments
        if arm.task_metric:
            weights, metric_work = task_metric(model, state, plan.commands, context, learning.metric_alpha)
            for key, value in metric_work.items():
                work[key] += value
        else:
            weights = torch.eye(5, device=state.device, dtype=state.dtype).expand(
                cfg.macro_steps, environments, 5, 5).clone()
        weight_records.append(weights)
        observed = []
        for micro in range(cfg.macro_steps):
            tick = macro * cfg.macro_steps + micro
            observation = actor_observation(state, command, context)
            with torch.no_grad():
                sample = policy.act(observation, deterministic=deterministic, generator=generator)
                actual_action = project_action(sample.action, cfg)
                following = step(state, actual_action, disturbance[tick], cfg)
                next_context = advance_context(context, config=cfg)
                real_cost = cfg.dt * task_cost(following, actual_action, next_context, config=cfg)
                if tick == cfg.horizon-1:
                    real_cost = real_cost + terminal_cost(following, next_context, cfg)
                feature = model.observed_features(following, actual_action)
                discrepancy = metric_quadratic(feature-targets[micro], weights[micro])
                shaping = cfg.dt * learning.response_weight * discrepancy if arm.response_reward else torch.zeros_like(real_cost)
                reward = -real_cost-shaping
            obs.append(observation.detach()); raw_actions.append(sample.raw_action)
            log_probs.append(sample.log_prob); values.append(sample.value); rewards.append(reward)
            true_rewards.append(-real_cost); penalties.append(shaping); actions.append(actual_action)
            observed.append(feature); states.append(following)
            state, context = following, next_context
            work["physical_steps"] += environments
            work["actor_forwards"] += environments
            work["critic_forwards"] += environments
        response_observations.append(torch.stack(observed))
    value_tensor = torch.stack(values)
    # The next observation at a macro boundary includes its newly replanned
    # command. Using value[t+1] avoids bootstrapping with a stale command.
    next_value = torch.cat((value_tensor[1:], torch.zeros_like(value_tensor[-1:])), dim=0)
    terminal = torch.zeros_like(value_tensor, dtype=torch.bool); terminal[-1] = True
    ppo = Rollout(torch.stack(obs), torch.stack(raw_actions), torch.stack(rewards),
                  terminal, torch.zeros_like(terminal), torch.stack(log_probs), value_tensor, next_value)
    response = ResponseBatch(torch.stack(starts), torch.stack(contexts), torch.stack(commands),
                             torch.stack(response_observations), torch.stack(weight_records), state0, context0)
    trace, action_tensor = torch.stack(states), torch.stack(actions)
    metrics = task_metrics(trace, action_tensor, context0, cfg)
    # Batch latency is never divided by environment count and called per-tick
    # deployment latency. Dedicated B=1 evaluation is required for deployment.
    latencies = torch.tensor(planner_latencies, dtype=torch.float64)
    metrics.update({"planner_batch_latency_p50_seconds": latencies.quantile(.5),
                    "planner_batch_latency_p95_seconds": latencies.quantile(.95),
                    "planner_batch_latency_max_seconds": latencies.max()})
    return Collection(ppo, response, trace, action_tensor, torch.stack(true_rewards),
                      torch.stack(penalties), metrics, work)
