"""A bounded, complete forward/gradient check. NO optimizer/update is invoked.

This checks implementation identity, not learned task performance. Random
untrained networks are used; the fixture is never saved as a training dataset.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
from torch.distributions import Independent, Normal

from response_ppo.ppo_backend import PPOBackend, PPOConfig, tensor_gae
from .collection import collect
from .environment import Config, make_context, initial_state
from .learning import ARMS, LearningConfig, model_objective
from .models import CompactResponse
from .planner import PlannerConfig, CEMPlanner, GradientPlanner


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def run_check(output: str | Path | None = None) -> dict:
    torch.manual_seed(20261019)
    torch.set_num_threads(1)
    env = Config()
    learning = LearningConfig(environment=env)
    # Bounded verification budget, deliberately different from the server
    # comparison budget. It must not be used as a performance result.
    planning = PlannerConfig(horizon=2, candidates=4, elites=2, iterations=1,
                             gradient_starts=2, gradient_iterations=2,
                             seed=20261019, environment=env)
    policy = PPOBackend(PPOConfig(obs_dim=14, action_dim=2, action_bound=env.input_bound,
                                 gamma=1.0, epochs=4, minibatch_size=256, device="cpu"))
    model = CompactResponse(env)
    originals = {name: {k: v.detach().clone() for k, v in module.state_dict().items()}
                 for name, module in {"actor": policy.actor, "critic": policy.critic, "model": model}.items()}
    started = time.perf_counter()
    trajectory = collect(policy, model, CEMPlanner(planning), ARMS["codesign"],
                         learning, environments=2, seed=20261019, deterministic=False)
    deployment = collect(policy, model, CEMPlanner(planning), ARMS["codesign"],
                         learning, environments=2, seed=20261019, deterministic=True)
    assert trajectory.trace.shape == (81, 2, 6)
    assert trajectory.response.observations.shape == (20, 4, 2, 5)
    assert trajectory.response.metrics.shape == (20, 4, 2, 5, 5)
    assert bool(trajectory.metrics["completed_horizon"].all())
    assert float(trajectory.actions.norm(dim=-1).max()) <= env.input_bound + 1e-6
    actual_cost_error = float((trajectory.metrics["task_cost"] + trajectory.raw_task_reward.sum(0)).abs().max())
    assert actual_cost_error < 2e-5
    metric = trajectory.response.metrics
    eigenvalues = torch.linalg.eigvalsh(metric)
    trace_error = float((metric.diagonal(dim1=-2, dim2=-1).sum(-1)-5).abs().max())
    assert trace_error < 2e-6
    assert float(eigenvalues.min()) >= 5/9-2e-6
    assert float(eigenvalues.max()) <= 25/9+2e-6
    gradients = {}
    for name in ("ordinary", "uniform", "weighted_id", "codesign"):
        objective, parts = model_objective(model, deployment.response, ARMS[name], learning)
        grad = torch.autograd.grad(objective, tuple(model.parameters()))
        assert all(bool(g.isfinite().all()) for g in grad)
        gradients[name] = {"model_objective": float(objective.detach()),
                           "model_gradient_norm": float(torch.sqrt(sum(g.square().sum() for g in grad))),
                           "terms": {key: float(value.detach()) for key, value in parts.items()}}

    rollout = trajectory.ppo
    (mean, std), _ = policy.actor(rollout.obs)
    logp = Independent(Normal(mean, std), 1).log_prob(rollout.raw_action)
    behavior_error = float((logp.detach()-rollout.log_prob).abs().max())
    assert behavior_error < 5e-5
    advantage, returns = tensor_gae(rollout.reward, rollout.value, rollout.next_value,
                                    rollout.terminated, rollout.truncated, 1.0, .95)
    # Tangent of the PPO/likelihood-ratio surrogate at behavior policy. Compute
    # gradients only; do not call PPOBackend.update, backward(), or optimizer.step.
    normalized = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    policy_loss = -(torch.exp(logp-rollout.log_prob)*normalized).mean()
    policy_grad = torch.autograd.grad(policy_loss, tuple(policy.actor.parameters()))
    assert all(bool(g.isfinite().all()) for g in policy_grad)
    assert bool(returns.isfinite().all())
    assert bool(rollout.terminated[-1].all()) and not bool(rollout.terminated[:-1].any())

    # Same-information command optimization, with its distinct actual budget.
    to = GradientPlanner(planning).plan(model, initial_state(2),
                                        torch.cat((make_context("go", 1), make_context("stop", 1))))
    assert bool(to.cost.isfinite().all()) and float(to.commands.abs().max()) <= 1.0
    unchanged = {}
    for name, module in {"actor": policy.actor, "critic": policy.critic, "model": model}.items():
        unchanged[name] = all(torch.equal(v, originals[name][k]) for k, v in module.state_dict().items())
        assert unchanged[name]
        assert all(parameter.grad is None for parameter in module.parameters())
    assert policy.update_count == 0 and len(policy.optimizer.state) == 0
    metrics = {key: _json_value(value.detach().cpu().tolist()) for key, value in trajectory.metrics.items()}
    with torch.no_grad():
        (deployment_mean, _), _ = policy.actor(deployment.ppo.obs)
        mean_action_error = float((deployment_mean-deployment.ppo.raw_action).abs().max())
    # Batched and per-tick GEMMs can differ by float32 rounding despite
    # identical weights and observations.
    assert mean_action_error < 5e-6
    combined_work = {key: trajectory.work[key] + deployment.work[key] for key in trajectory.work}
    result = {
        "status": "FORWARD_AND_GRADIENT_CHECK_PASSED_NO_TRAINING",
        "scope": "Random untrained neural actor/model; full 80-step synthetic task only. No learning benefit, robot ability, or server runtime claim.",
        "config": {"environment": asdict(env), "planner_verification_only": asdict(planning)},
        "work": combined_work,
        "collection_modes": {"deployment_model_data": "deterministic_mean", "ppo_data": "stochastic"},
        "deterministic_raw_action_error": mean_action_error,
        "additional_command_to_work": to.evaluations,
        "additional_objective_forward_model_steps": 4*2*80*2,
        "additional_objective_reverse_model_steps": 4*2*80*2,
        "gradient_call_note": "One likelihood-ratio actor gradient; four model objective gradients; no gradients applied.",
        "weights_unchanged": unchanged,
        "policy_optimizer_steps": 0, "model_optimizer_steps": 0, "checkpoint_writes": 0,
        "server_connections": 0, "real_task_cost_recalculation_error": actual_cost_error,
        "metric_trace_error": trace_error,
        "metric_eigenvalue_range": [float(eigenvalues.min()), float(eigenvalues.max())],
        "behavior_log_probability_error": behavior_error,
        "model_objectives": gradients,
        "actor_surrogate_gradient_norm": float(torch.sqrt(sum(g.square().sum() for g in policy_grad))),
        "untrained_fixture_metrics_not_performance": metrics,
        "wall_seconds_local_check": time.perf_counter()-started,
        "training_branch_execution": "NOT_RUN",
        "server_experiment": "NOT_RUN",
    }
    if output is not None:
        path = Path(output) / "verification.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result
