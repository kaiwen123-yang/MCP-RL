"""Explicit block objectives for the co-design candidate and its comparators.

The model block holds recorded commands/data/metrics fixed.  The policy block
uses fresh real-plant rollouts under a frozen model and an unchanged PPO
backend.  This is alternating surrogate optimization, not an exact gradient
through CEM or a proof of monotonic actual receding-horizon performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .environment import Config, advance_context, task_cost, terminal_cost
from .models import CompactResponse
from .planner import differentiable_rollout_cost, extend_commands_to_task_horizon


@dataclass(frozen=True)
class Arm:
    name: str
    response_reward: bool
    task_metric: bool
    model_task: bool
    actor_learning: bool = True


ARMS = {
    "ordinary": Arm("ordinary", False, False, False),
    "uniform": Arm("uniform", True, False, True),
    "weighted_id": Arm("weighted_id", True, True, False),
    "codesign": Arm("codesign", True, True, True),
    "frozen": Arm("frozen", False, False, False, False),
}


@dataclass(frozen=True)
class LearningConfig:
    response_weight: float = 0.2
    metric_alpha: float = 4.0
    model_task_weight: float = 0.1
    long_response_weight: float = 0.25
    model_learning_rate: float = 3e-4
    model_steps: int = 4
    model_grad_norm: float = 1.0
    environment: Config = field(default_factory=Config)


@dataclass
class ResponseBatch:
    # Time-major physical data. M macros, K microsteps/macro, B environments.
    starts: Tensor                 # [M,B,6]
    contexts: Tensor               # [M,B,6]
    commands: Tensor               # [M,B,2]
    observations: Tensor           # [M,K,B,5], normalized p/v/resource
    metrics: Tensor                # [M,K,B,5,5], detached at collection
    initial_state: Tensor          # [B,6]
    initial_context: Tensor        # [B,6]


def metric_quadratic(error: Tensor, metric: Tensor) -> Tensor:
    return torch.einsum("...i,...ij,...j->...", error, metric, error)


def predict_macro(model: CompactResponse, state: Tensor, command: Tensor,
                  context: Tensor) -> Tensor:
    current, ctx = model.encode(state), context
    features = []
    for _ in range(model.config.macro_steps):
        current, resource = model.step(current, command, ctx)
        ctx = advance_context(ctx, config=model.config)
        features.append(model.features(current, resource))
    return torch.stack(features)


def task_metric(model: CompactResponse, state: Tensor, commands: Tensor,
                context: Tensor, alpha: float) -> tuple[Tensor, dict[str, int]]:
    """Bounded PSD metrics from projected fixed-plan suffix costates.

    v_j is dJ/d[p_j,v_j,resource_j] in normalized coordinates, holding the
    response-memory coordinate at that node fixed.  It includes future model
    propagation because autograd targets the actual graph nodes, not a new
    stack detached from future computation.  Missing memory/representation
    effects remain outside this metric; it is not a safety or performance bound.
    """
    cfg = model.config
    full_commands = extend_commands_to_task_horizon(commands.detach(), context, cfg)
    with torch.enable_grad():
        current = model.encode(state.detach())
        ctx = context.detach()
        nodes, resources = [], []
        total = state.new_zeros(state.shape[0])
        for macro in range(full_commands.shape[1]):
            for _ in range(cfg.macro_steps):
                current, resource = model.step(current, full_commands[:, macro], ctx)
                ctx = advance_context(ctx, config=cfg)
                nodes.append(current)
                resources.append(resource)
                total = total + cfg.dt * task_cost(current, context=ctx,
                                                  predicted_resource=resource, config=cfg)
        total = total + terminal_cost(current, ctx, cfg)
        chosen = nodes[:cfg.macro_steps] + resources[:cfg.macro_steps]
        gradients = torch.autograd.grad(total.sum(), chosen, retain_graph=False)
        state_grad = torch.stack(gradients[:cfg.macro_steps])[..., :4]
        resource_grad = torch.stack(gradients[cfg.macro_steps:])
        v = torch.cat((state_grad, resource_grad.unsqueeze(-1)), dim=-1)
        v = v * model.feature_scale
        outer = v.unsqueeze(-1) * v.unsqueeze(-2)
        eye = torch.eye(5, device=state.device, dtype=state.dtype)
        unnormalized = eye + alpha * outer / (1.0 + v.square().sum(-1)[..., None, None])
        # Same total quadratic weight as W=I. This separates directional
        # allocation from a trivial increase of the regularizer coefficient.
        metric = 5.0 * unnormalized / unnormalized.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    count = state.shape[0] * full_commands.shape[1] * cfg.macro_steps
    return metric.detach(), {"metric_forward_model_steps": count,
                              "metric_reverse_model_steps": count}


def model_objective(model: CompactResponse, batch: ResponseBatch,
                    arm: Arm, config: LearningConfig,
                    calibration_only: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
    """Local open-loop realization + multi-macro recorded-command fitting.

    Every observation is real plant data.  The task term uses the SAME model
    and frozen recorded command sequence.  It is an explicitly model-based
    proposal, not evidence that the physical actor improved.
    """
    m, k, b, _ = batch.observations.shape
    predicted = predict_macro(model, batch.starts.reshape(m*b, 6),
                              batch.commands.reshape(m*b, 2), batch.contexts.reshape(m*b, 6))
    predicted = predicted.reshape(k, m, b, 5).transpose(0, 1)
    errors = predicted - batch.observations.detach()
    identity = torch.eye(5, device=errors.device, dtype=errors.dtype)
    weights = batch.metrics.detach() if arm.task_metric else identity
    local = config.environment.dt * metric_quadratic(errors, weights).sum(1).mean()

    rollout = differentiable_rollout_cost(model, batch.initial_state.detach(),
                                         batch.commands.transpose(0, 1).detach(),
                                         batch.initial_context.detach(),
                                         config.environment, return_trace=True)
    global_features = model.features(rollout.states[1:], rollout.resources)
    actual = batch.observations.reshape(m*k, b, 5).detach()
    long_error = global_features - actual
    long_weights = batch.metrics.reshape(m*k, b, 5, 5).detach() if arm.task_metric else identity
    # Average per-macro long rollout discrepancy, on the same response scale.
    long_fit = config.environment.dt * metric_quadratic(long_error, long_weights).sum(0).mean() / m
    identification = local + config.long_response_weight * long_fit
    predicted_task = rollout.cost.mean()
    task_coefficient = config.model_task_weight if arm.model_task and not calibration_only else 0.0
    loss = identification + task_coefficient * predicted_task
    return loss, {"model_loss": loss, "local_realization": local,
                  "long_realization": long_fit, "predicted_task": predicted_task,
                  "task_coefficient": loss.new_tensor(task_coefficient)}


def update_model(model: CompactResponse, optimizer: torch.optim.Optimizer,
                 batch: ResponseBatch, arm: Arm, config: LearningConfig,
                 calibration_only: bool = False) -> dict[str, float | int]:
    """Training operation. Implemented for the later server run; NOT run in verification."""
    last = {}
    for _ in range(config.model_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model_objective(model, batch, arm, config, calibration_only)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.model_grad_norm)
        optimizer.step()
        last = {name: float(value.detach()) for name, value in parts.items()}
    m, k, b, _ = batch.observations.shape
    return {**last, "model_optimizer_steps": config.model_steps,
            "fit_forward_model_steps": config.model_steps * 2*m*k*b,
            "fit_reverse_model_steps": config.model_steps * 2*m*k*b}
