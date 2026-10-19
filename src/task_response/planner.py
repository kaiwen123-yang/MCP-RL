"""Command-only MPC over the same learned microstep response model.

CEM and projected-gradient shooting use identical commands, model information,
costs, geometry penalties, and warm-start convention.  Neither optimizer is a
global oracle.  Work counters distinguish trajectory evaluations from forward
and reverse model steps; equal candidate counts are not equal total compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol

import torch
from torch import Tensor

from .environment import Config, DEFAULT_CONFIG, advance_context, task_cost, terminal_cost


class ResponseModel(Protocol):
    def encode(self, state: Tensor) -> Tensor:
        """Initialize z=[p2,v2,learned response memory2] from actual plant x."""

    def step(self, state: Tensor, command: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        """Return one-dt successor and normalized squared applied-input prediction."""


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 8
    candidates: int = 64
    elites: int = 8
    iterations: int = 3
    initial_std: float = 0.65
    min_std: float = 0.05
    smoothing: float = 0.20
    gradient_iterations: int = 12
    gradient_step_size: float = 0.15
    gradient_starts: int = 4
    seed: int = 0
    environment: Config = field(default_factory=Config)


@dataclass
class RolloutResult:
    cost: Tensor
    states: Tensor
    resources: Tensor
    contexts: Tensor
    stage_costs: Tensor
    active: Tensor


@dataclass
class PlanResult:
    command: Tensor
    cost: Tensor
    commands: Tensor
    evaluations: dict[str, int | str]
    method: str


def differentiable_rollout_cost(
    model: ResponseModel,
    state: Tensor,
    commands: Tensor,
    context: Tensor,
    config: Config = DEFAULT_CONFIG,
    return_trace: bool = False,
) -> Tensor | RolloutResult:
    """Roll out commands[B,H,2], holding each for config.macro_steps.

    The actual input state is encoded exactly once at entry.  Subsequent states
    are z=[p2,v2,response memory2]; its last two entries are not physical actuator
    truth.  The returned trace is time-major: states/contexts[T+1,B,...] include the
    initial point; resources/stage_costs/active[T,B] describe transitions.
    The function preserves gradients through states, commands, and model.
    It neither differentiates a sampling optimizer nor calls the actor/plant.
    For a short online horizon the endpoint term is a surrogate; a full-length
    held command tail still does not differentiate future replanning decisions.
    """
    current, ctx = model.encode(state), context
    states, contexts, resources, stages, masks = [current], [ctx], [], [], []
    total = state.new_zeros(state.shape[0])
    for macro in range(commands.shape[1]):
        command = commands[:, macro]
        for _ in range(config.macro_steps):
            enabled = ctx[..., 5] > config.dt * 1e-5
            predicted, resource = model.step(current, command, ctx)
            if resource.ndim == current.ndim:
                resource = resource.squeeze(-1)
            current = torch.where(enabled[:, None], predicted, current)
            ctx = advance_context(ctx, config=config)
            resource = torch.where(enabled, resource, torch.zeros_like(resource))
            stage = task_cost(current, context=ctx, predicted_resource=resource, config=config)
            stage = torch.where(enabled, stage, torch.zeros_like(stage))
            total = total + config.dt * stage
            states.append(current)
            contexts.append(ctx)
            resources.append(resource)
            stages.append(stage)
            masks.append(enabled)
    total = total + terminal_cost(current, ctx, config)
    if not return_trace:
        return total
    empty = state.new_empty((0, state.shape[0]))
    return RolloutResult(
        total, torch.stack(states), torch.stack(resources) if resources else empty,
        torch.stack(contexts), torch.stack(stages) if stages else empty,
        torch.stack(masks) if masks else empty.bool(),
    )


def extend_commands_to_task_horizon(
    commands: Tensor, context: Tensor, config: Config = DEFAULT_CONFIG,
) -> Tensor:
    """Pad a frozen plan with its last command to cover the remaining task.

    This is explicitly a held-tail surrogate.  It supplies no future disturbance
    information and does not represent the derivative of receding-horizon MPC.
    Shorter remaining durations in a batch are masked by rollout_cost.
    """
    # The finite pilot clock is an integer number of microsteps. Recover it
    # before division to avoid an extra macro caused by float32 subtraction.
    remaining_steps = round(float(context[..., 5].max().detach().cpu()) / config.dt)
    required = max(1, math.ceil(remaining_steps / config.macro_steps))
    required = min(required, math.ceil(config.horizon / config.macro_steps))
    if commands.shape[1] >= required:
        return commands[:, :required]
    if commands.shape[1] == 0:
        raise ValueError("a held-tail surrogate needs at least one command")
    return torch.cat((commands, commands[:, -1:].expand(-1, required - commands.shape[1], -1)), dim=1)


def shift_warm_start(commands: Tensor) -> Tensor:
    """Shift a previous solution by one macro step, holding its final command."""
    return torch.cat((commands[:, 1:], commands[:, -1:]), dim=1).detach()


def _initial_commands(state: Tensor, horizon: int, warm_start: Tensor | None) -> Tensor:
    if warm_start is None:
        return state.new_zeros((state.shape[0], horizon, 2))
    if warm_start.shape != (state.shape[0], horizon, 2):
        raise ValueError("warm_start must be [batch, planner horizon, 2]; shift explicitly before reuse")
    return warm_start.detach().clone().clamp(-1.0, 1.0)


def _score_candidates(model, state, context, candidates, config):
    batch, count, horizon, _ = candidates.shape
    expanded_state = state[:, None].expand(-1, count, -1).reshape(batch * count, -1)
    expanded_context = context[:, None].expand(-1, count, -1).reshape(batch * count, -1)
    costs = differentiable_rollout_cost(
        model, expanded_state, candidates.reshape(batch * count, horizon, 2),
        expanded_context, config=config,
    )
    return costs.reshape(batch, count)


class CEMPlanner:
    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        self.calls = 0

    def plan(self, model, state, context, warm_start=None) -> PlanResult:
        cfg = self.config
        if not 1 <= cfg.elites <= cfg.candidates or cfg.iterations < 1:
            raise ValueError("CEM requires 1 <= elites <= candidates and iterations >= 1")
        mean = _initial_commands(state, cfg.horizon, warm_start)
        std = torch.full_like(mean, cfg.initial_std)
        generator = torch.Generator(device=state.device).manual_seed(cfg.seed + self.calls)
        self.calls += 1
        best_commands = mean.clone()
        best_cost = state.new_full((state.shape[0],), float("inf"))
        evaluated = 0
        with torch.no_grad():
            for _ in range(cfg.iterations):
                noise = torch.randn((state.shape[0], cfg.candidates, cfg.horizon, 2),
                                    device=state.device, dtype=state.dtype, generator=generator)
                candidates = (mean[:, None] + std[:, None] * noise).clamp(-1.0, 1.0)
                candidates[:, 0] = best_commands
                costs = _score_candidates(model, state, context, candidates, cfg.environment)
                evaluated += state.shape[0] * cfg.candidates
                elite_costs, elite_ids = costs.topk(cfg.elites, dim=1, largest=False)
                gather_ids = elite_ids[..., None, None].expand(-1, -1, cfg.horizon, 2)
                elites = candidates.gather(1, gather_ids)
                improved = elite_costs[:, 0] < best_cost
                best_cost = torch.where(improved, elite_costs[:, 0], best_cost)
                best_commands = torch.where(improved[:, None, None], elites[:, 0], best_commands)
                new_mean = elites.mean(1)
                new_std = elites.std(1, unbiased=False).clamp_min(cfg.min_std)
                mean = cfg.smoothing * mean + (1.0 - cfg.smoothing) * new_mean
                std = cfg.smoothing * std + (1.0 - cfg.smoothing) * new_std
            # The fitted mean is evaluated; it is never returned as an unscored winner.
            mean_cost = differentiable_rollout_cost(model, state, mean, context, cfg.environment)
            evaluated += state.shape[0]
            improved = mean_cost < best_cost
            best_cost = torch.where(improved, mean_cost, best_cost)
            best_commands = torch.where(improved[:, None, None], mean, best_commands)
        return PlanResult(
            best_commands[:, 0].detach(), best_cost.detach(), best_commands.detach(),
            {"trajectory_evaluations": evaluated,
             "encoder_evaluations": evaluated,
             "model_steps": evaluated * cfg.horizon * cfg.environment.macro_steps,
             "reverse_model_steps": 0, "optimizer_steps": cfg.iterations,
             "model_information": "learned_response_only"}, "cem",
        )


class GradientPlanner:
    """Bounded multi-start projected gradient TO over exactly the same commands."""

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        self.calls = 0

    def plan(self, model, state, context, warm_start=None) -> PlanResult:
        cfg = self.config
        if cfg.gradient_starts < 1 or cfg.gradient_iterations < 1:
            raise ValueError("gradient shooting requires positive starts and iterations")
        base = _initial_commands(state, cfg.horizon, warm_start)
        generator = torch.Generator(device=state.device).manual_seed(cfg.seed + self.calls)
        self.calls += 1
        candidates = (base[:, None] + cfg.initial_std * torch.randn(
            (state.shape[0], cfg.gradient_starts, cfg.horizon, 2),
            device=state.device, dtype=state.dtype, generator=generator)).clamp(-1.0, 1.0)
        candidates[:, 0] = base
        best = base.clone()
        best_cost = state.new_full((state.shape[0],), float("inf"))
        evaluated, reversed_steps = 0, 0
        for iteration in range(cfg.gradient_iterations + 1):
            candidates = candidates.detach().requires_grad_(iteration < cfg.gradient_iterations)
            with torch.enable_grad():
                costs = _score_candidates(model, state, context, candidates, cfg.environment)
                evaluated += state.shape[0] * cfg.gradient_starts
                row_cost, index = costs.detach().min(dim=1)
                winner = candidates.detach()[torch.arange(state.shape[0], device=state.device), index]
                improved = row_cost < best_cost
                best_cost = torch.where(improved, row_cost, best_cost)
                best = torch.where(improved[:, None, None], winner, best)
                if iteration < cfg.gradient_iterations:
                    # autograd.grad requests command gradients only, never an actor/model update.
                    gradient, = torch.autograd.grad(costs.sum(), candidates)
                    reversed_steps += (state.shape[0] * cfg.gradient_starts
                                       * cfg.horizon * cfg.environment.macro_steps)
                    candidates = (candidates - cfg.gradient_step_size * gradient).clamp(-1.0, 1.0)
        return PlanResult(
            best[:, 0].detach(), best_cost.detach(), best.detach(),
            {"trajectory_evaluations": evaluated,
             "encoder_evaluations": evaluated,
             "model_steps": evaluated * cfg.horizon * cfg.environment.macro_steps,
             "reverse_model_steps": reversed_steps,
             "optimizer_steps": cfg.gradient_iterations,
             "model_information": "learned_response_only"}, "projected_gradient",
        )


def plan(model, state, context, warm_start=None, config=None) -> PlanResult:
    """Convenience one-shot CEM; reuse CEMPlanner to advance its random stream."""
    return CEMPlanner(config).plan(model, state, context, warm_start)
