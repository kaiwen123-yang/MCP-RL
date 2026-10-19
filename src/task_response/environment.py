"""A planar inertial task for studying learned feedback/response co-design.

The plant has two coupled input resources and persistent actuator state.  It
does not implement a tracking controller for the actor.  This is a synthetic
discrete-time plant, not a Go2/Piper model or a robot safety certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Config:
    dt: float = 0.05
    macro_steps: int = 4
    horizon: int = 80
    input_bound: float = 1.5
    tau: float = 0.18
    linear_drag: float = 0.12
    speed_drag: float = 0.10
    cross_drag: float = 0.08
    goal_x: float = 2.0
    go_velocity: float = 0.6
    corridor_halfwidth: float = 0.65
    position_weight: float = 0.20
    velocity_weight: float = 0.10
    resource_weight: float = 0.025
    geometry_weight: float = 80.0
    terminal_position_weight: float = 4.0
    terminal_velocity_weight: float = 1.5
    position_tolerance: float = 0.15
    velocity_tolerance: float = 0.18


DEFAULT_CONFIG = Config()


def project_action(action: Tensor, config: Config = DEFAULT_CONFIG) -> Tensor:
    """Physical saturation onto one shared Euclidean input ball."""
    scale = (action.norm(dim=-1, keepdim=True) / config.input_bound).clamp_min(1.0)
    return action / scale


def step(
    state: Tensor,
    action: Tensor,
    disturbance: Tensor | None = None,
    config: Config = DEFAULT_CONFIG,
) -> Tensor:
    """Advance one dt, with arbitrary leading batch dimensions.

    State is [px, py, vx, vy, ax, ay].  Disturbance is an external acceleration
    acting on velocity; its future value is not an observation or model input.
    Explicit Euler defines this plant, including its one-step actuator delay.
    All methods must use the same discretization and disturbance realization.
    """
    position, velocity, actuator = state.split(2, dim=-1)
    applied = project_action(action, config)
    external = torch.zeros_like(velocity) if disturbance is None else disturbance
    speed = velocity.norm(dim=-1, keepdim=True)
    # Each drag component depends on both velocity coordinates.  It is passive:
    # v dot drag >= 0, and does not require a scalar allocation coordinate.
    other_squared = velocity.flip(-1).square()
    drag = (config.linear_drag + config.speed_drag * speed
            + config.cross_drag * other_squared) * velocity
    next_position = position + config.dt * velocity
    next_velocity = velocity + config.dt * (actuator - drag + external)
    next_actuator = actuator + (config.dt / config.tau) * (applied - actuator)
    return torch.cat((next_position, next_velocity, next_actuator), dim=-1)


def make_context(
    kind: str,
    batch: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    config: Config = DEFAULT_CONFIG,
) -> Tensor:
    """[goal_x, goal_y, goal_vx, goal_vy, halfwidth, remaining_seconds].

    Both tasks begin in the same disturbed state and seek the same position.
    The downstream contract changes the desired exit velocity: pass through
    with forward motion (go), or arrive at rest (stop).  This context is public
    to the actor, response model, and every comparator.
    """
    if kind not in {"go", "stop"}:
        raise ValueError("kind must be 'go' or 'stop'")
    target_velocity = config.go_velocity if kind == "go" else 0.0
    row = torch.tensor(
        [config.goal_x, 0.0, target_velocity, 0.0,
         config.corridor_halfwidth, config.horizon * config.dt],
        device=device, dtype=dtype,
    )
    return row.expand(batch, -1).clone()


def initial_state(
    batch: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    lateral_sign: float = 1.0,
) -> Tensor:
    """A shared moving/sideways-disturbed initial condition, not a reset gate."""
    row = torch.tensor([0.0, 0.0, 0.45, 0.35 * lateral_sign, 0.0, 0.0],
                       device=device, dtype=dtype)
    return row.expand(batch, -1).clone()


def advance_context(
    context: Tensor,
    steps: int = 1,
    config: Config = DEFAULT_CONFIG,
) -> Tensor:
    remaining_steps = (torch.round(context[..., 5:6] / config.dt) - steps).clamp_min(0.0)
    remaining = remaining_steps * config.dt
    return torch.cat((context[..., :5], remaining), dim=-1)


def geometry_excess(state: Tensor, context: Tensor) -> Tensor:
    """Signed lateral violation of a straight point-body corridor."""
    return state[..., 1].abs() - context[..., 4]


def task_cost(
    state: Tensor,
    action: Tensor | None = None,
    context: Tensor | None = None,
    predicted_resource: Tensor | None = None,
    config: Config = DEFAULT_CONFIG,
) -> Tensor:
    """Per-time stage cost at the successor state; caller multiplies by dt.

    Actual resource is squared *applied* input divided by input_bound squared.
    A model supplies the same normalized quantity, not a physical upper bound.
    Prediction error in this resource is part of model error, never a safety
    certificate.  Geometry is an identical soft planning cost for all methods;
    evaluation reports actual boundary violations separately.
    """
    if context is None:
        raise ValueError("task_cost requires public task context")
    if predicted_resource is None:
        if action is None:
            raise ValueError("provide action or predicted_resource")
        resource = project_action(action, config).square().sum(-1) / config.input_bound**2
    else:
        resource = predicted_resource
        if resource.ndim == state.ndim:
            resource = resource.squeeze(-1)
    position_error = (state[..., :2] - context[..., :2]).square().sum(-1)
    velocity_error = (state[..., 2:4] - context[..., 2:4]).square().sum(-1)
    outside = geometry_excess(state, context).clamp_min(0.0).square()
    return (config.position_weight * position_error
            + config.velocity_weight * velocity_error
            + config.resource_weight * resource
            + config.geometry_weight * outside)


def terminal_cost(
    state: Tensor, context: Tensor, config: Config = DEFAULT_CONFIG,
) -> Tensor:
    """Shared endpoint surrogate, also used at the true episode endpoint."""
    position_error = (state[..., :2] - context[..., :2]).square().sum(-1)
    velocity_error = (state[..., 2:4] - context[..., 2:4]).square().sum(-1)
    outside = geometry_excess(state, context).clamp_min(0.0).square()
    return (config.terminal_position_weight * position_error
            + config.terminal_velocity_weight * velocity_error
            + config.geometry_weight * outside)


def task_metrics(
    trace: Tensor,
    actions: Tensor,
    context: Tensor,
    config: Config = DEFAULT_CONFIG,
) -> dict[str, Tensor]:
    """Episode evidence from trace[T+1,B,6], actions[T,B,2], context[B,6].

    Success means the final measured state satisfies the public position and
    velocity tolerances and no sampled corridor crossing occurred.  Reaching
    once is reported separately and is not substituted for terminal success.
    Partial traces remain explicitly truncated, even if their endpoint looks
    good.  This is a finite simulation test, not a continuous-time guarantee.
    """
    if trace.shape[0] != actions.shape[0] + 1:
        raise ValueError("trace must contain the initial state plus every successor")
    time_steps = actions.shape[0]
    position_error = (trace[..., :2] - context[None, ..., :2]).norm(dim=-1)
    velocity_error = (trace[..., 2:4] - context[None, ..., 2:4]).norm(dim=-1)
    corridor_excess = geometry_excess(trace, context[None])
    geometry_ok = (corridor_excess <= 0).all(dim=0)
    at_goal = ((position_error <= config.position_tolerance)
               & (velocity_error <= config.velocity_tolerance))
    completed_horizon = time_steps * config.dt >= context[..., 5] - 1e-6
    endpoint_ok = at_goal[-1] & geometry_ok
    success = endpoint_ok & completed_horizon
    indices = torch.arange(trace.shape[0], device=trace.device, dtype=trace.dtype)[:, None]
    prior_geometry_ok = (corridor_excess <= 0).to(torch.int64).cumprod(dim=0).bool()
    first_reach = torch.where(at_goal & prior_geometry_ok, indices * config.dt,
                              torch.full_like(position_error, float("inf"))).amin(dim=0)
    first_reach = torch.where(torch.isfinite(first_reach), first_reach,
                              torch.full_like(first_reach, float("nan")))
    applied = project_action(actions, config)
    resource = applied.square().sum(-1) / config.input_bound**2
    stage = task_cost(trace[1:], action=applied, context=context[None], config=config)
    total = config.dt * stage.sum(dim=0) + terminal_cost(trace[-1], context, config)
    if time_steps:
        max_applied = applied.norm(dim=-1).amax(dim=0)
        max_requested = actions.norm(dim=-1).amax(dim=0)
    else:
        max_applied = trace.new_zeros(trace.shape[1])
        max_requested = max_applied.clone()
    return {
        "success": success,
        "success_rate": success.float().mean(),
        "completed_horizon": completed_horizon,
        "evaluation_truncated": ~completed_horizon,
        "endpoint_conditions_met": endpoint_ok,
        "first_reach_time": first_reach,
        "terminal_position_error": position_error[-1],
        "terminal_velocity_error": velocity_error[-1],
        "corridor_violation_count": (corridor_excess > 0).sum(dim=0),
        "maximum_corridor_excess": corridor_excess.amax(dim=0),
        "max_applied_input_norm": max_applied,
        "max_requested_input_norm": max_requested,
        "normalized_input_integral": config.dt * resource.sum(dim=0),
        "task_cost": total,
    }
