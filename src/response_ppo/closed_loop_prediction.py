"""Functional prediction of the *executed* native three-state PPO loop.

This module creates no environment, RNG, network, planner or optimizer. The
caller supplies the current actor mean function, complete x/z state, command
timeline, design, goal preview and fixed native geometry. Reference z remains
continuous. Nominal prediction uses w=0; diagnostic scenarios must be explicit.

The support and fallback maps are piecewise smooth. Branch diagnostics are
returned; finite derivatives at a boundary are not a smoothness certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor

from .realization import FeasibleInterval


MeanActor = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class NativeContext:
    """Explicit native constants; P is the fixed trace-normalized metric.

    cfg follows EnvironmentConfig's fields. P/B/E must be the native fixed
    tensors, with shapes (3,3)/(3,)/(3,); B'PB>0 is required by the native
    minimax fallback. Reusing a saved float32 P in float64 analysis means
    converting that same P, not recomputing a different higher-precision DARE.
    """

    cfg: Any
    P: Tensor
    B: Tensor
    E: Tensor

    def geometry_like(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(torch.as_tensor(v, dtype=value.dtype, device=value.device)
                     for v in (self.P, self.B, self.E))


def _quadratic(x: Tensor, P: Tensor, y: Tensor) -> Tensor:
    return torch.einsum("...i,...ij,...j->...", x, P, y)


def physical_drift(x: Tensor, cfg: Any) -> Tensor:
    """Original discrete-Euler f0, without control or disturbance."""
    p, v, a = x.unbind(-1)
    h, tau = cfg.dt, cfg.tau
    return torch.stack((p + h*v,
                        v + h*(a-torch.sin(p)-.2*v-.1*v*v.abs()),
                        (1-h/tau)*a), -1)


def reference_step(z: Tensor, command: Tensor, raw_design: Tensor,
                   context: NativeContext) -> tuple[Tensor, Tensor, Tensor]:
    """Continuous native z update; command is physical (position c, speed eta)."""
    cfg = context.cfg
    _, B, _ = context.geometry_like(z)
    lower = raw_design.new_tensor((1.2, .7, 3., 0.))
    span = raw_design.new_tensor((1.8, .7, 11., .5))
    omega, zeta, kappa, factor = (lower + span*torch.sigmoid(raw_design)).unbind(-1)
    p, v, a = z.unbind(-1)
    c, eta = command.unbind(-1)
    omega_eff = omega*eta/(1+factor*v.square())
    q = a-torch.sin(p)-.2*v-.1*v*v.abs()
    jerk = kappa*(omega_eff.square()*(c-p)-2*zeta*omega_eff*v-q)
    raw_reference_input = a+cfg.tau*(torch.cos(p)*v+(.2+.2*v.abs())*q+jerk)
    reference_input = raw_reference_input.clamp(-1., 1.)
    return physical_drift(z, cfg)+B*reference_input[..., None], reference_input, raw_reference_input


def support_interval(f0: Tensor, znext: Tensor, error: Tensor,
                     context: NativeContext) -> tuple[FeasibleInterval, dict[str, Tensor]]:
    """Original quadratic interval, with safe evaluation of inactive sqrt.

    At disc<=0, sqrt receives a constant positive inactive operand and its
    reported value is zero. This preserves the original forward map, avoids
    0*inf in inactive backward paths and marks disc==0 as nonregular. It does
    not regularize a positive discriminant, widen an interval or alter a bound.
    """
    cfg = context.cfg
    P, B, E = context.geometry_like(f0)
    f0, B, E, znext, error = torch.broadcast_tensors(f0, B, E, znext, error)
    rho, delta, W, U = [torch.as_tensor(v, dtype=f0.dtype, device=f0.device)
                        for v in (cfg.rho, cfg.delta, cfg.disturbance_bound, cfg.input_bound)]
    offset = f0-znext
    endpoints = torch.stack((offset-E*W[..., None], offset+E*W[..., None]), dim=-2)
    a = _quadratic(B, P, B)
    b = 2*torch.stack((_quadratic(B, P, endpoints[..., 0, :]),
                       _quadratic(B, P, endpoints[..., 1, :])), dim=-1)
    budget = rho*_quadratic(error, P, error)+delta
    c = torch.stack((_quadratic(endpoints[..., 0, :], P, endpoints[..., 0, :]),
                     _quadratic(endpoints[..., 1, :], P, endpoints[..., 1, :])), dim=-1)-budget[..., None]
    discriminant = b.square()-4*a[..., None]*c
    controlled = a > 0
    safe_a = torch.where(controlled, a, torch.ones_like(a))[..., None]
    positive = discriminant > 0
    root_operand = torch.where(positive, discriminant, torch.ones_like(discriminant))
    square_root = torch.where(positive, torch.sqrt(root_operand), torch.zeros_like(discriminant))
    q = -.5*(b+torch.where(b >= 0, square_root, -square_root))
    safe_q = torch.where(q != 0, q, torch.ones_like(q))
    root_1 = q/safe_a
    root_2 = torch.where(q != 0, c/safe_q, -b/(2*safe_a))
    root_lower, root_upper = torch.minimum(root_1, root_2), torch.maximum(root_1, root_2)
    lower = root_lower.amax(dim=-1)
    upper = root_upper.amin(dim=-1)
    lower = torch.where(controlled, lower, torch.full_like(lower, -torch.inf))
    upper = torch.where(controlled, upper, torch.full_like(upper, torch.inf))
    lower, upper = torch.maximum(lower, -U), torch.minimum(upper, U)
    real_roots = (discriminant >= 0).all(dim=-1)
    constant_ok = (c <= 0).all(dim=-1)
    feasible = torch.where(controlled, real_roots, constant_ok) & (lower <= upper)
    interval = FeasibleInterval(torch.where(feasible, lower, torch.full_like(lower, torch.inf)),
                                torch.where(feasible, upper, torch.full_like(upper, -torch.inf)),
                                feasible, discriminant)
    low_candidates = torch.cat((root_lower, torch.full_like(lower[..., None], -cfg.input_bound)), -1)
    high_candidates = torch.cat((root_upper, torch.full_like(upper[..., None], cfg.input_bound)), -1)
    ordered_lower, ordered_upper = low_candidates.sort(-1).values, high_candidates.sort(-1).values
    lower_gap = ordered_lower[..., -1]-ordered_lower[..., -2]
    upper_gap = ordered_upper[..., 1]-ordered_upper[..., 0]
    diagnostics = {"support_lower_branch_gap": lower_gap,
                   "support_upper_branch_gap": upper_gap,
                   "support_lower_candidate_index": torch.where(feasible, low_candidates.argmax(-1), -1),
                   "support_upper_candidate_index": torch.where(feasible, high_candidates.argmin(-1), -1),
                   "support_discriminant_min": discriminant.amin(-1),
                   "support_regular_feasible": (feasible & (upper > lower) & positive.all(-1)
                                                & (lower_gap > 0) & (upper_gap > 0)),
                   "support_discriminant_boundary": (discriminant == 0).any(-1),
                   "support_width_before_infeasible_mask": upper-lower}
    return interval, diagnostics


def native_observation(x: Tensor, z: Tensor, command: Tensor, raw_design: Tensor,
                       interval: FeasibleInterval, context: NativeContext) -> Tensor:
    """Exactly the native 17-D information; no goal or disturbance is added."""
    cfg = context.cfg
    bounds = torch.stack((interval.lo, interval.hi), -1)
    bounds = torch.where(interval.feasible[..., None], bounds/cfg.input_bound, 0.)
    scale = x.new_tensor([1., 1., cfg.input_bound])
    return torch.cat((x/scale, z/scale, (x-z)/scale,
                      command[..., :1]/.45, (command[..., 1:]-1.)/.2,
                      2*torch.sigmoid(raw_design)-1, bounds), -1)


def fallback_action(x: Tensor, znext: Tensor,
                    context: NativeContext) -> tuple[Tensor, dict[str, Tensor]]:
    """Native scalar minimax fallback; selected-branch derivatives only."""
    cfg = context.cfg
    P, B, E = context.geometry_like(x)
    offset = physical_drift(x, cfg)-znext
    endpoints = torch.stack((offset-E*cfg.disturbance_bound,
                             offset+E*cfg.disturbance_bound), -2)
    # Flatten only leading batch axes to preserve the native einsum layout.
    shape = x.shape[:-1]
    flat_endpoints = endpoints.reshape(-1, 2, 3)
    a = B @ P @ B
    b = 2*torch.einsum("i,ij,bkj->bk", B, P, flat_endpoints)
    c = torch.einsum("bki,ij,bkj->bk", flat_endpoints, P, flat_endpoints)
    diff = b[:, 0]-b[:, 1]
    crossing_active = diff.abs() > 1e-20
    crossing = torch.where(crossing_active,
                           (c[:, 1]-c[:, 0])/torch.where(crossing_active, diff, 1.), 0.)
    candidates = torch.stack((-b[:, 0]/(2*a), -b[:, 1]/(2*a), crossing,
                               torch.full_like(crossing, -cfg.input_bound),
                               torch.full_like(crossing, cfg.input_bound)), -1).clamp(-cfg.input_bound, cfg.input_bound)
    values = a*candidates[:, :, None].square()+b[:, None, :]*candidates[:, :, None]+c[:, None, :]
    worst = values.amax(-1)
    indices = worst.argmin(-1)
    selected = candidates.gather(1, indices[:, None]).squeeze(1)
    ordered = worst.sort(-1).values
    diagnostics = {"fallback_candidate_index": indices.reshape(shape),
                   "fallback_objective_gap": (ordered[:, 1]-ordered[:, 0]).reshape(shape),
                   "fallback_crossing_branch_margin": (diff.abs()-1e-20).reshape(shape)}
    return selected.reshape(shape), diagnostics


def _broadcast_native(x: Tensor, z: Tensor, command: Tensor,
                      design: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    leading = torch.broadcast_shapes(x.shape[:-1], z.shape[:-1], command.shape[:-1], design.shape[:-1])
    return tuple(torch.broadcast_to(value, leading+(width,))
                 for value, width in ((x, 3), (z, 3), (command, 2), (design, 4)))


def native_step(actor_mean: MeanActor, x: Tensor, z: Tensor, command: Tensor,
                raw_design: Tensor, goal: Tensor | float, absolute_tick: int,
                context: NativeContext, *, support: bool,
                disturbance: Tensor | float | None = None) -> dict[str, Tensor]:
    """One actual closed-loop prediction step, with no reference reset.

    actor_mean(obs) returns latent mu of shape (...,) or (...,1). All leading
    dimensions are independent batch/candidate/scenario axes. command contains
    (c,eta) in execution units; goals are the planner's available preview.
    disturbance=None means nominal zero, never a hidden environment phase.
    """
    x, z, command, raw_design = _broadcast_native(x, z, command, raw_design)
    cfg = context.cfg
    P, B, E = context.geometry_like(x)
    znext, uref, raw_uref = reference_step(z, command, raw_design, context)
    f0 = physical_drift(x, cfg)
    interval, interval_info = support_interval(f0, znext, x-z, context)
    obs = native_observation(x, z, command, raw_design, interval, context)
    mean = actor_mean(obs)
    if mean.shape == obs.shape[:-1]+(1,):
        mean = mean.squeeze(-1)
    if mean.shape != obs.shape[:-1]:
        raise ValueError("actor_mean must return one latent scalar per native observation")
    normalized = torch.tanh(mean)
    emergency, fallback_info = fallback_action(x, znext, context)
    # Infeasible lo/hi are +/-inf in metadata. Mask them before arithmetic,
    # so the unselected action branch cannot create inf-inf in backward.
    safe_lo = torch.where(interval.feasible, interval.lo, 0.)
    safe_hi = torch.where(interval.feasible, interval.hi, 0.)
    midpoint, halfwidth = (safe_lo+safe_hi)/2, (safe_hi-safe_lo)/2
    mapped = midpoint+halfwidth*normalized
    action = torch.where(interval.feasible, mapped, emergency) if support else cfg.input_bound*normalized
    w = torch.zeros_like(action) if disturbance is None else torch.broadcast_to(
        torch.as_tensor(disturbance, dtype=x.dtype, device=x.device), action.shape)
    goal = torch.broadcast_to(torch.as_tensor(goal, dtype=x.dtype, device=x.device), action.shape)
    xnext = f0+B*action[..., None]+E*w[..., None]
    previous_value = _quadratic(x-z, P, x-z)
    value = _quadratic(xnext-znext, P, xnext-znext)
    excess = value-cfg.rho*previous_value-cfg.delta
    task = (xnext[..., 0]-goal).square()+.005*action.square()+.01*xnext[..., 1].square()
    reference_task = (znext[..., 0]-goal).square()+.005*uref.square()+.01*znext[..., 1].square()
    empty = ~interval.feasible
    total = task+cfg.violation_weight*excess.clamp_min(0)+cfg.empty_penalty*empty
    actual_margin = xnext.abs()-x.new_tensor(cfg.actual_bounds)
    reference_margin = znext.abs()-x.new_tensor(cfg.reference_bounds)
    input_margin = action.abs()-cfg.input_bound
    tube_margin = value-cfg.tube_budget
    center = f0+B*action[..., None]-znext
    worst_minus = center-E*cfg.disturbance_bound
    worst_plus = center+E*cfg.disturbance_bound
    worst_excess = torch.maximum(_quadratic(worst_minus, P, worst_minus),
                                  _quadratic(worst_plus, P, worst_plus))-cfg.rho*previous_value-cfg.delta
    geometric = torch.full_like(value, cfg.delta*(1-cfg.rho**(absolute_tick+1))/(1-cfg.rho))
    result = {"x_next": xnext, "z_next": znext, "obs": obs, "mean": mean,
              "command": command, "goal": goal, "disturbance": w,
              "action": action, "reference_input": uref, "reference_raw_input": raw_uref,
              "task": task, "reference_task_cost": reference_task, "total": total, "reward": -total,
              "V": value, "previous_V": previous_value, "excess": excess,
              "empty": empty.to(x.dtype), "fallback_used": empty & support,
              "interval_lo": interval.lo, "interval_hi": interval.hi,
              "interval_feasible": interval.feasible, "interval_discriminant": interval.discriminant,
              "support_width": safe_hi-safe_lo, "support_midpoint": midpoint,
              "support_halfwidth": halfwidth, "fallback_action": emergency,
              "position": xnext[..., 0], "reference_position": znext[..., 0], "target": goal,
              "actual_domain_margin": actual_margin, "reference_domain_margin": reference_margin,
              "input_margin": input_margin, "tube_margin": tube_margin,
              "support_worst_case_excess": worst_excess,
              "disturbance_assumption_margin": w.abs()-cfg.disturbance_bound,
              "actual_domain_violation": (xnext.abs() > x.new_tensor(cfg.actual_bounds)+1e-7).any(-1).to(x.dtype),
              "reference_domain_violation": (znext.abs() > x.new_tensor(cfg.reference_bounds)+1e-7).any(-1).to(x.dtype),
              "tube_violation": (value > cfg.tube_budget+1e-8).to(x.dtype),
              "geometric_bound": geometric, "geometric_bound_margin": value-geometric,
              "reference_clip_boundary": raw_uref.abs() == 1.,
              "reference_clip_active": raw_uref.abs() > 1.}
    result.update(interval_info)
    result.update(fallback_info)
    return result


def planner_coordinates_to_execution(commands: Tensor) -> Tensor:
    """Original native CEM coordinates: (c,s) -> (c,1+s*.2/.45)."""
    return torch.stack((commands[..., 0], 1.+commands[..., 1]*(.2/.45)), -1)


def predict_timeline(actor_mean: MeanActor, x0: Tensor, z0: Tensor,
                     command_timeline: Tensor, raw_design: Tensor, goal_preview: Tensor,
                     start_tick: int, context: NativeContext, *, support: bool,
                     disturbance_timeline: Tensor | None = None,
                     scenario_label: str | None = None) -> dict[str, Any]:
    """Predict an explicitly supplied low-level command timeline.

    command_timeline has shape (...,L,2), goal_preview (...,L), initial states
    (...,3), design (...,4), with broadcast-compatible leading axes. Outputs
    are time-major: x/z (L+1,...,3), all step records (L,...). No planning or
    command changes occur inside; this supports mid-macro initial ticks too.
    """
    horizon = command_timeline.shape[-2]
    if goal_preview.shape[-1] != horizon:
        raise ValueError("goal preview must contain one target for every low-level step")
    if disturbance_timeline is not None and disturbance_timeline.shape[-1] != horizon:
        raise ValueError("explicit disturbance scenario must match the timeline length")
    if disturbance_timeline is not None and not scenario_label:
        raise ValueError("an explicit disturbance timeline must have a diagnostic/scenario label")
    x, z, _, raw_design = _broadcast_native(x0, z0, command_timeline[..., 0, :], raw_design)
    xs, zs = [x], [z]
    records: dict[str, list[Tensor]] = {}
    for step in range(horizon):
        w = None if disturbance_timeline is None else disturbance_timeline[..., step]
        result = native_step(actor_mean, x, z, command_timeline[..., step, :], raw_design,
                             goal_preview[..., step], start_tick+step, context,
                             support=support, disturbance=w)
        for key, value in result.items():
            records.setdefault(key, []).append(value)
        x, z = result["x_next"], result["z_next"]
        xs.append(x)
        zs.append(z)
    stacked = {key: torch.stack(value, dim=0) for key, value in records.items()}
    return {"x": torch.stack(xs, dim=0), "z": torch.stack(zs, dim=0), "steps": stacked,
            "task_cost_mean": stacked["task"].mean(dim=0),
            "reference_task_cost_mean": stacked["reference_task_cost"].mean(dim=0),
            "task_cost_sum": stacked["task"].sum(dim=0),
            "total_cost_mean_diagnostic": stacked["total"].mean(dim=0),
            "start_tick": start_tick, "low_level_steps": horizon,
            "scenario_mode": "nominal_zero" if disturbance_timeline is None else "explicit_diagnostic",
            "scenario_label": scenario_label,
            "constraint_semantics": "Raw per-prefix native margins and original-tolerance flags; no aggregate feasibility rule or new threshold imposed"}


def predict_macro_commands(actor_mean: MeanActor, x0: Tensor, z0: Tensor,
                           execution_commands: Tensor, raw_design: Tensor, goal_preview: Tensor,
                           start_tick: int, context: NativeContext, *, support: bool,
                           disturbance_timeline: Tensor | None = None,
                           scenario_label: str | None = None) -> dict[str, Any]:
    """Planner wrapper for (B,N,H,2) execution-unit candidate sequences.

    x0/z0 are (B,3), design (B,4), goal preview (B,H*command_steps).
    An explicit scenario may be (B,L) shared across candidates or (B,N,L).
    This wrapper starts at a macro boundary; use predict_timeline otherwise.
    The original H=8 and command_steps=10 give 80 predicted low-level steps.
    """
    if execution_commands.ndim != 4 or execution_commands.shape[-1] != 2:
        raise ValueError("macro command candidates must have shape (B,N,H,2) in execution units")
    if start_tick % context.cfg.command_steps:
        raise ValueError("macro wrapper requires a command boundary; provide an explicit timeline otherwise")
    timeline = execution_commands.repeat_interleave(context.cfg.command_steps, dim=-2)
    goals = goal_preview[:, None, :]
    w = disturbance_timeline
    if w is not None and w.ndim == 2:
        w = w[:, None, :]
    return predict_timeline(actor_mean, x0[:, None, :], z0[:, None, :], timeline,
                            raw_design[:, None, :], goals, start_tick, context, support=support,
                            disturbance_timeline=w, scenario_label=scenario_label)


def prefix_violation_summary(prediction: dict[str, Any]) -> dict[str, Tensor]:
    """Separate native quantities; this function deliberately does not select a plan."""
    s = prediction["steps"]
    return {"actual_domain_margin_max": s["actual_domain_margin"].amax(dim=(0, -1)),
            "reference_domain_margin_max": s["reference_domain_margin"].amax(dim=(0, -1)),
            "input_margin_max": s["input_margin"].amax(dim=0),
            "tube_margin_max": s["tube_margin"].amax(dim=0),
            "actual_one_step_excess_max": s["excess"].amax(dim=0),
            "support_worst_case_excess_max": s["support_worst_case_excess"].amax(dim=0),
            "support_empty_any": (~s["interval_feasible"]).any(dim=0),
            "fallback_used_any": s["fallback_used"].any(dim=0),
            "reference_clip_boundary_any": s["reference_clip_boundary"].any(dim=0),
            "support_nonregular_feasible_any": (s["interval_feasible"] & ~s["support_regular_feasible"]).any(dim=0)}
