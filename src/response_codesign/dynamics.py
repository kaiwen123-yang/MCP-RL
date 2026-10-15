"""Differentiable batched dynamics for the small closed-loop learning pilot.

The physical state is ``[position, velocity, actuator_output]``; the response
state is ``[reference_position, reference_velocity, reference_acceleration]``.
All public ``eta`` arguments are *raw logits* of four bounded design variables.
Torch CPU float32 batches are the intended execution mode; dtype/device are
preserved to permit derivative checks. No environment, planner or training
loop is implemented here.

The analytic teacher produces shared warmstart labels only. A learned actor
must produce the full physical input at deployment: it is not a residual on
top of this teacher. Its existence is not evidence that RL is necessary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class PhysicalConfig:
    """Physical integration and constraint parameters (seconds and SI units)."""

    dt: float = 0.02
    tau: float = 0.12
    k: float = 1.0
    d: float = 0.2
    drag: float = 0.1
    U: float = 1.2
    W: float = 0.04
    substeps: int = 2

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.tau <= 0 or self.U <= 0 or self.W < 0:
            raise ValueError("Require dt, tau and U > 0 and W >= 0.")
        if self.substeps < 2:
            raise ValueError("RK4 requires at least two substeps per control step.")


def _scalar_like(value: Tensor | float, reference: Tensor) -> Tensor:
    """Accept a scalar, batch scalar, or batch scalar with a trailing one."""
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim == reference.ndim + 1 and result.shape[-1] == 1:
        result = result.squeeze(-1)
    return result


def _rk4(
    state: Tensor,
    rhs: Callable[[Tensor], Tensor],
    dt: float,
    substeps: int,
) -> Tensor:
    if substeps < 2:
        raise ValueError("Use at least two RK4 substeps.")
    step = dt / substeps
    result = state
    for _ in range(substeps):
        first = rhs(result)
        second = rhs(result + 0.5 * step * first)
        third = rhs(result + 0.5 * step * second)
        fourth = rhs(result + step * third)
        result = result + (step / 6.0) * (first + 2 * second + 2 * third + fourth)
    return result


def physical_rhs(
    x: Tensor,
    u: Tensor | float,
    w: Tensor | float,
    cfg: PhysicalConfig,
) -> Tensor:
    """Actual nonlinear plant with saturated command and exogenous disturbance.

    ``w`` is applied as supplied, not clipped to ``cfg.W``. W states a nominal
    disturbance-domain assumption; out-of-domain stress tests must remain
    visible instead of being silently mapped back into the assumed domain.
    The controller receives no access to ``w`` through this function.
    """
    p, v, a = x.unbind(-1)
    control = _scalar_like(u, p).clamp(-cfg.U, cfg.U)
    disturbance = _scalar_like(w, p)
    acceleration = a - cfg.k * torch.sin(p) - cfg.d * v - cfg.drag * v * v.abs()
    return torch.stack(
        (v, acceleration + disturbance, (control - a) / cfg.tau), dim=-1
    )


def physical_step(
    x: Tensor,
    u: Tensor | float,
    w: Tensor | float,
    cfg: PhysicalConfig,
) -> Tensor:
    """Advance the plant one control step, holding u and w over all RK4 stages."""
    return _rk4(x, lambda state: physical_rhs(state, u, w, cfg), cfg.dt, cfg.substeps)


def response_design(eta: Tensor) -> Tensor:
    """Map raw logits to [omega, zeta, kappa, velocity_bandwidth_factor].

    The respective ranges are [1.2, 3.0], [0.7, 1.4], [3.0, 14.0] and
    [0.0, 0.5]. Finite raw logits lie strictly inside these ranges; bounds
    can be approached and may round to endpoints in finite precision.
    """
    lower = eta.new_tensor((1.2, 0.7, 3.0, 0.0))
    span = eta.new_tensor((1.8, 0.7, 11.0, 0.5))
    return lower + span * torch.sigmoid(eta)


def response_rhs(z: Tensor, c: Tensor | float, eta: Tensor) -> Tensor:
    """Third-order response under a held position command and bounded jerk.

    omega_eff = omega / (1 + factor * reference_velocity**2), so the
    response family includes state-dependent bandwidth. Jerk clipping is
    part of the response dynamics; gradients are the ordinary piecewise
    Torch derivatives and are not asserted smooth at clipping boundaries.
    """
    r, s, q = z.unbind(-1)
    command = _scalar_like(c, r)
    omega, zeta, kappa, factor = response_design(eta).unbind(-1)
    omega_eff = omega / (1.0 + factor * s.square())
    jerk = kappa * (omega_eff.square() * (command - r) - 2.0 * zeta * omega_eff * s - q)
    return torch.stack((s, q, jerk.clamp(-8.0, 8.0)), dim=-1)


def response_step(
    z: Tensor,
    c: Tensor | float,
    eta: Tensor,
    dt: float = 0.02,
    substeps: int = 2,
) -> Tensor:
    """Advance the response with differentiable RK4, holding c and raw eta."""
    return _rk4(z, lambda state: response_rhs(state, c, eta), dt, substeps)


def nominal_input(
    z: Tensor,
    c: Tensor | float,
    eta: Tensor,
    cfg: PhysicalConfig,
) -> Tensor:
    """Physical input required by exact nominal response, before saturation.

    This returned input is deliberately *not* clipped: feasibility checks
    must observe a requested input that exceeds U. It includes actuator
    lag and the derivative of the nonlinear drag term.
    """
    r, s, q = z.unbind(-1)
    jerk = response_rhs(z, c, eta)[..., 2]
    a_ref = cfg.k * torch.sin(r) + cfg.d * s + cfg.drag * s * s.abs() + q
    a_ref_dot = cfg.k * torch.cos(r) * s + (cfg.d + 2.0 * cfg.drag * s.abs()) * q + jerk
    return a_ref + cfg.tau * a_ref_dot


def actor_features(
    x: Tensor,
    z: Tensor,
    c: Tensor | float,
    eta: Tensor,
    target: Tensor | float,
) -> Tensor:
    """Return 12 features: x/scale, z/scale, c/1.5, eta_norm, target/1.5.

    State scales are [1.5, 2.0, 3.0]. Design features are exactly
    ``2*sigmoid(raw_eta)-1``, equivalently the bounded physical designs
    normalized by their interval endpoints. Features are not clipped;
    that would discard out-of-range state information. All comparison
    arms must receive these same features.
    """
    scale = x.new_tensor((1.5, 2.0, 3.0))
    batch_shape = torch.broadcast_shapes(x.shape[:-1], z.shape[:-1], eta.shape[:-1])
    x_features = torch.broadcast_to(x / scale, batch_shape + (3,))
    z_features = torch.broadcast_to(z / scale, batch_shape + (3,))
    design_features = torch.broadcast_to(2.0 * torch.sigmoid(eta) - 1.0, batch_shape + (4,))
    reference = x_features[..., 0]
    command_feature = torch.broadcast_to(_scalar_like(c, reference), batch_shape) / 1.5
    target_feature = torch.broadcast_to(_scalar_like(target, reference), batch_shape) / 1.5
    return torch.cat(
        (
            x_features,
            z_features,
            command_feature.unsqueeze(-1),
            design_features,
            target_feature.unsqueeze(-1),
        ),
        dim=-1,
    )


def teacher_action(
    x: Tensor,
    z: Tensor,
    c: Tensor | float,
    eta: Tensor,
    cfg: PhysicalConfig,
) -> Tensor:
    """Saturated full-action backstepping teacher for shared warmstart only.

    Gains are kp=4, kv=3, ka=8, rho=1. The teacher observes x,z,c,eta and
    known nominal physics but not the external disturbance or its time
    derivative. Nonlinear drag is retained in both the desired actuator
    output and its nominal derivative. Saturating the teacher does not
    establish the earlier unsaturated analytic realization certificate.
    """
    kp, kv, ka, rho = 4.0, 3.0, 8.0, 1.0
    p, v, a = x.unbind(-1)
    r, s, q = z.unbind(-1)
    position_error = p - r
    velocity_error = v - s
    gravity_drag = cfg.k * torch.sin(p) + cfg.d * v + cfg.drag * v * v.abs()
    nominal_acceleration = a - gravity_drag
    desired_actuator = gravity_drag + q - kp * position_error - kv * velocity_error
    jerk = response_rhs(z, c, eta)[..., 2]
    desired_actuator_derivative = (
        cfg.k * torch.cos(p) * v
        + (cfg.d + 2.0 * cfg.drag * v.abs()) * nominal_acceleration
        + jerk
        - kp * velocity_error
        - kv * (nominal_acceleration - q)
    )
    actuator_error = a - desired_actuator
    raw_input = a + cfg.tau * (
        desired_actuator_derivative - ka * actuator_error - rho * velocity_error
    )
    return raw_input.clamp(-cfg.U, cfg.U)

