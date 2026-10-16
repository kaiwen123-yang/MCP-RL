"""Exact scalar-input one-step robust quadratic support for affine dynamics.

This is a standard control construction, not a new stability theorem. The
declared discrete model, disturbance bound and positive-definite metric are
assumptions; correctness for a different physical integrator is not implied.
"""

from __future__ import annotations

from typing import NamedTuple
import torch
from torch import Tensor


class FeasibleInterval(NamedTuple):
    lo: Tensor
    hi: Tensor
    feasible: Tensor
    discriminant: Tensor


def _quadratic(x: Tensor, P: Tensor, y: Tensor) -> Tensor:
    return torch.einsum("...i,...ij,...j->...", x, P, y)


def feasible_interval(
    f0: Tensor,
    B: Tensor,
    E: Tensor,
    znext: Tensor,
    error: Tensor,
    P: Tensor,
    rho: float | Tensor,
    delta: float | Tensor,
    W: float | Tensor,
    U: float | Tensor,
) -> FeasibleInterval:
    """Return all |u|<=U satisfying V(next error)<=rho*V(error)+delta.

    Dynamics: x_next=f0+B*u+E*w, w in [-W,W]. f0, B, E, znext and
    error broadcast to [...,n]; P is fixed symmetric positive definite [n,n]
    (batched metrics also broadcast). Scalar parameters broadcast to [...].
    The result has lo/hi/feasible shape [...] and discriminant [...,2] in
    disturbance order [-W,+W]. Caller guarantees W,U>=0 and the metric/model
    assumptions. Infeasibility is represented by lo=+inf, hi=-inf, feasible=False.
    No substitute action is returned or claimed safe.
    """
    f0, B, E, znext, error = torch.broadcast_tensors(f0, B, E, znext, error)
    def scalar(value):
        return torch.as_tensor(value, dtype=f0.dtype, device=f0.device)
    rho, delta, W, U = map(scalar, (rho, delta, W, U))
    offset = f0 - znext
    endpoints = torch.stack((offset - E * W[..., None], offset + E * W[..., None]), dim=-2)
    a = _quadratic(B, P, B)
    b = 2 * torch.stack((_quadratic(B, P, endpoints[..., 0, :]),
                          _quadratic(B, P, endpoints[..., 1, :])), dim=-1)
    budget = rho * _quadratic(error, P, error) + delta
    c = torch.stack((_quadratic(endpoints[..., 0, :], P, endpoints[..., 0, :]),
                     _quadratic(endpoints[..., 1, :], P, endpoints[..., 1, :])), dim=-1) - budget[..., None]
    discriminant = b.square() - 4 * a[..., None] * c

    # For SPD P, a==0 iff B==0; then b==0 and both inequalities are constant.
    controlled = a > 0
    safe_a = torch.where(controlled, a, torch.ones_like(a))[..., None]
    square_root = torch.sqrt(discriminant.clamp_min(0))
    # Stable quadratic roots: avoid cancellation when one root is near zero.
    q = -0.5 * (b + torch.where(b >= 0, square_root, -square_root))
    safe_q = torch.where(q != 0, q, torch.ones_like(q))
    root_1 = q / safe_a
    root_2 = torch.where(q != 0, c / safe_q, -b / (2 * safe_a))
    lower = torch.minimum(root_1, root_2).amax(dim=-1)
    upper = torch.maximum(root_1, root_2).amin(dim=-1)
    lower = torch.where(controlled, lower, torch.full_like(lower, -torch.inf))
    upper = torch.where(controlled, upper, torch.full_like(upper, torch.inf))
    lower = torch.maximum(lower, -U)
    upper = torch.minimum(upper, U)
    real_roots = (discriminant >= 0).all(dim=-1)
    constant_ok = (c <= 0).all(dim=-1)
    feasible = torch.where(controlled, real_roots, constant_ok) & (lower <= upper)
    return FeasibleInterval(
        torch.where(feasible, lower, torch.full_like(lower, torch.inf)),
        torch.where(feasible, upper, torch.full_like(upper, -torch.inf)),
        feasible,
        discriminant,
    )


def map_normalized_action(normalized_action: Tensor, interval: FeasibleInterval) -> Tensor:
    """Map s in [-1,1] to the support; infeasible entries return NaN.

    Pass shape [...] rather than [...,1]. Infeasible handling belongs to the
    caller. Singletons return their unique action and have no physical density.
    """
    midpoint = (interval.lo + interval.hi) / 2
    halfwidth = (interval.hi - interval.lo) / 2
    mapped = midpoint + halfwidth * normalized_action
    return torch.where(interval.feasible, mapped, torch.full_like(mapped, torch.nan))


def endpoint_residual(
    f0: Tensor, B: Tensor, E: Tensor, znext: Tensor, error: Tensor,
    P: Tensor, rho: float | Tensor, delta: float | Tensor,
    W: float | Tensor, action: Tensor,
) -> Tensor:
    """Maximum endpoint V excess; <=0 is the declared one-step inequality."""
    W = torch.as_tensor(W, dtype=f0.dtype, device=f0.device)
    center = f0 + B * action[..., None] - znext
    minus, plus = center - E * W[..., None], center + E * W[..., None]
    worst = torch.maximum(_quadratic(minus, P, minus), _quadratic(plus, P, plus))
    return worst - rho * _quadratic(error, P, error) - delta
