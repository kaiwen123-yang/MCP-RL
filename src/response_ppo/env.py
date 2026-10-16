"""Explicit discrete-Euler servo and continuous-reference rollout environment.

The physical system is the declared discrete affine model, not the previous
RK4 plant. Supports are one-step conditions, not recursive safety certificates.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_discrete_are
import torch
from torch import Tensor

from response_codesign.dynamics import response_design
from response_codesign.splates_bridge import build_splates_planner
from .realization import feasible_interval, map_normalized_action


@dataclass(frozen=True)
class EnvironmentConfig:
    dt: float = .02
    tau: float = .12
    input_bound: float = 1.2
    disturbance_bound: float = .04
    rho: float = .98
    delta: float = 4e-6
    command_steps: int = 10
    horizon: int = 8
    candidates: int = 16
    iterations: int = 2
    target_steps: int = 64
    violation_weight: float = 100.
    empty_penalty: float = .1
    reference_bounds: tuple[float, float, float] = (.6311721, .9401247, 1.2)
    actual_bounds: tuple[float, float, float] = (.65, 1., 1.2)
    tube_budget: float = 2e-4
    device: str = "cpu"


def initial_design(device="cpu") -> Tensor:
    omega = np.sqrt(32 / 11)
    physical = torch.tensor([omega, 29 / (22 * omega), 11., .02], dtype=torch.float32, device=device)
    return torch.logit((physical - physical.new_tensor([1.2, .7, 3., 0.])) /
                      physical.new_tensor([1.8, .7, 11., .5]))


def metric_from_dare(config: EnvironmentConfig) -> tuple[Tensor, dict]:
    h = config.dt
    A = np.array([[1., h, 0.], [-h, 1 - .2*h, h], [0., 0., 1-h/config.tau]])
    B = np.array([[0.], [0.], [h/config.tau]])
    Q, R = np.diag([2., .5, .02]), np.array([[.05]])
    unscaled = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ unscaled @ B, B.T @ unscaled @ A)
    P = unscaled / np.trace(unscaled)
    eigenvalues = np.linalg.eigvals(A + B @ K)
    contraction = np.max(np.linalg.eigvals(np.linalg.solve(P, (A+B@K).T@P@(A+B@K))).real)
    return torch.tensor(P, dtype=torch.float32, device=config.device), {
        "A": A.tolist(), "B": B.tolist(), "Q": Q.tolist(), "R": R.tolist(),
        "P_trace_normalized": P.tolist(), "nominal_K": K.tolist(),
        "nominal_closed_loop_eigenvalue_magnitudes": np.abs(eigenvalues).tolist(),
        "nominal_metric_contraction": float(contraction),
        "note": "Nominal linearization calculation, not a nonlinear whole-tube certificate",
    }


class ResponseServo:
    observation_dimension = 17

    def __init__(self, config: EnvironmentConfig, raw_design: Tensor, seed: int, steps: int, support: bool):
        self.config, self.raw_design, self.seed, self.steps, self.support = config, raw_design, seed, steps, support
        self.batch = raw_design.shape[0]
        self.P, self.metric_info = metric_from_dare(config)
        self.B = raw_design.new_tensor([0., 0., config.dt/config.tau])
        self.E = raw_design.new_tensor([0., config.dt, 0.])
        gen = torch.Generator(device=config.device).manual_seed(seed)
        self.x = .015 * torch.randn(self.batch, 3, generator=gen, device=config.device)
        self.z = self.x.clone()
        n_goals = (steps + config.horizon * config.command_steps) // config.target_steps + 3
        magnitude = .15 + .15 * torch.rand(self.batch, n_goals, generator=gen, device=config.device)
        sign = torch.where(torch.rand(self.batch, n_goals, generator=gen, device=config.device) < .5, -1., 1.)
        self.goals = magnitude * sign
        self.phase = 2 * np.pi * torch.rand(self.batch, generator=gen, device=config.device)
        self.t = 0
        self.command = torch.zeros(self.batch, 2, device=config.device)
        self.command[:, 1] = 1.
        self.planner_failures = 0
        self.planner_instances = 0
        self.planner = build_splates_planner(
            self.sequence_objective, horizon=config.horizon, command_dim=2,
            command_bound=.45, n_samples=config.candidates, n_iterations=config.iterations,
            device=config.device,
        )
        self._prepare_command()

    def f0(self, state):
        p, v, a = state.unbind(-1)
        h, tau = self.config.dt, self.config.tau
        return torch.stack((p + h*v, v + h*(a-torch.sin(p)-.2*v-.1*v*v.abs()), (1-h/tau)*a), -1)

    def response(self, z, c, eta, raw_design):
        p, v, a = z.unbind(-1)
        omega, zeta, kappa, factor = response_design(raw_design).unbind(-1)
        omega_eff = omega * eta / (1 + factor*v.square())
        q = a - torch.sin(p) - .2*v - .1*v*v.abs()
        jerk = kappa * (omega_eff.square()*(c-p) - 2*zeta*omega_eff*v - q)
        uref = (a + self.config.tau*(torch.cos(p)*v + (.2+.2*v.abs())*q + jerk)).clamp(-1., 1.)
        return self.f0(z) + self.B * uref[..., None], uref

    def target(self, step):
        return self.goals[:, step // self.config.target_steps]

    @torch.no_grad()
    def sequence_objective(self, state, commands, task):
        z = state[:, None, :3].expand(-1, commands.shape[1], -1)
        design = state[:, None, 3:].expand(-1, commands.shape[1], -1)
        costs = z.new_zeros(z.shape[:-1])
        margin = z.new_full(z.shape[:-1], -torch.inf)
        start = int(task)
        for macro in range(self.config.horizon):
            c = commands[..., macro, 0]
            eta = 1. + commands[..., macro, 1] * (.2/.45)
            for low in range(self.config.command_steps):
                z, u = self.response(z, c, eta, design)
                goal = self.target(start + macro*self.config.command_steps + low)[:, None]
                costs += (z[..., 0]-goal).square() + .005*u.square() + .01*z[..., 1].square()
                # Check every low-level prefix, not only macro endpoints.
                limits = z.new_tensor(self.config.reference_bounds)
                margin = torch.maximum(margin, (z.abs()-limits).amax(-1))
        return costs/(self.config.horizon*self.config.command_steps), margin

    @torch.no_grad()
    def _prepare_command(self):
        result = self.planner.plan(torch.cat((self.z, self.raw_design), -1), self.t,
                                   seed=self.seed*10000 + self.t)
        # A planner-domain failure remains visible. The returned finite best
        # candidate is executed; it is not called a feasible/safe backup.
        self.command[:, 0] = result.action[:, 0]
        self.command[:, 1] = 1. + result.action[:, 1]*(.2/.45)
        self.planner_failures += (~result.feasible).sum().item()
        self.planner_instances += self.batch

    def interval(self):
        znext, uref = self.response(self.z, self.command[:, 0], self.command[:, 1], self.raw_design)
        interval = feasible_interval(self.f0(self.x), self.B, self.E, znext,
                                     self.x-self.z, self.P, self.config.rho, self.config.delta,
                                     self.config.disturbance_bound, self.config.input_bound)
        return znext, uref, interval

    @torch.no_grad()
    def observation(self):
        _, _, interval = self.interval()
        bounds = torch.stack((interval.lo, interval.hi), -1)
        bounds = torch.where(interval.feasible[:, None], bounds/self.config.input_bound, 0.)
        scale = self.x.new_tensor([1., 1., self.config.input_bound])
        return torch.cat((self.x/scale, self.z/scale, (self.x-self.z)/scale,
                          self.command[:, :1]/.45, (self.command[:, 1:]-1.)/.2,
                          2*torch.sigmoid(self.raw_design)-1, bounds), -1)

    def fallback(self, znext):
        """Exact scalar minimax of the two endpoint quadratics, within U.

        Candidate stationary points, pairwise crossing and input endpoints
        contain a minimizer. This minimizes violation, does not erase it.
        """
        offset = self.f0(self.x)-znext
        endpoints = torch.stack((offset-self.E*self.config.disturbance_bound,
                                 offset+self.E*self.config.disturbance_bound), 1)
        a = self.B@self.P@self.B
        b = 2*torch.einsum("i,ij,bkj->bk", self.B, self.P, endpoints)
        c = torch.einsum("bki,ij,bkj->bk", endpoints, self.P, endpoints)
        diff = b[:, 0]-b[:, 1]
        crossing = torch.where(diff.abs() > 1e-20, (c[:, 1]-c[:, 0])/torch.where(diff.abs()>1e-20,diff,1.), 0.)
        candidates = torch.stack((-b[:, 0]/(2*a), -b[:, 1]/(2*a), crossing,
                                  torch.full_like(crossing, -self.config.input_bound),
                                  torch.full_like(crossing, self.config.input_bound)), -1).clamp(-self.config.input_bound, self.config.input_bound)
        values = a*candidates[:, :, None].square() + b[:, None, :]*candidates[:, :, None] + c[:, None, :]
        index = values.amax(-1).argmin(-1)
        return candidates[torch.arange(self.batch, device=self.x.device), index]

    @torch.no_grad()
    def step(self, normalized_action):
        znext, uref, interval = self.interval()
        emergency = self.fallback(znext)  # same computation in every arm
        mapped = map_normalized_action(normalized_action, interval)
        action = torch.where(interval.feasible, mapped, emergency) if self.support else self.config.input_bound*normalized_action
        before = self.x-self.z
        time = self.t*self.config.dt
        disturbance = self.config.disturbance_bound*(.7*torch.sin(3.1*time+self.phase)+.3*torch.cos(7.3*time+2*self.phase))
        self.x = self.f0(self.x) + self.B*action[:, None] + self.E*disturbance[:, None]
        self.z = znext  # Never reset to actual state at a planning boundary.
        error = self.x-self.z
        old_v = torch.einsum("bi,ij,bj->b", before, self.P, before)
        value = torch.einsum("bi,ij,bj->b", error, self.P, error)
        excess = value-self.config.rho*old_v-self.config.delta
        task = (self.x[:, 0]-self.target(self.t)).square()+.005*action.square()+.01*self.x[:, 1].square()
        empty = ~interval.feasible
        total = task + self.config.violation_weight*excess.clamp_min(0) + self.config.empty_penalty*empty
        self.t += 1
        if self.t % self.config.command_steps == 0:
            self._prepare_command()
        metric = {"task": task, "total": total, "V": value, "excess": excess,
                  "empty": empty.float(), "action": action, "reference_input": uref,
                  "support_width": torch.where(interval.feasible, interval.hi-interval.lo, 0.),
                  "position": self.x[:, 0], "reference_position": self.z[:, 0],
                  "actual_domain_violation": (self.x.abs()>self.x.new_tensor(self.config.actual_bounds)+1e-7).any(-1).float(),
                  "reference_domain_violation": (self.z.abs()>self.z.new_tensor(self.config.reference_bounds)+1e-7).any(-1).float(),
                  "tube_violation": (value>self.config.tube_budget+1e-8).float(),
                  "target": self.target(self.t-1),
                  "geometric_bound": torch.full_like(value, self.config.delta*(1-self.config.rho**self.t)/(1-self.config.rho))}
        return self.observation(), -total, metric
