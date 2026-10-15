"""On-policy joint constrained updates. No gradient is invented through CEM."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from itertools import product
import numpy as np
import torch
from torch import nn


def execution_features(x: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """Same twelve observed features for every arm; no design/target/reference input."""
    c = command.squeeze(-1)
    p, v, a = x.unbind(-1)
    return torch.stack((p / 1.5, v / 2, a / 3, c, (p-c) / 1.5,
                        torch.sin(p), torch.cos(p)-1, v*v.abs()/4,
                        (a-torch.sin(p)-0.2*v)/3, p*v/3, c*v/2,
                        torch.ones_like(p)), dim=-1)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 32), nn.Tanh(),
                                 nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


def parameter_vector(actor):
    return torch.nn.utils.parameters_to_vector(actor.parameters()).detach().clone()


def set_parameter_vector(actor, values):
    torch.nn.utils.vector_to_parameters(values, actor.parameters())


def _box_quadratic(hessian, linear, cap):
    """Solve the at-most-three-constraint dual by enumerating box faces.

    A convex quadratic on a face has a stationary free-coordinate solution or
    reaches its minimum on a smaller face. Enumeration also covers singular
    Hessians (duplicated/zero constraint gradients). A failed KKT check raises
    instead of silently applying a numerical optimizer's unfinished iterate.
    """
    count = len(linear)
    if count > 3 or cap <= 0:
        raise ValueError("Pilot dual requires at most three constraints and a positive cap")
    if count == 0:
        return np.zeros(0), {"dual_solver": "box-face-enumeration", "dual_kkt_residual": 0.0,
                             "dual_kkt_relative": 0.0, "dual_success": True}
    scale = max(1.0, float(np.linalg.norm(linear, np.inf)),
                float(np.linalg.norm(hessian, np.inf))*cap)
    candidates = []
    for face in product((0, 1, 2), repeat=count):
        face = np.asarray(face)
        free = np.flatnonzero(face == 1)
        fixed = np.flatnonzero(face != 1)
        value = np.where(face == 2, cap, 0.0)
        if len(free):
            rhs = -linear[free]-hessian[np.ix_(free, fixed)]@value[fixed]
            solution = np.linalg.lstsq(hessian[np.ix_(free, free)], rhs, rcond=None)[0]
            value[free] = solution
        if np.any(value < -1e-10) or np.any(value > cap+1e-10):
            continue
        value = np.clip(value, 0.0, cap)
        gradient = hessian@value+linear
        residual = gradient.copy()
        lower = value <= 1e-10
        upper = value >= cap-1e-10
        residual[lower] = np.minimum(gradient[lower], 0.0)
        residual[upper] = np.maximum(gradient[upper], 0.0)
        kkt = float(np.linalg.norm(residual, np.inf))
        if kkt <= 1e-8*scale:
            objective = float(0.5*value@hessian@value+linear@value)
            candidates.append((objective, value, kkt))
    if not candidates:
        raise RuntimeError("No box-dual candidate passed its KKT conditions")
    _, value, kkt = min(candidates, key=lambda item: item[0])
    return value, {"dual_solver": "box-face-enumeration", "dual_kkt_residual": kkt,
                   "dual_kkt_relative": kkt/scale, "dual_success": True}


def proximal_direction(g, rows, values, inverse_metric, alpha, dual_cap=2.0):
    """Solve the small box dual of a slack-penalized, joint proximal QP.

    Primal: g'd + d'Md/(2 alpha) + cap*sum(s), A d+G <= s, s>=0.
    Positive constraint values mean violations. This is not a hard safety filter.
    """
    g = np.asarray(g, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    metric = np.asarray(inverse_metric, dtype=np.float64)
    hessian = alpha * ((rows * metric) @ rows.T)
    linear = alpha * ((rows * metric) @ g) - values
    lam, solver_info = _box_quadratic(hessian, linear, dual_cap)
    direction = -alpha*metric*(g+rows.T@lam)
    slack = np.maximum(values+rows@direction, 0)
    return direction, {"dual": lam.tolist(), "linear_slack": slack.tolist(),
                       **solver_info}


@dataclass
class UpdateConfig:
    action_std: float = 0.12
    design_std: float = 0.25
    alpha: float = 0.01
    trust_radius: float = 0.08
    design_metric_inverse: float = 8.0
    task_scale: float = 0.05
    tracking_budget: float = 0.0002
    input_residual_budget: float = 0.03


class JointLearner:
    """Real-task score estimates, separately constrained gradients and joint QP."""
    def __init__(self, actor, design_mean, mode, config=None):
        self.actor = actor
        self.mean = design_mean.detach().clone()
        self.mode = mode
        self.config = config or UpdateConfig()
        self.baseline = None
        self.previous_means = np.zeros(3)
        self.updates = 0
        self.last_update = {}

    def sample_design(self, batch, generator):
        if self.mode == "fixed":
            return self.mean.expand(batch, -1).clone()
        noise = torch.randn(batch, 4, generator=generator)
        return self.mean[None]+self.config.design_std*noise

    def update(self, rollout):
        cfg = self.config
        features = rollout["features"].detach()
        raw = rollout["raw_actions"].detach()
        costs = rollout["costs"].detach().double().numpy()
        steps, batch, _ = costs.shape
        scales = np.array([cfg.task_scale, cfg.tracking_budget, cfg.input_residual_budget])
        normalized = costs/scales[None, None, :]
        returns = np.flip(np.cumsum(np.flip(normalized/steps, axis=0), axis=0), axis=0).copy()
        total = normalized.mean(axis=0)
        progress = np.broadcast_to(np.arange(steps)[:, None, None]/steps, (steps, batch, 1))
        critic_features = np.concatenate((features.double().numpy(), progress), axis=-1)
        flat_features = critic_features.reshape(-1, critic_features.shape[-1])
        if self.baseline is None:
            predictions = np.zeros_like(returns)
        else:
            predictions = (flat_features@self.baseline).reshape(returns.shape)
        mean_actions = self.actor(features)
        log_probability = -0.5*((raw-mean_actions)/cfg.action_std).square()
        actor_gradients = []
        for channel in range(3):
            advantage = torch.as_tensor(returns[:, :, channel]-predictions[:, :, channel],
                                        dtype=features.dtype)
            loss = (log_probability*advantage).sum(dim=0).mean()
            derivatives = torch.autograd.grad(loss, tuple(self.actor.parameters()),
                                               retain_graph=channel < 2)
            actor_gradients.append(torch.cat([item.reshape(-1) for item in derivatives]).double().numpy())
        design_score = (rollout["design"].double().numpy()-self.mean.double().numpy())/cfg.design_std**2
        design_gradients = (total-self.previous_means).T@design_score/batch
        if self.mode == "fixed":
            design_gradients[:] = 0
        gradients = np.concatenate((np.stack(actor_gradients), design_gradients), axis=1)
        g = gradients[0]
        rows = gradients[1:]
        values = total.mean(axis=0)[1:]-1
        actor_count = len(actor_gradients[0])
        inverse_metric = np.r_[np.ones(actor_count), np.full(4, cfg.design_metric_inverse)]
        alpha = cfg.alpha
        info = {}
        for _ in range(12):
            if self.mode == "independent":
                direction = np.r_[-alpha*g[:actor_count], np.zeros(4)]
                info = {"dual": [0, 0], "linear_slack": np.maximum(values+rows@direction, 0).tolist()}
            elif self.mode == "fixed":
                local, info = proximal_direction(g[:actor_count], rows[:, :actor_count], values,
                                                 inverse_metric[:actor_count], alpha)
                direction = np.r_[local, np.zeros(4)]
            elif self.mode == "tuning_only":
                local, info = proximal_direction(g[actor_count:], rows[:, actor_count:], values,
                                                 inverse_metric[actor_count:], alpha)
                direction = np.r_[np.zeros(actor_count), local]
            elif self.mode == "block":
                direction = np.zeros_like(g)
                blocks = [("actor", np.arange(actor_count)),
                          ("response", np.arange(actor_count, actor_count+4))]
                if self.updates % 2:
                    blocks.reverse()
                block_updates = []
                for name, indices in blocks:
                    shifted_values = values+rows@direction
                    local, block_info = proximal_direction(g[indices], rows[:, indices],
                                                           shifted_values, inverse_metric[indices], alpha)
                    direction[indices] = local
                    block_updates.append({"block": name,
                                          "constraint_before": shifted_values.tolist(), **block_info})
                info = {"dual": block_updates[-1]["dual"], "dual_scope": "last-block",
                        "block_order": [name for name, _ in blocks], "block_updates": block_updates,
                        "dual_success": all(item["dual_success"] for item in block_updates),
                        "linear_slack": np.maximum(values+rows@direction, 0).tolist()}
            else:
                direction, info = proximal_direction(g, rows, values, inverse_metric, alpha)
            step_norm = np.sqrt(np.sum(direction**2/inverse_metric))
            if step_norm <= cfg.trust_radius:
                break
            alpha *= 0.5
        assert np.isfinite(direction).all(), "Non-finite joint update"
        assert step_norm <= cfg.trust_radius*(1+1e-6), "Trust-region backtracking did not finish"
        before = parameter_vector(self.actor)
        set_parameter_vector(self.actor, before+torch.as_tensor(direction[:actor_count], dtype=before.dtype))
        self.mean += torch.as_tensor(direction[actor_count:], dtype=self.mean.dtype)
        # Baseline is fitted only after gradients from this batch have been computed.
        regularizer = 1e-2*np.eye(flat_features.shape[1])
        self.baseline = np.linalg.solve(flat_features.T@flat_features+regularizer,
                                        flat_features.T@returns.reshape(-1, 3))
        self.previous_means = total.mean(axis=0)
        self.updates += 1
        self.last_update = {"mode": self.mode, "update": self.updates,
                            "task_cost": float(costs[:, :, 0].mean()),
                            "constraints": values.tolist(), "alpha": alpha,
                            "step_metric_norm": float(step_norm),
                            "actor_step_norm": float(np.linalg.norm(direction[:actor_count])),
                            "response_step_norm": float(np.linalg.norm(direction[actor_count:])),
                            "linear_constraint_after": (values+rows@direction).tolist(), **info}
        return self.last_update

    def clone(self, mode):
        return JointLearner(copy.deepcopy(self.actor), self.mean.clone(), mode, self.config)
