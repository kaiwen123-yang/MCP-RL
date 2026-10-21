"""Actual-return PPO with optional CV or pathwise actor learning.

The ordinary clipped objective is unchanged. The CV importance ratio is NOT
clipped. Directions are calculated before any update using only pre-action
states, fixed plans and frozen old networks, never current-batch outcomes.
An optional pathwise loss is supplied by the runner. All trainable arms use
the same full-batch Gaussian KL acceptance and Adam rollback rule.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import torch

from .state import advance_memory, task_context


@contextmanager
def fixed_parameters(*modules):
    flags = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def actual_returns(rewards, next_values, terminated, truncated, gamma=1.0):
    """Monte Carlo suffixes, bootstrapping only an unfinished collection cut.

    Stage transitions are not boundaries. The original 35 s mission deadline
    is a terminal failure, not an artificial time-limit bootstrap.
    """
    # This scalar recurrence is cheaper on the CPU than thousands of CUDA
    # kernel launches and Python boolean synchronizations. Only the four small
    # return/termination vectors move if a caller already supplied CUDA data.
    output_device = rewards.device
    if output_device.type != "cpu":
        rewards, next_values, terminated, truncated = (
            tensor.detach().cpu()
            for tensor in (rewards, next_values, terminated, truncated))
    result = torch.empty_like(rewards)
    carry = rewards.new_zeros(())
    for t in range(len(rewards) - 1, -1, -1):
        if bool(terminated[t]):
            carry = rewards.new_zeros(())
        elif bool(truncated[t]) or t == len(rewards) - 1:
            carry = next_values[t]
        carry = rewards[t] + gamma * carry
        result[t] = carry
    return result.to(output_device)


def continuation_direction(actor, dynamics, value, states, memories, commands,
                           next_planned_commands, spec, steps, gamma=1.0):
    """Differentiate one fixed-command macro suffix plus the actual-return V.

    All rows in a call have the same number of steps (1..10). Future feedback
    uses the old mean actor, so autograd includes A+B*K, not just open-loop A.
    The terminal command comes from the already selected CEM plan. It is a
    local proxy, not a derivative through re-solving CEM or contact events.
    """
    start = perf_counter()
    output_device = states.device
    device = next(actor.parameters()).device
    states, memories, commands, next_planned_commands = (
        tensor.to(device)
        for tensor in (states, memories, commands, next_planned_commands))
    with fixed_parameters(actor, dynamics, value), torch.enable_grad():
        x, memory = states.detach(), memories.detach()
        command = commands.detach()
        context = task_context(x, memory, spec)
        first_mean = actor.mean(x, context, command).detach()
        first_raw = first_mean.clone().requires_grad_(True)
        proxy = states.new_zeros(len(states))
        alive = states.new_ones(len(states))
        for j in range(int(steps)):
            raw = first_raw if j == 0 else actor.mean(
                x, task_context(x, memory, spec), command)
            x = dynamics(x, raw)
            memory = advance_memory(x, memory, spec, dynamics.dt)
            proxy = proxy - (gamma ** j) * dynamics.dt * alive
            alive = alive * ((memory[..., 0] < 4) &
                             (memory[..., 1] < spec.timeout)).to(x.dtype)
        terminal = value(x, task_context(x, memory, spec),
                         next_planned_commands.detach())
        proxy = proxy + (gamma ** int(steps)) * alive * terminal
        direction = torch.autograd.grad(proxy.sum(), first_raw)[0]
    direction = direction.detach().to(output_device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return direction, {
        "adjoint_forward_steps": int(steps) * len(states),
        "adjoint_reverse_steps": int(steps) * len(states),
        "adjoint_seconds": perf_counter() - start,
    }


def ppo_objective(new_log_prob, old_log_prob, advantage, new_mean, old_mean,
                  raw_action, direction, clip=0.2, alpha=0.0):
    """Maximization surrogate; the CV has zero conditional population mean.

    The data, old mean, direction and coefficient are fixed during all epochs.
    No residual-advantage normalization and no clipped CV ratio are used.
    """
    ratio = torch.exp(new_log_prob - old_log_prob.detach())
    advantage = advantage.detach()
    ordinary = torch.minimum(ratio * advantage,
                             ratio.clamp(1 - clip, 1 + clip) * advantage)
    d, mu0 = direction.detach(), old_mean.detach()
    linear_sample = (d * (raw_action.detach() - mu0)).sum(-1)
    analytic_mean = (d * (new_mean - mu0)).sum(-1)
    correction = float(alpha) * (analytic_mean - ratio * linear_sample)
    return (ordinary + correction).mean(), {
        "ordinary_surrogate": ordinary.mean(),
        "cv_surrogate": correction.mean(),
        "linear_cv_rms": linear_sample.square().mean().sqrt(),
        "approx_kl": ((ratio - 1) - (new_log_prob - old_log_prob.detach())).mean(),
        "clip_fraction": ((ratio - 1).abs() > clip).float().mean(),
    }


@dataclass
class PPOSettings:
    epochs: int = 4
    minibatch_size: int = 256
    actor_lr: float = 3e-5
    value_lr: float = 3e-4
    clip: float = 0.2
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    entropy_coefficient: float = 0.0
    gamma: float = 1.0
    cv_alpha: float = 0.25


class PPOUpdater:
    """Shared transparent PyTorch PPO backend; not a Tianshou reproduction."""

    def __init__(self, actor, value, settings: PPOSettings, seed: int):
        self.actor, self.value, self.settings = actor, value, settings
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=settings.actor_lr)
        self.value_optimizer = torch.optim.Adam(value.parameters(), lr=settings.value_lr)
        self.rng = torch.Generator().manual_seed(seed)
        self.actor_updates = self.value_updates = 0

    def prepare(self, batch):
        returns = actual_returns(batch["reward"], batch["next_value"],
                                 batch["terminated"], batch["truncated"],
                                 self.settings.gamma)
        return returns, returns - batch["old_value"]

    def losses(self, batch, returns, advantage, direction, arm):
        distribution = self.actor.distribution(
            batch["state"], batch["context"], batch["command"])
        logp = distribution.log_prob(batch["raw_action"]).sum(-1)
        alpha = self.settings.cv_alpha if arm == "adjoint" else 0.0
        objective, metrics = ppo_objective(
            logp, batch["old_log_prob"], advantage, distribution.mean,
            batch["old_mean"], batch["raw_action"], direction,
            self.settings.clip, alpha)
        entropy = distribution.entropy().sum(-1).mean()
        actor_loss = -objective - self.settings.entropy_coefficient * entropy
        predicted = self.value(batch["state"], batch["context"], batch["command"],
                               batch["horizon"].to(batch["state"].dtype) / 10.0)
        value_loss = (predicted - returns.detach()).square().mean()
        return actor_loss, value_loss, {**metrics, "entropy": entropy,
                                      "value_loss": value_loss}

    @torch.no_grad()
    def _distribution_snapshot(self, batch):
        """Freeze the behavior distribution on all B states before any update."""
        means, deviations = [], []
        for begin in range(0, len(batch["state"]), self.settings.minibatch_size):
            part = slice(begin, begin + self.settings.minibatch_size)
            distribution = self.actor.distribution(
                batch["state"][part], batch["context"][part], batch["command"][part])
            means.append(distribution.mean.detach().clone())
            deviations.append(distribution.stddev.detach().clone())
        return torch.cat(means), torch.cat(deviations)

    @torch.no_grad()
    def _full_batch_kl(self, batch, old_mean, old_std):
        """Mean KL(old || candidate), summing all action dimensions per state."""
        total = old_mean.new_zeros((), dtype=torch.float64)
        for begin in range(0, len(old_mean), self.settings.minibatch_size):
            part = slice(begin, begin + self.settings.minibatch_size)
            distribution = self.actor.distribution(
                batch["state"][part], batch["context"][part], batch["command"][part])
            mean, std = distribution.mean.double(), distribution.stddev.double()
            previous_mean, previous_std = old_mean[part].double(), old_std[part].double()
            divergence = (torch.log(std / previous_std)
                          + (previous_std.square() + (previous_mean - mean).square())
                          / (2.0 * std.square()) - 0.5)
            total = total + divergence.sum()
        return float(total / len(old_mean))

    def update(self, batch, direction, arm, actor_auxiliary=None):
        """Update with full-B KL acceptance; rejected Adam steps are rolled back.

        The optional pathwise callback returns (scalar loss, scalar metrics).
        It runs once per attempted actor minibatch, not again during step-size
        retries. All retries use the same clipped gradient and pre-step Adam
        state. Six candidate attempts include the original configured rate.
        """
        start = perf_counter()
        # Collection uses CPU network copies; transfer the stored batch once,
        # before the minibatch loop. Keep the seeded shuffle stream on CPU.
        device = next(self.actor.parameters()).device
        returns, advantage = self.prepare(batch)
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        direction = direction.to(device)
        returns, advantage = returns.to(device), advantage.to(device)
        n, cfg = len(returns), self.settings
        before_actor, before_value = self.actor_updates, self.value_updates
        stopped = False
        last = {}
        attempts = rejected = auxiliary_calls = 0
        auxiliary_loss_total = returns.new_zeros(())
        auxiliary_metrics = {}
        full_batch_kl = max_accepted_kl = last_attempt_kl = 0.0
        last_accepted_lr = min_accepted_lr = None
        old_mean, old_std = self._distribution_snapshot(batch) if arm != "frozen" else (None, None)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = cfg.actor_lr
        try:
            for _ in range(cfg.epochs):
                permutation = torch.randperm(n, generator=self.rng).to(device)
                for index in permutation.split(cfg.minibatch_size):
                    sub = {k: v[index] for k, v in batch.items()}
                    actor_loss, value_loss, metrics = self.losses(
                        sub, returns[index], advantage[index], direction[index], arm)
                    if arm != "frozen" and not stopped:
                        if arm == "pathwise" and actor_auxiliary is not None:
                            auxiliary_loss, supplied_metrics = actor_auxiliary()
                            actor_loss = actor_loss + auxiliary_loss
                            auxiliary_calls += 1
                            auxiliary_loss_total = auxiliary_loss_total + auxiliary_loss.detach()
                            auxiliary_metrics = {
                                f"actor_auxiliary_{key}": (
                                    value.detach() if isinstance(value, torch.Tensor) else value)
                                for key, value in supplied_metrics.items()
                            }
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(), cfg.max_grad_norm,
                            error_if_nonfinite=True)
                        actor_before = copy.deepcopy(self.actor.state_dict())
                        optimizer_before = copy.deepcopy(self.actor_optimizer.state_dict())
                        accepted = False
                        for retry in range(6):
                            step_lr = cfg.actor_lr * (0.5 ** retry)
                            for group in self.actor_optimizer.param_groups:
                                group["lr"] = step_lr
                            attempts += 1
                            try:
                                self.actor_optimizer.step()
                                last_attempt_kl = self._full_batch_kl(batch, old_mean, old_std)
                                accepted = (isfinite(last_attempt_kl)
                                            and last_attempt_kl <= cfg.target_kl)
                            finally:
                                if not accepted:
                                    self.actor.load_state_dict(actor_before)
                                    # load_state_dict can share state tensors with its input.
                                    self.actor_optimizer.load_state_dict(copy.deepcopy(optimizer_before))
                            if accepted:
                                self.actor_updates += 1
                                full_batch_kl = last_attempt_kl
                                max_accepted_kl = max(max_accepted_kl, full_batch_kl)
                                last_accepted_lr = step_lr
                                min_accepted_lr = (step_lr if min_accepted_lr is None
                                                   else min(min_accepted_lr, step_lr))
                                break
                            rejected += 1
                        stopped = not accepted
                        for group in self.actor_optimizer.param_groups:
                            group["lr"] = cfg.actor_lr
                    self.value_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.value.parameters(), cfg.max_grad_norm,
                        error_if_nonfinite=True)
                    self.value_optimizer.step()
                    self.value_updates += 1
                    last = {k: v.detach() for k, v in metrics.items()}
        finally:
            for group in self.actor_optimizer.param_groups:
                group["lr"] = cfg.actor_lr
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        last = {key: float(value) for key, value in last.items()}
        auxiliary_metrics = {key: float(value) for key, value in auxiliary_metrics.items()}
        # Preserve the old field for callers, but give it the accepted analytic
        # KL, not the unrelated last minibatch's Monte Carlo diagnostic.
        last["last_minibatch_approx_kl"] = last.pop("approx_kl", 0.0)
        return {**last, **auxiliary_metrics,
                "approx_kl": full_batch_kl,
                "full_batch_kl": full_batch_kl,
                "max_accepted_full_batch_kl": max_accepted_kl,
                "last_attempt_full_batch_kl": last_attempt_kl,
                "actor_updates": self.actor_updates - before_actor,
                "actor_step_attempts": attempts,
                "actor_step_rejections": rejected,
                "actor_step_max_attempts": 6,
                "last_accepted_actor_lr": last_accepted_lr,
                "min_accepted_actor_lr": min_accepted_lr,
                "actor_auxiliary_calls": auxiliary_calls,
                "actor_auxiliary_loss_mean": (float(auxiliary_loss_total) / auxiliary_calls
                                               if auxiliary_calls else 0.0),
                "value_updates": self.value_updates - before_value,
                "kl_early_stop": stopped, "ppo_seconds": perf_counter() - start,
                "samples": n, "advantage_normalization": False}


def fit_facts(dynamics, response, dynamics_optimizer, response_optimizer,
              micro, macro, steps=128, batch_size=256, seed=0):
    """Supervised transitions only. The command model uses mean-policy data."""
    if not macro or not len(macro["state"]):
        raise RuntimeError("NO_CALIBRATION_TRANSITIONS: no complete 0.2 s mean-policy "
                           "transition; retain this failed branch without resampling")
    generator = torch.Generator().manual_seed(seed)
    start = perf_counter()
    dynamics_device = next(dynamics.parameters()).device
    response_device = next(response.parameters()).device
    micro = {key: micro[key].to(dynamics_device)
             for key in ("state", "raw_action", "next_state")}
    macro = {key: macro[key].to(response_device)
             for key in ("state", "context", "command", "next_state")}
    last_f = last_g = 0.0
    f_seconds = g_seconds = 0.0
    for _ in range(steps):
        part_start = perf_counter()
        index = torch.randint(len(micro["state"]), (min(batch_size, len(micro["state"])),),
                              generator=generator).to(dynamics_device)
        dynamics_optimizer.zero_grad(set_to_none=True)
        f_loss = dynamics.loss(micro["state"][index], micro["raw_action"][index],
                               micro["next_state"][index].detach())
        f_loss.backward()
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0, error_if_nonfinite=True)
        dynamics_optimizer.step()
        last_f = float(f_loss.detach())
        f_seconds += perf_counter() - part_start
        part_start = perf_counter()
        index = torch.randint(len(macro["state"]), (min(batch_size, len(macro["state"])),),
                              generator=generator).to(response_device)
        response_optimizer.zero_grad(set_to_none=True)
        g_loss = response.loss(macro["state"][index], macro["context"][index],
                               macro["command"][index], macro["next_state"][index].detach())
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(response.parameters(), 1.0, error_if_nonfinite=True)
        response_optimizer.step()
        last_g = float(g_loss.detach())
        g_seconds += perf_counter() - part_start
    return {"action_model_loss": last_f, "command_model_loss": last_g,
            "action_model_updates": steps, "command_model_updates": steps,
            "action_model_fitting_seconds": f_seconds,
            "command_model_fitting_seconds": g_seconds,
            "fitting_seconds": perf_counter() - start,
            "micro_samples": len(micro["state"]), "macro_samples": len(macro["state"]),
            "fitting_has_task_cost": False}
