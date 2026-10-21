"""Full twelve-joint actor initialized from the public robot_lab export.

The task-conditioned actor retains the original ELU architecture and all its
trainable weights. New task-input columns start at zero, so the initial mean
equals the public feedback. Gaussian samples are recorded before the original
fixed action clip. PPO ratios use those raw samples, not a Gaussian density
invented for the clipped action. Loading or sampling performs no optimization.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from .state import CONTEXT_DIM


class TaskActor(nn.Module):
    def __init__(self, export_path: str | Path, merged_policy_config: dict,
                 log_std_init: float = -2.3, context_dim: int = CONTEXT_DIM):
        super().__init__()
        self.config = copy.deepcopy(merged_policy_config)
        if self.config.get("observations_history"):
            raise ValueError("TaskActor initializes the 45-observation robot_lab export, without history.")
        exported = torch.jit.load(str(export_path), map_location="cpu").eval()
        weights = exported.state_dict()
        expected = ((512, 45), (256, 512), (128, 256), (12, 128))
        for index, shape in zip((0, 2, 4, 6), expected):
            if tuple(weights[f"actor.{index}.weight"].shape) != shape:
                raise ValueError("Export does not match the robot_lab 45-512-256-128-12 actor.")
        self.actor = nn.Sequential(
            nn.Linear(45 + context_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12),
        )
        with torch.no_grad():
            self.actor[0].weight.zero_()
            self.actor[0].weight[:, :45].copy_(weights["actor.0.weight"])
            self.actor[0].bias.copy_(weights["actor.0.bias"])
            for index in (2, 4, 6):
                self.actor[index].weight.copy_(weights[f"actor.{index}.weight"])
                self.actor[index].bias.copy_(weights[f"actor.{index}.bias"])
        self.log_std = nn.Parameter(torch.full((12,), float(log_std_init)))
        self.register_buffer("default_q", torch.tensor(self.config["default_dof_pos"], dtype=torch.float32))
        self.register_buffer("command_scale", torch.tensor(self.config["commands_scale"], dtype=torch.float32))
        self.register_buffer("action_lower", torch.tensor(self.config["clip_actions_lower"], dtype=torch.float32))
        self.register_buffer("action_upper", torch.tensor(self.config["clip_actions_upper"], dtype=torch.float32))

    def _tensor(self, value) -> Tensor:
        return torch.as_tensor(value, device=self.log_std.device, dtype=self.log_std.dtype)

    def public_observation(self, state: Tensor, command: Tensor) -> Tensor:
        """Build original scaled/clipped 45-vector, with actual previous action.

        Gravity uses R(rpy)^T [0,0,-1], matching the original quaternion path.
        State and command gradients are retained; no nominal/model state is
        substituted for physical observations during collection.
        """
        state, command = self._tensor(state), self._tensor(command)
        roll, pitch = state[..., 3], state[..., 4]
        gravity = torch.stack((torch.sin(pitch),
                               -torch.sin(roll) * torch.cos(pitch),
                               -torch.cos(roll) * torch.cos(pitch)), dim=-1)
        cfg = self.config
        terms = {
            "ang_vel": state[..., 9:12] * float(cfg["ang_vel_scale"]),
            "gravity_vec": gravity,
            "commands": command * self.command_scale,
            "dof_pos": (state[..., 12:24] - self.default_q) * float(cfg["dof_pos_scale"]),
            "dof_vel": state[..., 24:36] * float(cfg["dof_vel_scale"]),
            "actions": state[..., 36:48],
        }
        observation = torch.cat([terms[name] for name in cfg["observations"]], dim=-1)
        return observation.clamp(-float(cfg["clip_obs"]), float(cfg["clip_obs"]))

    def mean(self, state: Tensor, context: Tensor, command: Tensor) -> Tensor:
        observation = self.public_observation(state, command)
        return self.actor(torch.cat((observation, self._tensor(context)), dim=-1))

    def distribution(self, state: Tensor, context: Tensor, command: Tensor) -> Normal:
        mean = self.mean(state, context, command)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def log_prob(self, state: Tensor, context: Tensor, command: Tensor,
                 raw: Tensor) -> Tensor:
        """Joint raw-action log density, summed over the twelve joints."""
        return self.distribution(state, context, command).log_prob(self._tensor(raw)).sum(-1)

    def execute(self, raw: Tensor) -> Tensor:
        """Original fixed deployment clip; q_des scaling remains in the simulator."""
        raw = self._tensor(raw)
        return torch.maximum(torch.minimum(raw, self.action_upper), self.action_lower)

    def sample(self, state: Tensor, context: Tensor, command: Tensor,
               deterministic: bool = False,
               generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Sample behavior, keeping the realized raw sample fixed for its score.

        The raw sample is detached: its likelihood-ratio gradient acts through
        the distribution parameters, not a reparameterized copy of the action.
        Deterministic output is for evaluation/calibration, not an on-policy
        Gaussian sample eligible for an ordinary PPO update.
        """
        distribution = self.distribution(state, context, command)
        mean = distribution.loc
        if deterministic:
            raw = mean.detach()
        else:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                                generator=generator)
            raw = (mean + distribution.scale * noise).detach()
        return {"raw": raw, "executed": self.execute(raw),
                "log_prob": distribution.log_prob(raw).sum(-1), "mean": mean}
