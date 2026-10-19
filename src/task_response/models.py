"""Compact response dynamics, distinct from the physical actuator dynamics."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .environment import Config, project_action


def mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))


class CompactResponse(nn.Module):
    """z=[position2, velocity2, learned response memory2].

    The memory is an effective response coordinate, NOT measured actuator truth.
    Known kinematics organize a learned response; no inverse plant controller,
    actor parameters, disturbance oracle, or certified input support is embedded.
    All arms use exactly this model and input information.
    """

    def __init__(self, config: Config | None = None, hidden: int = 32):
        super().__init__()
        self.config = config or Config()
        self.hidden = hidden
        self.memory_encoder = mlp(6, hidden, 2)
        self.response = mlp(14, hidden, 3)
        self.register_buffer("state_scale", torch.tensor([2., 1., 1., 1., 1.5, 1.5]))
        self.register_buffer("context_scale", torch.tensor([2., 1., 1., 1., 1., 4.]))
        self.register_buffer("feature_scale", torch.tensor([2., 1., 1., 1., 1.]))

    def encode(self, state: Tensor) -> Tensor:
        memory = self.config.input_bound * torch.tanh(self.memory_encoder(state / self.state_scale))
        return torch.cat((state[..., :4], memory), dim=-1)

    def step(self, state: Tensor, command: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        inputs = torch.cat((state / self.state_scale, command, context / self.context_scale), dim=-1)
        output = self.response(inputs)
        equilibrium = self.config.input_bound * torch.tanh(output[..., :2])
        position, velocity, memory = state.split(2, dim=-1)
        following = torch.cat((position + self.config.dt * velocity,
                               velocity + self.config.dt * memory,
                               memory + self.config.dt / self.config.tau * (equilibrium - memory)), dim=-1)
        # Conditional average of normalized squared input; not an upper bound
        # on any realized trajectory or a learned actuator command.
        resource = torch.sigmoid(output[..., 2])
        return following, resource

    def features(self, state: Tensor, resource: Tensor) -> Tensor:
        return torch.cat((state[..., :4], resource.unsqueeze(-1)), dim=-1) / self.feature_scale

    def observed_features(self, state: Tensor, action: Tensor) -> Tensor:
        resource = project_action(action, self.config).square().sum(-1) / self.config.input_bound**2
        return self.features(state, resource)


def actor_observation(state: Tensor, command: Tensor, context: Tensor) -> Tensor:
    """Same 14 observed inputs for every actor; no hidden disturbance preview."""
    state_scale = state.new_tensor([2., 1., 1., 1., 1.5, 1.5])
    context_scale = state.new_tensor([2., 1., 1., 1., 1., 4.])
    return torch.cat((state / state_scale, command, context / context_scale), dim=-1)
