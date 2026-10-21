"""Shared factual models and actual-return critic for the Go2 learning study.

The models accept physical units.  Fixed scales are model buffers rather than
statistics fitted on evaluation trajectories.  Optimizer steps live in the
trainer: these modules expose only forward maps and supervised losses.

State order (48): position, ZYX roll/pitch/yaw, world linear velocity, body
angular velocity, 12 joint positions, 12 joint velocities, 12 applied actions.
The learned maps are approximations, not differentiable copies of MuJoCo.
"""
from __future__ import annotations

import torch
from torch import nn


STATE_DIM = 48
CONTEXT_DIM = 19
ACTION_DIM = 12
COMMAND_DIM = 3
ACTION_CLIP = 100.0


def state_scales() -> torch.Tensor:
    """Fixed physical input/output scales, in the documented state order."""
    return torch.tensor(
        [5.0, 5.0, 1.0, 1.0, 1.0, 3.141592653589793,
         1.0, 1.0, 1.0, 3.0, 3.0, 3.0]
        + [1.0] * 12 + [10.0] * 12 + [5.0] * 12,
        dtype=torch.float32,
    )


def _mlp(inputs: int, outputs: int, hidden: int) -> nn.Sequential:
    network = nn.Sequential(
        nn.Linear(inputs, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, outputs),
    )
    # Initially each factual model is exactly its declared engineering prior,
    # and the critic contributes zero.  This is not a learned dynamics claim.
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    return network


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _rpy_rate(rpy: torch.Tensor, omega_body: torch.Tensor) -> torch.Tensor:
    """ZYX Euler rate; the upright locomotion chart excludes pitch = +/- pi/2."""
    roll, pitch = rpy[..., 0], rpy[..., 1]
    wx, wy, wz = omega_body.unbind(-1)
    sr, cr = torch.sin(roll), torch.cos(roll)
    tangent = torch.tan(pitch)
    return torch.stack((
        wx + sr * tangent * wy + cr * tangent * wz,
        cr * wy - sr * wz,
        (sr * wy + cr * wz) / torch.cos(pitch),
    ), dim=-1)


def _physical_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = prediction - target
    return torch.cat((error[..., :3], _wrap(error[..., 3:6]), error[..., 6:]), dim=-1)


class ActionDynamics(nn.Module):
    """20 ms action-conditioned factual transition, with exact action memory.

    The raw Gaussian sample is mapped through the real deployment clip before
    it enters the dynamics network.  Its original, unclipped sample remains the
    random variable used by the policy likelihood and linear control variate.
    """

    def __init__(self, dt: float = 0.02, hidden: int = 128):
        super().__init__()
        self.dt = float(dt)
        self.register_buffer("state_scale", state_scales())
        self.register_buffer("action_scale", torch.full((ACTION_DIM,), 5.0))
        self.register_buffer("delta_scale", self.state_scale[:36].clone() * self.dt)
        self.network = _mlp(STATE_DIM + ACTION_DIM, 36, hidden)

    @staticmethod
    def execute(raw_action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(raw_action, -ACTION_CLIP, ACTION_CLIP)

    def prior(self, state: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
        applied = self.execute(raw_action)
        position = state[..., :3] + self.dt * state[..., 6:9]
        rpy = _wrap(state[..., 3:6] + self.dt * _rpy_rate(state[..., 3:6], state[..., 9:12]))
        joints = state[..., 12:24] + self.dt * state[..., 24:36]
        return torch.cat((position, rpy, state[..., 6:12],
                          joints, state[..., 24:36], applied), dim=-1)

    def forward(self, state: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
        applied = self.execute(raw_action)
        prior = self.prior(state, raw_action)
        inputs = torch.cat((state / self.state_scale, applied / self.action_scale), dim=-1)
        physical = prior[..., :36] + self.delta_scale * self.network(inputs)
        physical = torch.cat((physical[..., :3], _wrap(physical[..., 3:6]),
                              physical[..., 6:]), dim=-1)
        return torch.cat((physical, applied), dim=-1)

    def loss(self, state: torch.Tensor, raw_action: torch.Tensor,
             actual_next_state: torch.Tensor) -> torch.Tensor:
        """Factual transition regression; no task objective enters this loss."""
        error = _physical_error(self(state, raw_action), actual_next_state.detach())
        return (error[..., :36] / self.state_scale[:36]).square().mean()


class CommandResponse(nn.Module):
    """200 ms command response of the currently executing feedback policy.

    Every physical/memory coordinate has a learned residual output.  The
    untrained non-planar identity prior is declared explicitly; it is not an
    assertion that present joint states are known future joint states.
    """

    def __init__(self, dt: float = 0.2, hidden: int = 128,
                 velocity_time_constant: float = 0.25,
                 yaw_time_constant: float = 0.20, context_dim: int = CONTEXT_DIM):
        super().__init__()
        self.dt = float(dt)
        self.velocity_time_constant = float(velocity_time_constant)
        self.yaw_time_constant = float(yaw_time_constant)
        self.register_buffer("state_scale", state_scales())
        self.register_buffer("command_scale", torch.tensor([0.70, 0.30, 0.90]))
        self.register_buffer("delta_scale", self.state_scale.clone() * self.dt)
        self.network = _mlp(STATE_DIM + context_dim + COMMAND_DIM, STATE_DIM, hidden)

    def prior(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        yaw = state[..., 5]
        c, s = torch.cos(yaw), torch.sin(yaw)
        desired_xy = torch.stack((c * command[..., 0] - s * command[..., 1],
                                  s * command[..., 0] + c * command[..., 1]), dim=-1)
        av = 1.0 - torch.exp(state.new_tensor(-self.dt / self.velocity_time_constant))
        aw = 1.0 - torch.exp(state.new_tensor(-self.dt / self.yaw_time_constant))
        velocity_xy = state[..., 6:8] + av * (desired_xy - state[..., 6:8])
        yaw_rate = _rpy_rate(state[..., 3:6], state[..., 9:12])[..., 2]
        next_yaw_rate = yaw_rate + aw * (command[..., 2] - yaw_rate)
        # Preserve body wx/wy in this engineering prior and choose wz so the
        # ZYX yaw derivative matches the original planar lag model.
        roll, pitch = state[..., 3], state[..., 4]
        next_wz = (torch.cos(pitch) * next_yaw_rate
                   - torch.sin(roll) * state[..., 10]) / torch.cos(roll)
        position = torch.cat((state[..., :2] + self.dt * velocity_xy,
                              state[..., 2:3]), dim=-1)
        rpy = torch.cat((state[..., 3:5],
                         _wrap(yaw + self.dt * next_yaw_rate).unsqueeze(-1)), dim=-1)
        velocity = torch.cat((velocity_xy, state[..., 8:9]), dim=-1)
        omega = torch.cat((state[..., 9:11], next_wz.unsqueeze(-1)), dim=-1)
        return torch.cat((position, rpy, velocity, omega, state[..., 12:]), dim=-1)

    def forward(self, state: torch.Tensor, context: torch.Tensor,
                command: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((state / self.state_scale, context,
                            command / self.command_scale), dim=-1)
        prediction = self.prior(state, command) + self.delta_scale * self.network(inputs)
        return torch.cat((prediction[..., :3], _wrap(prediction[..., 3:6]),
                          prediction[..., 6:]), dim=-1)

    def loss(self, state: torch.Tensor, context: torch.Tensor,
             command: torch.Tensor, actual_next_state: torch.Tensor) -> torch.Tensor:
        error = _physical_error(self(state, context, command), actual_next_state.detach())
        return (error / self.state_scale).square().mean()


class ValueNet(nn.Module):
    """Actual-return critic with task context and the held-command clock.

    ``phase`` is the number of policy steps until the next MPC update divided
    by ten.  It is 1.0 at a fresh macro boundary and must be supplied explicitly
    for microstep values, so an identical physical state is not assigned the
    same continuation under different remaining command-hold durations.
    """

    def __init__(self, hidden: int = 128, context_dim: int = CONTEXT_DIM):
        super().__init__()
        self.register_buffer("state_scale", state_scales())
        self.register_buffer("command_scale", torch.tensor([0.70, 0.30, 0.90]))
        self.network = _mlp(STATE_DIM + context_dim + COMMAND_DIM + 1, 1, hidden)

    def forward(self, state: torch.Tensor, context: torch.Tensor,
                command: torch.Tensor, phase=None) -> torch.Tensor:
        if phase is None:
            clock = torch.ones_like(state[..., :1])
        else:
            clock = torch.as_tensor(phase, dtype=state.dtype, device=state.device)
            if clock.shape != state.shape[:-1] + (1,):
                clock = torch.broadcast_to(clock, state.shape[:-1]).unsqueeze(-1)
        inputs = torch.cat((state / self.state_scale, context,
                            command / self.command_scale, clock), dim=-1)
        return self.network(inputs).squeeze(-1)

    def loss(self, state: torch.Tensor, context: torch.Tensor,
             command: torch.Tensor, actual_return: torch.Tensor, phase=None) -> torch.Tensor:
        prediction = self(state, context, command, phase=phase)
        target = actual_return.detach().reshape_as(prediction)
        return (prediction - target).square().mean()
