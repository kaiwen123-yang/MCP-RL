"""Compact factual dynamics of one frozen Go2 feedback policy.

Inputs: newest-first 6x45 policy history, 42 current sensors and six current
context values. Context is [body vx, body vy, WORLD angular z, roll,pitch,height].
The model predicts local planar displacement/yaw change and the next planar
twist every 0.2s. It does not predict future contact, height, tilt or joint sweep.
No task objective, policy parameter, simulator or optimizer is embedded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


INPUT_DIM = 318
CONTEXT_START = 312


def wrap_angle(value: Tensor) -> Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


@dataclass
class CompactState:
    pose: Tensor       # [B,3]: world x/y/yaw
    twist: Tensor      # [B,3]: body vx/vy, world angular z
    hidden: Tensor     # [B,32]: learned response memory

    def as_state(self, index: int = 0) -> dict:
        """Planar task view only; no copied future height/tilt/contact facts."""
        pose = self.pose[index].detach().cpu().numpy().astype(float)
        twist = self.twist[index].detach().cpu().numpy().astype(float)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        world_xy = np.array([c*twist[0]-s*twist[1], s*twist[0]+c*twist[1]])
        return {"position": pose[:2], "yaw": float(pose[2]),
                "linear_velocity_world": world_xy, "linear_velocity_body": twist[:2],
                "angular_velocity_world": np.array([0., 0., twist[2]])}


class CompactResponse(nn.Module):
    def __init__(self, normalization: dict | None = None, macro_dt: float = .2,
                 hidden_size: int = 32, encoder_width: int = 64):
        super().__init__()
        self.macro_dt, self.hidden_size, self.encoder_width = float(macro_dt), hidden_size, encoder_width
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, encoder_width), nn.Tanh(),
                                     nn.Linear(encoder_width, hidden_size), nn.Tanh())
        self.transition = nn.GRUCell(6, hidden_size)
        self.decoder = nn.Linear(hidden_size, 6)
        # Both arms begin as the exact same ideal-command-response baseline.
        nn.init.zeros_(self.decoder.weight); nn.init.zeros_(self.decoder.bias)
        normalization = normalization or {}
        for name, default in (("input_mean", np.zeros(INPUT_DIM)), ("input_scale", np.ones(INPUT_DIM)),
                              ("command_mean", np.zeros(3)), ("command_scale", np.ones(3)),
                              ("residual_scale", np.ones(6))):
            self.register_buffer(name, torch.as_tensor(normalization.get(name, default), dtype=torch.float32))

    def encode(self, history: Tensor, sensors: Tensor, context: Tensor) -> Tensor:
        values = torch.cat((history.flatten(-2), sensors, context), dim=-1)
        return self.encoder((values-self.input_mean)/self.input_scale)

    def start(self, history: Tensor, sensors: Tensor, context: Tensor, pose: Tensor) -> CompactState:
        return CompactState(pose, context[..., :3], self.encode(history, sensors, context))

    def step(self, state: CompactState, command: Tensor) -> CompactState:
        normalized_twist = (state.twist-self.input_mean[CONTEXT_START:CONTEXT_START+3]) / self.input_scale[CONTEXT_START:CONTEXT_START+3]
        inputs = torch.cat((normalized_twist, (command-self.command_mean)/self.command_scale), dim=-1)
        hidden = self.transition(inputs, state.hidden)
        residual = self.decoder(hidden)*self.residual_scale
        ideal = torch.cat((self.macro_dt*command, command), dim=-1)
        response = ideal+residual
        yaw = state.pose[..., 2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        dx, dy = response[..., 0], response[..., 1]
        following_pose = torch.stack((state.pose[..., 0]+c*dx-s*dy,
                                      state.pose[..., 1]+s*dx+c*dy,
                                      wrap_angle(yaw+response[..., 2])), dim=-1)
        return CompactState(following_pose, response[..., 3:], hidden)


def state_to_inputs(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exactly the collector's pre-command snapshot convention."""
    history = np.asarray(state["history_frames"], dtype=np.float32)
    sensors = np.concatenate((np.asarray(state["angular_velocity_body"])*.25,
                              state["projected_gravity"], state["joint_position"],
                              np.asarray(state["joint_velocity"])*.05, state["action"])).astype(np.float32)
    context = np.array([*state["linear_velocity_body"][:2], state["angular_velocity_world"][2],
                        *state["rpy"][:2], state["position"][2]], dtype=np.float32)
    pose = np.array([*state["position"][:2], state["yaw"]], dtype=np.float32)
    if history.shape != (6,45) or sensors.shape != (42,):
        raise ValueError("The compact checkpoint requires native 6x45 history and 42 current sensors")
    return history, sensors, context, pose


class CompactPredictor:
    """Frozen CPU inference adapter used by the common command planner.

    start() reads current real history once. All later step() calls use only
    predicted pose/twist/hidden and proposed commands, never future truth.
    The caller measures end-to-end timing; this class makes no 200ms promise.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved.get("format") != "go2-compact-factual-v1":
            raise ValueError("Expected a Go2 compact factual checkpoint")
        self.metadata = {k:v for k,v in saved.items() if k != "state_dict"}
        self.model = CompactResponse(macro_dt=saved["architecture"]["macro_dt"],
                                     hidden_size=saved["architecture"]["hidden_size"],
                                     encoder_width=saved["architecture"]["encoder_width"]).to(device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        self.device = torch.device(device)
        self.macro_dt = self.model.macro_dt
        self.counters = {"encoder_calls": 0, "response_steps": 0}

    @torch.inference_mode()
    def start(self, state: dict) -> CompactState:
        values = [torch.as_tensor(a, device=self.device).unsqueeze(0) for a in state_to_inputs(state)]
        self.counters["encoder_calls"] += 1
        return self.model.start(*values)

    @torch.inference_mode()
    def step(self, state: CompactState, command) -> CompactState:
        command = torch.as_tensor(command, dtype=state.pose.dtype, device=self.device)
        if command.ndim == 1: command = command.unsqueeze(0).expand(state.pose.shape[0], -1)
        self.counters["response_steps"] += state.pose.shape[0]
        return self.model.step(state, command)
