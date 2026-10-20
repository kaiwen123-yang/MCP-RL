"""Shared physical state and remaining-task context for the Go2 learner.

The physical vector contains measured quantities, not an autonomous Markov
claim about contact dynamics. Task memory carries the original ordered stage
and stop timer. ``advance_memory`` uses the original spatial gates; its Boolean
branches are only piecewise differentiable and supply no derivative of event
time or of changing which gate is active. Actual contact/fall failure flags
remain the responsibility of the real Task and episode collector.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor


STATE_DIM = 48
MEMORY_DIM = 3
CONTEXT_DIM = 19


def _field(spec, name):
    return spec[name] if isinstance(spec, Mapping) else getattr(spec, name)


def state_from_observation(obs) -> np.ndarray:
    """Return pos/rpy/world velocity/body omega/q/dq/last applied action."""
    return np.concatenate((
        obs["position"], obs["rpy"], obs["velocity_world"],
        obs["omega_body"], obs["joint_pos"], obs["joint_vel"],
        obs["action"],
    )).astype(np.float32)


def memory_from_task(obs, task) -> np.ndarray:
    """Read actual ordered task state after ``task.update(obs)``."""
    elapsed = float(obs["time"])
    hold = -1.0 if task.stop_since is None else elapsed - float(task.stop_since)
    return np.asarray([task.stage_index, elapsed, hold], dtype=np.float32)


def planar_state(state: Tensor) -> Tensor:
    """Return x/y/yaw/world vx/world vy/Euler yaw rate from physical state."""
    roll, pitch = state[..., 3], state[..., 4]
    omega_y, omega_z = state[..., 10], state[..., 11]
    yaw_rate = (torch.sin(roll) * omega_y + torch.cos(roll) * omega_z) / torch.cos(pitch)
    return torch.cat((state[..., :2], state[..., 5:6],
                      state[..., 6:8], yaw_rate.unsqueeze(-1)), dim=-1)


def task_context(state: Tensor, memory: Tensor, spec) -> Tensor:
    """Encode the same remaining-task information for every comparison arm.

    Columns: stage one-hot[4], body-relative xy of four gates / 5 metres[8],
    sin/cos final-yaw error[2], remaining-time fraction[1], elapsed-hold
    fraction[1], door opening in metres[1], body-planar velocity[2].
    Stage 4 has an all-zero one-hot. The hold fraction is zero before holding.
    The four original route gates remain visible in every stage; their order
    and the stage vector together identify the remaining task.
    """
    length = float(_field(spec, "straight_length"))
    radius = float(_field(spec, "turn_radius"))
    goal_x = length + radius
    gates = state.new_tensor([
        [length, 0.0],
        [goal_x, radius],
        [goal_x, float(_field(spec, "door_y")) + 0.07
         + float(_field(spec, "footprint_half_length"))],
        [goal_x, float(_field(spec, "goal_y"))],
    ])
    delta = gates - state[..., None, :2]
    yaw = state[..., 5]
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    body_x = cosine[..., None] * delta[..., 0] + sine[..., None] * delta[..., 1]
    body_y = -sine[..., None] * delta[..., 0] + cosine[..., None] * delta[..., 1]
    relative_gates = torch.stack((body_x, body_y), dim=-1).flatten(-2) / 5.0
    stage = memory[..., 0]
    one_hot = torch.stack([(stage == i).to(state.dtype) for i in range(4)], dim=-1)
    yaw_error = torch.pi / 2.0 - yaw
    orientation = torch.stack((torch.sin(yaw_error), torch.cos(yaw_error)), dim=-1)
    timeout = float(_field(spec, "timeout"))
    remaining = torch.clamp((timeout - memory[..., 1]) / timeout, 0.0, 1.0)
    hold = torch.clamp(memory[..., 2] / float(_field(spec, "stop_hold_seconds")), 0.0, 1.0)
    opening = torch.full_like(remaining, float(_field(spec, "door_opening")))
    vx, vy = state[..., 6], state[..., 7]
    body_velocity = torch.stack((cosine * vx + sine * vy,
                                 -sine * vx + cosine * vy), dim=-1)
    return torch.cat((one_hot, relative_gates, orientation,
                      torch.stack((remaining, hold, opening), dim=-1),
                      body_velocity), dim=-1)


def advance_memory(state_after: Tensor, memory: Tensor, spec, dt: float) -> Tensor:
    """Advance original gate/hold logic from one actual or predicted successor.

    At most one spatial stage is crossed per call, as in ``Task.update``.
    The first settled sample starts the timer at zero rather than adding dt.
    Stage 4 is absorbing, including its elapsed time and hold timer. This
    geometric memory does not certify collision-free or fall-free completion;
    the collector checks those persistent physical flags separately.
    """
    stage = memory[..., 0]
    # Task.update converts observed coordinates to NumPy float64 before its
    # gates. Promote only the event predicates likewise, so a float32 number
    # just below (for example) 2.35 is not silently compared to a rounded-down
    # float32 threshold and accepted as having crossed.
    event_state = state_after.to(torch.float64)
    x, y, yaw = event_state[..., 0], event_state[..., 1], event_state[..., 5]
    length = float(_field(spec, "straight_length"))
    radius = float(_field(spec, "turn_radius"))
    goal_x = length + radius
    goal_y = float(_field(spec, "goal_y"))
    yaw_error = torch.abs(torch.atan2(torch.sin(yaw - torch.pi / 2.0),
                                     torch.cos(yaw - torch.pi / 2.0)))
    reached = (
        ((stage == 0) & (x >= length) & (torch.abs(y) < 0.55))
        | ((stage == 1) & (y >= radius) & (torch.abs(x - goal_x) < 0.55)
           & (yaw_error < 0.65))
        | ((stage == 2)
           & (y >= float(_field(spec, "door_y")) + 0.07
              + float(_field(spec, "footprint_half_length")))
           & (torch.abs(x - goal_x) < float(_field(spec, "door_opening")) / 2.0))
    )
    next_stage = stage + reached.to(memory.dtype)
    distance = torch.sqrt((x - goal_x).square() + (y - goal_y).square())
    speed = torch.linalg.vector_norm(event_state[..., 6:8], dim=-1)
    settled = (
        (next_stage == 3)
        & (distance <= float(_field(spec, "goal_position_tolerance")))
        & (speed <= float(_field(spec, "goal_speed_tolerance")))
        & (yaw_error <= float(_field(spec, "goal_yaw_tolerance")))
    )
    elapsed = memory[..., 1] + dt
    continued_hold = torch.where(memory[..., 2] < 0.0,
                                 torch.zeros_like(elapsed), memory[..., 2] + dt)
    hold = torch.where(settled, continued_hold, -torch.ones_like(elapsed))
    complete = settled & (hold >= float(_field(spec, "stop_hold_seconds")))
    next_stage = torch.where(complete, torch.full_like(next_stage, 4.0), next_stage)
    advanced = torch.stack((next_stage, elapsed, hold), dim=-1)
    return torch.where((stage == 4).unsqueeze(-1), memory, advanced)
