"""One differentiable mission proxy for planning and feedback learning.

The weights, route geometry and body-footprint penalty come from the frozen
mission controller. This is a piecewise differentiable control objective, not
an unbiased derivative of actual completion time. Discrete stage/hold events
are supplied by the caller at its rollout interval; 200 ms endpoints do not
certify continuous holding at the actual 20 ms evaluation interval.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from go2_frozen.fair_control import MISSION_WEIGHTS, SPEED_BANDS
from go2_frozen.task import Task

from .state import task_active


def _reference(progress, spec):
    length, radius = spec.straight_length, spec.turn_radius
    progress = progress.clamp(0.0, spec.total_length)
    angle = ((progress - length) / radius).clamp(0.0, torch.pi / 2.0)
    x = torch.where(progress <= length, progress, length + radius * angle.sin())
    y = torch.where(progress <= length, torch.zeros_like(progress),
                    radius * (1.0 - angle.cos()))
    y = torch.where(progress >= length + spec.arc_length,
                    radius + progress - length - spec.arc_length, y)
    return torch.stack((x, y, angle), dim=-1)


def route_progress(position, task):
    """Nearest route-piece projection, with derivatives within its chosen piece."""
    spec = getattr(task, "spec", task)
    length, radius = spec.straight_length, spec.turn_radius
    straight = position[..., 0].clamp(0.0, length)
    angle = torch.atan2(position[..., 0] - length, radius - position[..., 1])
    turn = length + radius * angle.clamp(0.0, torch.pi / 2.0)
    final = length + spec.arc_length + (position[..., 1] - radius).clamp(
        0.0, spec.goal_y - radius)
    candidates = torch.stack((straight, turn, final), dim=-1)
    references = _reference(candidates, spec)
    distances = (references[..., :2] - position[..., None, :2]).square().sum(-1)
    index = distances.argmin(dim=-1, keepdim=True)
    return candidates.gather(-1, index).squeeze(-1)


@lru_cache(maxsize=32)
def _time_table(length, radius, goal_y, tolerance, speed_band):
    """Original speed-profile quadrature ending at the legal goal-region entry."""
    total = length + np.pi * radius / 2.0 + goal_y - radius
    entry = total - tolerance
    turn_begin, turn_end = length - 0.35, length + np.pi * radius / 2.0 + 0.15
    edges = np.unique(np.r_[np.linspace(0.0, entry, 4097),
                            np.clip([turn_begin, turn_end], 0.0, entry)])
    midpoint = (edges[:-1] + edges[1:]) / 2.0
    cruise, turning = SPEED_BANDS[speed_band]
    speed = np.where((midpoint > turn_begin) & (midpoint < turn_end), turning, cruise)
    speed = np.minimum(speed, np.sqrt(np.maximum(0.0, 0.7 * (total - midpoint))))
    durations = np.diff(edges) / np.maximum(speed, 0.001)
    remaining = np.r_[np.cumsum(durations[::-1])[::-1], 0.0]
    return edges, remaining


def remaining_reference_time(position, task, speed_band="fast"):
    """Reference-route time to the position-tolerance region, retaining xy gradients.

    Route progress accounts for forward travel; the shared lateral and terminal
    distance penalties account for leaving the route or overshooting the goal.
    A position already inside the actual goal region has no remaining travel.
    """
    spec = getattr(task, "spec", task)
    edges, remaining = _time_table(spec.straight_length, spec.turn_radius,
                                   spec.goal_y, spec.goal_position_tolerance, speed_band)
    knots = position.new_tensor(edges)
    values = position.new_tensor(remaining)
    progress = route_progress(position, task).clamp(0.0, float(edges[-1]))
    right = torch.searchsorted(knots, progress.contiguous(), right=True).clamp(1, len(edges) - 1)
    left = right - 1
    fraction = (progress - knots[left]) / (knots[right] - knots[left])
    seconds = values[left] + fraction * (values[right] - values[left])
    goal = position.new_tensor((spec.goal_x, spec.goal_y))
    inside = torch.linalg.vector_norm(position[..., :2] - goal, dim=-1) <= spec.goal_position_tolerance
    return torch.where(inside, torch.zeros_like(seconds), seconds)


def footprint_penalty(states, task, collision_margin=0.04):
    """Original OBB/AABB separating-axis body penalty in Torch."""
    spec = getattr(task, "spec", task)
    geometry = task if hasattr(task, "obstacles") else Task(spec)
    x, y, yaw = states[..., 0], states[..., 1], states[..., 5]
    cosine, sine = yaw.cos(), yaw.sin()
    ac, ass = cosine.abs(), sine.abs()
    a = spec.footprint_half_length + collision_margin
    b = spec.footprint_half_width + collision_margin
    cost = torch.zeros_like(x)
    for obstacle in geometry.obstacles():
        ox, oy = obstacle["pos"][:2]
        hx, hy = obstacle["size"][:2]
        dx, dy = x - ox, y - oy
        gaps = torch.stack((
            dx.abs() - (hx + a * ac + b * ass),
            dy.abs() - (hy + a * ass + b * ac),
            (dx * cosine + dy * sine).abs() - (a + hx * ac + hy * ass),
            (-dx * sine + dy * cosine).abs() - (b + hx * ass + hy * ac),
        ), dim=-1)
        clearance = gaps.amax(dim=-1)
        cost = cost + 20000.0 * (0.015 - clearance).clamp_min(0.0).square()
        cost = cost + 2.0 * (0.08 - clearance).clamp_min(0.0).square()
    return cost


def mission_cost(states, memories, commands, initial_memory, previous_command,
                 task, dt, *, speed_band="fast", collision_margin=0.04):
    """Score post-step [B,T,48] states and [B,T,3] memories/issued commands.

    Each transition is charged only over its active time before the deadline.
    A transition into deadline failure receives the actual 2 * timeout penalty
    once; already terminated states receive no further cost. Terminal geometry
    is charged only for an active final state, and
    position/speed residuals vanish throughout their actual tolerance regions.
    The caller separately subtracts a continuation V for unfinished rollouts.
    Spatial, command and footprint costs retain their original 200 ms weights
    as rates over the active interval; the time cost uses that same interval.
    """
    spec = getattr(task, "spec", task)
    weight = MISSION_WEIGHTS
    predecessor_memory = torch.cat((initial_memory[:, None], memories[:, :-1]), dim=1)
    alive = task_active(predecessor_memory, spec)
    active_dt = (spec.timeout - predecessor_memory[..., 1]).clamp(0.0, float(dt))
    progress = route_progress(states[..., :2], task)
    reference = _reference(progress, spec)
    delta = states[..., :2] - reference[..., :2]
    lateral = -delta[..., 0] * reference[..., 2].sin() + delta[..., 1] * reference[..., 2].cos()
    yaw_delta = states[..., 5] - reference[..., 2]
    yaw_error = torch.atan2(yaw_delta.sin(), yaw_delta.cos())
    previous = torch.cat((previous_command[:, None], commands[:, :-1]), dim=1)
    spatial = weight["lateral_per_m2_per_step"] * lateral.square()
    spatial = spatial + weight["yaw_per_rad2_per_step"] * yaw_error.square()
    spatial = spatial + weight["command_squared_per_step"] * commands.square().sum(-1)
    spatial = spatial + weight["command_change_squared_per_step"] * (commands - previous).square().sum(-1)
    spatial = spatial + footprint_penalty(states, task, collision_margin)
    running = active_dt * weight["time_per_second"] + (active_dt / 0.2) * spatial
    total = torch.where(alive, running, torch.zeros_like(running)).sum(-1)
    deadline_failure = alive & (memories[..., 0] < getattr(spec, "complete_stage", 4)) & (memories[..., 1] >= spec.timeout)
    total = total + (2.0 * spec.timeout) * deadline_failure.to(states.dtype).sum(-1)

    terminal = states[:, -1]
    goal = terminal.new_tensor((spec.goal_x, spec.goal_y))
    distance = torch.linalg.vector_norm(terminal[:, :2] - goal, dim=-1)
    speed = torch.linalg.vector_norm(terminal[:, 6:8], dim=-1)
    position_excess = (distance - spec.goal_position_tolerance).clamp_min(0.0)
    speed_excess = (speed - spec.goal_speed_tolerance).clamp_min(0.0)
    terminal_cost = weight["remaining_reference_seconds_terminal"] * remaining_reference_time(
        terminal[:, :2], task, speed_band)
    near_goal = spec.total_length - progress[:, -1] < weight["terminal_goal_zone_m"]
    terminal_cost = terminal_cost + near_goal.to(states.dtype) * (
        weight["goal_position_per_m2_terminal"] * position_excess.square()
        + weight["goal_velocity_per_m2s2_terminal"] * speed_excess.square())
    if hasattr(spec, "checkpoints"):
        from .continuation import next_stop_y
        stop_y = next_stop_y(memories[:, -1], spec)
        stop_goal = torch.stack((torch.full_like(stop_y, spec.goal_x), stop_y), -1)
        stop_distance = torch.linalg.vector_norm(terminal[:, :2] - stop_goal, dim=-1)
        stop_progress = spec.straight_length + spec.arc_length + stop_y - spec.turn_radius
        intermediate = (stop_y < spec.goal_y) & (stop_progress - progress[:, -1] < weight["terminal_goal_zone_m"])
        terminal_cost = terminal_cost + intermediate.to(states.dtype) * (
            weight["goal_position_per_m2_terminal"] * (stop_distance - spec.goal_position_tolerance).clamp_min(0).square()
            + weight["goal_velocity_per_m2s2_terminal"] * speed_excess.square())
    unfinished = task_active(memories[:, -1], spec)
    return total + torch.where(unfinished, terminal_cost, torch.zeros_like(terminal_cost))
