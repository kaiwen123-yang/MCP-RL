"""Ordered pass/stop requirements on a shared physical Go2 route.

The initial straight, turn, door, world, and reset are identical within each
pair. Requirements are visible before motion starts. Intermediate stopping
does not end the episode, reset the robot, or cut the actual return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from go2_robot_lab.task import Task, TaskSpec, make_cases, wrap_angle


@dataclass(frozen=True)
class ContinuationSpec(TaskSpec):
    checkpoints: tuple = ()  # (world y, "pass" or "stop") in route order
    extra_doors: tuple = ()

    @property
    def complete_stage(self):
        return 3 + len(self.checkpoints)


def resolve_cases(config):
    if "task_specs" not in config:
        return make_cases()
    return [ContinuationSpec(name=name, **values)
            for name, values in config["task_specs"].items()]


def make_task(spec):
    return ContinuationTask(spec) if isinstance(spec, ContinuationSpec) else Task(spec)


def checkpoint_index(memory, spec):
    return (memory[..., 0].long() - 3).clamp(0, len(spec.checkpoints) - 1)


def next_stop_y(memory, spec):
    stops = [next(y for y, mode in spec.checkpoints[i:] if mode == "stop")
             for i in range(len(spec.checkpoints))]
    return memory.new_tensor(stops)[checkpoint_index(memory, spec)]


def future_features(state, memory, spec):
    """Next two checkpoint positions, stop flags and required hold durations."""
    index = checkpoint_index(memory, spec)
    indices = torch.stack((index, (index + 1).clamp(max=len(spec.checkpoints) - 1)), -1)
    targets = state.new_tensor([[spec.goal_x, y] for y, _ in spec.checkpoints])[indices]
    delta = targets - state[..., None, :2]
    c, s = state[..., 5].cos()[..., None], state[..., 5].sin()[..., None]
    relative = torch.stack((c * delta[..., 0] + s * delta[..., 1],
                            -s * delta[..., 0] + c * delta[..., 1]), -1).flatten(-2) / 5
    stop = state.new_tensor([mode == "stop" for _, mode in spec.checkpoints])[indices]
    return torch.cat((relative, stop, stop * spec.stop_hold_seconds), -1)


def advance_checkpoint_memory(state, memory, spec, dt):
    stage = memory[..., 0]
    x, y, yaw = state[..., 0].double(), state[..., 1].double(), state[..., 5].double()
    angle = (yaw - torch.pi / 2)
    yaw_error = torch.atan2(angle.sin(), angle.cos()).abs()
    crossed = (((stage == 0) & (x >= spec.straight_length) & (y.abs() < 0.55))
               | ((stage == 1) & (y >= spec.turn_radius) & ((x - spec.goal_x).abs() < 0.55)
                  & (yaw_error < 0.65))
               | ((stage == 2) & (y >= spec.door_y + .07 + spec.footprint_half_length)
                  & ((x - spec.goal_x).abs() < spec.door_opening / 2)))
    index = checkpoint_index(memory, spec)
    target_y = memory.new_tensor([point[0] for point in spec.checkpoints])[index].double()
    stop = memory.new_tensor([point[1] == "stop" for point in spec.checkpoints])[index].bool()
    speed = torch.linalg.vector_norm(state[..., 6:8].double(), dim=-1)
    distance = torch.sqrt((x - spec.goal_x).square() + (y - target_y).square())
    settled = ((stage >= 3) & stop & (distance <= spec.goal_position_tolerance)
               & (speed <= spec.goal_speed_tolerance) & (yaw_error <= spec.goal_yaw_tolerance))
    hold = torch.where(settled, torch.where(memory[..., 2] < 0,
                       torch.zeros_like(stage), memory[..., 2] + dt), -torch.ones_like(stage))
    passed = ((stage >= 3) & ~stop & (y >= target_y)
              & ((x - spec.goal_x).abs() < 0.55) & (yaw_error < 0.65))
    finished = crossed | passed | (settled & (hold >= spec.stop_hold_seconds - 1e-8))
    next_stage = stage + finished.to(stage.dtype)
    hold = torch.where(finished, -torch.ones_like(hold), hold)
    elapsed = memory[..., 1] + dt
    in_time = elapsed <= spec.timeout + 1e-8
    proposed = torch.stack((torch.where(in_time, next_stage, stage), elapsed.clamp(max=spec.timeout),
                            torch.where(in_time, hold, memory[..., 2])), -1)
    alive = (stage < spec.complete_stage) & (memory[..., 1] < spec.timeout)
    return torch.where(alive[..., None], proposed, memory)


class ContinuationTask(Task):
    @property
    def stage(self):
        if self.stage_index < 3:
            return ("straight", "turn", "door")[self.stage_index]
        if self.stage_index >= self.spec.complete_stage:
            return "complete"
        return f"{self.spec.checkpoints[self.stage_index - 3][1]}_{self.stage_index - 3}"

    def obstacles(self):
        boxes = super().obstacles()
        s = self.spec
        for index, y in enumerate(s.extra_doors):
            for side, lo, hi in (("left", s.goal_x - s.corridor_half_width, s.goal_x - s.door_opening / 2),
                                  ("right", s.goal_x + s.door_opening / 2, s.goal_x + s.corridor_half_width)):
                boxes.append({"name": f"extra_door_{index}_{side}",
                              "pos": [(lo + hi) / 2, y, .65], "size": [(hi - lo) / 2, .07, .65]})
        return boxes

    def update(self, obs):
        pos = np.asarray(obs["position"])[:2]
        velocity = np.asarray(obs["velocity_world"])[:2]
        time, yaw = float(obs["time"]), float(obs["yaw"])
        speed = float(np.linalg.norm(velocity))
        yaw_error = abs(float(wrap_angle(yaw - np.pi / 2)))
        self.contact_failure |= bool(obs.get("obstacle_contact_ever", False))
        self.fallen_failure |= bool(obs.get("fallen_ever", False))
        self.progress = max(self.progress, float(self.project(pos)))
        before = self.stage
        reached = False
        if self.stage_index < 3:
            reached = ((self.stage_index == 0 and pos[0] >= self.spec.straight_length and abs(pos[1]) < .55)
                       or (self.stage_index == 1 and pos[1] >= self.spec.turn_radius
                           and abs(pos[0] - self.spec.goal_x) < .55 and yaw_error < .65)
                       or (self.stage_index == 2 and pos[1] >= self.spec.door_y + .07 + self.spec.footprint_half_length
                           and abs(pos[0] - self.spec.goal_x) < self.spec.door_opening / 2))
        elif self.stage_index < self.spec.complete_stage:
            target_y, requirement = self.spec.checkpoints[self.stage_index - 3]
            if requirement == "pass":
                reached = pos[1] >= target_y and abs(pos[0] - self.spec.goal_x) < .55 and yaw_error < .65
            else:
                settled = (np.linalg.norm(pos - [self.spec.goal_x, target_y]) <= self.spec.goal_position_tolerance
                           and speed <= self.spec.goal_speed_tolerance and yaw_error <= self.spec.goal_yaw_tolerance)
                if settled:
                    if self.stop_since is None:
                        self.stop_since = time
                    reached = time - self.stop_since >= self.spec.stop_hold_seconds - 1e-8
                else:
                    self.stop_since = None
        if reached and time <= self.spec.timeout + 1e-8:
            self.crossings.append({"stage": before, "time": time, "position": pos.tolist(),
                                   "velocity_world": velocity.tolist(), "yaw": yaw,
                                   "yaw_rate": float(obs["yaw_rate"])})
            self.stage_index += 1
            self.stop_since = None
        self.success = (self.stage_index == self.spec.complete_stage
                        and not self.contact_failure and not self.fallen_failure)
        self.last_metrics = {"stage": self.stage, "stage_index": self.stage_index,
                             "progress": self.progress, "success": self.success,
                             "contact_failure": self.contact_failure, "fallen_failure": self.fallen_failure,
                             "timeout": time >= self.spec.timeout,
                             "goal_distance": float(np.linalg.norm(pos - self.spec.goal)),
                             "planar_speed": speed, "goal_yaw_error": yaw_error,
                             "crossings": list(self.crossings)}
        return self.last_metrics
