"""Fixed, disturbance-free Go2 route and ordered whole-task evaluation.

Lengths are metres; obstacle ``size`` values are MuJoCo box half extents.
These cases are declared before robot evaluation, not selected by outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class TaskSpec:
    name: str = "nominal"
    start_y: float = 0.0
    straight_length: float = 2.35
    turn_radius: float = 0.65
    door_y: float = 2.0
    door_opening: float = 1.0
    goal_y: float = 3.6
    corridor_half_width: float = 0.9
    timeout: float = 35.0
    goal_position_tolerance: float = 0.22
    goal_speed_tolerance: float = 0.12
    goal_yaw_tolerance: float = 0.20
    stop_hold_seconds: float = 1.0
    footprint_half_length: float = 0.32
    footprint_half_width: float = 0.18

    @property
    def goal_x(self):
        return self.straight_length + self.turn_radius

    @property
    def arc_length(self):
        return np.pi * self.turn_radius / 2.0

    @property
    def total_length(self):
        return self.straight_length + self.arc_length + self.goal_y - self.turn_radius

    @property
    def start(self):
        return np.array([0.0, self.start_y, 0.0])

    @property
    def goal(self):
        return np.array([self.goal_x, self.goal_y])

    def to_dict(self):
        return asdict(self)


def make_cases():
    return [TaskSpec(), TaskSpec(name="narrow", door_opening=0.9, start_y=0.06),
            TaskSpec(name="wide", door_opening=1.1, start_y=-0.06)]


class Task:
    """Route geometry is stateless; update() records ordered actual crossings."""

    stages = ("straight", "turn", "door", "stop", "complete")

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.stage_index = 0
        self.progress = 0.0
        self.success = False
        self.contact_failure = False
        self.fallen_failure = False
        self.stop_since = None
        self.crossings: list[dict[str, Any]] = []
        self.last_metrics: dict[str, Any] = {}

    @property
    def stage(self):
        return self.stages[self.stage_index]

    def reference(self, progress):
        """Vectorised arc-length reference; returns [..., (x,y,yaw)]."""
        s = np.clip(np.asarray(progress, dtype=float), 0.0, self.spec.total_length)
        length, radius = self.spec.straight_length, self.spec.turn_radius
        alpha = np.clip((s - length) / radius, 0.0, np.pi / 2.0)
        x = np.where(s <= length, s, length + radius * np.sin(alpha))
        y = np.where(s <= length, 0.0, radius * (1.0 - np.cos(alpha)))
        y = np.where(s >= length + self.spec.arc_length,
                     radius + s - length - self.spec.arc_length, y)
        return np.stack([x, y, alpha], axis=-1)

    def project(self, position):
        """Nearest exact piece of the route; does not change evaluation state."""
        p = np.asarray(position, dtype=float)[..., :2]
        length, radius = self.spec.straight_length, self.spec.turn_radius
        s0 = np.clip(p[..., 0], 0, length)
        alpha = np.clip(np.arctan2(p[..., 0] - length, radius - p[..., 1]),
                        0, np.pi / 2.0)
        s1 = length + radius * alpha
        s2 = length + self.spec.arc_length + np.clip(
            p[..., 1] - radius, 0, self.spec.goal_y - radius)
        candidates = np.stack([s0, s1, s2], axis=-1)
        refs = self.reference(candidates)
        dist2 = np.sum((refs[..., :2] - p[..., None, :]) ** 2, axis=-1)
        choice = np.argmin(dist2, axis=-1)
        return np.take_along_axis(candidates, choice[..., None], axis=-1)[..., 0]

    def nominal_speed(self, progress):
        """Fixed engineering reference, not identified from robot trials."""
        s = np.asarray(progress)
        turn_begin = self.spec.straight_length
        turn_end = turn_begin + self.spec.arc_length
        turn_region = (s > turn_begin - 0.35) & (s < turn_end + 0.15)
        cruise = np.where(turn_region, 0.32, 0.45)
        stopping_speed = np.sqrt(np.maximum(0.0, 0.7 * (self.spec.total_length - s)))
        return np.minimum(cruise, stopping_speed)

    def reference_horizon(self, initial_progress, horizon: int, dt: float):
        s = float(np.clip(initial_progress, 0, self.spec.total_length))
        values = []
        speeds = []
        for _ in range(horizon):
            speed = float(self.nominal_speed(s))
            s = min(self.spec.total_length, s + dt * speed)
            values.append(s)
            speeds.append(float(self.nominal_speed(s)))
        return self.reference(np.asarray(values)), np.asarray(speeds)

    def obstacles(self):
        """Physical L-corridor boundaries and two finite door panels."""
        s = self.spec
        gx, width, thick, height = s.goal_x, s.corridor_half_width, 0.06, 0.45
        boxes = []

        def box(name, xlo, xhi, ylo, yhi, h=height):
            boxes.append({"name": name,
                          "pos": [(xlo + xhi) / 2, (ylo + yhi) / 2, h],
                          "size": [(xhi - xlo) / 2, (yhi - ylo) / 2, h],
                          "rgba": [0.35, 0.45, 0.55, 1.0]})

        end = s.goal_y + 0.9
        box("south_wall", -1.0, gx + width + 2 * thick, -width - 2 * thick, -width)
        box("east_wall", gx + width, gx + width + 2 * thick, -width, end)
        inner_x = gx - width
        box("inner_horizontal", -1.0, inner_x, width, width + 2 * thick)
        box("inner_vertical", inner_x - 2 * thick, inner_x, width, end)
        box("door_left", inner_x, gx - s.door_opening / 2,
            s.door_y - 0.07, s.door_y + 0.07, 0.65)
        box("door_right", gx + s.door_opening / 2, gx + width,
            s.door_y - 0.07, s.door_y + 0.07, 0.65)
        return boxes

    def metrics(self):
        return dict(self.last_metrics)

    def update(self, obs):
        """Consume actual observations only. Contact permanently invalidates success."""
        pos = np.asarray(obs["position"], dtype=float)[:2]
        velocity = np.asarray(obs["velocity_world"], dtype=float)[:2]
        yaw = float(obs["yaw"])
        time = float(obs["time"])
        projected = float(self.project(pos))
        self.progress = max(self.progress, projected)
        self.contact_failure |= bool(obs.get("obstacle_contact_ever", obs.get("obstacle_contact", False)))
        self.fallen_failure |= bool(obs.get("fallen_ever", obs.get("fallen", False)))
        distance = float(np.linalg.norm(pos - self.spec.goal))
        speed = float(np.linalg.norm(velocity))
        yaw_error = abs(float(wrap_angle(yaw - np.pi / 2)))
        stage_before = self.stage
        # Sequential spatial gates prevent nearest-point projection from skipping phases.
        reached = (
            self.stage_index == 0 and pos[0] >= self.spec.straight_length
            and abs(pos[1]) < 0.55
        ) or (
            self.stage_index == 1 and pos[1] >= self.spec.turn_radius
            and abs(pos[0] - self.spec.goal_x) < 0.55 and yaw_error < 0.65
        ) or (
            self.stage_index == 2
            and pos[1] >= self.spec.door_y + 0.07 + self.spec.footprint_half_length
            and abs(pos[0] - self.spec.goal_x) < self.spec.door_opening / 2
        )
        if reached:
            self.crossings.append({"stage": stage_before, "time": time,
                                   "position": pos.tolist(), "velocity_world": velocity.tolist(),
                                   "yaw": yaw, "yaw_rate": float(obs["yaw_rate"])})
            self.stage_index += 1
        settled = (self.stage_index == 3 and distance <= self.spec.goal_position_tolerance
                   and speed <= self.spec.goal_speed_tolerance
                   and yaw_error <= self.spec.goal_yaw_tolerance)
        if settled:
            if self.stop_since is None:
                self.stop_since = time
            if time - self.stop_since >= self.spec.stop_hold_seconds:
                self.success = not self.contact_failure and not self.fallen_failure
                if self.success:
                    self.stage_index = 4
        else:
            self.stop_since = None
        self.last_metrics = {
            "stage": self.stage, "stage_index": self.stage_index,
            "progress": self.progress, "projected_progress": projected,
            "goal_distance": distance, "planar_speed": speed, "goal_yaw_error": yaw_error,
            "success": self.success, "contact_failure": self.contact_failure,
            "fallen_failure": self.fallen_failure,
            "timeout": time >= self.spec.timeout,
            "stop_hold_elapsed": 0.0 if self.stop_since is None else time - self.stop_since,
            "crossings": list(self.crossings),
        }
        return self.last_metrics
