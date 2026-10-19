"""Continuous route, contact gate and terminal hold shared by every controller."""
from __future__ import annotations
import math
import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def feature(state):
    return np.array([*state["position"][:2], state["yaw"],
                     *state["linear_velocity_world"][:2], state["angular_velocity_world"][2]], dtype=float)


def advance_stage(state, stage, config):
    route = config["route"]
    while stage < len(route) - 1 and np.linalg.norm(np.asarray(state["position"][:2]) - route[stage][:2]) <= config["waypoint_radius"]:
        stage += 1
    return stage


def nominal_command(state, stage, config):
    target = np.asarray(config["route"][stage])
    delta = target[:2] - state["position"][:2]
    yaw = state["yaw"]
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
    command = np.array([0.9 * local[0], 0.9 * local[1], 1.5 * wrap(target[2] - yaw)])
    return np.clip(command, config["command_lower"], config["command_upper"])


def remaining_distance(state, stage, config):
    route = np.asarray(config["route"])
    return float(np.linalg.norm(np.asarray(state["position"][:2]) - route[stage, :2])
                 + np.linalg.norm(np.diff(route[stage:, :2], axis=0), axis=1).sum())


def running_cost(state, stage, config):
    """Shared navigation objective; actuator resources are reported separately."""
    target = config["route"][stage]
    return remaining_distance(state, stage, config) + 0.2 * wrap(target[2] - state["yaw"]) ** 2


def terminal_surrogate(state, stage, config):
    """Explicit remaining-task heuristic, not a claim about a 1s final deadline."""
    cost = 1.5 * remaining_distance(state, stage, config)
    if stage == len(config["route"]) - 1:
        distance = np.linalg.norm(np.asarray(state["position"][:2]) - config["route"][-1][:2])
        if distance < 0.5:
            cost += 1.5 * np.linalg.norm(state["linear_velocity_world"][:2]) ** 2
        cost += 0.2 * wrap(config["route"][-1][2] - state["yaw"]) ** 2
    return float(cost)


def terminal_ok(state, config):
    goal, limits = config["route"][-1], config["success"]
    return bool(np.linalg.norm(np.asarray(state["position"][:2])-goal[:2]) <= limits["position_tolerance"]
                and abs(wrap(state["yaw"]-goal[2])) <= limits["yaw_tolerance"]
                and np.linalg.norm(state["linear_velocity_world"][:2]) <= limits["speed_tolerance"]
                and abs(state["angular_velocity_world"][2]) <= limits["yaw_rate_tolerance"])


class TerminalHold:
    """Elapsed time between qualifying samples, not frame-count times dt."""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def observe(self, timestamp, qualifies):
        if not qualifies:
            self.start_time = None
            self.elapsed = 0.0
        else:
            if self.start_time is None:
                self.start_time = timestamp
            self.elapsed = float(timestamp-self.start_time)
        return self.elapsed


def is_fallen(state, config):
    return bool(state["position"][2] < config["fall"]["minimum_height"]
                or state["tilt"] > config["fall"]["maximum_tilt"])


def kinematic_step(state, command, dt):
    """Transparent ideal body-twist assumption, not a learned response model."""
    out = dict(state)
    out["position"] = np.array(state["position"], dtype=float).copy()
    mid = state["yaw"] + command[2] * dt / 2
    c, s = math.cos(mid), math.sin(mid)
    out["linear_velocity_world"] = np.array([c*command[0]-s*command[1], s*command[0]+c*command[1], 0.0])
    out["angular_velocity_world"] = np.array([0.0, 0.0, command[2]])
    out["position"] += out["linear_velocity_world"] * dt
    out["yaw"] = float(wrap(state["yaw"] + command[2] * dt))
    return out
