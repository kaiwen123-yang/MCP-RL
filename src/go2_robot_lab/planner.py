"""Ordinary sampling MPC over body velocity commands for a frozen locomotion actor.

The lag model and cost are fixed engineering choices, not learned Go2 parameters.
CEM scores complete finite-horizon command sequences; a geometric feedback rollout
is included as a feasible candidate and initialisation, never relabelled as MPC.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .task import Task, wrap_angle


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.20
    horizon: int = 20
    samples: int = 256
    iterations: int = 3
    elites: int = 24
    velocity_time_constant: float = 0.25
    yaw_time_constant: float = 0.20
    command_min: tuple = (-0.15, -0.30, -0.90)
    command_max: tuple = (0.70, 0.30, 0.90)
    collision_margin: float = 0.04
    seed: int = 20261019


def observation_state(obs):
    return np.array([*np.asarray(obs["position"])[:2], float(obs["yaw"]),
                     *np.asarray(obs["velocity_world"])[:2], float(obs["yaw_rate"])])


class OrdinaryMPC:
    def __init__(self, config: MPCConfig | None = None):
        self.config = config or MPCConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.warm_sequence = None
        self.last_command = np.zeros(3)
        self.total_seconds = 0.0
        self.calls = 0

    def metadata(self):
        return {"controller": "ordinary_cem_mpc", "config": asdict(self.config),
                "model": "unidentified_first_order_world_velocity_and_yaw_response",
                "training_updates": 0}

    def step_model(self, state, command):
        """Vectorised state transition; [x,y,yaw,vx_world,vy_world,yaw_rate]."""
        out = np.array(state, copy=True)
        dt = self.config.dt
        yaw = state[..., 2]
        c, s = np.cos(yaw), np.sin(yaw)
        desired = np.stack([c * command[..., 0] - s * command[..., 1],
                            s * command[..., 0] + c * command[..., 1]], axis=-1)
        av = 1.0 - np.exp(-dt / self.config.velocity_time_constant)
        aw = 1.0 - np.exp(-dt / self.config.yaw_time_constant)
        out[..., 3:5] += av * (desired - state[..., 3:5])
        out[..., 5] += aw * (command[..., 2] - state[..., 5])
        out[..., :2] += dt * out[..., 3:5]
        out[..., 2] = wrap_angle(state[..., 2] + dt * out[..., 5])
        return out

    def feedback_command(self, state, task: Task):
        """Transparent geometric follower, also available as a diagnostic control."""
        progress = float(task.project(state[:2]))
        speed = float(task.nominal_speed(progress))
        ref = task.reference(min(task.spec.total_length, progress + 0.22))
        desired_world = speed * np.array([np.cos(ref[2]), np.sin(ref[2])])
        desired_world += 1.1 * (ref[:2] - state[:2])
        if task.spec.total_length - progress < 0.35:
            desired_world = 1.2 * (task.spec.goal - state[:2])
        c, s = np.cos(state[2]), np.sin(state[2])
        body = np.array([c * desired_world[0] + s * desired_world[1],
                         -s * desired_world[0] + c * desired_world[1],
                         1.8 * float(wrap_angle(ref[2] - state[2]))])
        return np.clip(body, self.config.command_min, self.config.command_max)

    def feedback_sequence(self, initial, task):
        states, commands = [], []
        state = initial.copy()
        for _ in range(self.config.horizon):
            command = self.feedback_command(state, task)
            state = self.step_model(state, command)
            commands.append(command)
            states.append(state)
        return np.asarray(commands), np.asarray(states)

    def footprint_penalty(self, state, task):
        """OBB versus AABB SAT, including an explicit conservative body margin."""
        x, y, yaw = state[..., 0], state[..., 1], state[..., 2]
        c, s = np.cos(yaw), np.sin(yaw)
        ac, ass = np.abs(c), np.abs(s)
        a = task.spec.footprint_half_length + self.config.collision_margin
        b = task.spec.footprint_half_width + self.config.collision_margin
        penalty = np.zeros_like(x)
        minimum_clearance = np.full_like(x, np.inf)
        for obstacle in task.obstacles():
            ox, oy = obstacle["pos"][:2]
            hx, hy = obstacle["size"][:2]
            dx, dy = x - ox, y - oy
            # Positive separating-axis distance excludes overlap; all four <=0 collide.
            gaps = np.stack([
                np.abs(dx) - (hx + a * ac + b * ass),
                np.abs(dy) - (hy + a * ass + b * ac),
                np.abs(dx * c + dy * s) - (a + hx * ac + hy * ass),
                np.abs(-dx * s + dy * c) - (b + hx * ass + hy * ac),
            ], axis=-1)
            clearance = np.max(gaps, axis=-1)
            minimum_clearance = np.minimum(minimum_clearance, clearance)
            penalty += 20000.0 * np.maximum(0.0, -clearance + 0.015) ** 2
            penalty += 2.0 * np.maximum(0.0, 0.08 - clearance) ** 2
        return penalty, minimum_clearance

    def score(self, initial, commands, task, refs, speeds, return_states=False):
        batch = len(commands)
        state = np.broadcast_to(initial, (batch, 6)).copy()
        costs = np.zeros(batch)
        previous = np.broadcast_to(self.last_command, (batch, 3))
        states = []
        for step in range(self.config.horizon):
            state = self.step_model(state, commands[:, step])
            ref = refs[step]
            delta = state[:, :2] - ref[:2]
            tangent = np.array([np.cos(ref[2]), np.sin(ref[2])])
            normal = np.array([-tangent[1], tangent[0]])
            longitudinal = delta @ tangent
            lateral = delta @ normal
            yaw_error = wrap_angle(state[:, 2] - ref[2])
            desired_velocity = speeds[step] * tangent
            cost = 8.0 * longitudinal ** 2 + 35.0 * lateral ** 2 + 4.0 * yaw_error ** 2
            cost += 2.0 * np.sum((state[:, 3:5] - desired_velocity) ** 2, axis=-1)
            cost += 0.03 * np.sum(commands[:, step] ** 2, axis=-1)
            cost += 0.25 * np.sum((commands[:, step] - previous) ** 2, axis=-1)
            collision_cost, _ = self.footprint_penalty(state, task)
            costs += cost + collision_cost
            previous = commands[:, step]
            if return_states:
                states.append(state.copy())
        costs += 20.0 * np.sum((state[:, :2] - refs[-1, :2]) ** 2, axis=-1)
        costs += 3.0 * np.sum((state[:, 3:5] - speeds[-1] * tangent) ** 2, axis=-1)
        if return_states:
            return costs, np.stack(states, axis=1)
        return costs

    def plan(self, obs, task: Task):
        start = perf_counter()
        initial = observation_state(obs)
        progress = float(task.project(initial[:2]))
        refs, speeds = task.reference_horizon(progress, self.config.horizon, self.config.dt)
        feedback, _ = self.feedback_sequence(initial, task)
        mean = feedback.copy() if self.warm_sequence is None else np.vstack([
            self.warm_sequence[1:], self.warm_sequence[-1:]])
        std = np.broadcast_to([0.17, 0.13, 0.32], mean.shape).copy()
        best = feedback.copy()
        best_cost = np.inf
        evaluations = 0
        for _ in range(self.config.iterations):
            noise = self.rng.normal(size=(self.config.samples, self.config.horizon, 3))
            # Temporally correlated samples still optimise a separate command each step.
            for k in range(1, self.config.horizon):
                noise[:, k] = 0.65 * noise[:, k - 1] + 0.76 * noise[:, k]
            samples = np.clip(mean[None] + std[None] * noise,
                              self.config.command_min, self.config.command_max)
            samples[0], samples[1], samples[2] = mean, feedback, best
            costs = self.score(initial, samples, task, refs, speeds)
            evaluations += len(samples)
            winner = int(np.argmin(costs))
            if costs[winner] < best_cost:
                best_cost = float(costs[winner])
                best = samples[winner].copy()
            elite = samples[np.argpartition(costs, self.config.elites)[:self.config.elites]]
            mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
            std = np.maximum(0.02, 0.2 * std + 0.8 * elite.std(axis=0))
        _, predicted = self.score(initial, best[None], task, refs, speeds, return_states=True)
        self.warm_sequence = best
        self.last_command = best[0].copy()
        elapsed = perf_counter() - start
        self.total_seconds += elapsed
        self.calls += 1
        info = {"controller": "ordinary_cem_mpc", "solve_seconds": elapsed,
                "candidate_rollouts": evaluations, "model_steps": evaluations * self.config.horizon,
                "horizon_steps": self.config.horizon, "horizon_seconds": self.config.horizon * self.config.dt,
                "predicted_cost": best_cost, "predicted_state": predicted[0].tolist(),
                "prediction_trace": predicted[0].tolist(), "dt": self.config.dt,
                "command_sequence": best.tolist(), "reference": refs.tolist(),
                "progress": progress, "model_dt": self.config.dt,
                "nominal_response": {"velocity_time_constant": self.config.velocity_time_constant,
                                     "yaw_time_constant": self.config.yaw_time_constant,
                                     "parameter_source": "fixed engineering initial values; not identified"}}
        return self.last_command.copy(), info


class ReferenceFollower(OrdinaryMPC):
    """Optional no-optimisation diagnostic; never reported as MPC."""

    def metadata(self):
        return {"controller": "geometric_reference_follower", "config": asdict(self.config),
                "model": "geometric feedback only; no online optimisation",
                "training_updates": 0}

    def plan(self, obs, task):
        start = perf_counter()
        command = self.feedback_command(observation_state(obs), task)
        return command, {"controller": "geometric_reference_follower",
                         "solve_seconds": perf_counter() - start,
                         "candidate_rollouts": 0, "model_steps": 0}


CommandMPC = OrdinaryMPC
