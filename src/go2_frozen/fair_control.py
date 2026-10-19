"""Predeclared, frozen-policy comparison of objectives under common speed limits.

No robot parameters are fitted here.  The original task, follower and MPC remain
unchanged in their existing files.  This module gives all three comparison modes
the same command governor and matched geometric-feedback proposal.

``score_trajectory`` consumes post-command states and *applied* commands.  It is
therefore also usable on full-physics counterfactual trajectories, with exactly
the same cost as the nominal model.  Its costs are controller proxies, not
cross-objective task-performance metrics or an exact minimum-time formulation.
"""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from .planner import MPCConfig, OrdinaryMPC, observation_state
from .task import Task, wrap_angle


SPEED_BANDS = {"standard": (0.45, 0.32), "fast": (0.69, 0.55)}
MISSION_WEIGHTS = {
    "time_per_second": 1.0,
    "lateral_per_m2_per_step": 35.0,
    "yaw_per_rad2_per_step": 4.0,
    "command_squared_per_step": 0.03,
    "command_change_squared_per_step": 0.25,
    "remaining_reference_seconds_terminal": 20.0,
    "goal_position_per_m2_terminal": 20.0,
    "goal_velocity_per_m2s2_terminal": 3.0,
    "terminal_goal_zone_m": 0.35,
}


class FairController(OrdinaryMPC):
    """Follower, time-reference MPC, or route-progress MPC with one governor."""

    def __init__(self, mode="tracking", speed_band="standard", config=None):
        if mode not in ("follower", "tracking", "mission"):
            raise ValueError(f"Unknown comparison mode: {mode}")
        if speed_band not in SPEED_BANDS:
            raise ValueError(f"Unknown speed band: {speed_band}")
        super().__init__(config or MPCConfig())
        self.mode = mode
        self.speed_band = speed_band
        self._time_table_key = None
        self._time_table = None

    def metadata(self):
        return {
            "controller": f"fair_{self.mode}",
            "mode": self.mode,
            "speed_band": self.speed_band,
            "config": asdict(self.config),
            "model": "unidentified_first_order_world_velocity_and_yaw_response",
            "training_updates": 0,
            "speed_semantics": "common body translation command norm ceiling",
            "speed_profile": {
                "straight_m_s": SPEED_BANDS[self.speed_band][0],
                "turn_m_s": SPEED_BANDS[self.speed_band][1],
                "turn_before_m": 0.35,
                "turn_after_m": 0.15,
                "braking_speed": "sqrt(0.7 * max(0, route_remaining_m))",
                "terminal_correction_ceiling_floor_m_s": 0.06,
                "beyond_endpoint_braking_distance": "Euclidean distance to goal",
            },
            "follower": {
                "lookahead_m": 0.22,
                "cross_track_gain_s_inv": 1.1,
                "yaw_gain_s_inv": 1.8,
                "terminal_position_gain_s_inv": 1.2,
                "terminal_position_region_m": 0.35,
                "lookahead_tangential_position_boost": False,
            },
            "tracking_cost": "all original planner.py score weights preserved",
            "mission_cost": dict(MISSION_WEIGHTS),
            "remaining_time_table": {
                "rule": "midpoint ds / nominal speed with 4096 uniform cells plus profile boundaries",
                "speed_floor_m_s": 0.001,
                "purpose": "continuous reference travel-time proxy; not an optimal value function",
            },
            "common_evaluation": "ordered success first, completion time, then path error and work proxy",
            "cost_comparison_boundary": "never compare tracking and mission scalar costs as task performance",
            "governor_in_prediction_and_execution": True,
        }

    def profile_speed(self, progress, task):
        """Common reference speed along the unchanged route, including braking."""
        s = np.asarray(progress, dtype=float)
        cruise, turn = SPEED_BANDS[self.speed_band]
        turn_begin = task.spec.straight_length
        turn_end = turn_begin + task.spec.arc_length
        in_turn = (s > turn_begin - 0.35) & (s < turn_end + 0.15)
        cap = np.where(in_turn, turn, cruise)
        braking = np.sqrt(np.maximum(0.0, 0.7 * (task.spec.total_length - s)))
        return np.minimum(cap, braking)

    def translation_cap(self, state, task):
        """Same cap for each model step, real command, and oracle callback."""
        state = np.asarray(state, dtype=float)
        progress = task.project(state[..., :2])
        cap = self.profile_speed(progress, task)
        remaining = task.spec.total_length - progress
        distance = np.linalg.norm(state[..., :2] - task.spec.goal, axis=-1)
        # Projection is clipped at the route endpoint.  Euclidean distance then
        # permits a genuine return from an overshoot instead of a zero-command trap.
        recovery_cap = np.minimum(SPEED_BANDS[self.speed_band][0],
                                  np.sqrt(np.maximum(0.0, 0.7 * distance)))
        cap = np.where(remaining <= 1e-9, recovery_cap, cap)
        # Small terminal corrections remain permitted even at zero projected
        # remaining distance.  This is a ceiling, not a minimum issued command.
        return np.where(remaining < 0.35, np.maximum(cap, 0.06), cap)

    def govern(self, state, raw_command, task):
        command = np.clip(np.asarray(raw_command, dtype=float),
                          self.config.command_min, self.config.command_max).copy()
        norm = np.linalg.norm(command[..., :2], axis=-1)
        cap = self.translation_cap(state, task)
        scale = np.minimum(1.0, cap / np.maximum(norm, 1e-12))
        command[..., :2] *= np.asarray(scale)[..., None]
        return command

    def reference_horizon(self, progress, horizon, task):
        s = float(np.clip(progress, 0, task.spec.total_length))
        positions, speeds = [], []
        for _ in range(horizon):
            s = min(task.spec.total_length,
                    s + self.config.dt * float(self.profile_speed(s, task)))
            positions.append(s)
            speeds.append(float(self.profile_speed(s, task)))
        return task.reference(np.asarray(positions)), np.asarray(speeds)

    def feedback_command(self, state, task):
        state = np.asarray(state, dtype=float)
        progress = float(task.project(state[:2]))
        ref = task.reference(min(task.spec.total_length, progress + 0.22))
        tangent = np.array([np.cos(ref[2]), np.sin(ref[2])])
        error = ref[:2] - state[:2]
        cross_track_error = error - np.dot(error, tangent) * tangent
        desired_world = float(self.translation_cap(state, task)) * tangent
        desired_world += 1.1 * cross_track_error
        if task.spec.total_length - progress < 0.35:
            desired_world = 1.2 * (task.spec.goal - state[:2])
        c, s = np.cos(state[2]), np.sin(state[2])
        raw = np.array([c * desired_world[0] + s * desired_world[1],
                        -s * desired_world[0] + c * desired_world[1],
                        1.8 * float(wrap_angle(ref[2] - state[2]))])
        return self.govern(state, raw, task)

    def nominal_rollout(self, initial, raw_commands, task):
        """Return post-step states and governed commands; single or batch input."""
        commands = np.asarray(raw_commands, dtype=float)
        single = commands.ndim == 2
        if single:
            commands = commands[None]
        state = np.broadcast_to(np.asarray(initial, dtype=float),
                                (len(commands), 6)).copy()
        states, applied = [], []
        for k in range(commands.shape[1]):
            command = self.govern(state, commands[:, k], task)
            state = super().step_model(state, command)
            applied.append(command)
            states.append(state.copy())
        states = np.stack(states, axis=1)
        applied = np.stack(applied, axis=1)
        return (states[0], applied[0]) if single else (states, applied)

    def feedback_sequence(self, initial, task):
        state = np.asarray(initial, dtype=float).copy()
        commands, states = [], []
        for _ in range(self.config.horizon):
            command = self.feedback_command(state, task)
            state = super().step_model(state, command)
            commands.append(command)
            states.append(state.copy())
        return np.asarray(commands), np.asarray(states)

    def remaining_reference_time(self, progress, task):
        key = (self.speed_band, task.spec.straight_length, task.spec.turn_radius,
               task.spec.goal_y)
        if key != self._time_table_key:
            length = task.spec.total_length
            turn_begin = task.spec.straight_length - 0.35
            turn_end = task.spec.straight_length + task.spec.arc_length + 0.15
            edges = np.unique(np.r_[np.linspace(0.0, length, 4097),
                                    np.clip([turn_begin, turn_end], 0.0, length)])
            mid = (edges[:-1] + edges[1:]) / 2
            duration = np.diff(edges) / np.maximum(self.profile_speed(mid, task), 0.001)
            remaining = np.r_[np.cumsum(duration[::-1])[::-1], 0.0]
            self._time_table_key = key
            self._time_table = (edges, remaining)
        return np.interp(progress, *self._time_table)

    def score_trajectory(self, states, commands, task, initial,
                         last_command=None, objective=None):
        """Score nominal or physics trajectories under either fixed objective.

        ``states`` is [H,6] or [N,H,6], sampled after each 0.2 s command.
        ``commands`` contains the actual governed commands, with matching H.
        No model is called and no command is re-governed inside this scorer.
        """
        objective = objective or self.mode
        if objective not in ("tracking", "mission"):
            raise ValueError("Trajectory objective must be tracking or mission")
        trajectory = np.asarray(states, dtype=float)
        issued = np.asarray(commands, dtype=float)
        single = trajectory.ndim == 2
        if single:
            trajectory, issued = trajectory[None], issued[None]
        batch, horizon, _ = trajectory.shape
        initial = np.broadcast_to(np.asarray(initial, dtype=float), (batch, 6))
        previous = np.broadcast_to(np.asarray(
            self.last_command if last_command is None else last_command, dtype=float),
            (batch, 3)).copy()
        costs = np.zeros(batch)
        if objective == "tracking":
            initial_progress = task.project(initial[:, :2])
            if np.all(initial_progress == initial_progress[0]):
                shared_refs, shared_speeds = self.reference_horizon(float(initial_progress[0]), horizon, task)
                refs = np.broadcast_to(shared_refs, (batch, horizon, 3))
                speeds = np.broadcast_to(shared_speeds, (batch, horizon))
            else:
                # Separate initial states are supported for the analysis API.
                reference_data = [self.reference_horizon(float(s), horizon, task)
                                  for s in initial_progress]
                refs = np.stack([entry[0] for entry in reference_data])
                speeds = np.stack([entry[1] for entry in reference_data])
        for k in range(horizon):
            state = trajectory[:, k]
            if objective == "tracking":
                ref = refs[:, k]
                tangent = np.stack([np.cos(ref[:, 2]), np.sin(ref[:, 2])], axis=-1)
                delta = state[:, :2] - ref[:, :2]
                longitudinal = np.sum(delta * tangent, axis=-1)
                lateral = -delta[:, 0] * tangent[:, 1] + delta[:, 1] * tangent[:, 0]
                yaw_error = wrap_angle(state[:, 2] - ref[:, 2])
                desired_velocity = speeds[:, k, None] * tangent
                cost = 8.0 * longitudinal ** 2 + 35.0 * lateral ** 2 + 4.0 * yaw_error ** 2
                cost += 2.0 * np.sum((state[:, 3:5] - desired_velocity) ** 2, axis=-1)
            else:
                ref = task.reference(task.project(state[:, :2]))
                tangent = np.stack([np.cos(ref[:, 2]), np.sin(ref[:, 2])], axis=-1)
                delta = state[:, :2] - ref[:, :2]
                lateral = -delta[:, 0] * tangent[:, 1] + delta[:, 1] * tangent[:, 0]
                yaw_error = wrap_angle(state[:, 2] - ref[:, 2])
                cost = (self.config.dt * MISSION_WEIGHTS["time_per_second"]
                        + MISSION_WEIGHTS["lateral_per_m2_per_step"] * lateral ** 2
                        + MISSION_WEIGHTS["yaw_per_rad2_per_step"] * yaw_error ** 2)
            cost += 0.03 * np.sum(issued[:, k] ** 2, axis=-1)
            cost += 0.25 * np.sum((issued[:, k] - previous) ** 2, axis=-1)
            collision_cost, _ = self.footprint_penalty(state, task)
            costs += cost + collision_cost
            previous = issued[:, k]
        terminal = trajectory[:, -1]
        if objective == "tracking":
            costs += 20.0 * np.sum((terminal[:, :2] - refs[:, -1, :2]) ** 2, axis=-1)
            costs += 3.0 * np.sum((terminal[:, 3:5] - speeds[:, -1, None] * tangent) ** 2, axis=-1)
        else:
            progress = task.project(terminal[:, :2])
            costs += (MISSION_WEIGHTS["remaining_reference_seconds_terminal"]
                      * self.remaining_reference_time(progress, task))
            # The terminal goal/velocity terms apply when the predicted endpoint
            # reaches the final approach, rather than pulling through the corner.
            near_goal = task.spec.total_length - progress < MISSION_WEIGHTS["terminal_goal_zone_m"]
            costs += near_goal * (
                MISSION_WEIGHTS["goal_position_per_m2_terminal"]
                * np.sum((terminal[:, :2] - task.spec.goal) ** 2, axis=-1)
                + MISSION_WEIGHTS["goal_velocity_per_m2s2_terminal"]
                * np.sum(terminal[:, 3:5] ** 2, axis=-1))
        return float(costs[0]) if single else costs

    def score(self, initial, commands, task, refs=None, speeds=None, return_states=False):
        states, applied = self.nominal_rollout(initial, commands, task)
        costs = self.score_trajectory(states, applied, task, initial,
                                      self.last_command, self.mode)
        return (costs, states) if return_states else costs

    def plan(self, obs, task):
        start = perf_counter()
        initial = observation_state(obs)
        if self.mode == "follower":
            command = self.feedback_command(initial, task)
            self.last_command = command.copy()
            elapsed = perf_counter() - start
            self.total_seconds += elapsed
            self.calls += 1
            return command, {"controller": "fair_follower", "speed_band": self.speed_band,
                             "solve_seconds": elapsed, "candidate_rollouts": 0,
                             "model_steps": 0, "dt": self.config.dt}
        feedback, _ = self.feedback_sequence(initial, task)
        mean = feedback.copy() if self.warm_sequence is None else np.vstack([
            self.warm_sequence[1:], self.warm_sequence[-1:]])
        std = np.broadcast_to([0.17, 0.13, 0.32], mean.shape).copy()
        best, best_cost = feedback.copy(), np.inf
        evaluations = 0
        for _ in range(self.config.iterations):
            noise = self.rng.normal(size=(self.config.samples, self.config.horizon, 3))
            for k in range(1, self.config.horizon):
                noise[:, k] = 0.65 * noise[:, k - 1] + 0.76 * noise[:, k]
            samples = np.clip(mean[None] + std[None] * noise,
                              self.config.command_min, self.config.command_max)
            samples[0], samples[1], samples[2] = mean, feedback, best
            costs = self.score(initial, samples, task)
            evaluations += len(samples)
            winner = int(np.argmin(costs))
            if costs[winner] < best_cost:
                best_cost, best = float(costs[winner]), samples[winner].copy()
            elite = samples[np.argpartition(costs, self.config.elites)[:self.config.elites]]
            mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
            std = np.maximum(0.02, 0.2 * std + 0.8 * elite.std(axis=0))
        predicted, applied = self.nominal_rollout(initial, best, task)
        self.warm_sequence = best.copy()
        self.last_command = applied[0].copy()
        elapsed = perf_counter() - start
        self.total_seconds += elapsed
        self.calls += 1
        refs, _ = self.reference_horizon(float(task.project(initial[:2])), self.config.horizon, task)
        return self.last_command.copy(), {
            "controller": f"fair_{self.mode}", "objective": self.mode,
            "speed_band": self.speed_band, "solve_seconds": elapsed,
            "candidate_rollouts": evaluations,
            "model_steps": evaluations * self.config.horizon,
            "horizon_steps": self.config.horizon,
            "horizon_seconds": self.config.horizon * self.config.dt,
            "predicted_cost": best_cost, "dt": self.config.dt,
            "predicted_state": predicted[0].tolist(),
            "prediction_trace": predicted.tolist(),
            "command_sequence": applied.tolist(),
            "raw_command_sequence": best.tolist(),
            "reference": refs.tolist(),
            "progress": float(task.project(initial[:2])),
            "model_dt": self.config.dt,
        }
