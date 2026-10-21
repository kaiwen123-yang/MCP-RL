"""Common finite-candidate MPC using the feedback learner's mission proxy.

The command governor, CEM budget and warm-start rule are retained. A factual
48-state command response supplies predictions and the actual-return critic
supplies active continuation value. Mission costs stop at predicted completion
or deadline failure and respect the actual goal tolerance. This module does not
differentiate through CEM or update networks.
"""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import torch

from go2_frozen.fair_control import FairController, SPEED_BANDS
from go2_frozen.planner import MPCConfig

from .mission import mission_cost
from .state import (
    advance_memory,
    memory_from_task,
    planar_state,
    state_from_observation,
    task_active,
    task_context,
)


class LearnedMPC(FairController):
    """Mission CEM with factual response, shared cost and continuation value.

    ``plan(obs, task)`` reads the actual complete state and task memory.  The
    optional observation fields ``learning_state48`` and ``learning_memory3``
    provide the same data explicitly, for adapters without all original keys.
    Model rollout helpers are scoped to that cached current decision state.
    """

    def __init__(self, response, value, actor=None, spec=None, speed="fast",
                 config: MPCConfig | None = None, controller_mode="mpc",
                 response_mode="learned", use_terminal_value=True):
        super().__init__(mode="mission", speed_band=speed, config=config or MPCConfig())
        if not math.isclose(float(response.dt), self.config.dt):
            raise ValueError("CommandResponse.dt must match the MPC macro interval")
        self.response = response
        self.value = value
        # The response describes this actor's executed behavior, but planning
        # uses g directly.  This reference is never queried as a hidden oracle.
        self.actor = actor
        self.spec = spec
        self.controller_mode = controller_mode
        self.response_mode = response_mode
        self.use_terminal_value = bool(use_terminal_value)
        self.current_state = None
        self.current_memory = None
        self.response_steps = 0
        self.value_calls = 0
        self._last_terminal = None
        self._last_full_trace = None
        self._last_memory_trace = None
        self._previous_measured_state = None
        self._previous_issued_command = None
        self._velocity_offset = None
        self._reference_residual = None

    def metadata(self):
        return {
            **super().metadata(),
            "controller": "shared_factual_response_mission_mpc",
            "model": "full48_command_response_fitted_only_to_actual_transitions",
            "prediction_closure": self.response_mode,
            "learned_terminal_value_enabled": self.use_terminal_value,
            "terminal_value": "minus actual-return critic; coefficient 1 in all arms",
            "terminal_value_inputs": "predicted full48, predicted task memory, final applied command, macro phase 1",
            "cem_policy": "original deterministic seed and shifted warm sequence",
            "policy_used_inside_mpc": False,
            "actor_credit": "shared mission proxy; CEM is not differentiated",
            "mission_cost": "go2_learning.mission.mission_cost; success/deadline absorbing, tolerance-region terminal",
            "hold_prediction": "macro endpoints only; not a continuous 20 ms hold certificate",
            "zero_update_prior": "original planar lag plus non-planar identity; critic zero",
            "response_steps": self.response_steps,
            "value_calls": self.value_calls,
        }

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.response.state_scale.dtype,
                               device=self.response.state_scale.device)

    def _predict_response(self, state, context, command):
        if self.response_mode == "prior":
            return self.response.prior(state, command)
        if self.response_mode == "local_response":
            return self._local_response(state, command)
        return self.response(state, context, command)

    @staticmethod
    def _body_velocity(state, reference_yaw):
        planar = planar_state(state)
        c, s = reference_yaw.cos(), reference_yaw.sin()
        return torch.stack((c * planar[..., 3] + s * planar[..., 4],
                            -s * planar[..., 3] + c * planar[..., 4], planar[..., 5]), -1)

    def _learned_velocity_residual(self, command):
        """Frozen checkpoint command dependence at the CURRENT measured state.

        Non-planar states and task features are measured scheduling variables,
        not autonomous 4-second predictions of joint phase or body attitude.
        """
        state = self.current_state.expand(command.shape[0], -1)
        memory = self.current_memory.expand(command.shape[0], -1)
        context = task_context(state, memory, self.spec)
        actual_yaw = state[:, 5]
        predicted = self.response(state, context, command)
        prior = self.response.prior(state, command)
        return self._body_velocity(predicted, actual_yaw) - self._body_velocity(prior, actual_yaw)

    @torch.no_grad()
    def _update_velocity_offset(self):
        """Measured disturbance state, never a network/optimizer update.

        Anchor the velocity intercept to the latest completed 200 ms execution.
        Filtering uses the response's existing physical time constants.
        """
        if self._velocity_offset is None:
            self._velocity_offset = self.current_state.new_zeros((1, 3))
        if self._previous_measured_state is not None:
            previous = self._previous_measured_state
            prior = self.response.prior(previous, self._previous_issued_command)
            observed = (self._body_velocity(self.current_state, previous[:, 5])
                        - self._body_velocity(prior, previous[:, 5]))
            time_constants = self.current_state.new_tensor([
                self.response.velocity_time_constant, self.response.velocity_time_constant,
                self.response.yaw_time_constant])
            gain = 1 - torch.exp(-self.response.dt / time_constants)
            self._velocity_offset = self._velocity_offset + gain * (observed - self._velocity_offset)
        reference_command = self._tensor(self.last_command).reshape(1, 3)
        self._reference_residual = self._learned_velocity_residual(reference_command)

    def _local_response(self, state, command):
        """Offset-corrected local command response with consistent kinematics.

        The learned model supplies changes with command, not an extrapolated
        absolute position bias. Current measured joint/attitude context stays
        fixed while planar motion evolves. Every real replan refreshes it.
        """
        prior = self.response.prior(state, command)
        correction = (self._learned_velocity_residual(command)
                      - self._reference_residual + self._velocity_offset)
        yaw = state[:, 5]
        c, s = yaw.cos(), yaw.sin()
        world_correction = torch.stack((c * correction[:, 0] - s * correction[:, 1],
                                         s * correction[:, 0] + c * correction[:, 1]), -1)
        velocity = prior[:, 6:8] + world_correction
        yaw_rate = planar_state(prior)[:, 5] + correction[:, 2]
        next_yaw = yaw + self.response.dt * yaw_rate
        next_yaw = torch.atan2(next_yaw.sin(), next_yaw.cos())
        roll, pitch = state[:, 3], state[:, 4]
        wz = (pitch.cos() * yaw_rate - roll.sin() * state[:, 10]) / roll.cos()
        return torch.cat((state[:, :2] + self.response.dt * velocity, state[:, 2:5],
                           next_yaw[:, None], velocity, state[:, 8:11], wz[:, None], state[:, 12:]), -1)

    def _initial_batch(self, batch):
        if self.current_state is None or self.current_memory is None:
            raise RuntimeError("Call plan(obs, task) or set_state(state, memory) before rollout")
        return (self.current_state.expand(batch, -1).clone(),
                self.current_memory.expand(batch, -1).clone())

    def set_state(self, state, memory):
        """Bind actual full state and ordered task memory for one decision."""
        self.current_state = self._tensor(state).reshape(1, 48)
        self.current_memory = self._tensor(memory).reshape(1, 3)

    def _stop_target(self, state, task, memory=None):
        from .continuation import next_stop_y
        if memory is None:
            memory = self._tensor([task.stage_index, 0., -1.])
        target_y = next_stop_y(memory, task.spec).detach().cpu().numpy()
        target_y = np.broadcast_to(target_y, np.asarray(state).shape[:-1])
        return np.stack((np.full_like(target_y, task.spec.goal_x), target_y), -1)

    def _stop_cap(self, state, task, memory=None):
        state = np.asarray(state, dtype=float)
        target = self._stop_target(state, task, memory)
        progress = task.project(state[..., :2])
        stop_progress = task.spec.straight_length + task.spec.arc_length + target[..., 1] - task.spec.turn_radius
        remaining = stop_progress - progress
        cruise, turn = SPEED_BANDS[self.speed_band]
        turning = ((progress > task.spec.straight_length - .35)
                   & (progress < task.spec.straight_length + task.spec.arc_length + .15))
        cap = np.minimum(np.where(turning, turn, cruise), np.sqrt(np.maximum(0., .7 * remaining)))
        distance = np.linalg.norm(state[..., :2] - target, axis=-1)
        cap = np.where(remaining <= 0., np.minimum(cruise, np.sqrt(.7 * distance)), cap)
        return np.where(remaining < .35, np.maximum(cap, .06), cap)

    def govern(self, state, raw_command, task, memory=None):
        if not hasattr(task.spec, "checkpoints"):
            return super().govern(state, raw_command, task)
        command = np.clip(np.asarray(raw_command, dtype=float),
                          self.config.command_min, self.config.command_max).copy()
        norm = np.linalg.norm(command[..., :2], axis=-1)
        cap = self._stop_cap(state, task, memory)
        command[..., :2] *= np.minimum(1., cap / np.maximum(norm, 1e-12))[..., None]
        return command

    def feedback_command(self, state, task, memory=None):
        if not hasattr(task.spec, "checkpoints"):
            return super().feedback_command(state, task)
        state = np.asarray(state, dtype=float)
        target = self._stop_target(state, task, memory).reshape(2)
        progress = float(task.project(state[:2]))
        stop_progress = task.spec.straight_length + task.spec.arc_length + target[1] - task.spec.turn_radius
        ref = task.reference(min(stop_progress, progress + .22))
        tangent = np.array([np.cos(ref[2]), np.sin(ref[2])])
        error = ref[:2] - state[:2]
        desired = float(np.asarray(self._stop_cap(state, task, memory)).item()) * tangent
        desired += 1.1 * (error - np.dot(error, tangent) * tangent)
        if stop_progress - progress < .35:
            desired = 1.2 * (target - state[:2])
        c, s = np.cos(state[2]), np.sin(state[2])
        angle = ref[2] - state[2]
        raw = np.array([c * desired[0] + s * desired[1], -s * desired[0] + c * desired[1],
                        1.8 * np.arctan2(np.sin(angle), np.cos(angle))])
        return self.govern(state, raw, task, memory)

    @torch.no_grad()
    def _full_rollout(self, raw_commands, task):
        commands = np.asarray(raw_commands, dtype=float)
        single = commands.ndim == 2
        if single:
            commands = commands[None]
        full, memory = self._initial_batch(len(commands))
        planar_trace, issued_trace, full_trace, memory_trace = [], [], [], []
        for index in range(commands.shape[1]):
            planar = planar_state(full).cpu().numpy()
            issued = self.govern(planar, commands[:, index], task, memory)
            command = self._tensor(issued)
            context = task_context(full, memory, task.spec)
            active = task_active(memory, task.spec).unsqueeze(-1)
            full = torch.where(active, self._predict_response(full, context, command), full)
            memory = advance_memory(full, memory, task.spec, self.config.dt)
            planar_trace.append(planar_state(full).cpu().numpy())
            issued_trace.append(issued)
            full_trace.append(full)
            memory_trace.append(memory)
        self.response_steps += len(commands) * commands.shape[1]
        states = np.stack(planar_trace, axis=1)
        applied = np.stack(issued_trace, axis=1)
        self._last_full_trace = torch.stack(full_trace, dim=1)
        self._last_memory_trace = torch.stack(memory_trace, dim=1)
        return states, applied, full, memory, single

    def nominal_rollout(self, initial, raw_commands, task):
        """Original six-state trace API; propagation uses cached actual full48."""
        states, applied, full, memory, single = self._full_rollout(raw_commands, task)
        self._last_terminal = (full, memory, self._tensor(applied[:, -1]))
        return (states[0], applied[0]) if single else (states, applied)

    @torch.no_grad()
    def feedback_sequence(self, initial, task):
        """Geometric follower proposal propagated through the same factual g."""
        full, memory = self._initial_batch(1)
        states, commands = [], []
        for _ in range(self.config.horizon):
            planar = planar_state(full)[0].cpu().numpy()
            issued = self.feedback_command(planar, task, memory[0])
            context = task_context(full, memory, task.spec)
            active = task_active(memory, task.spec).unsqueeze(-1)
            proposed = self._predict_response(full, context, self._tensor(issued).reshape(1, 3))
            full = torch.where(active, proposed, full)
            memory = advance_memory(full, memory, task.spec, self.config.dt)
            commands.append(issued)
            states.append(planar_state(full)[0].cpu().numpy())
        self.response_steps += self.config.horizon
        return np.asarray(commands), np.asarray(states)

    @torch.no_grad()
    def _continuation(self, full, memory, command, task):
        if not self.use_terminal_value:
            return full.new_zeros(full.shape[0])
        context = task_context(full, memory, task.spec)
        values = self.value(full, context, command)
        self.value_calls += int(full.shape[0])
        # The real episode has no continuation after completion or timeout.
        # Predicted geometric completion does not certify collision feasibility.
        return torch.where(task_active(memory, task.spec), values, torch.zeros_like(values))

    def score(self, initial, commands, task, refs=None, speeds=None, return_states=False):
        states, applied, full, memory, single = self._full_rollout(commands, task)
        batch = full.shape[0]
        costs = mission_cost(
            self._last_full_trace, self._last_memory_trace, self._tensor(applied),
            self.current_memory.expand(batch, -1),
            self._tensor(self.last_command).expand(batch, -1), task, self.config.dt,
            speed_band=self.speed_band, collision_margin=self.config.collision_margin)
        values = self._continuation(full, memory, self._tensor(applied[:, -1]), task)
        costs = (costs - values).detach().cpu().numpy()
        if single:
            return (float(costs[0]), states[0]) if return_states else float(costs[0])
        return (costs, states) if return_states else costs

    def plan(self, obs, task):
        start = perf_counter()
        state = (obs["learning_state48"] if "learning_state48" in obs
                 else state_from_observation(obs))
        memory = (obs["learning_memory3"] if "learning_memory3" in obs
                  else memory_from_task(obs, task))
        self.set_state(state, memory)
        self.spec = task.spec
        if self.response_mode == "local_response":
            self._update_velocity_offset()
        steps_before, values_before = self.response_steps, self.value_calls
        # The inherited CEM only needs a planar observation.  Derive it from
        # the same full state propagated by g, including when an adapter uses
        # the explicit learning_state48/learning_memory3 observation fields.
        planar = planar_state(self.current_state)[0].detach().cpu().numpy()
        if self.controller_mode == "follower":
            command = self.feedback_command(planar, task, self.current_memory[0])
            self.last_command = command.copy()
            return command, {"controller": "geometric_follower", "solve_seconds": perf_counter() - start,
                             "predicted_cost": None, "response_model_steps_including_proposals": 0,
                             "terminal_value_evaluations": 0,
                             "command_sequence": np.tile(command, (self.config.horizon, 1)).tolist()}
        planning_obs = {
            "position": self.current_state[0, :3].detach().cpu().numpy(),
            "yaw": float(planar[2]),
            "velocity_world": self.current_state[0, 6:9].detach().cpu().numpy(),
            "yaw_rate": float(planar[5]),
        }
        feedback_first = self.feedback_command(planar, task, self.current_memory[0])
        command, info = super().plan(planning_obs, task)
        self._previous_measured_state = self.current_state.detach().clone()
        self._previous_issued_command = self._tensor(command).reshape(1, 3).clone()
        full, predicted_memory, tail_command = self._last_terminal
        info.update({
            "controller": "shared_factual_response_mission_mpc",
            "terminal_predicted_full_state": full[0].cpu().tolist(),
            "terminal_predicted_memory": predicted_memory[0].cpu().tolist(),
            "terminal_applied_command": tail_command[0].cpu().tolist(),
            "terminal_value_coefficient": float(self.use_terminal_value),
            "response_mode": self.response_mode,
            "first_command_delta_from_feedback": float(np.linalg.norm(command - feedback_first)),
            "terminal_value_macro_phase": 1.0,
            "response_model_steps_including_proposals": self.response_steps - steps_before,
            "terminal_value_evaluations": self.value_calls - values_before,
            "response_model_training_in_plan": 0,
            "policy_training_in_plan": 0,
        })
        return command, info
