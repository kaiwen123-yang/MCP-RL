"""Common finite-candidate MPC for all three Go2 learning comparison arms.

The original mission proxy, command governor, CEM budget and warm-start rule
are retained.  A factual 48-state command response replaces the planar model;
the same actual-return critic adds a terminal continuation term in every arm.
The selected CEM sequence is frozen when a separate actor-credit calculation
uses it.  This module does not differentiate through CEM or update networks.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from go2_frozen.fair_control import FairController
from go2_frozen.planner import MPCConfig

from .state import (
    advance_memory,
    memory_from_task,
    planar_state,
    state_from_observation,
    task_context,
)


class LearnedMPC(FairController):
    """Original mission CEM with factual response and shared continuation value.

    ``plan(obs, task)`` reads the actual complete state and task memory.  The
    optional observation fields ``learning_state48`` and ``learning_memory3``
    provide the same data explicitly, for adapters without all original keys.
    Model rollout helpers are scoped to that cached current decision state.
    """

    def __init__(self, response, value, actor=None, spec=None, speed="fast",
                 config: MPCConfig | None = None):
        super().__init__(mode="mission", speed_band=speed, config=config or MPCConfig())
        if not math.isclose(float(response.dt), self.config.dt):
            raise ValueError("CommandResponse.dt must match the MPC macro interval")
        self.response = response
        self.value = value
        # The response describes this actor's executed behavior, but planning
        # uses g directly.  This reference is never queried as a hidden oracle.
        self.actor = actor
        self.spec = spec
        self.current_state = None
        self.current_memory = None
        self.response_steps = 0
        self.value_calls = 0
        self._last_terminal = None

    def metadata(self):
        return {
            **super().metadata(),
            "controller": "shared_factual_response_mission_mpc",
            "model": "full48_command_response_fitted_only_to_actual_transitions",
            "terminal_value": "minus actual-return critic; coefficient 1 in all arms",
            "terminal_value_inputs": "predicted full48, predicted task memory, final applied command, macro phase 1",
            "cem_policy": "original deterministic seed and shifted warm sequence",
            "policy_used_inside_mpc": False,
            "actor_credit": "separate fixed-plan local derivative; CEM is not differentiated",
            "zero_update_prior": "original planar lag plus non-planar identity; critic zero",
            "response_steps": self.response_steps,
            "value_calls": self.value_calls,
        }

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.response.state_scale.dtype,
                               device=self.response.state_scale.device)

    def _initial_batch(self, batch):
        if self.current_state is None or self.current_memory is None:
            raise RuntimeError("Call plan(obs, task) or set_state(state, memory) before rollout")
        return (self.current_state.expand(batch, -1).clone(),
                self.current_memory.expand(batch, -1).clone())

    def set_state(self, state, memory):
        """Bind actual full state and ordered task memory for one decision."""
        self.current_state = self._tensor(state).reshape(1, 48)
        self.current_memory = self._tensor(memory).reshape(1, 3)

    @torch.no_grad()
    def _full_rollout(self, raw_commands, task):
        commands = np.asarray(raw_commands, dtype=float)
        single = commands.ndim == 2
        if single:
            commands = commands[None]
        full, memory = self._initial_batch(len(commands))
        planar_trace, issued_trace = [], []
        for index in range(commands.shape[1]):
            planar = planar_state(full).cpu().numpy()
            issued = self.govern(planar, commands[:, index], task)
            command = self._tensor(issued)
            context = task_context(full, memory, task.spec)
            full = self.response(full, context, command)
            memory = advance_memory(full, memory, task.spec, self.config.dt)
            planar_trace.append(planar_state(full).cpu().numpy())
            issued_trace.append(issued)
        self.response_steps += len(commands) * commands.shape[1]
        states = np.stack(planar_trace, axis=1)
        applied = np.stack(issued_trace, axis=1)
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
            issued = self.feedback_command(planar, task)
            context = task_context(full, memory, task.spec)
            full = self.response(full, context, self._tensor(issued).reshape(1, 3))
            memory = advance_memory(full, memory, task.spec, self.config.dt)
            commands.append(issued)
            states.append(planar_state(full)[0].cpu().numpy())
        self.response_steps += self.config.horizon
        return np.asarray(commands), np.asarray(states)

    @torch.no_grad()
    def _continuation(self, full, memory, command, task):
        context = task_context(full, memory, task.spec)
        values = self.value(full, context, command)
        self.value_calls += int(full.shape[0])
        # The real episode has no continuation after completion or timeout.
        # Predicted geometric completion does not certify collision feasibility.
        terminal = (memory[..., 0] >= 4) | (memory[..., 1] >= task.spec.timeout)
        return torch.where(terminal, torch.zeros_like(values), values)

    def score(self, initial, commands, task, refs=None, speeds=None, return_states=False):
        states, applied, full, memory, single = self._full_rollout(commands, task)
        costs = self.score_trajectory(states, applied, task, initial,
                                      self.last_command, objective="mission")
        values = self._continuation(full, memory, self._tensor(applied[:, -1]), task)
        costs = np.asarray(costs) - values.cpu().numpy()
        if single:
            return (float(costs[0]), states[0]) if return_states else float(costs[0])
        return (costs, states) if return_states else costs

    def plan(self, obs, task):
        state = (obs["learning_state48"] if "learning_state48" in obs
                 else state_from_observation(obs))
        memory = (obs["learning_memory3"] if "learning_memory3" in obs
                  else memory_from_task(obs, task))
        self.set_state(state, memory)
        steps_before, values_before = self.response_steps, self.value_calls
        # The inherited CEM only needs a planar observation.  Derive it from
        # the same full state propagated by g, including when an adapter uses
        # the explicit learning_state48/learning_memory3 observation fields.
        planar = planar_state(self.current_state)[0].detach().cpu().numpy()
        planning_obs = {
            "position": self.current_state[0, :3].detach().cpu().numpy(),
            "yaw": float(planar[2]),
            "velocity_world": self.current_state[0, 6:9].detach().cpu().numpy(),
            "yaw_rate": float(planar[5]),
        }
        command, info = super().plan(planning_obs, task)
        full, predicted_memory, tail_command = self._last_terminal
        info.update({
            "controller": "shared_factual_response_mission_mpc",
            "terminal_predicted_full_state": full[0].cpu().tolist(),
            "terminal_predicted_memory": predicted_memory[0].cpu().tolist(),
            "terminal_applied_command": tail_command[0].cpu().tolist(),
            "terminal_value_coefficient": 1.0,
            "terminal_value_macro_phase": 1.0,
            "response_model_steps_including_proposals": self.response_steps - steps_before,
            "terminal_value_evaluations": self.value_calls - values_before,
            "response_model_training_in_plan": 0,
            "policy_training_in_plan": 0,
        })
        return command, info
