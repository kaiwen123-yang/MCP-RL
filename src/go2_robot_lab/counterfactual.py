"""Same-state physical rollouts for bounded, privileged model diagnostics.

This module never trains the actor and never changes the scene. It temporarily
forks a running ``Go2Sim`` at a complete policy boundary, executes a specified
command sequence, and restores the caller's exact simulator and policy state.
The additional simulator work is counted separately from the actual task run.
Snapshots stay in memory; no policy, asset or MuJoCo binary is exported.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import mujoco
import numpy as np

from .planner import observation_state
from .simulator import Go2Sim


@dataclass
class FullSnapshot:
    """Full ``MjData`` plus all mutable deployment and accounting state."""

    model_identity: int
    data: mujoco.MjData
    last_action: np.ndarray
    history: np.ndarray
    policy_calls: int
    command: np.ndarray
    q_des: np.ndarray
    last_tau_command: np.ndarray
    task_start_time: float
    physics_steps: int
    obstacle_contact_ever: bool
    fallen_ever: bool


class CounterfactualRollouts:
    """Physical candidate replay; deliberately not an online MPC baseline.

    ``Task`` is not accepted or updated. A caller needing stage transitions can
    update a separate copy of its task from the returned observations. The real
    task, controller RNG, and planning warm start are untouched.
    """

    def __init__(self, sim: Go2Sim):
        self.sim = sim
        if sim.policy.model.training or any(
            p.requires_grad for p in sim.policy.model.parameters()
        ):
            raise ValueError("Counterfactual replay requires a frozen eval actor")
        self.diagnostics: dict[str, Any] = {
            "purpose": "privileged same-state physical candidate diagnostic",
            "rollouts_attempted": 0,
            "rollouts_completed": 0,
            "physics_steps": 0,
            "policy_inference_calls": 0,
            "wall_seconds": 0.0,
            "policy_training_updates": 0,
            "model_training_updates": 0,
        }

    def capture(self) -> FullSnapshot:
        sim = self.sim
        data = mujoco.MjData(sim.model)
        # mj_copyData includes time, integration state, ctrl, warm start,
        # contact/constraint and sensor buffers. No forward/reset follows it.
        mujoco.mj_copyData(data, sim.model, sim.data)
        return FullSnapshot(
            model_identity=id(sim.model), data=data,
            last_action=sim.policy.last_action.copy(),
            history=sim.policy.history.copy(), policy_calls=sim.policy.calls,
            command=sim.command.copy(), q_des=sim.q_des.copy(),
            last_tau_command=sim.last_tau_command.copy(),
            task_start_time=sim.task_start_time, physics_steps=sim.physics_steps,
            obstacle_contact_ever=sim.obstacle_contact_ever,
            fallen_ever=sim.fallen_ever,
        )

    def restore(self, snapshot: FullSnapshot) -> None:
        sim = self.sim
        if snapshot.model_identity != id(sim.model):
            raise ValueError("Snapshot belongs to a different MuJoCo model")
        mujoco.mj_copyData(sim.data, sim.model, snapshot.data)
        sim.policy.last_action = snapshot.last_action.copy()
        sim.policy.history = snapshot.history.copy()
        sim.policy.calls = snapshot.policy_calls
        sim.command = snapshot.command.copy()
        sim.q_des = snapshot.q_des.copy()
        sim.last_tau_command = snapshot.last_tau_command.copy()
        sim.task_start_time = snapshot.task_start_time
        sim.physics_steps = snapshot.physics_steps
        sim.obstacle_contact_ever = snapshot.obstacle_contact_ever
        sim.fallen_ever = snapshot.fallen_ever

    def rollout(
        self, snapshot: FullSnapshot, commands: np.ndarray,
        command_dt: float = 0.2,
        governor: Callable[[dict, np.ndarray], np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Replay H commands and always restore the state held on entry.

        The governor is invoked once at each command boundary, using that
        branch's actual observation. A governed command is held until the next
        boundary. ``states``/``boundary_observations`` contain H+1 entries;
        ``observations`` contains initial plus every 50 Hz policy output.
        """
        raw = np.asarray(commands, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0 or not np.isfinite(raw).all():
            raise ValueError("Expected a finite, nonempty H by 3 command array")
        steps_per_command = round(command_dt / self.sim.policy_interval)
        if steps_per_command < 1 or not np.isclose(
            steps_per_command * self.sim.policy_interval, command_dt
        ):
            raise ValueError("Command interval must be a positive integer policy interval")
        # Validate before entering the branch so a mismatched snapshot cannot
        # alter accounting or the live simulator.
        if snapshot.model_identity != id(self.sim.model):
            raise ValueError("Snapshot belongs to a different MuJoCo model")
        caller = self.capture()
        started = perf_counter()
        self.diagnostics["rollouts_attempted"] += 1
        self.restore(snapshot)
        initial_physics = self.sim.physics_steps
        initial_calls = self.sim.policy.calls
        try:
            obs = self.sim.observe()
            observations = [obs]
            boundaries = [obs]
            states = [observation_state(obs)]
            applied = []
            for command in raw:
                actual_command = command.copy() if governor is None else np.asarray(
                    governor(obs, command.copy()), dtype=np.float64
                )
                if actual_command.shape != (3,) or not np.isfinite(actual_command).all():
                    raise ValueError("Governor must return one finite three-axis command")
                applied.append(actual_command.copy())
                for _ in range(steps_per_command):
                    obs = self.sim.step(actual_command)
                    observations.append(obs)
                boundaries.append(obs)
                states.append(observation_state(obs))
            result = {
                "states": np.asarray(states),
                "observations": observations,
                "boundary_observations": boundaries,
                "raw_commands": raw.copy(),
                "applied_commands": np.asarray(applied),
                "command_dt": float(command_dt),
                "policy_interval": float(self.sim.policy_interval),
                "simulation_seconds": len(raw) * float(command_dt),
                "physics_steps": self.sim.physics_steps - initial_physics,
                "policy_inference_calls": self.sim.policy.calls - initial_calls,
                "obstacle_contact_ever": bool(self.sim.obstacle_contact_ever),
                "fallen_ever": bool(self.sim.fallen_ever),
                "wall_seconds": perf_counter() - started,
            }
            self.diagnostics["rollouts_completed"] += 1
            return result
        finally:
            self.diagnostics["physics_steps"] += self.sim.physics_steps - initial_physics
            self.diagnostics["policy_inference_calls"] += self.sim.policy.calls - initial_calls
            self.restore(caller)
            self.diagnostics["wall_seconds"] += perf_counter() - started
