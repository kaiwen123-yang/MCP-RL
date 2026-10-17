"""Independent native adapter whose MPC predicts the current mean actor.

The historical ResponseServo is unchanged. This adapter retains its physical
step, disturbance process, reference continuity, CEM settings and reference
domain selection gate. Additional actual/input/tube quantities are diagnostics.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .env import EnvironmentConfig, ResponseServo
from .closed_loop_prediction import (MeanActor, NativeContext, predict_macro_commands,
                                     planner_coordinates_to_execution, prefix_violation_summary)


class ExecutionAwareServo(ResponseServo):
    def __init__(self, config: EnvironmentConfig, raw_design: Tensor, seed: int,
                 steps: int, support: bool, actor_mean: MeanActor):
        # ResponseServo.__init__ calls _prepare_command, so these exist first.
        self.actor_mean = actor_mean
        self.execution_prediction_calls = 0
        self.execution_prediction_low_steps = 0
        self.plan_diagnostics: list[dict[str, Any]] = []
        self._objective_records: list[dict[str, Tensor]] = []
        super().__init__(config, raw_design, seed, steps, support)

    @torch.no_grad()
    def sequence_objective(self, state: Tensor, commands: Tensor, task: Any):
        start = int(task)
        x, z, design = state[:, :3], state[:, 3:6], state[:, 6:10]
        low_steps = commands.shape[-2]*self.config.command_steps
        goals = torch.stack([self.target(start+step) for step in range(low_steps)], dim=-1)
        context = NativeContext(self.config, self.P, self.B, self.E)
        prediction = predict_macro_commands(
            self.actor_mean, x, z, planner_coordinates_to_execution(commands), design,
            goals, start, context, support=self.support)
        diagnostics = prefix_violation_summary(prediction)
        cost = prediction["task_cost_mean"]
        # Deliberately retain the old reference-domain primary selection gate.
        margin = diagnostics["reference_domain_margin_max"]
        self.execution_prediction_calls += 1
        self.execution_prediction_low_steps += commands.shape[0]*commands.shape[1]*low_steps
        self._objective_records.append({"commands": commands.detach().clone(),
                                        "actual_task_cost": cost.detach(),
                                        "reference_task_cost": prediction["reference_task_cost_mean"].detach(),
                                        **{key: value.detach() for key, value in diagnostics.items()}})
        return cost, margin

    @torch.no_grad()
    def _prepare_command(self):
        self._objective_records = []
        result = self.planner.plan(torch.cat((self.x, self.z, self.raw_design), -1), self.t,
                                   seed=self.seed*10000+self.t)
        execution = planner_coordinates_to_execution(result.action)
        self.command[:, 0], self.command[:, 1] = execution[:, 0], execution[:, 1]
        self.planner_failures += (~result.feasible).sum().item()
        self.planner_instances += self.batch
        # Look up the chosen candidate in the already evaluated pool. This adds
        # no prediction calls and avoids mistaking the last pool for the winner.
        chosen: dict[str, Tensor] = {}
        found = torch.zeros(self.batch, dtype=torch.bool, device=self.x.device)
        for record in self._objective_records:
            matching = (record["commands"] == result.sequence[:, None]).all(dim=(-1, -2))
            present = matching.any(dim=1)
            indices = matching.to(torch.int64).argmax(dim=1)
            rows = torch.arange(self.batch, device=self.x.device)
            for name, value in record.items():
                if name == "commands":
                    continue
                selected = value[rows, indices]
                if name not in chosen:
                    chosen[name] = torch.zeros_like(selected)
                chosen[name] = torch.where(present, selected, chosen[name])
            found |= present
        self.plan_diagnostics.append({"tick": self.t, "selected_found_in_evaluated_pool": found,
                                      "selected_reference_gate_feasible": result.feasible.detach(),
                                      "selected_sequence_planner_coordinates": result.sequence.detach().clone(),
                                      "selected_execution_command": self.command.detach().clone(),
                                      "selected_diagnostics": chosen,
                                      "candidate_sequences_evaluated": result.info["candidate_sequences"],
                                      "primary_gate": "original reference-domain gate only",
                                      "additional_constraints": "actual/input/tube/support diagnostics, not newly imposed hard gates"})
        self._objective_records = []
