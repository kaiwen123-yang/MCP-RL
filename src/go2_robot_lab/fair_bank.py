"""Bounded objective x predictor diagnostic, not an online-compute comparator.

Both selectors construct the same six command-plan candidates conditional on
the current actual state. Only the predictor used to rank them changes.
Full-physics selection has privileged simulator access and is costed separately.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from .fair_control import FairController
from .planner import observation_state


class CandidateBankController(FairController):
    def __init__(self, objective, predictor, forks=None, speed_band="fast"):
        super().__init__(objective, speed_band)
        if predictor not in ("nominal", "physics"):
            raise ValueError(predictor)
        self.predictor = predictor
        self.forks = forks
        self.generators = [FairController(m, speed_band) for m in ("tracking", "mission")]
        self.last_bank_detail = None

    def metadata(self):
        return {**super().metadata(), "controller": f"bank_{self.mode}_{self.predictor}",
                "purpose": "privileged bounded objective x predictor causal diagnostic",
                "prediction_selector": self.predictor,
                "candidate_count": 6,
                "candidate_names": ["follower", "tracking_cem", "mission_cem", "hold_previous",
                                    "mission_yaw_plus_0.12", "mission_yaw_minus_0.12"],
                "candidate_generation": "two nominal CEM proposals, same seeded noises conditional on state and decision index; no warm start",
                "physical_feasibility": "nominal planar footprint versus actual full-body contact/fall; infeasible candidates rank after feasible ones",
                "online_budget_comparable": False}

    def plan(self, obs, task):
        started = perf_counter()
        initial = observation_state(obs)
        previous = self.last_command.copy()
        proposals, proposal_work = [], []
        for generator in self.generators:
            generator.last_command = previous.copy()
            generator.warm_sequence = None
            generator.rng = np.random.default_rng(self.config.seed + 9973 * self.calls)
            _, info = generator.plan(obs, task)
            proposals.append(np.asarray(info["raw_command_sequence"]))
            proposal_work.append(info)
        follower, _ = self.feedback_sequence(initial, task)
        plus, minus = proposals[1].copy(), proposals[1].copy()
        plus[:, 2] += 0.12
        minus[:, 2] -= 0.12
        raw = np.stack([follower, proposals[0], proposals[1],
                        np.tile(previous, (self.config.horizon, 1)), plus, minus])
        raw = np.clip(raw, self.config.command_min, self.config.command_max)
        nominal_states, nominal_commands = self.nominal_rollout(initial, raw, task)
        nominal_scores = {m: np.asarray(self.score_trajectory(
            nominal_states, nominal_commands, task, initial, previous, m))
            for m in ("tracking", "mission")}
        _, clearance = self.footprint_penalty(nominal_states, task)
        nominal_feasible = np.all(clearance >= 0.0, axis=1)
        detail = {"candidate_names": self.metadata()["candidate_names"],
                  "raw_sequences": raw, "nominal_states": nominal_states,
                  "nominal_applied_commands": nominal_commands,
                  "nominal_scores": nominal_scores, "nominal_feasible": nominal_feasible}
        actual_states, actual_commands, actual_feasible = [], [], []
        branch_steps = branch_calls = 0
        branch_wall = 0.0
        if self.predictor == "physics":
            if self.forks is None:
                raise RuntimeError("Full-physics selector needs a snapshot runner")
            snapshot = self.forks.capture()
            for sequence in raw:
                result = self.forks.rollout(snapshot, sequence, command_dt=self.config.dt,
                    governor=lambda state, command: self.govern(observation_state(state), command, task))
                actual_states.append(result["states"][1:])
                actual_commands.append(result["applied_commands"])
                actual_feasible.append(not result["fallen_ever"] and not result["obstacle_contact_ever"])
                branch_steps += result["physics_steps"]
                branch_calls += result["policy_inference_calls"]
                branch_wall += result["wall_seconds"]
            actual_states, actual_commands = np.asarray(actual_states), np.asarray(actual_commands)
            actual_feasible = np.asarray(actual_feasible)
            actual_scores = {m: np.asarray(self.score_trajectory(
                actual_states, actual_commands, task, initial, previous, m))
                for m in ("tracking", "mission")}
            scores, feasible = actual_scores[self.mode], actual_feasible
            detail.update(physical_states=actual_states, physical_applied_commands=actual_commands,
                          physical_scores=actual_scores, physical_feasible=actual_feasible)
        else:
            scores, feasible = nominal_scores[self.mode], nominal_feasible
        eligible = np.flatnonzero(feasible)
        if not len(eligible):
            eligible = np.arange(len(raw))
        selected = int(eligible[np.argmin(scores[eligible])])
        command = self.govern(initial, raw[selected, 0], task)
        detail["selected_index"] = selected
        detail["nominal_selected_indices"] = {}
        for m in ("tracking", "mission"):
            ix = np.flatnonzero(nominal_feasible)
            if not len(ix):
                ix = np.arange(len(raw))
            detail["nominal_selected_indices"][m] = int(ix[np.argmin(nominal_scores[m][ix])])
        self.last_bank_detail = detail
        elapsed = perf_counter() - started
        self.last_command = command.copy()
        self.calls += 1
        self.total_seconds += elapsed
        return command, {
            "controller": f"bank_{self.mode}_{self.predictor}",
            "speed_band": self.speed_band, "solve_seconds": elapsed,
            "candidate_rollouts": sum(w["candidate_rollouts"] for w in proposal_work),
            "model_steps": sum(w["model_steps"] for w in proposal_work),
            "bank_nominal_model_steps": len(raw) * self.config.horizon,
            "raw_command_sequence": raw[selected].tolist(),
            "command_sequence": nominal_commands[selected].tolist(),
            "prediction_trace": nominal_states[selected].tolist(),
            "selected_candidate": detail["candidate_names"][selected],
            "oracle_physics_steps": branch_steps, "oracle_policy_calls": branch_calls,
            "oracle_wall_seconds": branch_wall,
            "model_ranking": detail,
        }
