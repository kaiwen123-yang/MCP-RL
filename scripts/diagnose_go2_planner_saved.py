"""Read-only diagnosis of stored Go2 decisions; no simulation or optimizer.

Replay the original CEM on stored states and RNG streams, then hold its candidate
pool fixed while replacing only saved model/value components. Such rescoring
explains model preferences, never counterfactual physical task performance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from go2_robot_lab.task import Task, make_cases
from go2_robot_lab.planner import MPCConfig
from go2_robot_lab.fair_control import MISSION_WEIGHTS
from go2_learning.models import CommandResponse, ValueNet
from go2_learning.planner import LearnedMPC
from go2_learning.state import advance_memory, planar_state, task_context


class CapturedMPC(LearnedMPC):
    def score(self, initial, commands, task, **kwargs):
        self.pool.append(np.asarray(commands).copy())
        return super().score(initial, commands, task, **kwargs)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def model_pair(checkpoint, config):
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    g = CommandResponse(dt=config["macro_dt"], hidden=config["model_hidden"])
    v = ValueNet(hidden=config["value_hidden"])
    g.load_state_dict(saved["response"])
    v.load_state_dict(saved["value"])
    return g.eval(), v.eval()


def task_at(spec, state, memory):
    task = Task(spec)
    task.stage_index = int(memory[0])
    task.progress = float(task.project(state[:2].numpy()))
    task.stop_since = None if memory[2] < 0 else float(memory[1] - memory[2])
    return task


def score_parts(controller, pool, state, memory, previous_command, task):
    controller.set_state(state, memory)
    controller.last_command = previous_command.copy()
    initial = planar_state(state).numpy()
    states, issued, full, predicted_memory, _ = controller._full_rollout(pool, task)
    proxy = np.asarray(controller.score_trajectory(
        states, issued, task, initial, previous_command, objective="mission"))
    values = controller._continuation(
        full, predicted_memory, controller._tensor(issued[:, -1]), task).numpy()
    return {
        "total": proxy - values, "proxy": proxy, "minus_value": -values,
        "first_command": issued[:, 0], "terminal_position": full[:, :2].numpy(),
        "terminal_speed": torch.linalg.vector_norm(full[:, 6:8], dim=-1).numpy(),
        "terminal_stage": predicted_memory[:, 0].numpy(),
        "terminal_hold": predicted_memory[:, 2].numpy(),
    }


def scalar_at(parts, index):
    return {
        "cost": float(parts["total"][index]),
        "geometry_proxy": float(parts["proxy"][index]),
        "minus_continuation_value": float(parts["minus_value"][index]),
        "command_x": float(parts["first_command"][index, 0]),
        "command_y": float(parts["first_command"][index, 1]),
        "command_yaw": float(parts["first_command"][index, 2]),
        "predicted_terminal_x": float(parts["terminal_position"][index, 0]),
        "predicted_terminal_y": float(parts["terminal_position"][index, 1]),
        "predicted_terminal_speed": float(parts["terminal_speed"][index]),
        "predicted_terminal_stage": int(parts["terminal_stage"][index]),
        "predicted_terminal_hold": float(parts["terminal_hold"][index]),
    }


def explain_trace(controller, sequence, state, memory, previous_command, task):
    """Decompose the unchanged production proxy and its predicted completion."""
    full, mem = state[None].clone(), memory[None].clone()
    previous = previous_command.copy()
    totals = dict(time=0.0, lateral=0.0, yaw=0.0, command=0.0,
                  command_change=0.0, collision=0.0)
    trace, completion = [], None
    after_completion = 0.0
    for k, raw in enumerate(sequence):
        before_complete = int(mem[0, 0]) >= 4
        planar = planar_state(full)[0].numpy()
        command = controller.govern(planar, raw, task)
        full = controller.response(full, task_context(full, mem, task.spec),
                                   controller._tensor(command)[None])
        mem = advance_memory(full, mem, task.spec, controller.config.dt)
        planar = planar_state(full)[0].numpy()
        ref = task.reference(task.project(planar[:2]))
        delta = planar[:2] - ref[:2]
        lateral = -delta[0] * np.sin(ref[2]) + delta[1] * np.cos(ref[2])
        yaw = np.arctan2(np.sin(planar[2] - ref[2]), np.cos(planar[2] - ref[2]))
        collision = controller.footprint_penalty(planar[None], task)[0]
        parts = dict(time=controller.config.dt, lateral=35.0 * lateral**2,
                     yaw=4.0 * yaw**2, command=.03 * float(np.sum(command**2)),
                     command_change=.25 * float(np.sum((command - previous)**2)),
                     collision=float(np.asarray(collision).reshape(-1)[0]))
        for name, value in parts.items():
            totals[name] += value
        if before_complete:
            after_completion += sum(parts.values())
        previous = command
        if completion is None and int(mem[0, 0]) >= 4:
            completion = (k + 1) * controller.config.dt
        trace.append({
            "prediction_step": k + 1, "seconds_from_observation": (k + 1) * controller.config.dt,
            "x": float(full[0, 0]), "y": float(full[0, 1]),
            "speed": float(torch.linalg.vector_norm(full[0, 6:8])),
            "stage": int(mem[0, 0]), "hold": float(mem[0, 2]),
            "command_x": float(command[0]), "command_y": float(command[1]),
            "command_yaw": float(command[2]),
            "running_proxy": sum(parts.values()), "after_predicted_completion": before_complete,
        })
    progress = float(task.project(full[0, :2].numpy()))
    near = task.spec.total_length - progress < MISSION_WEIGHTS["terminal_goal_zone_m"]
    remaining = 20.0 * float(controller.remaining_reference_time(progress, task))
    goal = 20.0 * float(np.sum((full[0, :2].numpy() - task.spec.goal)**2)) if near else 0.0
    velocity = 3.0 * float(torch.sum(full[0, 6:8]**2)) if near else 0.0
    value = float(controller._continuation(
        full, mem, controller._tensor(previous)[None], task)[0])
    result = {
        **{"running_" + k: v for k, v in totals.items()},
        "terminal_remaining_reference_penalty": remaining,
        "terminal_goal_position_penalty": goal,
        "terminal_velocity_penalty": velocity,
        "minus_value": -value,
        "reconstructed_total": sum(totals.values()) + remaining + goal + velocity - value,
        "first_predicted_complete_seconds": completion,
        "running_cost_strictly_after_predicted_complete": after_completion,
        "terminal_cost_despite_predicted_complete": remaining + goal + velocity if completion is not None else None,
    }
    return result, trace


@torch.no_grad()
def diagnose(runs, output):
    begin = perf_counter()
    torch.set_num_threads(1)
    config = json.loads((runs / "config.json").read_text())
    specs = {s.name: s for s in make_cases()}
    # Existing late-delay examples and their prior-round / ordinary comparators.
    selected = [
        (11, "adjoint", 4, "nominal"), (11, "ordinary", 4, "nominal"),
        (11, "frozen", 4, "narrow"),
        (11, "adjoint", 3, "nominal"), (11, "frozen", 3, "narrow"),
    ]
    source_files = {}
    replay_rows, rescoring_rows, model_rows, score_rows, trace_rows = [], [], [], [], []

    def register(path):
        source_files[str(path.relative_to(runs)).replace("\\", "/")] = (
            hashlib.sha256(path.read_bytes()).hexdigest())

    for seed, arm, round_number, case in selected:
        prefix = runs / f"seed{seed}" / arm / f"round{round_number}"
        before = runs / f"seed{seed}" / arm / f"round{round_number - 1}" / "checkpoint.pt"
        now = prefix / "checkpoint.pt"
        collection = prefix / "evaluation"
        for path in (before, now, collection / "transitions.pt", collection / "summary.json"):
            register(path)
        g, v = model_pair(now, config)
        old_g, old_v = model_pair(before, config)
        prior_g = CommandResponse(dt=config["macro_dt"], hidden=config["model_hidden"]).eval()
        zero_v = ValueNet(hidden=config["value_hidden"]).eval()
        systems = {
            "current_g_current_v": (g, v),
            "current_g_no_v": (g, zero_v),
            "previous_g_current_v": (old_g, v),
            "current_g_previous_v": (g, old_v),
            "previous_g_previous_v": (old_g, old_v),
            "engineering_prior_g_no_v": (prior_g, zero_v),
        }
        controllers = {
            name: LearnedMPC(response, value, speed=config["speed_band"],
                             config=MPCConfig(**config["planner"]))
            for name, (response, value) in systems.items()
        }
        planner = CapturedMPC(g, v, speed=config["speed_band"],
                              config=MPCConfig(**config["planner"]))
        case_index = config["cases"].index(case)
        planner.rng = np.random.default_rng(seed + 20000 + 1000 * case_index)
        data = torch.load(collection / "transitions.pt", weights_only=True, map_location="cpu")
        batch = data["batch"]
        indices = torch.where((batch["case_index"] == case_index) & (batch["horizon"] == 10))[0]
        summary = json.loads((collection / "summary.json").read_text())
        recorded = next(x["decisions"] for x in summary["decisions"] if x["case"] == case)
        assert len(indices) == len(recorded)
        spec = specs[case]
        terminal_indices = [
            j for j, index in enumerate(indices)
            if int(batch["memory"][index, 0]) == 3
            and float(torch.linalg.vector_norm(batch["state"][index, :2]
                                               - torch.tensor(spec.goal))) < .35
        ]
        # Sample up to five evenly spaced actual terminal decisions, endpoints included.
        selected_decisions = set()
        if terminal_indices:
            selected_decisions = {terminal_indices[int(k)] for k in np.linspace(
                0, len(terminal_indices) - 1, min(5, len(terminal_indices)), dtype=int)}

        for decision, (index, original) in enumerate(zip(indices, recorded)):
            state, memory = batch["state"][index], batch["memory"][index]
            task = task_at(spec, state, memory)
            previous_command = planner.last_command.copy()
            planner.pool = []
            command, info = planner.plan(
                {"learning_state48": state.numpy(), "learning_memory3": memory.numpy()}, task)
            error = float(np.max(np.abs(command - np.asarray(original["command"]))))
            replay_rows.append({
                "seed": seed, "arm": arm, "round": round_number, "case": case,
                "decision": decision, "time": original["time"], "stage": int(memory[0]),
                "command_max_abs_difference": error,
                "predicted_cost_abs_difference": abs(info["predicted_cost"] - original["predicted_cost"]),
                "actual_command_x": original["command"][0],
                "actual_command_y": original["command"][1],
                "actual_command_yaw": original["command"][2],
                "replay_command_x": float(command[0]),
                "replay_command_y": float(command[1]),
                "replay_command_yaw": float(command[2]),
            })
            if decision not in selected_decisions:
                continue
            pool = np.concatenate(planner.pool)
            baseline = score_parts(controllers["current_g_current_v"], pool, state, memory,
                                   previous_command, task)
            baseline_winner = int(np.argmin(baseline["total"]))
            # Candidate #1 is the production geometric-feedback proposal, present
            # in every CEM generation; #0 is the shifted warm-start sequence.
            follower_index = 1
            for role, candidate in (("original_winner", baseline_winner),
                                    ("geometric_follower", follower_index)):
                decomposition, trace = explain_trace(
                    controllers["current_g_current_v"], pool[candidate], state, memory,
                    previous_command, task)
                identity = {"seed": seed, "arm": arm, "round": round_number, "case": case,
                            "decision": decision, "time": original["time"], "candidate_role": role}
                score_rows.append({
                    **identity, **decomposition,
                    "production_score": float(baseline["total"][candidate]),
                    "decomposition_difference": decomposition["reconstructed_total"] - float(
                        baseline["total"][candidate]),
                })
                trace_rows.extend({**identity, **row} for row in trace)
            for name, controller in controllers.items():
                parts = baseline if name == "current_g_current_v" else score_parts(
                    controller, pool, state, memory, previous_command, task)
                own_winner = int(np.argmin(parts["total"]))
                for role, candidate in (
                        ("original_winner", baseline_winner),
                        ("geometric_follower", follower_index),
                        ("rescored_winner", own_winner)):
                    rescoring_rows.append({
                        "seed": seed, "arm": arm, "round": round_number, "case": case,
                        "decision": decision, "time": original["time"],
                        "distance_to_goal": float(torch.linalg.vector_norm(
                            state[:2] - torch.tensor(spec.goal))),
                        "actual_speed": float(torch.linalg.vector_norm(state[6:8])),
                        "actual_hold": float(memory[2]),
                        "replay_command_error": error, "model_components": name,
                        "candidate_role": role, "candidate_index": candidate,
                        "candidate_pool_size": len(pool), **scalar_at(parts, candidate),
                    })

        # True executed 200 ms command transitions in the final episode;
        # no model-generated endpoint is mistaken for observed ground truth.
        macro_indices = [int(i) for i in indices if int(i) + 9 < len(batch["state"])
                         and int(batch["case_index"][int(i) + 9]) == case_index
                         and not bool(batch["terminated"][int(i):int(i) + 9].any())]
        for index in macro_indices:
            state, context, command = (batch[k][index:index+1]
                                       for k in ("state", "context", "command"))
            target = batch["next_state"][index+9]
            memory = batch["memory"][index]
            for name, model in (("current_g", g), ("previous_g", old_g),
                                ("engineering_prior_g", prior_g)):
                predicted = model(state, context, command)[0]
                delta = predicted - target
                model_rows.append({
                    "seed": seed, "arm": arm, "round": round_number, "case": case,
                    "time": float(memory[1]), "stage": int(memory[0]), "model": name,
                    "distance_to_goal": float(torch.linalg.vector_norm(
                        state[0, :2] - torch.tensor(spec.goal))),
                    "command_translation_norm": float(torch.linalg.vector_norm(command[0, :2])),
                    "position_error_m": float(torch.linalg.vector_norm(delta[:2])),
                    "velocity_error_m_s": float(torch.linalg.vector_norm(delta[6:8])),
                    "y_error_m": float(delta[1]),
                    "vy_error_m_s": float(delta[7]),
                    "actual_speed": float(torch.linalg.vector_norm(target[6:8])),
                    "predicted_speed": float(torch.linalg.vector_norm(predicted[6:8])),
                })
        print(json.dumps({"case_complete": [seed, arm, round_number, case],
                          "replayed_decisions": len(indices),
                          "fixed_pool_probes": len(selected_decisions),
                          "elapsed_seconds": perf_counter() - begin}), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "decision_replay.csv", replay_rows)
    write_csv(output / "fixed_pool_rescoring.csv", rescoring_rows)
    write_csv(output / "observed_macro_prediction_errors.csv", model_rows)
    write_csv(output / "terminal_score_decomposition.csv", score_rows)
    write_csv(output / "terminal_predicted_traces.csv", trace_rows)
    write_json(output / "manifest.json", {
        "source": "unchanged completed clean run; existing trajectories and checkpoints",
        "selected_examples": selected,
        "selection_reason": "Previously reported seed11 terminal-delay examples, ordinary and prior-round controls",
        "input_sha256": source_files,
        "replayed_decisions": len(replay_rows),
        "replay_command_max_abs_difference": max(x["command_max_abs_difference"] for x in replay_rows),
        "replay_cost_max_abs_difference": max(x["predicted_cost_abs_difference"] for x in replay_rows),
        "recorded_fixed_pool_rows": len(rescoring_rows),
        "max_score_decomposition_abs_difference": max(
            abs(row["decomposition_difference"]) for row in score_rows),
        "wall_seconds": perf_counter() - begin,
        "torch": torch.__version__, "numpy": np.__version__,
        "additional_physics_steps": 0, "optimizer_updates": 0,
        "checkpoint_modifications": 0, "production_source_or_config_changes": 0,
        "limits": [
            "CEM replay only evaluates neural predictions on stored actual decision states.",
            "Fixed-pool alternatives explain component effects on scoring, not physical completion times.",
            "Previous g predicts the previous policy; it is a sensitivity control, not a validated replacement.",
            "Engineering prior and zero critic are the original declared initial maps, not retuned settings.",
            "Examples diagnose observed delays; they are not independent confirmatory test cases.",
        ],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.runs, args.output)
