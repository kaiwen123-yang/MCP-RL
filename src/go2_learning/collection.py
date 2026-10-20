"""One physical execution path, shared by all three comparison arms."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from go2_frozen.task import Task, make_cases

from .execution import LearningGo2Sim
from .planner import LearnedMPC
from .state import memory_from_task, state_from_observation, task_context


def stack_rows(rows):
    if not rows:
        return {}
    return {k: torch.as_tensor(np.asarray([r[k] for r in rows])) for k in rows[0]}


def concatenate_batches(*batches):
    present = [b for b in batches if b]
    return {k: torch.cat([b[k] for b in present], dim=0) for k in present[0]}


def collect(actor, response, value, assets, output, *, case_names, seed,
            stochastic, steps_per_case=None, speed="fast",
            episodes_per_case=1, planner_config=None):
    """Collect exact policy-step caps, or complete deterministic episodes.

    Warmup is the unchanged public deterministic stand procedure and is never
    a PPO sample. A cap ending an unfinished episode is recorded as a sampling
    truncation, not as task success/failure. Macro ID labels require all 10 steps.
    """
    if stochastic and steps_per_case is None:
        raise ValueError("Stochastic collection requires a declared per-case budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(int(seed))
    rows, macro_rows, episodes, decisions = [], [], [], []
    stats = {"policy_steps": 0, "physics_steps": 0, "warmup_physics_steps": 0,
             "planner_calls": 0, "planner_seconds": 0.0,
             "planner_model_steps": 0, "planner_value_calls": 0, "actor_calls": 0}
    start = perf_counter()
    cases = {s.name: s for s in make_cases()}
    for case_index, name in enumerate(case_names):
        spec, case_steps, episode = cases[name], 0, 0
        task = Task(spec)
        sim = LearningGo2Sim(
            Path(assets) / "rl_sar/policy/go2",
            Path(assets) / "rl_sar_zoo/go2_description/mjcf",
            output / name, obstacles=task.obstacles())
        macro_steps = round(0.2 / sim.policy_interval)
        while ((steps_per_case is not None and case_steps < steps_per_case) or
               (steps_per_case is None and episode < episodes_per_case)):
            task = Task(spec)
            obs = sim.reset(y=spec.start_y, warmup_seconds=2.0)
            metrics = task.update(obs)
            stats["warmup_physics_steps"] += sim.physics_steps
            controller = LearnedMPC(response, value, speed=speed, config=planner_config)
            # All arms share streams and solver settings; trajectories may diverge.
            controller.rng = np.random.default_rng(seed + 1000 * case_index + episode)
            ep_steps = 0
            work, torque2, path2, reward_sum = 0.0, 0.0, 0.0, 0.0
            ep_decisions, stage_exits = [], []

            def plan(current_obs):
                command, detail = controller.plan(current_obs, task)
                stats["planner_calls"] += 1
                stats["planner_seconds"] += detail["solve_seconds"]
                stats["planner_model_steps"] += detail["response_model_steps_including_proposals"]
                stats["planner_value_calls"] += detail["terminal_value_evaluations"]
                ep_decisions.append({"time": current_obs["time"],
                                     "command": command.tolist(),
                                     "solve_seconds": detail["solve_seconds"],
                                     "predicted_cost": detail["predicted_cost"]})
                return command, detail

            command, plan_info = plan(obs)
            while True:
                state = torch.from_numpy(state_from_observation(obs))
                memory = torch.from_numpy(memory_from_task(obs, task))
                context = task_context(state, memory, spec)
                command_t = torch.as_tensor(command, dtype=torch.float32)
                phase = ep_steps % macro_steps
                if phase == 0:
                    macro_start = {"state": state.numpy().copy(),
                                   "context": context.numpy().copy(),
                                   "command": command_t.numpy().copy()}
                planned = plan_info["command_sequence"]
                next_planned = np.asarray(planned[min(1, len(planned) - 1)],
                                          dtype=np.float32)
                with torch.no_grad():
                    sample = actor.sample(state, context, command_t,
                                          deterministic=not stochastic, generator=generator)
                    old_value = float(value(state, context, command_t,
                                            state.new_tensor((macro_steps - phase) / macro_steps)))
                before_physics = sim.physics_steps
                old_stage = task.stage
                obs = sim.step_action(command, sample["executed"].cpu().numpy())
                metrics = task.update(obs)
                stats["physics_steps"] += sim.physics_steps - before_physics
                stats["policy_steps"] += 1
                stats["actor_calls"] += 1
                ep_steps += 1
                case_steps += 1
                # Same floating-time tolerance as the original runner's loop:
                # the 35 s mission deadline must not become a bootstrapped cut.
                deadline_reached = obs["time"] >= spec.timeout - 1e-8
                failure = bool(metrics["contact_failure"] or metrics["fallen_failure"] or
                               (deadline_reached and not metrics["success"]))
                terminated = bool(metrics["success"] or failure)
                truncated = bool(not terminated and steps_per_case is not None and
                                 case_steps >= steps_per_case)
                dt = sim.policy_interval
                # Same real training objective in every arm. No model cost/reward.
                reward = -dt - (2.0 * spec.timeout if failure else 0.0)
                reward_sum += reward
                work += float(np.abs(obs["torque"] * obs["joint_vel"]).sum()) * dt
                torque2 += float(np.square(obs["torque"]).sum()) * dt
                reference = task.reference(task.project(obs["position"][:2]))
                path2 += float(np.square(obs["position"][:2] - reference[:2]).sum()) * dt
                if old_stage != task.stage:
                    stage_exits.append({"from": old_stage, "to": task.stage,
                                        "time": obs["time"],
                                        "state": state_from_observation(obs).tolist()})
                next_state = torch.from_numpy(state_from_observation(obs))
                next_memory = torch.from_numpy(memory_from_task(obs, task))
                if ep_steps % macro_steps == 0:
                    macro_rows.append({**macro_start,
                                       "next_state": next_state.numpy().copy()})
                # Bootstrap with the actual next decision (including replan),
                # rather than pretending the previous command remains active.
                next_command = command
                if not terminated and ep_steps % macro_steps == 0:
                    next_command, next_info = plan(obs)
                with torch.no_grad():
                    next_value = (0.0 if terminated else float(value(
                        next_state, task_context(next_state, next_memory, spec),
                        torch.as_tensor(next_command, dtype=torch.float32),
                        state.new_tensor((macro_steps - ep_steps % macro_steps) / macro_steps))))
                rows.append({
                    "state": state.numpy(), "memory": memory.numpy(),
                    "context": context.numpy(), "command": command_t.numpy(),
                    "next_state": next_state.numpy(), "next_memory": next_memory.numpy(),
                    "next_planned_command": next_planned,
                    "raw_action": sample["raw"].cpu().numpy(),
                    "executed_action": sample["executed"].cpu().numpy(),
                    "old_mean": sample["mean"].cpu().numpy(),
                    "old_log_prob": np.float32(sample["log_prob"].item()),
                    "old_value": np.float32(old_value),
                    "reward": np.float32(reward), "next_value": np.float32(next_value),
                    "terminated": terminated, "truncated": truncated,
                    "horizon": macro_steps - phase, "case_index": case_index,
                })
                if terminated or truncated:
                    reason = ("success" if metrics["success"] else
                              "collision" if metrics["contact_failure"] else
                              "fall" if metrics["fallen_failure"] else
                              "deadline" if deadline_reached else "sampling_cut")
                    episodes.append({
                        "case": name, "episode": episode, "reason": reason,
                        "success": bool(metrics["success"]), "complete": terminated,
                        "elapsed_seconds": obs["time"],
                        "completion_seconds": obs["time"] if metrics["success"] else None,
                        "real_return": reward_sum, "policy_steps": ep_steps,
                        "mechanical_work_proxy_J": work, "torque_squared_integral": torque2,
                        "path_rms_m": float(np.sqrt(path2 / max(obs["time"], dt))),
                        "stage_exits": stage_exits, "task": spec.to_dict(),
                        "source_policy_provenance": sim.provenance(),
                    })
                    decisions.append({"case": name, "episode": episode,
                                      "decisions": ep_decisions})
                    episode += 1
                    break
                if ep_steps % macro_steps == 0:
                    command, plan_info = next_command, next_info
    stats["warmup_and_execution_physics_steps"] = (
        stats["physics_steps"] + stats["warmup_physics_steps"])
    stats["collection_seconds"] = perf_counter() - start
    return {"batch": stack_rows(rows), "macro": stack_rows(macro_rows),
            "episodes": episodes, "decisions": decisions, "stats": stats,
            "behavior": "gaussian_raw_actions" if stochastic else "deterministic_mean"}
