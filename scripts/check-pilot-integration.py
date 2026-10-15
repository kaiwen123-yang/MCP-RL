"""Mechanism check: original planner/Hierarchy dispatch, not performance evidence."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import gymnasium as gym
import torch
from response_codesign.splates_bridge import build_splates_planner, build_hierarchy

torch.set_num_threads(1)


def objective(state, commands, target):
    predicted = state[:, None, :].expand(-1, commands.shape[1], -1)
    cost = torch.zeros(commands.shape[:2])
    for t in range(commands.shape[2]):
        predicted = predicted + 0.1 * commands[:, :, t]
        cost = cost + (predicted[..., 0] - target).square()
    margin = commands.abs().amax(dim=(-1, -2)) - 0.75
    return cost, margin


class ScalarPlant(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10, 10, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.state = np.zeros(1, dtype=np.float32)

    def step(self, action):
        self.state += np.asarray(action, dtype=np.float32) * 0.1
        return self.state.copy(), -float((self.state[0] - 0.75) ** 2), False, False, {}


class BatchedScalarPlant(gym.vector.VectorEnv):
    def __init__(self, count=8):
        self.num_envs = count
        self.single_observation_space = gym.spaces.Box(-10, 10, shape=(1,), dtype=np.float32)
        self.single_action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, count)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, count)
        self.state = np.zeros((count, 1), dtype=np.float32)

    def step(self, action):
        self.state += np.asarray(action, dtype=np.float32) * 0.01
        reward = -(self.state[:, 0] - 0.75) ** 2
        return self.state.copy(), reward, np.zeros(self.num_envs, bool), np.zeros(self.num_envs, bool), {}


def main():
    planner = build_splates_planner(objective, horizon=3, n_samples=24, n_iterations=2)
    plan = planner.plan(torch.zeros(1, 1), 0.75, seed=12)
    assert plan.feasible.all()
    assert plan.info["predictor_rollout_calls"] == 2
    assert "layeredrl-code" in plan.info["upstream_file"]
    assert plan.info["evaluated_sequences_per_instance"] == 50
    assert plan.info["objective_calls"] == 3
    assert plan.info["candidate_sequences"] == 50
    assert plan.info["candidate_macro_steps"] == 150
    neural_policy = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        neural_policy.weight[:] = torch.tensor([[0.0, 1.0]])
    def policy(obs, command):
        with torch.no_grad():
            return neural_policy(torch.cat((obs, command), dim=-1))
    events = {"transitions": 0, "joint_updates": 0}
    def record(*args):
        events["transitions"] += 1
    def update():
        events["joint_updates"] += 1
    env = ScalarPlant()
    planner.reset()
    before = planner.n_plans
    hierarchy = build_hierarchy(planner, policy, env, lambda obs, active: 0.75,
                                command_steps=2, transition_callback=record, joint_update=update)
    obs = torch.zeros(1, 1)
    for _ in range(6):
        action = hierarchy.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action[0].numpy())
        obs_next = torch.as_tensor(next_obs)[None]
        hierarchy.process_transition(obs_next, torch.tensor([reward]),
                                     torch.tensor([terminated]), torch.tensor([truncated]))
        obs = obs_next
    hierarchy.learn()
    assert planner.n_plans - before == 3
    assert events == {"transitions": 6, "joint_updates": 1}
    assert hierarchy.n_total_env_steps == 6
    impossible = build_splates_planner(lambda s, c, t: (c.square().sum((-1, -2)),
                                                       torch.ones(c.shape[:2])),
                                       horizon=2, n_samples=12, n_iterations=1)
    impossible_result = impossible.plan(torch.zeros(1, 1), None, seed=12)
    assert not impossible_result.feasible.any()
    assert impossible_result.info["objective_calls"] == 3
    assert impossible_result.info["candidate_sequences"] == 15
    assert impossible_result.info["candidate_macro_steps"] == 30
    actual_calls = {"calls": 0, "sequences": 0}
    def mixed_objective(state, commands, task):
        actual_calls["calls"] += 1
        actual_calls["sequences"] += commands.shape[0] * commands.shape[1]
        margins = torch.ones(commands.shape[:2])
        margins[0] = -1
        return commands.square().sum((-1, -2)), margins
    mixed = build_splates_planner(mixed_objective, horizon=2, n_samples=12, n_iterations=1)
    mixed_result = mixed.plan(torch.zeros(2, 1), None, seed=7)
    assert mixed_result.feasible.tolist() == [True, False]
    assert actual_calls == {"calls": 3, "sequences": 30}
    assert mixed_result.info["candidate_sequences"] == 30
    assert mixed_result.info["candidate_macro_steps"] == 60
    mixed_second = mixed.plan(torch.zeros(2, 1), None, seed=7)
    assert mixed_second.info["candidate_sequences"] == 30
    assert mixed.objective_counts()["candidate_sequences"] == 60
    batched_planner = build_splates_planner(objective, horizon=3, n_samples=24, n_iterations=2)
    batched_env = BatchedScalarPlant(8)
    batch_events = {"transitions": 0, "joint_updates": 0}
    def record_batch(obs, *args):
        batch_events["transitions"] += obs.shape[0]
    def update_batch():
        batch_events["joint_updates"] += 1
    batched_hierarchy = build_hierarchy(
        batched_planner, policy, batched_env, lambda obs, active: 0.75,
        command_steps=10, transition_callback=record_batch, joint_update=update_batch)
    batch_obs = torch.zeros(8, 1)
    for _ in range(100):
        batch_action = batched_hierarchy.get_action(batch_obs)
        nxt, rew, term, trunc, _ = batched_env.step(batch_action.numpy())
        next_batch_obs = torch.as_tensor(nxt)
        batched_hierarchy.process_transition(next_batch_obs, rew, term, trunc)
        batch_obs = next_batch_obs
    batched_hierarchy.learn()
    assert batched_planner.n_plans == 10
    assert batch_events == {"transitions": 800, "joint_updates": 1}
    assert batched_hierarchy.n_total_env_steps == 800
    assert batched_planner.total_objective_calls == 30
    assert batched_planner.total_candidate_sequences == 4000
    output = {
        "scope": "integration mechanism, not trained performance",
        "upstream_class": plan.info["upstream_class"],
        "upstream_file": plan.info["upstream_file"],
        "upstream_commit": plan.info["upstream_commit"],
        "hierarchy_class": type(hierarchy).__module__ + ".Hierarchy",
        "low_level_base": type(hierarchy.levels[1]).__bases__[0].__module__ + ".Level",
        "predictor_rollout_calls": plan.info["predictor_rollout_calls"],
        "candidate_sequences": plan.info["evaluated_sequences_per_instance"],
        "objective_calls": plan.info["objective_calls"],
        "all_infeasible_objective_counts": impossible.objective_counts(),
        "mixed_feasibility_external_counter": actual_calls,
        "mixed_feasibility_internal_counter": mixed.objective_counts(),
        "hierarchy_physical_steps": hierarchy.n_total_env_steps,
        "hierarchy_planner_calls": planner.n_plans - before,
        **events,
        "all_infeasible_reported_false": not bool(impossible_result.feasible.any()),
        "torch_version": torch.__version__,
        "upstream_methods": plan.info["upstream_methods"],
        "predictor_base_source": plan.info["predictor_base_source"],
        "hierarchy_provenance": hierarchy.integration_provenance,
        "batch8_physical_steps_per_env": 100,
        "batch8_total_env_steps": batched_hierarchy.n_total_env_steps,
        "batch8_planning_batch_calls": batched_planner.n_plans,
        "batch8_joint_updates": batch_events["joint_updates"],
        "batch8_objective_counts": batched_planner.objective_counts(),
        "first_command": plan.action.tolist(),
        "final_state": obs.tolist(),
    }
    path = ROOT / "work" / "learning-pilot-2026-10-15" / "integration-check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

