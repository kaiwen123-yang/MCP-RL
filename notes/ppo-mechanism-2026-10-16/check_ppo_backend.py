"""Small mechanism audit, not a performance or robotics training run."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import BasePolicy, PPOPolicy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from response_ppo import PPOBackend, PPOConfig, Rollout, tensor_gae


def run() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(1601)
    np.random.seed(1601)
    backend = PPOBackend(PPOConfig(obs_dim=3, hidden=(16, 16), epochs=3, minibatch_size=16))
    assert backend.policy.learn.__func__ is PPOPolicy.learn
    x = torch.zeros(4, 3)
    x[:, 0] = torch.tensor([0.3, -0.4, 0.2, -0.1])
    fields = {key: [] for key in Rollout.__dataclass_fields__}
    maximum_action = 0.0
    for t in range(16):
        sample = backend.act(x)
        successor = x.clone()
        successor[:, 1] = 0.9 * x[:, 1] + 0.1 * sample.action[:, 0]
        successor[:, 0] = x[:, 0] + 0.1 * successor[:, 1]
        successor[:, 2] = (t + 1) / 16
        reward = -successor[:, 0].square() - 0.01 * sample.action[:, 0].square()
        terminal = torch.zeros(4, dtype=torch.bool)
        truncated = torch.zeros(4, dtype=torch.bool)
        terminal[1] = t == 10
        truncated[0] = t == 7
        data = dict(obs=x.clone(), raw_action=sample.raw_action, reward=reward,
                    terminated=terminal, truncated=truncated, log_prob=sample.log_prob,
                    value=sample.value, next_value=backend.value(successor))
        for key, value in data.items():
            fields[key].append(value)
        maximum_action = max(maximum_action, sample.action.abs().max().item())
        x = successor
        x[terminal | truncated] = 0.0
    rollout = Rollout(**{key: torch.stack(value) for key, value in fields.items()})
    advantage, returns = tensor_gae(
        rollout.reward, rollout.value, rollout.next_value,
        rollout.terminated, rollout.truncated, 0.99, 0.95,
    )
    gae_error = 0.0
    for environment in range(4):
        terminated = rollout.terminated[:, environment].numpy()
        truncated = rollout.truncated[:, environment].numpy()
        buffer = SimpleNamespace(terminated=terminated, unfinished_index=lambda: np.array([15]))
        upstream_returns, upstream_advantage = BasePolicy.compute_episodic_return(
            Batch(rew=rollout.reward[:, environment].numpy(),
                  terminated=terminated, truncated=truncated),
            buffer, np.arange(16), rollout.next_value[:, environment].numpy(),
            rollout.value[:, environment].numpy(), gamma=0.99, gae_lambda=0.95,
        )
        gae_error = max(gae_error, float(np.max(np.abs(upstream_returns - returns[:, environment].numpy()))),
                        float(np.max(np.abs(upstream_advantage - advantage[:, environment].numpy()))))
    assert gae_error < 2e-6
    # Independent boundary checks: truncation includes successor value but
    # excludes the reset episode's advantage; true termination excludes both.
    truncation_target = rollout.reward[7, 0] + 0.99 * rollout.next_value[7, 0]
    terminal_target = rollout.reward[10, 1]
    truncation_error = (returns[7, 0] - truncation_target).abs().item()
    terminal_error = (returns[10, 1] - terminal_target).abs().item()
    assert truncation_error < 1e-6 and terminal_error < 1e-6
    obs = rollout.obs.reshape(-1, 3)
    raw = rollout.raw_action.reshape(-1, 1)
    with torch.no_grad():
        old_physical_logp = backend.action_log_prob(obs, raw).clone()
        old_raw_logp = backend.policy(Batch(obs=obs, info=Batch())).dist.log_prob(raw).clone()
    actor_before = torch.cat([p.detach().flatten().clone() for p in backend.actor.parameters()])
    critic_before = torch.cat([p.detach().flatten().clone() for p in backend.critic.parameters()])
    metrics = backend.update(rollout)
    with torch.no_grad():
        new_physical_logp = backend.action_log_prob(obs, raw)
        new_raw_logp = backend.policy(Batch(obs=obs, info=Batch())).dist.log_prob(raw)
        ratio_error = ((new_physical_logp - old_physical_logp) - (new_raw_logp - old_raw_logp)).abs().max().item()
    actor_delta = (torch.cat([p.detach().flatten() for p in backend.actor.parameters()]) - actor_before).norm().item()
    critic_delta = (torch.cat([p.detach().flatten() for p in backend.critic.parameters()]) - critic_before).norm().item()
    assert actor_delta > 0 and critic_delta > 0
    assert ratio_error < 1e-6
    assert maximum_action < backend.config.action_bound
    assert metrics["optimizer_steps"] == 12
    return {
        "status": "passed",
        "purpose": "64 synthetic transitions; 12 genuine upstream PPO optimizer steps; no performance claim",
        "provenance": backend.provenance(),
        "gae_max_error_vs_upstream": gae_error,
        "truncation_bootstrap_error": truncation_error,
        "terminal_bootstrap_error": terminal_error,
        "tanh_physical_vs_raw_logratio_error": ratio_error,
        "maximum_action": maximum_action,
        "actor_parameter_delta_l2": actor_delta,
        "critic_parameter_delta_l2": critic_delta,
        "metrics": metrics,
        "gpu_execution": "not tested; installed PyTorch is CPU-only",
    }


if __name__ == "__main__":
    result = run()
    output = Path(__file__).with_name("ppo-backend-smoke.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
