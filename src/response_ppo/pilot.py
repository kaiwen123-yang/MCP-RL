"""Matched small PPO mechanism pilot; default command runs development only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .env import EnvironmentConfig, ResponseServo, initial_design
from .ppo_backend import PPOBackend, PPOConfig, Rollout

ARMS = ("joint_support", "fixed_support", "joint_penalty", "fixed_penalty")


def collect(backend, design_mean, arm, seed, steps, environments=16, training=True,
            env_config=None, design_std=.2):
    config = env_config or EnvironmentConfig(device=backend.config.device)
    device = config.device
    design_generator = torch.Generator(device=device).manual_seed(seed+1000001)
    action_generator = torch.Generator(device=device).manual_seed(seed+2000003)
    # Both fixed and joint arms receive the same design randomization budget.
    # "Fixed" freezes its distribution mean; it is not an unrandomized policy.
    noise = torch.randn(environments, 4, generator=design_generator, device=device)
    raw_design = design_mean[None].expand(environments, -1) + (design_std*noise if training else 0.)
    started = time.perf_counter()
    env = ResponseServo(config, raw_design, seed, steps, support=arm.endswith("support"))
    obs = env.observation()
    fields = {key: [] for key in Rollout.__dataclass_fields__}
    history = {}
    for t in range(steps):
        sample = backend.act(obs, deterministic=not training, generator=action_generator)
        next_obs, reward, metric = env.step(torch.tanh(sample.raw_action[:, 0]))
        if training:
            terminal = torch.zeros(environments, dtype=torch.bool, device=device)
            truncated = torch.full((environments,), t == steps-1, dtype=torch.bool, device=device)
            data = dict(obs=obs, raw_action=sample.raw_action, reward=reward,
                        terminated=terminal, truncated=truncated, log_prob=sample.log_prob,
                        value=sample.value, next_value=backend.value(next_obs))
            for key, value in data.items():
                fields[key].append(value)
        for key, value in metric.items():
            history.setdefault(key, []).append(value)
        obs = next_obs
    arrays = {key: torch.stack(value) for key, value in history.items()}
    metrics = {
        "task_cost": arrays["task"].mean().item(), "total_cost": arrays["total"].mean().item(),
        "realized_violation_rate": (arrays["excess"]>1e-8).float().mean().item(),
        "mean_positive_excess": arrays["excess"].clamp_min(0).mean().item(),
        "max_excess": arrays["excess"].max().item(),
        "empty_support_rate": arrays["empty"].mean().item(),
        "mean_support_width": arrays["support_width"].mean().item(),
        "max_V": arrays["V"].max().item(),
        "geometric_bound_violation_rate": (arrays["V"]>arrays["geometric_bound"]+1e-8).float().mean().item(),
        "max_bound_excess": (arrays["V"]-arrays["geometric_bound"]).max().item(),
        "actual_domain_violation_rate": arrays["actual_domain_violation"].mean().item(),
        "reference_domain_violation_rate": arrays["reference_domain_violation"].mean().item(),
        "tube_violation_rate": arrays["tube_violation"].mean().item(),
        "actual_domain_violation_count": int(arrays["actual_domain_violation"].sum().item()),
        "reference_domain_violation_count": int(arrays["reference_domain_violation"].sum().item()),
        "tube_violation_count": int(arrays["tube_violation"].sum().item()),
        "geometric_bound_violation_count": int((arrays["V"]>arrays["geometric_bound"]+1e-8).sum().item()),
        "planner_domain_failure_rate": env.planner_failures/env.planner_instances,
        "wall_seconds": time.perf_counter()-started,
        "physical_steps": steps*environments,
        "planner_instances": env.planner_instances,
        "planner_counts": env.planner.objective_counts(),
        "planner_low_level_model_steps": env.planner.objective_counts()["candidate_macro_steps"]*config.command_steps,
        "planner_integration": "unmodified upstream CEMPlanner through response_codesign bridge; not full SPlaTES",
        "metric": env.metric_info,
    }
    rollout = Rollout(**{key: torch.stack(value) for key, value in fields.items()}) if training else None
    return rollout, metrics, arrays, noise


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ["src/response_ppo/ppo_backend.py", "src/response_ppo/env.py",
             "src/response_ppo/pilot.py", "src/response_ppo/realization.py",
             "src/response_codesign/splates_bridge.py", "src/response_codesign/dynamics.py"]
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def average_metrics(metrics):
    keys = [key for key, value in metrics[0].items() if isinstance(value, (float, int))]
    return {key: float(np.mean([m[key] for m in metrics])) for key in keys}


def run_seed(seed, output, updates=2, test_batches=1, environments=16, train_steps=256,
             short_steps=256, long_steps=1024, device="cpu"):
    output.mkdir(parents=True, exist_ok=True)
    ppo_config = PPOConfig(obs_dim=ResponseServo.observation_dimension, device=device)
    env_config = EnvironmentConfig(device=device)
    results = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        backend = PPOBackend(ppo_config)
        design_mean = initial_design(device)
        initial_weights_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in backend.actor.parameters())).hexdigest()
        history = []
        for update in range(updates):
            rollout, metrics, arrays, noise = collect(
                backend, design_mean, arm, seed*10000+update,
                train_steps, environments, True, env_config,
            )
            ppo_started = time.perf_counter()
            ppo_metrics = backend.update(rollout)
            ppo_metrics["wall_seconds"] = time.perf_counter()-ppo_started
            episode_cost = arrays["total"].mean(0)
            # Leave-one-environment-out baseline is independent of that
            # episode's sampled design. No pathwise CEM gradient is claimed.
            baseline = (episode_cost.sum()-episode_cost)/(environments-1)
            score_gradient = ((episode_cost-baseline)[:, None]*noise/.2).mean(0)
            proposed = -.25*score_gradient
            proposed *= min(1., .05/(proposed.norm().item()+1e-12))
            before = design_mean.clone()
            if arm.startswith("joint"):
                design_mean = design_mean + proposed
            history.append({"update": update, "rollout": metrics, "ppo": ppo_metrics,
                            "design_before": before.tolist(), "design_after": design_mean.tolist(),
                            "design_score_gradient": score_gradient.tolist(),
                            "applied_design_step_l2": (design_mean-before).norm().item()})
            print(json.dumps({"seed": seed, "arm": arm, "update": update,
                              "task": metrics["task_cost"], "empty": metrics["empty_support_rate"]}), flush=True)
        evaluation = {}
        for label, steps in (("short", short_steps), ("long", long_steps)):
            batches = []
            for batch in range(test_batches):
                _, metrics, arrays, _ = collect(backend, design_mean, arm,
                    seed*10000+5000+batch, steps, environments, False, env_config)
                batches.append(metrics)
                if batch == 0:
                    np.savez_compressed(output/f"{arm}-{label}-trace.npz", **{k:v.cpu().numpy() for k,v in arrays.items()})
            evaluation[label] = {"mean": average_metrics(batches), "batches": batches}
        actor_state = {k:v.detach().cpu() for k,v in backend.actor.state_dict().items()}
        critic_state = {k:v.detach().cpu() for k,v in backend.critic.state_dict().items()}
        torch.save({"actor": actor_state, "critic": critic_state, "design_mean": design_mean.cpu()}, output/f"{arm}.pt")
        results[arm] = {"history": history, "evaluation": evaluation, "design_final": design_mean.tolist(),
                        "initial_actor_sha256": initial_weights_hash, "provenance": backend.provenance(),
                        "budget": {"training_physical_steps": updates*environments*train_steps,
                                   "evaluation_physical_steps": test_batches*environments*(short_steps+long_steps),
                                   "optimizer_steps": sum(row["ppo"]["optimizer_steps"] for row in history)}}
        (output/f"{arm}.json").write_text(json.dumps(results[arm], indent=2)+"\n", encoding="utf-8")
    result = {"seed": seed, "environment": asdict(env_config), "ppo": asdict(ppo_config),
              "source_hashes": source_hashes(), "design_update": {"std": .2, "learning_rate": .25, "max_step_norm": .05,
              "fixed_arm": "same Gaussian randomization, distribution mean frozen", "score_cost": "same total cost as PPO"},
              "arms": results}
    (output/"result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[901])
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--test-batches", type=int, default=1)
    parser.add_argument("--environments", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=256)
    parser.add_argument("--short-steps", type=int, default=256)
    parser.add_argument("--long-steps", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"seeds": args.seeds, "updates": args.updates, "test_batches": args.test_batches,
                "environments": args.environments, "train_steps": args.train_steps,
                "short_steps": args.short_steps, "long_steps": args.long_steps,
                "source_hashes_before_run": source_hashes(), "status": "running"}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    for seed in args.seeds:
        run_seed(seed, args.output/f"seed-{seed}", args.updates, args.test_batches,
                 args.environments, args.train_steps, args.short_steps, args.long_steps, args.device)
    manifest["status"] = "completed"
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
