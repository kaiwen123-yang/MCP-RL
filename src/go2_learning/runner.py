"""Plan, no-update integration, and explicitly invoked later training.

No training is performed on import, by plan, or by verify. Raw trajectories and
derived policy checkpoints stay in work/. The original scene and success rules
are imported unchanged from go2_frozen.task.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["schema"] != 1 or config["ppo"]["gamma"] != 1.0:
        raise ValueError("This protocol uses schema 1 and undiscounted mission return")
    return config


def serial(value):
    if hasattr(value, "detach"):
        return serial(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return serial(value.tolist())
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(data), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def identity():
    paths = [ROOT / "configs/go2_adjoint_learning.json",
             ROOT / "scripts/run_go2_learning.py",
             *sorted((ROOT / "src/go2_learning").glob("*.py")),
             ROOT / "src/go2_frozen/task.py",
             ROOT / "src/go2_frozen/simulator.py",
             ROOT / "src/go2_frozen/fair_control.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in paths}


def plan(config):
    count_cases = len(config["cases"])
    # Complete deterministic calibration/evaluation may stop before 35 seconds.
    max_episode_steps = round(35.0 / config["policy_dt"])
    common_stochastic = count_cases * config["common_warmup_steps_per_case"]
    calibration_cap = count_cases * max_episode_steps * config["calibration_complete_episodes_per_case"]
    eval_cap = count_cases * max_episode_steps * config["evaluation_complete_episodes_per_case"]
    per_round_stochastic = count_cases * config["stochastic_steps_per_case_per_round"]
    per_arm_cap = (config["rounds"] * (per_round_stochastic + calibration_cap + eval_cap)
                   + eval_cap)
    return {
        "status": "PLANNED_NOT_RUN", "method": "continuation-adjoint control variate PPO",
        "task_source": config["frozen_task_source_commit"],
        "arms": config["arms"], "seeds": config["seeds"],
        "shared_once_per_seed": {
            "actor_training_updates": 0,
            "stochastic_policy_steps": common_stochastic,
            "deterministic_calibration_steps_upper": calibration_cap,
            "learned_before_branching": ["action_dynamics", "command_response", "actual_return_value"],
            "copy_exactly": ["actor", "f", "g", "V", "all optimizer states", "update counters"],
        },
        "per_arm_seed": {
            "rounds": config["rounds"],
            "exact_stochastic_policy_steps": config["rounds"] * per_round_stochastic,
            "calibration_policy_steps_upper": config["rounds"] * calibration_cap,
            "evaluation_policy_steps_upper": (config["rounds"] + 1) * eval_cap,
            "total_policy_steps_upper_excluding_stand_warmups": per_arm_cap,
            "wall_seconds": config["wall_seconds_per_arm_seed"],
            "wall_check": "before each round; one round may overrun; initial evaluation included",
        },
        "all_policy_steps_upper_excluding_stand_warmups": len(config["seeds"]) * (
            common_stochastic + calibration_cap + len(config["arms"]) * per_arm_cap),
        "physical_budget_note": "One policy step is 20 one-ms physics steps; each episode has an additional recorded 2 s public stand warmup. Actual calibration/evaluation lengths differ and must be reported.",
        "comparison": "Equal stochastic sample budgets and deterministic episode caps; NOT automatically equal wall/FLOP/total executed-transition counts.",
        "learning_signal": "same real time/failure return and same full task information; adjoint changes only gradient estimator",
        "cv_alpha": config["ppo"]["cv_alpha"],
        "fairness": "report performance versus all physics steps and end-to-end wall time; extra adjoint work is charged",
        "not_established": ["learning improvement", "variance reduction", "novel algorithm",
                            "held-out task composition generalization", "real-time deployment"],
    }


def make_system(config, assets, seed):
    import numpy as np
    import torch
    import yaml
    from .learning import PPOSettings, PPOUpdater
    from .models import ActionDynamics, CommandResponse, ValueNet
    from .policy import TaskActor

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    policy_root = Path(assets) / "rl_sar/policy/go2"
    base = yaml.safe_load((policy_root / "base.yaml").read_text(encoding="utf-8"))["go2"]
    specific = yaml.safe_load((policy_root / "robot_lab/config.yaml").read_text(encoding="utf-8"))["go2/robot_lab"]
    actor = TaskActor(policy_root / "robot_lab" / specific["model_name"],
                      {**base, **specific}, config["initial_log_std"])
    dynamics = ActionDynamics(dt=config["policy_dt"], hidden=config["model_hidden"])
    response = CommandResponse(dt=config["macro_dt"], hidden=config["model_hidden"])
    value = ValueNet(hidden=config["value_hidden"])
    updater = PPOUpdater(actor, value, PPOSettings(**config["ppo"]), seed + 71)
    return {"actor": actor, "dynamics": dynamics, "response": response, "value": value,
            "updater": updater,
            "f_optimizer": torch.optim.Adam(dynamics.parameters(), lr=config["model_learning_rate"]),
            "g_optimizer": torch.optim.Adam(response.parameters(), lr=config["model_learning_rate"])}


def hashes(system):
    result = {}
    for name in ("actor", "dynamics", "response", "value"):
        digest = hashlib.sha256()
        for key, value in sorted(system[name].state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def state_dict(system):
    updater = system["updater"]
    return copy.deepcopy({
        **{name: system[name].state_dict() for name in ("actor", "dynamics", "response", "value")},
        "actor_optimizer": updater.actor_optimizer.state_dict(),
        "value_optimizer": updater.value_optimizer.state_dict(),
        "f_optimizer": system["f_optimizer"].state_dict(),
        "g_optimizer": system["g_optimizer"].state_dict(),
        "actor_updates": updater.actor_updates, "value_updates": updater.value_updates,
    })


def load_state(system, state):
    for name in ("actor", "dynamics", "response", "value"):
        system[name].load_state_dict(state[name])
    updater = system["updater"]
    updater.actor_optimizer.load_state_dict(state["actor_optimizer"])
    updater.value_optimizer.load_state_dict(state["value_optimizer"])
    system["f_optimizer"].load_state_dict(state["f_optimizer"])
    system["g_optimizer"].load_state_dict(state["g_optimizer"])
    updater.actor_updates, updater.value_updates = state["actor_updates"], state["value_updates"]


def directions(system, batch, config):
    import torch
    from go2_frozen.task import make_cases
    from .learning import continuation_direction
    result = torch.zeros_like(batch["raw_action"])
    counters = {"adjoint_forward_steps": 0, "adjoint_reverse_steps": 0,
                "adjoint_seconds": 0.0}
    cases = {s.name: s for s in make_cases()}
    for case_index, name in enumerate(config["cases"]):
        for horizon in range(1, 11):
            indices = torch.where((batch["case_index"] == case_index) &
                                  (batch["horizon"] == horizon))[0]
            for index in indices.split(config["adjoint_batch_size"]):
                if not len(index):
                    continue
                d, stats = continuation_direction(
                    system["actor"], system["dynamics"], system["value"],
                    batch["state"][index], batch["memory"][index],
                    batch["command"][index], batch["next_planned_command"][index],
                    cases[name], horizon, config["ppo"]["gamma"])
                result[index] = d
                for key in counters:
                    counters[key] += stats[key]
    return result, counters


def save_collection(path, collection):
    import torch
    torch.save({"batch": collection["batch"], "macro": collection["macro"]},
               Path(path) / "transitions.pt")
    write_json(Path(path) / "summary.json", {k: v for k, v in collection.items()
                                           if k not in ("batch", "macro")})


def collect_phase(system, config, assets, path, seed, mode):
    from go2_frozen.planner import MPCConfig
    from .collection import collect
    stochastic = mode in ("warmup", "behavior")
    steps = (config["common_warmup_steps_per_case"] if mode == "warmup"
             else config["stochastic_steps_per_case_per_round"] if stochastic else None)
    episodes = (config["calibration_complete_episodes_per_case"] if mode == "calibration"
                else config["evaluation_complete_episodes_per_case"])
    collection = collect(system["actor"], system["response"], system["value"],
                         assets, path, case_names=config["cases"], seed=seed,
                         stochastic=stochastic, steps_per_case=steps,
                         speed=config["speed_band"], episodes_per_case=episodes,
                         planner_config=MPCConfig(**config["planner"]))
    save_collection(path, collection)
    return collection


def fit_phase(system, config, behavior, calibration, seed):
    from .collection import concatenate_batches
    from .learning import fit_facts
    micro = concatenate_batches(behavior["batch"], calibration["batch"])
    return fit_facts(system["dynamics"], system["response"], system["f_optimizer"],
                     system["g_optimizer"], micro, calibration["macro"],
                     config["model_fit_steps"], config["model_fit_batch_size"], seed)


def train(config, assets, output):
    """Implemented for a separately authorized run. Not called in this delivery."""
    import torch
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan(config))
    write_json(root / "source.json", identity())
    write_json(root / "config.json", config)
    for seed in config["seeds"]:
        common = make_system(config, assets, seed)
        prefix = root / f"seed{seed}" / "common"
        behavior = collect_phase(common, config, assets, prefix / "behavior", seed, "warmup")
        # This fits V only. Common warmup NEVER changes public feedback.
        zero = torch.zeros_like(behavior["batch"]["raw_action"])
        common_value_fit = common["updater"].update(behavior["batch"], zero, "frozen")
        calibration = collect_phase(common, config, assets, prefix / "calibration", seed + 1,
                                    "calibration")
        common_fact_fit = fit_phase(common, config, behavior, calibration, seed + 2)
        common_state = state_dict(common)
        torch.save(common_state, prefix / "initialization.pt")
        write_json(prefix / "fit.json", {"value": common_value_fit, "facts": common_fact_fit})
        for arm in config["arms"]:
            system = make_system(config, assets, seed)
            load_state(system, common_state)
            arm_root = root / f"seed{seed}" / arm
            arm_root.mkdir(parents=True, exist_ok=False)
            write_json(arm_root / "initialization_hashes.json", hashes(system))
            started, records, capped = perf_counter(), [], False
            collect_phase(system, config, assets, arm_root / "evaluation_round0",
                          seed + 20000, "evaluation")
            for round_index in range(config["rounds"]):
                if perf_counter() - started >= config["wall_seconds_per_arm_seed"]:
                    capped = True
                    break
                phase_root = arm_root / f"round{round_index + 1}"
                # f/g/V and actor stay unchanged through collection and direction construction.
                behavior = collect_phase(system, config, assets, phase_root / "behavior",
                                         seed + 100 * round_index + 10, "behavior")
                batch = behavior["batch"]
                if arm == "adjoint":
                    d, derivative_stats = directions(system, batch, config)
                else:
                    d = torch.zeros_like(batch["raw_action"])
                    derivative_stats = {"adjoint_forward_steps": 0,
                                        "adjoint_reverse_steps": 0, "adjoint_seconds": 0.0}
                # No current-batch fitting precedes the pre-action CV calculation.
                update = system["updater"].update(batch, d, arm)
                calibration = collect_phase(system, config, assets, phase_root / "calibration",
                                            seed + 100 * round_index + 11, "calibration")
                fact_update = fit_phase(system, config, behavior, calibration,
                                        seed + 100 * round_index + 12)
                evaluation = collect_phase(system, config, assets, phase_root / "evaluation",
                                           seed + 20000, "evaluation")
                torch.save(state_dict(system), phase_root / "checkpoint.pt")
                record = {"round": round_index + 1, "ppo": update, "facts": fact_update,
                          "derivatives": derivative_stats, "parameter_hashes": hashes(system),
                          "behavior": behavior["stats"], "calibration": calibration["stats"],
                          "evaluation": evaluation["stats"],
                          "evaluation_episodes": evaluation["episodes"],
                          "wall_seconds_including_initial_evaluation": perf_counter() - started}
                records.append(record)
                write_json(phase_root / "record.json", record)
            write_json(arm_root / "result.json", {
                "status": "BUDGET_CAPPED" if capped else "COMPLETED",
                "arm": arm, "seed": seed, "records": records,
                "common_initialization": common["actor"].config["model_name"],
                "actor_updates": system["updater"].actor_updates,
                "value_updates": system["updater"].value_updates,
                "wall_seconds": perf_counter() - started,
            })


def verify(config, assets, output, run_physics=False):
    """No optimizer step, no checkpoint write; optionally one complete mission."""
    import torch
    from go2_frozen.task import make_cases
    from .learning import continuation_direction
    from .state import task_context
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    system = make_system(config, assets, config["seeds"][0])
    before = hashes(system)
    state = torch.zeros(2, 48)
    state[:, 2] = .34
    state[:, 12:24] = system["actor"].default_q
    memory = torch.tensor([[0., 0., -1.], [1., 4., -1.]])
    command = torch.tensor([[.4, 0., 0.], [.2, 0., .4]])
    spec = make_cases()[0]
    context = task_context(state, memory, spec)
    with torch.no_grad():
        draw = system["actor"].sample(state, context, command,
                                      generator=torch.Generator().manual_seed(99))
        obs = system["actor"].public_observation(state, command)
        source = torch.jit.load(str(Path(assets) / "rl_sar/policy/go2/robot_lab/policy.pt")).eval()
        mean_error = float((source(obs) - draw["mean"]).abs().max())
        logp_error = float((system["actor"].log_prob(state, context, command, draw["raw"])
                           - draw["log_prob"]).abs().max())
    d, direction_stats = continuation_direction(
        system["actor"], system["dynamics"], system["value"],
        state, memory, command, command, spec, 10)
    result = {"status": "NO_UPDATE_INTEGRATION", "source": identity(),
              "original_actor_mean_max_error": mean_error,
              "raw_sample_log_prob_max_error": logp_error,
              "untrained_direction_max_abs": float(d.abs().max()),
              "untrained_direction_interpretation": "Zero initialized f/V; no learned physical action direction or efficacy is claimed.",
              "parameters": {name: sum(p.numel() for p in system[name].parameters())
                             for name in ("actor", "dynamics", "response", "value")},
              "direction_work": direction_stats, "optimizer_updates": 0,
              "new_physics_steps": 0, "scenario_modified": False}
    if run_physics:
        check_config = copy.deepcopy(config)
        check_config["cases"] = ["nominal"]
        # Keep the original fast band, scene, 4 s / 256 x 3 planner budget.
        collection = collect_phase(system, check_config, assets, root / "nominal_mean",
                                   config["seeds"][0], "evaluation")
        batch = collection["batch"]
        # Inspect both objectives on the same actual samples; deterministic data
        # are NOT valid stochastic PPO training, so no update is performed.
        selection = torch.arange(0, min(160, len(batch["state"])), 10)
        selected = {k: v[selection] for k, v in batch.items()}
        dirs, derivative_stats = directions(system, selected, check_config)
        returns, advantage = system["updater"].prepare(batch)
        losses = {}
        for arm in ("ordinary", "adjoint"):
            actor_loss, value_loss, metrics = system["updater"].losses(
                selected, returns[selection], advantage[selection], dirs, arm)
            losses[arm] = {"actor_loss": float(actor_loss.detach()),
                           "value_loss": float(value_loss.detach())}
        result.update({"new_physics_steps": collection["stats"]["warmup_and_execution_physics_steps"],
                       "complete_mission": collection["episodes"], "execution_stats": collection["stats"],
                       "same_batch_loss_evaluation": losses, "actual_state_adjoint_work": derivative_stats,
                       "loss_check_boundary": "Deterministic mean execution checks the wiring only, not an on-policy PPO update or candidate benefit."})
    result["parameter_hashes_before"] = before
    result["parameter_hashes_after"] = hashes(system)
    result["parameters_unchanged"] = before == hashes(system)
    if not result["parameters_unchanged"] or mean_error > 2e-5 or logp_error > 2e-5:
        raise AssertionError("No-update integration violated actor identity or changed weights")
    write_json(root / "verification.json", result)
    print(json.dumps({k: result[k] for k in ("status", "parameters_unchanged",
                     "optimizer_updates", "new_physics_steps")}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "verify", "train"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/go2_adjoint_learning.json")
    parser.add_argument("--assets", type=Path, default=ROOT / "work/go2-public-assets")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-physics", action="store_true",
                        help="verify only: one no-update original nominal mission")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.action == "plan":
        result = plan(config)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.output is None or not args.output.resolve().is_relative_to((ROOT / "work").resolve()):
        parser.error("Raw data and checkpoints require a fresh --output under this worktree's work/")
    output_preexisted = args.output.exists()
    try:
        if args.action == "verify":
            verify(config, args.assets, args.output, args.with_physics)
        else:
            if args.with_physics:
                parser.error("--with-physics belongs only to no-update verify")
            train(config, args.assets, args.output)
    except Exception as exc:
        # Preserve failure as failure, with partial trajectories left in place.
        # In particular, no complete calibration transition does not trigger
        # another rollout or silently keep claiming g is newly calibrated.
        if args.output.exists() and not output_preexisted:
            write_json(args.output / "execution_failure.json",
                       {"status": "FAILED", "action": args.action,
                        "exception_type": type(exc).__name__, "message": str(exc),
                        "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
