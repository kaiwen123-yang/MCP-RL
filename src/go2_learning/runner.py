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
             ROOT / "configs/go2_long_horizon.json",
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
    pathwise = "pathwise" in config["arms"]
    return {
        "status": "PLANNED_NOT_RUN",
        "method": ("whole-plan feedback learning with factual response differences"
                   if pathwise else "continuation-adjoint control variate PPO"),
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
        "learning_signal": (
            "real full-task PPO plus shared MPC task proxy across the whole selected plan; "
            "f/g/old actor/V remain frozen in the auxiliary objective"
            if pathwise else
            "same real time/failure return and full task information; adjoint changes only gradient estimator"),
        "auxiliary": config.get("pathwise", {"cv_alpha": config["ppo"]["cv_alpha"]}),
        "fairness": "report performance versus all physics steps and end-to-end wall time, including auxiliary model work",
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
    runtime = config.get("runtime", {})
    torch.set_num_threads(int(runtime.get("torch_threads", 1)))
    policy_root = Path(assets) / "rl_sar/policy/go2"
    base = yaml.safe_load((policy_root / "base.yaml").read_text(encoding="utf-8"))["go2"]
    specific = yaml.safe_load((policy_root / "robot_lab/config.yaml").read_text(encoding="utf-8"))["go2/robot_lab"]
    actor = TaskActor(policy_root / "robot_lab" / specific["model_name"],
                      {**base, **specific}, config["initial_log_std"],
                      context_dim=config.get("context_dim", 19))
    dynamics = ActionDynamics(dt=config["policy_dt"], hidden=config["model_hidden"])
    response = CommandResponse(dt=config["macro_dt"], hidden=config["model_hidden"],
                               context_dim=config.get("context_dim", 19))
    value = ValueNet(hidden=config["value_hidden"], context_dim=config.get("context_dim", 19))
    device = torch.device(runtime.get("device", "cpu"))
    for module in (actor, dynamics, response, value):
        module.to(device)
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


def optimizer_state_summary(system):
    """Fingerprint every saved optimizer value, including Adam moments/steps.

    Parameter identifiers are the deterministic indices in state_dict(), not
    Python object addresses. Tensor bytes include dtype and shape, so the
    fingerprint compares complete initial state without publishing the moments.
    """
    import torch

    updater = system["updater"]
    optimizers = {
        "actor_optimizer": updater.actor_optimizer,
        "value_optimizer": updater.value_optimizer,
        "f_optimizer": system["f_optimizer"],
        "g_optimizer": system["g_optimizer"],
    }
    result = {}
    for name, optimizer in optimizers.items():
        saved = optimizer.state_dict()
        digest = hashlib.sha256()
        tensor_count = 0

        def visit(value):
            nonlocal tensor_count
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(b"tensor")
                digest.update(str(tensor.dtype).encode())
                digest.update(json.dumps(list(tensor.shape)).encode())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
                tensor_count += 1
            elif isinstance(value, dict):
                digest.update(b"dict")
                for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                    visit(key)
                    visit(value[key])
            elif isinstance(value, (list, tuple)):
                digest.update(type(value).__name__.encode())
                digest.update(str(len(value)).encode())
                for item in value:
                    visit(item)
            else:
                digest.update(type(value).__name__.encode())
                digest.update(json.dumps(value, allow_nan=False, separators=(",", ":")).encode())
            digest.update(b"\0")

        visit(saved)
        steps = [
            float(parameter_state["step"].item())
            if isinstance(parameter_state["step"], torch.Tensor)
            else float(parameter_state["step"])
            for parameter_state in saved["state"].values() if "step" in parameter_state
        ]
        histogram = {}
        for step in steps:
            key = str(int(step)) if step.is_integer() else str(step)
            histogram[key] = histogram.get(key, 0) + 1
        result[name] = {
            "sha256": digest.hexdigest(),
            "parameter_groups": len(saved["param_groups"]),
            "parameters_in_groups": sum(len(group["params"]) for group in saved["param_groups"]),
            "initialized_parameter_states": len(saved["state"]),
            "state_tensor_count": tensor_count,
            "step_min": min(steps) if steps else None,
            "step_max": max(steps) if steps else None,
            "step_histogram": histogram,
        }
    return result


def load_state(system, state):
    # Optimizer.load_state_dict may retain same-device tensors from its input.
    # Each branch owns its entire copy: neither the immutable template nor any
    # other loaded branch can acquire this branch's Adam moments or step count.
    state = copy.deepcopy(state)
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
    torch.save({"batch": collection["batch"], "macro": collection["macro"],
                "plans": collection["plans"]},
               Path(path) / "transitions.pt")
    write_json(Path(path) / "summary.json", {k: v for k, v in collection.items()
                                           if k not in ("batch", "macro", "plans")})


def collect_phase(system, config, assets, path, seed, mode):
    from go2_frozen.planner import MPCConfig
    from .collection import collect
    stochastic = mode in ("warmup", "behavior")
    steps = (config["common_warmup_steps_per_case"] if mode == "warmup"
             else config["stochastic_steps_per_case_per_round"] if stochastic else None)
    episodes = (config["calibration_complete_episodes_per_case"] if mode == "calibration"
                else config["evaluation_complete_episodes_per_case"])
    # MuJoCo and small per-step feedback run on CPUs; GPU parameters are copied
    # once per phase, never shuttled across devices at every physical step.
    from .collection import collect_cases
    from .continuation import resolve_cases
    actors = [copy.deepcopy(system[name]).cpu().eval()
              for name in ("actor", "response", "value")]
    runtime = config.get("runtime", {})
    collection = collect_cases(*actors, assets, path,
                               case_names=(config.get("evaluation_cases", config["cases"])
                                           if mode == "evaluation" else config["cases"]), seed=seed,
                               stochastic=stochastic, steps_per_case=steps,
                               speed=config["speed_band"], episodes_per_case=episodes,
                               planner_config=MPCConfig(**config["planner"]),
                               workers=int(runtime.get("collection_workers", 1)),
                               threads=int(runtime.get("inference_threads", 1)),
                               case_specs=resolve_cases(config),
                               controller_mode=runtime.get("controller_mode", "mpc"))
    save_collection(path, collection)
    return collection


def fit_phase(system, config, behavior, calibration, seed):
    from .collection import concatenate_batches
    from .learning import fit_facts
    micro = concatenate_batches(behavior["batch"], calibration["batch"])
    return fit_facts(system["dynamics"], system["response"], system["f_optimizer"],
                     system["g_optimizer"], micro, calibration["macro"],
                     config["model_fit_steps"], config["model_fit_batch_size"], seed)


def learn_round(system, config, assets, phase_root, seed, round_index, arm):
    """One closed loop: actual B -> feedback update -> actual C -> facts -> task."""
    import torch
    behavior = collect_phase(system, config, assets, phase_root / "behavior",
                             seed + 100 * round_index + 10, "behavior")
    batch = behavior["batch"]
    derivative_stats = {"adjoint_forward_steps": 0,
                        "adjoint_reverse_steps": 0, "adjoint_seconds": 0.0}
    if arm == "adjoint":
        d, derivative_stats = directions(system, batch, config)
    else:
        d = torch.zeros_like(batch["raw_action"])
    auxiliary = None
    if arm == "pathwise":
        from .pathwise import PlannedResponseLoss
        auxiliary = PlannedResponseLoss(system, behavior["plans"], config)
    # All auxiliary models and the tail value are frozen at the B batch's
    # policy version. V updates below cannot change the pathwise teacher.
    update = system["updater"].update(batch, d, arm, actor_auxiliary=auxiliary)
    if auxiliary is not None:
        derivative_stats = auxiliary.statistics()
        del auxiliary
    calibration = collect_phase(system, config, assets, phase_root / "calibration",
                                seed + 100 * round_index + 11, "calibration")
    fact_update = fit_phase(system, config, behavior, calibration,
                            seed + 100 * round_index + 12)
    evaluation = collect_phase(system, config, assets, phase_root / "evaluation",
                               seed + 20000, "evaluation")
    torch.save(state_dict(system), phase_root / "checkpoint.pt")
    record = {"round": round_index + 1, "arm": arm, "ppo": update, "facts": fact_update,
              "derivatives": derivative_stats, "parameter_hashes": hashes(system),
              "behavior": behavior["stats"], "calibration": calibration["stats"],
              "evaluation": evaluation["stats"],
              "evaluation_episodes": evaluation["episodes"]}
    write_json(phase_root / "record.json", record)
    return record


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
        common_optimizer_state = optimizer_state_summary(common)
        torch.save(common_state, prefix / "initialization.pt")
        write_json(prefix / "fit.json", {"value": common_value_fit, "facts": common_fact_fit})
        write_json(prefix / "initialization_optimizer_state.json", common_optimizer_state)
        for arm in config["arms"]:
            system = make_system(config, assets, seed)
            load_state(system, common_state)
            arm_root = root / f"seed{seed}" / arm
            arm_root.mkdir(parents=True, exist_ok=False)
            write_json(arm_root / "initialization_hashes.json", hashes(system))
            loaded_optimizer_state = optimizer_state_summary(system)
            optimizer_matches = loaded_optimizer_state == common_optimizer_state
            write_json(arm_root / "initialization_optimizer_state.json", {
                "matches_common": optimizer_matches,
                "optimizers": loaded_optimizer_state,
            })
            if not optimizer_matches:
                raise RuntimeError("Loaded branch optimizer state differs from the common initialization")
            started, records, capped = perf_counter(), [], False
            collect_phase(system, config, assets, arm_root / "evaluation_round0",
                          seed + 20000, "evaluation")
            for round_index in range(config["rounds"]):
                if perf_counter() - started >= config["wall_seconds_per_arm_seed"]:
                    capped = True
                    break
                phase_root = arm_root / f"round{round_index + 1}"
                record = learn_round(system, config, assets, phase_root, seed, round_index, arm)
                record["wall_seconds_including_initial_evaluation"] = perf_counter() - started
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


def resume(config, assets, output, checkpoint, seed, *, update=False, arm="pathwise"):
    """Evaluate a saved policy, or execute exactly one existing-budget iteration."""
    import torch
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    system = make_system(config, assets, seed)
    load_state(system, torch.load(checkpoint, map_location="cpu", weights_only=True))
    write_json(root / "config.json", config)
    write_json(root / "source.json", identity())
    started = perf_counter()
    if update:
        record = learn_round(system, config, assets, root / "round1", seed, 0, arm)
        write_json(root / "result.json", {
            "status": "COMPLETED_SINGLE_ITERATION", "arm": arm, "seed": seed,
            "initial_checkpoint": str(checkpoint), "wall_seconds": perf_counter() - started,
            "record": record,
        })
    else:
        before = hashes(system)
        collected = collect_phase(system, config, assets, root / "evaluation",
                                  seed + 20000, "evaluation")
        write_json(root / "result.json", {
            "status": "COMPLETED_EVALUATION", "seed": seed,
            "initial_checkpoint": str(checkpoint), "episodes": collected["episodes"],
            "stats": collected["stats"], "parameters_unchanged": before == hashes(system),
            "wall_seconds": perf_counter() - started,
        })


def compare_one_round(config, assets, output, checkpoint, seed):
    """One matched ordinary/pathwise iteration, using the existing sample budget."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    print("Evaluating the shared starting policy on all complete tasks", flush=True)
    resume(config, assets, root / "initial", checkpoint, seed)
    for arm in ("ordinary", "pathwise"):
        print(f"Starting {arm}: actual B, feedback update, actual C, facts, evaluation", flush=True)
        resume(config, assets, root / arm, checkpoint, seed, update=True, arm=arm)
        result = json.loads((root / arm / "result.json").read_text())
        episodes = result["record"]["evaluation_episodes"]
        print(json.dumps({"arm": arm, "complete_tasks": [
            {k: ep[k] for k in ("case", "success", "completion_seconds", "reason")}
            for ep in episodes]}, ensure_ascii=False), flush=True)
    write_json(root / "comparison.json", {
        "status": "COMPLETED", "seed": seed, "rounds_per_arm": 1,
        "arms": ["ordinary", "pathwise"], "cases": config["cases"],
        "stochastic_policy_steps_per_arm": len(config["cases"]) * config["stochastic_steps_per_case_per_round"],
        "wall_seconds": perf_counter() - started,
        "initial": json.loads((root / "initial/result.json").read_text()),
        **{arm: json.loads((root / arm / "result.json").read_text())
           for arm in ("ordinary", "pathwise")},
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
    parser.add_argument("action", choices=("plan", "verify", "train", "evaluate", "iterate", "compare"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/go2_long_horizon.json")
    parser.add_argument("--assets", type=Path, default=ROOT / "work/go2-public-assets")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        help="evaluate/iterate: existing complete actor/f/g/V checkpoint")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel CPU complete-task cases")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--inference-threads", type=int, default=1)
    parser.add_argument("--arm", choices=("ordinary", "adjoint", "pathwise"), default="pathwise",
                        help="iterate only: update mechanism")
    parser.add_argument("--with-physics", action="store_true",
                        help="verify only: one no-update original nominal mission")
    args = parser.parse_args()
    config = read_config(args.config)
    config["runtime"] = {"device": args.device, "collection_workers": args.workers,
                         "torch_threads": args.torch_threads,
                         "inference_threads": args.inference_threads}
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
        elif args.action in ("evaluate", "iterate", "compare"):
            if args.checkpoint is None:
                parser.error("evaluate/iterate/compare requires --checkpoint")
            if args.action == "compare":
                compare_one_round(config, args.assets, args.output, args.checkpoint, args.seed)
            else:
                resume(config, args.assets, args.output, args.checkpoint, args.seed,
                       update=args.action == "iterate", arm=args.arm)
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
