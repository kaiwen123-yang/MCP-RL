"""Plan, fixed-input check, later server training, and frozen checkpoint evaluation.

Only the explicit train subcommand enters parameter-update code.  The plan
path uses JSON/arithmetic only and performs no model construction, collection,
optimizer operation, checkpoint write, or server connection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time


REPOSITORY = Path(__file__).resolve().parents[2]
WORK_ROOT = REPOSITORY / "work"
STAGES = ("A_collect", "model_proposal", "B_collect", "ppo_update",
          "C_collect", "model_calibration")


def read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected task_response schema_version 1")
    e, x, p = config["environment"], config["experiment"], config["ppo"]
    if e["horizon"] % e["macro_steps"]:
        raise ValueError("Episode horizon must comprise complete macro intervals")
    if x["environments"] % 2 or x["evaluation_environments"] % 2:
        raise ValueError("Paired go/stop collections require an even environment count")
    if not x["evaluation_deterministic"] or p["gamma"] != 1.0:
        raise ValueError("This configured finite-horizon pilot uses deterministic evaluation and gamma=1")
    if (p["obs_dim"], p["action_dim"], p["action_bound"]) != (14, 2, e["input_bound"]):
        raise ValueError("PPO observation/action contract must match the physical task")
    if set(x["arms"]) - {"ordinary", "uniform", "weighted_id", "codesign", "frozen"}:
        raise ValueError("Unknown comparison arm")
    if x["warmup_arm"] != "ordinary" or tuple(x["stage_order"]) != STAGES:
        raise ValueError("Common warmup and the fresh A/B/C lifecycle are fixed in schema v1")
    if x["collection_policy_modes"] != {"A": "deterministic_mean", "B": "stochastic_ppo", "C": "deterministic_mean"}:
        raise ValueError("A/C model the deterministic deployment actor; B provides stochastic PPO data")
    return config


def evaluation_rounds(config: dict) -> list[int]:
    x = config["experiment"]
    return sorted({0, x["comparison_rounds"],
                   *range(x["evaluation_every_rounds"], x["comparison_rounds"] + 1,
                          x["evaluation_every_rounds"])})


def build_plan(config: dict) -> dict:
    """Exact configured counts, not a speed estimate or executed result."""
    e, x, p, l = (config[k] for k in ("environment", "experiment", "planner", "learning"))
    seeds, arms = len(x["training_seeds"]), len(x["arms"])
    horizon, macro, batch = e["horizon"], e["horizon"] // e["macro_steps"], x["environments"]
    warm_rounds = seeds * x["common_warmup_rounds"]
    compare_rounds = seeds * arms * x["comparison_rounds"]
    rounds = warm_rounds + compare_rounds
    training_batches = rounds * 3
    training_steps = training_batches * batch * horizon
    checkpoints = evaluation_rounds(config)
    eval_batches = seeds * arms * len(checkpoints) * len(x["evaluation_seeds"])
    eval_steps = eval_batches * x["evaluation_environments"] * horizon
    sampled_sequences = p["candidates"] * p["iterations"] + 1
    planner_train = training_batches * macro * batch * sampled_sequences * p["horizon"] * e["macro_steps"]
    planner_eval = eval_batches * macro * x["evaluation_environments"] * sampled_sequences * p["horizon"] * e["macro_steps"]
    fit_steps_per_round = l["model_steps"] + x["model_calibration_steps"]
    fit_forward = rounds * fit_steps_per_round * 2 * horizon * batch
    weighted_arms = sum(a in {"weighted_id", "codesign"} for a in x["arms"])
    weighted_batches = seeds * weighted_arms * x["comparison_rounds"] * 3
    metric_forward = weighted_batches * batch * e["macro_steps"] * macro * (macro + 1) // 2
    # Upstream Tianshou 0.5.1 PPO uses Batch.split(..., merge_last=True).
    minibatches = max(1, (batch * horizon) // config["ppo"]["minibatch_size"])
    policy_rounds = warm_rounds + seeds * sum(a != "frozen" for a in x["arms"]) * x["comparison_rounds"]
    ppo_steps = policy_rounds * config["ppo"]["epochs"] * minibatches
    return {
        "status": "PLANNED_NOT_RUN", "name": config["name"],
        "training_seeds": x["training_seeds"], "arms": x["arms"],
        "common_warmup": {"arm": "ordinary", "rounds_per_seed": x["common_warmup_rounds"],
                          "physical_steps_total": warm_rounds * 3 * batch * horizon,
                          "shared_once_per_seed": True,
                          "clone": ["actor", "critic", "response_model", "policy_optimizer",
                                    "model_optimizer", "policy_update_count", "PPO_rng_state"],
                          "comparison_planner_stream": "same fresh seed and call counter for every arm"},
        "comparison": {"rounds_per_arm_seed": x["comparison_rounds"],
                       "physical_steps_total": compare_rounds * 3 * batch * horizon,
                       "fresh_collections_each_round": ["A: deterministic current actor before model proposal",
                                                        "B: stochastic PPO actor under the proposed model",
                                                        "C: deterministic updated actor before model calibration"],
                       "collection_policy_modes": x["collection_policy_modes"],
                       "response_identity": "Predict deterministic mean-actor execution; do not equate it to the mean stochastic trajectory"},
        "training_physical_steps": training_steps,
        "evaluation": {"rounds": checkpoints, "seeds": x["evaluation_seeds"],
                       "environments": x["evaluation_environments"], "deterministic": True,
                       "batches": eval_batches, "physical_steps": eval_steps,
                       "metric_derivatives": 0, "response_shaping": False,
                       "real_task_metrics_separate_from_training_reward": True},
        "total_physical_steps_including_evaluation": training_steps + eval_steps,
        "configured_model_work": {
            "training_planner_model_steps": planner_train,
            "evaluation_planner_model_steps": planner_eval,
            "collection_target_model_steps": training_steps + eval_steps,
            "fit_forward_model_steps": fit_forward, "fit_reverse_model_steps": fit_forward,
            "metric_forward_model_steps": metric_forward, "metric_reverse_model_steps": metric_forward,
            "total_forward_model_steps": planner_train + planner_eval + training_steps + eval_steps + fit_forward + metric_forward,
            "ordinary_uniform_metric_derivative_steps": 0,
            "model_optimizer_steps": rounds * fit_steps_per_round,
            "policy_optimizer_steps": ppo_steps,
            "ppo_minibatches_per_epoch": minibatches,
            "counting_note": "Model microsteps are work counters, not equal-cost FLOPs or measured runtime; encoder/NN overhead also occurs"
        },
        "wall_limits": {"warmup_seconds_per_seed": x["common_warmup_wall_seconds"],
                        "seconds_per_arm_seed_including_evaluation": x["per_arm_wall_seconds"],
                        "stop": "first stage boundary after cap; actual overrun recorded",
                        "incomplete_warmup": "no comparison arm starts"},
        "not_verified_by_plan": ["training update branches", "task success", "algorithm advantage",
                                 "server throughput", "robot transfer", "NX real-time performance"],
        "training_authorized_by_execution": "Only an explicit future train command runs training",
    }


def _jsonable(value):
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _work_path(value: str | Path) -> Path:
    resolved = Path(value).resolve()
    if not resolved.is_relative_to(WORK_ROOT.resolve()):
        raise ValueError("Outputs with model weights and intermediate state must stay under repository work/")
    return resolved


def _source_identity() -> dict:
    sources = sorted((REPOSITORY / "src" / "task_response").glob("*.py"))
    sources += [REPOSITORY / "src" / "response_ppo" / "ppo_backend.py"]
    return {str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in sources}


def _runtime_configs(config, device):
    from .environment import Config
    from .learning import LearningConfig
    from .planner import PlannerConfig
    from response_ppo.ppo_backend import PPOConfig
    environment = Config(**config["environment"])
    learning = LearningConfig(**config["learning"], environment=environment)
    planner = PlannerConfig(**config["planner"], environment=environment)
    ppo = PPOConfig(**{**config["ppo"], "hidden": tuple(config["ppo"]["hidden"]), "device": device})
    return environment, learning, planner, ppo


def _make_pair(config, device, seed):
    import numpy as np
    import torch
    from .models import CompactResponse
    from response_ppo.ppo_backend import PPOBackend
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env, learning, planner, ppo = _runtime_configs(config, device)
    policy = PPOBackend(ppo)
    model = CompactResponse(env, hidden=config["model"]["hidden"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning.model_learning_rate)
    return policy, model, optimizer, learning, planner


def _pair_state(policy, model, optimizer):
    return copy.deepcopy({"actor": policy.actor.state_dict(), "critic": policy.critic.state_dict(),
                          "response_model": model.state_dict(), "policy_optimizer": policy.optimizer.state_dict(),
                          "model_optimizer": optimizer.state_dict(), "policy_update_count": policy.update_count})


def _load_pair(state, policy, model, optimizer):
    policy.actor.load_state_dict(state["actor"]); policy.critic.load_state_dict(state["critic"])
    model.load_state_dict(state["response_model"])
    # Optimizer.load_state_dict may retain same-device tensor references.
    # Each comparison branch must own its Adam moments, not mutate the
    # common warmup snapshot used to initialize later arms.
    policy.optimizer.load_state_dict(copy.deepcopy(state["policy_optimizer"]))
    optimizer.load_state_dict(copy.deepcopy(state["model_optimizer"]))
    policy.update_count = int(state["policy_update_count"])


def _rng_snapshot():
    import numpy as np
    import torch
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(snapshot):
    import numpy as np
    import torch
    random.setstate(snapshot[0]); np.random.set_state(snapshot[1]); torch.set_rng_state(snapshot[2])
    if snapshot[3] is not None:
        torch.cuda.set_rng_state_all(snapshot[3])


def _merge_work(total, increment):
    for key, value in increment.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value


class StageBudget:
    def __init__(self, seconds, output):
        self.seconds, self.output = float(seconds), output
        self.started = time.perf_counter()
        self.work, self.records = {}, []

    @property
    def elapsed(self):
        return time.perf_counter() - self.started

    @property
    def exhausted(self):
        return self.elapsed >= self.seconds

    def record(self, stage, round_number, started, work, details):
        _merge_work(self.work, work)
        record = {"stage": stage, "round": round_number, "wall_seconds": time.perf_counter() - started,
                  "elapsed_wall_seconds": self.elapsed, "overrun_seconds": max(0.0, self.elapsed-self.seconds),
                  "work": work, "details": details}
        self.records.append(record)
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "stages.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), allow_nan=False) + "\n")


def _collection_details(collected):
    return {"task_metrics": collected.metrics,
            "real_task_return_per_environment": collected.raw_task_reward.sum(0),
            "response_penalty_per_environment": collected.response_penalty.sum(0),
            "shaped_training_return_per_environment": collected.ppo.reward.sum(0)}


def _data_seed(seed, round_number, stage_index, warmup):
    return seed * 100000 + (10000000 if warmup else 20000000) + round_number * 10 + stage_index


def _round(policy, model, optimizer, planner, arm, learning, config,
           seed, number, budget, warmup):
    from dataclasses import replace
    from .collection import collect
    from .learning import update_model
    x = config["experiment"]
    phase, calibrated, batches = "START_PAIRED", True, {}
    for stage_index, stage_name in enumerate(STAGES):
        if budget.exhausted:
            return False, calibrated, phase
        started = time.perf_counter()
        if stage_name.endswith("_collect"):
            label = stage_name[0]
            collected = collect(policy, model, planner, arm, learning, x["environments"],
                                _data_seed(seed, number, {"A": 1, "B": 2, "C": 3}[label], warmup),
                                deterministic=label != "B")
            batches[label] = collected
            work, details = collected.work, _collection_details(collected)
            details["collection_policy_mode"] = x["collection_policy_modes"][label]
        elif stage_name == "model_proposal":
            details = update_model(model, optimizer, batches["A"].response, arm, learning)
            work = {key: value for key, value in details.items() if key.endswith("_steps")}
            calibrated, phase = False, "MODEL_PROPOSED_NOT_RECALIBRATED"
        elif stage_name == "ppo_update":
            if arm.actor_learning:
                details = policy.update(batches["B"].ppo)
                work = {"policy_optimizer_steps": int(details["optimizer_steps"]),
                        "ppo_batch_samples": int(details["samples"])}
            else:
                details, work = {"status": "FROZEN_ACTOR_DIAGNOSTIC"}, {"policy_optimizer_steps": 0}
            calibrated, phase = False, "ACTOR_BLOCK_FINISHED_NOT_RECALIBRATED"
        else:
            calibration = replace(learning, model_steps=x["model_calibration_steps"])
            details = update_model(model, optimizer, batches["C"].response, arm, calibration, calibration_only=True)
            work = {key: value for key, value in details.items() if key.endswith("_steps")}
            calibrated, phase = True, "PAIRED_AFTER_CALIBRATION"
        budget.record(stage_name, number, started, work, details)
    return True, calibrated, phase


def _evaluate_pair(policy, model, learning, planner_config, config, round_number, budget, planner_name="cem"):
    import torch
    from .collection import collect
    from .learning import ARMS
    from .planner import CEMPlanner, GradientPlanner
    results = []
    for eval_seed in config["experiment"]["evaluation_seeds"]:
        if budget.exhausted:
            return results, False
        # Every arm/checkpoint gets the same independent evaluation planner stream.
        planner = (CEMPlanner if planner_name == "cem" else GradientPlanner)(planner_config)
        started = time.perf_counter()
        collected = collect(policy, model, planner, ARMS["ordinary"], learning,
                            config["experiment"]["evaluation_environments"], eval_seed, deterministic=True)
        # Evidence only: retain the already scheduled evaluation, without
        # extra sampling or a change to any update/selection rule.
        evidence_path = budget.output / f"evaluation-round-{round_number}-seed-{eval_seed}-{planner_name}.pt"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trace": collected.trace.detach().cpu(),
                    "actions": collected.actions.detach().cpu(),
                    "macro_starts": collected.response.starts.detach().cpu(),
                    "macro_contexts": collected.response.contexts.detach().cpu(),
                    "commands": collected.response.commands.detach().cpu(),
                    "observed_response": collected.response.observations.detach().cpu(),
                    "initial_state": collected.response.initial_state.detach().cpu(),
                    "initial_context": collected.response.initial_context.detach().cpu(),
                    "actor": {k: v.detach().cpu() for k, v in policy.actor.state_dict().items()},
                    "response_model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": config, "round": round_number, "evaluation_seed": eval_seed}, evidence_path)
        details = {"evaluation_seed": eval_seed, "planner": planner_name,
                   "retained_evidence": evidence_path.name,
                   "retained_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                   "deterministic": True, "training_shaping_disabled": True,
                   **_collection_details(collected)}
        budget.record("evaluation", round_number, started, collected.work, details)
        results.append(_jsonable(details))
    return results, True


def _save_checkpoint(path, config, seed, arm, completed, calibrated, phase, status,
                     policy, model, optimizer):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": config, "training_seed": seed, "arm": arm,
                "completed_rounds": completed, "pair_calibrated": calibrated, "lifecycle_phase": phase,
                "status": status, "source_identity": _source_identity(),
                "state": _pair_state(policy, model, optimizer)}, path)


def train(config, device, output):
    """Future server execution branch. Never invoked by plan or check."""
    from .learning import ARMS
    from .planner import CEMPlanner
    output = _work_path(output)
    if output.exists() and any(p.name != "plan.json" for p in output.iterdir()):
        raise FileExistsError("Use a new output directory; an existing training run is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", build_plan(config))
    _write_json(output / "config.json", config)
    _write_json(output / "source-identity.json", _source_identity())
    experiment, reports, grand_work = config["experiment"], [], {}
    for seed in experiment["training_seeds"]:
        seed_dir = output / f"seed-{seed}"
        warm_budget = StageBudget(experiment["common_warmup_wall_seconds"], seed_dir / "common-warmup")
        policy, model, optimizer, learning, planner_config = _make_pair(config, device, seed)
        _write_json(seed_dir / "ppo-provenance.json", policy.provenance())
        planner = CEMPlanner(planner_config)
        completed, calibrated, phase = 0, True, "INITIAL_PAIR_UNTRAINED"
        for number in range(1, experiment["common_warmup_rounds"] + 1):
            done, calibrated, phase = _round(policy, model, optimizer, planner, ARMS["ordinary"],
                                             learning, config, seed, number, warm_budget, True)
            if not done:
                break
            completed = number
        warm_complete = completed == experiment["common_warmup_rounds"] and calibrated
        warm_status = "COMPLETED_CONFIGURED_BUDGET" if warm_complete else "BUDGET_EXHAUSTED"
        _save_checkpoint(seed_dir / "common-warmup" / "checkpoint.pt", config, seed, "ordinary_warmup",
                         completed, calibrated, phase, warm_status, policy, model, optimizer)
        warm_report = {"status": warm_status, "completed_rounds": completed,
                       "wall_seconds": warm_budget.elapsed,
                       "overrun_seconds": max(0., warm_budget.elapsed-warm_budget.seconds),
                       "work": warm_budget.work, "pair_calibrated": calibrated}
        _write_json(seed_dir / "common-warmup" / "summary.json", warm_report)
        _merge_work(grand_work, warm_budget.work)
        seed_report = {"seed": seed, "warmup": warm_report, "arms": {}}
        reports.append(seed_report)
        if not warm_complete:
            seed_report["arms"] = {name: {"status": "NOT_STARTED_WARMUP_INCOMPLETE"}
                                   for name in experiment["arms"]}
            _write_json(output / "summary.json", {"status": "IN_PROGRESS", "seeds": reports, "work": grand_work})
            continue
        initial_pair, initial_rng = _pair_state(policy, model, optimizer), _rng_snapshot()
        for arm_name in experiment["arms"]:
            arm_dir = seed_dir / arm_name
            budget = StageBudget(experiment["per_arm_wall_seconds"], arm_dir)
            policy, model, optimizer, learning, planner_config = _make_pair(config, device, seed)
            _load_pair(initial_pair, policy, model, optimizer)
            _restore_rng(initial_rng)
            planner, arm = CEMPlanner(planner_config), ARMS[arm_name]
            evaluations, eval_complete = {}, True
            completed, calibrated, phase = 0, True, "CLONED_COMMON_CALIBRATED_PAIR"
            values, eval_complete = _evaluate_pair(policy, model, learning, planner_config, config, 0, budget)
            evaluations[0] = values
            for number in range(1, experiment["comparison_rounds"] + 1):
                if not eval_complete or budget.exhausted:
                    break
                done, calibrated, phase = _round(policy, model, optimizer, planner, arm,
                                                 learning, config, seed, number, budget, False)
                if not done:
                    break
                completed = number
                if number in evaluation_rounds(config):
                    values, eval_complete = _evaluate_pair(policy, model, learning, planner_config,
                                                          config, number, budget)
                    evaluations[number] = values
            all_evaluated = (eval_complete and set(evaluations) == set(evaluation_rounds(config)))
            complete = (completed == experiment["comparison_rounds"] and calibrated and all_evaluated)
            status = "COMPLETED_CONFIGURED_BUDGET" if complete else "BUDGET_EXHAUSTED"
            _save_checkpoint(arm_dir / "checkpoint.pt", config, seed, arm_name, completed,
                             calibrated, phase, status, policy, model, optimizer)
            report = {"status": status, "completed_rounds": completed, "pair_calibrated": calibrated,
                      "lifecycle_phase": phase, "wall_seconds": budget.elapsed,
                      "overrun_seconds": max(0., budget.elapsed-budget.seconds),
                      "work": budget.work, "evaluations": evaluations,
                      "frozen_is_diagnostic_only": arm_name == "frozen",
                      "completion_status_is_not_task_success": True}
            _write_json(arm_dir / "summary.json", report)
            seed_report["arms"][arm_name] = report
            _merge_work(grand_work, budget.work)
            _write_json(output / "summary.json", {"status": "IN_PROGRESS", "seeds": reports, "work": grand_work})
    complete = all(r["warmup"]["status"] == "COMPLETED_CONFIGURED_BUDGET"
                   and all(a["status"] == "COMPLETED_CONFIGURED_BUDGET" for a in r["arms"].values())
                   for r in reports)
    result = {"status": "COMPLETED_CONFIGURED_BUDGET" if complete else "BUDGET_EXHAUSTED",
              "seeds": reports, "work": grand_work,
              "completion_status_is_not_task_success_or_method_advantage": True}
    _write_json(output / "summary.json", result)
    return result


def evaluate(config, checkpoint, device, output, planner_name):
    import torch
    output = _work_path(output)
    checkpoint = Path(checkpoint).resolve()
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if not saved["pair_calibrated"]:
        raise ValueError("Checkpoint stopped before model/actor recalibration; it is not a completed evaluable pair")
    # CLI config may choose evaluation seeds/budget and CEM or command TO, but
    # must not silently change this checkpoint's physical/model contract.
    for key in ("environment", "model", "learning", "ppo"):
        if saved["config"][key] != config[key]:
            raise ValueError(f"Evaluation config changes checkpoint {key}; use its recorded contract")
    policy, model, optimizer, learning, planner_config = _make_pair(config, device, saved["training_seed"])
    _load_pair(saved["state"], policy, model, optimizer)
    budget = StageBudget(config["experiment"]["per_arm_wall_seconds"], output)
    values, complete = _evaluate_pair(policy, model, learning, planner_config, config,
                                     saved["completed_rounds"], budget, planner_name)
    result = {"status": "COMPLETED_CONFIGURED_BUDGET" if complete else "BUDGET_EXHAUSTED",
              "checkpoint": str(checkpoint), "checkpoint_status": saved["status"],
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "checkpoint_source_identity": saved["source_identity"],
              "current_source_identity": _source_identity(),
              "source_identity_matches_checkpoint": saved["source_identity"] == _source_identity(),
              "planner": planner_name, "model_information": "same learned response only",
              "evaluations": values, "work": budget.work, "wall_seconds": budget.elapsed,
              "overrun_seconds": max(0., budget.elapsed-budget.seconds),
              "comparison_warning": "CEM/TO budgets and batch timings are reported separately; no equal-wall-time or NX claim"}
    _write_json(output / "evaluation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "train", "evaluate"))
    parser.add_argument("--config", default="configs/task_response_pilot.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="work/task-response-pilot")
    parser.add_argument("--checkpoint")
    parser.add_argument("--planner", choices=("cem", "to"), default="cem")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.command == "plan":
        result = build_plan(config)
        _write_json(_work_path(args.output) / "plan.json", result)
    elif args.command == "check":
        from .verification import run_check
        result = run_check(output=_work_path(args.output))
    elif args.command == "train":
        result = train(config, args.device, args.output)
    else:
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        result = evaluate(config, args.checkpoint, args.device, args.output, args.planner)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
