"""Analyze saved deterministic task-response evaluations without new plant runs.

Example (run after the fixed evaluation files have been copied locally):
    python scripts/analyze_task_response_evidence.py \
        --input-root work/server-pilot-20261019-r1 \
        --output work/server-pilot-20261019-r1-analysis \
        --fixed-input-actor

Evaluation payloads are limited to evaluation-round-*-seed-91001-cem.pt;
run manifests/config and local source files are also read for identity checks.
CompactResponse is forward-evaluated on recorded starts/commands; no collector, plant.step,
planner, optimizer, training entry point, or network connection is called.
The optional actor diagnostic uses fixed observations from ordinary rounds
0/16. It predicts actions only, never their unobserved physical consequences.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
PATTERN = "evaluation-round-*-seed-91001-cem.pt"
ARMS = ("ordinary", "uniform", "weighted_id", "codesign", "frozen")
BASELINES = ("ordinary", "uniform", "weighted_id")
FEATURES = ("px_normalized", "py_normalized", "vx_normalized", "vy_normalized", "input_resource")
COST_COMPONENT_NAMES = (
    "actual_stage_position_cost", "actual_stage_velocity_cost",
    "actual_stage_resource_cost", "actual_stage_geometry_cost",
    "actual_terminal_position_cost", "actual_terminal_velocity_cost",
    "actual_terminal_geometry_cost",
)


def runtime() -> SimpleNamespace:
    # Keep --help usable without constructing any model or requiring PPO deps.
    sys.path.insert(0, str(REPOSITORY / "src"))
    import torch
    from task_response.environment import Config, advance_context, project_action, task_metrics
    from task_response.models import CompactResponse, actor_observation
    from task_response.learning import predict_macro
    from task_response.planner import differentiable_rollout_cost
    torch.set_num_threads(1)
    return SimpleNamespace(**locals())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_identity(root: Path, items: dict) -> dict:
    manifest_path, config_path = root/"source-identity.json", root/"config.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    files = {}
    for raw_name, expected in saved.items():
        name = raw_name.replace("\\", "/")
        path = REPOSITORY/name
        current = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if path.is_file() else None
        files[name] = {"recorded_sha256_lf": expected, "local_sha256_lf": current, "matches": current == expected}
    used = ("src/task_response/models.py", "src/task_response/environment.py",
            "src/task_response/learning.py", "src/task_response/planner.py",
            "src/response_ppo/ppo_backend.py")
    used_match = all(name in files and files[name]["matches"] for name in used) if saved else None
    recorded_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    run_config_hash = config_digest(recorded_config) if recorded_config is not None else None
    unique_config_hashes = sorted({item.config_hash for item in items.values()})
    configs_match = (all(item.config_hash == run_config_hash for item in items.values())
                     if recorded_config is not None and items else None)
    return {"source_manifest_available": bool(saved), "files": files,
            "used_prediction_and_actor_sources_match": used_match,
            "all_recorded_sources_match_local": all(v["matches"] for v in files.values()) if saved else None,
            "run_config_available": recorded_config is not None, "run_config_sha256": run_config_hash,
            "snapshot_config_sha256_values": unique_config_hashes,
            "snapshot_configs_match_run_config": configs_match,
            "snapshot_configs_mutually_identical": len(unique_config_hashes) == 1,
            "git_commit_in_evaluation_snapshot": None,
            "scope": "File-level source version and canonical config checks; no Git commit is invented from missing metadata."}


def tensor_state_digest(state: dict) -> str:
    sha = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        sha.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        sha.update(value.numpy().tobytes())
    return sha.hexdigest()


def initial_pair_identity(items: dict, rt: SimpleNamespace) -> list[dict]:
    torch, rows = rt.torch, []
    seeds = sorted({key[0] for key in items})
    for seed in seeds:
        reference = items.get((seed, "ordinary", 0))
        seed_items = [item for key, item in items.items() if key[0] == seed]
        expected_arms = seed_items[0].data["config"]["experiment"]["arms"]
        result = {"training_seed": seed, "round": 0, "reference_arm": "ordinary",
                  "scope": "Saved actor and response_model only; critic/Adam states were not retained in evaluation evidence.",
                  "arms": [], "all_configured_arms_have_identical_actor_and_model": None}
        if reference is None:
            result["status"] = "REFERENCE_ROUND0_MISSING"
            rows.append(result)
            continue
        result["reference_evidence"] = reference.relative_path
        present, identical = True, True
        for arm in expected_arms:
            target = items.get((seed, arm, 0))
            if target is None:
                result["arms"].append({"arm": arm, "status": "ROUND0_MISSING"})
                present = False
                continue
            row = {"arm": arm, "evidence": target.relative_path,
                   "config_matches_reference": target.config_hash == reference.config_hash}
            for name in ("actor", "response_model"):
                first, second = reference.data[name], target.data[name]
                differences = []
                for key in sorted(set(first) | set(second)):
                    if key not in first or key not in second:
                        differences.append({"key": key, "reason": "KEY_MISSING"})
                    elif first[key].shape != second[key].shape or first[key].dtype != second[key].dtype:
                        differences.append({"key": key, "reason": "SHAPE_OR_DTYPE_DIFFERENT"})
                    elif not torch.equal(first[key], second[key]):
                        differences.append({"key": key, "maximum_absolute_difference":
                                            plain((first[key].to(torch.float64)-second[key].to(torch.float64)).abs().max())})
                row[name+"_sha256"] = tensor_state_digest(second)
                row[name+"_equal_to_ordinary_round0"] = not differences
                row[name+"_different_tensors"] = differences
                identical = identical and not differences
            identical = identical and row["config_matches_reference"]
            row["status"] = "PRESENT"
            result["arms"].append(row)
        result["status"] = "COMPLETE" if present else "INCOMPLETE"
        result["all_configured_arms_have_identical_actor_and_model"] = identical if present else None
        rows.append(result)
    return rows


def plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        return plain(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class Evidence:
    path: Path
    relative_path: str
    seed: int
    arm: str
    round: int
    data: dict
    config_hash: str
    file_hash: str
    task_names: list[str]
    per_environment: dict[str, Any] | None = None

    def identity(self) -> dict:
        return {"training_seed": self.seed, "arm": self.arm, "round": self.round,
                "evaluation_seed": 91001, "evidence": self.relative_path,
                "evidence_sha256": self.file_hash, "config_sha256": self.config_hash}


def load_evidence(path: Path, root: Path, rt: SimpleNamespace) -> Evidence:
    torch = rt.torch
    filename = re.fullmatch(r"evaluation-round-(\d+)-seed-91001-cem\.pt", path.name)
    seed_dir = re.fullmatch(r"seed-(\d+)", path.parent.parent.name)
    if filename is None or seed_dir is None or path.parent.name not in ARMS:
        raise ValueError("Expected seed-N/arm/evaluation-round-R-seed-91001-cem.pt identity")
    # Saved payload is tensors and primitive config values; no unsafe pickle fallback.
    data = torch.load(path, map_location="cpu", weights_only=True)
    round_number, seed = int(filename[1]), int(seed_dir[1])
    if data["round"] != round_number or data["evaluation_seed"] != 91001:
        raise ValueError("Filename and saved round/evaluation seed disagree")
    cfg = rt.Config(**data["config"]["environment"])
    trace, actions = data["trace"], data["actions"]
    t, b, action_dim = actions.shape
    m, k = t // cfg.macro_steps, cfg.macro_steps
    shapes = {"trace": (t+1, b, 6), "actions": (t, b, 2),
              "macro_starts": (m, b, 6), "macro_contexts": (m, b, 6),
              "commands": (m, b, 2), "observed_response": (m, k, b, 5),
              "initial_state": (b, 6), "initial_context": (b, 6)}
    if t != cfg.horizon or t % k or action_dim != 2:
        raise ValueError("This analyzer requires a saved complete configured evaluation")
    for key, shape in shapes.items():
        if tuple(data[key].shape) != shape or not torch.isfinite(data[key]).all():
            raise ValueError(f"Invalid saved {key}: expected finite tensor of shape {shape}")
    if not torch.equal(trace[0], data["initial_state"]):
        raise ValueError("Saved initial state disagrees with the physical trace")
    if not torch.equal(trace[:-1:k], data["macro_starts"]):
        raise ValueError("Saved macro starts disagree with the physical trace")
    if not torch.equal(data["macro_contexts"][0], data["initial_context"]):
        raise ValueError("Saved initial and first macro contexts disagree")
    names = []
    for context in data["initial_context"]:
        velocity = context[2:4]
        if torch.allclose(velocity, velocity.new_tensor([cfg.go_velocity, 0.])):
            names.append("go")
        elif torch.allclose(velocity, torch.zeros_like(velocity)):
            names.append("stop")
        else:
            raise ValueError("Saved context is outside the configured go/stop task labels")
    return Evidence(path, path.relative_to(root).as_posix(), seed, path.parent.name,
                    round_number, data, config_digest(data["config"]), digest(path), names)


def error_statistics(error: Any, scale: Any, dt: float, macros: int, prefix: str,
                     rt: SimpleNamespace) -> dict:
    # Shape [time,B,5]; dispersion of these residuals is NOT policy conditional variance.
    per_feature_rmse = error.square().mean(0).sqrt()
    physical = error * scale
    return {
        prefix+"normalized_mse": error.square().mean((0, 2)),
        prefix+"feature_rmse": per_feature_rmse,
        prefix+"feature_signed_bias": error.mean(0),
        prefix+"position_rmse_l2": physical[..., :2].square().sum(-1).mean(0).sqrt(),
        prefix+"velocity_rmse_l2": physical[..., 2:4].square().sum(-1).mean(0).sqrt(),
        prefix+"resource_rmse": per_feature_rmse[..., 4],
        prefix+"unweighted_error_integral_per_macro": dt * error.square().sum((0, 2)) / macros,
    }


def actual_cost_components(item: Evidence, rt: SimpleNamespace) -> dict:
    """Algebra on saved states/actions, matching task_cost/terminal_cost exactly."""
    d = item.data
    cfg = rt.Config(**d["config"]["environment"])
    following, context = d["trace"][1:], d["initial_context"]
    position_squared = (following[..., :2]-context[None, ..., :2]).square().sum(-1)
    velocity_squared = (following[..., 2:4]-context[None, ..., 2:4]).square().sum(-1)
    geometry_squared = (following[..., 1].abs()-context[None, ..., 4]).clamp_min(0).square()
    rho = rt.project_action(d["actions"], cfg).square().sum(-1) / cfg.input_bound**2
    return {
        "actual_stage_position_cost": cfg.dt * cfg.position_weight * position_squared.sum(0),
        "actual_stage_velocity_cost": cfg.dt * cfg.velocity_weight * velocity_squared.sum(0),
        "actual_stage_resource_cost": cfg.dt * cfg.resource_weight * rho.sum(0),
        "actual_stage_geometry_cost": cfg.dt * cfg.geometry_weight * geometry_squared.sum(0),
        "actual_terminal_position_cost": cfg.terminal_position_weight * position_squared[-1],
        "actual_terminal_velocity_cost": cfg.terminal_velocity_weight * velocity_squared[-1],
        "actual_terminal_geometry_cost": cfg.geometry_weight * geometry_squared[-1],
    }


def logged_actual_cost(item: Evidence, rt: SimpleNamespace) -> tuple[Any, dict]:
    """Read the already recorded evaluation cost, never substitute a recomputation."""
    path = item.path.parent/"stages.jsonl"
    reference = {"file": str(Path(item.relative_path).with_name("stages.jsonl")).replace("\\", "/")}
    if not path.exists():
        return None, {**reference, "status": "LOG_NOT_RETAINED_OR_NOT_COPIED"}
    matches = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            details = record.get("details", {})
            if (record.get("stage") == "evaluation" and record.get("round") == item.round
                    and details.get("evaluation_seed") == 91001 and details.get("planner") == "cem"
                    and details.get("retained_evidence", item.path.name) == item.path.name):
                matches.append((line_number, details["task_metrics"]["task_cost"]))
    if not matches:
        return None, {**reference, "status": "MATCHING_EVALUATION_LOG_NOT_FOUND"}
    if len(matches) != 1:
        raise ValueError("Multiple logged costs match the same evaluation evidence identity")
    line_number, values = matches[0]
    logged = rt.torch.tensor(values, dtype=item.data["trace"].dtype)
    if tuple(logged.shape) != (item.data["actions"].shape[1],) or not rt.torch.isfinite(logged).all():
        raise ValueError("Logged actual task cost has a wrong shape or nonfinite values")
    return logged, {**reference, "status": "MATCHED_SAVED_EVALUATION_LOG",
                    "line": line_number, "file_sha256": digest(path)}


def analyze_one(item: Evidence, rt: SimpleNamespace) -> tuple[list[dict], list[dict], dict]:
    torch, d = rt.torch, item.data
    cfg = rt.Config(**d["config"]["environment"])
    model = rt.CompactResponse(cfg, hidden=d["config"]["model"]["hidden"])
    reference = next(iter(d["response_model"].values()))
    model = model.to(dtype=reference.dtype).eval()
    model.load_state_dict(d["response_model"], strict=True)
    model.requires_grad_(False)
    m, k, b, _ = d["observed_response"].shape
    with torch.inference_mode():
        # Labels are checked against SAVED states/actions, without advancing the plant.
        labels = model.observed_features(d["trace"][1:], d["actions"]).reshape(m, k, b, 5)
        label_difference = float((labels-d["observed_response"]).abs().max())
        if label_difference > 1e-5:
            raise ValueError("Saved response labels disagree with saved physical states/actions")
        local = rt.predict_macro(model, d["macro_starts"].reshape(m*b, 6),
                                 d["commands"].reshape(m*b, 2),
                                 d["macro_contexts"].reshape(m*b, 6))
        local = local.reshape(k, m, b, 5).transpose(0, 1)
        rollout = rt.differentiable_rollout_cost(
            model, d["initial_state"], d["commands"].transpose(0, 1),
            d["initial_context"], cfg, return_trace=True)
        long = model.features(rollout.states[1:], rollout.resources)
        actual = d["observed_response"].reshape(m*k, b, 5)
        local_error = (local-d["observed_response"]).reshape(m*k, b, 5)
        long_error = long-actual
        if not torch.isfinite(local_error).all() or not torch.isfinite(long_error).all():
            raise ValueError("Saved response model produces nonfinite offline predictions")
        metrics = rt.task_metrics(d["trace"], d["actions"], d["initial_context"], cfg)
        components = actual_cost_components(item, rt)
        component_sum = sum(components.values())
        logged_cost, logged_reference = logged_actual_cost(item, rt)
        if logged_cost is None:
            logged_cost = torch.full_like(metrics["task_cost"], float("nan"))
        values = {
            "actual_task_cost": metrics["task_cost"],
            **components,
            "actual_cost_component_sum": component_sum,
            "actual_cost_components_minus_recomputed_cost": component_sum-metrics["task_cost"],
            "actual_logged_task_cost": logged_cost,
            "actual_cost_components_minus_logged_cost": component_sum-logged_cost,
            "actual_recomputed_minus_logged_cost": metrics["task_cost"]-logged_cost,
            "actual_success": metrics["success"].to(torch.float64),
            "actual_terminal_position_error": metrics["terminal_position_error"],
            "actual_terminal_velocity_error": metrics["terminal_velocity_error"],
            "actual_input_integral": metrics["normalized_input_integral"],
            "actual_corridor_violation_count": metrics["corridor_violation_count"],
            "actual_maximum_corridor_excess": metrics["maximum_corridor_excess"],
            "actual_first_reach_time": metrics["first_reach_time"],
            "recorded_commands_model_cost": rollout.cost,
            # Positive means the model assigned less cost than was actually observed.
            "optimism_gap_actual_minus_model": metrics["task_cost"]-rollout.cost,
            **error_statistics(local_error, model.feature_scale, cfg.dt, m, "local_", rt),
            **error_statistics(long_error, model.feature_scale, cfg.dt, m, "long_", rt),
        }
    item.per_environment = values
    episodes = [{**item.identity(), "task": item.task_names[n], "environment_index": n,
                 **{name: plain(value[n]) for name, value in values.items()}} for n in range(b)]
    summaries = []
    local_by_step = local_error.reshape(m, k, b, 5)
    for task in sorted(set(item.task_names)):
        ids = [i for i, name in enumerate(item.task_names) if name == task]
        row = {**item.identity(), "task": task, "environments": len(ids)}
        for name, value in values.items():
            selected = value[ids].to(torch.float64)
            # Unreached first-reach times remain unavailable, with their denominator explicit.
            if name == "actual_first_reach_time":
                finite = torch.isfinite(selected)
                row[name+"_reached_mean"] = plain(selected[finite].mean()) if finite.any() else None
                row["actual_first_reach_count"] = int(finite.sum())
            else:
                row[name+"_mean"] = plain(selected.mean(0))
        row["local_feature_rmse_by_microstep"] = plain(
            local_by_step[:, :, ids].square().mean((0, 2)).sqrt())
        summaries.append(row)
    work = {**item.identity(), "saved_label_max_difference": label_difference,
            "logged_actual_task_cost_reference": logged_reference,
            "cost_components_vs_recomputed_max_abs_error": plain((component_sum-metrics["task_cost"]).abs().max()),
            "cost_components_vs_logged_max_abs_error": plain((component_sum-logged_cost).abs().max()),
            "recomputed_vs_logged_max_abs_error": plain((metrics["task_cost"]-logged_cost).abs().max()),
            "offline_model_microsteps": 2*m*k*b,
            "offline_model_encoder_calls": m*b+b,
            "new_physical_steps": 0, "optimizer_steps": 0}
    return episodes, summaries, work


def paired_outcomes(items: dict, rt: SimpleNamespace) -> list[dict]:
    torch, rows = rt.torch, []
    fields = ("actual_task_cost", *COST_COMPONENT_NAMES, "actual_success", "actual_input_integral",
              "actual_terminal_position_error", "actual_terminal_velocity_error",
              "local_normalized_mse", "long_normalized_mse",
              "optimism_gap_actual_minus_model", "recorded_commands_model_cost")
    for (seed, arm, round_number), candidate in sorted(items.items()):
        if arm != "codesign":
            continue
        for baseline_name in BASELINES:
            baseline = items.get((seed, baseline_name, round_number))
            if baseline is None:
                continue
            same_conditions = (candidate.config_hash == baseline.config_hash
                               and torch.equal(candidate.data["initial_state"], baseline.data["initial_state"])
                               and torch.equal(candidate.data["initial_context"], baseline.data["initial_context"]))
            if not same_conditions:
                rows.append({"training_seed": seed, "round": round_number,
                             "baseline": baseline_name, "status": "UNMATCHED_INITIAL_CONDITIONS_OR_CONFIG"})
                continue
            for task in sorted(set(candidate.task_names)):
                ids = [i for i, name in enumerate(candidate.task_names) if name == task]
                row = {"training_seed": seed, "round": round_number, "evaluation_seed": 91001,
                       "task": task, "candidate": "codesign", "baseline": baseline_name,
                       "environments": len(ids), "status": "PAIRED_RECORDED_OUTCOMES",
                       "candidate_evidence": candidate.relative_path, "baseline_evidence": baseline.relative_path}
                for name in fields:
                    delta = candidate.per_environment[name]-baseline.per_environment[name]
                    row[name+"_delta_codesign_minus_baseline"] = plain(delta[ids].to(torch.float64).mean())
                for key, channel, label in (("trace", slice(0, 2), "realized_position_path"),
                                             ("trace", slice(2, 4), "realized_velocity_path"),
                                             ("actions", slice(None), "realized_applied_action"),
                                             ("commands", slice(None), "realized_macro_command")):
                    difference = candidate.data[key][:, ids, channel]-baseline.data[key][:, ids, channel]
                    row[label+"_rms_l2_difference"] = plain(difference.square().sum(-1).mean().sqrt())
                rows.append(row)
    return rows


def recorded_actor_observations(item: Evidence, rt: SimpleNamespace) -> Any:
    d, torch = item.data, rt.torch
    cfg = rt.Config(**d["config"]["environment"])
    m, k = d["commands"].shape[0], cfg.macro_steps
    current_contexts = []
    for macro in range(m):
        context = d["macro_contexts"][macro]
        for _ in range(k):
            current_contexts.append(context)
            context = rt.advance_context(context, config=cfg)
    return rt.actor_observation(d["trace"][:-1],
                               d["commands"].repeat_interleave(k, dim=0),
                               torch.stack(current_contexts))


def saved_mean_actor_action(item: Evidence, observations: Any, rt: SimpleNamespace) -> Any:
    """Exact GaussianActor.mean forward from saved tensors; no PPO/optimizer object."""
    torch, weights = rt.torch, item.data["actor"]
    ppo = item.data["config"]["ppo"]
    dimensions = [ppo["obs_dim"], *ppo["hidden"], ppo["action_dim"]]
    value = observations.to(dtype=weights["mean.0.weight"].dtype)
    for layer in range(len(dimensions)-1):
        weight, bias = weights[f"mean.{2*layer}.weight"], weights[f"mean.{2*layer}.bias"]
        if tuple(weight.shape) != (dimensions[layer+1], dimensions[layer]):
            raise ValueError("Saved actor architecture differs from its declared GaussianActor config")
        value = torch.nn.functional.linear(value, weight, bias)
        if layer < len(dimensions)-2:
            value = value.tanh()
    # Matches PPOBackend.act(deterministic=True) followed by the fixed radial ball map.
    return rt.project_action(ppo["action_bound"]*value.tanh(),
                             rt.Config(**item.data["config"]["environment"]))


def fixed_actor_diagnostics(items: dict, reference_rounds: list[int], rt: SimpleNamespace) -> tuple[list[dict], int]:
    torch, rows, forwards = rt.torch, [], 0
    with torch.inference_mode():
        for (seed, arm, reference_round), reference in sorted(items.items()):
            if arm != "ordinary" or reference_round not in reference_rounds:
                continue
            observations = recorded_actor_observations(reference, rt)
            rounds = sorted({key[2] for key in items if key[0] == seed})
            for round_number in rounds:
                predictions = {}
                for arm_name in ("codesign", *BASELINES):
                    target = items.get((seed, arm_name, round_number))
                    if target is not None and target.config_hash == reference.config_hash:
                        predictions[arm_name] = saved_mean_actor_action(target, observations, rt)
                        forwards += observations.shape[0]*observations.shape[1]
                if "codesign" not in predictions:
                    continue
                for baseline in BASELINES:
                    if baseline not in predictions:
                        continue
                    for task in sorted(set(reference.task_names)):
                        ids = [i for i, name in enumerate(reference.task_names) if name == task]
                        difference = predictions["codesign"][:, ids]-predictions[baseline][:, ids]
                        row = {"training_seed": seed, "target_round": round_number,
                               "reference_arm": "ordinary", "reference_round": reference_round,
                               "reference_evidence": reference.relative_path,
                               "reference_sha256": reference.file_hash,
                               "task": task, "candidate": "codesign", "baseline": baseline,
                               "observation_count": difference.shape[0]*difference.shape[1],
                               "status": "COUNTERFACTUAL_FIXED_OBSERVATION_ACTION_ONLY",
                               "applied_action_rms_l2_difference": plain(difference.square().sum(-1).mean().sqrt()),
                               "applied_action_max_l2_difference": plain(difference.norm(dim=-1).max()),
                               "applied_action_component_rmse": plain(difference.square().mean((0, 1)).sqrt())}
                        rows.append(row)
    return rows, forwards


def across_seed_summary(task_rows: list[dict]) -> list[dict]:
    groups = {}
    for row in task_rows:
        groups.setdefault((row["arm"], row["round"], row["task"]), []).append(row)
    output = []
    for (arm, round_number, task), rows in sorted(groups.items()):
        result = {"arm": arm, "round": round_number, "task": task,
                  "training_seeds": sorted(r["training_seed"] for r in rows),
                  "training_seed_count": len(rows), "aggregation": "equal_weight_of_training_seed_means"}
        for key in rows[0]:
            if key.endswith("_mean") and all(isinstance(r.get(key), (int, float)) for r in rows):
                values = [r[key] for r in rows]
                result[key] = sum(values)/len(values)
                result[key+"_seed_min"] = min(values)
                result[key+"_seed_max"] = max(values)
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, required=True, help="One saved fixed-protocol run directory")
    parser.add_argument("--output", type=Path, required=True, help="New analysis directory; input files are never modified")
    parser.add_argument("--fixed-input-actor", action="store_true", help="Enable bounded fixed-observation actor forwards")
    parser.add_argument("--reference-rounds", type=int, nargs="+", default=[0, 16])
    args = parser.parse_args()
    root, output = args.input_root.resolve(), args.output.resolve()
    paths = sorted(root.rglob(PATTERN))
    if not paths:
        parser.error(f"No saved {PATTERN} files found; no analysis was generated")
    if output.exists():
        parser.error("Use a new output directory to preserve earlier analysis")
    rt = runtime()
    items, episodes, tasks, works, errors = {}, [], [], [], []
    for path in paths:
        try:
            item = load_evidence(path, root, rt)
            key = (item.seed, item.arm, item.round)
            if key in items:
                raise ValueError("Duplicate training-seed/arm/round evidence identity")
            episode_rows, task_rows, work = analyze_one(item, rt)
            items[key] = item
            episodes.extend(episode_rows); tasks.extend(task_rows); works.append(work)
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            errors.append({"evidence": path.relative_to(root).as_posix(),
                           "error_type": type(error).__name__, "error": str(error)})
    comparisons = paired_outcomes(items, rt)
    actor_rows, actor_forwards = (fixed_actor_diagnostics(items, args.reference_rounds, rt)
                                  if args.fixed_input_actor else ([], 0))
    expected, missing = [], []
    if items:
        config = next(iter(items.values())).data["config"]
        experiment = config["experiment"]
        rounds = sorted({0, experiment["comparison_rounds"],
                         *range(experiment["evaluation_every_rounds"], experiment["comparison_rounds"]+1,
                                experiment["evaluation_every_rounds"])})
        expected = [(s, a, r) for s in experiment["training_seeds"] for a in experiment["arms"] for r in rounds]
        missing = [{"training_seed": s, "arm": a, "round": r} for s, a, r in expected if (s, a, r) not in items]
    source_files = ("scripts/analyze_task_response_evidence.py", "src/task_response/models.py",
                    "src/task_response/environment.py", "src/task_response/learning.py", "src/task_response/planner.py")
    identity = source_identity(root, items)
    identity_verified = (identity["used_prediction_and_actor_sources_match"] is True
                         and identity["snapshot_configs_match_run_config"] is True
                         and identity["snapshot_configs_mutually_identical"])
    complete = bool(items) and not errors and not missing
    status = ("COMPLETE_SAVED_EVIDENCE_ANALYSIS" if complete and identity_verified
              else "SOURCE_OR_CONFIG_UNVERIFIED" if complete else "PARTIAL_OR_INVALID_EVIDENCE")
    result = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_pattern": PATTERN, "feature_order": FEATURES,
        "source_sha256": {name: digest(REPOSITORY/name) for name in source_files},
        "input_files_found": len(paths), "input_files_analyzed": len(items),
        "expected_configured_evaluations": len(expected), "missing_evaluations": missing,
        "source_and_config_identity": identity,
        "initial_actor_model_identity": initial_pair_identity(items, rt),
        "errors": errors, "work": works,
        "new_physical_steps": 0, "optimizer_steps": 0, "offline_fixed_input_actor_forwards": actor_forwards,
        "interpretation_boundaries": [
            "Offline predictions use the local source files; source/config identity must match before treating them as the saved model's behavior.",
            "Positive actual-minus-model cost gap means optimism on the saved command sequence.",
            "Actual cost components are algebraic sums on saved trajectories; missing evaluation logs remain unverified rather than being replaced by recomputed costs.",
            "The model cost uses recorded commands; it is not a fresh closed-loop replanning result or counterfactual actual cost.",
            "Local errors reset from each saved real macro start; long errors encode only once at the saved episode start.",
            "All residual metrics use a common unweighted feature scale; evaluation does not retain training W or shaping losses.",
            "Comparisons of visited trajectories/actions/commands combine feedback, planning, and changed visitation; they are not pure feedback effects.",
            "Fixed-observation actor differences establish a mapping difference only; they do not establish useful or feasible physical consequences.",
            "No conditional response variance is estimated by pooling different states or deterministic evaluation trajectories.",
            "The contribution of stronger effective training regularization cannot be identified from these saved evaluation files alone.",
            "Seed summaries weight training seeds equally; no statistical significance or independent-method advantage is inferred.",
        ],
        "per_environment": episodes, "by_training_seed_task": tasks,
        "across_training_seeds": across_seed_summary(tasks),
        "paired_recorded_outcomes": comparisons,
        "fixed_input_actor_comparisons": actor_rows,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output/"analysis.json").write_text(json.dumps(plain(result), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for filename, rows in (("episode_metrics.csv", episodes), ("task_summary.csv", tasks),
                           ("matched_outcome_comparisons.csv", comparisons),
                           ("fixed_input_actor_comparisons.csv", actor_rows),
                           ("across_training_seeds.csv", result["across_training_seeds"])):
        write_csv(output/filename, rows)
    print(json.dumps({"status": result["status"], "files_analyzed": len(items),
                      "expected_files": len(expected), "errors": len(errors),
                      "new_physical_steps": 0, "output": str(output)}, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETE_SAVED_EVIDENCE_ANALYSIS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
