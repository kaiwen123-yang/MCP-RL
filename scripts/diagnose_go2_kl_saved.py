"""Read-only KL diagnosis from saved behavior batches and actor checkpoints.

No simulator, optimizer, optimizer.step, policy sampling or fitting is used.
In a one-update round the saved post actor directly exposes the state at the
first rejected minibatch. For two-update rounds only the final stopping
location is inferred from the update count and re-evaluated; intermediate
actor states and the earlier sequence of checks are not recreated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from go2_learning.policy import TaskActor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


@torch.no_grad()
def means(actor, batch, index=None):
    if index is None:
        return torch.cat([
            actor.mean(batch["state"][ix], batch["context"][ix], batch["command"][ix])
            for ix in torch.arange(len(batch["state"])).split(256)
        ])
    return actor.mean(batch["state"][index], batch["context"][index],
                      batch["command"][index])


def gaussian_parts(old_mean, new_mean, old_log_std, new_log_std):
    old_mean, new_mean = old_mean.double(), new_mean.double()
    old_log_std, new_log_std = old_log_std.double(), new_log_std.double()
    mean = (old_mean - new_mean).square() / (2 * torch.exp(2 * new_log_std))
    std = new_log_std - old_log_std + .5 * torch.exp(
        2 * (old_log_std - new_log_std)) - .5
    return mean, std.expand_as(mean)


def sampled_kl(new_mean, new_log_std, raw, saved_old_logp):
    # Preserve production float32 arithmetic for the stopping-rule statistic.
    new_logp = torch.distributions.Normal(
        new_mean, new_log_std.exp().expand_as(new_mean)).log_prob(raw).sum(-1)
    logratio = new_logp - saved_old_logp
    return (logratio.exp() - 1) - logratio, new_logp


def statistics(values):
    x = values.detach().double()
    return {"mean": float(x.mean()), "median": float(x.median()),
            "p95": float(torch.quantile(x, .95)), "max": float(x.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "work/go2-public-assets")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notes/go2-adjoint-diagnosis-2026-10-20/kl")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((args.results / "config.json").read_text(encoding="utf-8"))
    policy_root = args.assets / "rl_sar/policy/go2"
    base = yaml.safe_load((policy_root / "base.yaml").read_text(encoding="utf-8"))["go2"]
    specific = yaml.safe_load(
        (policy_root / "robot_lab/config.yaml").read_text(encoding="utf-8"))["go2/robot_lab"]
    policy_config = {**base, **specific}
    actors = [
        TaskActor(policy_root / "robot_lab" / specific["model_name"],
                  policy_config, config["initial_log_std"]).eval()
        for _ in range(2)
    ]
    for actor in actors:
        actor.requires_grad_(False)
    pre_actor, post_actor = actors
    ppo = config["ppo"]
    target = ppo["target_kl"]
    rows, stages, joints, parameters, minibatches = [], [], [], [], []
    sources = {}
    first_batch_fingerprints = {}
    round_records = {}
    round1_means = {}
    adam_checks = []

    def source(path):
        relative = str(path.relative_to(args.results)).replace("\\", "/")
        sources[relative] = sha(path)
        return relative

    for seed in config["seeds"]:
        for arm in ("ordinary", "adjoint"):
            rng = torch.Generator().manual_seed(seed + 71)
            for number in range(1, config["rounds"] + 1):
                folder = args.results / f"seed{seed}" / arm / f"round{number}"
                batch_path = folder / "behavior/transitions.pt"
                saved = torch.load(batch_path, map_location="cpu", weights_only=True)
                batch = saved["batch"]
                n = len(batch["state"])
                pre_path = (args.results / f"seed{seed}/common/initialization.pt"
                            if number == 1 else folder.parent / f"round{number - 1}/checkpoint.pt")
                post_path = folder / "checkpoint.pt"
                before = torch.load(pre_path, map_location="cpu", weights_only=True)
                after = torch.load(post_path, map_location="cpu", weights_only=True)
                pre_actor.load_state_dict(before["actor"])
                post_actor.load_state_dict(after["actor"])
                record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
                update_count = int(record["ppo"]["actor_updates"])
                stopped = bool(record["ppo"]["kl_early_stop"])
                partitions = []
                for epoch in range(ppo["epochs"]):
                    order = torch.randperm(n, generator=rng)
                    partitions.extend((epoch + 1, j + 1, ix)
                                      for j, ix in enumerate(order.split(ppo["minibatch_size"])))
                # Count of accepted optimizer updates equals the zero-based
                # location of the first rejected check in this implementation.
                trigger_epoch, trigger_batch, trigger = (
                    partitions[update_count] if stopped else (None, None, None))
                old_mean = means(pre_actor, batch)
                new_mean = means(post_actor, batch)
                ls0, ls1 = pre_actor.log_std.detach(), post_actor.log_std.detach()
                mean_part, std_part = gaussian_parts(old_mean, new_mean, ls0, ls1)
                exact = (mean_part + std_part).sum(-1)
                estimate, _ = sampled_kl(new_mean, ls1, batch["raw_action"],
                                        batch["old_log_prob"])
                # This uses the exact stored behavior means, avoiding the
                # singleton-vs-batch GEMM rounding of an actor recomputation.
                stored_old_logp = torch.distributions.Normal(
                    batch["old_mean"], ls0.exp().expand_as(batch["old_mean"])
                ).log_prob(batch["raw_action"]).sum(-1)
                reconstructed_old_logp = torch.distributions.Normal(
                    old_mean, ls0.exp().expand_as(old_mean)
                ).log_prob(batch["raw_action"]).sum(-1)
                delta = (new_mean - old_mean).double()
                normalized_delta = delta / ls0.double().exp()
                common_fields = {"seed": seed, "arm": arm, "round": number}
                row = {
                    **common_fields, "samples": n, "actor_updates": update_count,
                    "kl_early_stop": stopped, "target_kl": target,
                    "change_identity": ("one_actual_optimizer_update" if update_count == 1
                                        else "cumulative_actual_updates_no_intermediate_weights"),
                    "all_exact_kl_mean": float(exact.mean()),
                    "all_exact_kl_p95": float(torch.quantile(exact, .95)),
                    "all_estimated_kl": float(estimate.mean()),
                    "all_mean_component": float(mean_part.sum(-1).mean()),
                    "all_std_component": float(std_part.sum(-1).mean()),
                    "std_fraction": float(std_part.sum(-1).mean() / exact.mean()),
                    "mean_raw_rms_per_joint": float(delta.square().mean().sqrt()),
                    "mean_shift_rms_in_old_std": float(normalized_delta.square().mean().sqrt()),
                    "mean_shift_l2_rms": float(delta.square().sum(-1).mean().sqrt()),
                    "old_std_min": float(ls0.exp().min()), "old_std_max": float(ls0.exp().max()),
                    "log_std_change_max": float((ls1 - ls0).abs().max()),
                    "stored_mean_old_logp_max_error": float((stored_old_logp - batch["old_log_prob"]).abs().max()),
                    "recomputed_old_mean_max_error": float((old_mean - batch["old_mean"]).abs().max()),
                    "recomputed_old_logp_max_error": float((reconstructed_old_logp - batch["old_log_prob"]).abs().max()),
                    "trigger_epoch": trigger_epoch, "trigger_minibatch": trigger_batch,
                    "stop_check_identity": (
                        "one_update_first_rejection_reconstruction"
                        if update_count == 1 else
                        "post_actor_at_final_stop_location_inferred_from_update_count_no_intermediate_weights"),
                    "trigger_size": len(trigger) if trigger is not None else None,
                    "trigger_indices_sha256": (
                        hashlib.sha256(trigger.numpy().tobytes()).hexdigest()
                        if trigger is not None else None),
                    "reported_last_approx_kl": record["ppo"]["approx_kl"],
                }
                if trigger is not None:
                    old_trigger, new_trigger = means(pre_actor, batch, trigger), means(post_actor, batch, trigger)
                    mt, st = gaussian_parts(old_trigger, new_trigger, ls0, ls1)
                    est, _ = sampled_kl(new_trigger, ls1, batch["raw_action"][trigger],
                                        batch["old_log_prob"][trigger])
                    row.update(trigger_exact_kl=float((mt + st).sum(-1).mean()),
                               trigger_estimated_kl=float(est.mean()),
                               trigger_estimate_minus_exact=float(est.mean() - (mt + st).sum(-1).mean()))
                    first = partitions[0][2]
                    initial_est, _ = sampled_kl(means(pre_actor, batch, first), ls0,
                                               batch["raw_action"][first], batch["old_log_prob"][first])
                    row["first_checked_preupdate_estimated_kl"] = float(initial_est.mean())
                    last = partitions[-1][2]
                    last_est, _ = sampled_kl(means(post_actor, batch, last), ls1,
                                            batch["raw_action"][last], batch["old_log_prob"][last])
                    row["reconstructed_last_approx_kl"] = float(last_est.mean())
                    row["last_metric_reconstruction_error"] = abs(float(last_est.mean()) - record["ppo"]["approx_kl"])
                for epoch, index, ix in partitions:
                    # Diagnostic final-actor evaluations; these are not the
                    # intermediate actors used before the real stop check.
                    minibatches.append({
                        **common_fields, "epoch": epoch, "minibatch": index,
                        "size": len(ix),
                        "is_one_update_first_rejection": stopped and update_count == 1 and epoch == trigger_epoch and index == trigger_batch,
                        "is_final_stop_location": stopped and epoch == trigger_epoch and index == trigger_batch,
                        "final_actor_exact_kl": float(exact[ix].mean()),
                        "final_actor_estimated_kl": float(estimate[ix].mean()),
                    })
                for stage in range(4):
                    ix = batch["memory"][:, 0] == stage
                    if not bool(ix.any()):
                        continue
                    stages.append({
                        **common_fields, "stage": ("straight", "turn", "door", "stop")[stage],
                        "samples": int(ix.sum()), "exact_kl_mean": float(exact[ix].mean()),
                        "exact_kl_p95": float(torch.quantile(exact[ix], .95)),
                        "estimated_kl": float(estimate[ix].mean()),
                        "mean_component": float(mean_part[ix].sum(-1).mean()),
                        "std_component": float(std_part[ix].sum(-1).mean()),
                        "normalized_mean_shift_rms": float(normalized_delta[ix].square().mean().sqrt()),
                        "trigger_samples": int((batch["memory"][trigger, 0] == stage).sum()) if trigger is not None else 0,
                    })
                for joint in range(12):
                    joints.append({
                        **common_fields, "action_index": joint,
                        "mean_kl": float(mean_part[:, joint].mean()),
                        "std_kl": float(std_part[:, joint].mean()),
                        "raw_mean_shift_rms": float(delta[:, joint].square().mean().sqrt()),
                        "shift_rms_in_old_std": float(normalized_delta[:, joint].square().mean().sqrt()),
                        "old_std": float(ls0[joint].exp()), "new_std": float(ls1[joint].exp()),
                    })
                for name, _ in pre_actor.named_parameters():
                    a, b = before["actor"][name].double(), after["actor"][name].double()
                    change = b - a
                    parameters.append({
                        **common_fields, "parameter": name, "elements": change.numel(),
                        "delta_l2": float(change.norm()), "delta_rms": float(change.square().mean().sqrt()),
                        "delta_abs_max": float(change.abs().max()),
                        "fraction_abs_delta_above_half_lr": float(
                            (change.abs() > .5 * ppo["actor_lr"]).double().mean()),
                    })
                if update_count == 1:
                    # Algebraically inspect the already saved Adam moments;
                    # do not instantiate an optimizer or apply any update.
                    group = after["actor_optimizer"]["param_groups"][0]
                    beta1, beta2 = group["betas"]
                    for (name, _), identifier in zip(pre_actor.named_parameters(), group["params"]):
                        moment = after["actor_optimizer"]["state"][identifier]
                        step = float(moment["step"])
                        predicted_change = (
                            -group["lr"] * moment["exp_avg"].double() / (1 - beta1 ** step)
                            / ((moment["exp_avg_sq"].double() / (1 - beta2 ** step)).sqrt()
                               + group["eps"]))
                        actual_change = after["actor"][name].double() - before["actor"][name].double()
                        adam_checks.append({
                            **common_fields, "parameter": name, "adam_step": step,
                            "saved_moment_delta_l2": float(predicted_change.norm()),
                            "observed_delta_l2": float(actual_change.norm()),
                            "max_delta_residual": float((actual_change - predicted_change).abs().max()),
                            "interpretation": "saved-moment arithmetic only; no optimizer step",
                        })
                source(batch_path); source(pre_path); source(post_path); source(folder / "record.json")
                if number == 1:
                    round1_means[(seed, arm)] = (old_mean.clone(), new_mean.clone())
                    digest = hashlib.sha256()
                    for field in ("state", "memory", "context", "command", "raw_action", "old_mean",
                                  "old_log_prob", "reward", "next_value", "terminated", "truncated"):
                        digest.update(field.encode())
                        digest.update(batch[field].contiguous().numpy().tobytes())
                    first_batch_fingerprints[(seed, arm)] = digest.hexdigest()
                rows.append(row)
                round_records[(seed, arm, number)] = row
                print(json.dumps({**common_fields, "updates": update_count,
                                  "analytic_kl": row["all_exact_kl_mean"],
                                  "trigger_kl": row.get("trigger_estimated_kl")}), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "round_kl.csv", rows)
    write_csv(args.output / "stage_kl.csv", stages)
    write_csv(args.output / "joint_kl.csv", joints)
    write_csv(args.output / "parameter_changes.csv", parameters)
    write_csv(args.output / "saved_adam_delta_check.csv", adam_checks)
    write_csv(args.output / "final_actor_minibatch_kl.csv", minibatches)
    pairs = []
    for seed in config["seeds"]:
        for number in range(1, config["rounds"] + 1):
            ordinary = round_records[(seed, "ordinary", number)]
            adjoint = round_records[(seed, "adjoint", number)]
            pair = {
                "seed": seed, "round": number,
                "same_behavior_batch_verified": (
                    first_batch_fingerprints[(seed, "ordinary")] == first_batch_fingerprints[(seed, "adjoint")]
                    if number == 1 else False),
                "comparison": "same initialization and same batch" if number == 1 else "different learned policies and visited states; descriptive only",
                "ordinary_updates": ordinary["actor_updates"], "adjoint_updates": adjoint["actor_updates"],
                "ordinary_exact_kl": ordinary["all_exact_kl_mean"],
                "adjoint_exact_kl": adjoint["all_exact_kl_mean"],
                "ordinary_trigger_estimated_kl": ordinary["trigger_estimated_kl"],
                "adjoint_trigger_estimated_kl": adjoint["trigger_estimated_kl"],
                "post_mean_difference_max": None,
                "post_mean_difference_rms": None,
                "ordinary_old_to_post_mean_rms": None,
                "adjoint_old_to_post_mean_rms": None,
                "post_difference_over_ordinary_change": None,
                "post_difference_over_adjoint_change": None,
            }
            if number == 1:
                old_o, post_o = round1_means[(seed, "ordinary")]
                old_a, post_a = round1_means[(seed, "adjoint")]
                diff = (post_o - post_a).double()
                rms = float(diff.square().mean().sqrt())
                rms_o = float((post_o - old_o).double().square().mean().sqrt())
                rms_a = float((post_a - old_a).double().square().mean().sqrt())
                pair.update(
                    post_mean_difference_max=float(diff.abs().max()),
                    post_mean_difference_rms=rms,
                    ordinary_old_to_post_mean_rms=rms_o,
                    adjoint_old_to_post_mean_rms=rms_a,
                    post_difference_over_ordinary_change=rms / rms_o,
                    post_difference_over_adjoint_change=rms / rms_a,
                )
            pairs.append(pair)
    write_csv(args.output / "paired_arms.csv", pairs)
    summary = {
        "schema": 1, "status": "READ_ONLY_SAVED_BATCH_KL_DIAGNOSIS",
        "saved_run_config_file_sha256": sha(args.results / "config.json"),
        "fixed_source_configuration_sha256": sha(ROOT / "configs/go2_adjoint_learning.json"),
        "saved_config_semantically_equals_fixed_source": config == json.loads(
            (ROOT / "configs/go2_adjoint_learning.json").read_text(encoding="utf-8")),
        "config_hash_note": "The runner reserializes its copy of config.json; file bytes differ from the fixed source while parsed configuration is identical.",
        "rounds": len(rows), "one_update_rounds": sum(r["actor_updates"] == 1 for r in rows),
        "first_rejection_reconstruction_scope": "22 one-update rounds; two two-update rounds use post actor at the inferred final stop location without recreating intermediate actors",
        "early_stopped_rounds": sum(r["kl_early_stop"] for r in rows),
        "all_batch_exact_kl_above_threshold": sum(r["all_exact_kl_mean"] > target for r in rows),
        "trigger_exact_kl_above_threshold": sum(r["trigger_exact_kl"] > target for r in rows),
        "trigger_estimated_kl_above_threshold": sum(r["trigger_estimated_kl"] > target for r in rows),
        "all_batch_exact_kl_range": [min(r["all_exact_kl_mean"] for r in rows), max(r["all_exact_kl_mean"] for r in rows)],
        "trigger_estimated_kl_range": [min(r["trigger_estimated_kl"] for r in rows), max(r["trigger_estimated_kl"] for r in rows)],
        "max_std_fraction": max(r["std_fraction"] for r in rows),
        "max_stored_mean_old_logp_error": max(r["stored_mean_old_logp_max_error"] for r in rows),
        "max_recomputed_old_mean_error": max(r["recomputed_old_mean_max_error"] for r in rows),
        "max_recomputed_old_logp_error": max(r["recomputed_old_logp_max_error"] for r in rows),
        "max_last_metric_reconstruction_error": max(r["last_metric_reconstruction_error"] for r in rows),
        "max_saved_adam_delta_rounding_residual": max(r["max_delta_residual"] for r in adam_checks),
        "actual_actor_updates_total": sum(r["actor_updates"] for r in rows),
        "available_minibatch_slots_total": len(minibatches),
        "round1_same_batch_each_seed": {str(seed): first_batch_fingerprints[(seed, "ordinary")] == first_batch_fingerprints[(seed, "adjoint")] for seed in config["seeds"]},
        "arm_summaries": {
            arm: {
                "rounds": 12,
                "one_update_rounds": sum(r["actor_updates"] == 1 for r in rows if r["arm"] == arm),
                "analytic_kl_mean": float(np.mean([r["all_exact_kl_mean"] for r in rows if r["arm"] == arm])),
                "analytic_kl_median": float(np.median([r["all_exact_kl_mean"] for r in rows if r["arm"] == arm])),
                "normalized_mean_shift_rms_median": float(np.median([r["mean_shift_rms_in_old_std"] for r in rows if r["arm"] == arm])),
            } for arm in ("ordinary", "adjoint")
        },
        "interpretation_limits": [
            "Analytic KL is old raw Gaussian to final raw Gaussian, averaged over saved old-policy states; it is not the KL of clipped physical actions.",
            "The first rejected minibatch is reconstructed in 22 one-update rounds using the original per-branch generator. In two-update rounds only the final stop location is inferred from the update count and re-evaluated with the saved final actor; intermediate actors and the full earlier check sequence are not reconstructed.",
            "In 22 one-update rounds the final actor directly exposes the one actual update. Two-update rounds expose only their cumulative change; intermediate policy weights were not saved or recreated.",
            "Round1 arm pairs share data and initialization. Later arm differences also include different learned policies, optimizer histories and visited states.",
            "This diagnoses the actual fixed run. It does not test alternative learning rates, standard deviations, rewards, normalization or KL thresholds.",
        ],
        "additional_physics_steps": 0, "optimizer_steps": 0, "model_fits": 0,
        "script_sha256": sha(__file__),
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "input_manifest.json", {
        "path_base": "preserved clean-rerun results", "files_sha256": sources,
        "read_only": True, "additional_physics_steps": 0, "optimizer_steps": 0,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
