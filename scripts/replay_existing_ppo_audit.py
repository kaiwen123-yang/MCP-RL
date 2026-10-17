"""Replay exactly the saved first-short deterministic PPO evaluations.

This runs fresh simulations of an existing fixed test and records native
observations. It is not training, old training-batch recovery, or a zero-
simulation audit. No PPOBackend, optimizer, critic or checkpoint write is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from response_ppo.env import EnvironmentConfig, ResponseServo
from response_ppo.ppo_backend import GaussianActor, PPOConfig

OUT = ROOT / "notes/neural-demand-audit-2026-10-17"
WORK = ROOT / "work/neural-demand-audit-2026-10-17/replays"
PROTOCOL = OUT / "replay-protocol.md"
PROTOCOL_SHA256 = "5aa134ab63ba113567ecc8e850fdd662e3f3085d0ff0db8661556473df654a99"
SEEDS = (17, 29, 43)
ARMS = ("joint_support", "fixed_support", "joint_penalty", "fixed_penalty")
STEPS, ENVIRONMENTS = 256, 16


def hashes(path):
    data = path.read_bytes()
    return {"sha256_bytes": hashlib.sha256(data).hexdigest(),
            "sha256_canonical_lf": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def keep(storage, **values):
    for key, value in values.items():
        storage.setdefault(key, []).append(value.detach().cpu().clone())


def run_case(seed, arm):
    base = ROOT / f"work/ppo-mechanism-2026-10-16/formal/seedrun-{seed}"
    folder = base / f"seed-{seed}"
    checkpoint_path = folder / f"{arm}.pt"
    original_path = folder / f"{arm}-short-trace.npz"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    original_run = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    source_checks = []
    for name, expected in manifest["source_hashes_before_run"].items():
        actual = hashes(ROOT / name)
        canonical = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        equivalent_hashes = {actual['sha256_bytes'], actual['sha256_canonical_lf'],
                             hashlib.sha256(canonical.replace(b"\n", b"\r\n")).hexdigest()}
        source_checks.append({"path": name, "recorded_sha256_bytes": expected, **actual,
                              "bytes_match_recorded": actual["sha256_bytes"] == expected,
                              "source_matches_up_to_line_endings": expected in equivalent_hashes})
    if not all(row["source_matches_up_to_line_endings"] for row in source_checks):
        return {"seed": seed, "arm": arm, "status": "source_hash_mismatch_not_run",
                "source_checks": source_checks, "budget": {"simulated_physical_transitions": 0,
                "planner_instances": 0, "candidate_sequences": 0, "candidate_macro_steps": 0,
                "candidate_low_level_steps": 0, "training_updates": 0}}
    before_hash = hashes(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor = GaussianActor(PPOConfig(**original_run["ppo"]))
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()
    config = EnvironmentConfig(**original_run["environment"])
    assert config.device == "cpu"
    assert all(p.dtype == torch.float32 for p in actor.parameters())
    design = checkpoint["design_mean"][None].expand(ENVIRONMENTS, -1)
    with np.load(original_path, allow_pickle=False) as old:
        original = {key: old[key].copy() for key in old.files}
    assert len(original) == 15
    assert all(value.shape == (STEPS, ENVIRONMENTS) for value in original.values())

    started = time.perf_counter()
    evaluation_seed = seed * 10000 + 5000
    env = ResponseServo(config, design, evaluation_seed, STEPS, support=arm.endswith("support"))
    obs = env.observation()
    records, metric_records = {}, {}
    comparison = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                       "numeric_exact": True, "bytes_exact": True, "max_abs_error": 0.}
                  for key, value in original.items()}
    first_mismatch = None
    completed = 0
    with torch.no_grad():
        for step in range(STEPS):
            znext, uref, interval = env.interval()
            (mean, _), _ = actor(obs)
            keep(records, obs=obs, x=env.x, z=env.z, command=env.command,
                 raw_design=env.raw_design, interval_lo=interval.lo, interval_hi=interval.hi,
                 interval_feasible=interval.feasible, interval_discriminant=interval.discriminant,
                 znext=znext, uref=uref, mean=mean)
            next_obs, _, metric = env.step(torch.tanh(mean[:, 0]))
            keep(metric_records, **metric)
            completed += 1
            for key, expected in original.items():
                actual = metric[key].detach().cpu().numpy()
                wanted = expected[step]
                identical = actual.shape == wanted.shape and actual.dtype == wanted.dtype and np.array_equal(actual, wanted, equal_nan=True)
                bytes_identical = actual.tobytes() == wanted.tobytes()
                error = float(np.max(np.abs(actual.astype(np.float64) - wanted.astype(np.float64))))
                comparison[key]["numeric_exact"] &= bool(identical)
                comparison[key]["bytes_exact"] &= bool(bytes_identical)
                comparison[key]["max_abs_error"] = max(comparison[key]["max_abs_error"], error)
                if not identical and first_mismatch is None:
                    first_mismatch = {"step_zero_based": step, "field": key,
                                      "actual": actual.tolist(), "expected": wanted.tolist(),
                                      "max_abs_error": error}
            obs = next_obs
            if first_mismatch is not None:
                break
    wall = time.perf_counter() - started
    arrays = {key: torch.stack(value).numpy() for key, value in records.items()}
    arrays.update({key: torch.stack(value).numpy() for key, value in metric_records.items()})
    arrays.update(step_index=np.arange(completed, dtype=np.int64),
                  terminal_x=env.x.detach().cpu().numpy(), terminal_z=env.z.detach().cpu().numpy(),
                  P=env.P.detach().cpu().numpy(), B=env.B.detach().cpu().numpy(), E=env.E.detach().cpu().numpy(),
                  actor_log_std=actor.log_std.detach().cpu().numpy(),
                  evaluation_seed=np.array(evaluation_seed, dtype=np.int64))
    destination = WORK / f"{seed}-{arm}.npz"
    np.savez_compressed(destination, **arrays)
    after_hash = hashes(checkpoint_path)
    tensor_unchanged = all(torch.equal(value, checkpoint["actor"][key]) for key, value in actor.state_dict().items())
    assert before_hash["sha256_bytes"] == after_hash["sha256_bytes"] and tensor_unchanged
    counts = env.planner.objective_counts()
    return {"seed": seed, "arm": arm, "status": "exact_replay_pass" if first_mismatch is None and completed == STEPS else "metric_mismatch_stopped",
            "evaluation_seed": evaluation_seed, "completed_steps": completed, "environments": ENVIRONMENTS,
            "source_checks": source_checks,
            "checkpoint": {"path": relative(checkpoint_path), "before": before_hash, "after": after_hash,
                           "bytes_unchanged": before_hash == after_hash, "actor_tensors_unchanged": tensor_unchanged},
            "original_trace": {"path": relative(original_path), **hashes(original_path)},
            "reconstructed_trace": {"path": relative(destination), **hashes(destination),
                                    "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
                                    "identity": "New deterministic resimulation of the fixed existing evaluation, with new complete native observations; not original logged observations or training batch"},
            "metric_comparison": comparison, "first_mismatch": first_mismatch,
            "wall_seconds": wall, "planner_batch_calls": env.planner.n_plans,
            "budget": {"simulated_physical_transitions": completed * ENVIRONMENTS,
                       "planner_instances": env.planner_instances, "objective_calls": counts["objective_calls"],
                       "candidate_sequences": counts["candidate_sequences"],
                       "candidate_macro_steps": counts["candidate_macro_steps"],
                       "candidate_low_level_steps": counts["candidate_macro_steps"] * config.command_steps,
                       "training_updates": 0, "actor_weight_updates": 0, "gradient_optimizer_steps": 0,
                       "checkpoint_writes": 0}}


def write_outputs(results, finished):
    totals = {}
    for result in results:
        for key, value in result["budget"].items():
            totals[key] = totals.get(key, 0) + value
    report = {"date": "2026-10-17", "status": "completed" if finished else "running",
              "scope": "Fixed first-short deterministic PPO evaluation resimulation; no training or weight updates",
              "protocol": {"path": relative(PROTOCOL), **hashes(PROTOCOL)},
              "script": {"path": relative(Path(__file__).resolve()), **hashes(Path(__file__).resolve())},
              "planned_cases": 12, "recorded_cases": len(results),
              "passed_cases": sum(r["status"] == "exact_replay_pass" for r in results),
              "all_cases_passed": finished and len(results) == 12 and all(r["status"] == "exact_replay_pass" for r in results),
              "cases": results, "budget": totals,
              "fingerprint_convention": "Byte fingerprints and LF content fingerprints are both retained; source guards accept only exact bytes or LF/CRLF-equivalent source text.",
              "boundary": "The complete native observations were newly reconstructed by deterministic simulation and checked against all old metrics. This does not recover the missing stochastic PPO training batch, old advantages or raw sampled actions."}
    (OUT / "replay-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    rows = []
    for r in results:
        maxerr = max((v["max_abs_error"] for v in r.get("metric_comparison", {}).values()), default=float('nan'))
        rows.append(f"| {r['seed']} | {r['arm']} | {r['status']} | {r.get('completed_steps',0)} | {maxerr:.3g} | {r.get('wall_seconds',0):.3f} |")
    text = f"""# 已有PPO短测的确定性重放结果

日期：2026-10-17（Asia/Shanghai）。[冻结协议](replay-protocol.md)、[逐字段证据](replay-results.json)、[脚本](../../scripts/replay_existing_ppo_audit.py)。当前状态：`{report['status']}`；已记录{len(results)}/12组，完全一致通过{report['passed_cases']}组。全部完成且通过：`{report['all_cases_passed']}`。

**这是本轮重新执行原有固定评估的仿真，不是零仿真读取，也不是旧随机训练batch恢复。**原PPO网络、design、seed、16env、256step、原环境及原MPC全部保留；逐step核对旧NPZ的全部15项metric，首次不一致即停止该case。没有更换模型、调参数或根据结果选组。

| seed | arm | 状态 | 已执行step/env | 15metric最大差 | 仿真墙钟秒 |
|---:|---|---|---:|---:|---:|
{chr(10).join(rows)}

完整重建观测保存在`work/neural-demand-audit-2026-10-17/replays/{{seed}}-{{arm}}.npz`。新增字段包含step前obs/x/z/command/raw_design、支持区间及discriminant、znext/uref和网络mean；原15metric保持原字段名。末态terminal_x/z可与step前状态移位组合得到next序列。每组实际shape、dtype、字节hash和规范化hash分别列于JSON。

本次执行时源文件字节hash与正式manifest完全匹配。脚本额外支持仅LF/CRLF换行不同的同一源文本，并分别记录字节匹配与内容等价，不允许其它源码改动。所有已经执行的checkpoint在前后字节一致，actor张量未变；仅实例化GaussianActor，不创建PPOBackend、optimizer或critic。物理状态由原离散servo重新推进，原MPC在线规划，所以实际仿真与预测成本必须记账。

本次累计实际预算：仿真转移`{totals.get('simulated_physical_transitions',0)}`，planner实例`{totals.get('planner_instances',0)}`，候选序列`{totals.get('candidate_sequences',0)}`，预测微步`{totals.get('candidate_low_level_steps',0)}`；训练更新、actor权重更新、梯度优化器步及checkpoint写入均为0。结果中原始观察缺失的事实不变：本轮新记录不能被称作原本已保存的随机PPO数据或真实训练优势。
"""
    (OUT/'replay-results.md').write_text(text, encoding='utf-8')


def main():
    assert hashes(PROTOCOL)["sha256_canonical_lf"] == PROTOCOL_SHA256
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in SEEDS:
        for arm in ARMS:
            result = run_case(seed, arm)
            results.append(result)
            write_outputs(results, finished=False)
            print(json.dumps({"seed": seed, "arm": arm, "status": result["status"],
                              "completed_steps": result.get('completed_steps',0),
                              "wall_seconds": result.get('wall_seconds',0),
                              "simulated_physical_transitions": result['budget']['simulated_physical_transitions']}, ensure_ascii=False), flush=True)
    write_outputs(results, finished=True)
    print(json.dumps({"completed": len(results), "passed": sum(r['status']=='exact_replay_pass' for r in results),
                      "simulated_physical_transitions": sum(r['budget']['simulated_physical_transitions'] for r in results),
                      "training_updates": 0}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
