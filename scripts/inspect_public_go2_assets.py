"""Read two pinned public policies, with exactly 12 synthetic CPU forwards.

This is an ABI/file check, never an environment or robot evaluation.
Download sources and protocol: notes/platform-assets-2026-10-17/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import torch
import yaml


FILES = {
    "robot_lab": ("policy.pt", "9f14cb95e74ac9e5e30954da0fc0c33eaa41e337852ebed19fcee869279ade0b"),
    "himloco": ("himloco.pt", "d9945f85a39944aaf241a67f7d2b25a15326a0222d0e503aeb05a7f188430509"),
}
COMMANDS = [("zero", [0., 0., 0.]), ("forward", [.2, 0., 0.]), ("yaw", [0., 0., .2])]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation(cfg, command):
    values = {
        "ang_vel": [0.] * 3,
        "gravity_vec": [0., 0., -1.],
        "commands": [a * b for a, b in zip(command, cfg["commands_scale"])],
        "dof_pos": [0.] * 12,
        "dof_vel": [0.] * 12,
        "actions": [0.] * 12,
    }
    frame = torch.tensor([x for key in cfg["observations"] for x in values[key]], dtype=torch.float32)
    frame = frame.clamp(-cfg["clip_obs"], cfg["clip_obs"])
    assert frame.numel() == cfg["num_observations"] == 45
    history = cfg["observations_history"]
    if history:
        assert history == list(range(6)) and cfg["observations_history_priority"] == "time"
        frame = frame.repeat(6)
    return frame.unsqueeze(0)


def run(root: Path):
    started = time.perf_counter()
    records = []
    forwards = 0
    for name, (filename, expected) in FILES.items():
        folder = root / "policy" / "go2" / name
        path, config_path = folder / filename, folder / "config.yaml"
        before = sha(path)
        assert before == expected, "Only the pinned, previously inspected weight is authorized by this protocol"
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))[f"go2/{name}"]
        with zipfile.ZipFile(path) as archive:
            code = {entry: hashlib.sha256(archive.read(entry)).hexdigest()
                    for entry in archive.namelist() if "/code/" in entry and entry.endswith(".py")}
        model = torch.jit.load(str(path), map_location="cpu").eval()
        record = {
            "name": name, "source_relative_path": f"policy/go2/{name}/{filename}",
            "bytes": path.stat().st_size, "sha256_before": before,
            "config_sha256": sha(config_path), "config": cfg, "archive_code_sha256": code,
            "parameter_shapes": {key: list(value.shape) for key, value in model.named_parameters()},
            "parameter_count": sum(value.numel() for value in model.parameters()),
            "torchscript_operators": torch.jit.export_opnames(model), "cases": [],
        }
        with torch.inference_mode():
            for label, command in COMMANDS:
                obs = observation(cfg, command)
                first, second = model(obs), model(obs)
                forwards += 2
                assert first.shape == (1, 12) and torch.isfinite(first).all() and torch.equal(first, second)
                action = first[0].clamp(torch.tensor(cfg["clip_actions_lower"]), torch.tensor(cfg["clip_actions_upper"]))
                target = torch.tensor(cfg["default_dof_pos"]) + action * torch.tensor(cfg["action_scale"])
                sdk_target = torch.empty_like(target)
                sdk_target[torch.tensor(cfg["joint_mapping"])] = target
                record["cases"].append({
                    "name": label, "unscaled_command": command, "input": obs[0].tolist(),
                    "input_shape": list(obs.shape), "output_shape": list(first.shape),
                    "finite": bool(torch.isfinite(first).all()), "repeat_exact": bool(torch.equal(first, second)),
                    "raw_action": first[0].tolist(), "q_target_policy_order": target.tolist(),
                    "q_target_sdk_order": sdk_target.tolist(), "sent_to_robot": False,
                })
        record["sha256_after"] = sha(path)
        assert record["sha256_after"] == before
        records.append(record)
    return {
        "evidence_level": "FIXED_SYNTHETIC_INPUT_INFERENCE_ONLY",
        "source_commit": "376d42c9b128f963ab08579762d5a216a976ce39", "torch_version": torch.__version__,
        "script_sha256": sha(Path(__file__)), "policy_forwards": forwards,
        "actor_forwards": forwards, "estimator_forwards": 6,
        "gradient_calls": 0, "optimizer_steps": 0, "environment_steps": 0,
        "hardware_commands": 0, "checkpoint_writes": 0,
        "wall_seconds_not_a_runtime_benchmark": time.perf_counter() - started,
        "policies": records,
        "limitation": "Synthetic inputs are not reachable histories; this does not establish feedback ability, task success, deployment compatibility, or training value.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"forwards": result["policy_forwards"], "models": len(result["policies"]),
                      "all_finite_repeat_exact": True, "wall_seconds": result["wall_seconds_not_a_runtime_benchmark"]}))
