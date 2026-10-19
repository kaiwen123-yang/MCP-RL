"""Pinned rl_sar TorchScript observation/history/action adapter.

Policy order is obtained from base.yaml joint_names and the policy's
joint_mapping; MuJoCo array order is deliberately not assumed.  This module
contains no optimizer, SDK, ROS, DDS, or hardware imports.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import yaml


SDK_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
                        for joint in ("hip", "thigh", "calf"))


def _yaml_mapping(path):
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if len(document) == 1 and isinstance(next(iter(document.values())), dict):
        return next(iter(document.values()))
    return document


def load_policy_config(policy_path, policy_config=None):
    path = Path(policy_path).resolve()
    base = path.parent.parent / "base.yaml"
    cfg = _yaml_mapping(base) if base.exists() else {
        "dt": .005, "decimation": 4, "joint_names": list(SDK_JOINT_NAMES)}
    if isinstance(policy_config, dict):
        supplied = copy.deepcopy(policy_config)
        if len(supplied) == 1 and isinstance(next(iter(supplied.values())), dict):
            supplied = next(iter(supplied.values()))
    else:
        supplied = _yaml_mapping(policy_config or path.with_name("config.yaml"))
    cfg.update(supplied)
    if cfg["num_of_dofs"] != 12 or cfg["num_observations"] != 45:
        raise ValueError("This adapter supports the inspected 12-joint/45-observation Go2 policies")
    if cfg.get("observations_history_priority", "time") != "time":
        raise ValueError("The inspected Go2 checkpoints use time-major history")
    mapping = cfg["joint_mapping"]
    if sorted(mapping) != list(range(12)):
        raise ValueError("joint_mapping must be a permutation of the twelve SDK joints")
    cfg["policy_joint_names"] = [cfg["joint_names"][index] for index in mapping]
    return cfg


class FrozenGo2Policy:
    """Stateless TorchScript network plus explicitly owned recurrent inputs.

    HIM's estimator+actor are both inside the exported forward; its six-frame
    observation history and previous action are maintained here.  History starts
    with zeros, matching rl_sar ObservationBuffer construction, not six copies of
    an invented standing observation.
    """

    def __init__(self, policy_path, policy_config=None, device="cpu"):
        import torch
        self.path = Path(policy_path).resolve()
        self.config = load_policy_config(self.path, policy_config)
        self.device = str(device)
        self.module = torch.jit.load(str(self.path), map_location=self.device).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.history_indices = tuple(self.config.get("observations_history", []))
        self.history_length = max(self.history_indices) + 1 if self.history_indices else 1
        self.joint_names = tuple(self.config["policy_joint_names"])
        self.default_position = np.asarray(self.config["default_dof_pos"], dtype=np.float64)
        self.action_scale = np.asarray(self.config["action_scale"], dtype=np.float64)
        self.kp = np.asarray(self.config["rl_kp"], dtype=np.float64)
        self.kd = np.asarray(self.config["rl_kd"], dtype=np.float64)
        self.inference_count = 0
        self.reset()

    @property
    def control_dt(self):
        return float(self.config["dt"]) * int(self.config["decimation"])

    def reset(self):
        self.history = np.zeros((self.history_length, 45), dtype=np.float32)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.last_raw_action = np.zeros(12, dtype=np.float32)
        self.last_observation = np.zeros(45, dtype=np.float32)
        self.last_input = np.zeros(45 * (len(self.history_indices) or 1), dtype=np.float32)
        self.last_target = self.default_position.copy()

    def infer(self, command, angular_velocity_body, projected_gravity,
              joint_position, joint_velocity):
        import torch
        cfg = self.config
        terms = {
            "commands": np.asarray(command) * np.asarray(cfg["commands_scale"]),
            "ang_vel": np.asarray(angular_velocity_body) * cfg["ang_vel_scale"],
            "gravity_vec": np.asarray(projected_gravity),
            "dof_pos": (np.asarray(joint_position) - self.default_position) * cfg["dof_pos_scale"],
            "dof_vel": np.asarray(joint_velocity) * cfg["dof_vel_scale"],
            "actions": self.previous_action,
        }
        observation = np.concatenate([terms[name] for name in cfg["observations"]]).astype(np.float32)
        observation = np.clip(observation, -cfg["clip_obs"], cfg["clip_obs"])
        self.history[1:] = self.history[:-1].copy()
        self.history[0] = observation
        network_input = (self.history[list(self.history_indices)].reshape(-1)
                         if self.history_indices else observation).copy()
        with torch.inference_mode():
            output = self.module(torch.from_numpy(network_input[None]).to(self.device))
        action = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
        self.inference_count += 1
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError("Frozen policy returned a nonfinite or wrongly shaped action")
        self.last_raw_action = action.copy()
        action = np.clip(action, np.asarray(cfg["clip_actions_lower"], dtype=np.float32),
                         np.asarray(cfg["clip_actions_upper"], dtype=np.float32))
        self.previous_action = action.copy()
        self.last_observation = observation.copy()
        self.last_input = network_input
        self.last_target = self.default_position + self.action_scale * action
        return self.last_target.copy()

    def snapshot(self):
        return {name: getattr(self, name).copy() for name in
                ("history", "previous_action", "last_raw_action", "last_observation", "last_input", "last_target")}

    def restore(self, state):
        for name, value in state.items():
            setattr(self, name, np.asarray(value).copy())

    def clone(self):
        # The inspected export is a feed-forward estimator+actor; only these
        # explicit observation arrays carry history. Shared weights remain frozen.
        result = copy.copy(self)
        result.inference_count = 0
        result.restore(self.snapshot())
        return result

    def provenance(self):
        return {"sha256": self.sha256, "configuration": copy.deepcopy(self.config),
                "joint_order": list(self.joint_names), "policy_period_seconds": self.control_dt,
                "history_order": "newest_first_time_major", "startup_history": "zeros",
                "history_frames": self.history_length, "weights_frozen": True,
                "training": False, "hardware_calls": False}
