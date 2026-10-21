"""Frozen public Go2 locomotion in full MuJoCo contact dynamics.

This adapter has no trainer, optimizer, networking or hardware entry point.
The observation/action contract follows fan-ziqi/rl_sar's Go2 deployment
configuration. Robot inertias, contacts and motor limits come from rl_sar_zoo.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quat_wxyz)
    return matrix.reshape(3, 3)


def _rpy(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ])


class FrozenPolicy:
    """Load an exported policy and preserve its original deployment scaling."""

    def __init__(self, policy_root: str | Path, policy_name: str = "robot_lab"):
        self.root = Path(policy_root).resolve()
        self.name = policy_name
        self.config_path = self.root / policy_name / "config.yaml"
        self.base_path = self.root / "base.yaml"
        base = yaml.safe_load(self.base_path.read_text(encoding="utf-8"))["go2"]
        specific = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))[
            f"go2/{policy_name}"
        ]
        self.config = {**base, **specific}
        self.weight_path = self.root / policy_name / self.config["model_name"]
        self.weight_sha256_before = _sha256(self.weight_path)
        torch.set_num_threads(1)
        self.model = torch.jit.load(str(self.weight_path), map_location="cpu")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tensor_sha256_before = self.tensor_hash()
        self.default_q = np.asarray(self.config["default_dof_pos"], dtype=np.float64)
        self.action_scale = np.asarray(self.config["action_scale"], dtype=np.float64)
        self.kp = np.asarray(self.config["rl_kp"], dtype=np.float64)
        self.kd = np.asarray(self.config["rl_kd"], dtype=np.float64)
        self.fixed_kp = np.asarray(self.config["fixed_kp"], dtype=np.float64)
        self.fixed_kd = np.asarray(self.config["fixed_kd"], dtype=np.float64)
        sdk_names = base["joint_names"]
        self.joint_names = [sdk_names[i] for i in self.config["joint_mapping"]]
        self.interval = float(base["dt"]) * int(base["decimation"])
        self.calls = 0
        self.reset()

    def tensor_hash(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def reset(self) -> None:
        self.last_action = np.zeros(12, dtype=np.float64)
        history_ids = self.config.get("observations_history", [])
        self.history = np.zeros((max(history_ids, default=0) + 1, 45), dtype=np.float32)

    def infer(self, quat: np.ndarray, omega: np.ndarray, command: np.ndarray,
              q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        cfg = self.config
        terms = {
            "ang_vel": omega * cfg["ang_vel_scale"],
            "gravity_vec": _rotation(quat).T @ np.array([0.0, 0.0, -1.0]),
            "commands": command * np.asarray(cfg["commands_scale"]),
            "dof_pos": (q - self.default_q) * cfg["dof_pos_scale"],
            "dof_vel": dq * cfg["dof_vel_scale"],
            "actions": self.last_action,
        }
        observation = np.concatenate([terms[name] for name in cfg["observations"]])
        observation = np.clip(observation, -cfg["clip_obs"], cfg["clip_obs"]).astype(np.float32)
        history_ids = cfg.get("observations_history", [])
        if history_ids:
            if cfg.get("observations_history_priority", "time") != "time":
                raise ValueError("This adapter supports the public Go2 time-major history only")
            self.history[1:] = self.history[:-1].copy()
            self.history[0] = observation
            observation = self.history[history_ids].reshape(-1)
        with torch.inference_mode():
            action = self.model(torch.from_numpy(observation).unsqueeze(0))
        self.last_action = np.clip(
            action.detach().cpu().numpy().reshape(-1),
            cfg["clip_actions_lower"], cfg["clip_actions_upper"],
        ).astype(np.float64)
        self.calls += 1
        return self.last_action.copy()

    def provenance(self) -> dict[str, Any]:
        return {
            "policy_name": self.name,
            "weight_file": self.weight_path.name,
            "weight_sha256_before": self.weight_sha256_before,
            "weight_sha256_after": _sha256(self.weight_path),
            "tensor_sha256_before": self.tensor_sha256_before,
            "tensor_sha256_after": self.tensor_hash(),
            "config_sha256": _sha256(self.config_path),
            "base_config_sha256": _sha256(self.base_path),
            "inference_calls": self.calls,
            "training": self.model.training,
            "any_parameter_requires_grad": any(p.requires_grad for p in self.model.parameters()),
            "optimizer_updates": 0,
            "joint_names": self.joint_names,
            "observation_terms": self.config["observations"],
        }


class Go2Sim:
    """One robot, no resets between task stages, frozen 50 Hz policy."""

    def __init__(self, policy_root: str | Path, model_root: str | Path,
                 output_dir: str | Path, obstacles: list[dict] | None = None,
                 policy_name: str = "robot_lab", dt: float = 0.005):
        self.policy = FrozenPolicy(policy_root, policy_name)
        self.model_root = Path(model_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.control_interval = float(dt)
        self.policy_interval = self.policy.interval
        self.physics_interval = self.control_interval / 5.0
        self.controls_per_policy = int(round(self.policy_interval / self.control_interval))
        if not np.isclose(self.controls_per_policy * self.control_interval, self.policy_interval):
            raise ValueError("PD period must divide the policy interval")
        self.obstacles = list(obstacles or [])
        self.robot_path = self.model_root / "go2.xml"
        xml_root = ET.parse(self.robot_path).getroot()
        xml_root.find("compiler").set("meshdir", (self.model_root / "assets").as_posix())
        xml_root.find("option").set("timestep", str(self.physics_interval))
        visual = ET.SubElement(xml_root, "visual")
        ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.4 0.4 0.4")
        ET.SubElement(visual, "global", azimuth="135", elevation="-25", offwidth="1280", offheight="720")
        world = xml_root.find("worldbody")
        ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true")
        ET.SubElement(world, "geom", name="floor", size="0 0 0.05", type="plane", rgba="0.8 0.82 0.85 1")
        self.obstacle_names = []
        for index, obstacle in enumerate(self.obstacles):
            name = obstacle.get("name", f"obstacle_{index}")
            self.obstacle_names.append(name)
            ET.SubElement(world, "geom", name=name, type="box",
                          pos=" ".join(map(str, obstacle["pos"])),
                          size=" ".join(map(str, obstacle["size"])),
                          rgba="0.45 0.35 0.25 1")
        self.scene_path = self.output_dir / "scene.xml"
        ET.ElementTree(xml_root).write(self.scene_path, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.joint_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                   for name in self.policy.joint_names])
        self.q_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.v_addresses = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([np.flatnonzero(self.model.actuator_trnid[:, 0] == joint)[0]
                                     for joint in self.joint_ids])
        self.foot_geom_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, leg)
                                      for leg in ["FR", "FL", "RR", "RL"]])
        self.obstacle_geom_ids = set(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                                     for name in self.obstacle_names)
        self.robot_geom_ids = set(np.flatnonzero(self.model.geom_bodyid > 0))
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.gyro_address = int(self.model.sensor_adr[gyro_id])
        self.task_start_time = 0.0
        self.command = np.zeros(3)
        self.last_tau_command = np.zeros(12)
        self.q_des = self.policy.default_q.copy()
        self.warmup_trace: list[dict] = []
        self.obstacle_contact_ever = False
        self.fallen_ever = False
        self.physics_steps = 0

    def _pd_step(self, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        q = self.data.qpos[self.q_addresses]
        dq = self.data.qvel[self.v_addresses]
        # Match RL_Sim::SetCommand: no YAML torque clip; MJCF motor ctrlrange
        # supplies the physical actuator saturation. Recompute PD at 200 Hz.
        self.last_tau_command = kp * (q_des - q) - kd * dq
        self.data.ctrl[self.actuator_ids] = self.last_tau_command
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            self._update_contact_flags()

    def _update_contact_flags(self) -> None:
        for contact in self.data.contact[:self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.obstacle_geom_ids and pair & self.robot_geom_ids:
                self.obstacle_contact_ever = True
        quat = self.data.qpos[3:7]
        tilt_cosine = _rotation(quat)[2, 2]
        if self.data.qpos[2] < 0.16 or tilt_cosine < 0.5:
            self.fallen_ever = True

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
              warmup_seconds: float = 2.0) -> dict[str, Any]:
        mujoco.mj_resetData(self.model, self.data)
        self.policy.reset()
        self.data.qpos[:3] = [x, y, 0.34]
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.data.qpos[self.q_addresses] = self.policy.default_q
        self.data.qvel[:] = 0.0
        self.q_des = self.policy.default_q.copy()
        self.command = np.zeros(3)
        self.obstacle_contact_ever = False
        self.fallen_ever = False
        self.physics_steps = 0
        self.task_start_time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.warmup_trace = [self.observe()]
        # The public deployment enters locomotion after standing. Use fixed
        # gains for the first half, then run the unchanged policy at zero command.
        fixed_time = warmup_seconds / 2
        for _ in range(round(fixed_time / self.control_interval)):
            self._pd_step(self.q_des, self.policy.fixed_kp, self.policy.fixed_kd)
            self.warmup_trace.append(self.observe())
        for _ in range(round((warmup_seconds - fixed_time) / self.policy_interval)):
            self.warmup_trace.append(self.step(np.zeros(3)))
        self.task_start_time = float(self.data.time)
        return self.observe()

    def step(self, command3: np.ndarray | list[float]) -> dict[str, Any]:
        self.command = np.asarray(command3, dtype=np.float64).reshape(3)
        state = self.observe()
        action = self.policy.infer(state["quat"], state["omega_body"], self.command,
                                   state["joint_pos"], state["joint_vel"])
        self.q_des = self.policy.default_q + self.policy.action_scale * action
        for _ in range(self.controls_per_policy):
            self._pd_step(self.q_des, self.policy.kp, self.policy.kd)
        return self.observe()

    def observe(self) -> dict[str, Any]:
        quat = self.data.qpos[3:7].copy()
        rotation = _rotation(quat)
        rpy = _rpy(quat)
        omega_body = self.data.sensordata[self.gyro_address:self.gyro_address + 3].copy()
        # ZYX Euler yaw derivative, distinct from gyro-z when tilted.
        yaw_rate = (math.sin(rpy[0]) * omega_body[1] + math.cos(rpy[0]) * omega_body[2]) / math.cos(rpy[1])
        velocity_world = self.data.qvel[:3].copy()
        foot_contact = np.zeros(4, dtype=bool)
        obstacle_contact = False
        for contact in self.data.contact[:self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            foot_contact |= np.array([int(gid) in pair for gid in self.foot_geom_ids])
            obstacle_contact |= bool(pair & self.obstacle_geom_ids and pair & self.robot_geom_ids)
        fallen = bool(self.data.qpos[2] < 0.16 or rotation[2, 2] < 0.5)
        return {
            "time": float(self.data.time - self.task_start_time),
            "simulation_time": float(self.data.time),
            "position": self.data.qpos[:3].copy(),
            "quat": quat,
            "rpy": rpy,
            "yaw": float(rpy[2]),
            "yaw_rate": float(yaw_rate),
            "velocity_world": velocity_world,
            "velocity_body": rotation.T @ velocity_world,
            "omega_body": omega_body,
            "joint_pos": self.data.qpos[self.q_addresses].copy(),
            "joint_vel": self.data.qvel[self.v_addresses].copy(),
            "torque": self.data.actuator_force[self.actuator_ids].copy(),
            "torque_command": self.last_tau_command.copy(),
            "action": self.policy.last_action.copy(),
            "command": self.command.copy(),
            "foot_contact": foot_contact,
            "obstacle_contact": bool(obstacle_contact),
            "obstacle_contact_ever": bool(self.obstacle_contact_ever),
            "fallen": fallen,
            "fallen_ever": bool(self.fallen_ever),
        }

    def provenance(self) -> dict[str, Any]:
        return {
            **self.policy.provenance(),
            "robot_mjcf_sha256": _sha256(self.robot_path),
            "physics_interval": self.physics_interval,
            "pd_interval": self.control_interval,
            "policy_interval": self.policy_interval,
            "physics_steps": self.physics_steps,
            "actuator_ctrlrange_policy_order": self.model.actuator_ctrlrange[self.actuator_ids].tolist(),
            "obstacles": self.obstacles,
            "mujoco_version": mujoco.__version__,
            "torch_version": torch.__version__,
            "warmup_actual_seconds": self.task_start_time,
            "fallen_ever": self.fallen_ever,
            "obstacle_contact_ever": self.obstacle_contact_ever,
        }
