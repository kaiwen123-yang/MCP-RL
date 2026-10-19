"""MuJoCo Go2 mechanics and frozen policy, with exact in-process branch copies.

The nominal scene is unitree_rl_mjlab's scene_go2.xml. rl_sar's MuJoCo
SetCommand writes position PD torque to motor ctrl; it does NOT consume the
separately computed/clipped torque queue. Consequently this adapter uses the
scene's motor ctrlrange, not the policy YAML's unconsumed torque_limits.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .policy import FrozenGo2Policy


@dataclass
class SimulationSnapshot:
    model: Any
    data: Any
    policy_state: dict
    ticks: int
    command: np.ndarray
    target_position: np.ndarray
    requested_torque: np.ndarray
    applied_torque: np.ndarray
    integrals: dict
    interval_contacts: dict


class Go2Simulation:
    """Headless, CPU MuJoCo body with 200 Hz PD and 50 Hz frozen policy.

    obstacles are world-fixed boxes: name, pos[3], size[3] (HALF extents).
    Optional quat is wxyz. No source XML is modified. clone() shares the static
    model and TorchScript weights but owns MjData, history, actions and clocks.
    Counters count actual computation and do not rewind on restore().
    """

    def __init__(self, model_path, policy_path, policy_config=None, *,
                 physics_dt=.005, obstacles=None, device="cpu"):
        import mujoco
        self.mj = mujoco
        self.model_path = Path(model_path).resolve()
        self.policy = FrozenGo2Policy(policy_path, policy_config, device=device)
        self.obstacles = copy.deepcopy(obstacles or [])
        if self.obstacles:
            root = ET.fromstring(self.model_path.read_text(encoding="utf-8"))
            compiler = root.find("compiler")
            if compiler is not None:
                for attr in ("meshdir", "texturedir"):
                    if attr in compiler.attrib:
                        compiler.set(attr, str((self.model_path.parent / compiler.get(attr)).resolve()))
            world = root.find("worldbody")
            if world is None:
                world = ET.SubElement(root, "worldbody")
            for index, box in enumerate(self.obstacles):
                attributes = {"name": box.get("name", f"task_box_{index}"), "type": "box",
                              "pos": self._xml_numbers(box["pos"]), "size": self._xml_numbers(box["size"]),
                              "contype": "1", "conaffinity": "1", "friction": "0.6 0.01 0.001",
                              "rgba": "0.6 0.25 0.12 1"}
                for key in ("quat", "friction", "rgba"):
                    if key in box:
                        attributes[key] = self._xml_numbers(box[key])
                ET.SubElement(world, "geom", attributes)
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        else:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.model.opt.timestep = float(physics_dt)
        self.dt = float(physics_dt)
        self.policy_dt = self.policy.control_dt
        self.total_mass = float(self.model.body_mass.sum())
        self.policy_stride = int(round(self.policy.control_dt / self.dt))
        if self.policy_stride < 1 or not np.isclose(self.policy_stride * self.dt, self.policy.control_dt):
            raise ValueError("Physics timestep must divide the frozen policy period")
        if not np.isclose(self.dt, self.policy.config["dt"]):
            raise ValueError("This adapter refreshes PD every physics step; use upstream dt=.005")
        self.data = mujoco.MjData(self.model)
        self.base_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.gyro_id = self._id(mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.joint_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.policy.joint_names])
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids].copy()
        self.qvel_indices = self.model.jnt_dofadr[self.joint_ids].copy()
        actuator_ids = []
        for joint_id in self.joint_ids:
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
            if len(matches) != 1:
                raise ValueError("Each policy joint needs exactly one MuJoCo motor")
            actuator_ids.append(matches[0])
        self.actuator_ids = np.asarray(actuator_ids)
        self.foot_geom_ids = {self._id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("FL", "FR", "RL", "RR")}
        self.obstacle_geom_ids = {self._id(mujoco.mjtObj.mjOBJ_GEOM, box.get("name", f"task_box_{index}"))
                                  for index, box in enumerate(self.obstacles)}
        self.robot_geom_ids = np.array([i for i in range(self.model.ngeom)
                                       if self.model.geom_bodyid[i] != 0
                                       and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])])
        self.counters = {"physics_steps": 0, "policy_inferences": 0, "actor_forwards": 0,
                         "estimator_forwards": 0, "pd_updates": 0, "dynamics_forward_calls": 0,
                         "snapshots": 0, "restores": 0}
        self.reset()

    @staticmethod
    def _xml_numbers(values):
        return " ".join(str(float(value)) for value in values)

    def _id(self, kind, name):
        identifier = self.mj.mj_name2id(self.model, kind, name)
        if identifier < 0:
            raise ValueError(f"Required nominal Go2 object missing: {name}")
        return identifier

    @staticmethod
    def _empty_contacts():
        return {"normal_impulse_ns": 0., "obstacle_normal_impulse_ns": 0.,
                "obstacle_contact_steps": 0, "nonfoot_ground_contact_steps": 0,
                "self_contact_steps": 0, "peak_normal_force_n": 0.}

    def _rotation(self):
        return self.data.xmat[self.base_id].reshape(3, 3)

    def _gyro(self):
        start = self.model.sensor_adr[self.gyro_id]
        return self.data.sensordata[start:start + 3].copy()

    def reset(self, position=(0., 0., .34), yaw=0., settle_seconds=0.):
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = position
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
        self.data.qpos[self.qpos_indices] = self.policy.default_position
        self.data.qvel[:] = 0.
        self.data.ctrl[:] = 0.
        self.policy.reset()
        self.ticks = 0
        self.command = np.zeros(3)
        self.target_position = self.policy.default_position.copy()
        self.requested_torque = np.zeros(12)
        self.applied_torque = np.zeros(12)
        self.integrals = {"torque_squared_integral": 0., "absolute_mechanical_work_j": 0.,
                          "external_force_impulse_world_ns": np.zeros(3), "motor_limit_samples": 0,
                          **self._empty_contacts()}
        self.interval_contacts = self._empty_contacts()
        self.mj.mj_forward(self.model, self.data)
        self.counters["dynamics_forward_calls"] += 1
        if settle_seconds:
            self.step_command([0., 0., 0.], settle_seconds, record=False)
        return self.state()

    def snapshot(self):
        # MuJoCo 3.3.7 exposes complete MjData copying through __copy__,
        # not a Python-level mj_copyData function.
        saved_data = copy.copy(self.data)
        self.counters["snapshots"] += 1
        return SimulationSnapshot(self.model, saved_data, self.policy.snapshot(), self.ticks,
                                  self.command.copy(), self.target_position.copy(),
                                  self.requested_torque.copy(), self.applied_torque.copy(),
                                  copy.deepcopy(self.integrals), copy.deepcopy(self.interval_contacts))

    def restore(self, saved):
        if saved.model is not self.model:
            raise ValueError("Restore snapshots only into the same immutable model or its clone")
        self.data = copy.copy(saved.data)
        self.policy.restore(saved.policy_state)
        self.ticks = saved.ticks
        self.command, self.target_position = saved.command.copy(), saved.target_position.copy()
        self.requested_torque, self.applied_torque = saved.requested_torque.copy(), saved.applied_torque.copy()
        self.integrals, self.interval_contacts = copy.deepcopy(saved.integrals), copy.deepcopy(saved.interval_contacts)
        self.counters["restores"] += 1

    def clone(self):
        result = copy.copy(self)
        result.data = self.mj.MjData(self.model)
        result.policy = self.policy.clone()
        result.counters = dict.fromkeys(self.counters, 0)
        result.restore(self.snapshot())
        return result

    def _contact_stats(self):
        stats = {"normal_force_n": 0., "obstacle_normal_force_n": 0., "obstacle_contacts": 0,
                 "nonfoot_ground_contacts": 0, "self_contacts": 0, "peak_normal_force_n": 0.,
                 "foot_normal_force_n": {leg: 0. for leg in ("FL", "FR", "RL", "RR")}}
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            self.mj.mj_contactForce(self.model, self.data, index, wrench)
            normal = max(0., float(wrench[0]))
            # MuJoCo can keep positive-gap, zero-force margin candidates in
            # data.contact. Count actual touching/penetration or resolved force,
            # not the mere presence of such a broadphase/constraint candidate.
            if contact.dist > 0. and normal <= 1e-8:
                continue
            stats["normal_force_n"] += normal
            stats["peak_normal_force_n"] = max(stats["peak_normal_force_n"], normal)
            bodies = (self.model.geom_bodyid[g1], self.model.geom_bodyid[g2])
            if g1 in self.obstacle_geom_ids or g2 in self.obstacle_geom_ids:
                stats["obstacle_contacts"] += 1
                stats["obstacle_normal_force_n"] += normal
            elif 0 in bodies:
                robot_geom = g2 if bodies[0] == 0 else g1
                if robot_geom in self.foot_geom_ids:
                    name = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, robot_geom)
                    stats["foot_normal_force_n"][name] += normal
                else:
                    stats["nonfoot_ground_contacts"] += 1
            else:
                stats["self_contacts"] += 1
        return stats

    def _collision_bounds(self):
        lower, upper = [], []
        for geom in self.robot_geom_ids:
            centre = self.data.geom_xpos[geom]
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            size, kind = self.model.geom_size[geom], self.model.geom_type[geom]
            types = self.mj.mjtGeom
            if kind == types.mjGEOM_SPHERE:
                radius = np.full(3, size[0])
            elif kind == types.mjGEOM_BOX:
                radius = np.abs(rotation) @ size
            elif kind == types.mjGEOM_CAPSULE:
                radius = np.abs(rotation[:, 2]) * size[1] + size[0]
            elif kind == types.mjGEOM_CYLINDER:
                axis = rotation[:, 2]
                radius = np.abs(axis) * size[1] + size[0] * np.sqrt(np.maximum(0., 1. - axis**2))
            elif kind == types.mjGEOM_ELLIPSOID:
                radius = np.sqrt(rotation**2 @ size**2)
            else:
                radius = np.full(3, self.model.geom_rbound[geom])
            lower.append(centre - radius); upper.append(centre + radius)
        return np.min(lower, axis=0), np.max(upper, axis=0)

    def state(self):
        rotation = self._rotation()
        velocity = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY,
                                  self.base_id, velocity, 0)
        quaternion = self.data.xquat[self.base_id].copy()
        w, x, y, z = quaternion
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = np.arcsin(np.clip(2*(w*y-z*x), -1., 1.))
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        lower, upper = self._collision_bounds()
        return {"time": float(self.data.time), "position": self.data.xpos[self.base_id].copy(),
                "quaternion_wxyz": quaternion, "rpy": np.array([roll, pitch, yaw]), "yaw": float(yaw),
                "linear_velocity_world": velocity[3:].copy(), "angular_velocity_world": velocity[:3].copy(),
                "linear_velocity_body": rotation.T @ velocity[3:],
                "tilt": float(np.arccos(np.clip(rotation[2, 2], -1., 1.))),
                "angular_velocity_body": self._gyro(), "projected_gravity": rotation.T @ np.array([0., 0., -1.]),
                "joint_position": self.data.qpos[self.qpos_indices].copy(),
                "joint_velocity": self.data.qvel[self.qvel_indices].copy(),
                "command": self.command.copy(), "q_target": self.target_position.copy(),
                "action": self.policy.previous_action.copy(), "raw_action": self.policy.last_raw_action.copy(),
                "observation": self.policy.last_observation.copy(), "policy_input_history": self.policy.last_input.copy(),
                "history_frames": self.policy.history.copy(), "policy_phase": self.ticks % self.policy_stride,
                "requested_torque": self.requested_torque.copy(), "applied_torque": self.applied_torque.copy(),
                "collision_aabb_min": lower, "collision_aabb_max": upper,
                "robot_aabb_min": lower.copy(), "robot_aabb_max": upper.copy(),
                "contacts": self._contact_stats(), "resource_integrals": copy.deepcopy(self.integrals),
                "policy_interval_contacts": self.interval_contacts.copy()}

    def step_command(self, command, duration, external_force=None, *, record=True, trace=None):
        """Advance physics, retaining command/history/PD phase across calls.

        command is body [vx,vy,yaw_rate]. external_force is world [Fx,Fy,Fz]
        or [Fx,Fy,Fz,Tx,Ty,Tz] at the base COM, constant for this call. It can
        also be a callable(time) returning such a vector for actual execution.
        Predictors should pass None; no future force enters policy observations.
        trace is recorded after each completed 20ms policy interval, plus a
        partial final interval. Model/force/limit identity is never inferred.
        """
        command = np.asarray(command, dtype=np.float64)
        steps = int(round(float(duration) / self.dt))
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Command must be finite [vx,vy,yaw_rate]")
        if steps < 0 or not np.isclose(steps * self.dt, duration, atol=1e-10):
            raise ValueError("Duration must be a nonnegative multiple of physics_dt")
        if trace is not None:
            record = bool(trace)
        before = self.counters.copy()
        previous_torque_integral = self.integrals["torque_squared_integral"]
        previous_obstacle_steps = self.integrals["obstacle_contact_steps"]
        previous_nonfoot_steps = self.integrals["nonfoot_ground_contact_steps"]
        previous_work = self.integrals["absolute_mechanical_work_j"]
        frames = []
        self.command = command.copy()
        for index in range(steps):
            if self.ticks % self.policy_stride == 0:
                self.interval_contacts = self._empty_contacts()
                self.target_position = self.policy.infer(
                    command, self._gyro(), self._rotation().T @ np.array([0., 0., -1.]),
                    self.data.qpos[self.qpos_indices], self.data.qvel[self.qvel_indices])
                self.counters["policy_inferences"] += 1
                self.counters["actor_forwards"] += 1
                self.counters["estimator_forwards"] += int(bool(self.policy.history_indices))
            joint_velocity = self.data.qvel[self.qvel_indices].copy()
            self.requested_torque = (self.policy.kp * (self.target_position - self.data.qpos[self.qpos_indices])
                                     - self.policy.kd * joint_velocity)
            self.data.ctrl[self.actuator_ids] = self.requested_torque
            force = external_force(float(self.data.time)) if callable(external_force) else external_force
            wrench = np.zeros(6)
            if force is not None:
                force = np.asarray(force, dtype=np.float64)
                if force.shape not in {(3,), (6,)} or not np.isfinite(force).all():
                    raise ValueError("External force must be a finite world force or wrench")
                wrench[:len(force)] = force
            self.data.xfrc_applied[:] = 0.
            self.data.xfrc_applied[self.base_id] = wrench
            self.mj.mj_step(self.model, self.data)
            self.applied_torque = self.data.qfrc_actuator[self.qvel_indices].copy()
            contacts = self._contact_stats()
            increments = {"normal_impulse_ns": contacts["normal_force_n"] * self.dt,
                          "obstacle_normal_impulse_ns": contacts["obstacle_normal_force_n"] * self.dt,
                          "obstacle_contact_steps": int(contacts["obstacle_contacts"] > 0),
                          "nonfoot_ground_contact_steps": int(contacts["nonfoot_ground_contacts"] > 0),
                          "self_contact_steps": int(contacts["self_contacts"] > 0)}
            for key, value in increments.items():
                self.integrals[key] += value
                self.interval_contacts[key] += value
            for container in (self.integrals, self.interval_contacts):
                container["peak_normal_force_n"] = max(container["peak_normal_force_n"], contacts["peak_normal_force_n"])
            self.integrals["torque_squared_integral"] += float(self.applied_torque @ self.applied_torque) * self.dt
            self.integrals["absolute_mechanical_work_j"] += float(np.abs(self.applied_torque * joint_velocity).sum()) * self.dt
            self.integrals["external_force_impulse_world_ns"] += wrench[:3] * self.dt
            limits = self.model.actuator_ctrlrange[self.actuator_ids]
            limited = self.model.actuator_ctrllimited[self.actuator_ids].astype(bool)
            self.integrals["motor_limit_samples"] += int(np.count_nonzero(limited & (
                (self.requested_torque < limits[:, 0]) | (self.requested_torque > limits[:, 1]))))
            self.ticks += 1
            self.counters["physics_steps"] += 1
            self.counters["pd_updates"] += 1
            # Refresh post-integration kinematics/sensors identically in actual
            # execution and prediction, independently of whether trace is saved.
            self.mj.mj_forward(self.model, self.data)
            self.counters["dynamics_forward_calls"] += 1
            if record and (self.ticks % self.policy_stride == 0 or index == steps - 1):
                frames.append(self.state())
        work = {key: self.counters[key] - before[key] for key in self.counters}
        work["simulated_seconds"] = steps * self.dt
        return {"state": self.state(), "trace": frames, "work": work,
                "torque_squared_integral": self.integrals["torque_squared_integral"] - previous_torque_integral,
                "absolute_work": self.integrals["absolute_mechanical_work_j"] - previous_work,
                "obstacle_contacts": self.integrals["obstacle_contact_steps"] - previous_obstacle_steps,
                "nonfoot_ground_contacts": self.integrals["nonfoot_ground_contact_steps"] - previous_nonfoot_steps}

    def provenance(self):
        return {"model_xml_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
                "physics_dt": self.dt, "policy_stride": self.policy_stride, "obstacles": copy.deepcopy(self.obstacles),
                "base_body": "base_link", "policy": self.policy.provenance(),
                "torque_semantics": "position PD -> motor ctrl; MuJoCo scene ctrlrange applies",
                "policy_yaml_torque_clip_applied": False,
                "motor_ctrlrange_policy_order": self.model.actuator_ctrlrange[self.actuator_ids].copy(),
                "nominal_simulation_not_user_hardware": True, "training": False}
