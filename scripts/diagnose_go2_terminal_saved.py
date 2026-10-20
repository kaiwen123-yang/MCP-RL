"""Describe saved Go2 terminal motion and commands; no rollout or model fitting.

Loads trusted tensor-only transition dictionaries with weights_only=True and
the original follower CSV. Reads all 27 final episodes, all 27 round-zero
episodes, seed11's nine round-three episodes, and three follower references.
No controller, physics engine, optimizer, checkpoint policy or CEM is invoked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DT = 0.02


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def first_time(time, mask):
    indices = np.flatnonzero(mask)
    return float(time[indices[0]]) if len(indices) else None


def mean(values):
    return float(np.mean(values)) if len(values) else None


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def wrapped(value):
    return np.arctan2(np.sin(value), np.cos(value))


def intervals(mask):
    padded = np.r_[False, mask, False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded) == 1),
                    np.flatnonzero(np.diff(padded) == -1) - 1))


class Sources:
    def __init__(self):
        self.files = {}

    def add(self, path):
        path = Path(path)
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        if relative not in self.files:
            raw = path.read_bytes()
            self.files[relative] = {
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return relative


def learning_episodes(runs, seed, arm, round_number, sources):
    folder = (runs / f"seed{seed}/{arm}/evaluation_round0" if round_number == 0 else
              runs / f"seed{seed}/{arm}/round{round_number}/evaluation")
    summary_path, tensor_path = folder / "summary.json", folder / "transitions.pt"
    summary_source, tensor_source = sources.add(summary_path), sources.add(tensor_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved = torch.load(tensor_path, map_location="cpu", weights_only=True)
    batch = {key: value.detach().cpu().numpy() for key, value in saved["batch"].items()}
    offset = 0
    episodes = []
    for ep_index, episode in enumerate(summary["episodes"]):
        count = int(episode["policy_steps"])
        ix = slice(offset, offset + count)
        state = np.vstack((batch["state"][offset], batch["next_state"][ix])).astype(float)
        memory = np.vstack((batch["memory"][offset], batch["next_memory"][ix])).astype(float)
        command = np.vstack((batch["command"][offset], batch["command"][ix])).astype(float)
        time = np.arange(count + 1) * DT
        max_time_error = float(np.max(np.abs(memory[:, 1] - time)))
        if max_time_error > 4e-6 or not bool(batch["terminated"][offset + count - 1]):
            raise ValueError(f"Episode boundary or timestamp mismatch in {summary_source}")
        decisions = next(x["decisions"] for x in summary["decisions"]
                         if x["case"] == episode["case"] and x["episode"] == episode["episode"])
        gate = next(x for x in episode["stage_exits"] if x["from"] == "door" and x["to"] == "stop")
        trajectory_digest = hashlib.sha256()
        for field in ("state", "next_state", "memory", "next_memory", "command", "executed_action"):
            trajectory_digest.update(np.ascontiguousarray(batch[field][ix]).tobytes())
        episodes.append({
            "id": f"s{seed}_{arm}_r{round_number}_{episode['case']}",
            "seed": seed, "arm": arm, "round": round_number, "case": episode["case"],
            "scope": "round0" if round_number == 0 else "final" if round_number == 4 else "seed11_round3",
            "state": state, "memory": memory, "command": command, "time": time,
            "spec": episode["task"], "decisions": decisions,
            "door_time": float(gate["time"]),
            "complete_time": float(episode["completion_seconds"]),
            "success": bool(episode["success"]), "hold_authoritative": True,
            "summary_source": summary_source, "trajectory_source": tensor_source,
            "episode_json_pointer": f"/episodes/{ep_index}",
            "tensor_row_begin": offset, "tensor_row_end_exclusive": offset + count,
            "timestamp_float32_max_error_seconds": max_time_error,
            "trajectory_state_command_hash": trajectory_digest.hexdigest(),
        })
        offset += count
    if offset != len(batch["state"]):
        raise ValueError(f"Unassigned transition rows in {tensor_source}")
    return episodes


def follower_episode(reference, case, sources):
    label = f"primary_fast_follower_{case}"
    evidence, raw = reference / "evidence" / label, reference / "raw" / label
    summary_path, trajectory_path = evidence / "summary.json", raw / "trajectory_50hz.csv"
    summary_source, trajectory_source = sources.add(summary_path), sources.add(trajectory_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with trajectory_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    array = lambda key, count: np.array([[float(r[f"{key}_{i}"]) for i in range(count)] for r in rows])
    state = np.column_stack((array("position", 3), array("rpy", 3), array("velocity_world", 3),
                             array("omega_body", 3), array("joint_pos", 12),
                             array("joint_vel", 12), array("action", 12)))
    time = np.array([float(r["time"]) for r in rows])
    stages = ["straight", "turn", "door", "stop", "complete"]
    memory = np.column_stack(([stages.index(r["stage"]) for r in rows], time, np.full(len(rows), -1.0)))
    spec = summary["task"]
    goal = np.array([spec["straight_length"] + spec["turn_radius"], spec["goal_y"]])
    good = ((memory[:, 0] >= 3)
            & (np.linalg.norm(state[:, :2] - goal, axis=1) <= spec["goal_position_tolerance"])
            & (np.linalg.norm(state[:, 6:8], axis=1) <= spec["goal_speed_tolerance"])
            & (np.abs(wrapped(state[:, 5] - np.pi / 2)) <= spec["goal_yaw_tolerance"]))
    for begin, end in intervals(good):
        memory[begin:end + 1, 2] = time[begin:end + 1] - time[begin]
    decisions_path = evidence / "decisions.json"
    sources.add(decisions_path)
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    return {
        "id": f"follower_{case}", "seed": None, "arm": "follower", "round": None,
        "case": case, "scope": "reference", "state": state, "memory": memory,
        "command": array("command", 3), "time": time, "spec": spec, "decisions": decisions,
        "door_time": first_time(time, memory[:, 0] >= 3),
        "complete_time": float(summary["elapsed_sim_seconds"]), "success": bool(summary["success"]),
        "hold_authoritative": False, "summary_source": summary_source,
        "trajectory_source": trajectory_source, "episode_json_pointer": None,
        "tensor_row_begin": None, "tensor_row_end_exclusive": None,
        "timestamp_float32_max_error_seconds": 0.0, "trajectory_state_command_hash": None,
    }


def analyze_episode(ep):
    x, m, c, t, spec = (ep[k] for k in ("state", "memory", "command", "time", "spec"))
    goal = np.array([spec["straight_length"] + spec["turn_radius"], spec["goal_y"]])
    delta = goal - x[:, :2]
    distance = np.linalg.norm(delta, axis=1)
    speed = np.linalg.norm(x[:, 6:8], axis=1)
    yaw_error = np.abs(wrapped(x[:, 5] - np.pi / 2))
    pos_ok = distance <= spec["goal_position_tolerance"]
    speed_ok = speed <= spec["goal_speed_tolerance"]
    yaw_ok = yaw_error <= spec["goal_yaw_tolerance"]
    after = t >= ep["door_time"] - 1e-6
    duration_mask = t > ep["door_time"] + 1e-6
    geometric_good = after & pos_ok & speed_ok & yaw_ok
    held_good = after & (m[:, 2] >= 0)
    first_pos = first_time(t, after & pos_ok)
    first_pos_yaw = first_time(t, after & pos_ok & yaw_ok)
    first_good = first_time(t, held_good)
    first_speed = first_time(t, after & speed_ok)
    first_yaw = first_time(t, after & yaw_ok)
    runs = intervals(held_good)
    final_start = float(t[runs[-1][0]]) if runs else None
    final_hold = float(t[runs[-1][1]] - t[runs[-1][0]]) if runs else None
    cosine, sine = np.cos(x[:, 5]), np.sin(x[:, 5])
    body_velocity = np.column_stack((cosine * x[:, 6] + sine * x[:, 7],
                                     -sine * x[:, 6] + cosine * x[:, 7]))
    world_command = np.column_stack((cosine * c[:, 0] - sine * c[:, 1],
                                     sine * c[:, 0] + cosine * c[:, 1]))
    command_speed = np.linalg.norm(c[:, :2], axis=1)
    radial_unit = delta / np.maximum(distance[:, None], 1e-12)
    commanded_approach = np.sum(world_command * radial_unit, axis=1)
    actual_approach = np.sum(x[:, 6:8] * radial_unit, axis=1)
    rpy = x[:, 3:6]
    yaw_rate = (np.sin(rpy[:, 0]) * x[:, 10] + np.cos(rpy[:, 0]) * x[:, 11]) / np.cos(rpy[:, 1])
    tracking_error = np.linalg.norm(body_velocity - c[:, :2], axis=1)
    within_pose = duration_mask & pos_ok & yaw_ok
    speed_obstruction = within_pose & ~speed_ok
    within_position = duration_mask & pos_ok
    position_indices = np.flatnonzero(after & pos_ok)
    after_first_position = (t >= first_pos - 1e-6) if first_pos is not None else np.zeros(len(t), bool)
    pos_exit = np.flatnonzero(pos_ok[:-1] & ~pos_ok[1:] & after_first_position[:-1])
    marginal = after & ((np.abs(distance - spec["goal_position_tolerance"]) < 1e-6)
                        | (np.abs(speed - spec["goal_speed_tolerance"]) < 1e-6)
                        | (np.abs(yaw_error - spec["goal_yaw_tolerance"]) < 1e-6))
    after_ix = np.flatnonzero(after)
    dist_diff = np.diff(distance[after_ix])
    xy_diff = np.diff(x[after_ix, :2], axis=0)
    actual_acceleration = np.diff(x[:, 6:8], axis=0) / np.diff(t)[:, None]
    row = {
        key: ep[key] for key in ("id", "seed", "arm", "round", "case", "scope", "success",
                                "summary_source", "trajectory_source", "episode_json_pointer",
                                "tensor_row_begin", "tensor_row_end_exclusive",
                                "timestamp_float32_max_error_seconds", "trajectory_state_command_hash")
    }
    row.update({
        "door_clear_seconds": ep["door_time"], "completion_seconds": ep["complete_time"],
        "door_to_complete_seconds": ep["complete_time"] - ep["door_time"],
        "first_position_zone_seconds": first_pos, "first_speed_ok_seconds_after_door": first_speed,
        "first_yaw_ok_seconds_after_door": first_yaw,
        "first_position_and_yaw_ok_seconds": first_pos_yaw,
        "first_recorded_joint_good_seconds": first_good,
        "final_unbroken_hold_start_seconds": final_start,
        "final_unbroken_hold_elapsed_seconds": final_hold,
        "door_to_first_position_zone_seconds": first_pos - ep["door_time"] if first_pos is not None else None,
        "position_entry_to_final_hold_start_seconds": final_start - first_pos
        if final_start is not None and first_pos is not None else None,
        "first_joint_good_to_final_hold_start_seconds": final_start - first_good
        if final_start is not None and first_good is not None else None,
        "hold_attempt_count": len(runs), "hold_reset_count": max(0, len(runs) - 1),
        "position_zone_exit_count_after_first_entry": len(pos_exit),
        "postdoor_position_bad_seconds": float(np.count_nonzero(duration_mask & ~pos_ok) * DT),
        "postdoor_speed_bad_seconds": float(np.count_nonzero(duration_mask & ~speed_ok) * DT),
        "postdoor_yaw_bad_seconds": float(np.count_nonzero(duration_mask & ~yaw_ok) * DT),
        "pose_good_but_speed_bad_seconds": float(np.count_nonzero(speed_obstruction) * DT),
        "pose_good_speed_bad_with_command_above_speed_threshold_seconds": float(np.count_nonzero(
            speed_obstruction & (command_speed > spec["goal_speed_tolerance"])) * DT),
        "pose_good_speed_bad_with_command_at_or_below_speed_threshold_seconds": float(np.count_nonzero(
            speed_obstruction & (command_speed <= spec["goal_speed_tolerance"])) * DT),
        "postdoor_joint_good_sample_occupancy_seconds": float(np.count_nonzero(duration_mask & held_good) * DT),
        "postdoor_command_speed_mean_m_s": mean(command_speed[duration_mask]),
        "postdoor_actual_speed_mean_m_s": mean(speed[duration_mask]),
        "position_zone_command_speed_mean_m_s": mean(command_speed[within_position]),
        "position_zone_actual_speed_mean_m_s": mean(speed[within_position]),
        "postdoor_translation_tracking_rmse_m_s": rms(tracking_error[duration_mask]),
        "position_zone_translation_tracking_rmse_m_s": rms(tracking_error[within_position]),
        "postdoor_yaw_rate_tracking_rmse_rad_s": rms((yaw_rate - c[:, 2])[duration_mask]),
        "postdoor_command_radial_away_sample_fraction": mean((commanded_approach[duration_mask] < 0).astype(float)),
        "postdoor_actual_radial_away_sample_fraction": mean((actual_approach[duration_mask] < 0).astype(float)),
        "postdoor_path_length_m": float(np.linalg.norm(xy_diff, axis=1).sum()),
        "postdoor_distance_increase_total_m": float(np.maximum(dist_diff, 0).sum()),
        "postdoor_backward_y_travel_m": float(np.maximum(-xy_diff[:, 1], 0).sum()),
        "postdoor_goal_y_overshoot_max_m": float(np.maximum(x[after_ix, 1] - goal[1], 0).max()),
        "postdoor_actual_velocity_difference_rms_m_s2": rms(
            np.linalg.norm(actual_acceleration[duration_mask[1:]], axis=-1)),
        "hold_recomputed_vs_recorded_disagreement_samples": int(np.count_nonzero(
            geometric_good != held_good)) if ep["hold_authoritative"] else None,
        "threshold_marginal_samples_within_1e_minus6": int(np.count_nonzero(marginal)),
        "hold_source": "recorded_Task_memory" if ep["hold_authoritative"] else "recomputed_from_original_50Hz_CSV",
    })
    attempts = []
    for number, (begin, end) in enumerate(runs):
        reset = end + 1 < len(t)
        at = end + 1 if reset else end
        attempts.append({
            "id": ep["id"], "attempt": number + 1, "start_seconds": float(t[begin]),
            "last_eligible_sample_seconds": float(t[end]),
            "eligible_span_seconds": float(t[end] - t[begin]),
            "reset": reset, "first_ineligible_sample_seconds": float(t[at]) if reset else None,
            "reset_position_bad": bool(not pos_ok[at]) if reset else None,
            "reset_speed_bad": bool(not speed_ok[at]) if reset else None,
            "reset_yaw_bad": bool(not yaw_ok[at]) if reset else None,
            "distance_at_reset_m": float(distance[at]) if reset else None,
            "speed_at_reset_m_s": float(speed[at]) if reset else None,
            "yaw_error_at_reset_rad": float(yaw_error[at]) if reset else None,
            "command_speed_at_reset_m_s": float(command_speed[at]) if reset else None,
        })
    reset_rows = [attempt for attempt in attempts if attempt["reset"]]
    for label, selected in (
        ("speed_only", lambda a: a["reset_speed_bad"] and not a["reset_position_bad"] and not a["reset_yaw_bad"]),
        ("yaw_only", lambda a: a["reset_yaw_bad"] and not a["reset_position_bad"] and not a["reset_speed_bad"]),
        ("position_only", lambda a: a["reset_position_bad"] and not a["reset_speed_bad"] and not a["reset_yaw_bad"]),
        ("multiple", lambda a: sum(a[k] for k in ("reset_position_bad", "reset_speed_bad", "reset_yaw_bad")) > 1),
    ):
        row[f"hold_reset_{label}_count"] = sum(bool(selected(attempt)) for attempt in reset_rows)
    zone_events = []
    for at in np.flatnonzero(pos_ok[1:] != pos_ok[:-1]) + 1:
        if after[at]:
            zone_events.append({
                "id": ep["id"], "time_seconds": float(t[at]),
                "event": "position_zone_entry" if pos_ok[at] else "position_zone_exit",
                "world_x_m": float(x[at, 0]), "world_y_m": float(x[at, 1]),
                "goal_distance_m": float(distance[at]), "actual_speed_m_s": float(speed[at]),
                "command_speed_m_s": float(command_speed[at]),
            })
    command_rows = []
    previous_command = None
    previous_time = None
    change_norms, yaw_changes = [], []
    for decision in ep["decisions"]:
        decision_time = float(decision["time"])
        current = np.asarray(decision["command"], dtype=float)
        at = min(round(decision_time / DT), len(t) - 1)
        if decision_time >= ep["door_time"] - 1e-6:
            rate = (current - previous_command) / (decision_time - previous_time) if previous_command is not None else None
            if rate is not None:
                change_norms.append(float(np.linalg.norm(rate[:2])))
                yaw_changes.append(float(rate[2]))
            command_rows.append({
                "id": ep["id"], "time_seconds": decision_time,
                "time_after_door_seconds": decision_time - ep["door_time"],
                "world_x_m": float(x[at, 0]), "world_y_m": float(x[at, 1]),
                "goal_distance_m": float(distance[at]), "actual_speed_m_s": float(speed[at]),
                "goal_yaw_error_rad": float(yaw_error[at]), "hold_elapsed_seconds": float(m[at, 2]),
                "command_vx_m_s": float(current[0]), "command_vy_m_s": float(current[1]),
                "command_yaw_rate_rad_s": float(current[2]),
                "command_planar_speed_m_s": float(np.linalg.norm(current[:2])),
                "command_change_rate_norm_m_s2": float(np.linalg.norm(rate[:2])) if rate is not None else None,
                "command_yaw_change_rate_rad_s2": float(rate[2]) if rate is not None else None,
                "logged_predicted_cost": decision.get("predicted_cost"),
            })
        previous_command, previous_time = current, decision_time
    row["postdoor_macro_command_change_rate_rms_m_s2"] = rms(change_norms)
    row["postdoor_macro_yaw_command_change_rate_rms_rad_s2"] = rms(yaw_changes)
    trace_rows = []
    focus = (ep["scope"] == "reference" or
             (ep["seed"] == 11 and ep["round"] in (3, 4) and
              ((ep["arm"] in ("ordinary", "adjoint") and ep["case"] == "nominal")
               or (ep["arm"] == "frozen" and ep["case"] == "narrow"))))
    if focus:
        for at in after_ix:
            trace_rows.append({
                "id": ep["id"], "time_seconds": float(t[at]),
                "world_x_m": float(x[at, 0]), "world_y_m": float(x[at, 1]),
                "goal_distance_m": float(distance[at]), "actual_speed_m_s": float(speed[at]),
                "goal_yaw_error_rad": float(yaw_error[at]),
                "position_ok": bool(pos_ok[at]), "speed_ok": bool(speed_ok[at]),
                "yaw_ok": bool(yaw_ok[at]), "hold_elapsed_seconds": float(m[at, 2]),
                "command_vx_m_s": float(c[at, 0]), "command_vy_m_s": float(c[at, 1]),
                "command_yaw_rate_rad_s": float(c[at, 2]),
                "command_planar_speed_m_s": float(command_speed[at]),
                "actual_yaw_frame_vx_m_s": float(body_velocity[at, 0]),
                "actual_yaw_frame_vy_m_s": float(body_velocity[at, 1]),
                "actual_yaw_rate_rad_s": float(yaw_rate[at]),
                "commanded_approach_speed_m_s": float(commanded_approach[at]),
                "actual_approach_speed_m_s": float(actual_approach[at]),
            })
    return row, attempts, zone_events, command_rows, trace_rows


def differences(left, right, label):
    keys = ("completion_seconds", "door_clear_seconds", "door_to_complete_seconds",
            "door_to_first_position_zone_seconds", "position_entry_to_final_hold_start_seconds",
            "hold_reset_count", "position_zone_exit_count_after_first_entry",
            "pose_good_but_speed_bad_seconds",
            "postdoor_command_speed_mean_m_s", "postdoor_actual_speed_mean_m_s",
            "postdoor_translation_tracking_rmse_m_s", "postdoor_path_length_m")
    return {
        "contrast": label, "left": left["id"], "right": right["id"],
        **{f"right_minus_left_{key}": right[key] - left[key]
           if right[key] is not None and left[key] is not None else None for key in keys},
    }


def plot_saved_figure(output, skill_scripts=None):
    """One requested figure from existing diagnostic CSV; no raw-data replay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image

    if skill_scripts is not None:
        sys.path.insert(0, str(skill_scripts))
        from setup_style import setup_style
        from visual_qa import audit_layout
        from layout_tools import add_panel_labels
        setup_style(journal="general", lang="en", use_sciplots=False)
    else:
        audit_layout = None
        add_panel_labels = None
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.constrained_layout.use": False, "figure.autolayout": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .18, "lines.linewidth": 1.05,
    })
    with (output / "focus_postdoor_50hz.csv").open(newline="", encoding="utf-8") as stream:
        traces = list(csv.DictReader(stream))
    with (output / "episode_terminal_metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = {row["id"]: row for row in csv.DictReader(stream)}
    identities = [
        ("s11_ordinary_r4_nominal", "Ordinary PPO / nominal", "follower_nominal"),
        ("s11_adjoint_r4_nominal", "Adjoint PPO / nominal", "follower_nominal"),
        ("s11_frozen_r4_narrow", "Frozen actor / narrow", "follower_narrow"),
    ]
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    arrays, profiles = {}, []
    for identity in {item for selected in identities for item in (selected[0], selected[2])}:
        selected = [r for r in traces if r["id"] == identity]
        fields = ("time_seconds", "goal_distance_m", "actual_speed_m_s",
                  "command_planar_speed_m_s", "hold_elapsed_seconds")
        arrays[identity] = {key: np.array([float(r[key]) for r in selected]) for key in fields}
        profiles.append({"id": identity, "samples": len(selected),
                         "sampling": "saved 50 Hz time series; one episode, not independent samples",
                         **{key: {"minimum": float(values.min()), "maximum": float(values.max()),
                                  "missing": int(np.count_nonzero(~np.isfinite(values)))}
                            for key, values in arrays[identity].items()}})
    json_write(figure_dir / "data_profile.json", {
        "method": "Direct CSV type/range/count inspection; no pandas dependency or environment modification",
        "profiles": profiles,
        "chart_choice": "Three aligned time-series columns expose arrival, speed threshold crossings and hold resets. Shared axes preserve actual duration.",
        "alternative_not_used": "A completion-time bar would hide the continuous resets and command sequence.",
    })
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 6.5), sharex=True, sharey="row")
    fig.subplots_adjust(left=.105, right=.985, top=.87, bottom=.22, hspace=.23, wspace=.15)
    thresholds = (.22, .12, 1.0)
    labels = ("Position error (m)", "Planar speed (m/s)", "Hold clock (s)")
    blue, orange, gray = "#0072B2", "#D55E00", "#999999"
    for col, (identity, title, reference) in enumerate(identities):
        current, baseline = arrays[identity], arrays[reference]
        time = current["time_seconds"] - float(metrics[identity]["door_clear_seconds"])
        ref_time = baseline["time_seconds"] - float(metrics[reference]["door_clear_seconds"])
        actual_series = (current["goal_distance_m"], current["actual_speed_m_s"],
                         np.maximum(current["hold_elapsed_seconds"], 0.0))
        baseline_series = (baseline["goal_distance_m"], baseline["actual_speed_m_s"],
                           np.maximum(baseline["hold_elapsed_seconds"], 0.0))
        for row in range(3):
            ax = axes[row, col]
            ax.plot(ref_time, baseline_series[row], color=gray, ls="-.", lw=1.05, alpha=.85, zorder=1)
            if row == 2:
                ax.step(time, actual_series[row], where="post", color=blue, lw=1.0, zorder=3)
            else:
                ax.plot(time, actual_series[row], color=blue, zorder=3)
            ax.plot(time[-1], actual_series[row][-1], marker="o", color=blue, markersize=3, zorder=4)
            if row == 1:
                ax.step(time, current["command_planar_speed_m_s"], where="pre",
                        color=orange, ls="--", lw=.8, alpha=.85, zorder=2)
            ax.axhline(thresholds[row], color="#333333", ls=(0, (4, 3)), lw=.7, zorder=0)
            ax.set_xlim(0, 18.6)
            ax.set_xticks([0, 5, 10, 15])
            if col == 0:
                ax.set_ylabel(labels[row])
            if row == 2:
                ax.set_xlabel("Time after door clearance (s)")
        duration = float(metrics[identity]["door_to_complete_seconds"])
        resets = int(metrics[identity]["hold_reset_count"])
        axes[0, col].set_title(f"{title}\n{duration:.2f} s; {resets} hold resets", pad=10)
    axes[0, 0].set_ylim(0, 1.3)
    axes[0, 0].set_yticks([0, .22, .6, 1.2], ["0", ".22", ".6", "1.2"])
    axes[1, 0].set_ylim(0, .85)
    axes[1, 0].set_yticks([0, .12, .4, .8], ["0", ".12", ".4", ".8"])
    axes[2, 0].set_ylim(0, 1.13)
    axes[2, 0].set_yticks([0, .5, 1.0], ["0", ".5", "1.0"])
    fig.suptitle("Final-approach motion repeatedly interrupts the required hold\nSeed 11, fixed round 4; saved complete trajectories",
                 fontsize=10, y=.975)
    if add_panel_labels:
        add_panel_labels(fig, axes=axes.flatten(), style="paren", fontsize=7.5,
                         x_offset_pt=0, y_offset_pt=1, ha="left", va="bottom")
    handles = [
        Line2D([0], [0], color=blue, label="Actual trajectory / hold"),
        Line2D([0], [0], color=orange, ls="--", label="Issued command norm"),
        Line2D([0], [0], color=gray, ls="-.", label="Original follower, same case"),
        Line2D([0], [0], color="#333333", ls="--", lw=.7, label="Original task threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .105),
               ncol=2, frameon=False, columnspacing=1.4)
    caption_lines = [
        "The stop stage includes final approach and the 1 s hold. Hold = 0 also denotes ineligible samples.",
        "Command norm above .12 m/s does not prove an instantaneous causal speed response.",
        "All curves stop at their recorded completion; each column and reference is one saved trajectory.",
    ]
    fig.text(.105, .067, "\n".join(caption_lines), ha="left", va="top", fontsize=7, linespacing=1.35)
    fig.canvas.draw()
    issues = audit_layout(fig) if audit_layout else []
    clipping = []
    for panel, ax in enumerate(axes.flat):
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        for line in ax.lines:
            xs, ys = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
            # axhline x coordinates are axes fractions; both bounds are still inside.
            if np.any((xs < xlo - 1e-5) | (xs > xhi + 1e-5) | (ys < ylo - 1e-5) | (ys > yhi + 1e-5)):
                clipping.append(panel)
    fig.savefig(figure_dir / "terminal_hold_chain_preview.png", dpi=150)
    fig.savefig(figure_dir / "terminal_hold_chain.png", dpi=600)
    fig.savefig(figure_dir / "terminal_hold_chain.svg")
    with Image.open(figure_dir / "terminal_hold_chain_preview.png") as picture:
        picture.convert("L").save(figure_dir / "terminal_hold_chain_grayscale.png")
    json_write(figure_dir / "figure_check.json", {
        "layout_issues": issues, "data_out_of_axis_panels": sorted(set(clipping)),
        "figure_inches": [7.16, 6.5], "png_dpi": 600,
        "source": ["focus_postdoor_50hz.csv", "episode_terminal_metrics.csv"],
        "visual_review": "pending image inspection",
        "caption_zh": "同一固定第四轮三个实例，横轴为实际过门后的时间。stop阶段包含终点接近和最终1秒保持。灰线是相同条件原跟随器的一次已保存轨迹；保持时钟不合格时显示为0。命令速度模长超过0.12m/s与瞬时实测速度超过阈值不是因果等价，也未通过该图隔离策略、预测、价值或规划的原因。",
    })
    plt.close(fig)


def coarse_hold_check(output):
    """Observe aliasing at original 0.2 s boundaries in saved 50 Hz traces only."""
    with (output / "focus_postdoor_50hz.csv").open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    with (output / "episode_terminal_metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = {row["id"]: row for row in csv.DictReader(stream)}
    results, windows = [], []
    for identity in dict.fromkeys(row["id"] for row in all_rows):
        rows = [row for row in all_rows if row["id"] == identity]
        ticks = np.array([round(float(row["time_seconds"]) / DT) for row in rows])
        tests = np.array([[row[key] == "True" for key in ("position_ok", "speed_ok", "yaw_ok")]
                          for row in rows])
        good = tests.all(axis=1)
        macro = np.flatnonzero(ticks % 10 == 0)
        eligible_since = None
        coarse_completion = None
        for at in macro:
            if good[at]:
                if eligible_since is None:
                    eligible_since = ticks[at]
                if ticks[at] - eligible_since >= 50 and coarse_completion is None:
                    coarse_completion = float(ticks[at] * DT)
            else:
                eligible_since = None
        bad_windows = []
        for before, after in zip(macro[:-1], macro[1:]):
            if ticks[after] - ticks[before] != 10:
                raise ValueError("Missing saved samples inside a macro interval")
            if good[before] and good[after] and not good[before:after + 1].all():
                failures = ~tests[before + 1:after]
                record = {
                    "id": identity, "start_seconds": float(ticks[before] * DT),
                    "end_seconds": float(ticks[after] * DT),
                    "both_boundary_samples_settled": True,
                    "interior_ineligible_20ms_samples": int(np.count_nonzero(~good[before + 1:after])),
                    "interior_position_bad": bool(failures[:, 0].any()),
                    "interior_speed_bad": bool(failures[:, 1].any()),
                    "interior_yaw_bad": bool(failures[:, 2].any()),
                }
                windows.append(record)
                bad_windows.append(record)
        actual = float(metrics[identity]["completion_seconds"])
        results.append({
            "id": identity, "available_macro_boundary_samples": len(macro),
            "both_boundaries_good_but_interior_bad_windows": len(bad_windows),
            "aliased_windows_with_position_bad": sum(w["interior_position_bad"] for w in bad_windows),
            "aliased_windows_with_speed_bad": sum(w["interior_speed_bad"] for w in bad_windows),
            "aliased_windows_with_yaw_bad": sum(w["interior_yaw_bad"] for w in bad_windows),
            "first_coarse_only_hold_completion_seconds": coarse_completion,
            "actual_complete_seconds": actual,
            "coarse_check_lead_seconds": actual - coarse_completion if coarse_completion is not None else None,
            "coarse_check_status": ("REACHED_IN_SAVED_SAMPLES" if coarse_completion is not None
                                    else "NOT_REACHED_WITHIN_SAVED_TRAJECTORY"),
        })
    csv_write(output / "macro_hold_aliasing.csv", results)
    csv_write(output / "macro_boundary_missed_windows.csv", windows)
    json_write(output / "macro_hold_aliasing_scope.json", {
        "source": "focus_postdoor_50hz.csv", "macro_interval_seconds": .2,
        "saved_interval_seconds": .02, "original_position_speed_yaw_thresholds_unchanged": True,
        "coarse_hold_rule": "First qualifying macro sample starts at zero; five subsequent consecutive qualifying 0.2 s boundaries complete one second.",
        "unobserved_future_not_extrapolated": True,
        "interpretation": "This is a downsampled predicate check on existing actual trajectories, not a newly completed physical task or an evaluation of a model prediction. It tests whether checking only macro boundaries can miss recorded subinterval violations.",
        "new_physics_steps": 0, "model_or_policy_updates": 0,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path,
                        default=ROOT / "work/server-adjoint-20261020-r2/preserved/results")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "work/server-adjoint-20261020-r2/preserved/follower-reference")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notes/go2-adjoint-diagnosis-2026-10-20/terminal")
    parser.add_argument("--figure-only", action="store_true",
                        help="Draw the one requested figure from existing analytical CSV only")
    parser.add_argument("--skill-scripts", type=Path,
                        help="Optional scipilot scripts folder for style and deterministic layout audit")
    args = parser.parse_args()
    if args.figure_only:
        plot_saved_figure(args.output, args.skill_scripts)
        return
    sources = Sources()
    episodes = []
    for seed in (11, 23, 37):
        for arm in ("frozen", "ordinary", "adjoint"):
            for round_number in (0, 4):
                episodes.extend(learning_episodes(args.runs, seed, arm, round_number, sources))
            if seed == 11:
                episodes.extend(learning_episodes(args.runs, seed, arm, 3, sources))
    episodes.extend(follower_episode(args.reference, case, sources)
                    for case in ("nominal", "narrow", "wide"))
    rows, attempts, zones, commands, traces = [], [], [], [], []
    for episode in episodes:
        result, holds, crossings, issued, trace = analyze_episode(episode)
        rows.append(result)
        attempts.extend(holds)
        zones.extend(crossings)
        commands.extend(issued)
        traces.extend(trace)
    lookup = {(r["seed"], r["arm"], r["round"], r["case"]): r for r in rows}
    paired = []
    for seed in (11, 23, 37):
        for case in ("nominal", "narrow", "wide"):
            paired.append(differences(lookup[seed, "ordinary", 4, case],
                                      lookup[seed, "adjoint", 4, case], "adjoint_minus_ordinary_final"))
    round_change = [differences(lookup[11, arm, 3, case], lookup[11, arm, 4, case], "round4_minus_round3")
                    for arm in ("frozen", "ordinary", "adjoint")
                    for case in ("nominal", "narrow", "wide")]
    initial_identity = []
    for seed in (11, 23, 37):
        for case in ("nominal", "narrow", "wide"):
            hashes = {arm: lookup[seed, arm, 0, case]["trajectory_state_command_hash"]
                      for arm in ("frozen", "ordinary", "adjoint")}
            initial_identity.append({"seed": seed, "case": case,
                                     "all_three_saved_trajectories_identical": len(set(hashes.values())) == 1,
                                     "state_command_hashes": hashes})
    aggregate = []
    for scope, arm in sorted({(r["scope"], r["arm"]) for r in rows}):
        group = [r for r in rows if r["scope"] == scope and r["arm"] == arm]
        fields = ("door_to_complete_seconds", "door_to_first_position_zone_seconds",
                  "position_entry_to_final_hold_start_seconds", "hold_reset_count",
                  "position_zone_exit_count_after_first_entry", "pose_good_but_speed_bad_seconds",
                  "postdoor_translation_tracking_rmse_m_s", "postdoor_path_length_m")
        aggregate.append({"scope": scope, "arm": arm, "episodes": len(group),
                          **{f"mean_{key}": statistics.mean(r[key] for r in group if r[key] is not None)
                             for key in fields}})
    args.output.mkdir(parents=True, exist_ok=True)
    csv_write(args.output / "episode_terminal_metrics.csv", rows)
    csv_write(args.output / "hold_attempts_and_resets.csv", attempts)
    csv_write(args.output / "position_zone_crossings.csv", zones)
    csv_write(args.output / "all_postdoor_macro_commands.csv", commands)
    csv_write(args.output / "focus_postdoor_50hz.csv", traces)
    coarse_hold_check(args.output)
    csv_write(args.output / "final_adjoint_vs_ordinary.csv", paired)
    csv_write(args.output / "seed11_round3_to_round4.csv", round_change)
    csv_write(args.output / "scope_summary.csv", aggregate)
    json_write(args.output / "round0_identity.json", initial_identity)
    json_write(args.output / "summary.json", {
        "schema": 1, "status": "SAVED_TRAJECTORY_ANALYSIS",
        "coverage": {"final": 27, "round0_all_arms": 27, "seed11_round3": 9, "follower": 3,
                     "total_episode_records": len(rows), "all_recorded_success": all(r["success"] for r in rows)},
        "source_availability": {
            "learning": "50 Hz physical48, task stage/time/actual hold clock, executed body commands and joint actions; 200 ms command decisions and total predicted score",
            "reference": "Original 50 Hz CSV with pose/velocity/commands/actions/torques; hold reconstructed from original predicates",
            "not_saved": "Full per-candidate predicted trajectories and score components, float64 full simulator state, or controlled counterfactual physical outcomes",
        },
        "thresholds_unchanged": {"position_m": 0.22, "planar_speed_m_s": 0.12,
                                 "final_yaw_rad": 0.2, "hold_seconds": 1.0},
        "scope_summary": aggregate,
        "interpretation": [
            "Stop is the original stage name after door clearance; it includes final approach and one-second hold, not only braking or standing.",
            "The authoritative learning-run hold clock comes from Task memory; position/speed/yaw decomposition uses saved float32 states. Threshold marginality and disagreement counts are reported.",
            "Time occupancies classify 20 ms endpoint samples; they are not continuous-time dwell proofs. Hold spans use endpoint time differences, not frame count times dt.",
            "Commands attached to a 50 Hz endpoint were executed over the preceding interval. Macro decision rows instead pair a newly chosen command with the pre-action state.",
            "Command change divided by 0.2 s is a command finite-difference rate, not commanded continuous physical acceleration; actual velocity differences are unfiltered at 50 Hz.",
            "Velocity-command tracking mismatch, movement requested near goal, and hold resets are observations. They do not alone assign causality to actor, CV, model, value or CEM.",
            "All final cases and round-zero arms are included. The focused dense CSV contains the three discussed examples, their seed11 round-three counterparts, and all three follower references.",
            "Known cases, repeated rounds and identical round-zero trajectories are not independent generalization samples.",
        ],
        "additional_physics_steps": 0, "additional_model_fitting_updates": 0,
        "additional_policy_updates": 0, "checkpoint_inference_or_planner_replay": False,
    })
    sources.add(Path(__file__))
    json_write(args.output / "source_manifest.json", sources.files)
    print(json.dumps({"episodes": len(rows), "hold_attempts": len(attempts),
                      "postdoor_macro_commands": len(commands), "new_physics_steps": 0,
                      "model_or_policy_updates": 0}))


if __name__ == "__main__":
    main()
