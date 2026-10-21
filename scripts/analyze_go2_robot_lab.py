"""Reproduce descriptive tables/figures from six saved frozen-Go2 evaluations.

Reads existing logs only: no simulator, policy, optimiser, training, or replay of
the full robot. 'Replay' below always means the already logged nominal-model
open-loop rollout driven by the actual recorded body commands.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from go2_robot_lab.task import Task, TaskSpec, wrap_angle

COLORS = {"mpc": "#0072B2", "follower": "#D55E00"}
STYLES = {"mpc": "-", "follower": "--"}
LABELS = {"mpc": "MPC", "follower": "Follower"}
CASES = ("nominal", "narrow", "wide")
CONTROLLERS = ("mpc", "follower")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.asarray([row[key] for row in rows], dtype=str if key == "stage" else float)
            for key in rows[0]}


def matrix(data, prefix, count):
    return np.column_stack([data[f"{prefix}_{i}"] for i in range(count)])


def rmse(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def vector_rmse(values):
    return float(np.sqrt(np.mean(np.sum(np.asarray(values) ** 2, axis=-1))))


def schema(value, path="", output=None):
    """Visit every JSON field without discarding structural or missingness checks."""
    output = [] if output is None else output
    if isinstance(value, dict):
        for key, entry in value.items():
            schema(entry, path + "." + key, output)
    elif isinstance(value, list):
        output.append({"path": path, "type": "array", "length": len(value)})
        for index, entry in enumerate(value):
            schema(entry, path + f"[{index}]", output)
    else:
        output.append({"path": path, "type": type(value).__name__,
                       "null": value is None,
                       "finite": bool(np.isfinite(value)) if isinstance(value, (float, int)) else None})
    return output


def profile_csv(path, data):
    columns = {}
    for name, values in data.items():
        record = {"type": str(values.dtype), "rows": len(values),
                  "missing": int(np.sum(values == "")) if values.dtype.kind in "US" else int(np.isnan(values).sum())}
        if values.dtype.kind not in "US":
            record.update(finite=bool(np.isfinite(values).all()), min=float(values.min()),
                          max=float(values.max()), mean=float(values.mean()))
        else:
            record["counts"] = {str(k): int(v) for k, v in zip(*np.unique(values, return_counts=True))}
        columns[name] = record
    time = data["time"]
    return {"file": path.name, "rows": len(time), "columns": columns,
            "strictly_increasing_time": bool(np.all(np.diff(time) > 0)),
            "time_start": float(time[0]), "time_end": float(time[-1]),
            "sample_interval_min": float(np.diff(time).min()),
            "sample_interval_max": float(np.diff(time).max())}


def collect(run_root, out):
    runs, metrics, exits, errors_by_stage, integrity = {}, [], [], [], []
    source_checks, inputs = [], []
    for case in CASES:
        for controller in CONTROLLERS:
            name = f"{controller}_{case}"
            directory = run_root / name
            s = load_json(directory / "summary.json")
            transitions = load_json(directory / "stage_transitions.json")
            decisions = load_json(directory / "decisions.json")
            trajectory = load_csv(directory / "trajectory_10hz.csv")
            pred = load_csv(directory / "prediction_errors.csv")
            replay = load_csv(directory / "recorded_command_replay.csv")
            json_profiles = {}
            for filename in ("summary.json", "stage_transitions.json", "decisions.json"):
                fields = schema(load_json(directory / filename))
                json_profiles[filename] = {
                    "visited_fields": len(fields),
                    "null_fields": [v["path"] for v in fields if v.get("null")],
                    "nonfinite_fields": [v["path"] for v in fields if v.get("finite") is False],
                    "root_keys": list(load_json(directory / filename)) if filename == "summary.json" else None,
                }
            csv_profiles = [profile_csv(directory / filename, data) for filename, data in (
                ("trajectory_10hz.csv", trajectory), ("prediction_errors.csv", pred),
                ("recorded_command_replay.csv", replay))]
            for path in sorted(directory.iterdir()):
                inputs.append({"file": path.relative_to(ROOT).as_posix(),
                               "sha256_bytes": hashlib.sha256(path.read_bytes()).hexdigest()})
            for relative, expected in s["source_sha256"].items():
                observed = hashlib.sha256((ROOT / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                source_checks.append({"run": name, "file": relative, "expected": expected,
                                      "observed": observed, "match": expected == observed})
            error = matrix(pred, "error", 6)
            replay_error = matrix(replay, "actual", 6) - matrix(replay, "replay", 6)
            replay_error[:, 2] = wrap_angle(replay_error[:, 2])
            recomputed = np.sqrt(np.mean(error ** 2, axis=0))
            replay_recomputed = np.sqrt(np.mean(replay_error ** 2, axis=0))
            checks = {
                "run": name, "csv": csv_profiles, "json": json_profiles,
                "decision_count": len(decisions), "transition_count": len(transitions),
                "decision_count_matches_summary": len(decisions) == s["planner_calls"],
                "transition_order": [event["to_stage"] for event in transitions],
                "one_interval_rmse_matches": bool(np.allclose(recomputed, s["one_interval_prediction_rmse_xy_yaw_vxy_yawrate"])),
                "open_loop_rmse_matches": bool(np.allclose(replay_recomputed, s["recorded_command_open_loop_rmse_xy_yaw_vxy_yawrate"])),
                "final_time_matches": bool(np.isclose(trajectory["time"][-1], s["elapsed_sim_seconds"])),
                "actual_replay_matches_decision_states": bool(np.allclose(matrix(replay, "actual", 6), [d["actual_state"] for d in decisions])),
                "policy_file_hash_unchanged": s["provenance"]["weight_sha256_before"] == s["provenance"]["weight_sha256_after"],
                "policy_tensor_hash_unchanged": s["provenance"]["tensor_sha256_before"] == s["provenance"]["tensor_sha256_after"],
            }
            integrity.append(checks)
            row = {"controller": controller, "case": case, "success": s["success"],
                   "elapsed_sim_s": s["elapsed_sim_seconds"],
                   "contact_ever": s["final_observation"]["obstacle_contact_ever"],
                   "fall_ever": s["final_observation"]["fallen_ever"],
                   "final_distance_m": s["task_metrics"]["goal_distance"],
                   "final_speed_m_s": s["task_metrics"]["planar_speed"],
                   "final_yaw_error_rad": s["task_metrics"]["goal_yaw_error"],
                   "absolute_mechanical_work_proxy_J": s["absolute_mechanical_work_proxy_J"],
                   "torque_squared_integral_Nm2_s": s["torque_squared_integral_Nm2_s"],
                   "path_rms_distance_m": np.sqrt(s["path_squared_error_integral_m2_s"] / s["elapsed_sim_seconds"]),
                   "max_abs_roll_pitch_rad": s["max_abs_roll_pitch_rad"],
                   "planner_p95_ms": 1000 * s["planner_latency_p95_seconds"],
                   "planner_max_ms": 1000 * s["planner_latency_max_seconds"],
                   "planner_period_exceedances": s["planner_latency_exceeds_period_count"],
                   "planner_calls": s["planner_calls"],
                   "cem_candidate_model_steps": s["planner_model_steps"],
                   "other_controller_model_steps": 40 * s["planner_calls"] if controller == "mpc" else 0,
                   "one_interval_position_rmse_m": vector_rmse(error[:, :2]),
                   "one_interval_yaw_rmse_rad": rmse(error[:, 2]),
                   "one_interval_velocity_rmse_m_s": vector_rmse(error[:, 3:5]),
                   "one_interval_yawrate_rmse_rad_s": rmse(error[:, 5]),
                   "open_loop_position_rmse_m": vector_rmse(replay_error[:, :2]),
                   "open_loop_yaw_rmse_rad": rmse(replay_error[:, 2]),
                   "open_loop_final_boundary_position_error_m": float(np.linalg.norm(replay_error[-1, :2])),
                   "open_loop_last_boundary_time_s": float(replay["time"][-1]),
                   "trajectory_rows": len(trajectory["time"]), "prediction_intervals": len(pred["time"])}
            metrics.append(row)
            for event in transitions:
                ob = event["observation"]
                exits.append({"controller": controller, "case": case,
                              "from_stage": event["from_stage"], "to_stage": event["to_stage"],
                              "time_s": ob["time"], "x_m": ob["position"][0], "y_m": ob["position"][1],
                              "yaw_rad": ob["yaw"], "vx_world_m_s": ob["velocity_world"][0],
                              "vy_world_m_s": ob["velocity_world"][1],
                              "planar_speed_m_s": float(np.linalg.norm(ob["velocity_world"][:2])),
                              "yawrate_rad_s": ob["yaw_rate"],
                              "vx_command_m_s": ob["command"][0], "vy_command_m_s": ob["command"][1],
                              "yawrate_command_rad_s": ob["command"][2],
                              "foot_contacts": "".join("1" if v else "0" for v in ob["foot_contact"])})
            for stage in ("straight", "turn", "door", "stop"):
                # Error at t[k+1] belongs to the actual command selected in stage at t[k].
                mask = np.asarray([d["stage"] == stage for d in decisions[:-1]])
                e = error[mask]
                errors_by_stage.append({"controller": controller, "case": case, "start_stage": stage,
                                        "intervals": len(e), "position_rmse_m": vector_rmse(e[:, :2]),
                                        "yaw_rmse_rad": rmse(e[:, 2]), "velocity_rmse_m_s": vector_rmse(e[:, 3:5]),
                                        "yawrate_rmse_rad_s": rmse(e[:, 5]),
                                        "yaw_mean_signed_error_rad": float(e[:, 2].mean())})
            runs[(controller, case)] = {"summary": s, "trajectory": trajectory, "errors": pred,
                                        "replay": replay, "transitions": transitions, "decisions": decisions}
    save_csv(out / "case_metrics.csv", metrics)
    save_csv(out / "stage_exit_states.csv", exits)
    save_csv(out / "prediction_by_start_stage.csv", errors_by_stage)
    save_json(out / "case_metrics.json", metrics)
    save_json(out / "data_profile.json", {"design": "3 declared conditions x 2 controllers x 1 deterministic rollout; no independent repeats",
                                          "runs": integrity, "source_hash_checks": source_checks,
                                          "all_source_hashes_match": all(r["match"] for r in source_checks),
                                          "input_file_hashes": inputs})
    return runs, metrics, exits, errors_by_stage


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "lines.linewidth": 1.35, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.dpi": 600, "axes.grid": True, "grid.alpha": 0.16})


def map_plot(ax, task):
    for item in task.obstacles():
        x, y = item["pos"][:2]; hx, hy = item["size"][:2]
        ax.add_patch(Rectangle((x - hx, y - hy), 2 * hx, 2 * hy,
                               facecolor="#D4D8DD", edgecolor="#8B9197", linewidth=0.4, zorder=0))
    route = task.reference(np.linspace(0, task.spec.total_length, 250))
    ax.plot(*route[:, :2].T, color="#6B6B6B", ls=":", lw=1.2, label="Route", zorder=1)
    ax.plot(*task.spec.goal, marker="*", color="black", ms=7, linestyle="none")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def save_figure(fig, out, stem, audit):
    fig.canvas.draw()
    issues = []
    if audit is not None:
        issues = audit(fig)
    else:
        renderer = fig.canvas.get_renderer()
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_visible() or not text.get_text(): continue
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                issues.append(("WARN", "Text outside figure: " + text.get_text()))
    fig.savefig(out / f"{stem}.png", dpi=600)
    fig.savefig(out / f"{stem}_preview.png", dpi=150)
    svg_path = out / f"{stem}.svg"
    fig.savefig(svg_path)
    # Matplotlib emits trailing spaces in multi-line path data; normalise text
    # without changing the vector geometry so the evidence stays diff-clean.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")
    with Image.open(out / f"{stem}_preview.png") as im:
        im.convert("L").save(out / f"{stem}_grayscale.png")
    plt.close(fig)
    return {"figure": stem, "issues": issues}


def figures(runs, out, audit):
    qa = []
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.8), constrained_layout=True)
    for i, (case, ax) in enumerate(zip(CASES, axes)):
        task = Task(TaskSpec(**runs[("mpc", case)]["summary"]["task"]))
        map_plot(ax, task)
        for controller in CONTROLLERS:
            run = runs[(controller, case)]; tr = run["trajectory"]
            ax.plot(tr["position_0"], tr["position_1"], color=COLORS[controller], ls=STYLES[controller], label=LABELS[controller])
            for event in run["transitions"][:3]:
                ax.plot(*event["observation"]["position"][:2], marker="o" if controller == "mpc" else "s",
                        color=COLORS[controller], ms=3, markerfacecolor="white")
        ax.set_xlim(-0.30, 4.12); ax.set_ylim(-0.45, 4.15)
        ax.set_title(f"({chr(97+i)}) {case}; door {task.spec.door_opening:.1f} m")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    qa.append(save_figure(fig, out, "fig1_full_task_paths", audit))

    fig, axes = plt.subplots(3, 2, figsize=(7.16, 6.1), sharex="col", sharey="row", constrained_layout=True)
    for col, controller in enumerate(CONTROLLERS):
        run = runs[(controller, "nominal")]; tr = run["trajectory"]
        task = Task(TaskSpec(**run["summary"]["task"]))
        refs = task.reference(task.project(np.column_stack([tr["position_0"], tr["position_1"]])))
        pairs = ((tr["planar_speed"], tr["command_0"], "Planar speed", "Body vx command", "Speed (m/s)"),
                 (tr["yaw"], refs[:, 2], "Body yaw", "Route heading", "Yaw (rad)"),
                 (tr["yaw_rate"], tr["command_2"], "Body yaw rate", "Yaw-rate command", "Yaw rate (rad/s)"))
        edges = [0.0] + [t["observation"]["time"] for t in run["transitions"]]
        for row, (actual, desired, alabel, dlabel, ylabel) in enumerate(pairs):
            ax = axes[row, col]
            for k in range(len(edges)-1):
                ax.axvspan(edges[k], edges[k+1], color="#9A9A9A", alpha=0.06 if k % 2 else 0.16, zorder=0)
                if row == 0:
                    ax.text((edges[k]+edges[k+1])/2, 0.96, ("straight", "turn", "door", "stop")[k],
                            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
            ax.plot(tr["time"], actual, color=COLORS[controller], label=alabel)
            ax.plot(tr["time"], desired, color="black", ls="--", lw=0.95, label=dlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"({chr(97 + row*2 + col)}) {LABELS[controller]}" if row == 0 else f"({chr(97 + row*2 + col)})", loc="left")
        axes[0, col].set_ylim(-0.05, 1.05)
        axes[2, col].set_xlabel("Simulation task time (s)")
    fig.legend([Line2D([0], [0], color="#555555", lw=1.4),
                Line2D([0], [0], color="black", lw=1.0, ls="--")],
               ["Actual body response", "Command / route heading"],
               loc="outside lower center", ncol=2, frameon=False)
    qa.append(save_figure(fig, out, "fig2_nominal_stage_response", audit))

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.0), constrained_layout=True)
    all_xy = np.concatenate([matrix(runs[(ctrl, "nominal")]["replay"], prefix, 6)[:, :2]
                              for ctrl in CONTROLLERS for prefix in ("actual", "replay")])
    for col, controller in enumerate(CONTROLLERS):
        run = runs[(controller, "nominal")]; replay = run["replay"]
        task = Task(TaskSpec(**run["summary"]["task"]))
        ax = axes[0, col]; map_plot(ax, task)
        ax.plot(replay["actual_0"], replay["actual_1"], color=COLORS[controller], label="Actual robot")
        ax.plot(replay["replay_0"], replay["replay_1"], color="black", ls="--", label="Model open loop")
        ax.set_xlim(min(-0.25, all_xy[:, 0].min() - 0.15), max(4.15, all_xy[:, 0].max() + 0.15))
        ax.set_ylim(min(-0.3, all_xy[:, 1].min() - 0.15), max(4.3, all_xy[:, 1].max() + 0.15))
        ax.plot(replay["replay_0"][-1], replay["replay_1"][-1], "x", color="black", ms=5)
        ax.set_title(f"({chr(97+col)}) {LABELS[controller]}: same recorded commands")
        ax.legend(loc="upper left", frameon=False)
        error = matrix(run["errors"], "error", 6)
        axes[1, 0].plot(run["errors"]["time"], 100 * np.linalg.norm(error[:, :2], axis=1),
                        color=COLORS[controller], ls=STYLES[controller], label=LABELS[controller])
        full_error = matrix(replay, "actual", 6) - matrix(replay, "replay", 6)
        axes[1, 1].plot(replay["time"], np.linalg.norm(full_error[:, :2], axis=1),
                        color=COLORS[controller], ls=STYLES[controller], label=LABELS[controller])
    axes[1, 0].set_title("(c) Reset from actual state every 0.2 s")
    axes[1, 1].set_title("(d) Open loop from initial state only")
    axes[1, 0].set_ylabel("Position error norm (cm)")
    axes[1, 1].set_ylabel("Position error norm (m)")
    for ax in axes[1]:
        ax.set_xlabel("Simulation task time (s)"); ax.set_ylim(bottom=0); ax.legend(frameon=False)
    qa.append(save_figure(fig, out, "fig3_recorded_command_prediction", audit))
    save_json(out / "figure_layout_audit.json", qa)


def write_readme(out, metrics, stages):
    lookup = {(r["controller"], r["case"]): r for r in metrics}
    lines = ["# 冻结 Go2 完整任务：保存结果的描述性分析", "",
             "本分析只读取已有六条轨迹，没有新增机器人仿真或训练。三条件 × 两控制器，各一条确定性评价；不进行显著性检验，不将时间点当作独立实验。", "",
             "| 条件 | MPC 完成时间/s | Follower 完成时间/s | MPC 机械功代理/J | Follower 机械功代理/J |",
             "|---|---:|---:|---:|---:|"]
    for case in CASES:
        m, f = lookup[("mpc",case)], lookup[("follower",case)]
        lines.append(f"| {case} | {m['elapsed_sim_s']:.2f} | {f['elapsed_sim_s']:.2f} | {m['absolute_mechanical_work_proxy_J']:.1f} | {f['absolute_mechanical_work_proxy_J']:.1f} |")
    lines.extend(["", "六条均无障碍接触、无跌倒并满足完整成功条件。这里的停止是平面位置、速度和朝向误差连续满足阈值 1 s，并非全身完全静止，末段仍有小幅偏航及关节运动。Follower 在三个条件均更快、累计绝对机械功代理和力矩平方积分均更低。这是当前固定参考速度、代价及控制器实现的描述，不能表述为 MPC 性能增益；两控制器有效行进速度不同，也不能将差异归于模型预测这一单一因素。机械功代理不是电池电能。", "",
                  "## 预测与跨阶段含义", "",
                  "每 0.2 s 从实际状态重置的名义模型短步位置误差较小，但同一模型从初态开始、沿实际记录机体命令连续开环回放会累积明显漂移。实际 MPC 闭环不断读取真实状态，所以长期开环漂移与任务成功可以同时出现。这不能单独证明 PINN 必要，也不是对尚未执行的 CEM 候选命令序列所作 4 s 机器人实测。", "",
                  "| 控制器 | 条件 | 0.2s 位置向量 RMSE/m | 开环位置向量 RMSE/m | 开环最后边界位置误差/m |",
                  "|---|---|---:|---:|---:|"])
    for r in metrics:
        lines.append(f"| {r['controller']} | {r['case']} | {r['one_interval_position_rmse_m']:.4f} | {r['open_loop_position_rmse_m']:.3f} | {r['open_loop_final_boundary_position_error_m']:.3f} |")
    lines.extend(["", "跨阶段状态表保存真实切换时刻的瞬时身体速度、偏航角/角速度、命令及四足接触；这些值受步态相位影响，不是阶段平均能力或因果效应。阶段误差表按命令发出时所属阶段分组，保留切换前后连续动态，不在阶段处复位模型实际状态以外的机器人状态。", "",
                  "## 文件与图注", "",
                  "- `case_metrics.csv/json`：每条轨迹的完整结果、终点状态、代价代理、实际求解时延和预测误差。",
                  "- `stage_exit_states.csv`：完整四次任务切换的实际状态；`prediction_by_start_stage.csv`：按命令开始阶段归类的短步误差。",
                  "- `data_profile.json`：所有 CSV 列类型/行数/缺失/有限值、全部 JSON 字段遍历、来源 hash 和指标重算核验。",
                  "- 图 1：三条件实际全程 xy、物理墙/门和预声明参考；圆/方点分别为两控制器阶段出口。每条曲线来自一次完整真实机器人仿真。",
                  "- 图 2：nominal 两控制器的实际平面速度、偏航角和角速度；虚线为机体前向速度命令、路线方向或偏航率命令。速度标量与机体前向命令定义不同，不将其差直接称跟踪误差。灰色区分顺序阶段，不是置信区间。",
                  "- 图 3：nominal 真实执行与沿同一实际记录命令的名义模型开环回放；下方分别是每 0.2 s 从实际状态重置和从初态单次开始的误差。回放不是另一条完整机器人仿真，也没有执行未采纳的候选计划。",
                  "- `figure_layout_audit.json`：出图布局检查；PNG 600 dpi、SVG 矢量和灰度预览并存。", "",
                  "真实求解时延只说明本机这次计算低于 0.2 s 指令周期。仿真物理时钟在求解时暂停，尚未验证带计算延迟的墙钟闭环部署。`planner_model_steps` / `cem_candidate_model_steps` 只计 CEM 候选评分，控制器每次另有 20 步反馈初值和 20 步最终最优序列预测；这些 40 步在 `other_controller_model_steps` 单列，真实求解耗时已包含。runner 为日志评价另做的两次名义步不在控制器求解预算内。真值机体位姿/速度由仿真器提供；公开策略原观测接口保持不变。", "",
                  "```powershell", "work/venv/Scripts/python.exe scripts/analyze_go2_robot_lab.py", "```", "",
                  "图型按 SciPilot 科研绘图工作流选择：完整路径与连续时间线直接展示原始结果，避免小样本均值柱、显著性与双 Y 轴。无训练、无新仿真、无新增物理约束。"])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "notes/go2-frozen-2026-10-19/runs")
    parser.add_argument("--output", type=Path, default=ROOT / "notes/go2-frozen-2026-10-19/analysis")
    parser.add_argument("--skill-scripts", type=Path, default=None, help="Optional external SciPilot scripts directory for layout audit")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    audit = None
    if args.skill_scripts is not None:
        sys.path.insert(0, str(args.skill_scripts))
        from visual_qa import audit_layout
        audit = audit_layout
    runs, metrics, stages, _ = collect(args.runs, args.output)
    style(); figures(runs, args.output, audit); write_readme(args.output, metrics, stages)
    save_json(args.output / "analysis_manifest.json", {
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256_normalized_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "additional_robot_simulations": 0, "training_updates": 0,
        "figure_selection": "Direct per-condition paths and continuous time-series; no means, confidence intervals or significance tests",
        "external_scipilot_layout_audit_used": audit is not None})
    print(json.dumps({"runs": len(metrics), "all_success": all(m["success"] for m in metrics),
                      "output": args.output.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    main()
