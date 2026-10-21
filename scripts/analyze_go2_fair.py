"""Analyse saved fair-Go2 evaluations; no new simulation, fitting or training.

The primary design is case x speed x controller (18 deterministic runs).
The separate six-candidate diagnostic is objective x ranking predictor (4 runs).
Its physical predictor is privileged and is never labelled a PINN benefit.
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
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from go2_robot_lab.fair_control import FairController
from go2_robot_lab.task import Task, TaskSpec, wrap_angle

CASES = ("nominal", "narrow", "wide")
SPEEDS = ("standard", "fast")
MODES = ("follower", "tracking", "mission")
OBJECTIVES = ("tracking", "mission")
PREDICTORS = ("nominal", "physics")
COLORS = {"follower": "#666666", "tracking": "#0072B2", "mission": "#D55E00"}
MARKERS = {"follower": "o", "tracking": "s", "mission": "^"}


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def json_write(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path, rows):
    if not rows:
        path.write_text("status\nNO_OBSERVATIONS\n", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(clean(rows))


def csv_read(path):
    if not path.exists(): return {}
    with path.open(encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
    if not rows: return {}
    return {k: np.asarray([r[k] for r in rows], dtype=str if k == "stage" else float) for k in rows[0]}


def matrix(data, prefix, count):
    return np.column_stack([data[f"{prefix}_{i}"] for i in range(count)])


def sha(path, normalized=False):
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if normalized else data).hexdigest()


def expected_labels():
    return ([f"primary_{speed}_{mode}_{case}" for speed in SPEEDS for case in CASES for mode in MODES]
            + [f"bank_fast_{objective}_{predictor}_nominal" for objective in OBJECTIVES for predictor in PREDICTORS])


def numeric_json_profile(value):
    result = {"numeric_leaves": 0, "nonfinite": 0, "null_leaves": 0, "string_leaves": 0}
    def walk(v):
        if isinstance(v, dict):
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
        elif v is None: result["null_leaves"] += 1
        elif isinstance(v, (float, int)):
            result["numeric_leaves"] += 1; result["nonfinite"] += int(not np.isfinite(v))
        elif isinstance(v, str): result["string_leaves"] += 1
    walk(value)
    return result


def csv_profile(data):
    if not data: return {"rows": 0, "status": "NO_DATA"}
    return {"rows": len(data["time"]), "time_strictly_increasing": bool(np.all(np.diff(data["time"]) > 0)),
            "columns": {k: {"kind": "category" if v.dtype.kind in "US" else "numeric",
                              "missing": int(np.sum(v == "")) if v.dtype.kind in "US" else int(np.isnan(v).sum()),
                              "nonfinite": 0 if v.dtype.kind in "US" else int((~np.isfinite(v)).sum())}
                        for k, v in data.items()}}


def chosen(scores, feasible):
    eligible = np.flatnonzero(feasible)
    if not len(eligible): eligible = np.arange(len(scores))
    return int(eligible[np.argmin(np.asarray(scores)[eligible])])


def ranks(scores, feasible):
    """Deterministic feasibility-first ranks; ties use candidate index."""
    order = sorted(range(len(scores)), key=lambda i: (not bool(feasible[i]), float(scores[i]), i))
    result = np.empty(len(scores), dtype=int)
    for rank, index in enumerate(order, 1): result[index] = rank
    return result


def command_check(controller, task, states_before, commands):
    states, cmd = np.asarray(states_before), np.asarray(commands)
    cap = controller.translation_cap(states, task)
    violation = np.maximum(0.0, np.linalg.norm(cmd[..., :2], axis=-1) - cap)
    box = max(float(np.max(cmd - controller.config.command_max)),
              float(np.max(np.asarray(controller.config.command_min) - cmd)), 0.0)
    return float(violation.max(initial=0)), box


def collect(run_root, output):
    runs, metrics, stages, audit, candidates, rankings = {}, [], [], [], [], []
    schemas = {}; source_checks = []; input_hashes = []
    for label in expected_labels():
        folder = run_root / label
        s = json_read(folder / "summary.json")
        d = json_read(folder / "decisions.json")
        transitions = json_read(folder / "stage_transitions.json")
        tr = csv_read(folder / "trajectory_10hz.csv")
        errors = csv_read(folder / "prediction_errors.csv")
        schemas[label] = {"summary_keys": list(s), "decision_keys": sorted(set(k for x in d for k in x)),
                          "json": {n: numeric_json_profile(v) for n, v in
                                   (("summary", s), ("decisions", d), ("transitions", transitions))},
                          "trajectory": csv_profile(tr), "prediction_errors": csv_profile(errors)}
        for path in sorted(folder.iterdir()):
            if path.is_file():
                input_hashes.append({"file": path.relative_to(ROOT).as_posix(),
                                     "sha256": sha(path), "sha256_lf": sha(path, normalized=True)})
        for path, expected in s["source_sha256"].items():
            observed = sha(ROOT / path, normalized=True)
            source_checks.append({"run": label, "file": path, "expected": expected, "observed": observed, "match": expected == observed})
        prior = json_read(ROOT / f"notes/go2-frozen-2026-10-19/runs/mpc_{s['case']}/summary.json")
        ctrl = FairController(s["mode"], s["speed_band"])
        task = Task(TaskSpec(**s["task"]))
        state = np.asarray([x["actual_state"] for x in d]); command = np.asarray([x["command"] for x in d])
        cap_error, box_error = command_check(ctrl, task, state, command) if len(d) else (None, None)
        e = matrix(errors, "error", 6) if errors else np.empty((0, 6))
        rmse = np.sqrt(np.mean(e * e, axis=0)) if len(e) else None
        budget = s["counterfactual_budget"]
        is_physics = s["suite"] == "bank" and s["predictor"] == "physics"
        record = {
            "run": label, "status": s["status"], "termination": s["termination"],
            "suite": s["suite"], "case": s["case"], "speed": s["speed_band"], "mode": s["mode"], "predictor": s["predictor"],
            "success": s["success"], "elapsed_sim_s": s["elapsed_sim_seconds"],
            "completion_time_s": s["elapsed_sim_seconds"] if s["success"] else None,
            "path_rms_m": s["path_rms_error_m"], "mechanical_work_proxy_J": s["absolute_mechanical_work_proxy_J"],
            "torque_squared_integral_Nm2_s": s["torque_squared_integral_Nm2_s"],
            "goal_distance_m": s["task_metrics"]["goal_distance"], "final_speed_m_s": s["task_metrics"]["planar_speed"],
            "final_yaw_error_rad": s["task_metrics"]["goal_yaw_error"],
            "contact_ever": s["final_observation"]["obstacle_contact_ever"], "fallen_ever": s["final_observation"]["fallen_ever"],
            "actual_policy_steps": s["policy_steps"], "actual_policy_inferences_including_warmup": s["actual_policy_inferences_including_warmup"],
            "actual_physics_steps_including_warmup": s["actual_physics_steps_including_warmup"],
            "planner_calls": s["planner_calls"], "planner_p95_s": s["planner_latency_p95_seconds"],
            "planner_max_s": s["planner_latency_max_seconds"], "period_exceedances": s["planner_latency_exceeds_period_count"],
            "cem_scored_model_steps": s["cem_scored_model_steps"], "bank_nominal_model_steps": s["bank_nominal_model_steps"],
            "other_controller_model_steps": s["planner_calls"] * (100 if s["suite"] == "bank" else 40 if s["mode"] != "follower" else 0),
            "oracle_rollouts": budget["rollouts_completed"], "oracle_physics_steps": budget["physics_steps"],
            "oracle_policy_inferences": budget["policy_inference_calls"], "oracle_wall_s": budget["wall_seconds"],
            "one_interval_position_rmse_m": float(np.sqrt(np.sum(rmse[:2] ** 2))) if rmse is not None else None,
            "one_interval_yaw_rmse_rad": rmse[2] if rmse is not None else None,
            "command_cap_violation_max": cap_error, "command_box_violation_max": box_error,
            "policy_unchanged": s["policy_unchanged"], "runtime_error": s["runtime_error"]}
        metrics.append(record)
        provenance = s["provenance"]
        cap_saved = np.asarray([x["translation_speed_ceiling"] for x in d])
        rowaudit = {
            "run": label, "task_equals_original": s["task"] == prior["task"],
            "scene_hash_equals_original": s["scene_matches_previous_evaluation"],
            "obstacles_equal_original": provenance["obstacles"] == prior["provenance"]["obstacles"],
            "robot_hash_equals_original": provenance["robot_mjcf_sha256"] == prior["provenance"]["robot_mjcf_sha256"],
            "policy_hash_equals_original": provenance["weight_sha256_before"] == prior["provenance"]["weight_sha256_before"],
            "policy_file_frozen": provenance["weight_sha256_before"] == provenance["weight_sha256_after"],
            "policy_tensors_frozen": provenance["tensor_sha256_before"] == provenance["tensor_sha256_after"],
            "no_training": s["policy_training_updates"] == s["model_training_updates"] == 0,
            "decision_count_matches": len(d) == s["planner_calls"],
            "actual_physics_steps_match": s["actual_physics_steps_including_warmup"] == 2000 + 20 * s["policy_steps"],
            "actual_inferences_match": s["actual_policy_inferences_including_warmup"] == 50 + s["policy_steps"],
            "cap_recomputed_matches": bool(np.allclose(cap_saved, ctrl.translation_cap(state, task))) if len(d) else None,
            "command_cap_ok": cap_error < 1e-9 if cap_error is not None else None,
            "command_box_ok": box_error < 1e-9 if box_error is not None else None,
            "prediction_rmse_matches": bool(np.allclose(rmse, s["one_interval_prediction_rmse_xy_yaw_vxy_yawrate"])) if rmse is not None else None,
            "oracle_rollouts_match": budget["rollouts_completed"] == 6 * len(d) if is_physics else budget["rollouts_completed"] == 0,
            "oracle_steps_match": budget["physics_steps"] == 24000 * len(d) if is_physics else budget["physics_steps"] == 0,
            "oracle_inferences_match": budget["policy_inference_calls"] == 1200 * len(d) if is_physics else budget["policy_inference_calls"] == 0,
            "oracle_steps_within_protocol": budget["physics_steps"] <= 4200000,
            "oracle_actual_steps_distinct": s["actual_physics_steps_including_warmup"] < 40000,
            "initial_state_equal_original": bool(np.allclose(s["initial_observation"]["position"], prior["initial_observation"]["position"]))}
        previous_time = 0.0
        for event in transitions:
            ob = event["observation"]
            stages.append({"run": label, "suite": s["suite"], "case": s["case"], "speed": s["speed_band"],
                           "mode": s["mode"], "predictor": s["predictor"], "stage": event["from_stage"],
                           "next_stage": event["to_stage"], "exit_time_s": ob["time"],
                           "stage_duration_s": ob["time"] - previous_time,
                           "x_m": ob["position"][0], "y_m": ob["position"][1], "yaw_rad": ob["yaw"],
                           "vx_world_m_s": ob["velocity_world"][0], "vy_world_m_s": ob["velocity_world"][1],
                           "yawrate_rad_s": ob["yaw_rate"], "observed_transition": True})
            previous_time = ob["time"]
        if not s["success"]:
            stages.append({"run": label, "suite": s["suite"], "case": s["case"], "speed": s["speed_band"],
                           "mode": s["mode"], "predictor": s["predictor"], "stage": s["task_metrics"]["stage"],
                           "next_stage": "UNCOMPLETED", "exit_time_s": None, "stage_duration_s": None,
                           "x_m": None, "y_m": None, "yaw_rad": None, "vx_world_m_s": None,
                           "vy_world_m_s": None, "yawrate_rad_s": None, "observed_transition": False})
        if s["suite"] == "bank":
            rowaudit.update(analyse_bank(label, s, d, ctrl, task, candidates, rankings))
        audit.append(rowaudit)
        runs[label] = {"summary": s, "decisions": d, "trajectory": tr, "transitions": transitions, "metrics": record}
    csv_write(output / "all_22_metrics.csv", metrics); json_write(output / "all_22_metrics.json", metrics)
    csv_write(output / "stage_times_and_exits.csv", stages)
    csv_write(output / "candidate_scores_and_ranks.csv", candidates)
    csv_write(output / "counterfactual_ranking_and_regret.csv", rankings)
    json_write(output / "data_and_contract_audit.json", {
        "run_checks": audit, "source_hash_checks": source_checks, "all_source_hashes_match": all(x["match"] for x in source_checks),
        "schema_and_missingness": schemas, "input_file_hashes": input_hashes,
        "command_contract": "Translation command norm cap and box checked at every 0.2s command boundary, not continuous body speed.",
        "execution_status": "All declared result files present; success/failure retained independently."})
    return runs, metrics, rankings


def analyse_bank(label, summary, decisions, ctrl, task, candidates, rankings):
    maximum_score_error = 0.0; max_nominal_cap = 0.0; max_physical_cap = 0.0
    max_selected_replay_error = 0.0; selection_matches = True; nominal_selection_matches = True
    for index, decision in enumerate(decisions):
        b = decision["model_ranking"]; initial = np.asarray(decision["actual_state"])
        previous = np.asarray(decision["last_command"])
        nstates, ncmd = np.asarray(b["nominal_states"]), np.asarray(b["nominal_applied_commands"])
        nfeas = np.asarray(b["nominal_feasible"], dtype=bool)
        nprev = np.concatenate([np.broadcast_to(initial, (len(nstates), 1, 6)), nstates[:, :-1]], axis=1)
        max_nominal_cap = max(max_nominal_cap, command_check(ctrl, task, nprev, ncmd)[0])
        physical = "physical_scores" in b
        if physical:
            pstates, pcmd = np.asarray(b["physical_states"]), np.asarray(b["physical_applied_commands"])
            pfeas = np.asarray(b["physical_feasible"], dtype=bool)
            pprev = np.concatenate([np.broadcast_to(initial, (len(pstates), 1, 6)), pstates[:, :-1]], axis=1)
            max_physical_cap = max(max_physical_cap, command_check(ctrl, task, pprev, pcmd)[0])
            if index + 1 < len(decisions):
                selected_error = np.asarray(decisions[index+1]["actual_state"]) - pstates[b["selected_index"], 0]
                selected_error[2] = wrap_angle(selected_error[2])
                max_selected_replay_error = max(max_selected_replay_error, float(np.max(np.abs(selected_error))))
        for objective in OBJECTIVES:
            nscores = np.asarray(b["nominal_scores"][objective]); nrank = ranks(nscores, nfeas)
            nbest = chosen(nscores, nfeas)
            nominal_selection_matches &= nbest == b["nominal_selected_indices"][objective]
            ncheck = ctrl.score_trajectory(nstates, ncmd, task, initial, previous, objective)
            maximum_score_error = max(maximum_score_error, float(np.max(np.abs(ncheck - nscores))))
            if physical:
                pscores = np.asarray(b["physical_scores"][objective]); prank = ranks(pscores, pfeas)
                pbest = chosen(pscores, pfeas)
                pcheck = ctrl.score_trajectory(pstates, pcmd, task, initial, previous, objective)
                maximum_score_error = max(maximum_score_error, float(np.max(np.abs(pcheck - pscores))))
                same_feasibility_regret = float(pscores[nbest] - pscores[pbest]) if pfeas.any() and pfeas[nbest] else None
                ranking_correlation = float(np.corrcoef(nrank, prank)[0, 1])
            else:
                pscores = prank = None; pbest = None; same_feasibility_regret = ranking_correlation = None
            for c, candidate in enumerate(b["candidate_names"]):
                candidates.append({"run": label, "decision": index, "time_s": decision["time"], "stage": decision["stage"],
                                   "objective": objective, "candidate_index": c, "candidate": candidate,
                                   "nominal_score": nscores[c], "nominal_feasible": nfeas[c], "nominal_rank": nrank[c],
                                   "physical_measured": physical, "physical_score": pscores[c] if physical else None,
                                   "physical_feasible": pfeas[c] if physical else None, "physical_rank": prank[c] if physical else None,
                                   "actually_selected": c == b["selected_index"]})
            rankings.append({"run": label, "run_objective": summary["mode"], "decision": index,
                             "time_s": decision["time"], "stage": decision["stage"], "objective_evaluated": objective,
                             "physical_measured": physical, "nominal_best_index": nbest, "physical_best_index": pbest,
                             "argmin_disagrees": nbest != pbest if physical else None,
                             "feasibility_first_rank_correlation": ranking_correlation,
                             "physical_score_regret_nominal_choice": same_feasibility_regret,
                             "regret_scope": "same objective, snapshot and candidate bank; only if nominal choice is physically feasible",
                             "physical_rank_nominal_choice": prank[nbest] if physical else None,
                             "nominal_choice_physically_feasible": pfeas[nbest] if physical else None,
                             "any_physical_feasible": bool(pfeas.any()) if physical else None,
                             "false_safe_count": int(np.sum(nfeas & ~pfeas)) if physical else None,
                             "false_unsafe_count": int(np.sum(~nfeas & pfeas)) if physical else None,
                             "nominal_feasible_count": int(nfeas.sum()),
                             "physical_feasible_count": int(pfeas.sum()) if physical else None})
            if objective == summary["mode"]:
                selection_matches &= b["selected_index"] == (pbest if physical else nbest)
    return {"bank_score_recomputed_max_abs_error": maximum_score_error,
            "bank_score_recomputed_matches": maximum_score_error < 1e-8,
            "nominal_candidate_command_cap_max_violation": max_nominal_cap,
            "physical_candidate_command_cap_max_violation": max_physical_cap if summary["predictor"] == "physics" else None,
            "selected_physical_first_interval_max_abs_state_error": max_selected_replay_error if summary["predictor"] == "physics" else None,
            "selection_matches_scores_and_feasibility": bool(selection_matches),
            "nominal_selection_indices_match": bool(nominal_selection_matches)}


def effect(left, right, metric):
    valid = bool(left["success"] and right["success"] and left[metric] is not None and right[metric] is not None)
    return {"left_run": left["run"], "right_run": right["run"], "metric": metric,
            "left_value": left[metric], "right_value": right[metric],
            "delta_right_minus_left": right[metric] - left[metric] if valid else None,
            "interpretation": "paired condition contrast, one deterministic result per cell" if valid else "UNKNOWN: at least one full task unsuccessful; elapsed partial time not a performance win"}


def comparisons(metrics, rankings, output):
    lookup = {m["run"]: m for m in metrics}; contrasts = []; history = []; effects = []
    values = ("elapsed_sim_s", "path_rms_m", "mechanical_work_proxy_J", "torque_squared_integral_Nm2_s")
    for case in CASES:
        for speed in SPEEDS:
            for left, right, kind in (("tracking", "mission", "objective"), ("follower", "mission", "controller")):
                for metric in values:
                    contrasts.append({"comparison": kind, "case": case, "speed": speed,
                                      **effect(lookup[f"primary_{speed}_{left}_{case}"], lookup[f"primary_{speed}_{right}_{case}"], metric)})
        for mode in MODES:
            for metric in values:
                contrasts.append({"comparison": "speed", "case": case, "speed": "standard_to_fast",
                                  **effect(lookup[f"primary_standard_{mode}_{case}"], lookup[f"primary_fast_{mode}_{case}"], metric)})
        for oldmode, newmode in (("follower", "follower"), ("mpc", "tracking")):
            old = json_read(ROOT / f"notes/go2-frozen-2026-10-19/runs/{oldmode}_{case}/summary.json")
            for speed in SPEEDS:
                new = lookup[f"primary_{speed}_{newmode}_{case}"]
                history.append({"case": case, "old_controller": oldmode, "new_run": new["run"],
                                "old_success": old["success"], "new_success": new["success"],
                                "old_time_s": old["elapsed_sim_seconds"], "new_time_s": new["elapsed_sim_s"],
                                "delta_time_s": new["elapsed_sim_s"] - old["elapsed_sim_seconds"] if old["success"] and new["success"] else None,
                                "interpretation": "historical descriptive change: governor and follower tangential term differ; not a uniquely identified causal percentage"})
    for metric in (*values, "planner_p95_s", "oracle_physics_steps"):
        nt = lookup["bank_fast_tracking_nominal_nominal"]; pt = lookup["bank_fast_tracking_physics_nominal"]
        nm = lookup["bank_fast_mission_nominal_nominal"]; pm = lookup["bank_fast_mission_physics_nominal"]
        for label, left, right in (("predictor_at_tracking", nt, pt), ("predictor_at_mission", nm, pm),
                                   ("objective_at_nominal", nt, nm), ("objective_at_physics", pt, pm)):
            effects.append({"effect": label, **effect(left, right, metric)})
        valid = all(r["success"] and r[metric] is not None for r in (nt, pt, nm, pm))
        left_difference = pt[metric] - nt[metric] if pt[metric] is not None and nt[metric] is not None else None
        right_difference = pm[metric] - nm[metric] if pm[metric] is not None and nm[metric] is not None else None
        effects.append({"effect": "interaction_(mission_physics-mission_nominal)-(tracking_physics-tracking_nominal)",
                        "metric": metric, "left_run": "tracking predictor effect", "right_run": "mission predictor effect",
                        "left_value": left_difference, "right_value": right_difference,
                        "delta_right_minus_left": (pm[metric]-nm[metric])-(pt[metric]-nt[metric]) if valid else None,
                        "interpretation": "conditional finite-bank interaction; privileged physics costs separate" if valid else "UNKNOWN: incomplete task cell"})
    aggregates = []
    for run in ("bank_fast_tracking_physics_nominal", "bank_fast_mission_physics_nominal"):
        for objective in OBJECTIVES:
            rs = [r for r in rankings if r["run"] == run and r["objective_evaluated"] == objective]
            valid_regret = [r["physical_score_regret_nominal_choice"] for r in rs if r["physical_score_regret_nominal_choice"] is not None]
            aggregates.append({"run": run, "objective": objective, "snapshots": len(rs),
                               "argmin_disagreement_count": sum(r["argmin_disagrees"] for r in rs),
                               "nominal_choice_infeasible_count": sum(not r["nominal_choice_physically_feasible"] for r in rs),
                               "all_candidates_physically_infeasible_count": sum(not r["any_physical_feasible"] for r in rs),
                               "false_safe_candidate_count": sum(r["false_safe_count"] for r in rs),
                               "false_unsafe_candidate_count": sum(r["false_unsafe_count"] for r in rs),
                               "feasible_regret_count": len(valid_regret),
                               "feasible_regret_median": float(np.median(valid_regret)) if valid_regret else None,
                               "feasible_regret_max": max(valid_regret) if valid_regret else None,
                               "rank_correlation_median": float(np.median([r["feasibility_first_rank_correlation"] for r in rs])) if rs else None})
    csv_write(output / "primary_paired_contrasts.csv", contrasts)
    csv_write(output / "historical_condition_changes.csv", history)
    csv_write(output / "bank_2x2_effects_and_interaction.csv", effects)
    csv_write(output / "counterfactual_ranking_summary.csv", aggregates)


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "lines.linewidth": 1.3, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18})


def save_fig(fig, output, name, audit):
    fig.canvas.draw()
    issues = audit(fig) if audit is not None else []
    data_bounds = []
    for index, ax in enumerate(fig.axes):
        xmin, xmax = sorted(ax.get_xlim()); ymin, ymax = sorted(ax.get_ylim())
        for line in ax.lines:
            x, y = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            clipped = finite & ((x < xmin - 1e-9) | (x > xmax + 1e-9)
                                | (y < ymin - 1e-9) | (y > ymax + 1e-9))
            if clipped.any():
                data_bounds.append({"panel": index, "line": line.get_label(), "clipped_samples": int(clipped.sum())})
    fig.savefig(output / (name + ".png"), dpi=600)
    fig.savefig(output / (name + "_preview.png"), dpi=150)
    svg_path = output / (name + ".svg")
    fig.savefig(svg_path)
    # Matplotlib emits trailing blanks in path data. Preserve SVG content while
    # making the generated text clean under git diff --check on every platform.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8", newline="\n")
    with Image.open(output / (name + "_preview.png")) as img: img.convert("L").save(output / (name + "_grayscale.png"))
    plt.close(fig)
    return {"figure": name, "issues": issues, "data_clipping": data_bounds,
            "all_line_data_inside_axes": not data_bounds}


def figures(runs, output, audit=None):
    qa = []; setup_style()
    fields = (("elapsed_sim_s", "Task time (s)"), ("path_rms_m", "Path RMS error (m)"), ("mechanical_work_proxy_J", "Absolute work proxy (J)"))
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 6.2), sharex=True, sharey="row", constrained_layout=True)
    for col, speed in enumerate(SPEEDS):
        for row, (field, ylabel) in enumerate(fields):
            ax = axes[row, col]
            for case_index, case in enumerate(CASES):
                y = []; xx = []
                for mode_index, mode in enumerate(MODES):
                    m = runs[f"primary_{speed}_{mode}_{case}"]["metrics"]; x = case_index + (mode_index - 1) * .19
                    ax.plot(x, m[field], marker=MARKERS[mode] if m["success"] else "x", color=COLORS[mode],
                            markersize=5, linestyle="none", label=mode if case_index == 0 else None)
                    xx.append(x); y.append(m[field])
                ax.plot(xx, y, color="#AAAAAA", lw=.7, zorder=0)
            ax.set_title(f"({chr(97+row*2+col)}) {speed}", loc="left"); ax.set_ylabel(ylabel)
            ax.set_xticks(range(3), CASES)
    # Shared axes must use all plotted conditions; set_ylim during point insertion
    # otherwise freezes autoscaling before subsequent higher-valued points exist.
    for row, (field, _) in enumerate(fields):
        maximum = max(runs[f"primary_{speed}_{mode}_{case}"]["metrics"][field]
                      for speed in SPEEDS for case in CASES for mode in MODES)
        axes[row, 0].set_ylim(0, maximum * 1.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    qa.append(save_fig(fig, output, "fig1_primary_paired_tradeoffs", audit))
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0), constrained_layout=True)
    bankfields = (*fields, ("planner_p95_s", "Planning p95 (s; log scale)"))
    for i, ((field, ylabel), ax) in enumerate(zip(bankfields, axes.flat)):
        for objective_index, objective in enumerate(OBJECTIVES):
            xx, yy = [], []
            for predictor_index, predictor in enumerate(PREDICTORS):
                m = runs[f"bank_fast_{objective}_{predictor}_nominal"]["metrics"]
                x = objective_index + (predictor_index-.5) * .2
                color = "#0072B2" if predictor == "nominal" else "#D55E00"
                ax.plot(x, m[field], marker=("o" if predictor == "nominal" else "s") if m["success"] else "x",
                        color=color, linestyle="none", ms=5, label=predictor if objective_index == 0 else None)
                xx.append(x); yy.append(m[field])
            ax.plot(xx, yy, color="#999999", lw=.8)
        ax.set_xticks([0, 1], OBJECTIVES); ax.set_xlim(-.35, 1.35); ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97+i)}) Finite candidate bank", loc="left")
        observed = [runs[f"bank_fast_{objective}_{predictor}_nominal"]["metrics"][field]
                    for objective in OBJECTIVES for predictor in PREDICTORS]
        if field == "planner_p95_s":
            ax.set_yscale("log"); ax.set_ylim(min(observed) / 1.4, max(observed) * 1.4)
        else:
            ax.set_ylim(0, max(observed) * 1.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    qa.append(save_fig(fig, output, "fig2_bank_objective_predictor", audit))
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.9), sharex="col", sharey="row", constrained_layout=True)
    for col, mode in enumerate(MODES):
        r = runs[f"primary_fast_{mode}_nominal"]; tr = r["trajectory"]
        task = Task(TaskSpec(**r["summary"]["task"]))
        refs = task.reference(task.project(matrix(tr, "position", 3)[:, :2]))
        edges = [0.] + [e["observation"]["time"] for e in r["transitions"]]
        if edges[-1] < tr["time"][-1] - 1e-8: edges.append(float(tr["time"][-1]))
        for row, (actual, target, label) in enumerate(((tr["planar_speed"], np.linalg.norm(matrix(tr, "command", 3)[:, :2], axis=1), "Speed (m/s)"),
                                                      (tr["yaw"], refs[:, 2], "Yaw (rad)"))):
            ax = axes[row, col]
            for k, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                ax.axvspan(left, right, color="#999999", alpha=.14 if k % 2 == 0 else .05)
                if row == 0:
                    ax.text((left+right)/2, .97, ("S", "T", "D", "Stop")[min(k, 3)],
                            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
            ax.plot(tr["time"], actual, color=COLORS[mode], label="Actual")
            ax.plot(tr["time"], target, color="black", ls="--", lw=.9, label="Command / route")
            ax.set_title(f"({chr(97+row*3+col)}) {mode}" + (" [FAIL]" if not r["metrics"]["success"] else ""), loc="left")
            ax.set_ylabel(label)
        axes[1,col].set_xlabel("Task time (s)")
    speed_max = max(float(runs[f"primary_fast_{mode}_nominal"]["trajectory"]["planar_speed"].max()) for mode in MODES)
    axes[0,0].set_ylim(-.05, max(.95, speed_max + .12))
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    qa.append(save_fig(fig, output, "fig3_nominal_fast_stage_response", audit))
    json_write(output / "figure_layout_audit.json", qa)


def readme(output, metrics):
    lookup = {r["run"]: r for r in metrics}
    lines = ["# Go2 公平条件对照：保存结果的分析", "",
             "本目录只分析已保存的22条评价，不启动机器人仿真、不拟合模型、不训练策略。每格一次确定性运行；三个条件不是独立重复，不进行显著性检验。", "",
             "| suite | case | speed | controller/objective | predictor | success | elapsed/s | path RMS/m | work proxy/J |",
             "|---|---|---|---|---|---|---:|---:|---:|"]
    for r in metrics:
        lines.append(f"| {r['suite']} | {r['case']} | {r['speed']} | {r['mode']} | {r['predictor']} | {r['success']} | {r['elapsed_sim_s']:.2f} | {r['path_rms_m']:.4f} | {r['mechanical_work_proxy_J']:.1f} |")
    lines.extend(["", "## 本轮条件内读数", ""])
    for speed in SPEEDS:
        pairs = [(lookup[f"primary_{speed}_tracking_{case}"], lookup[f"primary_{speed}_mission_{case}"]) for case in CASES]
        if all(a["success"] and b["success"] for a, b in pairs):
            deltas = "/".join(f"{b['elapsed_sim_s']-a['elapsed_sim_s']:+.2f}" for a, b in pairs)
            lines.append(f"- {speed} 下目标tracking→mission的完整时间变化，按nominal/narrow/wide顺序为 {deltas} s。负号代表mission更快。路径与功代价另列，不能只看时间宣布全面优势。")
        else:
            lines.append(f"- {speed} 目标对照存在未完成格，完整任务收益UNKNOWN，保留所有原轨迹。")
    for objective in OBJECTIVES:
        n = lookup[f"bank_fast_{objective}_nominal_nominal"]; p = lookup[f"bank_fast_{objective}_physics_nominal"]
        if n["success"] and p["success"]:
            lines.append(f"- 候选库{objective}：nominal→physics时间 {n['elapsed_sim_s']:.2f}→{p['elapsed_sim_s']:.2f} s；路径RMS {n['path_rms_m']:.5f}→{p['path_rms_m']:.5f} m；机械功代理 {n['mechanical_work_proxy_J']:.2f}→{p['mechanical_work_proxy_J']:.2f} J。physics额外 {p['oracle_physics_steps']:,} 物理步/{p['oracle_policy_inferences']:,} 策略推理。")
    lines.append("- 历史到本轮同时改变了多项控制条件；用历史旧MPC大幅慢于旧follower的差距直接证明模型瓶颈是不成立的。当前物理分叉能复现选中首步，并不保证内部代理目标对完整时间的排序正确。")
    lines.extend(["", "## 解释边界与文件", "",
                  "- `all_22_metrics.csv/json` 保留全部成功、失败、超时和执行异常。失败的elapsed是已运行时间，不是完成时间；`completion_time_s`为空，图中用×标识。任务未完成的时间对比为UNKNOWN。",
                  "- `stage_times_and_exits.csv` 保存阶段时间和瞬时出口状态；未完成阶段明确列UNCOMPLETED。停止仍是平面位置、速度和朝向误差保持阈值1s，不代表全身完全静止。",
                  "- `primary_paired_contrasts.csv`：同speed同case的tracking→mission目标表达对照、follower→mission对照和同mode跨speed条件对照。差值是right−left，不作总体统计推断。",
                  "- `historical_condition_changes.csv`：与旧六条结果的描述性比较。命令模长governor及跟随器切向前视增益共同改变，不能计算唯一的独立因果贡献百分比。",
                  "- `bank_2x2_effects_and_interaction.csv`：固定候选构造下objective×predictor对完整任务的条件效应。interaction=(mission physics−mission nominal)−(tracking physics−tracking nominal)。它不是跨objective内部损失的差值，也不是统计ANOVA结论。",
                  "- `candidate_scores_and_ranks.csv`：每快照六候选分别在tracking/mission下的名义分数与已测物理分数、可行性与排名。不同objective的score量纲与权重不同，不能直接比较数值大小。",
                  "- `counterfactual_ranking_and_regret.csv`/`counterfactual_ranking_summary.csv`：仅在physics组访问的实际状态上计算同objective、同快照、同候选库regret。nominal组未测物理分叉留空；名义选择物理不可行时只报告可行性错误，不把标量差称可行regret。",
                  "- 各物理分叉共享raw序列与governor规则；governor根据各预测/物理状态限速，因此名义与物理的最终applied命令不保证逐值相同。不同组状态随闭环分歧，所以后续候选数值也会改变；不是跨组始终共享完全相同的序列。",
                  "- 精确physics拥有完整模拟器/策略历史特权且追加大量物理步；预算、推理次数与时延单列。它同时改变轨迹预测和可行性判断（名义身体轮廓 versus 实际全机器人接触/跌倒），本轮没有再把这两项拆开归因。可能排序改变却不缩短完整任务，须按完整结果判断。即使有收益也不称PINN收益或等算力在线优越性。",
                  "- 所有候选物理不可行时，固定实现退回最低代理score并只执行首0.2s，随后重规划。完整4s分叉不可行不意味着实际下一0.2s已经失败；全程成功也不能反推所有候选预测安全或已有硬安全保证。",
                  "- `data_and_contract_audit.json` 验证原TaskSpec/实体场景/hash、冻结权重、实际和分叉计数、评分重算、选中物理分叉首步与真实下一状态、命令边界模长约束，以及所有输入列/JSON字段有限值。",
                  "- 输入文件hash同时保留`sha256`（本地原始文件字节）和`sha256_lf`（先将CRLF规范化为LF）。Git可能规范化文本换行；公开仓库跨机器核验JSON/CSV时使用`sha256_lf`，本地原始字节存档使用`sha256`。运行源码与分析脚本hash同样采用LF规范化口径。",
                  "- CEM计数只含候选评分。primary MPC每次另40名义步，bank每次另100名义步加单列120 bank步；runner日志预测每次另1名义步。实际求解时间包含控制器内全部计算。",
                  "- 物理时钟在求解时暂停。低于0.2s只说明本机计算耗时；physics超过周期同样如实报告，不冒称真实墙钟部署。机械功代理不是电池电能。", "",
                  "图1为18条同条件配对点与路径/功取舍；灰线仅连接同一个case的三种控制器，不连接不同case。图2为固定候选库2×2结果及特权计算，规划时延使用明确log轴。图3为nominal-fast三种模式速度与偏航连续轨迹，S/T/D/Stop为直行/转弯/过门/停止，灰色不是置信区间。全部图以单条原始运行表示，不画均值柱或显著性。", "",
                  "```powershell", "work/venv/Scripts/python.exe scripts/analyze_go2_fair.py", "```", "",
                  "可选`--skill-scripts`启用本地SciPilot布局审计；数据和成图不依赖该路径。"])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=ROOT / "notes/go2-fair-2026-10-19/runs")
    p.add_argument("--output", type=Path, default=ROOT / "notes/go2-fair-2026-10-19/analysis")
    p.add_argument("--skill-scripts", type=Path, default=None)
    args = p.parse_args()
    missing = [name for name in expected_labels() if not (args.runs / name / "summary.json").exists()]
    if missing:
        raise SystemExit("UNKNOWN: declared evaluations not all available; no final analysis written. Missing: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    audit = None
    if args.skill_scripts is not None:
        sys.path.insert(0, str(args.skill_scripts))
        from visual_qa import audit_layout
        audit = audit_layout
    runs, metrics, rankings = collect(args.runs, args.output)
    comparisons(metrics, rankings, args.output)
    figures(runs, args.output, audit); readme(args.output, metrics)
    json_write(args.output / "analysis_manifest.json", {
        "script": Path(__file__).relative_to(ROOT).as_posix(), "script_sha256_lf": sha(Path(__file__), normalized=True),
        "protocol_sha256_lf": sha(ROOT / "notes/go2-fair-2026-10-19/PROTOCOL.md", normalized=True),
        "config_sha256_lf": sha(ROOT / "configs/go2_fair_comparison.json", normalized=True),
        "expected_runs": 22, "observed_runs": len(metrics), "additional_simulation_steps": 0, "training_updates": 0,
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "external_layout_audit": audit is not None})
    print(json.dumps({"runs": len(metrics), "successes": sum(r["success"] for r in metrics),
                      "output": args.output.relative_to(ROOT).as_posix()}))


if __name__ == "__main__": main()
