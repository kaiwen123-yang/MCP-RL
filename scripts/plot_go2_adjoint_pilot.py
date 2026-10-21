"""Plot saved Go2 pilot analysis only; never run physics or load checkpoints.

Example:
  python scripts/plot_go2_adjoint_pilot.py --analysis notes/pilot/analysis --output notes/pilot/figures
  # Optionally add --reference <same-environment-follower.json>.

Each column is one training seed. Cases are known development conditions, not
independent generalization trials. Failed elapsed times never become completion
times. Missing/partial evaluations and unavailable costs break the curves.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np


STYLES = {
    "frozen": dict(label="Frozen feedback", color="#666666", linestyle=":", marker="s"),
    "ordinary": dict(label="Ordinary PPO", color="#0072B2", linestyle="--", marker="o"),
    "adjoint": dict(label="Adjoint CV-PPO", color="#D55E00", linestyle="-", marker="^"),
}
AXES = {
    "physics": {
        "field": "cumulative_standalone_physics_steps_including_common_and_stands",
        "scale": 1e6,
        "label": "Cumulative physics\nsteps (million)",
        "title": "Performance versus cumulative physics",
    },
    "wall": {
        "field": "reported_cumulative_wall_seconds_including_common",
        "scale": 60.0,
        "label": "Recorded elapsed\ntime (min)",
        "title": "Performance versus recorded elapsed time",
    },
}


def numeric(value):
    if value is None or str(value).strip().lower() in ("", "none", "null", "nan"):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def flag(value):
    return value is True or str(value).lower() == "true"


def load_analysis(directory):
    summary_path = directory / "summary.json"
    rounds_path = directory / "round_summary.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != 1:
        raise ValueError("Expected schema 1 from analyze_go2_adjoint_pilot.py.")
    protocol = summary["protocol"]
    with rounds_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "seed", "arm", "round", "status", "expected_evaluation_cells",
        "completed_evaluation_cells", "successes_observed", "failures_observed",
        "success_time_mean_seconds", "cumulative_cost_accounting_complete",
        *(axis["field"] for axis in AXES.values()),
    }
    if rows and not required.issubset(rows[0]):
        missing = ", ".join(sorted(required - set(rows[0])))
        raise ValueError(f"Re-run the current analyzer; missing columns: {missing}")
    lookup = {}
    for row in rows:
        key = (int(row["seed"]), row["arm"], int(row["round"]))
        if key in lookup:
            raise ValueError(f"Duplicate evaluation checkpoint in analysis: {key}")
        lookup[key] = row
    points = []
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            for round_number in range(int(protocol["rounds"]) + 1):
                row = lookup.get((seed, arm, round_number), {})
                status = row.get("status", "NOT_RUN")
                expected = numeric(row.get("expected_evaluation_cells"))
                completed = numeric(row.get("completed_evaluation_cells"))
                successes = numeric(row.get("successes_observed"))
                failures = numeric(row.get("failures_observed"))
                complete = (status == "COMPLETE_EVALUATION"
                            and expected is not None and expected > 0
                            and completed == expected and successes is not None)
                completion = numeric(row.get("success_time_mean_seconds"))
                # This is deliberately not elapsed_seconds or observed_elapsed_sum.
                if not complete or successes <= 0:
                    completion = None
                physics = numeric(row.get(AXES["physics"]["field"]))
                if not flag(row.get("cumulative_cost_accounting_complete")):
                    physics = None
                points.append({
                    "seed": seed,
                    "arm": arm,
                    "round": round_number,
                    "evaluation_status": status,
                    "source_row_present": bool(row),
                    "complete_evaluation": complete,
                    "expected_cases": int(expected) if expected is not None else None,
                    "completed_cases": int(completed) if completed is not None else None,
                    "not_run_cases": numeric(row.get("not_run_cells")),
                    "incomplete_episode_cases": numeric(row.get("incomplete_episode_cells")),
                    "successes": int(successes) if successes is not None else None,
                    "failures": int(failures) if failures is not None else None,
                    "success_fraction": successes / expected if complete else None,
                    "successful_completion_mean_seconds": completion,
                    "physics": physics,
                    "wall": numeric(row.get(AXES["wall"]["field"])),
                    "wall_basis": row.get("reported_cumulative_wall_basis") or None,
                    "common_wall_basis": row.get("common_wall_basis") or None,
                })
    return summary, points, (summary_path, rounds_path), len(rows)


def load_reference(path, protocol):
    if path is None:
        return None
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    cases = record.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("--reference must contain a nonempty cases array.")
    names = [case["case"] for case in cases]
    if len(names) != len(set(names)) or set(names) != set(protocol["known_cases"]):
        raise ValueError("Reference must contain each declared development case exactly once.")
    normalized, times = [], []
    for case in cases:
        if not isinstance(case.get("success"), bool):
            raise ValueError("Each reference case must declare a boolean success.")
        time = numeric(case.get("completion_seconds")) if case["success"] else None
        if case["success"] and (time is None or time <= 0):
            raise ValueError("A successful reference case requires its actual completion_seconds.")
        if time is not None:
            times.append(time)
        normalized.append({"case": case["case"], "success": case["success"],
                           "completion_seconds": time})
    calculated = sum(times) / len(times) if times else None
    reported = numeric(record.get("mean_success_completion_seconds"))
    if reported is not None and (calculated is None or not math.isclose(
            reported, calculated, rel_tol=1e-8, abs_tol=1e-8)):
        raise ValueError("Reference mean disagrees with its successful-case completion times.")
    return {
        "label": record.get("label", "Fixed follower"),
        "case_count": len(cases),
        "success_count": len(times),
        "cases": normalized,
        "mean_success_completion_seconds": calculated,
        "measurement_identity": record.get("measurement_identity", record.get("identity")),
        "separate_cost_record": record.get("cost", record.get("costs", record.get("budget"))),
        "source_file": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_cost_added_to_method_axes": False,
        "independent_reference_evaluations": len(cases),
        "interpretation": (
            "One fixed, untrained evaluation of the declared cases in the current server/runtime. "
            "The same horizontal guide is repeated across seed panels, not counted as new "
            "seed-level replicates or nine evaluations. Its measurement cost is separate."),
    }


def style_for(arm, index):
    if arm in STYLES:
        return STYLES[arm]
    return dict(label=arm, color=plt.get_cmap("tab10")(index % 10),
                linestyle=("-", "--", ":")[index % 3],
                marker=("o", "s", "^", "D")[index % 4])


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.3,
        "svg.fonttype": "none",
        "svg.hashsalt": "go2-adjoint-pilot-task-results-v1",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def finite_values(points, key):
    return [p[key] for p in points if p.get(key) is not None]


def numeric_profile(points, key):
    values = finite_values(points, key)
    return {
        "known": len(values),
        "missing": len(points) - len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def data_profile(summary, points, raw_rows):
    """Equivalent lightweight EDA using stdlib, without adding pandas."""
    protocol = summary["protocol"]
    grouped = []
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            group = [p for p in points if p["seed"] == seed and p["arm"] == arm]
            grouped.append({
                "seed": seed, "arm": arm, "declared_checkpoints": len(group),
                "source_rows_present": sum(p["source_row_present"] for p in group),
                "checkpoint_status_counts": dict(Counter(p["evaluation_status"] for p in group)),
                "complete_evaluation_checkpoints": sum(p["complete_evaluation"] for p in group),
                "checkpoints_with_failed_cases": sum((p["failures"] or 0) > 0 for p in group),
                "checkpoints_with_no_success": sum(
                    p["complete_evaluation"] and p["successes"] == 0 for p in group),
                "success_fraction": numeric_profile(group, "success_fraction"),
                "successful_completion_mean_seconds": numeric_profile(
                    group, "successful_completion_mean_seconds"),
                "physics": numeric_profile(group, "physics"),
                "recorded_elapsed_seconds": numeric_profile(group, "wall"),
            })
    return {
        "raw_round_summary_csv_row_count": raw_rows,
        "declared_checkpoint_count": len(points),
        "checkpoint_grain": "one training seed x one method x one declared round",
        "rounds_including_initial": list(range(int(protocol["rounds"]) + 1)),
        "seeds": protocol["seeds"],
        "arms": protocol["arms"],
        "known_development_cases": protocol["known_cases"],
        "expected_case_evaluations_per_checkpoint": (
            len(protocol["known_cases"]) * int(protocol["evaluation_episodes_per_case"])),
        "checkpoint_status_counts": dict(Counter(p["evaluation_status"] for p in points)),
        "by_seed_and_method": grouped,
        "completion_time_source": "success_time_mean_seconds; never elapsed time",
        "completion_time_estimand": (
            "Within a seed/method/round having the complete declared evaluation set, "
            "mean actual completion seconds over successful cases only. Null if there "
            "are no successes or evaluation is incomplete. It is not an unconditional "
            "task time, seed-level confidence interval, or independent generalization sample."),
        "missing_values": (
            "NOT_RUN, partial evaluations, no successful time and unavailable costs remain "
            "null; none are imputed. Complete evaluations with failures retain their success fraction."),
        "case_independence": "Known conditions are not independent training seeds or unseen tasks.",
        "profiling_backend": "Python stdlib and NumPy; no dependency installation",
    }


def grayscale_figure(figure):
    """Re-render native Matplotlib artists in gray, never edit a saved image."""
    def gray(color):
        r, g, b, alpha = to_rgba(color)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return luminance, luminance, luminance, alpha

    for line in figure.findobj(Line2D):
        # Resolve all original colors before mutating automatic marker colors.
        colors = (line.get_color(), line.get_markeredgecolor(), line.get_markerfacecolor())
        line.set_color(gray(colors[0]))
        line.set_markeredgecolor(gray(colors[1]))
        line.set_markerfacecolor(gray(colors[2]))
    for text in figure.findobj(Text):
        text.set_color(gray(text.get_color()))


def layout_check(figure):
    """Check visible figure text bounds; visual inspection is still required."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    outside = []
    artists = list(figure.texts) + list(figure.legends)
    for axis in figure.axes:
        artists += [axis.title, axis.xaxis.label, axis.yaxis.label]
        artists += axis.get_xticklabels() + axis.get_yticklabels()
        artists += list(axis.texts)
    for artist in artists:
        if not artist.get_visible():
            continue
        if hasattr(artist, "get_text") and not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if (box.x0 < canvas.x0 - 1 or box.y0 < canvas.y0 - 1
                or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1):
            outside.append(artist.get_text() if hasattr(artist, "get_text") else "legend")
    return {"text_outside_canvas": outside,
            "visual_inspection_required": True}


def make_figure(summary, points, budget, reference=None):
    protocol, spec = summary["protocol"], AXES[budget]
    seeds, arms = protocol["seeds"], protocol["arms"]
    columns = len(seeds)
    # Three seeds fit an IEEE-sized double-column figure without pooled means.
    width = 7.16 if columns <= 3 else 2.4 * columns
    figure, axes = plt.subplots(2, columns, figsize=(width, 5.6),
                                squeeze=False, sharex=True, sharey="row")
    figure.subplots_adjust(left=0.105, right=0.985, top=0.75, bottom=0.28,
                           hspace=0.46, wspace=0.24)
    snapshot = summary["status"] != "COMPLETED"
    heading = f"Go2 fixed-task pilot | {spec['title']}"
    figure.suptitle(heading, y=0.985, fontsize=10.0, fontweight="bold")
    status_label = "PARTIAL SNAPSHOT" if snapshot else "COMPLETED FIXED PROTOCOL"
    figure.text(0.5, 0.945,
                f"{status_label}  |  Separate training seeds; known development cases",
                ha="center", fontsize=8.2, color="#8D5500" if snapshot else "#444444")
    handles = []
    for index, arm in enumerate(arms):
        style = style_for(arm, index)
        handles.append(Line2D([], [], **style, markerfacecolor="white",
                              markersize=4.3))
    handles.append(Line2D([], [], color="#222222", marker="x", linestyle="none",
                          markersize=5, label="At least one failed case"))
    reference_time = (reference["mean_success_completion_seconds"] if reference else None)
    reference_style = dict(color="#009E73", linestyle="-.", linewidth=1.1)
    if reference_time is not None:
        handles.append(Line2D([], [], **reference_style, label=(
            f"Fixed follower ({reference_time:.2f} s; "
            f"{reference['success_count']}/{reference['case_count']})")))
    figure.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, 0.905), ncol=2, frameon=False,
                  columnspacing=2.2, handlelength=2.8)

    usable_costs = [p[budget] / spec["scale"] for p in points
                    if p["complete_evaluation"] and p[budget] is not None]
    maximum_x = max(usable_costs, default=1.0)
    observed_times = [p["successful_completion_mean_seconds"] for p in points
                      if p["complete_evaluation"] and p[budget] is not None
                      and p["successful_completion_mean_seconds"] is not None]
    if reference_time is not None:
        observed_times.append(reference_time)
    maximum_time = max(observed_times, default=1.0)
    expected = len(protocol["known_cases"]) * int(protocol["evaluation_episodes_per_case"])
    ticks = np.linspace(0.0, 1.0, expected + 1) if expected <= 6 else np.linspace(0, 1, 5)
    labels = ([f"{i}/{expected}" for i in range(expected + 1)]
              if expected <= 6 else [f"{v:.0%}" for v in ticks])

    counts = {"success_points": 0, "completion_points": 0, "failure_markers": 0,
              "missing_or_partial_evaluations": 0, "complete_evaluations_missing_cost": 0}
    for col, seed in enumerate(seeds):
        upper, lower = axes[:, col]
        upper.set_title(f"Seed {seed}", fontweight="bold", pad=7)
        has_success_axis_point = has_completion = False
        seed_points = [p for p in points if p["seed"] == seed]
        for index, arm in enumerate(arms):
            line = [p for p in seed_points if p["arm"] == arm]
            style = style_for(arm, index).copy()
            style.pop("label")
            x, success, completion = [], [], []
            failed_x, failed_y = [], []
            for point in line:
                complete = point["complete_evaluation"]
                raw_cost = point[budget]
                if not complete:
                    counts["missing_or_partial_evaluations"] += 1
                elif raw_cost is None:
                    counts["complete_evaluations_missing_cost"] += 1
                visible = complete and raw_cost is not None
                x.append(raw_cost / spec["scale"] if visible else np.nan)
                success.append(point["success_fraction"] if visible else np.nan)
                time = point["successful_completion_mean_seconds"]
                completion.append(time if visible and time is not None else np.nan)
                if visible:
                    has_success_axis_point = True
                    counts["success_points"] += 1
                    if point["failures"]:
                        failed_x.append(x[-1])
                        failed_y.append(success[-1])
                        counts["failure_markers"] += 1
                    if time is not None:
                        has_completion = True
                        counts["completion_points"] += 1
            upper.plot(x, success, **style, markersize=4.4, markerfacecolor="white",
                       markeredgewidth=0.9, zorder=3)
            lower.plot(x, completion, **style, markersize=4.4, markerfacecolor="white",
                       markeredgewidth=0.9, zorder=3)
            upper.plot(failed_x, failed_y, linestyle="none", marker="x",
                       color=style["color"], markersize=7.2, markeredgewidth=1.25,
                       zorder=4)
        for row, axis in enumerate((upper, lower)):
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#D5D5D5", linewidth=0.5, alpha=0.8)
            axis.set_axisbelow(True)
            axis.set_xlim(0.0, maximum_x * 1.06)
            locator = MaxNLocator(3, min_n_ticks=2)
            axis.set_xticks([v for v in locator.tick_values(0, maximum_x * 1.06)
                             if 0 <= v <= maximum_x * 1.06])
            axis.tick_params(length=3, width=0.7)
            axis.text(-0.02, 1.07, f"({chr(97 + row * columns + col)})",
                      transform=axis.transAxes, ha="left", va="bottom",
                      fontsize=8.2, fontweight="bold")
        upper.set_ylim(-0.06, 1.06)
        upper.set_yticks(ticks, labels)
        lower.set_ylim(0.0, maximum_time * 1.13)
        if reference_time is not None:
            lower.axhline(reference_time, **reference_style, zorder=2)
        lower.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=2))
        lower.set_xlabel(spec["label"], labelpad=5)
        if not has_success_axis_point:
            note = ("NOT RUN" if all(p["evaluation_status"] == "NOT_RUN"
                                    for p in seed_points)
                    else "No complete evaluation\nwith recorded cost")
            upper.text(0.5, 0.45, note, transform=upper.transAxes,
                       ha="center", va="center", color="#666666", fontsize=8)
        if not has_completion:
            note = ("No successful\ncompletions"
                    if has_success_axis_point else "No completion-time\nestimate")
            lower.text(0.5, 0.5, note, transform=lower.transAxes,
                       ha="center", va="center", color="#666666", fontsize=7.8,
                       wrap=True)
    axes[0, 0].set_ylabel("Complete-task successes\n/ declared cases", labelpad=6)
    axes[1, 0].set_ylabel("Mean completion time (s)\nsuccessful cases only", labelpad=6)
    common_note = ("Includes one shared preparation per method, round 0, behavior, "
                   "calibration, evaluation and all stand warmups."
                   if budget == "physics" else
                   "Method elapsed intervals + one shared preparation interval; "
                   "loading/setup and recovery downtime are excluded.")
    footer = (
        f"Points: rounds 0–{protocol['rounds']}; each seed has {expected} known-case evaluations per round.\n"
        "Lines break for NOT_RUN, incomplete evaluation, missing cost or no successful time. "
        "Failures are never timed as completions.\n"
        f"{common_note}\n"
        "Successful-time means are conditional on success; read with the success panel. "
        "No seed pooling or confidence intervals."
    )
    if reference is not None:
        footer += (f"\nFollower reference: the same {reference['case_count']} fixed cases once, "
                   "no training; repeated guide, not extra seed replicates. Cost is separate.")
    figure.text(0.105, 0.15 if reference else 0.13, footer, ha="left", va="top", fontsize=6.7,
                linespacing=1.5, color="#444444")
    return figure, counts


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def plot_analysis(analysis, output, dpi, reference_path=None):
    summary, points, input_paths, raw_rows = load_analysis(analysis)
    if summary.get("comparison_valid") is False or summary.get("status") == "INVALID_COMPARISON":
        reasons = summary.get("comparison_validity", {}).get("reasons", [])
        detail = "; ".join(reasons) or "The analysis marks this comparison invalid."
        raise ValueError(
            f"INVALID_COMPARISON: no method-comparison figures will be generated. {detail} "
            "Retain the descriptive CSV files and cost ledger; use a separate analysis directory for the repaired run.")
    if summary.get("comparison_valid") is not True:
        raise ValueError(
            "Analysis lacks an explicit comparison_valid field. Re-run the current analyzer "
            "with the saved comparison-validity marker before generating method figures.")
    if not any(p["complete_evaluation"] for p in points):
        raise ValueError("No complete actual evaluation is available; no result figures generated.")
    reference = load_reference(reference_path, summary["protocol"])
    setup_style()
    output.mkdir(parents=True, exist_ok=True)
    prefix = "" if summary["status"] == "COMPLETED" else "snapshot_"
    plots, layout = {}, {}
    for budget in AXES:
        figure, counts = make_figure(summary, points, budget, reference)
        check = layout_check(figure)
        if check["text_outside_canvas"]:
            plt.close(figure)
            raise ValueError(f"{budget} figure has clipped text: {check['text_outside_canvas']}")
        name = f"{prefix}go2_task_vs_{budget}"
        figure.savefig(output / f"{name}.png", dpi=dpi)
        figure.savefig(output / f"{name}.svg")
        grayscale_figure(figure)
        figure.savefig(output / f"{name}_grayscale.png", dpi=dpi)
        plots[budget] = {"files": [f"{name}.png", f"{name}.svg",
                                   f"{name}_grayscale.png"], **counts,
                         "grayscale_preview": f"{name}_grayscale.png",
                         "grayscale_method": "Native artist recoloring; original line styles and markers preserved",
                         "size_inches": list(figure.get_size_inches()), "png_dpi": dpi}
        layout[budget] = check
        plt.close(figure)
    with (output / "plotted_checkpoints.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    manifest = {
        "schema": 1,
        "source_status": summary["status"],
        "protocol_complete": summary["protocol_complete"],
        "source_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in input_paths},
        "plot_script_sha256_lf": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "profile": data_profile(summary, points, raw_rows),
        "fixed_same_environment_reference": reference,
        "plots": plots,
        "layout_check": layout,
        "interpretation": [
            "Each column is one training seed; no seed-by-case independence or generalization claim.",
            "Success fractions require the complete declared evaluation set for that seed and round.",
            "Completion times are means over successes; changes in the successful subset can change the mean.",
            "Missing outcomes, incomplete evaluations and unknown costs are gaps, not failure or zero.",
            "Standalone-method axes allocate one full shared preparation to each method; do not sum across methods.",
            "Physics includes all recorded sampling, calibration, evaluation and stand warmups through each checkpoint.",
            "Wall uses measured/reported cumulative intervals plus measured shared preparation, not phase-time sums.",
            "For exact-resume runs, active process intervals exclude recovery downtime, which is listed separately in the analysis.",
            "Shared preparation excludes preceding model loading and subsequent initialization checkpoint saving.",
            "Per-round interval boundaries are retained in plotted_checkpoints.csv; no unrecorded running work is invented.",
            "No uncertainty intervals or significance claims are inferred from three seeds.",
        ],
        "additional_training_updates": 0,
        "additional_physics_steps": 0,
        "checkpoint_or_pickle_loading": False,
    }
    write_json(output / "plot_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="Directory produced by analyze_go2_adjoint_pilot.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Figure directory; use a distinct directory for snapshots")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--reference", type=Path,
                        help="Separate same-server/runtime fixed follower JSON; costs are not added to training axes")
    args = parser.parse_args()
    if not args.analysis.is_dir():
        parser.error("--analysis must be an existing analysis directory")
    if args.output.resolve() == args.analysis.resolve():
        parser.error("--output must differ from --analysis")
    if args.dpi < 300:
        parser.error("--dpi must be at least 300")
    result = plot_analysis(args.analysis, args.output, args.dpi, args.reference)
    print(json.dumps({"source_status": result["source_status"],
                      "plots": result["plots"], "additional_physics_steps": 0},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
