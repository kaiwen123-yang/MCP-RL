"""Offline figure of all nine complete-task records, no simulation or inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="work/go2-frozen-20261019/downloaded", type=Path)
    parser.add_argument("--output", default="notes/go2-frozen-2026-10-19", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {}
    # Repaired-duration records supersede only their three affected identities;
    # all originals remain in the input archive and offline audit output.
    parents = ["tasks-v1", "tasks-v1-policy-rollout"] + sorted(p.name for p in args.input.glob("holdfix-*") if p.is_dir())
    for parent in parents:
        for summary in (args.input/parent).glob("*/summary.json"):
            data = json.loads(summary.read_text(encoding="utf-8"))
            trace = json.loads(summary.with_name("trace.json").read_text(encoding="utf-8"))
            cases[(data["method"], data["condition"])] = (data, trace)
    methods = ["waypoint", "kinematic_mpc", "policy_rollout_mpc"]
    names = ["Waypoint", "Kinematic MPC", "Policy-rollout MPC"]
    conditions = ["nominal", "push_positive_y", "push_negative_y"]
    titles = ["No disturbance", "Positive-y push", "Negative-y push"]
    colors, styles, markers = ["#0072B2", "#D55E00", "#009E73"], ["-", "--", "-."], ["o", "^", "s"]
    if len(cases) != 9:
        raise ValueError("Figure requires all nine retained paired task records")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.2))
    fig.subplots_adjust(top=.86, bottom=.12, left=.07, right=.98, wspace=.30, hspace=.42)
    for column, condition in enumerate(conditions):
        ax = axes[0, column]
        for x in [-.1, 2.6]:
            ax.add_patch(Rectangle((x, 1.42), 1.5, .16, color="#888888", alpha=.6))
        ax.add_patch(Circle((2, 2.4), .25, fill=False, color="#555555", linestyle=":"))
        ax.plot([0, 1.5, 2, 2], [0, 0, 1, 2.4], ":", color="#aaaaaa", linewidth=1)
        for i, method in enumerate(methods):
            summary, trace = cases[(method, condition)]
            positions = np.array([r["position"][:2] for r in trace])
            ax.plot(positions[:, 0], positions[:, 1], color=colors[i], linestyle=styles[i], label=names[i], linewidth=1.6)
            ax.scatter(*positions[-1], marker=markers[i], color=colors[i], s=24)
            event = summary["push_event"]
            if event is not None:
                ax.scatter(*event["position"][:2], marker="x", color=colors[i], s=22)
        ax.set(xlabel="World x (m)", ylabel="World y (m)", title=f"{chr(97+column)}  {titles[column]}",
               xlim=(-.2, 3.2), ylim=(-.35, 2.85), xticks=[0, 1, 2, 3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.15)
    metrics = [("task_seconds", "Completion time (s)", "d  All tasks completed"),
               ("absolute_mechanical_work", "Absolute joint work (J)", "e  Mechanical work, not battery energy"),
               ("end_to_end_p95_seconds", "p95 macro-step latency (s)", "f  CPU cost against 0.2 s period")]
    for column, (field, label, title) in enumerate(metrics):
        ax = axes[1, column]
        for i, method in enumerate(methods):
            for j, condition in enumerate(conditions):
                ax.scatter(i+(j-1)*.12, cases[(method, condition)][0][field],
                           color=colors[i], marker=markers[j], s=43, edgecolors="white", linewidths=.4)
        ax.set(xticks=range(3), xticklabels=["Waypoint", "Kinematic\nMPC", "Policy-rollout\nMPC"], ylabel=label, title=title, xlim=(-.4, 2.4))
        ax.grid(axis="y", alpha=.2)
        if column == 2:
            ax.set_yscale("log")
            ax.set_ylim(.005, 1.0)
            ax.set_yticks([.01, .1, 1.0])
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.axhline(.2, color="#444444", linestyle=":", linewidth=1)
            ax.text(.0, .23, "5 Hz period", fontsize=8)
        else:
            ax.set_ylim(0, max(v[0][field] for v in cases.values())*1.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .965), ncol=3, frameon=False)
    fig.suptitle("Frozen Go2: task completion is established; exact-policy prediction is computationally expensive", fontsize=12, y=.995)
    fig.text(.07, .025, "Three deterministic conditions: circle = nominal, triangle = positive-y push, square = negative-y push.\n"
             "No training. Success requires full-body clearance and qualifying 50 Hz samples spanning 1 s. Synchronous simulation.", fontsize=9)
    with warnings.catch_warnings(record=True) as captured:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        outside = []
        for obj in fig.findobj(matplotlib.text.Text):
            if obj.get_visible() and obj.get_text():
                bounds = obj.get_window_extent(renderer)
                if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.width+1 or bounds.y1 > fig.bbox.height+1:
                    outside.append(obj.get_text())
        fig.savefig(args.output/"complete-task-comparison.png", dpi=180)
        fig.savefig(args.output/"complete-task-comparison.svg")
    checks = {"case_count": len(cases), "success_count": sum(x[0]["success"] for x in cases.values()),
              "missing_glyph_warnings": [str(w.message) for w in captured if "Glyph" in str(w.message)],
              "text_outside_canvas": outside, "error_bars": False,
              "interpretation": "All paired cases shown; no statistical confidence or learned-feedback gain claimed"}
    (args.output/"figure-check.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
