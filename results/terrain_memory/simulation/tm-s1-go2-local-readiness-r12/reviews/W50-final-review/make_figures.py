"""Read committed W50 evidence and draw review figures; no simulation or solver."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    base = args.repo / "results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12"
    out = base / "reviews/W50-final-review"
    a = json.loads((base / "w50/task-a.json").read_text())
    b = json.loads((base / "w50/task-b.json").read_text())
    data = {"scope": "W49 plant evidence reanalyzed in W50; W50 native finite predictions are not new plant videos", "early_support": [], "native_comparisons": []}
    for stream in a["streams"]:
        w = stream["support_windows"][0]
        four = next(x["fraction"] for x in w["load_patterns"] if x["loaded_FL_FR_RL_RR"] == [1, 1, 1, 1])
        data["early_support"].append(dict(stream=stream["stream"], start_s=w["start_s"], end_exclusive_s=w["end_exclusive_s"],
            loaded_during_target_swing_fraction=w["actual_loaded_during_target_swing_fraction"], four_feet_loaded_fraction=four,
            pd_saturation_any_fraction=w["actual_held_pd_saturation_any_fraction"], dx_m=w["displacement_x_m"],
            foot_order=["FL", "FR", "RL", "RR"], diagnostic_load_threshold_N=1))
    for event in b["events"]:
        selected = event["selected"]
        row = dict(label=event["label"], origin_s=event["origin_s"], baseline=event["baseline"], selected=selected,
                   qualification="native finite prediction only")
        data["native_comparisons"].append(row)
    (out / "video-analysis.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7), layout="constrained")
    colors = ["#2864b4", "#d76b2d"]
    x = np.arange(4)
    for i, row in enumerate(data["early_support"]):
        values = 100 * np.array(row["loaded_during_target_swing_fraction"])
        bars = axes[0].bar(x + (i-.5)*.36, values, width=.34, color=colors[i], label=f"W49 {row['stream'].upper()}")
        axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axes[0].set(xticks=x, xticklabels=["FL", "FR", "RL", "RR"], ylim=(0, 105), ylabel="Loaded during target swing (%)", title="Before collapse: 2.002–4.300 s")
    axes[0].legend(loc="lower right")
    axes[0].text(.03, .12, "All four feet loaded: A 85.72%, B 86.74%\nPD saturation: 0% in both streams\nLoad >1 N is a diagnostic, not a new stop rule", transform=axes[0].transAxes, fontsize=9)
    axes[0].grid(axis="y", alpha=.2)
    palette = {"A237":"#2864b4", "B236":"#d76b2d", "A328":"#43835a", "B625":"#797979"}
    for row in data["native_comparisons"]:
        label = row["label"]; base_row = row["baseline"]
        bx = 1000 * base_row["forward_displacement_m"]; bz = 1000 * base_row["minimum_z_m"]
        axes[1].scatter([bx], [bz], marker="o", s=65, facecolors="white", edgecolors=palette[label], linewidths=1.8)
        if row["selected"]:
            m = row["selected"]["motion"]
            sx = 1000 * m["forward_displacement_m"]; sz = 1000 * m["minimum_z_m"]
            axes[1].annotate("", xy=(sx,sz), xytext=(bx,bz), arrowprops=dict(arrowstyle="->", color=palette[label], lw=1.6))
            axes[1].scatter([sx], [sz], s=40, color=palette[label])
            axes[1].annotate(label, (sx,sz), xytext=(-2,8), textcoords="offset points", color=palette[label], fontsize=10)
        else:
            axes[1].annotate(label+" retained", (bx,bz), xytext=(6,0), textcoords="offset points", color=palette[label], fontsize=9)
    axes[1].axvline(0, color="#999999", lw=.8, ls="--")
    axes[1].set(xlim=(-29,14), ylim=(126,159), xlabel="Net forward displacement over 0.35 s (mm)", ylabel="Minimum body height (mm)", title="W50 native prediction: baseline → selected")
    axes[1].text(.04,.04,"Open circle: original baseline\nFilled circle: strict selected candidate\nThese are predictions, not new plant runs",transform=axes[1].transAxes,fontsize=9)
    axes[1].grid(alpha=.2)
    fig.suptitle("W50 evidence: the gait was already ineffective before the visible collapse", fontsize=14)
    fig.savefig(out / "support-and-candidate-evidence.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"figure":"support-and-candidate-evidence.png", "data":"video-analysis.json", "new_physics":0}))


if __name__ == "__main__":
    main()
