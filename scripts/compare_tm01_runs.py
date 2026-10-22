#!/usr/bin/env python3
"""Matched-seed R1 contrasts using full-horizon outcomes and common prefixes."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"results/terrain_memory/reproduction/pympc"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", default="r1-foothold-restored")
a = p.parse_args()
dest = BASE/a.run
contrasts = [
    ("acados_footholds", "r1-v7-paper-model", "acados", a.run, "acados"),
    ("naive_footholds", "r1-v7-paper-model", "naive", a.run, "naive"),
    ("adaptive_footholds", "r1-v7-paper-model", "adaptive", a.run, "adaptive"),
    ("frequency_only", a.run, "adaptive_fixed", a.run, "adaptive"),
]


def rows(run, method):
    with (BASE/run/"table1"/method/"episodes.csv").open() as stream:
        return {int(row["seed"]): row for row in csv.DictReader(stream)}


def dump_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


summary, pairs = [], []
rng = np.random.default_rng(20261022)
plt.rcParams.update({"font.size": 9, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.3), layout="constrained")
for axis, (name, run0, method0, run1, method1) in zip(axes.flat, contrasts):
    left, right = rows(run0, method0), rows(run1, method1)
    group = []
    for seed in sorted(left):
        x, y = left[seed], right[seed]
        with np.load(ROOT/x["trajectory"]) as dx, np.load(ROOT/y["trajectory"]) as dy:
            # All runs use 500 Hz with the same zero-time start. A pair's common
            # prefix ends when the first member falls; it is NOT a 60 s score.
            n = min(len(dx["time"]), len(dy["time"]))
            ex = np.linalg.norm(dx["velocity"][:n]-dx["command"][:n], axis=1)
            ey = np.linalg.norm(dy["velocity"][:n]-dy["command"][:n], axis=1)
            record = dict(contrast=name, seed=seed, run_a=run0, method_a=method0,
                run_b=run1, method_b=method1, completed_a=int(x["success"]), completed_b=int(y["success"]),
                duration_a_s=float(x["duration_s"]), duration_b_s=float(y["duration_s"]),
                duration_difference_s=float(y["duration_s"])-float(x["duration_s"]),
                common_prefix_s=n*.002, common_prefix_error_a_m_s=float(ex.mean()),
                common_prefix_error_b_m_s=float(ey.mean()),
                trajectory_a=x["trajectory"], trajectory_b=y["trajectory"])
            group.append(record)
    pairs.extend(group)
    delta = np.array([row["duration_difference_s"] for row in group])
    boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
    ci = np.quantile(boot, [.025, .975])
    s = dict(contrast=name, episodes=len(group),
        completed_a=sum(row["completed_a"] for row in group),
        completed_b=sum(row["completed_b"] for row in group),
        mean_time_without_fall_a_s=float(np.mean([row["duration_a_s"] for row in group])),
        mean_time_without_fall_b_s=float(np.mean([row["duration_b_s"] for row in group])),
        paired_mean_difference_s=float(delta.mean()),
        paired_mean_difference_ci95_low_s=float(ci[0]), paired_mean_difference_ci95_high_s=float(ci[1]),
        longer_b=int((delta > 0).sum()), shorter_b=int((delta < 0).sum()), tied=int((delta == 0).sum()),
        gained_completion=sum(not row["completed_a"] and row["completed_b"] for row in group),
        lost_completion=sum(row["completed_a"] and not row["completed_b"] for row in group))
    summary.append(s)
    axis.scatter([row["duration_a_s"] for row in group], [row["duration_b_s"] for row in group],
        s=23, color="#0072B2", alpha=.7, edgecolor="white", linewidth=.3)
    axis.plot([0, 60], [0, 60], "k--", lw=.8)
    axis.set(xlim=(0, 62), ylim=(0, 62),
        xlabel=("Adaptive implementation at 1.3 Hz" if name == "frequency_only" else "v7 "+method0)+" (s)",
        ylabel=("Adaptive implementation, variable Hz" if name == "frequency_only" else "Restored "+method1)+" (s)")
    axis.set_title(f"{s['completed_a']}/50 to {s['completed_b']}/50 completed\n"
        f"Paired mean time change {delta.mean():+.2f} s [{ci[0]:+.2f}, {ci[1]:+.2f}]", fontsize=9)
    axis.grid(alpha=.2)
fig.suptitle("R1 matched seeds | time without a fall within the fixed 60 s horizon")
fig.supxlabel("Each point: one matched disturbance seed. Dashed line: equal duration.\n"
    "Intervals: paired bootstrap, 5,000 resamples (analysis seed 20261022); all 50 outcomes retained.", fontsize=8)
for extension in ("png", "svg"):
    fig.savefig(dest/("matched_durations."+extension), dpi=220, bbox_inches="tight")
plt.close(fig)
dump_csv(dest/"paired_durations.csv", pairs)
dump_csv(dest/"paired_comparisons.csv", summary)
(dest/"paired_comparisons.json").write_text(json.dumps(dict(
    command=f"python scripts/compare_tm01_runs.py --run {a.run}", comparisons=summary,
    interpretation="Completion and mean time without a fall use the same complete 60 s task. Common-prefix error in paired_durations.csv is diagnostic only; it does not assign an error after a fall or rank unequal surviving intervals as full-task performance.",
    intervention="The foothold contrasts change two declared behaviors together: 0.083 to 0.1 m lateral offset and body-following to world-fixed lift-off. They do not separate the two contributions. The frequency-only contrast uses the same adaptive controller with [1.3] versus [1.3,2.0,2.4] Hz."
    ), indent=2)+"\n")
print(json.dumps(summary, indent=2))
