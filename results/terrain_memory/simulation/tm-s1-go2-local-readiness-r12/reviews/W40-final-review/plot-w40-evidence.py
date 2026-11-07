#!/usr/bin/env python3
"""Plot published W40 evidence. Reads records only; never advances physics.

python plot-w40-evidence.py --raw-root /path/to/w40-final-raw
Figures are companion evidence for the repository's Markdown review.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    base = here.parent.parent
    audit = json.loads((here / "raw-progress-audit.json").read_text())
    names = ["model_derivatives", "nominal_rollout", "candidate_rollouts",
             "backward_pass", "policy_update", "cost_derivatives"]
    episodes = [x for x in audit["episodes"] if x["label"] != "evaluation-E30"]
    seconds = np.array([sum(x["planner_calls"]["timers_us_sum"].get(k, 0)
                            for x in episodes) / 1e6 for k in names])
    timer_summary = {
        "population": "All 64 W40 training episodes; excludes fixed-policy evaluation",
        "denominator": "Sum of six recorded planner stage timers, not shared elapsed time",
        "recorded_subtotal_s": float(seconds.sum()),
        "stages": {k: {"seconds": float(v), "share": float(v / seconds.sum())}
                   for k, v in zip(names, seconds)},
        "interpretation": "A workload map, not a GPU speedup measurement. Nominal/candidate time-series have sequential dependencies; only independent nodes/candidates/perturbations can be batched.",
    }
    (here / "planner-stage-summary.json").write_text(
        json.dumps(timer_summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10.5, 4.6), layout="constrained")
    labels = ["Dynamics derivatives", "Nominal trajectory", "Candidate trajectories",
              "Backward pass", "Policy update", "Cost derivatives"]
    values = seconds / seconds.sum() * 100
    bars = ax.barh(labels[::-1], values[::-1], color=["#9aa8b6"] * 3 + ["#176d8f"] * 3,
                   height=.62)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values[::-1]], padding=6)
    ax.set_xlim(0, 51)
    ax.set_xlabel("Share of the six recorded planner stage timers (%)")
    ax.set_title("W40: most planner work is dynamics and forward trajectories", loc="left", pad=16)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.savefig(here / "planner-stage-shares.png", dpi=160)
    fig.savefig(here / "planner-stage-shares.svg")
    plt.close(fig)

    raw = args.raw_root / "w40-b/evaluation-E30"
    with np.load(raw / "trajectory.npz", allow_pickle=False) as f:
        t = f["time"].copy()
        q = f["qpos"].copy()
    with np.load(raw / "physical-000.npz", allow_pickle=False) as f:
        physical = f["physical"].copy()
    result_path = base / "job-20261106T103028Z-w40-b-eval-PANN_PPO_S5-s2-E30.json"
    result = json.loads(result_path.read_text())
    columns = {k: i for i, k in enumerate(result["physical_columns"])}
    requested = physical[:, [columns[f"unclipped_pd_request_{i}"] for i in range(12)]]
    limits = np.array([23.7, 23.7, 45.43] * 4)
    excess = np.maximum(np.abs(requested) - limits, 0).max(axis=1)
    held = excess.reshape(-1, 10)
    physical_t = physical[:, columns["force_evaluation_time_s"]]
    hold_t = physical_t.reshape(-1, 10)[:, 0]
    first = float(physical_t[np.flatnonzero(excess > 1e-4)[0]])
    peak_later = float(held[:, 1:].max())
    fig, axs = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True, layout="constrained")
    fig.suptitle("W40 B final-policy evaluation: forward progress and held-action saturation", x=.03, ha="left", fontsize=14)
    axs[0].plot(t, q[:, 0], color="#176d8f", lw=1.8, label="Base x")
    axs[0].axhspan(.8, 2.2, color="#b9dcb4", alpha=.4, label="Material strip x range")
    axs[0].axhline(3, color="#555", ls="--", lw=1, label="Far goal")
    axs[0].set_ylim(-.03, 3.2)
    axs[0].set_ylabel("Base x (m)")
    axs[0].legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    axs[0].text(.98, .43, f"Maximum base x = {q[:,0].max():.3f} m", transform=axs[0].transAxes, ha="right")
    axs[1].plot(t, q[:, 2], color="#176d8f", lw=1.5)
    axs[1].set_ylabel("Base z (m)")
    axs[1].set_ylim(0, max(.35, float(q[:,2].max()) + .02))
    axs[2].plot(hold_t, held[:, 1:].max(axis=1), color="#b54932", lw=1.2,
                label="Maximum of the later nine substeps")
    axs[2].plot(hold_t, held[:, 0], color="#176d8f", lw=1,
                label="First substep")
    axs[2].set_ylabel("PD request excess (Nm)")
    axs[2].set_ylim(-.5, 33)
    axs[2].set_xlabel("Simulation time (s)")
    axs[2].legend(loc="upper left", frameon=False, fontsize=9)
    axs[2].annotate(f"First excess > 1e-4 Nm\n{first:.4f} s", xy=(first, 0),
                    xytext=(7.2, 14), arrowprops={"arrowstyle": "->", "color": "#555"}, fontsize=9)
    axs[2].text(.98, .88, f"Later-substep maximum = {peak_later:.3f} Nm",
                transform=axs[2].transAxes, ha="right", fontsize=9)
    for ax in axs:
        ax.axvline(t[-1], color="#666", ls=":", lw=1)
        ax.grid(alpha=.17)
        ax.set_xlim(0, 16.3)
    axs[1].text(.98, .12, "body_fall at 15.680 s", transform=axs[1].transAxes, ha="right")
    fig.savefig(here / "w40-b-held-action.png", dpi=160)
    fig.savefig(here / "w40-b-held-action.svg")
    plt.close(fig)
    print(json.dumps({"planner_timer_episodes": len(episodes), "first_hold_excess_s": first,
                      "later_substep_excess_Nm": peak_later,
                      "note": "PD request excess is clipped by the actuator; it is not applied torque above the physical limit. Temporal association does not establish a unique failure cause."}))


if __name__ == "__main__":
    main()
