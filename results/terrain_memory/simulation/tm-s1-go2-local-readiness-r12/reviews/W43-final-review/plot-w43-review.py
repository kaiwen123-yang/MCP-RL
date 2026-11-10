#!/usr/bin/env python3
"""Render W43 final E30 figures from downloaded original files, not index aggregates."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex=True)
    colors = {"a": "#0072B2", "b": "#D55E00"}
    for stream in ("a", "b"):
        with np.load(args.raw_dir / f"{stream}-e30-trajectory.npz") as data:
            time = data["time"]
            qpos = data["qpos"]
            take = np.flatnonzero(time >= 2.002 - 1e-8)[::10]
            label = f"W43-{stream.upper()} / update55"
            axes[0, 0].plot(time[take], 1000 * qpos[take, 0], color=colors[stream], label=label)
            axes[0, 1].plot(time[take], 1000 * qpos[take, 2], color=colors[stream], label=label)
            if stream == "a":
                forces = data["actual_grf"]
                if forces.ndim == 2:
                    forces = forces.reshape(len(time), 4, 3)
                for foot, name in enumerate(("FL", "FR", "RL", "RR")):
                    axes[1, 0].plot(time[take], forces[take, foot, 2], label=name, linewidth=1.5)
        with gzip.open(args.raw_dir / f"{stream}-e30-planner-calls.jsonl.gz", "rt") as handle:
            calls = [json.loads(line) for line in handle if line.strip()]
        scored = [call for call in calls if call["origin"] >= 2.002 - 1e-8
                  and call.get("deployed_current_cost") is not None]
        axes[1, 1].plot([call["origin"] for call in scored],
                        [call["deployed_current_cost"] for call in scored],
                        color=colors[stream], label=label)
    titles = ["No forward locomotion", "Body slowly settles lower",
              "All four feet remain loaded (W43-A)", "Current native deployed rollout cost"]
    ylabels = ["Base x [mm]", "Base z [mm]", "Vertical foot force [N]", "J (new MPC problem at each time)"]
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.set_xlim(2, 30)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].axhline(0, color="#777777", linestyle=":", linewidth=1)
    axes[1, 0].set_ylim(0, 50)
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].legend(frameon=False, fontsize=9, ncol=4, loc="lower left")
    axes[1, 0].set_xlabel("Simulation time [s]")
    axes[1, 1].set_xlabel("Simulation time [s]")
    fig.suptitle("W43 final fixed-policy E30: solver activity without task progress", fontsize=15, weight="bold")
    fig.text(0.07, 0.017, "Source: both original E30 trajectories and planner-call logs. "
             "Local accepted decreases do not imply J decreases across different MPC problems.", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"cpu-final-motion.{suffix}", dpi=180)
    svg_path = args.output_dir / "cpu-final-motion.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
