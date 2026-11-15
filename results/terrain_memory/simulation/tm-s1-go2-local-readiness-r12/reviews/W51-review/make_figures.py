#!/usr/bin/env python3
"""Plot already-recorded W51 data and extract original video frames; no replay.

Usage: python make_figures.py --originals EXTRACTED --videos DOWNLOADS --out OUT
Requires NumPy, matplotlib, Pillow and ffmpeg. All clocks remain original.
"""
import argparse
import io
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ATTEMPTS = {
    "D0": "job-20261115T112019Z-w51-d0-eval-PANN_PPO_S5-s1-D8-attempt01",
    "D1": "job-20261115T113036Z-w51-d1-command-repair-eval-PANN_PPO_S5-s1-D8-attempt01",
}
COLORS = {"D0": "#bb462e", "D1": "#176a99"}


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, layout="constrained")
    for arm, attempt in ATTEMPTS.items():
        root = args.originals / attempt
        with np.load(root / "trajectory.npz", allow_pickle=False) as archive:
            t, q = archive["time"], archive["qpos"]
            loads = archive["actual_grf"][:, :, 2]
        with np.load(root / "physical-000.npz", allow_pickle=False) as archive:
            physical = archive["physical"]
        repo = Path(__file__).resolve().parents[6]
        summary = json.loads((repo / "results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12" /
                              (attempt + ".json")).read_text())
        columns = {name: i for i, name in enumerate(summary["physical_columns"])}
        pd = physical[:, [columns[f"unclipped_pd_request_{i}"] for i in range(12)]]
        axes[0, 0].plot(t, q[:, 0], color=COLORS[arm], label=arm)
        axes[0, 1].plot(t, q[:, 2], color=COLORS[arm], label=arm)
        ax = axes[1, 0 if arm == "D0" else 1]
        for leg, label in enumerate(("FL", "FR", "RL", "RR")):
            ax.step(t, (loads[:, leg] > 1).astype(float) * .65 + 3 - leg,
                    where="post", linewidth=.8, label=label)
        ax.set_yticks([.325, 1.325, 2.325, 3.325], ["RR", "RL", "FR", "FL"])
        ax.set_ylim(-.15, 3.85)
        ax.set_title(f"{arm}: actual foot load (upper level = force > 1 N)")
        actions = {r["query_id"]: r for r in read_rows(root / "action-execution.jsonl")
                   if r["query_kind"] == "applied_hold"}
        feedback = [r for r in read_rows(root / "feedback-actions.jsonl") if r["query_id"] in actions]
        ft = np.array([actions[r["query_id"]]["time_s"] for r in feedback])
        gain = np.array([max(abs(v) for v in r["raw_K_delta_x"]) for r in feedback])
        axes[2, 0].plot(ft, gain, color=COLORS[arm], linewidth=.8, label=arm)
        limits = np.array([23.7, 23.7, 45.43] * 4)
        axes[2, 1].plot(physical[:, 0], np.max(np.abs(pd) / limits, axis=1), color=COLORS[arm],
                        linewidth=.8, label=arm)
    axes[0, 0].set(title="Body forward position", ylabel="x (m)")
    axes[0, 1].set(title="Body height", ylabel="z (m)")
    axes[0, 1].axhline(.12, linestyle=":", color="#666666", label="original fall height")
    axes[2, 0].set(title="Applied policy: largest raw feedback correction", ylabel="|K delta x| (rad)")
    axes[2, 0].set_yscale("symlog", linthresh=.01)
    axes[2, 0].set_ylim(bottom=0)
    axes[2, 1].set(title="5000 Hz PD request / original actuator torque limit", ylabel="maximum ratio")
    axes[2, 1].axhline(1, linestyle=":", color="#666666")
    for ax in axes.flat:
        ax.set_xlim(0, 8.05)
        ax.axvline(2.002, linestyle="--", color="#888888", linewidth=.7)
        ax.grid(alpha=.18)
    for ax in axes[0]:
        ax.legend(loc="best", fontsize=9)
    for ax in axes[2]:
        ax.set_xlabel("Original simulation time (s)")
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("W51: D0 falls after feedback escalation; D1 remains loaded and stationary", fontsize=13)
    fig.savefig(args.out / "w51-physical-feedback.png", dpi=150)
    plt.close(fig)
    for arm, times in {"D0": [0, 2, 4, 4.6, 5.4, 5.8], "D1": [0, 2, 3, 4.2, 6, 8]}.items():
        source = args.videos / ("w51-" + ATTEMPTS[arm] + "-full-mission-material-overlay.mp4")
        sheet = Image.new("RGB", (960, 580), "#fafafa")
        draw = ImageDraw.Draw(sheet)
        for j, t in enumerate(times):
            # Frame number at the recorded 30 fps, rather than interpolated images.
            n = round(30 * t)
            result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-vf",
                                     f"select=eq(n\\,{n})", "-frames:v", "1", "-f", "image2pipe",
                                     "-vcodec", "png", "-"], capture_output=True, check=True)
            frame = Image.open(io.BytesIO(result.stdout))
            crop = frame.crop((490, 300, 730, 490)).resize((320, 254))
            x, y = j % 3 * 320, j // 3 * 290
            sheet.paste(crop, (x, y + 30))
            draw.text((x + 8, y + 8), f"{arm} original video {t:.1f}s | frame {n}", fill="black")
        sheet.save(args.out / f"{arm}-video-closeups.png")


if __name__ == "__main__":
    main()
