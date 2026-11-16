"""Plot saved W52 call128 geometry and native contact forces; no physics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--originals", required=True, type=Path)
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
job = args.originals / "job-20261116T110800Z-w52-d2-eval-PANN_PPO_S5-s1-D8-attempt01"
seed = next(json.loads(line) for line in (job / "w51-seeds.jsonl").read_text().splitlines()
            if json.loads(line)["call_id"] == 128)
with np.load(job / seed["objects"]["IK"]["file"]) as z:
    raw = z["kinematics_payload"].copy()
n = int(raw[3])
t = raw[5:5+n]
targets = raw[5+n:5+13*n].reshape(n, 4, 3)[:, [0, 2, 1, 3]]
with np.load(job / "r33-candidates-call000128.npz") as z:
    native = z["candidate010_foot_rows"].reshape(-1, 29).copy()
feet = native[:, 1:].reshape(-1, 4, 7)
# This saved call is entirely on the original flat floor; foot radius is 0.022 m.
clearance = (targets[:, :, 2] - 0.022) * 1000
force = feet[:, :, 4] + feet[:, :, 5]
apex_i = int(np.argmin(abs(t - 2.1885)))
colors = ["#64748b", "#007c91", "#d2691e", "#a78bba"]
names = ["FL", "FR", "RL", "RR"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.2), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.05]})
fig.suptitle("W52: a valid 5 mm swing target did not create unloading",
             x=.09, ha="left", fontsize=15, fontweight="bold")
fig.text(.09, .917, "First valid seed S | call 128 at 2.140 s | D2 and D3 share this exact native seed",
         color="#475569", fontsize=10)
for i in (1, 2):
    axes[0].plot(t, clearance[:, i], color=colors[i], linewidth=2,
                 linestyle="-" if i == 1 else "--", label=names[i]+" target")
    axes[0].plot(native[:, 0], feet[:, i, 3]*1000, color=colors[i], linewidth=1.7,
                 linestyle=":" if i == 1 else "-.", label=names[i]+" native")
axes[0].axhline(0, color="#94a3b8", linewidth=.7)
axes[0].set_ylabel("Foot-bottom clearance (mm)")
axes[0].set_ylim(-.5, 6.1)
axes[0].legend(loc="upper right", ncol=2, frameon=False)
axes[0].annotate("Target: ~5.000 mm\nNative: -0.00276 / -0.00335 mm",
                 xy=(t[apex_i], 4.99995), xytext=(2.245, 3.0),
                 arrowprops={"arrowstyle":"-", "color":"#475569"}, fontsize=10)
for i in range(4):
    axes[1].plot(native[:, 0], force[:, i], color=colors[i],
                 linewidth=2 if i in (1,2) else 1.2, label=names[i])
axes[1].axhline(1, color="#94a3b8", linestyle=":", linewidth=1)
axes[1].set_ylabel("Native vertical foot load (N)")
axes[1].set_xlabel("Simulation time (s)")
axes[1].set_ylim(0, 60)
axes[1].legend(loc="upper right", ncol=4, frameon=False)
axes[1].annotate("Planned swing feet at first apex:\nFR 28.20 N   |   RL 33.75 N",
                 xy=(t[apex_i], force[apex_i, 1]), xytext=(2.245, 12),
                 arrowprops={"arrowstyle":"-", "color":"#475569"}, fontsize=10)
for ax in axes:
    ax.axvline(t[apex_i], color="#64748b", linestyle="--", linewidth=.8)
    ax.grid(axis="y", alpha=.2)
    ax.set_xlim(2.14, 2.49)
fig.text(.09, .037,
         "All 176 native nodes retain four loaded feet. The 1 N line is a diagnostic reference, not a success rule.\n"
         "No clipping or saturation explains this seed: saved raw/action reconstruction error < 1.2e-16 rad.",
         color="#475569", fontsize=9)
fig.subplots_adjust(left=.09, right=.97, bottom=.14, top=.87, hspace=.18)
args.output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output_dir / "first-seed-mismatch.png", dpi=160, facecolor="white")
plt.close(fig)
print(json.dumps({"figure":"first-seed-mismatch.png", "call":128,
                  "apex_sample_s":float(t[apex_i]), "new_physics":0}))
