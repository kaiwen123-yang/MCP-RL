"""Plot existing W51 trajectory records; no simulation or controller execution.

Usage: python make_command_support_figure.py --originals /path/to/extracted
The extracted layout follows W51 paired-D8 delivery/sources.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", type=Path, required=True)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    jobs = {
        "D0": "job-20261115T112019Z-w51-d0-eval-PANN_PPO_S5-s1-D8-attempt01",
        "D1": "job-20261115T113036Z-w51-d1-command-repair-eval-PANN_PPO_S5-s1-D8-attempt01",
    }
    # Runtime trajectory/native candidate order; W51 IK uses FL, RL, FR, RR.
    foot_names = ["FL", "FR", "RL", "RR"]
    colors = ["#006d77", "#e29578", "#586ba4", "#a44a3f"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1, .8, 1]})
    evidence = {
        "schema": "w51_command_support_followup_v1",
        "source_commit": "a621d7d35e6016634b388a3aef0d39d8d008aab0",
        "review_base_commit": "28b865477d35a1ef04dc04e0da3521c6c9dc4c2b",
        "new_physics": 0,
        "foot_order": foot_names,
        "ik_to_trajectory_permutation": [0, 2, 1, 3],
        "scope": "Existing 500 Hz trajectory rows, including each final row. Fz > 1 N is descriptive only. Flat pre-material foot radius is 0.022 m.",
        "arms": {},
    }
    for col, (arm, job) in enumerate(jobs.items()):
        path = args.originals / job / "trajectory.npz"
        with np.load(path, allow_pickle=False) as record:
            t = record["time"]
            vx = record["qvel"][:, 0]
            command = record["command_velocity"][:, 0]
            position = record["qpos"][:, :3]
            fz = record["actual_grf"][:, :, 2]
            clearance_mm = (record["feet"][:, :, 2] - .022) * 1000
            loaded = fz > 1.
            selected = []
            for target in (2.002, 3., 4.3, 6., 8.):
                if target > float(t[-1]) + 1e-9:
                    continue
                k = int(np.argmin(np.abs(t-target)))
                selected.append({"time_s": float(t[k]), "command_vx_m_s": float(command[k]),
                                 "actual_vx_m_s": float(vx[k]), "base_x_m": float(position[k, 0]),
                                 "base_z_m": float(position[k, 2]), "feet_Fz_N": fz[k].tolist()})
            windows = []
            for lo, hi in ((0., 2.), (2.002, 4.3), (4.3, 8.)):
                stop = min(hi, float(t[-1]))
                take = (t >= lo-1e-9) & (t <= stop+1e-9)
                idx = np.flatnonzero(take)
                if not len(idx):
                    continue
                loads = loaded[take]
                windows.append({"from_s": lo, "to_s": stop, "rows_including_boundaries": len(idx),
                    "delta_x_mm": float((position[idx[-1], 0]-position[idx[0], 0])*1000),
                    "four_feet_loaded_fraction": float(np.mean(np.all(loads, axis=1))),
                    "per_foot_load_transitions": np.sum(loads[1:] != loads[:-1], axis=0).tolist(),
                    "max_foot_clearance_mm": float(np.max(clearance_mm[take])),
                    "min_base_z_m": float(np.min(position[take, 2]))})
            evidence["arms"][arm] = {"job_id": job, "record": f"{job}/trajectory.npz",
                "end_s": float(t[-1]), "rows": len(t), "samples": selected, "windows": windows}
            axes[0, col].plot(t, command, color="#d76b2f", lw=1.8, label="Mission command vx")
            axes[0, col].plot(t, vx, color="#176b87", lw=1.1, label="Actual base vx")
            axes[0, col].axhline(0, color="#b8b8b8", lw=.6)
            axes[0, col].set_ylabel("Forward velocity (m/s)")
            axes[0, col].legend(loc="upper left", fontsize=8)
            axes[0, col].set_title("D0 / r38: body fall at 5.814 s" if arm == "D0"
                                   else "D1 / r39 corrected: D8 ends without walking", weight="bold", pad=12)
            axes[1, col].plot(t, position[:, 2], color="#176b87", lw=1.4)
            axes[1, col].axhline(.12, color="#ad3d3d", ls="--", lw=.8, label="Original z fall boundary")
            axes[1, col].set_ylim(.10, .32)
            axes[1, col].set_ylabel("Base height (m)")
            axes[1, col].legend(loc="lower left", fontsize=8)
            axes[2, col].imshow(loaded.T, origin="upper", aspect="auto", interpolation="nearest",
                                extent=[float(t[0]), float(t[-1]), 3.5, -.5],
                                cmap=matplotlib.colors.ListedColormap(["#f6f6f6", "#4c7a8b"]), vmin=0, vmax=1)
            axes[2, col].set_yticks(range(4), foot_names)
            axes[2, col].set_ylabel("Loaded foot\nFz > 1 N")
            for j in range(4):
                axes[3, col].plot(t, clearance_mm[:, j], color=colors[j], lw=.8, label=foot_names[j])
            axes[3, col].axhline(0, color="#777777", lw=.6)
            axes[3, col].axhline(5, color="#777777", ls=":", lw=.9,
                                 label="Nominal peak (not instantaneous reference)")
            if arm == "D1":
                axes[3, col].set_ylim(-.3, 5.8)
            axes[3, col].set_ylabel("Actual foot-bottom\nclearance (mm)")
            axes[3, col].set_xlabel("Simulation time (s)")
            axes[3, col].legend(loc="upper left", fontsize=7, ncol=2)
            for ax in axes[:, col]:
                ax.axvline(2., color="#666666", ls="--", lw=.7)
                ax.axvline(4.3, color="#999999", ls=":", lw=.7)
                ax.set_xlim(0, 8)
                if arm == "D0":
                    ax.axvspan(float(t[-1]), 8, color="#e2e2e2", alpha=.8)
            axes[1, 0].text(6.9, .24, "No data\nafter termination", ha="center", color="#5c5c5c", fontsize=9)
    fig.suptitle("W51: nonzero movement commands did not become sustained locomotion", fontsize=16, weight="bold", y=.99)
    fig.text(.5, .013, "Existing original records only. Load threshold describes contact; it does not change task qualification.\n"
             "0–2 s: Stand. Vertical dotted line: start of the late observation window. Axes preserve each arm's actual duration.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, .05, 1, .965])
    fig.savefig(output / "command-support.png", dpi=160)
    (output / "command-support.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
