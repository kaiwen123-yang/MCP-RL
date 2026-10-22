#!/usr/bin/env python3
"""Analyze recorded R1 episodes, without running or relabeling a simulation.

Event thresholds below describe the time series; they never change success or
termination. Old trajectories lack planned_contact, so reconstruct it from the
recorded phase using the executed PeriodicGaitGenerator rule phase < 0.65.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/terrain_memory/reproduction/pympc"
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
LEGS = ["FL", "FR", "RL", "RR"]
plt.rcParams.update({"font.size": 9, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False})


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_sustained(time, mask, seconds):
    count = round(seconds / np.median(np.diff(time)))
    indices = np.flatnonzero(np.convolve((mask & (time >= 2)).astype(int),
        np.ones(count, dtype=int), mode="valid") == count)
    return float(time[indices[0]]) if len(indices) else None


def signals(d):
    time = d["time"]
    planned = d["planned_contact"].astype(bool) if "planned_contact" in d else d["phase"] < .65
    actual = d["contact"].astype(bool)
    force = d["actual_grf"]
    predicted = d["planned_grf"]
    arms = d["feet"] - d["com"][:, None, :]
    foot_velocity = (d["feet_velocity"] if "feet_velocity" in d
        else np.gradient(d["feet"], time, axis=0))
    return dict(time=time, planned=planned, actual=actual,
        missing=(planned & ~actual).sum(axis=1),
        saturation=(np.abs(d["torque"]) >= 39.959).any(axis=1),
        tilt=np.rad2deg(np.abs(d["orientation"][:, :2])).max(axis=1),
        support=force[:, :, 2].sum(axis=1),
        planned_moment=np.cross(arms, predicted).sum(axis=1),
        actual_moment=np.cross(arms, force).sum(axis=1),
        loaded_slip=((force[:, :, 2] > 20) &
            (np.linalg.norm(foot_velocity[:, :, :2], axis=2) > .2)).any(axis=1),
        error=np.linalg.norm(d["velocity"] - d["command"], axis=1))


def plot_episode(d, row, dest):
    s = signals(d)
    time = s["time"]
    fig, axes = plt.subplots(8, 2, figsize=(11.5, 13.0), layout="constrained")
    # Whole episode plus the onset of substantial tilt. The latter is a
    # deterministic selection, not the most dramatic-looking interval.
    onset = first_sustained(time, s["tilt"] > 20, .1)
    centre = onset if onset is not None else time[-1] - 1
    windows = [(0, time[-1]), (max(0, centre - 1), min(time[-1], centre + 2))]
    for column, (left, right) in enumerate(windows):
        ax = axes[:, column]
        for component, color in enumerate(COLORS[:2]):
            ax[0].plot(time, np.rad2deg(d["orientation"][:, component]),
                color=color, lw=.8, label=["roll", "pitch"][component])
            ax[0].plot(time, np.rad2deg(d["reference_state"][:, 6+component]),
                color=color, ls="--", lw=.8, label=["roll ref", "pitch ref"][component])
        for i in range(3):
            ax[1].plot(time, d["velocity"][:, i], color=COLORS[i], lw=.8, label=["vx", "vy", "vz"][i])
        ax[1].axhline(.5, color="black", ls="--", lw=.7)
        ax[2].plot(time, d["planned_grf"][:, :, 2].sum(axis=1), color=COLORS[0], lw=.8, label="planned")
        ax[2].plot(time, s["support"], color=COLORS[1], lw=.7, label="measured")
        ax[2].axhline(24.638*9.81, color="black", ls=":", lw=.7)
        ax[3].step(time, s["planned"].sum(axis=1), color=COLORS[0], lw=.7, label="planned contacts")
        ax[3].step(time, s["actual"].sum(axis=1), color=COLORS[1], lw=.7, label="actual contacts")
        ax[3].step(time, s["missing"], color=COLORS[3], ls=":", lw=.8, label="missing stance feet")
        ax[4].plot(time, np.abs(d["torque"]).max(axis=1), color=COLORS[0], lw=.8, label="applied max")
        if "requested_torque" in d:
            ax[4].plot(time, np.abs(d["requested_torque"]).max(axis=1), color=COLORS[1], lw=.6, label="requested max")
        ax[4].axhline(39.96, color="black", ls="--", lw=.7, label="execution limit")
        for i, leg in enumerate(LEGS):
            ax[5].plot(time, d["feet"][:, i, 2], color=COLORS[i], lw=.7, label=leg)
        ax[6].plot(time, s["planned_moment"][:, 0], color=COLORS[0], lw=.8, label="planned contact")
        ax[6].plot(time, s["actual_moment"][:, 0], color=COLORS[1], lw=.7, label="measured contact")
        ax[6].plot(time, d["wrench"][:, 3], color="black", ls="--", lw=.8, label="external")
        ax[7].step(time, d["solver_status"], color=COLORS[0], lw=.8, label="acados status" if row["method"] == "acados" else "sampling placeholder")
        for i, axis in enumerate(ax):
            axis.set_xlim(left, right)
            axis.grid(alpha=.2)
            axis.legend(loc="upper left", fontsize=6.6, ncol=4 if i in (0, 5) else 3,
                bbox_to_anchor=(0, 1.26), frameon=False)
        ax[-1].set_xlabel("Simulation time (s)")
    labels = ["Attitude (deg)", "Velocity (m/s)", "Vertical force (N)", "Contact count",
        "Joint torque (N m)", "Foot height (m)", "World roll moment (N m)", "Solver status"]
    for axis, label in zip(axes[:, 0], labels):
        axis.set_ylabel(label)
    fig.suptitle(f"{row['run']} | {row['method']} | seed {row['seed']} | "
        f"{'completed' if int(row['success']) else 'fell'} at {float(row['duration_s']):.3f} s\n"
        "Left: full trajectory. Right: first sustained 20-degree tilt (or final interval).", fontsize=11)
    fig.supxlabel("Measured contacts precede the current torque action by one 2 ms step. "
        "Moment uses measured foot positions.\nDashed attitude lines are executed references; sampling has no native convergence status.", fontsize=8)
    for extension in ("png", "svg"):
        fig.savefig(dest.with_suffix("."+extension), dpi=190, bbox_inches="tight")
    plt.close(fig)


def analyze(run):
    folder = BASE / run
    out = folder / "failure-analysis"
    out.mkdir(exist_ok=True)
    event_rows, summaries, figures = [], [], []
    for method_dir in sorted((folder / "table1").iterdir()):
        if not (method_dir / "episodes.csv").exists():
            continue
        # v8 reused both fixed groups from v7; analyze those only once.
        if run == "r1-v8-phase-consistent" and method_dir.name != "adaptive":
            continue
        rows = csv_rows(method_dir / "episodes.csv")
        chosen = {0}
        failures = [int(r["seed"]) for r in rows if not int(r["success"])]
        if failures:
            chosen.add(min(failures))
        group = []
        for row in rows:
            path = ROOT / row["trajectory"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Recorded trajectory identity differs: {path}")
            d = np.load(path)
            s = signals(d)
            time = s["time"]
            tail = time >= max(2, time[-1]-2)
            events = dict(run=run, method=method_dir.name, seed=int(row["seed"]),
                success=int(row["success"]), duration_s=float(row["duration_s"]),
                first_tilt20_100ms_s=first_sustained(time, s["tilt"] > 20, .1),
                first_saturation50ms_s=first_sustained(time, s["saturation"], .05),
                first_missing_stance100ms_s=first_sustained(time, s["missing"] >= 1, .1),
                first_low_support100ms_s=first_sustained(time, s["support"] < .5*24.638*9.81, .1),
                tail_saturation_fraction=float(s["saturation"][tail].mean()),
                tail_missing_stance_fraction=float((s["missing"][tail] >= 1).mean()),
                tail_loaded_slip_fraction=float(s["loaded_slip"][tail].mean()),
                status2_fraction=float((d["solver_status"] == 2).mean()),
                max_reference_pitch_deg=float(np.abs(np.rad2deg(d["reference_state"][:, 7])).max()),
                trajectory=row["trajectory"], sha256=digest)
            event_rows.append(events)
            group.append(events)
            if int(row["seed"]) in chosen:
                name=f"{method_dir.name}-seed-{int(row['seed']):03d}"
                plot_episode(d, dict(row, run=run), out/name)
                figures.append(name+".png")
        failed = [r for r in group if not r["success"]]
        summaries.append(dict(method=method_dir.name, episodes=len(group), failures=len(failed),
            success=sum(r["success"] for r in group),
            mean_time_without_fall_s=float(np.mean([r["duration_s"] for r in group])),
            median_duration_s=float(np.median([r["duration_s"] for r in group])),
            missing_before_tilt=sum(r["first_missing_stance100ms_s"] is not None and
                r["first_tilt20_100ms_s"] is not None and
                r["first_missing_stance100ms_s"] < r["first_tilt20_100ms_s"] for r in failed),
            saturation_before_tilt=sum(r["first_saturation50ms_s"] is not None and
                r["first_tilt20_100ms_s"] is not None and
                r["first_saturation50ms_s"] < r["first_tilt20_100ms_s"] for r in failed),
            median_tail_missing_fraction=float(np.median([r["tail_missing_stance_fraction"] for r in failed]))
            if failed else None))
    write_csv(out/"events.csv", event_rows)
    (out/"index.json").write_text(json.dumps(dict(run=run, command=f"python scripts/analyze_tm01_failures.py --runs {run}",
        unique_episode_count=len(event_rows), summaries=summaries, figures=figures,
        event_definition="First event after 2 s: tilt >20 degrees for 100 ms; any joint at 39.96 N m for 50 ms; at least one planned stance foot absent for 100 ms; total measured Fz <0.5 mg for 100 ms. Descriptive events only, not failure criteria or causal proof.",
        old_contact_reconstruction="phase < 0.65, matching the executed PeriodicGaitGenerator; v7/v8 did not save planned_contact, liftoff, desired feet, or pre-clipping torque.",
        retained_limit="An event can precede a later recovered episode. Ordering alone does not establish that an implementation difference caused failure. All outcomes retained; no post-fall tracking error is invented."
        ), indent=2)+"\n")
    print(json.dumps(dict(run=run, summaries=summaries), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    for run_name in parser.parse_args().runs:
        analyze(run_name)
