#!/usr/bin/env python3
"""Recover unsaved lift-off references using the executed helper and saved states.

This is an offline replay of the original state trajectory, not a counterfactual
closed-loop run. It quantifies the reference drift; the restored full experiments
are needed to measure its effect on outcomes.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from gym_quadruped.utils.quadruped_utils import LegsAttr
from quadruped_pympc import config as cfg
from quadruped_pympc.helpers.foothold_reference_generator import FootholdReferenceGenerator
from quadruped_pympc.helpers.terrain_estimator import TerrainEstimator
from quadruped_pympc.helpers.quadruped_utils import GaitType

ROOT = Path(__file__).resolve().parents[1]
LEGS = ("FL", "FR", "RL", "RR")
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--runs", nargs="+", required=True)
a = p.parse_args()

for run in a.runs:
    root = ROOT/"results/terrain_memory/reproduction/pympc"/run
    output = root/"failure-analysis"
    output.mkdir(exist_ok=True)
    touchdown_rows, summary_rows = [], []
    for method_dir in sorted((root/"table1").iterdir()):
        if not (method_dir/"episodes.csv").exists():
            continue
        if run == "r1-v8-phase-consistent" and method_dir.name != "adaptive":
            continue
        rows = list(csv.DictReader((method_dir/"episodes.csv").open()))
        meta = json.loads((method_dir/"run.json").read_text())
        # v7/v8 predate the explicit frame field; their executed helper followed
        # the body. Later runs record the actual choice in the runtime metadata.
        frame = "body" if run in ("r1-v7-paper-model", "r1-v8-phase-consistent") else meta["liftoff_frame"]
        cfg.locomotion_params["liftoff_frame"] = frame
        cfg.hip_offset_x, cfg.hip_offset_y = 0., .1 if frame == "world" else .083
        for row in rows:
            # Load each compressed array once before the 500 Hz replay loop.
            with np.load(ROOT/row["trajectory"]) as archive:
                d = {key: archive[key] for key in archive.files}
            time, feet, base = d["time"], d["feet"], d["qpos"][:, :3]
            plan = d["planned_contact"] if "planned_contact" in d else (d["phase"] < .65).astype(float)
            frg = FootholdReferenceGenerator(.65/1.3, LegsAttr(**dict(zip(LEGS, feet[0]))), hip_height=.35)
            terrain = TerrainEstimator()
            previous = np.ones(4)
            liftoff = feet[0].copy()
            reference_error = []
            stored_liftoff_error = []
            for i, t in enumerate(time):
                roll, pitch, height, _ = terrain.compute_terrain_estimation(
                    base[i], d["orientation"][i, 2], frg.lift_off_positions, previous)
                if np.isfinite(d["solve_seconds"][i]):
                    expected = np.array([roll, pitch, .35+height-base[i, 2]+d["com"][i, 2]])
                    reference_error.append(expected-d["reference_state"][i, [6, 7, 2]])
                for leg in np.flatnonzero((previous == 0) & (plan[i] == 1)):
                    drift = frg.lift_off_positions[LEGS[leg]]-liftoff[leg]
                    end = min(len(time), i+round(.2/.002))
                    loaded = np.flatnonzero(d["actual_grf"][i:end, leg, 2] > 5)
                    touchdown_rows.append(dict(run=run, method=method_dir.name, seed=int(row["seed"]),
                        leg=LEGS[leg], planned_touchdown_s=float(t),
                        body_tilt_deg=float(np.rad2deg(np.abs(d["orientation"][i,:2])).max()),
                        original_liftoff_z_m=float(liftoff[leg,2]),
                        reconstructed_liftoff_z_m=float(frg.lift_off_positions[LEGS[leg]][2]),
                        liftoff_drift_xy_m=float(np.linalg.norm(drift[:2])),
                        liftoff_drift_z_m=float(drift[2]),
                        actual_foot_z_m=float(feet[i, leg, 2]),
                        loaded_contact_delay_s=float(loaded[0]*.002) if len(loaded) else None,
                        trajectory=row["trajectory"]))
                new_lift = (previous == 1) & (plan[i] == 0)
                liftoff[new_lift] = feet[i, new_lift]
                frg.update_lift_off_positions(previous, plan[i], LegsAttr(**dict(zip(LEGS, feet[i]))),
                    LEGS, GaitType.TROT.value, base[i], d["orientation"][i])
                if "liftoff" in d:
                    stored_liftoff_error.append(np.array([frg.lift_off_positions[l] for l in LEGS])-d["liftoff"][i])
                previous = plan[i]
            err = np.array(reference_error)
            summary_rows.append(dict(run=run, method=method_dir.name, seed=int(row["seed"]),
                liftoff_frame=frame, samples=len(time),
                reference_pitch_replay_rmse_rad=float(np.sqrt(np.mean(err[:, 1]**2))),
                reference_height_replay_rmse_m=float(np.sqrt(np.mean(err[:, 2]**2))),
                stored_liftoff_max_error_m=float(np.abs(stored_liftoff_error).max()) if stored_liftoff_error else None))
        print(json.dumps(dict(run=run, method=method_dir.name, episodes=len(rows))), flush=True)
    for name, rows in [("touchdowns.csv", touchdown_rows), ("reference_replay.csv", summary_rows)]:
        with (output/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
