#!/usr/bin/env python3
"""Measure swing-reference discontinuities on actual complete R1 trajectories."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/terrain_memory/reproduction/pympc"
LEGS=("FL","FR","RL","RR")
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--runs",nargs="+",required=True)
args=p.parse_args()
plt.rcParams.update({"font.size":9,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
for run in args.runs:
    folder=BASE/run
    out=folder/"clock-analysis"
    out.mkdir(exist_ok=True)
    rows=list(csv.DictReader((folder/"table1/adaptive/episodes.csv").open()))
    meta=json.loads((folder/"table1/adaptive/run.json").read_text())
    events,episodes=[],[]
    for row in rows:
        with np.load(ROOT/row["trajectory"]) as archive:
            d={key:archive[key] for key in ("time","frequency","planned_contact","swing_target","feet","requested_torque","torque","orientation","swing_time")}
        time=d["time"];frequency=d["frequency"];plan=d["planned_contact"]
        same_swing=(plan[:-1]==0)&(plan[1:]==0)
        jumps=np.where(same_swing,np.linalg.norm(np.diff(d["swing_target"],axis=0),axis=2),0)
        changed=np.flatnonzero(np.abs(np.diff(frequency))>.01)+1
        group=[]
        for i in changed:
            leg=int(np.argmax(jumps[i-1]))
            indices=[j for j,name in enumerate(meta["actuator_names"]) if name.startswith(LEGS[leg]+"_")]
            e=dict(seed=int(row["seed"]),time_s=float(time[i]),frequency_before_hz=float(frequency[i-1]),
                frequency_after_hz=float(frequency[i]),leg=LEGS[leg],
                continuing_swing=bool(same_swing[i-1,leg]),
                reference_jump_m=float(jumps[i-1,leg]),
                swing_time_before_s=float(d["swing_time"][i-1,leg]),swing_time_after_s=float(d["swing_time"][i,leg]),
                target_z_before_m=float(d["swing_target"][i-1,leg,2]),target_z_after_m=float(d["swing_target"][i,leg,2]),
                actual_z_m=float(d["feet"][i,leg,2]),
                leg_requested_torque_max_Nm=float(np.abs(d["requested_torque"][i,indices]).max()),
                all_requested_torque_max_Nm=float(np.abs(d["requested_torque"][i]).max()),
                body_tilt_deg=float(np.rad2deg(np.abs(d["orientation"][i,:2])).max()),
                trajectory=row["trajectory"])
            events.append(e);group.append(e)
        episodes.append(dict(seed=int(row["seed"]),success=int(row["success"]),duration_s=float(row["duration_s"]),
            changes=len(group),median_reference_jump_m=float(np.median([e["reference_jump_m"] for e in group])),
            fraction_jumps_over_3cm=float(np.mean([e["reference_jump_m"]>.03 for e in group]))))
        if int(row["seed"])==0:
            first=next(e for e in group if e["frequency_after_hz"]>e["frequency_before_hz"])
            # Plot the same right-rear leg across conditions, even if the other
            # member of the diagonal pair has a slightly larger residual step.
            centre=first["time_s"];leg=LEGS.index("RR")
            event_i=np.searchsorted(time,centre)
            component=int(np.argmax(np.abs(d["swing_target"][event_i,leg]-d["swing_target"][event_i-1,leg])))
            indices=[j for j,name in enumerate(meta["actuator_names"]) if name.startswith("RR_")]
            mask=(time>=centre-.1)&(time<=centre+.16)
            f,ax=plt.subplots(3,1,figsize=(7.16,6.3),sharex=True,layout="constrained")
            ax[0].step(time[mask],frequency[mask],where="post",color="#0072B2")
            ax[0].set_ylabel("Frequency (Hz)")
            ax[1].plot(time[mask],d["swing_target"][mask,leg,component],label="desired foot",color="#D55E00")
            ax[1].plot(time[mask],d["feet"][mask,leg,component],label="measured foot",color="#0072B2")
            ax[1].set_ylabel(f"Foot {'xyz'[component]} (m)");ax[1].legend()
            ax[2].plot(time[mask],np.abs(d["requested_torque"][mask][:,indices]).max(axis=1),label="requested",color="#D55E00")
            ax[2].plot(time[mask],np.abs(d["torque"][mask][:,indices]).max(axis=1),label="applied",color="#0072B2")
            ax[2].axhline(39.96,color="black",ls="--",lw=.8,label="limit")
            ax[2].set(ylabel="Max leg torque (N m)",xlabel="Simulation time (s)");ax[2].legend()
            for axis in ax:
                axis.axvline(centre,color="gray",ls=":",lw=.8);axis.grid(alpha=.2)
            f.suptitle(f"{run} | seed 0 | first frequency increase | RR")
            f.supxlabel("Same continuing swing on both sides of the change; data at 500 Hz.\n"
                "This window is selected by the first frequency increase, not by a failure or a large jump.",fontsize=8)
            for extension in ("png","svg"):f.savefig(out/("first_change."+extension),dpi=220,bbox_inches="tight")
            plt.close(f)
    for name,data in [("events.csv",events),("episodes.csv",episodes)]:
        with (out/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    summary=dict(run=run,command=f"python scripts/analyze_tm01_clock.py --runs {run}",episodes=len(rows),
        frequency_changes=len(events),median_jump_m=float(np.median([e["reference_jump_m"] for e in events])),
        fraction_jumps_over_3cm=float(np.mean([e["reference_jump_m"]>.03 for e in events])),
        low_tilt_jumps_over_3cm=sum(e["reference_jump_m"]>.03 and e["body_tilt_deg"]<20 for e in events),
        episode_median_jump_median_m=float(np.median([r["median_reference_jump_m"] for r in episodes])),
        first_change_seed0=next(e for e in events if e["seed"]==0 and e["frequency_after_hz"]>e["frequency_before_hz"]),
        example_plot_leg="RR in both conditions; event rows identify the largest-changing leg independently",
        source="quadruped_pympc/helpers/swing_trajectory_controller.py:35-42; swing_time is scaled by new_period/old_period only when phase_consistent_swing is enabled.",
        interpretation="A measured reference discontinuity and ensuing torque request are implementation evidence; their contribution to full-episode failure must be assessed with the complete paired clock-correction cohorts.")
    (out/"index.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
