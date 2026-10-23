#!/usr/bin/env python3
"""The four declared R1 frequency cohorts: full-task outcomes and matched work."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/terrain_memory/reproduction/pympc"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", default="r1-closeout-20261022")
args = parser.parse_args()
DEST = BASE / args.run
DEST.mkdir(parents=True, exist_ok=True)
GROUPS = [
    ("fixed_1p3", "Fixed 1.3 Hz", "r1-foothold-clock-repaired", "adaptive_fixed", False),
    ("fixed_2p0", "Fixed 2.0 Hz", args.run + "-fixed-2p0", "adaptive_fixed", True),
    ("fixed_2p4", "Fixed 2.4 Hz", args.run + "-fixed-2p4", "adaptive_fixed", True),
    ("adaptive", "Adaptive", "r1-foothold-clock-repaired", "adaptive", False),
]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
STYLES = [":", "--", "-.", "-"]
MARKERS = ["o", "s", "^", "D"]
DT = 0.002
BOOTSTRAPS = 5000
ANALYSIS_SEED = 20261022
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_csv(name, records):
    with (DEST / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None, None
    # The independent resampling unit is an episode/paired seed, never a 500 Hz tick.
    rng = np.random.default_rng(ANALYSIS_SEED)
    boot = values[rng.integers(0, len(values), (BOOTSTRAPS, len(values)))].mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    # At zero/all successes the Wilson endpoint is exactly zero/one;
    # subtracting its two equal floating-point terms can leave a tiny residue.
    return (0. if k == 0 else center-half), (1. if k == n else center+half)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(DEST / (name + "." + ext), dpi=300)
    plt.close(fig)


def model_mapping(path):
    model = mujoco.MjModel.from_binary_path(str(path))
    joint_ids = model.actuator_trnid[:, 0]
    dofs = model.jnt_dofadr[joint_ids]
    # These checks establish the physical meaning of the saved action, rather
    # than guessing that actuator order equals the order of generalized speeds.
    assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
    assert np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
    assert np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
    assert np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
    assert np.all(model.actuator_gainprm[:, 0] == 1)
    assert np.all(model.actuator_gear[:, 0] == 1)
    record = dict(model=str(path.relative_to(ROOT)), model_sha256=sha(path),
        timestep_s=float(model.opt.timestep), nu=model.nu, nv=model.nv,
        transmission="Direct joint torque: unit gear, unit fixed gain, no bias or activation dynamics",
        actuator_qvel_indices=dofs.tolist(), actuators=[dict(
            actuator_index=i,
            actuator=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_ids[i])),
            qvel_index=int(dofs[i]), gear=model.actuator_gear[i].tolist(),
            ctrlrange=model.actuator_ctrlrange[i].tolist()) for i in range(model.nu)])
    return dofs, record


def prefix_metrics(trace, count):
    return dict(velocity_error_mean_m_s=float(trace["error_cumsum"][count-1]/count),
        velocity_error_rmse_m_s=float(np.sqrt(trace["square_cumsum"][count-1]/count)),
        positive_work_J=float(trace["power_cumsum"][count-1]*DT),
        mean_positive_power_W=float(trace["power_cumsum"][count-1]/count))


episodes, models, cohorts, cohort_sources = [], [], {}, []
for key, label, run, method, new in GROUPS:
    folder = BASE / run / "table1" / method
    with (folder / "episodes.csv").open() as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r["seed"]))
    assert [int(r["seed"]) for r in rows] == list(range(50)), (run, "Incomplete fixed budget")
    meta = json.loads((folder / "run.json").read_text())
    model_path = (ROOT / rows[0]["trajectory"]).parent / "model.mjb"
    dofs, mapping = model_mapping(model_path)
    mapping.update(group=key, source_run=run, method=method)
    models.append(mapping)
    cohort_sources.append(dict(group=key, label=label, source_run=run, method=method,
        newly_executed=new, episodes=len(rows), run_record=str((folder/"run.json").relative_to(ROOT)),
        configuration=meta["configuration"], configuration_sha256=meta["configuration_sha256"],
        source_snapshot=meta["source_snapshot"], source_commit=meta["source_commit"],
        model_sha256=mapping["model_sha256"], candidate_hz=meta["candidate_hz"],
        timing_context=meta["timing_context"], cpu=meta["cpu"], device=meta["device"], gpu=meta["gpu"]))
    cohort = {}
    for row in rows:
        seed = int(row["seed"])
        path = ROOT / row["trajectory"]
        assert sha(path) == row["sha256"], path
        with np.load(path) as archive:
            t = archive["time"]
            error = np.linalg.norm(archive["velocity"] - archive["command"], axis=1)
            torque = archive["torque"]
            power = np.maximum(torque * archive["qvel"][:, dofs], 0).sum(axis=1)
            wrench = archive["wrench"]
            assert np.allclose(t, np.arange(len(t))*DT, atol=1e-12, rtol=0)
            assert np.isclose(len(t)*DT, float(row["duration_s"]))
            assert float(row["requested_duration_s"]) == 60
            cohort[seed] = dict(row=row, count=len(t), error_cumsum=error.cumsum(),
                square_cumsum=(error**2).cumsum(), power_cumsum=power.cumsum(),
                wrench=wrench, schedule=archive["wrench_schedule"], command=archive["command"],
                initial_qpos=archive["qpos"][0], initial_qvel=archive["qvel"][0])
            episodes.append(dict(group=key, label=label, source_run=run,
                newly_executed=int(new), **row,
                **{"observed_"+k:v for k,v in prefix_metrics(cohort[seed], len(t)).items()},
                samples_500hz=len(t)))
    cohorts[key] = cohort

dump_csv("episodes.csv", episodes)
(DEST/"actuator_mapping.json").write_text(json.dumps(models, indent=2)+"\n")
summary = []
for key, label, run, method, new in GROUPS:
    rows = [c["row"] for c in cohorts[key].values()]
    durations = np.minimum(60, [float(r["duration_s"]) for r in rows])
    successes = sum(int(r["success"]) for r in rows)
    lo, hi = wilson(successes, 50)
    mean, mlo, mhi = mean_ci(durations)
    summary.append(dict(group=key, label=label, source_run=run, method=method,
        newly_executed=int(new), episodes=50, successes=successes,
        success_percent=2*successes, success_ci95_low_percent=100*lo,
        success_ci95_high_percent=100*hi, mean_capped_duration_s=mean,
        mean_capped_duration_ci95_low_s=mlo, mean_capped_duration_ci95_high_s=mhi,
        median_capped_duration_s=float(np.median(durations))))
dump_csv("frequency_summary.csv", summary)

# The event grid includes every failure time and both horizon endpoints.
grid = sorted({0., 60.} | {float(r["duration_s"]) for r in episodes})
survival = []
fig, ax = plt.subplots(figsize=(7.16, 4.1), layout="constrained")
for (key, label, _, _, _), color, style in zip(GROUPS, COLORS, STYLES):
    alive = []
    for time in grid:
        k = sum(float(c["row"]["duration_s"]) > time or
            (int(c["row"]["success"]) and float(c["row"]["duration_s"]) >= time)
            for c in cohorts[key].values())
        lo, hi = wilson(k, 50)
        alive.append(2*k)
        survival.append(dict(group=key, time_s=time, alive=k, episodes=50,
            survival_percent=2*k, pointwise_ci95_low_percent=100*lo,
            pointwise_ci95_high_percent=100*hi))
    ax.step(grid, alive, where="post", label=label, color=color, ls=style, lw=1.6)
ax.set(xlim=(0, 60), ylim=(0, 102), xlabel="Simulation time (s)",
    ylabel="Episodes without a fall (%)", title="The entire fixed 60 s task | 50 matched seeds per group")
ax.grid(alpha=.2)
ax.legend(loc="upper right", frameon=False)
fig.supxlabel("Failure times shown exactly; completed episodes are censored at 60 s.\n"
    "Fixed 1.3 Hz and adaptive reuse the clock-repaired cohorts; high-frequency groups are new.", fontsize=8)
save(fig, "frequency_survival")
dump_csv("frequency_survival.csv", survival)

fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.1), layout="constrained")
for i, (s, color, marker) in enumerate(zip(summary, COLORS, MARKERS)):
    axes[0].errorbar(i, s["success_percent"], yerr=[[
        s["success_percent"]-s["success_ci95_low_percent"]], [
        s["success_ci95_high_percent"]-s["success_percent"]]], fmt=marker, color=color, capsize=4)
    axes[0].annotate(f'{s["successes"]}/50', (i, s["success_percent"]),
        xytext=(0, 13), textcoords="offset points", ha="center", fontsize=8)
    values = [float(c["row"]["duration_s"]) for c in cohorts[s["group"]].values()]
    axes[1].scatter(i+np.linspace(-.14, .14, 50), values, s=12, marker=marker,
        color=color, alpha=.6, linewidths=0)
    axes[1].plot(i, s["mean_capped_duration_s"], "_", color="black", ms=18, mew=2)
axes[0].set(ylabel="Completed 60 s episodes (%)", ylim=(0, 112))
axes[1].set(ylabel="Time without a fall, capped at 60 s", ylim=(0, 63))
for axis in axes:
    axis.set_xticks(range(4), ["Fixed\n1.3 Hz", "Fixed\n2.0 Hz", "Fixed\n2.4 Hz", "Adaptive"])
    axis.grid(axis="y", alpha=.2)
fig.suptitle("Four frequencies under identical task and controller settings")
fig.supxlabel("Left: 95% Wilson intervals. Right: all 50 episodes; black bar = mean.\n"
    "Failures remain in every full-task outcome statistic.", fontsize=8)
save(fig, "frequency_outcomes")

pairs, paired_summary, windows, metric_summary = [], [], [], []
for key, label, run, method, _ in GROUPS[:3]:
    group_pairs = []
    for seed in range(50):
        fixed, adaptive = cohorts[key][seed], cohorts["adaptive"][seed]
        x, y = fixed["row"], adaptive["row"]
        n = min(fixed["count"], adaptive["count"])
        # Establish that a pair saw the same disturbance and reset, not just
        # that two filenames happen to contain the same integer seed.
        assert np.array_equal(fixed["schedule"], adaptive["schedule"])
        assert np.array_equal(fixed["wrench"][:n], adaptive["wrench"][:n])
        assert np.array_equal(fixed["command"][:n], adaptive["command"][:n])
        assert np.array_equal(fixed["initial_qpos"], adaptive["initial_qpos"])
        assert np.array_equal(fixed["initial_qvel"], adaptive["initial_qvel"])
        record = dict(fixed_group=key, seed=seed, fixed_source_run=run,
            fixed_method=method, adaptive_source_run=GROUPS[3][2], adaptive_method="adaptive",
            fixed_success=int(x["success"]), adaptive_success=int(y["success"]),
            fixed_duration_s=float(x["duration_s"]), adaptive_duration_s=float(y["duration_s"]),
            adaptive_minus_fixed_duration_s=float(y["duration_s"])-float(x["duration_s"]),
            common_start_s=0, common_end_exclusive_s=n*DT, common_samples=n,
            fixed_trajectory=x["trajectory"], fixed_sha256=x["sha256"],
            adaptive_trajectory=y["trajectory"], adaptive_sha256=y["sha256"])
        for side, trace in (("fixed", fixed), ("adaptive", adaptive)):
            record.update({side+"_"+k:v for k,v in prefix_metrics(trace, n).items()})
        group_pairs.append(record)
        windows.append(dict(comparison=key, window="pair_common_survival", **record))
        if int(x["success"]) and int(y["success"]):
            windows.append(dict(comparison=key, window="pair_both_complete_60s", **record))
    pairs.extend(group_pairs)
    delta = np.array([r["adaptive_minus_fixed_duration_s"] for r in group_pairs])
    mean, lo, hi = mean_ci(delta)
    completion_delta = [r["adaptive_success"]-r["fixed_success"] for r in group_pairs]
    cp, cl, ch = mean_ci(np.array(completion_delta)*100)
    paired_summary.append(dict(fixed_group=key, n_pairs=50,
        adaptive_minus_fixed_mean_duration_s=mean, duration_ci95_low_s=lo, duration_ci95_high_s=hi,
        adaptive_longer=int((delta>0).sum()), adaptive_shorter=int((delta<0).sum()), tied=int((delta==0).sum()),
        both_complete=sum(r["fixed_success"] and r["adaptive_success"] for r in group_pairs),
        adaptive_only_complete=sum(not r["fixed_success"] and r["adaptive_success"] for r in group_pairs),
        fixed_only_complete=sum(r["fixed_success"] and not r["adaptive_success"] for r in group_pairs),
        both_fail=sum(not r["fixed_success"] and not r["adaptive_success"] for r in group_pairs),
        adaptive_minus_fixed_success_pp=cp, success_difference_ci95_low_pp=cl, success_difference_ci95_high_pp=ch))
    for window in ("pair_common_survival", "pair_both_complete_60s"):
        selected = [r for r in windows if r["comparison"]==key and r["window"]==window]
        for metric in ("velocity_error_mean_m_s", "velocity_error_rmse_m_s", "positive_work_J", "mean_positive_power_W"):
            va = np.array([r["fixed_"+metric] for r in selected])
            vb = np.array([r["adaptive_"+metric] for r in selected])
            am, al, ah = mean_ci(va)
            bm, bl, bh = mean_ci(vb)
            dm, dl, dh = mean_ci(vb-va)
            metric_summary.append(dict(fixed_group=key, window=window, metric=metric, n_pairs=len(selected),
                start_s=0, mean_end_s=float(np.mean([r["common_end_exclusive_s"] for r in selected])) if selected else None,
                fixed_mean=am, fixed_ci95_low=al, fixed_ci95_high=ah,
                adaptive_mean=bm, adaptive_ci95_low=bl, adaptive_ci95_high=bh,
                adaptive_minus_fixed_mean=dm, paired_ci95_low=dl, paired_ci95_high=dh))

dump_csv("paired_episodes.csv", pairs)
dump_csv("paired_outcomes.csv", paired_summary)
dump_csv("matched_tracking_work.csv", windows)
dump_csv("matched_tracking_work_summary.csv", metric_summary)

# A four-way common prefix supports an additional direct four-group comparison.
# There are no four-way complete episodes because fixed 1.3 Hz completes none.
common_rows = []
for seed in range(50):
    n = min(cohorts[key][seed]["count"] for key, *_ in GROUPS)
    for key, label, run, method, _ in GROUPS:
        trace = cohorts[key][seed]
        common_rows.append(dict(group=key, seed=seed, source_run=run, method=method,
            start_s=0, end_exclusive_s=n*DT, samples=n,
            trajectory=trace["row"]["trajectory"], sha256=trace["row"]["sha256"],
            **prefix_metrics(trace, n)))
dump_csv("four_group_common_prefix.csv", common_rows)
common_summary = []
for key, *_ in GROUPS:
    selected = [r for r in common_rows if r["group"]==key]
    for metric in ("velocity_error_mean_m_s", "velocity_error_rmse_m_s", "positive_work_J", "mean_positive_power_W"):
        mean, lo, hi = mean_ci([r[metric] for r in selected])
        common_summary.append(dict(group=key, window="four_way_common_survival", metric=metric,
            n_episodes=len(selected), mean=mean, ci95_low=lo, ci95_high=hi,
            start_s=0, mean_end_exclusive_s=float(np.mean([r["end_exclusive_s"] for r in selected]))))
dump_csv("four_group_common_prefix_summary.csv", common_summary)

fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.4), layout="constrained")
for axis, group, s, color in zip(axes, GROUPS[:3], paired_summary, COLORS):
    rr = [r for r in pairs if r["fixed_group"]==group[0]]
    axis.scatter([r["fixed_duration_s"] for r in rr], [r["adaptive_duration_s"] for r in rr],
        s=15, alpha=.65, color=color)
    axis.plot([0,60], [0,60], "k--", lw=.8)
    axis.set(xlim=(0,62), ylim=(0,62), xlabel=group[1]+" (s)", xticks=[0,20,40,60], yticks=[0,20,40,60])
    axis.set_title(f"Change {s['adaptive_minus_fixed_mean_duration_s']:+.2f} s\n"
        f"[{s['duration_ci95_low_s']:+.2f}, {s['duration_ci95_high_s']:+.2f}]", fontsize=9)
    axis.grid(alpha=.2)
axes[0].set_ylabel("Adaptive time without a fall (s)")
fig.suptitle("Adaptive versus each fixed frequency | all 50 paired seeds")
fig.supxlabel("Titles: paired mean duration change and 95% bootstrap interval (5,000 seed resamples).\n"
    "Points at (60, 60) overlap; exact outcome counts are in paired_outcomes.csv.", fontsize=8)
save(fig, "paired_durations")

fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.3), layout="constrained")
for col, window in enumerate(("pair_common_survival", "pair_both_complete_60s")):
    for row, metric in enumerate(("velocity_error_rmse_m_s", "positive_work_J")):
        axis = axes[row, col]
        for i, (key, _, _, _, _) in enumerate(GROUPS[:3]):
            s = next(r for r in metric_summary if r["fixed_group"]==key and r["window"]==window and r["metric"]==metric)
            rr = [r for r in windows if r["fixed_group"]==key and r["window"]==window]
            if rr:
                d = [r["adaptive_"+metric]-r["fixed_"+metric] for r in rr]
                axis.scatter(i+np.linspace(-.13,.13,len(d)), d, s=10, color=COLORS[i], alpha=.45)
                axis.errorbar(i, s["adaptive_minus_fixed_mean"],
                    yerr=[[s["adaptive_minus_fixed_mean"]-s["paired_ci95_low"]],
                          [s["paired_ci95_high"]-s["adaptive_minus_fixed_mean"]]],
                    fmt=MARKERS[i], color="black", capsize=3, markersize=4)
            else:
                axis.annotate("No shared\ncompletions", (i, 0), xytext=(0, 15),
                    textcoords="offset points", ha="center", fontsize=8)
        axis.axhline(0, color="black", ls="--", lw=.8)
        counts = [next(r["n_pairs"] for r in metric_summary if r["fixed_group"]==g[0] and r["window"]==window) for g in GROUPS[:3]]
        axis.set_xticks(range(3), [f"{hz} Hz\nn={n}" for hz,n in zip((1.3,2.0,2.4),counts)])
        axis.set_xlim(-.5, 2.5)
        axis.grid(axis="y", alpha=.2)
axes[0,0].set_title("Pairwise common survival: 0 to first fall", fontsize=9)
axes[0,1].set_title("Both controllers complete: 0 to 60 s", fontsize=9)
axes[0,0].set_ylabel("Adaptive - fixed\nvelocity RMSE (m/s)")
axes[1,0].set_ylabel("Adaptive - fixed\nmechanical positive work (J)")
fig.suptitle("Tracking and work over equal within-pair time intervals")
fig.supxlabel("Dots: paired seeds. Black: mean and pointwise 95% paired bootstrap interval.\n"
    "Completion-conditioned subsets are descriptive; no battery-energy claim. Negative = lower for adaptive.", fontsize=8)
save(fig, "matched_tracking_work")

index = dict(run=args.run, command=" ".join(sys.argv), training=0,
    new_frequency_episodes=100, referenced_frequency_episodes=100, included_frequency_episodes=200,
    cohorts=cohort_sources, summary=summary, paired_outcomes=paired_summary,
    statistics=dict(success_interval="95% Wilson", duration="min(observed duration, 60 s), including every failure",
        bootstrap="95% percentile, 5,000 resamples of paired seeds; equal episode weight",
        analysis_seed=ANALYSIS_SEED, interval_scope="Pointwise descriptive intervals, not simultaneous multi-comparison claims",
        median="Median of observed times capped at 60 s; this is not an uncensored median time to failure",
        survival="Exact failure times; completed episodes right-censored only at the fixed 60 s endpoint"),
    tracking=dict(error="World-frame linear-velocity error norm ||v(t)-v_command(t)||, in m/s",
        rmse="sqrt(mean_t ||v(t)-v_command(t)||^2) within the explicitly recorded window",
        aggregate="Arithmetic mean of per-episode mean errors or per-episode RMSE; 95% intervals resample episode pairs",
        windows="matched_tracking_work.csv gives each [0,T) and its 500 Hz sample count; complete-pair windows have T=60 s",
        interpretation="Matched observed-state diagnostics, not an assigned post-fall error or the paper's incompletely defined Mean Cost"),
    mechanical_work=dict(formula="0.002 * sum_t sum_i max(torque[t,i] * qvel[t,model_dof[i]], 0)",
        torque="Saved torque is the clipped executed control passed to env.step; actual saved model has unit joint transmission and gain",
        integration="Left rectangle using pre-step joint velocities and held executed torque for each 0.002 s step",
        meaning="Estimate of positive actuator mechanical work; excludes electrical losses and is not battery energy",
        mapping="actuator_mapping.json", pair_windows="matched_tracking_work.csv",
        complete_windows="Both-success pairs only, entire [0,60); conditional on both completing",
        common_windows="Each matched pair uses [0,min(duration_fixed,duration_adaptive)); no zero-padding after a fall",
        four_way_windows="four_group_common_prefix.csv; 50 seeds with the same prefix for all four groups",
        four_way_complete_episodes=sum(all(int(cohorts[g[0]][s]["row"]["success"]) for g in GROUPS) for s in range(50)),
        aggregation="Mean of episode work and per-episode mean positive power; intervals resample episode pairs",
        observed_columns="episodes.csv observed_* describe unequal observed durations and are not used as energy rankings"),
    matched_identity="All 150 adaptive/fixed pairs have identical wrench schedules, executed common-prefix wrenches, commands and initial qpos/qvel; every trajectory hash verified",
    figures=["frequency_outcomes.png", "frequency_outcomes.svg", "frequency_survival.png", "frequency_survival.svg",
        "paired_durations.png", "paired_durations.svg", "matched_tracking_work.png", "matched_tracking_work.svg"],
    data=["episodes.csv", "frequency_summary.csv", "frequency_survival.csv", "paired_episodes.csv", "paired_outcomes.csv",
        "matched_tracking_work.csv", "matched_tracking_work_summary.csv", "four_group_common_prefix.csv",
        "four_group_common_prefix_summary.csv", "actuator_mapping.json"],
    source_snapshot=dict(script="scripts/analyze_tm01_closeout.py", sha256=sha(Path(__file__))),
    reference_downloads=json.loads((BASE/"r1-foothold-clock-repaired/index.json").read_text())["downloads"],
    protocol_verification=dict(
        paper="https://arxiv.org/html/2403.11383v2#S5.SS2",
        paper_pdf_sha256="7497d4b461c83f7593d2d5afe2af626a593a239a3c8e89dba3b97654f0f3f7ed",
        checked_scope="Figure 4, Table I and their directly associated text; dated author simulation entry and config",
        fig4_confirmed="40 N lateral push at the CoM and an additional lateral velocity command of 0.1 m/s",
        fig4_timing_conflict="Caption states a 3 s push; main text describes onset around 2 s and removal around 4.2 s. These do not define one consistent exact schedule.",
        fig4_missing=["Complete time histories of forward, vertical and angular velocity commands and their frames", "Unambiguous exact force onset, duration and removal"],
        table1_confirmed="50 episodes; random CoM wrenches in the +/-20 N,Nm range",
        table1_missing=["Episode duration and complete command", "Reset pose/velocity and gait-phase initialization linked to Table I", "Distribution, cross-axis dependence, random seeds and temporal wrench schedule", "Exact failure/scoring definition"],
        historical_source="https://github.com/iit-DLSLab/Quadruped-PyMPC/tree/5c763068126499255c5e80898999aa83f3226ff8",
        historical_files=["simulation/simulation.py", "quadruped_pympc/config.py"],
        historical_entry="Generic interactive human-command runner, 500 episodes, reset(random=False), 20,000 steps in human mode or 2,000 otherwise. No Figure 4/Table I-specific random-wrench script was found in these directly associated files. Its defaults are not treated as the original table protocol.",
        executed_protocol="Frozen configurations retain the declared 60 s, 0.5 m/s, +/-20 N,Nm reconstruction with 2 s holds starting at 2 s and native invalid-contact termination. No tuning or shortened task was used to fit published percentages."),
    r1_complete=False, r2_started=False)
(DEST/"index.json").write_text(json.dumps(index, indent=2)+"\n")
print(json.dumps(dict(summary=summary, paired_outcomes=paired_summary), indent=2))
