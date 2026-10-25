#!/usr/bin/env python3
"""Summarize task support and paired outcomes from the frozen second S1 run.

No simulation, training, tests or archive download. Reads existing compact data.
Cross-leg rows are material-column events, not independent footstep trials.
Use actual e_before_step for pre-reload state, not the free-boundary candidate.
Missing legacy B0 diagnostics remain blank. Paired performance differences use
only pairs in which both full missions succeeded; success counts remain separate.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "5af2c25c5223c7b74a473133fe97aed3d9962c34"
RESULT = ROOT / "results/terrain_memory/simulation/tm-s1-control-repair-20261024"
CONFIG = ROOT / "experiments/terrain_memory/s1-control-repair.json"


def read_csv(name):
    with (RESULT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def bounds(values):
    return (min(values), max(values)) if values else (None, None)


def main():
    episode_files = sorted((RESULT / "formal-v1").glob("*-seed-*.json"))
    episodes = [json.loads(path.read_text()) for path in episode_files]
    events = read_csv("actual-cross-leg-revisits.csv")
    stance = read_csv("stance-tracking.csv")
    decisions = read_csv("decisions.csv")
    commitments = read_csv("footstep-commitments.csv")
    config = json.loads(CONFIG.read_text())
    terrains = {row["id"]: row for row in config["terrains"]}
    inputs = episode_files + [CONFIG] + [RESULT / name for name in (
        "actual-cross-leg-revisits.csv", "stance-tracking.csv", "decisions.csv",
        "footstep-commitments.csv")]
    manifest = "\n".join(
        path.relative_to(ROOT).as_posix() + " " + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(inputs)) + "\n"
    manifest_sha = hashlib.sha256(manifest.encode()).hexdigest()
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode["terrain"], episode["controller"]].append(episode)

    output = []
    for (terrain, controller), group in sorted(grouped.items()):
        reference = "B0" if controller == "B1" else "B1" if controller in ("B2", "B3") else None
        base = {row["seed"]: row for row in grouped.get((terrain, reference), [])}
        pairs = [(base[row["seed"]], row) for row in group
                 if row["success"] and row["seed"] in base and base[row["seed"]]["success"]]
        completed = [row["duration_s"] for row in group if row["success"]]
        cross = [row for row in events if (row["terrain"], row["controller"]) == (terrain, controller)]
        gaps = [float(row["unloaded_gap_s"]) for row in cross if row["unloaded_gap_s"]]
        # TM02 e is dimensionless strain; KV e is signed displacement in metres.
        scale_mm = terrains[terrain].get("thickness_m", 1.) * 1000
        actual = [(float(row["area_m2"]), abs(json.loads(row["e_before_step"])[0]) * scale_mm)
                  for row in cross if row["e_before_step"]]
        residual = [value for _, value in actual]
        tau = (terrains[terrain]["d_N_s_per_m"] / terrains[terrain]["k_N_per_m"]
               if terrain.startswith("kv_") else None)
        rates = {name: [row["material_rate_extrema"][name]["magnitude"]
                        for row in group if row.get("material_rate_extrema", {}).get(name)]
                 for name in ("continuous_loaded_strain_rate_s_inv", "actual_foot_vz_over_H_s_inv", "internal_qdot_s_inv")}
        active = [row for row in stance if (row["terrain"], row["controller"]) == (terrain, controller)]
        inactive_seeds = [seed for seed in sorted({int(row["seed"]) for row in active})
                          if sum(int(row["active_reference_samples"]) for row in active
                                 if int(row["seed"]) == seed) == 0]
        selected = [row for row in decisions if (row["terrain"], row["controller"]) == (terrain, controller)
                    and row["selected"] == "True"]
        committed = [row for row in commitments
                     if (row["terrain"], row["controller"]) == (terrain, controller)]
        item = dict(
            terrain=terrain, controller=controller, episodes=len(group), successes=len(completed),
            entered_turn=sum(any(p["phase"] == "turn" for p in row["transitions"]) for row in group),
            entered_return=sum(any(p["phase"] == "return" for p in row["transitions"]) for row in group),
            terminations=json.dumps(dict(sorted(Counter(row["termination"] for row in group).items()))),
            successful_completion_time_mean_s=mean(completed) if completed else None,
            paired_reference=reference, paired_both_successful=len(pairs) if reference else None,
        )
        for field in ("duration_s", "joint_positive_mechanical_work_J", "contact_slip_distance_m", "attitude_rms_rad", "grf_rmse_N"):
            delta = [b[field] - a[field] for a, b in pairs]
            item["paired_" + field + "_mean_difference"] = mean(delta) if delta else None
        item.update(
            cross_leg_patch_events=len(cross),
            front_to_hind_patch_events=sum(int(row["previous_leg"]) in (0, 1) and int(row["leg"]) in (2, 3) for row in cross),
            observed_gap_min_s=min(gaps) if gaps else None,
            observed_gap_median_s=median(gaps) if gaps else None,
            kv_zero_load_tau_s=tau,
            minimum_observed_gap_over_tau=min(gaps) / tau if gaps and tau else None,
            actual_preload_state_recorded_events=len(actual),
            actual_preload_state_displacement_max_mm=max(residual) if residual else None,
            actual_preload_state_displacement_median_mm=median(residual) if residual else None,
            actual_preload_state_displacement_area_weighted_mean_mm=(
                sum(area * value for area, value in actual) / sum(area for area, _ in actual)
                if actual else None),
            fully_inactive_stance_seeds=json.dumps(inactive_seeds) if active else None,
            selected_planning_decisions=len(selected) if selected else None,
            selected_continuation_decisions=sum(row["continuation"] == "True" for row in selected) if selected else None,
            actual_offset_commitments=len(committed) if committed else None,
            actual_nonzero_offset_commitments=sum(float(row["offset_m"]) != 0 for row in committed) if committed else None,
        )
        for name, values in rates.items():
            item[name + "_episode_maximum_min"], item[name + "_episode_maximum_max"] = bounds(values)
        item.update(source_commit=SOURCE_COMMIT, source_input_files=len(inputs),
                    source_input_manifest_sha256=manifest_sha,
                    interpretation="Patch events are not independent tasks; pre-reload state is actual e_before_step; paired deltas exclude failed pairs while success counts retain failures.")
        output.append(item)

    destination = RESULT / "task-support-summary.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(dict(output=str(destination), groups=len(output), episodes=len(episodes),
                         successes=sum(row["success"] for row in episodes),
                         source_input_manifest_sha256=manifest_sha,
                         new_simulations=0, new_training_runs=0), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
