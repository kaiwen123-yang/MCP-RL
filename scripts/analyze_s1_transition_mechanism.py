#!/usr/bin/env python3
"""Derive mechanism-relevant quantities from the frozen third S1 round.

Only existing compact records are read. No simulation, training, tests or archive
retrieval. Repeated queries and material/contact events are not independent tasks.
Exact floating-point inequality is reported separately from difference magnitude.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "f0a1632f445415be6c33a7eac31c3a2c974d1f4a"
RESULT = ROOT / "results/terrain_memory/simulation/tm-s1-contact-transition-20261025"
OLD = ROOT / "results/terrain_memory/simulation/tm-s1-control-repair-20261024/formal-v1"


def main():
    episode_files = sorted((RESULT / "formal-v1").glob("*-seed-*.json"))
    episodes = [json.loads(path.read_text()) for path in episode_files]
    old_files = [OLD / f"{terrain}-B1-seed-{seed}.json"
                 for terrain, seed in sorted({(row["terrain"], row["seed"])
                                               for row in episodes})]
    old = {(row["terrain"], row["seed"]): row
           for row in (json.loads(path.read_text()) for path in old_files)}
    csv_names = ("same-start-queries.csv", "grouped-contact-revisits.csv")
    tables = {}
    for name in csv_names:
        with (RESULT / name).open(newline="", encoding="utf-8") as stream:
            tables[name] = list(csv.DictReader(stream))
    inputs = episode_files + old_files + [RESULT / name for name in csv_names]
    manifest = "\n".join(path.relative_to(ROOT).as_posix() + " "
                         + hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(inputs)) + "\n"
    input_sha = hashlib.sha256(manifest.encode()).hexdigest()
    groups = defaultdict(list)
    for row in episodes:
        groups[row["terrain"], row["controller"]].append(row)

    # The matching quantities are physical outcomes, not runtime-dependent NPZ
    # bytes. A terrain-independent pre-contact failure is not three independent
    # pieces of evidence against material-history information.
    physical_keys = ("duration_s", "termination", "final_position_m",
                     "velocity_rmse_m_s", "attitude_rms_rad", "grf_rmse_N",
                     "angular_velocity_rmse_rad_s", "max_roll_pitch_rad",
                     "torque_saturation_fraction", "max_joint_limit_excess_rad",
                     "contact_slip_distance_m", "joint_positive_mechanical_work_J",
                     "signed_robot_contact_work_J", "positive_ground_contact_work_J",
                     "transitions")
    common_failure_seeds = []
    for seed in range(5):
        same_seed = [row for row in episodes if row["controller"] == "T1" and row["seed"] == seed]
        signatures = [{key: row[key] for key in physical_keys} for row in same_seed]
        if (len(same_seed) == 3 and all(not row["success"] for row in same_seed)
                and all(row["patches"] == 0 and row["contact_events"] == 0 for row in same_seed)
                and all(signature == signatures[0] for signature in signatures[1:])):
            common_failure_seeds.append(seed)

    output = []
    for (terrain, controller), group in sorted(groups.items()):
        reference = ({row["seed"]: old[terrain, row["seed"]] for row in group}
                     if controller == "B1" else
                     {row["seed"]: row for row in groups[terrain, "B1"]})
        pairs = [(reference[row["seed"]], row) for row in group
                 if row["success"] and reference[row["seed"]]["success"]]
        complete = [row for row in group if row["success"]]
        material_loaded = [row for row in group
                           if terrain != "rigid" and row["contact_events"] > 0]
        loaded_vz = [abs(row["material_rate_extrema"]["actual_foot_vz_over_H_s_inv"]["foot_vz_m_s"])
                     for row in group
                     if row["material_rate_extrema"].get("actual_foot_vz_over_H_s_inv")]
        queries = [row for row in tables["same-start-queries.csv"]
                   if (row["terrain"], row["controller"]) == (terrain, controller)]
        revisits = [row for row in tables["grouped-contact-revisits.csv"]
                    if (row["terrain"], row["controller"]) == (terrain, controller)]
        cross = [row for row in revisits if row["leg"] != row["previous_leg"]]
        front_hind = [row for row in cross if row["front_to_hind"] == "True"]
        speed_differences = sorted(
            [(abs(float(row["speed_T1_m_s"]) - float(row["speed_T2_m_s"])), row)
             for row in queries], key=lambda item: item[0], reverse=True)
        largest_query = speed_differences[0][1] if speed_differences else None
        row = dict(
            terrain=terrain, controller=controller, episodes=len(group), successes=len(complete),
            termination_counts=json.dumps(dict(sorted(Counter(r["termination"] for r in group).items()))),
            final_phase_counts=json.dumps(dict(sorted(Counter(r["transitions"][-1]["phase"] for r in group).items()))),
            initial_stand_failures=sum(not r["success"] and r["transitions"][-1]["phase"] == "initial_stand" for r in group),
            soft_terrain_failures_before_any_material_patch_load=(
                sum(not r["success"] and r["patches"] == 0 and r["contact_events"] == 0 for r in group)
                if terrain != "rigid" else None),
            soft_terrain_episodes_with_recorded_material_load=len(material_loaded) if terrain != "rigid" else None,
            tm02_material_domain_exit_episodes=(sum(r["material_domain_exit"] is not None for r in group)
                                               if terrain == "tm02_anchor" else None),
            tm02_episodes_with_loaded_vz_record=len(loaded_vz) if terrain == "tm02_anchor" else None,
            tm02_per_episode_peak_abs_loaded_foot_vz_min_m_s=min(loaded_vz) if loaded_vz else None,
            tm02_per_episode_peak_abs_loaded_foot_vz_max_m_s=max(loaded_vz) if loaded_vz else None,
            tm02_per_loaded_episode_max_principal_strain_min=(
                min(r["max_abs_principal_strain"] for r in material_loaded)
                if terrain == "tm02_anchor" and material_loaded else None),
            tm02_per_loaded_episode_max_principal_strain_max=(
                max(r["max_abs_principal_strain"] for r in material_loaded)
                if terrain == "tm02_anchor" and material_loaded else None),
            episodes_without_precontact_query=sum(not r["precontact_query_used"] for r in group),
            matching_precontact_physical_failure_seeds_across_three_terrains=(
                json.dumps(common_failure_seeds) if controller == "T1" else None),
            observed_duration_min_s=min(r["duration_s"] for r in group),
            observed_duration_max_s=max(r["duration_s"] for r in group),
            successful_completion_time_mean_s=mean(r["duration_s"] for r in complete) if complete else None,
            paired_reference="second_round_frozen_B1" if controller == "B1" else "third_round_common_correction_B1",
            paired_reference_successes=sum(r["success"] for r in reference.values()),
            paired_both_completed=len(pairs),
        )
        for metric in ("duration_s", "joint_positive_mechanical_work_J", "velocity_rmse_m_s",
                       "attitude_rms_rad", "contact_slip_distance_m", "grf_rmse_N"):
            row["paired_" + metric + "_mean_difference"] = mean(b[metric] - a[metric] for a, b in pairs) if pairs else None
        row.update(
            episodes_with_front_to_hind_revisit=sum(r["front_to_hind_revisit"] for r in group),
            cross_leg_contact_segment_pairs=len(cross),
            front_to_hind_contact_segment_pairs=len(front_hind),
            front_to_hind_distinct_current_contact_segments=len({
                (r["seed"], r["leg"], r["foot_contact_segment"]) for r in front_hind}),
            cross_leg_minimum_unloaded_gap_s=min(float(r["minimum_unloaded_gap_s"]) for r in cross) if cross else None,
            kv_cross_leg_pre_reload_displacement_max_nm=(
                max(float(r["max_abs_before_reload_e"]) for r in cross) * 1e9
                if terrain.startswith("kv") and cross else None),
            precontact_queries=len(queries),
            exact_unequal_query_initial_states=sum(r["initial_state_differs"] == "True" for r in queries),
            exact_unequal_query_actions=sum(r["action_differs"] == "True" for r in queries),
            nonzero_speed_difference_queries=sum(delta != 0 for delta, _ in speed_differences),
            query_abs_speed_difference_max_m_s=speed_differences[0][0] if speed_differences else None,
            query_abs_speed_difference_second_largest_m_s=speed_differences[1][0] if len(speed_differences) > 1 else None,
            query_abs_compression_difference_max_m=max(abs(float(r["compression_T1_m"]) - float(r["compression_T2_m"])) for r in queries) if queries else None,
            query_abs_surface_difference_max_m=max(abs(float(r["surface_T1_m"]) - float(r["surface_T2_m"])) for r in queries) if queries else None,
            largest_speed_difference_query_seed=int(largest_query["seed"]) if largest_query else None,
            largest_speed_difference_query_id=int(largest_query["query_id"]) if largest_query else None,
            largest_speed_difference_query_time_s=float(largest_query["time_s"]) if largest_query else None,
            source_commit=SOURCE_COMMIT, source_input_files=len(inputs),
            source_input_manifest_sha256=input_sha,
            interpretation=("Repeated queries/contact-segment pairs are not independent missions. Exact unequal floating values do not establish a meaningful decision effect. T2 queries are not executed T2 missions. Paired performance uses only both-completed tasks; success counts retain failures. Rigid-ground revisit flags are geometric events, not material memory. Missing/not-applicable quantities are blank."),
        )
        output.append(row)
    destination = RESULT / "mechanism-summary.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(dict(output=str(destination), groups=len(output), episodes=len(episodes),
                         successes=sum(row["success"] for row in episodes),
                         matched_common_failure_seeds=common_failure_seeds,
                         source_input_manifest_sha256=input_sha,
                         new_simulations=0, new_training_runs=0), indent=2))


if __name__ == "__main__":
    main()
