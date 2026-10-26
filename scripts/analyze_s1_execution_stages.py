#!/usr/bin/env python3
"""Summarize existing T1R development stages; no simulation or new trials."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/terrain_memory/simulation/tm-s1-contact-execution-20261026"
SOURCE_COMMIT = "56fabe0024108fe933dd8207c3a97331b8fa6102"
SEEDS = (4, 3, 0)


def main():
    episode_paths = [RESULT / f"development-v1-seed{s}/rigid-T1R-seed-{s}.json"
                     for s in SEEDS]
    control_path = RESULT / "control-time-series.csv"
    event_path = RESULT / "contact-events.csv"
    inputs = episode_paths + [control_path, event_path]
    manifest = "\n".join(p.relative_to(ROOT).as_posix() + " "
                         + hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(inputs)) + "\n"
    manifest_sha = hashlib.sha256(manifest.encode()).hexdigest()
    with control_path.open(newline="", encoding="utf-8") as stream:
        controls = list(csv.DictReader(stream))
    with event_path.open(newline="", encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    output = []
    for path in episode_paths:
        episode = json.loads(path.read_text())
        seed = episode["seed"]
        rows = [r for r in controls if int(r["seed"]) == seed]
        outbound = next(e for e in episode["transitions"] if e["phase"] == "outbound")
        start = outbound["time_s"]
        at_start = next(r for r in rows if float(r["time_s"]) == start)
        stand = [r for r in rows if 0 < float(r["time_s"]) < start]
        offset = float(rows[0]["control_simulator_time_s"]) - float(rows[0]["time_s"])
        # Event times use simulator time; remove its 0.2 ms origin. Rounding to
        # nanoseconds only aligns floating-point boundary labels, not physics.
        lost = [e for e in events if int(e["seed"]) == seed
                and e["event"] == "support_lost_force_invalidated"
                and 0 < round(float(e["time_s"]) - offset, 9) < start]
        output.append(dict(
            seed=seed, terrain=episode["terrain"], result_kind="development",
            success=episode["success"], termination=episode["termination"],
            duration_s=episode["duration_s"],
            final_x_m=episode["final_position_m"][0], final_y_m=episode["final_position_m"][1],
            outbound_clock_transition_s=start,
            base_z_at_outbound_start_m=float(at_start["base_z_m"]),
            roll_at_outbound_start_rad=float(at_start["roll_rad"]),
            pitch_at_outbound_start_rad=float(at_start["pitch_rad"]),
            support_legs_at_outbound_start=int(at_start["actual_support_legs"]),
            commanded_Fz_at_outbound_start_N=float(at_start["commanded_Fz_N"]),
            actual_Fz_at_control_outbound_start_N=float(at_start["actual_Fz_at_control_N"]),
            initial_stand_control_samples_excluding_t0=len(stand),
            initial_stand_commanded_Fz_sample_mean_N=mean(float(r["commanded_Fz_N"]) for r in stand),
            initial_stand_actual_Fz_at_control_sample_mean_N=mean(float(r["actual_Fz_at_control_N"]) for r in stand),
            initial_stand_support_lost_events=len(lost),
            torque_saturation_fraction=episode["torque_saturation_fraction"],
            tm02_mission_executed=episode["terrain"] == "tm02_anchor",
            material_memory_evaluated=False, event_clock_origin_offset_s=offset,
            source_commit=SOURCE_COMMIT, source_input_files=len(inputs),
            source_input_paths=";".join(p.relative_to(ROOT).as_posix() for p in sorted(inputs)),
            source_input_manifest_sha256=manifest_sha,
            interpretation="All records are failed rigid development missions, not formal trials. "
                "The 2 s outbound transition is a clock event, not evidence of stable standing. "
                "Force means are samples, not physical impulse integrals. actual_Fz_at_control "
                "retains the preceding physics-end measurement; the separate original post-step "
                "GRF is about 2 ms after the command and is not used here. Event counts use "
                "simulator time minus its 0.2 ms origin and do not establish instantaneous causality. "
                "Global means do not rule out local instability or establish its unique cause."
        ))
    with (RESULT / "execution-stage-summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"rows": len(output), "source_input_files": len(inputs),
                      "source_input_manifest_sha256": manifest_sha}))


if __name__ == "__main__":
    main()
