#!/usr/bin/env python3
"""Locate S1 failures within the already executed complete-mission protocol.

Reads the 85 formal-v1 episode JSON files. Produces one group table only.
No physics, training, archive download or mutation of original results occurs.

A positive recorded material indentation marks actual soft-material loading.
Rigid-ground episodes have no soft-material exposure denominator: their loading
and pre-material-failure cells are deliberately blank (not zero). Phase entry
does not mean phase completion. Reaching 'return' is not completing the mission.

The input manifest digest hashes sorted relative JSON paths and their SHA256
digests. The source commit is provenance for this frozen input campaign.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HEAD = "1bfcf244bf5787cf0086f3f1846c19c42576f9d7"
DEFAULT_RESULT = ROOT / "results/terrain_memory/simulation/tm-s1-go2-contact-20261024"
TERRAINS = ("rigid", "tm02_anchor", "kv_soft", "kv_moderate")
CONTROLLERS = ("C0", "C1", "C2", "C3", "C4")


def analyze(directory):
    records = []
    manifest = []
    for path in sorted((directory / "formal-v1").glob("*-seed-*.json")):
        raw = path.read_bytes()
        records.append(json.loads(raw))
        relative = path.relative_to(directory).as_posix()
        manifest.append(relative + " " + hashlib.sha256(raw).hexdigest())
    digest = hashlib.sha256(("\n".join(manifest) + "\n").encode()).hexdigest()

    groups = defaultdict(list)
    for record in records:
        groups[record["terrain"], record["controller"]].append(record)

    summaries = []
    for terrain in TERRAINS:
        for controller in CONTROLLERS:
            group = groups.get((terrain, controller), [])
            if not group:
                continue
            soft = terrain != "rigid"
            no_intent = [r for r in group
                         if r["termination"] == "controller_no_feasible_intent"]
            summaries.append({
                "terrain": terrain,
                "controller": controller,
                "episodes": len(group),
                "entered_turn": sum(any(t["phase"] == "turn" for t in r["transitions"])
                                    for r in group),
                "entered_return": sum(any(t["phase"] == "return" for t in r["transitions"])
                                      for r in group),
                "entered_terminal_stand": sum(
                    any(t["phase"] == "terminal_stand" for t in r["transitions"])
                    for r in group),
                "completed_missions": sum(r["success"] for r in group),
                "material_loading_scope": (
                    "positive recorded material indentation in this soft-terrain episode"
                    if soft else "N/A: rigid ground has no soft-material loading"),
                "material_loaded_episodes": (
                    sum(r["max_indentation_mm"] > 0 for r in group) if soft else None),
                "no_feasible_intent_before_material_loading": (
                    sum(r["max_indentation_mm"] == 0 for r in no_intent) if soft else None),
                "no_feasible_intent_episodes": len(no_intent),
                "material_domain_exit_episodes": sum(
                    r["material_domain_exit"] is not None for r in group),
                "last_recorded_phase_counts": json.dumps(dict(sorted(Counter(
                    r["transitions"][-1]["phase"] for r in group).items())), sort_keys=True),
                "source_commit": SOURCE_HEAD,
                "source_episode_count": len(records),
                "source_episode_manifest_sha256": digest,
                "interpretation": (
                    "phase entry is not completion; material loading uses actual "
                    "positive indentation, not the planner's future prediction"),
            })

    destination = directory / "failure-stage-summary.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    # Exact equality of saved numeric values; no rounded-position matching.
    shared = defaultdict(list)
    for record in records:
        if (record["termination"] == "controller_no_feasible_intent"
                and (record["terrain"] == "rigid"
                     or record["max_indentation_mm"] == 0)):
            key = (record["seed"], record["duration_s"],
                   tuple(record["final_position_m"]))
            shared[key].append({"terrain": record["terrain"],
                                "controller": record["controller"]})
    identical = [
        {"seed": key[0], "termination_time_s": key[1],
         "final_position_m": list(key[2]), "episodes": values}
        for key, values in sorted(shared.items())
        if len({row["terrain"] for row in values}) > 1
    ]
    totals = {
        key: sum(row[key] for row in summaries)
        for key in ("episodes", "entered_turn", "entered_return",
                    "entered_terminal_stand", "completed_missions",
                    "no_feasible_intent_episodes", "material_domain_exit_episodes")
    }
    totals["soft_material_loaded_episodes"] = sum(
        row["material_loaded_episodes"] for row in summaries
        if row["material_loaded_episodes"] is not None)
    totals["soft_terrain_no_intent_before_loading"] = sum(
        row["no_feasible_intent_before_material_loading"] for row in summaries
        if row["no_feasible_intent_before_material_loading"] is not None)
    print(json.dumps({
        "output": str(destination),
        "group_rows": len(summaries),
        "source_commit": SOURCE_HEAD,
        "source_episode_manifest_sha256": digest,
        "totals": totals,
        "exact_shared_failure_signatures": identical,
        "new_physics_predictions": 0,
        "new_training_runs": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    analyze(parser.parse_args().result_dir)

