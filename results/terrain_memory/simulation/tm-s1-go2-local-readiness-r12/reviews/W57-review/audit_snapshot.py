"""Read-only aggregation of a published W57 snapshot; never launches physics.

Run from the repository root:
python results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W57-review/audit_snapshot.py
The output is a review artifact, not an execution receipt or new qualification.
"""
from collections import Counter
import json
from pathlib import Path
import subprocess


def main():
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    rel = Path("results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
    source = repo / rel / "w57"
    output = Path(__file__).resolve().parent
    idx = json.loads((source / "index.json").read_text())
    updates = {}
    for head in ("full", "blind"):
        rows = [json.loads(p.read_text()) for p in sorted((source / "r/pilot02" / head).glob("update-*.json"))]
        keys = ("distinct", "effective_liftoff", "only_ramp", "unexecuted", "optimizer_steps", "optimization_wall_s")
        updates[head] = {k: sum(r[k] for r in rows) for k in keys}
        updates[head]["updates"] = len(rows)
        updates[head]["effective_liftoff_fraction"] = updates[head]["effective_liftoff"] / updates[head]["distinct"]
    node = json.loads((source / "r/pilot02/full/node-08192.json").read_text())
    paired = json.loads((source / "p/physical/w57-m1-p-a-original-e30-a01-paired-B.json").read_text())
    history = json.loads((source / "p/physical/w57-m1-p-a-original-e30-a01-all-protected.json").read_text())
    census = json.loads((source / "p/full-force-off-summary.json").read_text())
    parsed_count = 0
    parsed_bytes = 0
    # Complete JSON decoding is a syntax/data coverage count, not semantic reading.
    for path in source.rglob("*.json"):
        data = path.read_bytes()
        json.loads(data)
        parsed_count += 1
        parsed_bytes += len(data)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).split(b"\0")
    files = [repo / p.decode() for p in tracked if p]
    tracked_bytes = sum(p.stat().st_size for p in files if p.is_file())
    measurement = idx["measurement"]
    result = {
        "schema": "s1_w57_readonly_review_v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "execution_snapshot_shanghai": idx["as_of_shanghai"],
        "execution_status": idx["status"],
        "review_new_physics": 0,
        "review_new_PPO_updates": 0,
        "attempt_status_counts": dict(Counter(x["status"] for x in idx["attempts"])),
        "task_mode_counts": dict(Counter(x["task_mode"] for x in idx["attempts"])),
        "distinct_physics_attempts": idx["distinct_physics_attempts"],
        "completed_physics_samples": idx["completed_physics_samples"],
        "completed_process_cpu_s": idx["completed_process_cpu_s"],
        "original_qualifications": idx["original_qualifications"],
        "observed_on_policy": idx["observed_on_policy"],
        "committed_distinct": sum(x["distinct"] for x in updates.values()),
        "observed_distinct_including_partial": sum(x["total_distinct"] for x in idx["observed_on_policy"].values()),
        "updates_independent_sum": updates,
        "full_node_reward_rate": {k: node[k] for k in ("episode_reward_per_s", "first_vs_second_half_return_rate", "within_case_episode_standard_error")},
        "paired_P_minus_B": paired["paired_differences"],
        "protected_private_history": history,
        "full_E600_force_off_census": census,
        "measurement": measurement,
        "simultaneous_advancement_fraction": measurement["concurrently_advancing_interval_wall_s"] / measurement["covered_wall_s"],
        "release_index": {k: idx[k] for k in ("verified_release_assets", "release_bytes")},
        "coverage": {
            "tracked_paths_inventory": len(files),
            "tracked_regular_file_bytes": tracked_bytes,
            "w57_json_fully_decoded": parsed_count,
            "w57_json_bytes_fully_decoded": parsed_bytes,
            "semantic_scope": "Current architecture, W57 material/planner/learning/dispatcher sources, historical review/plan/memory, all committed update receipts, event census and new P/A/E30 raw verification; see companion evidence report.",
            "limits": "Inventory/JSON parsing is not a claim to understand every repository byte. Old four-E600 census is a published full-raw analysis, not four newly downloaded raw archives. New P/A/E30 raw arrays were independently recomputed. Videos are timecoded samples, not every frame; no final8192/blind frozen evaluation video existed in this snapshot.",
        },
    }
    (output / "snapshot-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("source_commit", "execution_snapshot_shanghai", "attempt_status_counts", "committed_distinct", "observed_distinct_including_partial", "coverage")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
