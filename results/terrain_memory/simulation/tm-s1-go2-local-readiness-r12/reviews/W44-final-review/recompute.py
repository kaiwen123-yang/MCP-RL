#!/usr/bin/env python3
"""Recompute W44 evidence; reads existing records only, never starts simulation.

Run from any directory:
  python recompute.py --raw-dir /path/to/selected-originals

The optional directory contains A/B-trajectory.npz and A/B-planner-calls.jsonl.gz
listed in selected-originals.json. No downloads or server mutations are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent
JOBS = {
    "A": "job-20261110T143358Z-w44-a",
    "B": "job-20261110T143400Z-w44-b",
}


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(paths):
    return [json.loads(line) for p in sorted(paths)
            for line in p.read_text().splitlines() if line.strip()]


def physical_failure(reason):
    return reason in {"body_fall", "material_domain_exit"} or reason.startswith("invalid_contact:")


def repo_summary():
    out = {}
    for label, job in JOBS.items():
        jd = read_json(RESULTS / "jobs" / f"{job}.json")
        files = [p for p in RESULTS.glob(f"train-{job}-PANN_PPO_S5-s*-ep*.json")
                 if not p.name.endswith("-resolved.json")]
        episodes = [read_json(p) for p in sorted(files)]
        reasons = Counter(e["reason"] for e in episodes)
        n = RESULTS / "jobs" / job / "numeric-evidence"
        transitions = read_jsonl((n / "training").glob("*-transitions.jsonl"))
        pann = read_jsonl((n / "training").glob("*-pann-values.jsonl"))
        eval_pann = read_jsonl((n / "evaluation").rglob("*-pann-values.jsonl"))
        evaluation_file = next(RESULTS.glob(f"{job}-eval-PANN_PPO_S5-s*-E30.json"))
        evaluation = read_json(evaluation_file)
        training = jd["training"]
        out[label] = {
            "job": job,
            "training_counts": {k: training[k] for k in (
                "actual_transitions", "updates", "initial_transitions", "new_transitions",
                "initial_updates", "new_updates", "unupdated_fragment")},
            "training_physical_episodes": len(episodes),
            "training_reasons": dict(reasons),
            "training_physical_failures": sum(v for k, v in reasons.items() if physical_failure(k)),
            "training_loaded_episodes": sum(e["material"]["loaded_ever_cells"] > 0 for e in episodes),
            "training_max_x_m": max(e["max_x_m"] for e in episodes),
            "training_simulation_s": sum(e["simulation_s"] for e in episodes),
            "legal_training": {k: sum(e["legal_route"][k] for e in episodes) for k in (
                "legal_outbound_traversals", "legal_return_traversals", "completed_roundtrips", "side_exits")},
            "numeric": {
                "transitions": len(transitions), "unique_serials": len({r["transition_serial"] for r in transitions}),
                "behavior_updates": dict(Counter(r["behavior_updates"] for r in transitions)),
                "pann_rows": len(pann),
                "current_pann_nonzero": sum(any(x != 0 for x in r["pann_recovery_m"]) for r in pann),
                "next_pann_nonzero": sum(any(x != 0 for x in r["next_pann_recovery_m"]) for r in pann),
                "saved_array_matches": sum(bool(r["saved_array_match"]) for r in pann),
                "evaluation_actor_records": len(eval_pann),
                "evaluation_pann_nonzero": sum(any(x != 0 for x in r["pann_recovery_m"]) for r in eval_pann),
            },
            "final_E30": {k: evaluation[k] for k in (
                "attempt", "simulation_s", "reason", "success", "error", "max_x_m",
                "final_qpos", "stage_progress", "legal_route")},
            "source_result_files": [str(p.relative_to(RESULTS)) for p in sorted(files)] + [evaluation_file.name],
        }
    return out


def raw_summary(raw_dir, output):
    manifest = read_json(HERE / "selected-originals.json")
    checks = []
    for item in manifest:
        path = raw_dir / item["local_name"]
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["sha256"] or len(data) != item["size"]:
            raise ValueError(f"Original does not match published digest/size: {path.name}")
        checks.append({"file": path.name, "bytes": len(data), "sha256": digest})
    trajectories = {}
    summary = {"verified_originals": checks, "streams": {}}
    for label in JOBS:
        with np.load(raw_dir / f"{label}-trajectory.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        trajectories[label] = a
        with gzip.open(raw_dir / f"{label}-planner-calls.jsonl.gz", "rt") as f:
            calls = [json.loads(line) for line in f if line.strip()]
        summary["streams"][label] = {
            "trajectory_nodes": len(a["time"]), "arrays": list(a),
            "all_numeric_arrays_finite": all(np.isfinite(x).all() for x in a.values() if x.dtype.kind in "fiu"),
            "final_time_s": float(a["time"][-1]),
            "maximum_body_x_m": float(a["qpos"][:, 0].max()),
            "final_xyz_m": a["qpos"][-1, :3].tolist(),
            "final_rpy_rad": a["rpy"][-1].tolist(),
            "planner_calls": len(calls),
        }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex="col", constrained_layout=True)
    colors = {"A": "#176B87", "B": "#AA3E39"}
    for col, (label, a) in enumerate(trajectories.items()):
        t = a["time"]
        axes[0, col].plot(t, a["qpos"][:, 0], color=colors[label], label="body x")
        axes[0, col].axhline(.8, color="#777777", ls="--", lw=1, label="strip west edge (body only)")
        axes[0, col].set_title(f"W44-{label}: {t[-1]:.3f} s, body fall; no material loading")
        axes[0, col].set_ylabel("Forward position (m)")
        axes[1, col].plot(t, a["qpos"][:, 2], color=colors[label])
        axes[1, col].axhline(.12, color="#B33A3A", ls="--", lw=1, label="fall threshold")
        axes[1, col].set_ylabel("Body height (m)")
        axes[2, col].plot(t, a["rpy"][:, 0], color="#176B87", label="roll")
        axes[2, col].plot(t, a["rpy"][:, 1], color="#C77A20", label="pitch")
        for threshold in (-1.2, 1.2):
            axes[2, col].axhline(threshold, color="#B33A3A", ls="--", lw=1)
        axes[2, col].set_ylabel("Orientation (rad)")
        axes[2, col].set_xlabel("Simulation time (s)")
        for ax in axes[:, col]:
            ax.axvspan(0, 2, color="#87939E", alpha=.12)
            ax.grid(alpha=.2)
            ax.set_xlim(0, 13)
        axes[0, col].legend(fontsize=8, loc="upper left")
        axes[1, col].legend(fontsize=8, loc="lower left")
        axes[2, col].legend(fontsize=8, loc="upper left")
    fig.suptitle("Final deterministic policies, update 57: forward motion returned, stable traversal did not", fontsize=13)
    fig.savefig(output / "final-evaluations.png", dpi=170)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"source_snapshot": "69d5a6ff2aa9803fddc8ac3a78a906008ea526c6",
               "new_physics": False, "streams": repo_summary()}
    if args.raw_dir:
        summary["selected_originals"] = raw_summary(args.raw_dir, args.output_dir)
    (args.output_dir / "recomputed-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("training_physical_episodes", "training_physical_failures", "training_loaded_episodes", "numeric")} for k, v in summary["streams"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
