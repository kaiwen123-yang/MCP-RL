"""Re-evaluate the identical saved controllers after a shared planner correction.

No data collection for training, policy update, value update, or model fitting
is performed. Keep this evaluation distinct from the original learning run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2_learning.runner import identity, resume, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.original_run / "comparison.json").read_text())
    if original["status"] != "COMPLETED":
        raise ValueError("Preserve and finish the original fixed learning run first")
    config = original["config"]
    config["runtime"]["device"] = "cpu"
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    result = {"status": "RUNNING", "seed": original["seed"], "config": config,
              "actor_updates": 0, "model_updates": 0, "value_updates": 0,
              "change": "unfulfilled mandatory stops block route credit; same checkpoints and tasks",
              "training_source": "original fixed run; no training with the corrected planning objective",
              "source_sha256": identity(), "arms": {}}
    for arm in config["arms"]:
        print(f"Evaluating unchanged {arm} checkpoint on four complete missions", flush=True)
        checkpoint = args.original_run / arm / "round1/checkpoint.pt"
        resume(config, args.assets, root / arm, checkpoint, original["seed"])
        evaluation = json.loads((root / arm / "result.json").read_text())
        result["arms"][arm] = {key: evaluation[key] for key in
                              ("episodes", "stats", "parameters_unchanged", "wall_seconds")}
        result["wall_seconds"] = perf_counter() - started
        write_json(root / "comparison.json", result)
        print(json.dumps({"arm": arm, "episodes": [
            {key: e[key] for key in ("case", "reason", "completion_seconds", "elapsed_seconds")}
            for e in evaluation["episodes"]]}, ensure_ascii=False), flush=True)
    result["follower"] = original["follower"]
    result["follower_reused"] = "The corrected mission cost is never called by the geometric follower"
    result["status"] = "COMPLETED"
    result["wall_seconds"] = perf_counter() - started
    write_json(root / "comparison.json", result)
    print("COMPLETED, no additional optimization", flush=True)


if __name__ == "__main__":
    main()
