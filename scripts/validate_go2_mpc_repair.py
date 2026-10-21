"""Evaluate repaired MPC on the saved checkpoints, with no training.

Both fixed 27 m tasks run from start to success or the original 120 s deadline.
Default: kinematic response prior without an uncalibrated terminal critic.
Use --response-mode local_response for the measured local learned response.
Only the deployed prediction closure changes; no optimizer is called.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

from run_go2_mpc_attribution import execute
from go2_learning.runner import identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-mode", choices=("prior", "local_response"), default="prior")
    parser.add_argument("--arms", nargs="+", choices=("frozen", "ordinary", "pathwise"),
                        default=["frozen", "ordinary", "pathwise"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.original_run / "config.json").read_text())
    options = {"response_mode": args.response_mode, "use_terminal_value": False}
    write_json(args.output / "protocol.json", {"task_config": config, "options": options,
               "arms": args.arms, "cem_seeds": [22011, 23011],
               "source_sha256": identity(), "optimization_updates": 0,
               "observer": ("measured velocity residual, filtered by existing response time constants"
                            if args.response_mode == "local_response" else "none"),
               "controller": "CEM at every decision; no switch to follower"})
    jobs = [(config, args.assets, args.original_run / arm / "round1/checkpoint.pt",
             args.output, arm, options) for arm in args.arms]
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        records = list(pool.map(execute, jobs))
    write_json(args.output / "comparison.json", {"status": "COMPLETED", "records": records,
               "wall_seconds": perf_counter() - started, "optimization_updates": 0})
    for record in records:
        print(json.dumps({"arm": record["name"], "episodes": [
            {k: ep[k] for k in ("case", "reason", "elapsed_seconds")} for ep in record["episodes"]],
            "mpc_calls": record["stats"]["planner_calls"],
            "non_follower_first_commands": record["stats"]["mpc_non_follower_first_commands"]}), flush=True)


if __name__ == "__main__":
    main()
