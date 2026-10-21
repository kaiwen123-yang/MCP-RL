"""Full 27 m missions with frozen checkpoints; isolate prediction and value.

The existing learned-response/learned-value complete runs are the baseline.
This script executes the remaining three cells of the 2 x 2 comparison.
No follower output is used as a fallback. All cells run CEM at every decision.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from go2_learning.runner import collect_phase, hashes, identity, load_state, make_system, write_json


def execute(payload):
    config, assets, checkpoint, output, name, options = payload
    config = copy.deepcopy(config)
    config["evaluation_cases"] = ["long_continue_first", "long_stop_first"]
    config["runtime"] = {"device": "cpu", "torch_threads": 2, "inference_threads": 2,
                         "collection_workers": 1, "controller_mode": "mpc", "planner_options": options}
    root = Path(output) / name
    root.mkdir(parents=True, exist_ok=False)
    system = make_system(config, assets, 11)
    load_state(system, torch.load(checkpoint, map_location="cpu", weights_only=True))
    before = hashes(system)
    started = perf_counter()
    # These are cases 2 and 3 of the original four-case evaluation. Keep their
    # original CEM streams (22011, 23011), despite selecting only the long pair.
    result = collect_phase(system, config, assets, root / "evaluation", 22011, "evaluation")
    record = {"name": name, "options": options, "status": "COMPLETED",
              "parameter_hashes": before, "parameters_unchanged": before == hashes(system),
              "episodes": result["episodes"], "stats": result["stats"],
              "wall_seconds": perf_counter() - started}
    write_json(root / "result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.original_run / "config.json").read_text())
    variants = [("learned_no_value", {"response_mode": "learned", "use_terminal_value": False}),
                ("prior_value", {"response_mode": "prior", "use_terminal_value": True}),
                ("prior_no_value", {"response_mode": "prior", "use_terminal_value": False})]
    checkpoint = args.original_run / "frozen/round1/checkpoint.pt"
    write_json(args.output / "protocol.json", {"task_config": config, "variants": variants,
               "checkpoint_arm": "frozen", "cem_seeds": [22011, 23011],
               "source_sha256": identity(), "optimization_updates": 0})
    started = perf_counter()
    jobs = [(config, args.assets, checkpoint, args.output, name, options) for name, options in variants]
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        records = list(pool.map(execute, jobs))
    write_json(args.output / "comparison.json", {"status": "COMPLETED", "records": records,
               "wall_seconds": perf_counter() - started, "optimization_updates": 0})
    for record in records:
        print(json.dumps({"variant": record["name"], "episodes": [
            {k: ep[k] for k in ("case", "reason", "elapsed_seconds")} for ep in record["episodes"]]}), flush=True)


if __name__ == "__main__":
    main()
