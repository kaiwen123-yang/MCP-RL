"""One fixed-budget paired-task learning comparison plus held-out long missions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from go2_learning.runner import (collect_phase, fit_phase, hashes, identity,
                                 learn_round, load_state, make_system, read_config,
                                 state_dict, write_json)


def import_initial_weights(system, checkpoint):
    """Keep the public feedback and old factual models; add eight zero columns.

    New task value starts from zero. Optimizers start fresh and are shared by
    copying the resulting common initialization, not by sharing Adam tensors.
    The old 19 context columns and command columns keep their original roles.
    """
    old = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name in ("actor", "dynamics", "response"):
        target = system[name].state_dict()
        for key, value in old[name].items():
            if target[key].shape == value.shape:
                target[key] = value
            elif name == "actor" and key == "actor.0.weight":
                target[key].zero_()
                target[key][:, :value.shape[1]] = value
            elif name == "response" and key == "network.0.weight":
                target[key].zero_()
                target[key][:, :67] = value[:, :67]
                target[key][:, -3:] = value[:, -3:]
            else:
                raise ValueError(f"Unexpected checkpoint shape: {name}/{key}")
        system[name].load_state_dict(target)


def compact_episodes(episodes):
    return [{key: episode[key] for key in
             ("case", "success", "reason", "completion_seconds", "elapsed_seconds",
              "mechanical_work_proxy_J", "path_rms_m", "stage_exits")}
            for episode in episodes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/go2_continuation.json"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    config = read_config(args.config)
    config["runtime"] = {"device": args.device, "collection_workers": args.workers,
                         "torch_threads": 2, "inference_threads": 2}
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    seed = config["seeds"][0]
    write_json(root / "config.json", config)
    source = identity()
    for p in (args.config, Path(__file__)):
        source[p.name] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    write_json(root / "source.json", source)
    started = perf_counter()
    common = make_system(config, args.assets, seed)
    import_initial_weights(common, args.checkpoint)
    print("Common initialization: complete paired missions with public feedback", flush=True)
    calibration = collect_phase(common, config, args.assets, root / "common/calibration",
                                seed + 1, "calibration")
    zeros = torch.zeros_like(calibration["batch"]["raw_action"])
    common_value = common["updater"].update(calibration["batch"], zeros, "frozen")
    common_facts = fit_phase(common, config, calibration, calibration, seed + 2)
    initial = state_dict(common)
    torch.save(initial, root / "common/initialization.pt")
    write_json(root / "common/fit.json", {"value": common_value, "facts": common_facts})
    results = {"status": "RUNNING", "seed": seed, "config": config,
               "common_calibration": calibration["stats"], "arms": {}}
    for arm in config["arms"]:
        print(f"Starting {arm}: fixed 5250-step batch, complete calibration and four full evaluations", flush=True)
        system = make_system(config, args.assets, seed)
        load_state(system, initial)
        initial_hashes = hashes(system)
        arm_start = perf_counter()
        record = learn_round(system, config, args.assets, root / arm / "round1", seed, 0, arm)
        record["wall_seconds"] = perf_counter() - arm_start
        record["initial_parameter_hashes"] = initial_hashes
        results["arms"][arm] = record
        results["wall_seconds"] = perf_counter() - started
        write_json(root / "comparison.json", results)
        print(json.dumps({"arm": arm, "episodes": compact_episodes(record["evaluation_episodes"]),
                          "actor_updates": record["ppo"]["actor_updates"],
                          "wall_seconds": record["wall_seconds"]}, ensure_ascii=False), flush=True)
    print("Evaluating the unchanged public feedback with the geometric follower", flush=True)
    reference = make_system(config, args.assets, seed)
    load_state(reference, initial)
    reference_config = copy.deepcopy(config)
    reference_config["runtime"]["controller_mode"] = "follower"
    follower = collect_phase(reference, reference_config, args.assets,
                             root / "follower/evaluation", seed + 20000, "evaluation")
    results["follower"] = {"stats": follower["stats"], "episodes": follower["episodes"]}
    results["status"] = "COMPLETED"
    results["wall_seconds"] = perf_counter() - started
    results["source_sha256"] = source
    write_json(root / "comparison.json", results)
    print(json.dumps({"status": "COMPLETED", "wall_seconds": results["wall_seconds"],
                      "follower": compact_episodes(follower["episodes"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
