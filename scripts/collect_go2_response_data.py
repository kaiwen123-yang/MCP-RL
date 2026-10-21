"""Fixed nominal-scene excitation data for response identification only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--task-config", default="configs/go2_frozen_tasks.json", type=Path)
    parser.add_argument("--fit-config", default="configs/go2_compact_fit.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.task_config.read_text(encoding="utf-8"))
    protocol = json.loads(args.fit_config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.nice(protocol["resource_budget"]["nice"])
    os.sched_setaffinity(0, set(protocol["resource_budget"]["cpu_affinity"]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
    import numpy as np
    import torch
    from go2_frozen.simulation import Go2Simulation
    from go2_frozen.task import nominal_command, advance_stage, is_fallen
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    policy_path = args.asset_root/"himloco/himloco.pt"
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    sim = Go2Simulation(args.asset_root/"go2/scene_go2.xml", policy_path,
                        args.asset_root/"himloco/config.yaml", physics_dt=config["physics_dt"], obstacles=config["obstacles"])
    start, cpu_start = time.perf_counter(), time.process_time()
    record = {"status": "RUNNING", "episodes": [], "macro_dt": config["macro_dt"],
              "policy_sha256_before": policy_sha, "task_config_sha256": hashlib.sha256(args.task_config.read_bytes()).hexdigest(),
              "fit_config_sha256": hashlib.sha256(args.fit_config.read_bytes()).hexdigest(),
              "fit_protocol": protocol, "policy_updates": 0, "response_model_updates": 0,
              "future_force_available": False, "scene_changed": False,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "nice": os.getpriority(os.PRIO_PROCESS, 0),
              "state_semantics": {"history": "newest first; actual previous policy inputs, not future state",
                                  "sensors": "current body gyro*.25,gravity,q,dq*.05,previous action",
                                  "context": "body vx,body vy,world wz,roll,pitch,height",
                                  "pose": "world x,y,yaw",
                                  "labels": "actual endpoints; no future target or task cost used as labels"}}
    def save():
        (args.output/"dataset.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    save()
    def pack(state):
        return {"history": np.asarray(state["history_frames"], dtype=np.float32),
                "sensors": np.concatenate([np.asarray(state["angular_velocity_body"])*.25,
                                            state["projected_gravity"], state["joint_position"],
                                            np.asarray(state["joint_velocity"])*.05, state["action"]]).astype(np.float32),
                "context": np.asarray([*state["linear_velocity_body"][:2], state["angular_velocity_world"][2],
                                       *state["rpy"][:2], state["position"][2]], dtype=np.float32),
                "pose": np.asarray([*state["position"][:2], state["yaw"]], dtype=np.float32)}
    for split, seeds in [("train", protocol["collection"]["train_seeds"]), ("validation", protocol["collection"]["validation_seeds"])]:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            before_counts = sim.counters.copy()
            state = sim.reset(settle_seconds=config["settle_seconds"])
            states, commands, costs, contacts = [pack(state)], [], [], []
            stage, noise, fallen = 0, np.zeros(3), False
            rho = protocol["collection"]["noise_ar_coefficient"]
            sigma = np.asarray(protocol["collection"]["command_noise_std"])
            for _ in range(round(protocol["collection"]["seconds_per_episode"]/config["macro_dt"])):
                stage = advance_stage(state, stage, config)
                noise = rho*noise+np.sqrt(1-rho*rho)*sigma*rng.normal(size=3)
                command = np.clip(nominal_command(state, stage, config)+noise, config["command_lower"], config["command_upper"])
                step = sim.step_command(command, config["macro_dt"], trace=False)
                state = step["state"]
                states.append(pack(state)); commands.append(command)
                costs.append([step["torque_squared_integral"], step["absolute_work"]])
                contacts.append([step["obstacle_contacts"], step["nonfoot_ground_contacts"]])
                fallen = is_fallen(state, config)
                if fallen:
                    break
            arrays = {key: np.stack([s[key] for s in states]) for key in states[0]}
            arrays.update(commands=np.asarray(commands, dtype=np.float32), resources=np.asarray(costs), contacts=np.asarray(contacts))
            filename = f"episode-{seed}.npz"
            np.savez_compressed(args.output/filename, **arrays)
            episode = {"file": filename, "seed": seed, "split": split, "transitions": len(commands),
                       "seconds": len(commands)*config["macro_dt"], "fallen": fallen,
                       "obstacle_contact_steps": int(np.asarray(contacts)[:,0].sum()),
                       "nonfoot_ground_contact_steps": int(np.asarray(contacts)[:,1].sum()),
                       "physics_steps_including_settle": sim.counters["physics_steps"]-before_counts["physics_steps"],
                       "policy_calls_including_settle": sim.counters["policy_inferences"]-before_counts["policy_inferences"],
                       "sha256": hashlib.sha256((args.output/filename).read_bytes()).hexdigest()}
            record["episodes"].append(episode); save()
            print(json.dumps(episode), flush=True)
    record.update(status="COMPLETED", wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu_start,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  policy_sha256_after=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                  actual_transitions=sum(e["transitions"] for e in record["episodes"]))
    save()


if __name__ == "__main__":
    main()
