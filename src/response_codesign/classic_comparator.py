"""Predeclared analytic-feedback comparator on the *same* pilot interfaces.

The planner, physical integration, tasks, costs and screening are imported from
the frozen pilot; they are not reimplemented here. No neural network is trained.
Four feedback gains are declared before evaluating any of them. The selected
pair is frozen using validation, then tested on the pilot's paired test seeds.

This asks whether the known low-dimensional plant needs free neural feedback.
It is not a model-free sample-efficiency comparison: analytic feedback uses
known nominal physics structurally, while RL receives the same teacher/nominal
information without retaining that analytic feedback law at deployment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .dynamics import PhysicalConfig, response_design, teacher_action
from .learning import UpdateConfig, execution_features
from .pilot import average_metrics, identify, rollout, select_feasible


ROOT = Path(__file__).resolve().parents[2]
GAIN_CANDIDATES = (
    {"name": "shared_teacher", "kp": 4.0, "kv": 3.0, "ka": 8.0, "rho": 1.0},
    {"name": "slower", "kp": 3.0, "kv": 3.0, "ka": 7.0, "rho": 1.0},
    {"name": "faster", "kp": 5.0, "kv": 3.5, "ka": 9.0, "rho": 1.0},
    {"name": "more_damped", "kp": 4.0, "kv": 4.0, "ka": 8.0, "rho": 1.0},
)


def response_initialization(gains):
    kappa = gains["ka"]+gains["kv"]
    omega = np.sqrt(gains["ka"]*gains["kp"]/kappa)
    zeta = (gains["kp"]+gains["ka"]*gains["kv"]+gains["rho"])/(2*kappa*omega)
    physical = torch.tensor([omega, zeta, kappa, 0.02], dtype=torch.float32)
    low = torch.tensor([1.2, 0.7, 3.0, 0.0])
    span = torch.tensor([1.8, 0.7, 11.0, 0.5])
    probability = (physical-low)/span
    assert ((probability > 0) & (probability < 1)).all()
    return torch.logit(probability)


class ClassicActor(nn.Module):
    """Full-action saturated feedback, with no trainable network or residual.

    The input is exactly the pilot's twelve-feature vector. It reconstructs
    only observed p/v/a and command c; no disturbance or future target is used.
    Returning atanh allows the unchanged rollout's U*tanh transform to execute
    this controller. Endpoint clipping differs from exact +/-U by <=1.2e-7.
    """
    def __init__(self, gains):
        super().__init__()
        self.gains = dict(gains)
        self.cfg = PhysicalConfig()

    def forward(self, features):
        p, v, a, c = (features[..., 0]*1.5, features[..., 1]*2,
                       features[..., 2]*3, features[..., 3])
        kp, kv, ka, rho = (self.gains[name] for name in ("kp", "kv", "ka", "rho"))
        gravity_drag = self.cfg.k*torch.sin(p)+self.cfg.d*v+self.cfg.drag*v*v.abs()
        acceleration = a-gravity_drag
        desired_actuator = gravity_drag-kp*(p-c)-kv*v
        desired_derivative = (self.cfg.k*torch.cos(p)*v
                              +(self.cfg.d+2*self.cfg.drag*v.abs())*acceleration
                              -kp*v-kv*acceleration)
        physical_input = a+self.cfg.tau*(desired_derivative-ka*(a-desired_actuator)-rho*v)
        unit_input = (physical_input/self.cfg.U).clamp(-1+1e-7, 1-1e-7)
        return torch.atanh(unit_input)


def teacher_identity_check():
    """Check the shared candidate against the pre-existing teacher, not a task."""
    generator = torch.Generator().manual_seed(9183)
    x = (torch.rand(256, 3, generator=generator)-0.5)*torch.tensor([0.8, 0.8, 1.0])
    c = (torch.rand(256, 1, generator=generator)-0.5)*0.8
    reference = torch.cat((c, torch.zeros(256, 2)), -1)
    cfg = PhysicalConfig()
    expected = teacher_action(x, reference, c, torch.zeros(256, 4), cfg)
    actual = cfg.U*torch.tanh(ClassicActor(GAIN_CANDIDATES[0])(execution_features(x, c)))
    error = float((actual-expected).abs().max())
    assert error <= 2e-6
    return error


def run_seed(seed, output, test_batches):
    cfg = UpdateConfig()
    candidates, candidate_records, validations = [], [], []
    for index, gains in enumerate(GAIN_CANDIDATES):
        learner = SimpleNamespace(actor=ClassicActor(gains),
                                  mean=response_initialization(gains), config=cfg)
        initial_mean = learner.mean.clone()
        calibration = [rollout(learner, 200000+seed*1000+trial, deterministic=True,
                               random_commands=True) for trial in range(5)]
        segments = [segment for item in calibration for segment in item["segments"]]
        started = time.perf_counter()
        fitted, id_info = identify(segments, learner.mean, updates=100)
        id_info["wall_seconds"] = time.perf_counter()-started
        id_info["model_response_transitions"] = 100*64*10
        learner.mean = fitted
        validation = rollout(learner, 300000+seed, deterministic=True)
        validations.append(validation)
        candidates.append(learner)
        record = {"candidate_index": index, "gains": gains,
                  "initial_mean": initial_mean.tolist(),
                  "initial_response": response_design(initial_mean).tolist(),
                  "fitted_mean": fitted.tolist(),
                  "fitted_response": response_design(fitted).tolist(),
                  "calibration": average_metrics(calibration), "identification": id_info,
                  "validation": validation["metrics"],
                  "actor_training_updates": 0, "actor_parameter_count": 0,
                  "feedback_gain_count": 4}
        candidate_records.append(record)
        folder = output/f"seed-{seed}"/f"candidate-{index}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder/"result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({"phase": "classic-validation", "seed": seed,
                          "candidate": index, "gains": gains,
                          **validation["metrics"]}), flush=True)

    selected, screening = select_feasible(validations, cfg)
    learner = candidates[selected]
    evaluations = [rollout(learner, 500000+seed*1000+trial,
                           deterministic=True, trace=trial == 0)
                   for trial in range(test_batches)]
    # Selection was frozen before observing any test result.
    evaluation = average_metrics(evaluations)
    calibration_totals = average_metrics([{"metrics": r["calibration"]} for r in candidate_records])
    validation_totals = average_metrics(validations)
    stages = [calibration_totals, validation_totals, evaluation]
    count_keys = ("physical_steps", "planner_batch_calls", "planner_instance_calls",
                  "candidate_sequences", "objective_calls", "response_transitions")
    budget = {key: sum(stage[key] for stage in stages) for key in count_keys}
    budget.update(actor_training_updates=0, response_identification_updates=4*100,
                  identification_model_response_transitions=4*100*64*10,
                  rollout_wall_seconds=sum(stage["wall_seconds"] for stage in stages),
                  identification_wall_seconds=sum(r["identification"]["wall_seconds"] for r in candidate_records))
    result = {"seed": seed, "selected_candidate": selected, "selected_gains": GAIN_CANDIDATES[selected],
              "selection": screening, "candidates": candidate_records,
              "evaluation": evaluation, "budget": budget,
              "scope": "Four predeclared classical gains; validation-selected pair, not continuous optimum"}
    folder = output/f"seed-{seed}"
    np.savez(folder/"trace.npz", **{name: value.numpy() for name, value in evaluations[0]["trace"].items()})
    torch.save({"gains": GAIN_CANDIDATES[selected], "mean": learner.mean,
                "config": asdict(cfg)}, folder/"checkpoint.pt")
    (output/f"seed-{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "classic-test", "seed": seed, "selected": selected,
                      **evaluation, "budget": budget}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT/"work/learning-pilot-2026-10-15/classic-comparison")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--test-batches", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    manifest = {"stage": "before_run", "seeds": args.seeds, "test_batches": args.test_batches,
                "gains": GAIN_CANDIDATES, "physical": asdict(PhysicalConfig()),
                "thresholds": asdict(UpdateConfig()), "batch": 8, "episode_steps": 100,
                "calibration_batches_per_candidate": 5, "identification_updates_per_candidate": 100,
                "validation_batches_per_candidate": 1, "actor_training_updates": 0,
                "planned_physical_steps": len(args.seeds)*(4*6+args.test_batches)*8*100,
                "calibration_seeds": "200000 + seed*1000 + index; index=0..4",
                "validation_seed": "300000 + seed",
                "test_seeds": "500000 + seed*1000 + index",
                "selection": "Original pilot select_feasible; freeze pair before test",
                "actor": "Analytic static backstepping full action, no neural network",
                "known_physics_scope": "Known nominal physics used structurally; not a model-free sample-efficiency comparison",
                "source_sha256": {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                   for name in ("classic_comparator.py", "pilot.py", "learning.py",
                                                "dynamics.py", "splates_bridge.py")}}
    (args.output/"before-run-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    started = time.perf_counter()
    identity_error = teacher_identity_check()
    records = [run_seed(seed, args.output, args.test_batches) for seed in args.seeds]
    summary = {"manifest": manifest, "teacher_identity_max_error": identity_error,
               "wall_seconds": time.perf_counter()-started, "seeds": records,
               "total_physical_steps": sum(r["budget"]["physical_steps"] for r in records)}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "completed", "total_physical_steps": summary["total_physical_steps"],
                      "wall_seconds": summary["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
