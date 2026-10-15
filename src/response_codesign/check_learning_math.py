"""Numerical mechanism checks; no environment rollout or RL training is run.

Run with the project venv. Writes a small JSON record under the ignored work
directory. Checks the independently specified primal problem, dual/KKT gap,
inactive-constraint equivalence and actual frozen-coordinate update behavior.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize
import torch

from .learning import Actor, JointLearner, UpdateConfig, parameter_vector, proximal_direction


def primal_value(d, g, rows, values, inv, alpha, cap):
    slack = np.maximum(values+rows@d, 0.0)
    return float(g@d+0.5/alpha*np.sum(d*d/inv)+cap*slack.sum())


def check_qp(g, rows, values, inv, alpha, cap=2.0):
    g, rows, values, inv = map(np.asarray, (g, rows, values, inv))
    d, info = proximal_direction(g, rows, values, inv, alpha, cap)
    lam = np.asarray(info["dual"])
    slack = np.asarray(info["linear_slack"])
    primal = primal_value(d, g, rows, values, inv, alpha, cap)
    v = g+rows.T@lam
    dual = float(-0.5*alpha*np.sum(v*v*inv)+lam@values)
    stationarity = float(np.max(np.abs(d/(alpha*inv)+v)))
    complementarity = max(float(np.max(np.abs(lam*(values+rows@d-slack)))),
                          float(np.max(np.abs((cap-lam)*slack))))
    assert abs(primal-dual) < 1e-8*max(1.0, abs(primal), abs(dual))
    assert stationarity < 1e-9*max(1.0, np.max(np.abs(v)))
    assert complementarity < 1e-8*max(1.0, abs(primal))
    assert np.min(slack) >= 0 and np.max(values+rows@d-slack) < 1e-10
    # Independently solve in primal variables [d, s], using linear constraints.
    n, m = len(g), len(values)
    objective = lambda x: float(g@x[:n]+0.5/alpha*np.sum(x[:n]**2/inv)+cap*x[n:].sum())
    jacobian = lambda x: np.r_[g+x[:n]/(alpha*inv), np.full(m, cap)]
    result = minimize(objective, np.r_[np.zeros(n), np.maximum(values, 0)+1e-6],
                      jac=jacobian, method="SLSQP", bounds=[(None, None)]*n+[(0, None)]*m,
                      constraints=LinearConstraint(np.c_[rows, -np.eye(m)], -np.inf, -values),
                      options={"maxiter": 300, "ftol": 1e-12})
    assert result.success, result.message
    assert abs(result.fun-primal) < 1e-7*max(1.0, abs(primal))
    return d, {"primal_dual_gap": abs(primal-dual), "stationarity": stationarity,
               "complementarity": complementarity,
               "independent_primal_difference": abs(float(result.fun)-primal), **info}


def main():
    torch.set_num_threads(1)
    np.set_printoptions(precision=12)
    records = []
    hard_like, hard_info = check_qp([-2.0], [[1.0]], [-0.5], [1.0], 1.0, 10.0)
    assert np.allclose(hard_like, [0.5], atol=1e-12)
    soft, soft_info = check_qp([-2.0], [[1.0]], [-0.5], [1.0], 1.0, 0.25)
    assert np.allclose(soft, [1.75], atol=1e-12)
    assert np.allclose(soft_info["linear_slack"], [1.25], atol=1e-12)
    impossible, impossible_info = check_qp([0.0], [[0.0]], [0.6], [1.0], 0.1)
    assert np.allclose(impossible, [0.0]) and np.allclose(impossible_info["linear_slack"], [0.6])
    records.extend([hard_info, soft_info, impossible_info])

    rng = np.random.default_rng(3917)
    for case in range(48):
        n, m = 7, 2+(case % 2)
        g = rng.normal(size=n)
        rows = rng.normal(size=(m, n))
        if case % 6 == 0:
            rows[-1] = rows[0]  # singular dual, distinct right-hand sides
        if case % 7 == 0:
            rows[0] = 0        # uncontrollable linearized constraint
        values = rng.normal(size=m)*0.4
        inv = np.exp(0.5*rng.normal(size=n))
        alpha = float(rng.uniform(0.02, 0.2))
        _, record = check_qp(g, rows, values, inv, alpha)
        records.append(record)

    # With inactive constraints and a separable metric, joint and sequential
    # blocks must give exactly the same direction; no synthetic advantage.
    g = np.array([-2.0, -1.0]); rows = np.array([[1.0, 1.0]])
    inv = np.array([1.0, 1.0]); alpha = 0.3
    equivalences = []
    coupled = {}
    for values in (np.array([-10.0]), np.array([-0.4])):
        joint, _ = proximal_direction(g, rows, values, inv, alpha, 10.0)
        block = np.zeros(2)
        for index in (0, 1):
            local, _ = proximal_direction(g[index:index+1], rows[:, index:index+1],
                                           values+rows@block, inv[index:index+1], alpha, 10.0)
            block[index] = local[0]
        if values[0] == -10.0:
            assert np.allclose(joint, block, atol=1e-12)
            equivalences.append(float(np.max(np.abs(joint-block))))
        else:
            jv = primal_value(joint, g, rows, values, inv, alpha, 10.0)
            bv = primal_value(block, g, rows, values, inv, alpha, 10.0)
            assert jv < bv-1e-6
            coupled = {"joint_direction": joint.tolist(), "block_direction": block.tolist(),
                       "joint_objective": jv, "one_sweep_block_objective": bv,
                       "scope": "Known convex QP, not a learning-performance result"}

    # One fixed synthetic derivative batch verifies real implementation paths.
    # There is no physical interaction, environment fit, or learning experiment.
    torch.manual_seed(813)
    actor = Actor()
    cfg = UpdateConfig(alpha=1e-5, trust_radius=10.0)
    features = torch.randn(6, 8, 12)
    with torch.no_grad():
        raw = actor(features)+cfg.action_std*torch.randn(6, 8)
    costs = torch.rand(6, 8, 3)
    costs *= torch.tensor([cfg.task_scale, cfg.tracking_budget*1e-3,
                           cfg.input_residual_budget*1e-3])
    payload = {"features": features, "raw_actions": raw, "costs": costs,
               "design": cfg.design_std*torch.randn(8, 4)}
    frozen = {}
    vectors = {}
    for mode in ("joint", "block", "fixed", "tuning_only", "independent"):
        learner = JointLearner(copy.deepcopy(actor), torch.zeros(4), mode, cfg)
        actor_before = parameter_vector(learner.actor)
        mean_before = learner.mean.clone()
        info = learner.update(payload)
        actor_change = float(torch.max(torch.abs(parameter_vector(learner.actor)-actor_before)))
        mean_change = float(torch.max(torch.abs(learner.mean-mean_before)))
        if mode in ("fixed", "independent"):
            assert mean_change == 0.0
        if mode == "tuning_only":
            assert actor_change == 0.0
        if mode == "block":
            assert info["block_order"] == ["actor", "response"]
            assert len(info["block_updates"]) == 2
        assert learner.baseline is not None
        frozen[mode] = {"actor_max_change": actor_change, "response_max_change": mean_change,
                        "constraints": info["constraints"], "dual": info["dual"]}
        vectors[mode] = torch.cat((parameter_vector(learner.actor), learner.mean))
    assert torch.equal(vectors["joint"], vectors["block"])
    reverse = JointLearner(copy.deepcopy(actor), torch.zeros(4), "block", cfg)
    reverse.updates = 1
    assert reverse.update(payload)["block_order"] == ["response", "actor"]

    summary = {
        "scope": "Numerical math/implementation checks only; no rollout or RL training",
        "qp_cases": len(records), "all_passed": True,
        "max_primal_dual_gap": max(r["primal_dual_gap"] for r in records),
        "max_stationarity_residual": max(r["stationarity"] for r in records),
        "max_complementarity_residual": max(r["complementarity"] for r in records),
        "max_independent_primal_difference": max(r["independent_primal_difference"] for r in records),
        "soft_constraint_example": {"direction": soft.tolist(), "slack": soft_info["linear_slack"]},
        "zero_gradient_infeasibility_example": impossible_info,
        "inactive_joint_block_difference": equivalences[0],
        "coupled_qp": coupled, "synthetic_update": frozen, "synthetic_config": asdict(cfg),
    }
    destination = Path(__file__).resolve().parents[2]/"work/learning-pilot-2026-10-15/learning-math-check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
