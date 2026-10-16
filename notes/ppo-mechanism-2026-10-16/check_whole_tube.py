"""Recompute the sufficient whole-tube conditions; no simulator or training.

Run with the repository Python environment. This is floating-point verification
of an analytically derived sufficient condition, not interval certification.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import solve_discrete_are


def recompute() -> dict:
    directory = Path(__file__).resolve().parent
    repository = directory.parents[1]
    source_path = repository / "src/response_ppo/env.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    config_class = next(node for node in module.body
                        if isinstance(node, ast.ClassDef) and node.name == "EnvironmentConfig")
    config = {node.target.id: ast.literal_eval(node.value) for node in config_class.body
              if isinstance(node, ast.AnnAssign)}
    h, tau = config["dt"], config["tau"]
    rho, delta, W = config["rho"], config["delta"], config["disturbance_bound"]
    actual = np.asarray(config["actual_bounds"])
    reference = np.asarray(config["reference_bounds"])
    A = np.array([[1., h, 0.], [-h, 1-.2*h, h], [0., 0., 1-h/tau]])
    B = np.array([[0.], [0.], [h/tau]])
    E = np.array([0., h, 0.])
    Q, R = np.diag([2., .5, .02]), np.array([[.05]])
    S = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R+B.T@S@B, B.T@S@A)
    P = S/np.trace(S)
    vbar = delta/(1-rho)
    alpha_bound, beta_bound = .204, .2
    assert 1-np.cos(actual[0]) <= alpha_bound
    assert .2*actual[1] <= beta_bound
    vertices = []
    for alpha in (0., alpha_bound):
        for beta in (0., beta_bound):
            M = A-B@K
            M[1, 0] += h*alpha
            M[1, 1] -= h*beta
            Qi = rho*P-M.T@P@M
            b = M.T@P@E*W
            c = float(W*W*(E@P@E))
            argmax = np.linalg.solve(Qi, b)
            vertices.append({
                "alpha": alpha, "beta": beta,
                "min_Q_eigenvalue": float(np.linalg.eigvalsh(Qi).min()),
                "max_excess_before_delta": float(c+b@argmax),
                "argmax_e": argmax.tolist(),
                "argmax_V": float(argmax@P@argmax),
            })
    input_correction = float(np.sqrt(vbar*(K@np.linalg.solve(P, K.T))[0, 0]))
    margins = np.sqrt(vbar*np.diag(np.linalg.inv(P)))
    reference_exact_pv = actual[:2]-margins[:2]
    worst_excess = max(row["max_excess_before_delta"] for row in vertices)
    assert min(row["min_Q_eigenvalue"] for row in vertices) > 0
    assert worst_excess < delta
    assert 1+input_correction < config["input_bound"]
    return {
        "kind": "analytical sufficient condition, deterministic float64 matrix check, no training",
        "source": {"path": "src/response_ppo/env.py",
                   "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "A": A.tolist(), "B": B.tolist(), "E": E.tolist(),
        "Q": Q.tolist(), "R": R.tolist(), "P": P.tolist(), "K": K.tolist(),
        "K_sign_convention": "witness u = u_reference - K e; env nominal_K is -K",
        "rho": rho, "delta": delta, "vbar": vbar,
        "alpha_bound": alpha_bound, "analytic_alpha": float(1-np.cos(actual[0])),
        "beta_bound": beta_bound, "vertices": vertices,
        "worst_excess_before_delta": worst_excess,
        "dissipation_numerical_margin": delta-worst_excess,
        "input_bound": config["input_bound"], "reference_input_bound": 1.,
        "max_input_correction": input_correction,
        "input_numerical_margin": config["input_bound"]-1-input_correction,
        "state_margins": margins.tolist(),
        "reference_bounds_exact_pv": reference_exact_pv.tolist(),
        "reference_bounds_config": reference.tolist(),
        "reference_bounds_float32": reference.astype(np.float32).astype(float).tolist(),
        "config_pv_rounding_excess": (reference[:2]-reference_exact_pv).tolist(),
        "float32_pv_rounding_excess": (reference.astype(np.float32)[:2]-reference_exact_pv).tolist(),
        "note": "Configured decimal bounds are rounded approximations. No strict zero-tolerance domain claim follows from rounding. No interval arithmetic or complete MPC recursive feasibility certificate.",
    }


if __name__ == "__main__":
    result = recompute()
    destination = Path(__file__).resolve().with_name("whole-tube-check.json")
    destination.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                     ("worst_excess_before_delta", "dissipation_numerical_margin",
                      "max_input_correction", "reference_bounds_exact_pv",
                      "float32_pv_rounding_excess")}, indent=2))
