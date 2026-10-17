"""Frozen full-chain algebraic witness using a finite analytic policy population.

No environment/training runner is imported. All states below are constructed
by a declared affine surrogate. They are not new physical trajectories, an
old PPO batch, or fitted robot data. All parameter steps are single virtual
vectors; no optimizer or checkpoint is created.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notes/joint-update-2026-10-16/source-segment.json"
OUT = ROOT / "notes/joint-revision-2026-10-17"
SOURCE_HASH = "42e6c3623d808de882277c6e22c4cf8f30b38de4c54430105a130301c97d2bca"
DT, K, U = 0.02, 5, 1.8
COMMANDS = (0.3, 0.65, 0.8)
MODES = np.array(list(itertools.product((-1.0, 1.0), repeat=2)))
PATHS = np.array(list(itertools.product(range(4), repeat=K)), dtype=np.int64)
WEIGHTS = np.array([0.3, 18.0, 5.0, 0.6, 0.025])
TRI = np.triu_indices(2 * K)
D = 2 * K + len(TRI[0]) + 2 * K * 6
LEDGER = {"actor_probability_calls": 0, "actor_path_probabilities_evaluated": 0,
          "model_mean_calls": 0, "autograd_calls": 0}
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def tensor(a):
    return torch.as_tensor(a, dtype=torch.float64)


def serial(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def differentiate(value, parameters, retain_graph=False):
    LEDGER["autograd_calls"] += 1
    return torch.autograd.grad(value, parameters, retain_graph=retain_graph)[0]


def affine_matrices():
    ac = np.zeros((6, 6))
    ac[0, 2] = ac[1, 3] = 1
    ac[2, 2] = -0.2
    ac[2, 4:6] = 0.5
    ac[3, 1], ac[3, 3], ac[3, 4], ac[3, 5] = -1, -0.15, 0.5, -0.5
    ac[4, 4] = ac[5, 5] = -10
    bc = np.zeros((6, 2))
    bc[4, 0] = bc[5, 1] = 10
    augmented = np.zeros((8, 8))
    augmented[:6, :6], augmented[:6, 6:] = ac, bc
    discrete = expm(DT * augmented)
    return ac, bc, discrete[:6, :6], discrete[:6, 6:]


def lifted_relations(ad, bd, x0):
    f = np.concatenate([np.linalg.matrix_power(ad, j + 1) @ x0 for j in range(K)])
    t = np.zeros((K * 6, K * 2))
    for j in range(K):
        for k in range(j + 1):
            t[j * 6:(j + 1) * 6, k * 2:(k + 1) * 2] = np.linalg.matrix_power(ad, j - k) @ bd
    uoff, moff, xoff, soff = 0, 10, 65, 95
    hmat = np.zeros((60, D))
    rhs = np.concatenate((f, f**2))
    hmat[:30, uoff:moff] = -t
    hmat[:30, xoff:soff] = np.eye(30)
    hmat[30:, uoff:moff] = -2 * f[:, None] * t
    quad = t[:, TRI[0]] * t[:, TRI[1]]
    quad[:, TRI[0] != TRI[1]] *= 2
    hmat[30:, moff:xoff] = -quad
    hmat[30:, soff:] = np.eye(30)
    normal = hmat.T @ np.linalg.pinv(hmat @ hmat.T) @ hmat
    tangent = np.eye(D) - normal
    return f, t, hmat, rhs, normal, tangent


def reference(t):
    # This frozen 0.1 s witness stays wholly within the original first segment.
    assert np.min(t) >= 1.0 and np.max(t) <= 1.1 + 1e-14
    return 0.65 * t, np.full_like(t, 0.65)


def direct_cost(states_before, actions, terminal):
    times = 1.0 + np.arange(K) * DT
    pr, vr = reference(times)
    step = DT * (WEIGHTS[0] * (states_before[:, :, 0] - pr)**2
                 + WEIGHTS[1] * states_before[:, :, 1]**2
                 + WEIGHTS[2] * (states_before[:, :, 2] - vr)**2
                 + WEIGHTS[3] * states_before[:, :, 3]**2
                 + WEIGHTS[4] * np.sum(actions**2, axis=2))
    terminal_cost = 0.2 * (WEIGHTS[1] * terminal[:, 1]**2 + WEIGHTS[3] * terminal[:, 3]**2)
    return step, terminal_cost, step.sum(axis=1) + terminal_cost


def task_feature_weights(x0):
    w = np.zeros(D)
    const = 0.0
    for j in range(K):
        time = 1.0 + j * DT
        pr, vr = 0.65 * time, 0.65
        if j == 0:
            const += DT * (WEIGHTS[0] * (x0[0] - pr)**2 + WEIGHTS[1] * x0[1]**2
                           + WEIGHTS[2] * (x0[2] - vr)**2 + WEIGHTS[3] * x0[3]**2)
        else:
            o = (j - 1) * 6
            w[65 + o] += -2 * DT * WEIGHTS[0] * pr
            w[65 + o + 2] += -2 * DT * WEIGHTS[2] * vr
            w[95 + o:95 + o + 4] += DT * WEIGHTS[:4]
            const += DT * (WEIGHTS[0] * pr**2 + WEIGHTS[2] * vr**2)
        for k in (j * 2, j * 2 + 1):
            diag = np.flatnonzero((TRI[0] == k) & (TRI[1] == k))[0]
            w[10 + diag] += DT * WEIGHTS[4]
    w[95 + (K - 1) * 6 + 1] += 0.2 * WEIGHTS[1]
    w[95 + (K - 1) * 6 + 3] += 0.2 * WEIGHTS[3]
    return const, w


def construct_population(command, x0, ad, bd, hmat, rhs, const, taskw):
    """Exactly 1024 mode paths propagated once in the offline affine surrogate."""
    n = len(PATHS)
    x = np.tile(x0, (n, 1))
    before, after, actions = [], [], []
    for j in range(K):
        before.append(x.copy())
        signs = MODES[PATHS[:, j]]
        f = 0.2 * x[:, 2] + 3 * (command - x[:, 2]) + 0.1 * U * signs[:, 0]
        g = -3 * np.sin(x[:, 1]) - 1.5 * x[:, 3] + 0.1 * U * signs[:, 1]
        u = U * np.tanh(np.column_stack((f + g, f - g)) / U)
        x = x @ ad.T + u @ bd.T
        actions.append(u)
        after.append(x.copy())
    before, after, actions = [np.stack(value, axis=1) for value in (before, after, actions)]
    flat_u, flat_x = actions.reshape(n, -1), after.reshape(n, -1)
    moment = flat_u[:, TRI[0]] * flat_u[:, TRI[1]]
    features = np.concatenate((flat_u, moment, flat_x, flat_x**2), axis=1)
    step, terminal, costs = direct_cost(before, actions, after[:, -1])
    physical_residual = features @ hmat.T - rhs
    feature_cost = const + features @ taskw
    assert np.max(np.abs(physical_residual)) < 1e-11
    assert np.max(np.abs(feature_cost - costs)) < 1e-11
    return {"command": command, "before": tensor(before), "after": tensor(after),
            "actions": tensor(actions), "features": tensor(features),
            "stage_costs": tensor(step), "terminal_cost": tensor(terminal),
            "costs": tensor(costs), "p0": torch.full((n,), 1 / n),
            "b0": tensor(features.mean(axis=0)),
            "construction_checks": {"max_physics_residual": float(np.max(np.abs(physical_residual))),
                                    "max_feature_cost_error": float(np.max(np.abs(feature_cost - costs))),
                                    "max_input_magnitude": float(np.max(np.abs(actions)))}}


def path_probability(theta, population):
    LEDGER["actor_probability_calls"] += 1
    LEDGER["actor_path_probabilities_evaluated"] += len(PATHS)
    state = population["before"]
    sf, sg = tensor(MODES[:, 0]), tensor(MODES[:, 1])
    logits = (theta[0] * (population["command"] - state[:, :, 2, None]) * sf
              + theta[1] * (-state[:, :, 3, None]) * sg)
    log_all = torch.log_softmax(logits, dim=-1)
    indices = torch.as_tensor(PATHS, dtype=torch.long)[:, :, None]
    log_each = log_all.gather(2, indices).squeeze(2)
    log_path = log_each.sum(dim=1)
    return log_path.exp(), log_path, log_each


def response(phi, population, normal, structured):
    LEDGER["model_mean_calls"] += 1
    centered = population["features"] - population["b0"]
    omega = torch.softmax(torch.log(population["p0"]) + centered @ phi, dim=0)
    q = omega @ population["features"]
    if not structured:
        q = q + normal @ phi
    return q


def ppo_advantages(population):
    rewards = -population["stage_costs"].clone()
    rewards[:, -1] -= population["terminal_cost"]
    returns = torch.flip(torch.cumsum(torch.flip(rewards, dims=(1,)), dim=1), dims=(1,))
    values = torch.empty_like(returns)
    for j in range(K):
        suffixes = 4**(K - j)
        values[:, j] = returns[:, j].reshape(4**j, suffixes).mean(dim=1).repeat_interleave(suffixes)
    return returns - values, returns, values


def actor_derivatives(q_fixed, population, advantages):
    theta = torch.zeros(2, requires_grad=True)
    probabilities, _, log_each = path_probability(theta, population)
    ratio = torch.exp(log_each + math.log(4))
    clipped = ratio.clamp(0.8, 1.2)
    lclip = (population["p0"] * torch.minimum(ratio * advantages, clipped * advantages).sum(dim=1)).sum()
    mu = probabilities @ population["features"]
    residual = q_fixed - mu
    lpred = (probabilities * ((q_fixed - population["features"])**2).sum(dim=1)).sum()
    lpair = 0.5 * residual.square().sum()
    gtask = -differentiate(lclip, theta, retain_graph=True)
    gpred = differentiate(lpred, theta, retain_graph=True)
    gpair = differentiate(lpair, theta)
    return {"task": gtask.detach(), "predictability": gpred.detach(), "paired": gpair.detach(),
            "total": (gtask + gpred + gpair).detach(),
            "Lclip": float(lclip.detach()), "Lpred": float(lpred.detach()),
            "Lpair": float(lpair.detach()), "C": residual.detach()}


def model_step(populations, oldwinner, normal, taskw, const):
    modelgrad = {}
    initial_checks = {}
    for label, structured in (("S", True), ("U", False)):
        phi0 = torch.zeros(D, requires_grad=True)
        q0 = response(phi0, populations[oldwinner], normal, structured)
        model_cost = const + taskw @ q0
        grad = differentiate(model_cost, phi0, retain_graph=True)
        p = populations[oldwinner]
        pred = (p["p0"] * ((q0 - p["features"])**2).sum(dim=1)).sum()
        paired = 0.5 * (q0 - p["b0"]).square().sum()
        pred_grad = differentiate(pred, phi0, retain_graph=True)
        pair_grad = differentiate(paired, phi0)
        modelgrad[label] = grad.detach()
        initial_checks[label] = {"q0_minus_exact_mean_norm": float(torch.linalg.vector_norm(q0.detach() - p["b0"])),
                                 "prediction_phi_gradient_norm": float(torch.linalg.vector_norm(pred_grad)),
                                 "pair_phi_gradient_norm": float(torch.linalg.vector_norm(pair_grad)),
                                 "task_gradient": grad.detach()}
    step = 0.01 / (1 + torch.linalg.vector_norm(modelgrad["S"]) + torch.linalg.vector_norm(modelgrad["U"]))
    phis = {label: -step * grad for label, grad in modelgrad.items()}
    return modelgrad, phis, float(step), initial_checks


def summarize_prediction(phi, structured, populations, normal, hmat, rhs, taskw, const):
    result = []
    for p in populations:
        q = response(phi, p, normal, structured).detach()
        result.append({"command": p["command"], "q": q,
                       "nominal_cost": float(const + taskw @ q),
                       "physics_residual_l2": float(torch.linalg.vector_norm(hmat @ q - rhs)),
                       "physics_residual_max": float(torch.max(torch.abs(hmat @ q - rhs))),
                       "minimum_predicted_state_square": float(q[95:].min())})
    costs = [v["nominal_cost"] for v in result]
    winner = int(np.argmin(costs))
    return {"candidates": result, "winner_index": winner, "winner_command": COMMANDS[winner],
            "minimum_nominal_cost": costs[winner]}


def virtual_refit(theta, populations, hmat, rhs, taskw, const):
    """Oracle readout of exact current-policy moments; no common phi is fitted.

    Legacy JSON keys say exact_refit, but this function does not establish
    representability by the shared 125-dimensional exponential-tilt model.
    """
    result = []
    for p in populations:
        probs, logs, _ = path_probability(theta, p)
        q = probs @ p["features"]
        c = probs @ p["costs"]
        kl = (p["p0"] * (torch.log(p["p0"]) - logs)).sum()
        result.append({"command": p["command"], "probability_sum": float(probs.sum()),
                       "full_path_KL_old_to_new": float(kl), "exact_refit_q": q,
                       "expected_nominal_cost": float(c),
                       "linear_feature_cost_error": float(const + taskw @ q - c),
                       "physics_residual_l2": float(torch.linalg.vector_norm(hmat @ q - rhs))})
    winner = int(np.argmin([r["expected_nominal_cost"] for r in result]))
    return {"theta_virtual": theta, "candidates": result, "winner_index": winner,
            "winner_command": COMMANDS[winner],
            "minimum_expected_nominal_cost": result[winner]["expected_nominal_cost"]}


def angle_degrees(a, b):
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return math.degrees(math.acos(np.clip(float(a @ b) / denominator, -1, 1))) if denominator else None


def main():
    source_bytes = SOURCE.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_HASH
    excerpt = json.loads(source_bytes)
    x0 = np.asarray(excerpt["x"][-1])
    assert excerpt["t"][-1] == 1.0
    ac, bc, ad, bd = affine_matrices()
    f, t, h, rhs, normal, tangent = lifted_relations(ad, bd, x0)
    const, taskw = task_feature_weights(x0)
    populations = [construct_population(c, x0, ad, bd, h, rhs, const, taskw) for c in COMMANDS]
    normal_t, h_t, rhs_t, w_t = [tensor(v) for v in (normal, h, rhs, taskw)]
    oldcosts = [float(p["p0"] @ p["costs"]) for p in populations]
    oldwinner = int(np.argmin(oldcosts))
    ordered = sorted(oldcosts)
    unique_gap = ordered[1] - ordered[0]
    modelgrad, phis, hm, initial_checks = model_step(populations, oldwinner, normal_t, w_t, const)
    prediction_s = summarize_prediction(phis["S"], True, populations, normal_t, h_t, rhs_t, w_t, const)
    prediction_u = summarize_prediction(phis["U"], False, populations, normal_t, h_t, rhs_t, w_t, const)
    q_s = prediction_s["candidates"][oldwinner]["q"]
    q_u = prediction_u["candidates"][oldwinner]["q"]
    p = populations[oldwinner]
    advantages, returns, values = ppo_advantages(p)
    js = actor_derivatives(q_s, p, advantages)
    ju = actor_derivatives(q_u, p, advantages)
    oldfit = actor_derivatives(p["b0"], p, advantages)
    theta0 = torch.zeros(2, requires_grad=True)
    prob0, _, _ = path_probability(theta0, p)
    exact_task_gradient = differentiate(prob0 @ p["costs"], theta0).detach()
    signs = tensor(MODES[PATHS])
    score0 = torch.stack(((signs[:, :, 0] * (p["command"] - p["before"][:, :, 2])).sum(dim=1),
                          (signs[:, :, 1] * (-p["before"][:, :, 3])).sum(dim=1)), dim=1)
    jmu = p["features"].T @ (p["p0"][:, None] * score0)
    planning_increment = -3 * jmu.T @ (q_s - p["b0"])
    equivalent_gradient = exact_task_gradient + oldfit["predictability"] + planning_increment
    q_s_at_phi_u = response(phis["U"], p, normal_t, True).detach()
    normal_shift = normal_t @ phis["U"]
    ha = float(0.01 / (1 + torch.linalg.vector_norm(exact_task_gradient)
                      + torch.linalg.vector_norm(js["total"])))
    theta_b = -ha * exact_task_gradient
    theta_js = -ha * js["total"]
    theta_ju = -ha * ju["total"]
    norm_js = float(torch.linalg.vector_norm(js["total"]))
    matched_scale = float(torch.linalg.vector_norm(exact_task_gradient)) / norm_js if norm_js else 0.0
    theta_matched = -ha * matched_scale * js["total"]
    refits = {"strong_O1_task_PPO_exact_refit": virtual_refit(theta_b, populations, h_t, rhs_t, w_t, const),
              "JS_joint_structured_exact_refit": virtual_refit(theta_js, populations, h_t, rhs_t, w_t, const),
              "JU_joint_unstructured_exact_refit": virtual_refit(theta_ju, populations, h_t, rhs_t, w_t, const),
              "JS_matched_step_norm_exact_refit": virtual_refit(theta_matched, populations, h_t, rhs_t, w_t, const)}
    for v in refits.values():
        v["winner_changed"] = v["winner_index"] != oldwinner
        v["improvement_from_old_minimum"] = oldcosts[oldwinner] - v["minimum_expected_nominal_cost"]
        v["old_winner_command_cost_after_step"] = v["candidates"][oldwinner]["expected_nominal_cost"]

    checks = {
        "H_rank": int(np.linalg.matrix_rank(h)),
        "normal_projector_idempotence_max": float(np.max(np.abs(normal @ normal - normal))),
        "normal_projector_symmetry_max": float(np.max(np.abs(normal - normal.T))),
        "normal_times_tangent_max": float(np.max(np.abs(normal @ tangent))),
        "PPO_task_gradient_vs_exact_expectation_max": float(torch.max(torch.abs(js["task"] - exact_task_gradient))),
        "score_expected_value_max": float(torch.max(torch.abs(p["p0"] @ score0))),
        "moment_jacobian_normal_component_max": float(torch.max(torch.abs(normal_t @ jmu))),
        "S_U_model_gradient_difference_minus_Nw_max": float(torch.max(torch.abs(modelgrad["U"] - modelgrad["S"] - normal_t @ w_t))),
        "S_model_gradient_normal_component_max": float(torch.max(torch.abs(normal_t @ modelgrad["S"]))),
        "same_tangent_tilt_qS_phiS_vs_phiU_max": float(torch.max(torch.abs(q_s - q_s_at_phi_u))),
        "normal_shift_actor_sensitivity_max": float(torch.max(torch.abs(jmu.T @ normal_shift))),
        "JS_JU_actor_gradient_difference_max": float(torch.max(torch.abs(js["total"] - ju["total"]))),
        "A13_vs_equivalent_ordinary_full_objective_gradient_max": float(torch.max(torch.abs(js["total"] - equivalent_gradient))),
        "matched_actor_step_norm_difference": float(torch.abs(torch.linalg.vector_norm(theta_matched) - torch.linalg.vector_norm(theta_b))),
        "S_U_phi_difference_minus_negative_hNw_max": float(torch.max(torch.abs(phis["U"] - phis["S"] + hm * normal_t @ w_t))),
        "S_U_q_difference_minus_negative_hNw_max": max(float(torch.max(torch.abs(u["q"] - s["q"] + hm * normal_t @ w_t)))
                                                       for s, u in zip(prediction_s["candidates"], prediction_u["candidates"])),
        "S_U_cost_difference_minus_negative_hNw_squared_max": max(abs(u["nominal_cost"] - s["nominal_cost"]
                                                                    + hm * float((normal_t @ w_t).square().sum()))
                                                                 for s, u in zip(prediction_s["candidates"], prediction_u["candidates"])),
        "planning_credit_component_difference_vs_moment_formula_max": float(torch.max(torch.abs(
            js["predictability"] - oldfit["predictability"] + js["paired"] - planning_increment))),
        "S_U_model_candidate_ranking_equal": bool(np.array_equal(
            np.argsort([v["nominal_cost"] for v in prediction_s["candidates"]]),
            np.argsort([v["nominal_cost"] for v in prediction_u["candidates"]]))),
    }
    for name in ("PPO_task_gradient_vs_exact_expectation_max", "JS_JU_actor_gradient_difference_max",
                 "A13_vs_equivalent_ordinary_full_objective_gradient_max"):
        assert checks[name] < 1e-10, (name, checks[name])
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
    checks["source_excerpt_unchanged"] = True
    report = {
        "date": "2026-10-17",
        "identity": "No-training analytic finite-population chain witness, not old PPO replay or physical execution",
        "source": {"public_excerpt_path": SOURCE.relative_to(ROOT).as_posix(), "excerpt_sha256": SOURCE_HASH,
                   "selection": "last stored state at t=1.0 after the first recorded five-step segment; the first physical disturbance lasts longer", "x0": x0,
                   "original_mat_sha256_recorded": excerpt["source_mat_sha256"],
                   "original_mat_verified_this_run": False,
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "frozen_design": {"commands": COMMANDS, "command_origin": "Three existing task-reference speeds, frozen before comparison",
                          "K": K, "dt": DT, "U": U, "theta_old": [0, 0], "phi_old": [0] * D,
                          "modes_sign_f_sign_g": MODES, "population_paths_per_command": len(PATHS),
                          "state_conditioning": "Actor reads six-state-derived q,v,r and command; no task reference input",
                          "actor_logits": "theta1*sign_f*(c-v) + theta2*sign_g*(-r)",
                          "actor_fixed_action_map": "f=.2v+3(c-v)+.1U sign_f; g=-3sin(q)-1.5r+.1U sign_g; u=U tanh([f+g,f-g]/U)",
                          "task_stage_weights": WEIGHTS, "task_terminal": ".2*(18*q^2+.6*r^2)",
                          "task_reference": "p_ref=.65*t, v_ref=.65 on t=1.0..1.1",
                          "prediction_scope": "One macro segment, not original 12-macro MPC horizon",
                          "replanning_anchor": "All before/after scores use the same historical t=1.0 state, not a newly measured t=1.1 state",
                          "objective_semantics": "Exact expected quadratic cost from first and second moments; differs from the main A5 default cost evaluated only on a mean trajectory, equally applied to all arms",
                          "J_U_ablation_scope": "Normal directions of the entire lifted motion/input moment relation H; not only the deadline capacity inequality and not a full neural J-U implementation",
                          "model_step_rule": ".01/(1+norm(gS)+norm(gU))",
                          "actor_step_rule": ".01/(1+norm(gtask)+norm(gJS))",
                          "joint_coefficients": {"etaP": 1, "etaJ": 1, "rho": 1, "lambda_dual": 0},
                          "PPO": {"gamma": 1, "gae_lambda": 1, "clip": .2, "time_reduction": "sum, not mean", "baseline": "exact prefix-conditioned old value"}},
        "local_surrogate": {"Ac": ac, "Bc": bc, "Ad": ad, "Bd": bd,
                            "discretization": "Exact ZOH of q=0 linearized affine model; w=0 nominal",
                            "nonlinear_plant_difference": "rdot error q-sin(q); no nonlinear physical predictions claimed",
                            "F_lifted": f, "T_lifted": t},
        "feature_representation": {"D": D, "layout": {"U_mean": [0, 10], "U_upper_triangle_second_moment": [10, 65],
                                                           "X_after_mean": [65, 95], "X_after_square_mean": [95, 125]},
                                   "triangle_indices": np.column_stack(TRI), "H_shape": list(h.shape),
                                   "H_sha256_float64": hashlib.sha256(h.astype('<f8').tobytes()).hexdigest(),
                                   "h": rhs, "task_constant": const, "task_weights": taskw,
                                   "constraints": "H b = h for every constructed path; N=H.T pinv(H H.T) H, P=I-N",
                                   "structured_model": "qS(phi)=sum softmax(log p0+(b-b0).T phi)*b",
                                   "unstructured_model": "qU(phi)=qS(phi)+N phi; same parameter count, but intentionally more infeasible output freedom"},
        "construction_checks": [{"command": p["command"], **p["construction_checks"]} for p in populations],
        "old_planning": {"candidate_expected_costs": oldcosts, "winner_index": oldwinner,
                         "winner_command": COMMANDS[oldwinner], "unique_winner_gap": unique_gap,
                         "unique_winner": unique_gap > 1e-12},
        "model_virtual_step": {"initial_fit_checks": initial_checks, "gradients": modelgrad,
                               "h_m": hm, "phi_S": phis["S"], "phi_U": phis["U"],
                               "S_predictions": prediction_s, "U_predictions": prediction_u,
                               "normal_output_shift_at_old_winner": normal_shift,
                               "normal_optimistic_cost_shift_theory": -hm * float((normal_t @ w_t).square().sum()),
                               "ranking_invariance_scope": "This construction shares H and task w across all candidates, so the same normal cost shift cannot reorder S/U candidates; this is not a general MPC ranking result"},
        "actor_credit": {"command_distribution": "Only old winner command for every method; no command-distribution confound",
                         "advantages_shape": list(advantages.shape), "advantage_min": float(advantages.min()),
                         "advantage_max": float(advantages.max()), "returns_shape": list(returns.shape),
                         "old_value_shape": list(values.shape), "exact_task_gradient": exact_task_gradient,
                         "JS_components": js, "JU_components": ju,
                         "old_exact_fit_components": oldfit,
                         "old_prediction_variance_gradient": oldfit["predictability"],
                         "planning_pairing_increment": planning_increment,
                         "planning_pairing_increment_from_components": js["predictability"] - oldfit["predictability"] + js["paired"],
                         "prediction_variance_to_planning_increment_norm_ratio": float(torch.linalg.vector_norm(oldfit["predictability"]) / torch.linalg.vector_norm(planning_increment)),
                         "decomposition": "gJS=gtask+gvariance-3*Jmu.T*(qS-b0) at theta_old",
                         "Jmu": jmu, "ordinary_same_full_objective_gradient": equivalent_gradient,
                         "angle_gJS_vs_gtask_degrees": angle_degrees(js["total"], exact_task_gradient),
                         "norm_gJS": float(torch.linalg.vector_norm(js["total"])),
                         "norm_gtask": float(torch.linalg.vector_norm(exact_task_gradient))},
        "actor_virtual_steps_and_exact_refits": {"h_a": ha, "matched_JS_scale": matched_scale, "methods": refits,
                                                 "readout_identity": "Same exact analytic current-policy moment oracle for all methods, ideal sufficient-fit evaluation; no shared phi fitted and no model release performed"},
        "checks": checks,
        "budget": {"historical_data_used": "One already-published six-state endpoint; no new measurements",
                   "constructed_candidate_paths": len(COMMANDS) * len(PATHS),
                   "constructed_path_expanded_model_transitions": len(COMMANDS) * len(PATHS) * K,
                   "distinct_prefix_model_edges": len(COMMANDS) * sum(4**j for j in range(1, K + 1)),
                   "construction_repeated_after_virtual_steps": False, **LEDGER,
                   "model_virtual_vectors": 2, "actor_virtual_vectors": 4,
                   "real_physical_transitions": 0, "training_updates": 0, "optimizer_steps": 0,
                   "checkpoint_files_read": 0, "checkpoint_files_written": 0,
                   "new_random_samples": 0, "old_PPO_batch_replay": False,
                   "max_feature_tensor_bytes_one_population": p["features"].numel() * 8,
                   "note": "Offline model propagation is counted and is not free physical data; exact finite population has no Monte Carlo/SNIS sampling error."},
        "runtime": {"torch": torch.__version__, "scipy": scipy.__version__, "numpy": np.__version__, "dtype": "float64", "torch_threads": 1},
        "boundaries": ["No training or physical rollout; finite analytic categorical policy is not the old Gaussian PPO actor.",
                       "Finite known surrogate population is much easier than unknown-system/high-dimensional PPO data estimation.",
                       "Original task weights are preserved, but only one 0.1-second macro stage and three fixed commands are compared.",
                       "Unstructured output permits infeasible normal motion/input statistics and is a structural ablation, not a fair deployable performance competitor.",
                       "Legacy exact_refit keys mean an exact analytic current-policy moment oracle shared by all methods. No shared phi+ is fitted or proven to represent all commands; the main algorithm's model-release step is not tested.",
                       "Every before/after candidate score is evaluated at the same historical t=1.0 anchor; no advanced measured state or actual replanning event occurs.",
                       "This instance uses exact expected quadratic cost from first/second moments, not the main A5 default mean-trajectory nominal cost; all comparisons use the same convention.",
                       "J-U relaxes the entire lifted motion/input moment normal relation, not only a deadline capacity constraint. All finite-tree actions already obey hardware bounds; no active resource-boundary learning effect is demonstrated.",
                       "Gradient/nominal-cost differences alone do not establish improved physical feedback capability or long-horizon benefit."],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chain-witness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=serial) + "\n", encoding="utf-8")
    write_markdown(report)
    print(json.dumps({"old_costs": oldcosts, "old_winner": COMMANDS[oldwinner],
                      "model_step": hm, "actor_step": ha, "gtask": exact_task_gradient,
                      "gJS": js["total"], "gJU": ju["total"], "gvariance": oldfit["predictability"],
                      "planning_pairing_increment": planning_increment,
                      "JS_vs_JU_gradient_max": checks["JS_JU_actor_gradient_difference_max"],
                      "refit_results": {key: {"cost": v["minimum_expected_nominal_cost"], "command": v["winner_command"],
                                                "winner_changed": v["winner_changed"]} for key, v in refits.items()},
                      "real_physical_transitions": 0, "training_updates": 0}, ensure_ascii=False, default=serial))


def write_markdown(r):
    old, model, credit = r["old_planning"], r["model_virtual_step"], r["actor_credit"]
    steps = r["actor_virtual_steps_and_exact_refits"]
    rows_old = "\n".join(f"| {c:g} | {v:.15f} |" for c, v in zip(COMMANDS, old["candidate_expected_costs"]))
    rows_model = "\n".join(
        f"| {label} | {v['command']:g} | {v['nominal_cost']:.15f} | {v['physics_residual_l2']:.6g} |"
        for label, key in (("S物理结构", "S_predictions"), ("U去结构", "U_predictions"))
        for v in model[key]["candidates"]
    )
    def pair(t):
        return "[" + ", ".join(f"{float(v):.12g}" for v in t) + "]"
    rows_actor = "\n".join(
        f"| {label} | {pair(v['task'])} | {pair(v['predictability'])} | {pair(v['paired'])} | {pair(v['total'])} |"
        for label, v in (("JS", credit["JS_components"]), ("JU", credit["JU_components"]), ("旧精确拟合", credit["old_exact_fit_components"]))
    )
    rows_refit = "\n".join(
        f"| {key} | {pair(v['theta_virtual'])} | {v['winner_command']:g} | {v['minimum_expected_nominal_cost']:.15f} | {v['improvement_from_old_minimum']:.8g} | {v['candidates'][old['winner_index']]['full_path_KL_old_to_new']:.6g} |"
        for key, v in steps["methods"].items()
    )
    checks = r["checks"]
    methods = steps["methods"]
    base = methods["strong_O1_task_PPO_exact_refit"]
    joint = methods["JS_joint_structured_exact_refit"]
    matched = methods["JS_matched_step_norm_exact_refit"]
    norm_ratio = credit["norm_gJS"] / credit["norm_gtask"]
    matched_excess_cost = matched["minimum_expected_nominal_cost"] - base["minimum_expected_nominal_cost"]
    text = f"""# 任务需求—模型步—actor梯度—下一次名义规划：不训练整链见证

日期：2026-10-17（Asia/Shanghai）。**本次完成的是固定解析替代模型上的整链代数计算，没有训练、物理执行或旧PPO批次重放。** [完整数值](chain-witness.json)、[复算脚本](../../scripts/joint_update_chain_witness.py)、[来源身份](provenance.md)。所有设计常数在运行前冻结，没有为了得到正结果搜索命令、步长、权重或状态。

## 1. 输入与构造的身份

唯一真实历史输入是公开旧片段`t=1.0`的六维末状态：`{pair(r['source']['x0'])}`。后续状态全部由声明的`q=0`局部仿射模型、精确ZOH、名义`w=0`构造；它们不是新增真实轨迹。原任务stage权重`[.3,18,5,.6,.025]`和终端`.2*(18q²+.6r²)`沿用，预测范围缩为一宏段5步/0.1秒，参考`p_ref=.65t,v_ref=.65`。**所有更新前后评分仍在同一个历史`t=1.0`锚点，未前进到新测得的`t=1.1`状态。** 不能把这个短名义对象写成原12宏段完整MPC测试。

固定候选命令`.3,.65,.8`取自原任务参考速度，不围绕赢家边界选择。解析actor有四种`(sign_f,sign_g)`模式，softmax logits为`θ1 sign_f(c−v)+θ2 sign_g(−r)`，`θold=[0,0]`。固定动作映射为`f=.2v+3(c−v)+.1U sign_f`、`g=−3sin(q)−1.5r+.1U sign_g`、`u=U tanh([f+g,f−g]/U)`。策略不读任务参考，避免模型遗漏其条件。它是本轮解析策略，**不是原Gaussian PPO checkpoint**。

每candidate枚举全部`4^5=1024`模式路径，无随机采样。给定模式序列，状态和动作不随θ变化；θ只改变该路径概率。虚拟更新后只重加权同一有限population，不重新传播状态。所有均值与梯度均为这个有限解析对象的完整求和，不能用“无采样误差”暗示真实未知系统也有此条件。

## 2. 真正由任务产生的模型需求

每路径特征包含10维动作、55维动作二阶矩上三角、30维动作后状态、30维状态平方，总125维。lifted仿射关系`X=F+TU`导出60行线性矩约束`Hb=h`，包括状态均值和状态平方与动作矩的对应关系。原任务精确写作`const+wᵀE[b]`，并与每条路径直接计算成本核对。这里通过二阶矩计算**精确期望二次成本**，与主稿A5默认在均值轨迹上计算名义成本的语义不同；这是对所有比较统一采用的有限实例，不冒称原A5未经改变的运行。

S模型为`qS=Σ softmax(log p0+(b−b0)ᵀφ)b`；U为`qU=qS+Nφ`，`N=Hᵀ(HHᵀ)^†H`。两者125个参数、相同状态/特征/population，φ0时都精确拟合。U放开的是**整个lifted运动/输入矩关系的法向约束**，不是只去掉deadline容量界，也不是完整J-U神经实现。U允许物理不一致的法向输出，是结构消融，不能拿其虚假低成本作可部署性能比较。树上全部动作本来就满足硬件权限，本例没有展示活跃资源界带来的学习效应。

初始固定候选池成本：

| 命令c | 精确期望名义成本 |
|---:|---:|
{rows_old}

唯一winner为`c={old['winner_command']:g}`，次佳差`{old['unique_winner_gap']:.12g}`。对`min_c(const+wᵀq)`实际求导产生`gS/gU`，**不使用人工d_req**。共同虚拟模型步长`h_m=.01/(1+||gS||+||gU||)={model['h_m']:.12g}`。初始预测拟合和配对项对φ的梯度为数值零，故这一步与已定义联合目标的初始模型步相符。

| 模型 | c | 虚拟模型步后名义成本 | Hb−h的L2残差 |
|---|---:|---:|---:|
{rows_model}

U的名义成本在本例降为负数，同时违反物理矩关系；这正是输出空间的乐观失真，不能解释为能力提高。理论预言每个candidate额外成本变化`−h_m||Nw||²={model['normal_optimistic_cost_shift_theory']:.15f}`；数值最大残差`{checks['S_U_cost_difference_minus_negative_hNw_squared_max']:.3g}`。`φU−φS=−h_m Nw`与`qU−qS=−h_m Nw`的最大残差分别为`{checks['S_U_phi_difference_minus_negative_hNw_max']:.3g}`、`{checks['S_U_q_difference_minus_negative_hNw_max']:.3g}`。

本构造的所有candidate共享同一个H和任务w，所以U对每个candidate的成本偏移相同，S/U模型步后的候选排序必然相同，数值也确认一致。这是本构造的特殊性质，不能推广为一般MPC排序结论。后面oracle读出时O1/JS赢家是否改变，仍是另一个有限actor步的实际数值结果。

## 3. 经过同一批构造路径计算actor信用

所有actor比较都只使用旧winner命令的同一完整population。每拍reward为负名义成本，末拍含terminal；`γ=λ=1`，精确prefix条件value提供baseline，保留ratio/clip=.2并对5拍求和。初始任务PPO梯度与直接精确期望成本梯度最大差`{checks['PPO_task_gradient_vs_exact_expectation_max']:.3g}`。

模型候选q停止梯度后，分别求`gtask=−∇Lclip`、`gpred=∇Eθ||q−b||²`、`gpair=∇[.5||q−Eθb||²]`，系数均1、初始dual0：

| 对象 | task | predictability | paired | total |
|---|---|---|---|---|
{rows_actor}

旧精确拟合处的predictability也可能有非零梯度，因为actor改变响应方差；这不来自MPC需求。精确分解为

`gJS=gtask+gvariance−3 Jμᵀ(qS−b0)`。

本次`gvariance={pair(credit['old_prediction_variance_gradient'])}`，模型需求/配对增量为`{pair(credit['planning_pairing_increment'])}`。前者范数是后者的`{credit['prediction_variance_to_planning_increment_norm_ratio']:.9g}`倍，因而本例联合梯度的主要改变来自旧预测方差项。直接计算`(gPred(qnew)−gPred(q0))+gPair(qnew)`与`−3Jμᵀ(qnew−q0)`的差为`{checks['planning_credit_component_difference_vs_moment_formula_max']:.3g}`。完整分解与普通自动微分完整目标的最大差为`{checks['A13_vs_equivalent_ordinary_full_objective_gradient_max']:.3g}`；没有新的求解器或信用公式优势。

**本例的S/U actor梯度相同，最大差`{checks['JS_JU_actor_gradient_difference_max']:.3g}`。** 原因是所有真实构造特征位于`Hb=h`，actor统计导数位于其切空间；`gU_model−gS_model=Nw`，共享步长只造成法向参数差，指数倾斜不变。法向乐观模型误差与真实actor统计方向正交，`JμᵀNφ`最大`{checks['normal_shift_actor_sensitivity_max']:.3g}`。因此这里控制结构修复了模型不可能的预测，**没有因该法向限制产生新的物理actor方向**。

## 4. 一次虚拟actor步与下一次名义选择

共同`h_a=.01/(1+||gtask||+||gJS||)={steps['h_a']:.12g}`。强O1使用同物理结构和全部路径，只用任务PPO方向；JS/JU使用上述联合梯度。随后所有方法都获得**同一个解析oracle给出的当前策略精确矩**，作为理想充分拟合评估。同一批路径以新θ重加权，没有拟合训练；尚未证明这些命令条件均值能由同一个125维tilt参数φ表示，**未拟合φ+，也未检验主稿的模型发布步骤**。JSON中历史命名`exact_refit`仅表示这项oracle读出。另只做一次JS方向的步长范数匹配，比例`{steps['matched_JS_scale']:.12g}`；未做搜索。

| 方法 | 虚拟θ | 新winner | 最优期望名义成本 | 相对旧成本下降 | oldwinner完整路径KL(old∥new) |
|---|---|---:|---:|---:|---:|
{rows_refit}

JS与任务梯度夹角为`{credit['angle_gJS_vs_gtask_degrees']:.9g}°`。**四种虚拟actor步后winner都仍是`c={old['winner_command']:g}`，没有发生命令切换。** 同一h下JS成本下降较多，但步长范数为任务PPO的`{norm_ratio:.9g}`倍；范数匹配后JS成本比任务PPO高`{matched_excess_cost:.9g}`，两者几乎一致，没有显示额外任务优势。JSON逐candidate列出成本与完整路径KL。这里不把“计算了不同梯度”写成“学到更有用能力”；名义收益来自已知局部模型与解析population，不是实际反馈学习收益。

## 5. 证据结论与边界

这次从原任务目标生成模型需求，并贯通模型步、actor导数、单次虚拟参数和oracle矩读出后的同锚点名义选择；补上了旧人工二维需求见证未覆盖的局部计算链，尚未完成真实模型拟合/发布链。与此同时，S/U actor梯度等价表明：在完整population和精确物理统计条件下，法向结构约束的直接作用止于消除不可能模型预测，不能单独确立新的RL能力学习优势。

JS相对强O1的差异须拆成预测方差项与任务需求/配对项，再看同范数有限步结果。本例没有解决未知系统、局部条件稀疏样本、长程滚动访问分布或全机器人问题。解析策略动作映射本身含经典反馈，也不能把这次构造的名义成本称为RL新增贡献。

资源：3×1024条构造模式路径，15360个按完整路径展开的离线模型转移（独立prefix边4092）；虚拟步后不重新推进模型。实际物理转移0、训练更新0、optimizer.step 0、checkpoint读写0。概率求值、模型均值/导数调用、虚拟向量数量均列于JSON；离线模型计算不冒充免费真实样本。

复算：`& '.venv/Scripts/python.exe' 'scripts/joint_update_chain_witness.py'`。只依赖公开摘录、NumPy/SciPy/PyTorch；不依赖忽略MAT或旧checkpoint。此脚本只输出本报告与JSON，不写训练权重。
"""
    (OUT / "chain-witness.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
