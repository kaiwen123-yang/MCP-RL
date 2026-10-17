"""Frozen nominal-null feedback-jet derivative witness; no parameter update.

Only public historical data, local affine matrices and disturbance-response
matrix recurrences are used. No physical trajectory or training runner is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes/feedback-sensitivity-2026-10-17"
SOURCE = ROOT / "notes/joint-update-2026-10-16/source-segment.json"
PROTOCOL = OUT / "protocol.md"
SOURCE_HASH = "42e6c3623d808de882277c6e22c4cf8f30b38de4c54430105a130301c97d2bca"
PROTOCOL_HASH = "06db73e7345b27d04e618f3c87717db3ae5d7341250b053c57b868cc971bd1c2"
DT, K, U, QMAX, SIGMA = 0.02, 5, 1.8, 0.35, 0.35
FD_EPS = (1e-5, 1e-6)
Q = np.diag([.3, 18., 5., .6, 0., 0.])
R = .025 * np.eye(2)
QF = .2 * np.diag([0., 18., 0., .6, 0., 0.])
RESIDUALS = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
COUNTS = {"response_matrix_calls": 0, "response_matrix_steps": 0,
          "finite_difference_objective_evaluations": 0}
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def tt(value):
    return torch.as_tensor(value, dtype=torch.float64)


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def affine_model():
    ac = np.zeros((6, 6))
    ac[0, 2] = ac[1, 3] = 1
    ac[2, 2], ac[2, 4], ac[2, 5] = -.2, .5, .5
    ac[3, 1], ac[3, 3], ac[3, 4], ac[3, 5] = -1., -.15, .5, -.5
    ac[4, 4] = ac[5, 5] = -10.
    bc = np.zeros((6, 2))
    bc[4, 0] = bc[5, 1] = 10.
    ec = np.zeros((6, 1))
    ec[3, 0] = 1.
    aug = np.zeros((9, 9))
    aug[:6, :6], aug[:6, 6:8], aug[:6, 8:] = ac, bc, ec
    discrete = expm(DT * aug)
    return ac, bc, ec, discrete[:6, :6], discrete[:6, 6:8], discrete[:6, 8:]


def baseline_jacobians(xbar):
    gains = np.zeros((K, 2, 6))
    gains[:, 0, 1] = -3 * np.cos(xbar[:-1, 1])
    gains[:, 1, 1] = 3 * np.cos(xbar[:-1, 1])
    gains[:, :, 2] = -2.8
    gains[:, 0, 3], gains[:, 1, 3] = -1.5, 1.5
    return gains


def response(eta, ad, bd, ed, gains, horizon=K):
    """Map normalized disturbance energy to incremental task outputs."""
    COUNTS["response_matrix_calls"] += 1
    COUNTS["response_matrix_steps"] += horizon
    L = torch.zeros((6, horizon))
    states, controls, closed = [L], [], []
    outputs = []
    sqrtq, sqrtr, sqrtqf = tt(np.sqrt(DT * Q)), tt(np.sqrt(DT * R)), tt(np.sqrt(QF))
    for j in range(horizon):
        gain = gains[j] + eta
        f = ad + bd @ gain
        action = gain @ L
        outputs.extend((sqrtq @ L, sqrtr @ action))
        selector = torch.zeros((1, horizon))
        selector[0, j] = 1.
        L = f @ L + (ed / np.sqrt(DT)) @ selector
        states.append(L)
        controls.append(action)
        closed.append(f)
    outputs.append(sqrtqf @ L)
    transfer = torch.cat(outputs, dim=0)
    # With E=0, the squared norm is identically zero; no unique SVD is claimed.
    if float(torch.linalg.vector_norm(transfer).detach()) == 0.:
        gamma = transfer.square().sum()
        singulars = torch.zeros(horizon)
    else:
        singulars = torch.linalg.svdvals(transfer)
        gamma = singulars[0].square()
    return gamma, transfer, states, controls, closed, singulars


def capacity(states, controls, xbar, ubar):
    values, descriptions = [], []
    for j, L in enumerate(states):
        for sign in (-1., 1.):
            row = sign * L[1]
            norm = torch.linalg.vector_norm(row)
            slack = QMAX - sign * xbar[j, 1]
            value = torch.tensor(float('inf')) if float(norm.detach()) == 0. else slack / norm
            values.append(value)
            descriptions.append({"kind": "angle", "index": j, "time": .9 + j * DT,
                                 "sign": sign, "nominal": float(xbar[j, 1]), "slack": float(slack),
                                 "response_row_norm": float(norm.detach())})
    for j, C in enumerate(controls):
        for actuator in range(2):
            for sign in (-1., 1.):
                row = sign * C[actuator]
                norm = torch.linalg.vector_norm(row)
                slack = U - sign * ubar[j, actuator]
                value = torch.tensor(float('inf')) if float(norm.detach()) == 0. else slack / norm
                values.append(value)
                descriptions.append({"kind": "input_linearization_budget", "index": j,
                                     "time": .9 + j * DT, "actuator": actuator + 1,
                                     "sign": sign, "nominal": float(ubar[j, actuator]),
                                     "slack": float(slack), "response_row_norm": float(norm.detach())})
    stacked = torch.stack(values)
    result, ix = torch.min(stacked, dim=0)
    for d, v in zip(descriptions, stacked):
        d["rho_bound"] = None if not torch.isfinite(v) else float(v.detach())
        d["zero_response_unbounded"] = not bool(torch.isfinite(v))
    finite = sorted(float(v.detach()) for v in stacked if torch.isfinite(v))
    return result, int(ix), descriptions, finite[1] - finite[0]


def policy_identity(xbar, ubar, gains):
    mu0 = np.arctanh(ubar / U)
    d = U * (1 - (ubar / U)**2)
    nominal_actions, jacobian_errors, center_scores, uncenter_scores = [], [], [], []
    fisher_center, fisher_uncenter = np.zeros((12, 12)), np.zeros((12, 12))
    for j in range(K):
        def latent(eta, x, centered=True):
            delta = x - tt(xbar[j]) if centered else x
            return tt(mu0[j]) + ((tt(gains[j]) + eta) @ delta) / tt(d[j])
        eta0 = torch.zeros((2, 6), requires_grad=True)
        point = tt(xbar[j])
        action = U * torch.tanh(latent(eta0, point))
        nominal_actions.append(action.detach().numpy())
        k_actual = torch.autograd.functional.jacobian(
            lambda state: U * torch.tanh(latent(eta0, state)), point)
        jacobian_errors.append(float(torch.max(torch.abs(k_actual - tt(gains[j])))))
        centered = torch.autograd.functional.jacobian(lambda eta: latent(eta, point), eta0).reshape(2, 12).detach().numpy()
        uncentered = torch.autograd.functional.jacobian(lambda eta: latent(eta, point, False), eta0).reshape(2, 12).detach().numpy()
        center_scores.append(RESIDUALS @ centered / SIGMA)
        uncenter_scores.append(RESIDUALS @ uncentered / SIGMA)
        fisher_center += centered.T @ centered / SIGMA**2
        fisher_uncenter += uncentered.T @ uncentered / SIGMA**2
    return {"latent_mean_at_centers": mu0, "D_diagonal": d,
            "nominal_action_max_error": float(np.max(np.abs(np.array(nominal_actions) - ubar))),
            "physical_Jacobian_vs_K0_max_error": max(jacobian_errors),
            "mixed_dK_deta": "identity on all 12 physical Jacobian coordinates, analytically D*D^-1=I at every center",
            "fixed_standardized_residuals": RESIDUALS,
            "centered_scores": np.array(center_scores), "uncentered_scores": np.array(uncenter_scores),
            "centered_score_max": float(np.max(np.abs(center_scores))),
            "uncentered_score_max": float(np.max(np.abs(uncenter_scores))),
            "centered_fisher_trace": float(np.trace(fisher_center)),
            "uncentered_fisher_trace": float(np.trace(fisher_uncenter)),
            "centered_fisher_rank": int(np.linalg.matrix_rank(fisher_center)),
            "uncentered_fisher_rank": int(np.linalg.matrix_rank(fisher_uncenter)),
            "sample_policy_gradient": [0.] * 12,
            "sample_policy_gradient_identity": "Deduced for any fixed advantages from eta-invariant recorded-center density; not a computed old PPO advantage gradient",
            "fisher_identity": "Exact conditional Fisher sum J_mu.T Sigma^-1 J_mu, not an empirical covariance of the four fixed residual probes",
            "score_claim_scope": "Every fixed recorded-center density is eta-independent for any latent action and fixed advantage; not a claim about new off-center rollouts",
            "stochastic_vs_deterministic": "Gamma uses U*tanh(mu), not E[U*tanh(a)] under the Gaussian; no guarantee is transferred between them"}


def adjoint_check(transfer, states, closed, gains, bd):
    matrix = transfer.detach().numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.T @ matrix)
    v = eigenvectors[:, -1]
    e = np.stack([L.detach().numpy() @ v for L in states])
    p = 2 * QF @ e[-1]
    pieces = np.zeros((K, 2, 6))
    for j in range(K - 1, -1, -1):
        gain = gains[j]
        pieces[j] = np.outer(bd.T @ p + 2 * DT * R @ gain @ e[j], e[j])
        p = 2 * DT * (Q + gain.T @ R @ gain) @ e[j] + closed[j].detach().numpy().T @ p
    return {"eigenvalues": eigenvalues, "top_eigenvalue_gap": float(eigenvalues[-1] - eigenvalues[-2]),
            "unit_worst_direction_for_derivative_only": v, "norm_v": float(np.linalg.norm(v)),
            "incremental_states_from_stored_response": e,
            "gradient_by_time": pieces, "gradient": pieces.sum(axis=0),
            "j0_gradient_max": float(np.max(np.abs(pieces[0]))), "adjoint_reverse_steps": K,
            "note": "Eigenvector is the induced-norm definition, not an adversarial physical rollout or selected disturbance amplitude"}


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_HASH
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    xbar, ubar = np.asarray(source["x"]), np.asarray(source["u"])
    ac, bc, ec, ad, bd, ed = affine_model()
    gains = baseline_jacobians(xbar)
    offsets = xbar[1:] - xbar[:-1] @ ad.T - ubar @ bd.T
    identities = policy_identity(xbar, ubar, gains)
    at, bt, et, kt = [tt(v) for v in (ad, bd, ed, gains)]
    eta = torch.zeros((2, 6), requires_grad=True)
    gamma, transfer, states, controls, closed, singulars = response(eta, at, bt, et, kt)
    rho, facet_index, facets, facet_gap = capacity(states, controls, xbar, ubar)
    ggamma = torch.autograd.grad(gamma, eta, retain_graph=True)[0].detach().numpy()
    grho = torch.autograd.grad(rho, eta)[0].detach().numpy()
    adjoint = adjoint_check(transfer, states, closed, gains, bd)
    fd_results = []
    for epsilon in FD_EPS:
        fd_gamma, fd_rho = np.zeros((2, 6)), np.zeros((2, 6))
        active_indices = []
        for index in range(12):
            direction = torch.zeros((2, 6))
            direction.reshape(-1)[index] = epsilon
            values = []
            for sign in (1., -1.):
                val, _, ls, cs, _, _ = response(sign * direction, at, bt, et, kt)
                allowed, selected, _, _ = capacity(ls, cs, xbar, ubar)
                values.append((float(val), float(allowed)))
                active_indices.append(selected)
                COUNTS["finite_difference_objective_evaluations"] += 1
            fd_gamma.reshape(-1)[index] = (values[0][0] - values[1][0]) / (2 * epsilon)
            fd_rho.reshape(-1)[index] = (values[0][1] - values[1][1]) / (2 * epsilon)
        abs_g = float(np.max(np.abs(fd_gamma - ggamma)))
        rel_g = float(np.linalg.norm(fd_gamma - ggamma) / max(np.linalg.norm(ggamma), 1e-30))
        abs_r = float(np.max(np.abs(fd_rho - grho)))
        rel_r = float(np.linalg.norm(fd_rho - grho) / max(np.linalg.norm(grho), 1e-30))
        fd_results.append({"epsilon": epsilon, "Gamma_gradient": fd_gamma, "rho_gradient": fd_rho,
                           "Gamma_max_abs_error": abs_g, "Gamma_relative_L2_error": rel_g,
                           "rho_max_abs_error": abs_r, "rho_relative_L2_error": rel_r,
                           "Gamma_protocol_tolerance_pass": abs_g <= 5e-8 and rel_g <= 1e-4,
                           "rho_same_tolerance_pass": abs_r <= 5e-8 and rel_r <= 1e-4,
                           "all_perturbations_keep_same_limiting_facet": all(i == facet_index for i in active_indices)})
    eta_zero = torch.zeros((2, 6), requires_grad=True)
    no_disturbance, _, _, _, _, _ = response(eta_zero, at, bt, torch.zeros_like(et), kt)
    no_disturbance_grad = torch.autograd.grad(no_disturbance, eta_zero)[0].detach().numpy()
    eta_one = torch.zeros((2, 6), requires_grad=True)
    one_step, _, _, _, _, _ = response(eta_one, at, bt, et, kt, horizon=1)
    one_step_grad = torch.autograd.grad(one_step, eta_one)[0].detach().numpy()
    nominal_weighted_cost = 0.
    for j in range(K):
        target = np.array([.65 * source['t'][j], 0., .65, 0., 0., 0.])
        error = xbar[j] - target
        nominal_weighted_cost += DT * (error @ Q @ error + ubar[j] @ R @ ubar[j])
    nominal_weighted_cost += xbar[-1] @ QF @ xbar[-1]
    result = {
        "date": "2026-10-17", "identity": "Frozen historical-center, deterministic mean-feedback jet sensitivity; no training or eta update",
        "source": {"path": str(SOURCE.relative_to(ROOT)).replace('\\', '/'), "sha256": SOURCE_HASH,
                   "protocol_sha256": PROTOCOL_HASH, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "original_mat_sha256_recorded": source['source_mat_sha256'], "original_mat_verified_this_run": False,
                   "time": source['t'], "xbar": xbar, "ubar": ubar,
                   "condition_identity": "Historical data, not a PPO batch; j and xbar_j are fixed public policy conditions shared by all comparisons"},
        "frozen_settings": {"horizon": K, "dt": DT, "input_limit": U, "q_limit": QMAX, "Gaussian_sigma": SIGMA,
                            "eta_shape": [2, 6], "eta_evaluation_point": np.zeros((2, 6)),
                            "FD_eps": FD_EPS, "Q": Q, "R": R, "Qf": QF,
                            "input_normalization": "v_j=sqrt(dt)*delta_w_j, sum v_j^2 equals sampled disturbance energy",
                            "policy_formula": "mu=atanh(ubar/U)+D^-1(K0+eta)(x-xbar); u_mean_controller=U*tanh(mu)",
                            "Gaussian_action_formula": "latent a~Normal(mu,.35^2 I); actual stochastic u=U*tanh(a)",
                            "meaning_of_Gamma": "Finite-horizon induced L2 gain squared of task-weighted incremental outputs, not infinite-horizon H-infinity stability or exact full nominal task cost change"},
        "local_model": {"Ac": ac, "Bc": bc, "Ec": ec, "Ad": ad, "Bd": bd, "Ed": ed,
                        "K0": gains, "nominal_alignment_offsets": offsets,
                        "nominal_alignment_error": float(np.max(np.abs(xbar[:-1] @ ad.T + ubar @ bd.T + offsets - xbar[1:]))),
                        "model_identity": "Externally specified q=0 linearization and exact ZOH; not identified from five-step data",
                        "offset_identity": "Historical alignment including old disturbance/model defect, not future privileged information"},
        "policy_identities": identities,
        "response": {"T": transfer.detach(), "T_shape": list(transfer.shape), "L_by_time": [v.detach() for v in states],
                     "input_response_by_time": [v.detach() for v in controls], "closed_loop_matrices": [v.detach() for v in closed],
                     "singular_values": singulars.detach(), "Gamma": float(gamma.detach()),
                     "Gamma_gradient": ggamma, "Gamma_gradient_norm": float(np.linalg.norm(ggamma)),
                     "top_eigenvalue_gap": adjoint['top_eigenvalue_gap'],
                     "top_eigenvalue_is_simple_numerically": adjoint['top_eigenvalue_gap'] > 1e-9,
                     "no_eta_step_taken": True},
        "adjoint": {**adjoint, "max_error_vs_autograd": float(np.max(np.abs(adjoint['gradient'] - ggamma)))},
        "capacity": {"rho_allow": float(rho.detach()), "eta_gradient": grho, "gradient_norm": float(np.linalg.norm(grho)),
                     "limiting_facet_index": facet_index, "limiting_facet": facets[facet_index],
                     "second_smallest_minus_smallest": facet_gap, "unique_limiting_facet_numerically": facet_gap > 1e-9,
                     "facets": facets,
                     "symbolic_cases": {"rho_less_than_allow": "Every normalized disturbance in the ball satisfies the stated linearized facet budgets",
                                        "rho_equal_to_allow": "At least one worst direction can attain a limiting facet in the linearized model",
                                        "rho_greater_than_allow": "There exists a direction violating a linearized facet; not every direction and not actual hardware violation"},
                     "input_scope": "Conservative first-order budget for historical input slack; tanh already enforces actual hardware bounds",
                     "planner_scope": "Readout for the fixed old local plan; no MPC, dual, plan update or chosen disturbance radius"},
        "finite_difference_checks": fd_results,
        "negative_controls": {"E_zero": {"Gamma": float(no_disturbance.detach()), "gradient": no_disturbance_grad,
                                            "gradient_max": float(np.max(np.abs(no_disturbance_grad))),
                                            "zero_T_SVD_nonunique": True},
                              "one_step_post_action_disturbance": {"Gamma": float(one_step.detach()), "gradient": one_step_grad,
                                                                    "gradient_max": float(np.max(np.abs(one_step_grad))),
                                                                    "independent_formula": float((ed.T @ QF @ ed / DT)[0, 0])},
                              "remove_center": {"score_max": identities['uncentered_score_max'],
                                                "fisher_trace": identities['uncentered_fisher_trace'],
                                                "meaning": "Breaks the nominal-null condition; not a performance comparator"}},
        "task_cost_scope": {"historical_nominal_weighted_cost": nominal_weighted_cost,
                            "cross_term": "Full quadratic task difference has 2*zbar.T*delta_z + ||delta_z||^2",
                            "conditional_upper_bound": "DeltaJ <= 2*sqrt(Jnom)*sqrt(Gamma)*rho + Gamma*rho^2 for this fixed nominal, local response and energy ball",
                            "rho_chosen": None, "full_task_improvement_demonstrated": False},
        "preconditioner_fact": {"recorded_center_sample_gradient": [0.] * 12,
                                "Gamma_gradient_nonzero": bool(np.linalg.norm(ggamma) > 1e-12),
                                "claim": "No linear M maps the zero fixed-center nominal sample gradient to a nonzero gain gradient",
                                "does_not_rule_out": "Model-based robust policy gradients, same-disturbance PPO data, objective changes or standard optimization"},
        "budget": {**COUNTS, "adjoint_reverse_matrix_steps": K, "physical_state_Jacobian_checks": K,
                   "conditional_latent_parameter_Jacobian_checks": 2*K,
                   "fixed_standardized_score_probes": 4*K*2,
                   "constructed_model_trajectory_trees": 0, "new_physical_transitions": 0,
                   "training_updates": 0, "eta_updates": 0, "optimizer_steps": 0,
                   "checkpoint_reads": 0, "checkpoint_writes": 0, "new_random_samples": 0,
                   "FD_interpretation": "48 fixed coordinate probes only for derivative precision, no gain tuning or candidate selection"},
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "dtype": "float64"},
        "limits": ["Derivatives are externally supplied, not identified by the one historical trajectory.",
                   "Mean-feedback sensitivity does not prove stochastic rollout risk or stochastic-to-deterministic transfer.",
                   "Five-step local induced gain does not prove infinite-horizon stability, robust nonlinear safety or global performance.",
                   "No feedback parameter was updated, no plan was changed and no improved learned capability was executed.",
                   "Nominal zero score is conditional on these fixed centers; sufficiently excited on-policy data can provide ordinary PPO with off-center feedback credit.",
                   "The gain is incremental task-weighted energy, not the full nonzero-nominal worst-case task cost; the cross term is retained explicitly."]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=plain)+'\n', encoding='utf-8')
    write_report(result)
    print(json.dumps({"Gamma": result['response']['Gamma'], "gradient": ggamma,
                      "gradient_norm": result['response']['Gamma_gradient_norm'], "eigen_gap": adjoint['top_eigenvalue_gap'],
                      "adjoint_max_error": result['adjoint']['max_error_vs_autograd'],
                      "rho_allow": result['capacity']['rho_allow'], "limiting_facet": result['capacity']['limiting_facet'],
                      "rho_gradient": grho, "FD": [{k:v for k,v in z.items() if 'error' in k or 'pass' in k or k=='epsilon'} for z in fd_results],
                      "center_score": identities['centered_score_max'], "uncenter_score": identities['uncentered_score_max'],
                      "E0_gradient": result['negative_controls']['E_zero']['gradient_max'],
                      "K1_gradient": result['negative_controls']['one_step_post_action_disturbance']['gradient_max'],
                      "new_physical_transitions": 0, "eta_updates": 0}, default=plain, ensure_ascii=False))


def write_report(r):
    response = r['response']; capacity = r['capacity']; identity = r['policy_identities']; adjoint = r['adjoint']
    def vector(v): return '['+', '.join(f'{float(x):.12g}' for x in v)+']'
    grow = '\n'.join(f"| u{i+1} | "+' | '.join(f'{float(x):.12g}' for x in row)+' |' for i,row in enumerate(response['Gamma_gradient']))
    rhrow = '\n'.join(f"| u{i+1} | "+' | '.join(f'{float(x):.12g}' for x in row)+' |' for i,row in enumerate(capacity['eta_gradient']))
    fdrow = '\n'.join(f"| {v['epsilon']:g} | {v['Gamma_max_abs_error']:.3g} | {v['Gamma_relative_L2_error']:.3g} | {v['rho_max_abs_error']:.3g} | {v['rho_relative_L2_error']:.3g} | {v['Gamma_protocol_tolerance_pass'] and v['rho_same_tolerance_pass']} |" for v in r['finite_difference_checks'])
    facet = capacity['limiting_facet']
    text = f"""# 名义记录分布不变，反馈Jacobian能否带来不同任务信用

日期：2026-10-17（Asia/Shanghai）。[冻结协议](protocol.md)先于本次数值运行保存，SHA256为`{PROTOCOL_HASH}`。本次只做12维局部反馈jet的η0导数与矩阵核验，**没有η更新、训练、物理轨迹、候选树或MPC执行**。[完整数值](results.json)，[公开复算脚本](../../scripts/feedback_sensitivity_witness.py)。

## 1. 数据和策略身份

使用原公开`t=.9→1.0`完整五步记录，未另选更有利状态；相位和各`xbar_j`为所有方法共同已知的固定局部策略条件。基准K0由已使用的经典反馈实际Jacobian给出，不是神经PPO权重。潜动作均值定义为`atanh(ubar/U)+D_j^−1(K0,j+η)(x−xbar_j)`，`D_j=U diag(1−(ubar/U)^2)`；协方差固定`.35²I`。

在每个记录中心，所有η的潜动作条件分布完全一致。中心score最大绝对值`{identity['centered_score_max']:.3g}`、Fisher trace=`{identity['centered_fisher_trace']:.3g}`；因此任何只在这些固定记录条件上计算、采用固定advantage的样本PPO方向为0。**这个零向量由密度恒等式对任意固定advantage推出，未计算原PPO优势或重放原PPO梯度。** Fisher是解析`J_muᵀΣ^−1J_mu`，不是四个固定残差的经验协方差。这不等于随机策略新rollout的总体梯度为0：离开中心后，策略可出现不同状态和动作，普通PPO也可能获得反馈信用。

确定性mean-controller使用`U*tanh(mu)`，记录动作误差最大`{identity['nominal_action_max_error']:.3g}`，中心实际状态Jacobian与K0误差最大`{identity['physical_Jacobian_vs_K0_max_error']:.3g}`。混合导数`dK/dη`为12维恒等映射。Gaussian+tanh的真实动作期望一般不等于mean-controller动作；本次诱导增益只属于明确的确定性局部切片。

## 2. 控制关系给出的非名义响应信用

局部过程扰动模型为`δx_(j+1)=(Ad+Bd(K0,j+η))δx_j+(Ed/sqrt(dt))v_j`，`δx0=0`，`v_j=sqrt(dt)δw_j`。Ad/Bd/Ed来自已声明的q=0局部线性化与精确ZOH，不是由五步数据辨识。名义偏移e_j仅对齐历史记录，不赋予未来扰动知识。

输出使用原任务Q/R及终端Qf对`δx,δu`加权，得到`z=Tηv`。**目标Gamma是5步有限时域L2诱导增益平方**，不是无限时域H∞稳定性结论。T形状`{response['T_shape']}`；η0下`Gamma={response['Gamma']:.15g}`，12维梯度范数`{response['Gamma_gradient_norm']:.12g}`。最大特征值间隙为`{response['top_eigenvalue_gap']:.12g}`，本点唯一性数值判定为`{response['top_eigenvalue_is_simple_numerically']}`。

| 参数动作行 | p | q | v | r | a1 | a2 |
|---|---:|---:|---:|---:|---:|---:|
{grow}

名义固定样本梯度为全0而上表非零，故不能由任何线性预条件矩阵乘该零梯度得到上表。它来自外给动力学导数和额外扰动响应目标，**不排除同目标的标准模型化策略梯度，也不证明胜过拥有同扰动数据的普通PPO**。本例没有调整η沿此方向下降，更没有把非零导数写成新能力已学到。

## 3. 固定旧计划能容许多少局部扰动能量

使用已存在的`qMax=.35`和`U=1.8`，对状态时点j=0..5的正负角度facet、动作时点j=0..4的两执行器正负facet，共32项，计算`slack/||response row||₂`，零响应行记为无穷。η0的`rho_allow={capacity['rho_allow']:.15g}`。

最紧项：`{json.dumps(facet, ensure_ascii=False)}`。第一、第二紧界差`{capacity['second_smallest_minus_smallest']:.12g}`，唯一facet数值判定为`{capacity['unique_limiting_facet_numerically']}`。该容量对η的局部导数为：

| 参数动作行 | p | q | v | r | a1 | a2 |
|---|---:|---:|---:|---:|---:|---:|
{rhrow}

本次没有选择ρ来使约束活跃。对声明的线性对象：ρ小于该值时球内扰动满足这些一阶facet预算；相等时某个最坏方向可触界；大于时至少存在一个方向使局部预测越过facet。**输入项只是保持线性化响应处于历史输入余量内的保守一阶预算，tanh本来保证硬件动作有界。**它不是防止真实硬件越界的新增保证；状态项也没有升级为未知非线性对象的安全证书。

该值可以作为MPC针对同一反馈局部能力的条件读出，但本次只读出固定旧计划，没有运行规划、取得dual、改变命令或发布新策略。

## 4. 导数核验和负对照

独立伴随从已存响应矩阵读出单位主特征向量响应，不新做物理传播。伴随梯度与autograd最大差`{adjoint['max_error_vs_autograd']:.3g}`；第0拍梯度最大`{adjoint['j0_gradient_max']:.3g}`，符合初始偏差为零时第一拍反馈不能处理尚未进入的扰动。

| 固定FD eps | Gamma最大绝对误差 | Gamma相对L2误差 | rho最大绝对误差 | rho相对L2误差 | 预声明误差门槛通过 |
|---:|---:|---:|---:|---:|---|
{fdrow}

每个eps核12个坐标的正负点，两组共48次矩阵求值，只检查导数精度，不属于反馈性能扫描。三个负对照结果：

- `E=0`时Gamma=`{r['negative_controls']['E_zero']['Gamma']:.12g}`，梯度最大=`{r['negative_controls']['E_zero']['gradient_max']:.3g}`；零T没有唯一SVD方向，不伪造唯一性。
- `K=1`、扰动在动作后进入时Gamma=`{r['negative_controls']['one_step_post_action_disturbance']['Gamma']:.12g}`，梯度最大=`{r['negative_controls']['one_step_post_action_disturbance']['gradient_max']:.3g}`；说明仅有终端扰动不自动产生反馈学习信用。
- 去掉中心化后，同四个固定标准化潜动作残差给出score最大=`{identity['uncentered_score_max']:.12g}`、Fisher trace=`{identity['uncentered_fisher_trace']:.12g}`；零score取决于名义不变结构，不是计算遗漏了actor导数。去中心对照不具有相同名义行为，不作性能比较。

## 5. 结论范围

本次验证的是：历史名义条件不足以区分某些反馈Jacobian，而给定局部动力学导数和扰动通道后，任务加权的扰动能力及输入/状态余量可以对这些自由度给出非零需求信用。控制理论在这里补充的是off-nominal反馈响应信息，不是上轮纯法向不可能模型预测的修正。

Gamma度量`||δz||²`。完整原二次任务在非零名义误差下还含`2 zbarᵀδz`，不能直接等同Gamma。对固定名义和本线性对象，可用`ΔJ≤2 sqrt(Jnom) sqrt(Gamma)ρ+Gammaρ²`作条件上界；本次未选扰动半径或声称完整任务改善。

外给Jacobian的误差与可辨识性、有限扰动的非线性余项、随机→确定性关系、可泛化反馈函数和模型/策略发布仍未解决。这里用了局部解析12维jet，不是学习算法已优于成熟鲁棒控制或模型化RL的证据。

资源：响应矩阵调用`{r['budget']['response_matrix_calls']}`次、递推矩阵步`{r['budget']['response_matrix_steps']}`步，另5步伴随。没有候选树、真实物理转移、训练更新、η更新、优化器或checkpoint读写。复算：`& '.venv/Scripts/python.exe' 'scripts/feedback_sensitivity_witness.py'`。
"""
    (OUT/'results.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
