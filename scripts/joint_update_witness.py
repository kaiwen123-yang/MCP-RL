"""One fixed, historical-data algebraic witness; no plant, MPC or RL run.

The requested capability vector is declared by the analyst, not extracted
from a PPO batch or an MPC solution. This checks the capability projection
and an actuator-memory/input relation on a previously collected segment.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work/capability-probe-2026-10-16/study.mat"
OUTPUT = ROOT / "notes/joint-update-2026-10-16"
EXCERPT = OUTPUT / "source-segment.json"
SOURCE_SHA256 = "076c3e11416c8c87dc3fe59a10940f97cefbb5c99476789e089a0fe1e99a0a01"
START, K = 45, 5
D_REQUEST = 1.0


def numbers(value):
    """Turn NumPy scalars/arrays into plain JSON numbers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def read_source(mode):
    """Public excerpt replay and original MAT verification have distinct provenance."""
    excerpt_hash = hashlib.sha256(EXCERPT.read_bytes()).hexdigest()
    excerpt = json.loads(EXCERPT.read_text(encoding="utf-8"))
    assert excerpt["source_mat_sha256"] == SOURCE_SHA256
    assert excerpt["start_index_zero_based"] == START and excerpt["K"] == K
    if mode == "mat":
        source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        if source_hash != SOURCE_SHA256:
            raise ValueError("The frozen historical study hash has changed.")
        run = loadmat(SOURCE, simplify_cells=True)["evaluation"][0, 0]
        extracted = {
            "t": run.t[START:START + K + 1],
            "x": run.x[START:START + K + 1],
            "u": run.u[START:START + K],
            "command": run.command[START:START + K],
            "w": run.disturbance[START:START + K],
        }
        for name, value in extracted.items():
            if not np.array_equal(value, np.asarray(excerpt[name])):
                raise ValueError(f"The public excerpt differs from the MAT field {name}.")
        cfg = {"U": run.cfg.U, "dt": run.cfg.dt,
               "integrationStep": run.cfg.integrationStep, "tau": run.cfg.tau}
        for name, value in cfg.items():
            if not np.array_equal(np.asarray(value), np.asarray(excerpt["cfg"][name])):
                raise ValueError(f"The public excerpt differs from MAT cfg.{name}.")
    return excerpt, {
        "mode": mode,
        "excerpt_path": EXCERPT.relative_to(ROOT).as_posix(),
        "excerpt_sha256": excerpt_hash,
        "original_mat_verified_this_run": mode == "mat",
        "original_mat_matches_excerpt_this_run": True if mode == "mat" else None,
        "explanation": "Original MAT hash and slice verified this run" if mode == "mat" else
                       "Recomputed from public excerpt only; original MAT hash was not reverified this run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("mat", "excerpt"), default="excerpt")
    args = parser.parse_args()
    segment, provenance = read_source(args.source)
    source_hash = segment["source_mat_sha256"]  # Recorded origin, not necessarily read this run.
    x = np.asarray(segment["x"])
    actual_u = np.asarray(segment["u"])
    cfg = segment["cfg"]
    f = actual_u.mean(axis=1)
    g = (actual_u[:, 0] - actual_u[:, 1]) / 2
    actuator_difference = (x[:, 4] - x[:, 5]) / 2
    limit = float(cfg["U"])

    # Match the stored numerical integration, not continuous exact ZOH.
    substeps = round(cfg["dt"] / cfg["integrationStep"])
    h = cfg["dt"] / substeps
    tau = np.asarray(cfg["tau"])
    assert tau[0] == tau[1]
    z = h / tau[0]
    alpha = (1 - z + z**2 / 2 - z**3 / 6 + z**4 / 24) ** substeps
    beta = (1 - alpha) * alpha ** np.arange(K - 1, -1, -1)
    initial_memory = alpha**K * actuator_difference[0]
    old_d = float(beta @ g)
    old_endpoint = float(actuator_difference[-1])
    old_endpoint_residual = old_endpoint - initial_memory - old_d

    # Both features have actuator-input units. b is deliberately time weighted.
    old_b = float(beta @ np.abs(f))
    total_capacity = float(limit * beta.sum())
    fixed_common_capacity = float(total_capacity - old_b)
    requested = np.array([D_REQUEST, old_b])
    independent_solution = requested.copy()
    independent_gap = float(requested.sum() - total_capacity)

    # In this fixed positive sector, the active half-space projection is exact.
    multiplier = independent_gap / 2
    joint_solution = requested - multiplier
    assert independent_gap > 0 and np.all(joint_solution > 0)

    def objective(v):
        return 0.5 * float((v - requested) @ (v - requested))

    qp = minimize(
        objective,
        requested,
        method="SLSQP",
        jac=lambda v: v - requested,
        bounds=[(0, None), (0, None)],
        constraints={
            "type": "ineq",
            "fun": lambda v: total_capacity - v.sum(),
            "jac": lambda v: -np.ones(2),
        },
        options={"ftol": 1e-13, "maxiter": 30},
    )
    if not qp.success:
        raise RuntimeError(qp.message)
    same_problem_error = float(np.max(np.abs(qp.x - joint_solution)))

    # This is a feasible-input existence witness, never a deployed feedback.
    d_new, b_new = joint_solution
    f_new = (b_new / old_b) * f
    headroom = limit - np.abs(f_new)
    g_new = (d_new / (total_capacity - b_new)) * headroom
    u_new = np.column_stack([f_new + g_new, f_new - g_new])
    reconstructed = np.array([beta @ g_new, beta @ np.abs(f_new)])
    new_endpoint = float(initial_memory + reconstructed[0])
    kkt_stationarity = joint_solution - requested + multiplier * np.ones(2)
    kkt_primal = max(float(joint_solution.sum() - total_capacity),
                     float(-joint_solution.min()), 0.0)
    kkt_complementarity = multiplier * (joint_solution.sum() - total_capacity)

    # Exactly two fixed orderings; this is not a profile search or a rollout.
    reverse_g = g[::-1]
    reverse_d = float(beta @ reverse_g)
    reversed_u = np.column_stack([f + reverse_g, f - reverse_g])

    assert abs(old_endpoint_residual) < 1e-12
    assert same_problem_error < 1e-9
    assert np.max(np.abs(u_new)) <= limit + 1e-12
    assert np.max(np.abs(reconstructed - joint_solution)) < 1e-12
    assert np.max(np.abs(kkt_stationarity)) < 1e-12

    result = {
        "date": "2026-10-16",
        "scope": "One declared local request projected through an exact actuator-memory/input relation using one fixed historical segment",
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": source_hash,
            "matlab_cell": "evaluation{1,1}",
            "case": "case3 linear_current_mpc",
            "start_index_zero_based": START,
            "K": K,
            "selection_rule": "first disturbance onset at t=0.9; fixed before numerical comparison; no segment search",
            "identity": "classical-feedback data, not PPO batch",
            "verification_this_run": provenance,
        },
        "segment": {
            "time": segment["t"],
            "x": x,
            "actual_u": actual_u,
            "f": f,
            "g": g,
            "actuator_difference": actuator_difference,
            "command": segment["command"],
            "disturbance": segment["w"],
        },
        "actuator_relation": {
            "U": limit,
            "dt": float(cfg["dt"]),
            "tau": tau,
            "rk4_substeps": substeps,
            "rk4_step": h,
            "alpha_rk": alpha,
            "alpha_continuous_zoh_not_used": float(np.exp(-cfg["dt"] / tau[0])),
            "beta": beta,
            "initial_memory_alphaK_a0": initial_memory,
            "recorded_endpoint": old_endpoint,
            "reconstructed_old_endpoint": initial_memory + old_d,
            "old_endpoint_identity_residual": old_endpoint_residual,
            "A": total_capacity,
            "old_d": old_d,
            "old_b": old_b,
            "fixed_b_positive_d_max": fixed_common_capacity,
        },
        "request": {
            "features": ["d = weighted differential input = aK - alpha^K a0", "b = weighted absolute common-mode input"],
            "target": requested,
            "planner_demand_origin": "declared fixed local request, not measured MPC request",
            "d_request_reason": "Explicit unit input d_req=1.0 to expose an active deadline/resource constraint; not fitted from or selected by task performance",
            "b_request_reason": "Keep the time-weighted forward input usage of the fixed existing segment",
            "objective": "0.5 * [(d - 1.0)^2 + (b - b0)^2]",
            "weights": [1.0, 1.0],
            "weights_selected_by_search": False,
        },
        "independent_endpoint_matching": {
            "solution": independent_solution,
            "objective": objective(independent_solution),
            "coupled_resource_violation": independent_gap,
            "meaning": "Algebraic reference omitting the deadline/resource relation; not a measured PPO step or a universal characterization of ordinary co-training",
        },
        "coupled_projection": {
            "constraint": "d + b <= A; d >= 0; b >= 0 (this positive-sector example)",
            "solution": joint_solution,
            "delta_from_request": joint_solution - requested,
            "objective": objective(joint_solution),
            "lagrange_multiplier": multiplier,
            "kkt_stationarity": kkt_stationarity,
            "kkt_primal_violation": kkt_primal,
            "kkt_complementarity": kkt_complementarity,
            "f_new_existence_witness": f_new,
            "g_new_existence_witness": g_new,
            "u_new_existence_witness": u_new,
            "reconstructed_features": reconstructed,
            "reconstruction_error": reconstructed - joint_solution,
            "predicted_endpoint_from_actuator_identity": new_endpoint,
            "input_peak": float(np.max(np.abs(u_new))),
            "minimum_input_margin": float(limit - np.max(np.abs(u_new))),
            "release_status": "not executed, not policy realizability proof, not published to MPC as learned capability",
        },
        "same_constraint_ordinary_solver": {
            "method": "SciPy SLSQP on the identical convex two-variable quadratic problem",
            "success": bool(qp.success),
            "solution": qp.x,
            "objective": objective(qp.x),
            "iterations": int(qp.nit),
            "maximum_difference_from_closed_form": same_problem_error,
            "interpretation": "Equivalent solutions; no solver novelty is claimed",
        },
        "timing_information_check": {
            "profiles": ["recorded g order", "recorded g reversed"],
            "recorded_g": g,
            "reversed_g": reverse_g,
            "unweighted_sum_original": float(g.sum()),
            "unweighted_sum_reversed": float(reverse_g.sum()),
            "weighted_d_original": old_d,
            "weighted_d_reversed": reverse_d,
            "difference": old_d - reverse_d,
            "reversed_profile_input_peak_with_same_f": float(np.max(np.abs(reversed_u))),
            "meaning": "Two algebraic actuator-memory moments; no second physical trajectory was simulated",
        },
        "budget": {
            "training_steps": 0,
            "learning_optimizer_steps": 0,
            "new_physical_transitions": 0,
            "new_mpc_calls": 0,
            "parameter_scan_candidates": 0,
            "capability_projection_solver_iterations": int(qp.nit),
            "fixed_timing_profiles": 2,
            "ppo_batch_replay": False,
        },
        "limitations": [
            "Demand is declared, not generated by MPC; a real PPO batch was not persisted and cannot be replayed.",
            "Constructed input vectors prove finite-segment actuator feasibility only; they are not a learned causal feedback policy.",
            "For stochastic feedback, jointly learned distributional response/resource statistics and current batch score estimates remain required.",
            "Positive-sector actuator relation does not establish whole-system nonlinear stability, full-task improvement, or recursive feasibility.",
            "The request projection can sacrifice forward input usage; benefit must later be assessed against actual task cost and matched baselines.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "witness.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=numbers) + "\n", encoding="utf-8"
    )

    rows = "\n".join(
        f"| {j} | {beta[j]:.12f} | {f[j]:.12f} | {g[j]:.12f} | {f_new[j]:.12f} | {g_new[j]:.12f} |"
        for j in range(K)
    )
    report = f"""# 固定历史片段上的不训练决策差异见证

日期：2026-10-16（Asia/Shanghai）。**这是一项代数见证：目标由本稿明确给定，数据来自既有经典反馈轨迹。它不是实际PPO批次重算，不是实际MPC需求提取，也没有执行构造的新动作。** [逐项数值与来源](witness.json)，[数据可用性审计](data-audit.md)，[可重运行脚本](../../scripts/joint_update_witness.py)。

## 同一个需求，增加控制关系后改变了什么

固定case3的linear_current_mpc轨迹首次扰动开始的`t=0.90→1.00 s`五步段。先定义两个同单位特征：`d=a_K−alpha^K a_0`是截至该段终点已形成的执行器差模记忆；`b=sum beta_j abs(f_j)`是带时序权重的前进共模输入占用。

声明的局部需求为`(d_req,b_req)=(1.0,{old_b:.15f})`，第二项保持已存片段的前进占用，第一项是明确的人为单元输入。这个值不来自已有MPC解、PPO advantage或数据选优。使用等权目标`0.5[(d−1)^2+(b−b0)^2]`，未搜索权重。这里把需求层输入固定，是为了单独检查实现关系会不会改变后续决策；不能据此声称已经验证完整学习闭环。

| 同一次局部需求更新 | d | b | 判定 |
|---|---:|---:|---|
| 忽略时序实现关系的独立目标匹配 | {independent_solution[0]:.12f} | {independent_solution[1]:.12f} | 超出共同资源界{independent_gap:.12f} |
| 固定原前进占用时的差模上界 | {fixed_common_capacity:.12f} | {old_b:.12f} | 原始d需求不可实现 |
| 加入精确deadline/输入关系后投影 | {d_new:.12f} | {b_new:.12f} | 降低前进占用并缩小差模需求，落在界上 |
| 同约束普通SLSQP/凸QP | {qp.x[0]:.12f} | {qp.x[1]:.12f} | 与闭式投影相同，最大差{same_problem_error:.3g} |

“独立目标匹配”是显式删除该关系的局部代数对照，不代表SPlaTES、ReCo或所有普通共同训练必然给出这一步。将同一关系加入普通约束优化后得到相同解是正确结果；**本例没有求解器创新。**

## 控制关系与精确投影

执行器时间常数相同，定义`a=(x5−x6)/2`、实际`f=(u1+u2)/2`、`g=(u1−u2)/2`。已保存轨迹使用RK4四子步，因此数值核验使用`alpha_RK={alpha:.16f}`，而不是连续ZOH指数。令`beta_j=(1−alpha)alpha^(K−1−j)`，则：

`d=sum beta_j g_j`，`A=U sum beta_j={total_capacity:.15f}`。

实际输入限额与共差模关系给出`abs(f_j)+abs(g_j)≤U`。因此`abs(d)+b≤A`；本次正扇区为`d+b≤A,d≥0,b≥0`。该关系同时连接末端记忆、动作时序和前进资源。

请求超出界的量为`gap={independent_gap:.15f}`。两个变量均为正的本例，闭式欧氏投影分别减去`gap/2={multiplier:.15f}`。KKT站立性残差最大`{np.max(np.abs(kkt_stationarity)):.3g}`，原始可行性违反`{kkt_primal:.3g}`，互补残差`{kkt_complementarity:.3g}`。

给定投影后的`(d*,b*)`，下面构造仅证明这两个特征存在满足输入界的实现：`f_new=(b*/b0)f_old`，`m_j=U−abs(f_new_j)`，`g_new_j=[d*/(A−b*)]m_j`。

| j | beta | 原f | 原g | 构造f_new | 构造g_new |
|---:|---:|---:|---:|---:|---:|
{rows}

构造输入为`u1=f_new+g_new,u2=f_new−g_new`，最大幅值`{np.max(np.abs(u_new)):.15f}≤U={limit:g}`。重建特征误差最大`{np.max(np.abs(reconstructed-joint_solution)):.3g}`，对应差模终端状态的代数预测为`{new_endpoint:.15f}`。**构造输入未执行，不能据此判断PPO能学出这一时序反馈；它也不是已发布给MPC的能力。**

## 为什么不能只记录动作总量

只比较两种事先固定的时序：已记录g与它的倒序。二者非加权总和都是`{g.sum():.15f}`，但执行器记忆增量分别为`{old_d:.15f}`和`{reverse_d:.15f}`，相差`{old_d-reverse_d:.15f}`。同一f下倒序输入最大幅值为`{np.max(np.abs(reversed_u)):.15f}`，也在输入界内。

这只计算两个固定向量的加权矩，没有第二条新物理轨迹或时序扫描。它说明该有限执行器关系所需的时间信息会被非加权平均/总量丢失；若普通共同训练已经建模全部相关时序，其增量须另行判断。

## 对既有记录的核验与边界

记录的差模终点为`{old_endpoint:.15f}`，用实际输入代入执行器公式得到`{initial_memory+old_d:.15f}`，残差`{old_endpoint_residual:.3g}`。公开的[五步原始片段](source-segment.json)记录MAT来源SHA256为`{source_hash}`。本次运行模式为`{args.source}`，`original_mat_verified_this_run={str(provenance['original_mat_verified_this_run']).lower()}`；原始MAT hash只有在`--source mat`模式才会重新核验，默认excerpt模式只重算公开片段。片段文件本次SHA256为`{provenance['excerpt_sha256']}`。这个核验只支持当前轨迹与该关系一致，不支持新策略任务收益。

本例已经说明：在相同需求下，模型/能力目标不能独立向更好的数值移动，必须同时遵守动作时序和前进资源之间的实现关系；保持原资源时某些要求应被拒绝，允许资源重新分配时可保留更多差模需求。它尚未显示真正的PPO更新差异，因为原训练批次没有持久化；也未验证任务如何自动选出这个需求、因果策略能否实现、分布估计精度或滚动任务收益。

本轮资源：新增物理转移0，MPC调用0，训练步0，学习优化器步0；仅一次二维能力投影的普通数值求解（{qp.nit}次迭代）和两个固定时序向量的代数比较。没有新仿真、反馈增益扫描或换系统。

公开复查：在项目根目录运行`& '.venv/Scripts/python.exe' 'scripts/joint_update_witness.py' --source excerpt`（默认模式），只需公开的五步片段及NumPy/SciPy，无需忽略目录中的MAT。使用本机原始数据复核时运行同一命令并指定`--source mat`；该模式先核原始MAT hash，并逐字段确认提取片段与公开片段完全一致，再重算同一关系。两种模式均输出本报告与JSON，不导入训练器或物理仿真入口。JSON分开记录本次原MAT验证状态、片段hash与历史来源hash。
"""
    (OUTPUT / "witness.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT / "witness.json"),
        "source_mode": args.source,
        "original_mat_verified_this_run": provenance["original_mat_verified_this_run"],
        "excerpt_sha256": provenance["excerpt_sha256"],
        "request": requested,
        "independent_violation": independent_gap,
        "joint_solution": joint_solution,
        "same_problem_solver_difference": same_problem_error,
        "input_peak": np.max(np.abs(u_new)),
        "old_endpoint_residual": old_endpoint_residual,
        "timing_moment_difference": old_d - reverse_d,
        "new_physical_transitions": 0,
        "training_steps": 0,
        "ppo_batch_replay": False,
    }, ensure_ascii=False, default=numbers))


if __name__ == "__main__":
    main()
