"""Independent direct optimization of the existing two-sign allocation family.

No project evaluator, actor, environment, or joint-core module is imported.
The decision variable r is an analytic feedback-family coordinate, not weights.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import minimize_scalar


ARITHMETIC_TOLERANCE = 1e-10
OPTIMIZER_XATOL = 1e-12
OPTIMIZER_MAXITER = 200


@dataclass
class EvaluationLedger:
    family_evaluations: int = 0
    signed_case_evaluations: int = 0
    exact_polynomial_phase_maps: int = 0
    prefix_phase_checks: int = 0
    scalar_extremum_position_evaluations: int = 0
    gate_equation_solves: int = 0
    within_phase_gate_maps: int = 0
    terminal_case_checks: int = 0


def family_bounds() -> dict[str, float | str]:
    """Derive the domain from lateral motion and nonnegative suffix times."""
    # y1 = 1/(2 sqrt(1-r^2)) <= 1.
    clearance = math.sqrt(1.0 - (1.0 / 2.0) ** 2)
    # x1 + vx1^2/2 = r/[2(1-r)] <= L=4.
    nonnegative_suffix = (2.0 * 4.0) / (1.0 + 2.0 * 4.0)
    return {
        "lower": 0.0,
        "upper": min(clearance, nonnegative_suffix),
        "clearance_upper": clearance,
        "nonnegative_suffix_upper": nonnegative_suffix,
        "domain_scope": "constant allocation with non-overshooting forward/brake suffix",
        "not_a_task_constraint": "No global x(t)<=L constraint is added to the task.",
    }


def _position_extrema(
    p: np.ndarray, v: np.ndarray, u: np.ndarray, duration: float,
    ledger: EvaluationLedger,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = [], []
    for axis in range(2):
        times = [0.0, duration]
        if u[axis] != 0.0:
            stationary = -v[axis] / u[axis]
            if 0.0 < stationary < duration:
                times.append(float(stationary))
        values = [float(p[axis] + v[axis] * t + 0.5 * u[axis] * t * t) for t in times]
        ledger.scalar_extremum_position_evaluations += len(times)
        lower.append(min(values))
        upper.append(max(values))
    return np.asarray(lower), np.asarray(upper)


def _gate_roots(p_x: float, v_x: float, u_x: float, duration: float) -> list[float]:
    """Roots of p_x+v_x*t+u_x*t^2/2=2 on this phase."""
    offset = p_x - 2.0
    if u_x == 0.0:
        roots = [] if v_x == 0.0 else [-offset / v_x]
    else:
        discriminant = v_x * v_x - 2.0 * u_x * offset
        if discriminant < -ARITHMETIC_TOLERANCE:
            roots = []
        else:
            radical = math.sqrt(max(0.0, discriminant))
            roots = [(-v_x - radical) / u_x, (-v_x + radical) / u_x]
    return sorted({min(duration, max(0.0, float(t))) for t in roots
                   if -ARITHMETIC_TOLERANCE <= t <= duration + ARITHMETIC_TOLERANCE})


def _phase(
    name: str, start_time: float, p: np.ndarray, v: np.ndarray,
    u: np.ndarray, duration: float, ledger: EvaluationLedger,
) -> dict:
    ledger.exact_polynomial_phase_maps += 1
    p_next = p + v * duration + 0.5 * u * duration * duration
    v_next = v + u * duration
    p_min, p_max = _position_extrema(p, v, u, duration, ledger)
    ledger.prefix_phase_checks += 1
    ledger.gate_equation_solves += 1
    gate_events = []
    for local_t in _gate_roots(float(p[0]), float(v[0]), float(u[0]), duration):
        ledger.within_phase_gate_maps += 1
        event_p = p + v * local_t + 0.5 * u * local_t * local_t
        event_v = v + u * local_t
        gate_events.append({
            "time": start_time + local_t,
            "position": event_p.tolist(), "velocity": event_v.tolist(),
        })
    return {
        "phase": name, "start_time": start_time, "duration": duration,
        "position_start": p.tolist(), "velocity_start": v.tolist(),
        "acceleration": u.tolist(), "position_end": p_next.tolist(),
        "velocity_end": v_next.tolist(), "position_min": p_min.tolist(),
        "position_max": p_max.tolist(), "input_norm": float(np.linalg.norm(u)),
        "gate_events": gate_events,
    }


def _evaluate_sign(r: float, sign: int, ledger: EvaluationLedger) -> dict:
    ledger.signed_case_evaluations += 1
    s = math.sqrt(1.0 - r * r)
    p, v = np.asarray([0.0, 0.0]), np.asarray([0.0, float(sign)])
    recovery = _phase(
        "observed_sign_recovery", 0.0, p, v,
        np.asarray([r, -sign * s]), 1.0 / s, ledger,
    )
    phases = [recovery]
    p = np.asarray(recovery["position_end"])
    v = np.asarray(recovery["velocity_end"])
    # Solve the two suffix phase equations, not the simplified total-time T(r).
    radical_squared = 4.0 - float(p[0]) + 0.5 * float(v[0]) ** 2
    if radical_squared < 0.0:
        return {"r": r, "sign": sign, "status": "UNSUPPORTED_SUFFIX",
                "reason": "negative suffix radical", "phases": phases}
    peak = math.sqrt(radical_squared)
    accel_duration = peak - float(v[0])
    if accel_duration < -ARITHMETIC_TOLERANCE:
        return {"r": r, "sign": sign, "status": "UNSUPPORTED_SUFFIX",
                "reason": "outside the nonnegative acceleration suffix subfamily",
                "phases": phases}
    # Only remove floating-point negative zero at the subfamily boundary.
    accel_duration = max(0.0, accel_duration)
    accelerate = _phase(
        "forward", recovery["duration"], p, v,
        np.asarray([1.0, 0.0]), accel_duration, ledger,
    )
    phases.append(accelerate)
    p, v = np.asarray(accelerate["position_end"]), np.asarray(accelerate["velocity_end"])
    brake = _phase(
        "brake", recovery["duration"] + accel_duration, p, v,
        np.asarray([-1.0, 0.0]), float(v[0]), ledger,
    )
    phases.append(brake)
    duration = sum(phase["duration"] for phase in phases)
    final_p, final_v = np.asarray(brake["position_end"]), np.asarray(brake["velocity_end"])
    lateral_occupancy = max(max(abs(phase["position_min"][1]), abs(phase["position_max"][1]))
                            for phase in phases)
    max_input = max(phase["input_norm"] for phase in phases)
    gate_events = []
    for phase in phases:
        for event in phase["gate_events"]:
            if not gate_events or abs(event["time"] - gate_events[-1]["time"]) > ARITHMETIC_TOLERANCE:
                gate_events.append(event)
    residuals = {
        "clearance_excess": lateral_occupancy - 1.0,
        "input_norm_excess": max_input - 1.0,
        "terminal_x_error": float(final_p[0] - 4.0),
        "terminal_velocity_norm": float(np.linalg.norm(final_v)),
    }
    ledger.terminal_case_checks += 1
    checks = {
        "whole_prefix_clearance": residuals["clearance_excess"] <= ARITHMETIC_TOLERANCE,
        "whole_prefix_input_ball": residuals["input_norm_excess"] <= ARITHMETIC_TOLERANCE,
        "terminal_x": abs(residuals["terminal_x_error"]) <= ARITHMETIC_TOLERANCE,
        "terminal_velocity": residuals["terminal_velocity_norm"] <= ARITHMETIC_TOLERANCE,
        "gate_crossed": bool(gate_events),
    }
    return {
        "r": r, "sign": sign, "status": "COMPLETE" if all(checks.values()) else "CONSTRAINT_VIOLATION",
        "duration": duration, "recovery_duration": recovery["duration"],
        "lateral_occupancy": lateral_occupancy, "max_input_norm": max_input,
        "terminal_position": final_p.tolist(), "terminal_velocity": final_v.tolist(),
        "squared_acceleration_integral": sum(phase["input_norm"] ** 2 * phase["duration"] for phase in phases),
        "residuals": residuals, "checks": checks, "gate_events": gate_events,
        "subfamily_diagnostic_not_task_constraint": {
            "maximum_x": max(phase["position_max"][0] for phase in phases),
            "minimum_x_velocity": min(min(phase["velocity_start"][0], phase["velocity_end"][0]) for phase in phases),
        },
        "phases": phases,
    }


def evaluate_family(r: float, ledger: EvaluationLedger | None = None) -> dict:
    """Evaluate both observed signs; unsupported continuations get no fake cost."""
    ledger = ledger if ledger is not None else EvaluationLedger()
    ledger.family_evaluations += 1
    r = float(r)
    if not math.isfinite(r) or not 0.0 <= r < 1.0:
        return {"r": r, "status": "OUTSIDE_ALLOCATION_FAMILY", "cases": [],
                "complete_case_count": 0, "total_case_count": 2,
                "mean_duration": None}
    cases = [_evaluate_sign(r, sign, ledger) for sign in (-1, 1)]
    complete = sum(case["status"] == "COMPLETE" for case in cases)
    # A failed/unsupported continuation is not silently removed from an average.
    mean = sum(case["duration"] for case in cases) / 2.0 if complete == 2 else None
    return {
        "r": r, "status": "COMPLETE" if complete == 2 else "NOT_ALL_COMPLETE",
        "complete_case_count": complete, "total_case_count": 2,
        "mean_duration": mean, "cases": cases,
    }


def solve_direct() -> dict:
    """Numerically solve the same constant-allocation family, then check endpoints."""
    started = time.perf_counter()
    ledger = EvaluationLedger()
    bounds = family_bounds()
    evaluations: list[dict] = []
    objective_records: list[dict] = []

    def objective(r: float) -> float:
        result = evaluate_family(r, ledger)
        if result["status"] != "COMPLETE":
            raise ArithmeticError("The independently derived optimizer interval produced an invalid continuation.")
        evaluations.append(result)
        objective_records.append({
            "call": len(objective_records) + 1, "r": float(r),
            "mean_duration": result["mean_duration"],
            "max_clearance_excess": max(case["residuals"]["clearance_excess"] for case in result["cases"]),
            "complete_case_count": result["complete_case_count"],
        })
        return float(result["mean_duration"])

    solution = minimize_scalar(
        objective, bounds=(float(bounds["lower"]), float(bounds["upper"])),
        method="bounded", options={"xatol": OPTIMIZER_XATOL, "maxiter": OPTIMIZER_MAXITER},
    )
    raw_evaluation = next(item for item in evaluations if item["r"] == float(solution.x))
    endpoint_evaluations = [evaluate_family(float(bounds[key]), ledger) for key in ("lower", "upper")]
    eligible = [raw_evaluation] + [item for item in endpoint_evaluations if item["status"] == "COMPLETE"]
    best = min(eligible, key=lambda item: item["mean_duration"])
    script_bytes = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return {
        "status": "COMPLETED" if solution.success else "OPTIMIZER_DID_NOT_CONVERGE",
        "task": {"a": 1.0, "v0": 1.0, "b": 1.0, "L": 4.0, "gate_x": 2.0, "observed_signs": [-1, 1]},
        "bounds": bounds,
        "scope": "Independent same-family scalar direct optimization; no general time-varying TO or robot claim.",
        "code_identity": {"lf_canonical_sha256": hashlib.sha256(script_bytes).hexdigest(), "scipy_version": scipy.__version__},
        "optimizer": {
            "method": "scipy.optimize.minimize_scalar/bounded", "xatol": OPTIMIZER_XATOL,
            "maxiter": OPTIMIZER_MAXITER, "success": bool(solution.success),
            "message": str(solution.message), "nfev": int(solution.nfev), "nit": int(solution.nit),
            "raw_r": float(solution.x), "raw_objective": float(solution.fun),
            "objective_records": objective_records,
        },
        "raw_optimizer_evaluation": raw_evaluation,
        "endpoint_evaluations": endpoint_evaluations,
        "best_closed_interval_evaluation": best,
        "best_source": "optimizer" if best is raw_evaluation else "explicit_endpoint_check",
        "budget": {
            **asdict(ledger), "numerical_optimizer_invocations": 1,
            "optimizer_objective_calls": len(objective_records), "explicit_endpoint_evaluations": 2,
            "wall_seconds": time.perf_counter() - started,
            "neural_actor_calls": 0, "environment_steps": 0, "PPO_batches": 0,
            "training_updates": 0, "weight_writes": 0, "checkpoint_reads": 0,
        },
        "interpretation": {
            "same_wall_or_total_budget_comparison": False,
            "independent_joint_algorithm_advantage_established": False,
            "exact_polynomial_maps_are_model_computation": True,
            "macro_command_caveat": "If an actual frozen controller's command can choose r, it must be allowed to use this allocation too.",
        },
    }


def direct_anchor_gradient_and_step() -> dict:
    """Right derivative at r=0 from phase equations, with no root/packet import."""
    # s(0)=1, s'(0)=0. Recovery endpoint derivatives follow directly
    # from t_y=1/s and constant-acceleration position/velocity equations.
    tau_derivative = 0.0
    x_recovery_derivative = 0.5
    vx_recovery_derivative = 1.0
    y_magnitude_derivative = 0.0
    peak_at_anchor = math.sqrt(4.0)
    peak_derivative = -x_recovery_derivative / (2.0 * peak_at_anchor)
    accelerate_derivative = peak_derivative - vx_recovery_derivative
    brake_derivative = peak_derivative
    direct_gradient = tau_derivative + accelerate_derivative + brake_derivative
    metric, trust_radius = 1.0, 0.1
    # Exact minimizer of g*d + d^2/2 on [0,0.1]. The local clearance
    # and input rows have zero gradients and values -0.5 and 0 here.
    step = min(trust_radius, max(0.0, -direct_gradient / metric))
    return {
        "anchor_r": 0.0, "derivative_kind": "right derivative of independently constructed phase cost",
        "response_derivatives": {"recovery_time": tau_derivative, "x_end": x_recovery_derivative,
                                 "vx_end": vx_recovery_derivative, "abs_y_end": y_magnitude_derivative},
        "peak_velocity_derivative": peak_derivative,
        "accelerate_duration_derivative": accelerate_derivative,
        "brake_duration_derivative": brake_derivative,
        "direct_task_gradient": direct_gradient,
        "local_metric": metric, "trust_radius": trust_radius,
        "local_constraint_rows": [{"value": -0.5, "gradient": 0.0}, {"value": 0.0, "gradient": 0.0}],
        "subproblem": "min g*d + d*d/2 subject to 0<=d<=0.1 and unchanged local physical rows",
        "closed_form_step": step,
        "finite_difference_evaluations": 0,
        "not_neural_gradient_or_PPO_KL": True,
    }


def compare_saved_round(report_path: str | Path) -> dict:
    """One fixed-r evaluation and independent local-QP comparison, no TO rerun."""
    path = Path(report_path)
    report_bytes = path.read_bytes()
    root_report = json.loads(report_bytes)
    supported = root_report["rounds"]["supported"]
    if supported["anchor_input"]["r"] != 0.0 or supported["accepted_feedback"]["r"] != 0.1:
        raise ValueError("This requested audit is specifically the recorded r=0 to r=0.1 round.")
    accepted = next(attempt for attempt in supported["attempts"] if attempt["accepted"])
    ledger = EvaluationLedger()
    independently_evaluated = evaluate_family(0.1, ledger)
    independent_by_sign = {case["sign"]: case for case in independently_evaluated["cases"]}
    plans_by_id = {item["case_id"]: item["plan"] for item in supported["next_planning_decisions"]}
    paired_by_id = {item["case_id"]: item for item in supported["active_response_pair"]["predictions"]}
    rows = []

    def maximum_error(left, right) -> float:
        return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))

    for recorded in accepted["outcomes"]:
        independent = independent_by_sign[recorded["sign"]]
        recovery, accelerate, brake = independent["phases"]
        response = [recovery["duration"], recovery["position_end"][0],
                    recovery["velocity_end"][0], abs(recovery["position_end"][1])]
        independent_plan = {
            "accelerate_duration": accelerate["duration"], "brake_duration": brake["duration"],
            "brake_position": brake["position_start"][0], "brake_start_time": brake["start_time"],
            "completion_time": independent["duration"],
            "lateral_endpoint_magnitude": abs(independent["terminal_position"][1]),
        }
        root_plans = {
            "case_suffix": recorded["details"]["suffix_plan"],
            "paired_model": paired_by_id[recorded["case_id"]]["plan"],
            "next_macro_plan": plans_by_id[recorded["case_id"]],
        }
        plan_errors = {name: {key: abs(float(plan[key]) - value) for key, value in independent_plan.items()}
                       for name, plan in root_plans.items()}
        phase_errors = []
        for root_phase, direct_phase in zip(recorded["details"]["phases"], independent["phases"]):
            phase_errors.extend([
                abs(root_phase["duration"] - direct_phase["duration"]),
                maximum_error(root_phase["position_end"], direct_phase["position_end"]),
                maximum_error(root_phase["velocity_end"], direct_phase["velocity_end"]),
                maximum_error(root_phase["input"], direct_phase["acceleration"]),
            ])
        errors = {
            "task_time": abs(recorded["task_time"] - independent["duration"]),
            "response": maximum_error(recorded["response"], response),
            "paired_response": maximum_error(paired_by_id[recorded["case_id"]]["response"], response),
            "terminal_position": maximum_error(recorded["details"]["terminal_position"], independent["terminal_position"]),
            "terminal_velocity": maximum_error(recorded["details"]["terminal_velocity"], independent["terminal_velocity"]),
            "clearance_constraint": abs(recorded["constraints"]["clearance"] - independent["residuals"]["clearance_excess"]),
            "input_constraint": abs(recorded["constraints"]["input_norm"] - independent["residuals"]["input_norm_excess"]),
            "phase_data": max(phase_errors),
            "plan": max(error for plan_error in plan_errors.values() for error in plan_error.values()),
        }
        rows.append({
            "case_id": recorded["case_id"], "sign": recorded["sign"],
            "root_status": recorded["status"], "independent_status": independent["status"],
            "independent_response": response, "independent_plan": independent_plan,
            "absolute_errors": errors, "plan_absolute_errors": plan_errors,
            "passed": recorded["status"] == "COMPLETED" and independent["status"] == "COMPLETE"
                      and all(independent["checks"].values()) and max(errors.values()) <= ARITHMETIC_TOLERANCE,
        })
    direct_local = direct_anchor_gradient_and_step()
    proposal = supported["proposal"]
    local_comparison = {
        "root_gradient": proposal["gradient"][0], "direct_gradient": direct_local["direct_task_gradient"],
        "gradient_absolute_error": abs(proposal["gradient"][0] - direct_local["direct_task_gradient"]),
        "root_step": proposal["step"][0], "direct_closed_form_step": direct_local["closed_form_step"],
        "step_absolute_error": abs(proposal["step"][0] - direct_local["closed_form_step"]),
        "root_metric_description": proposal["metric"],
        "root_trust_radius": proposal["trust_radius"], "direct_trust_radius": direct_local["trust_radius"],
        "same_information_local_step_equivalent": proposal["metric"] == "identity on analytic coordinate, not empirical PPO Fisher"
          and proposal["trust_radius"] == direct_local["trust_radius"]
          and abs(proposal["step"][0] - direct_local["closed_form_step"]) <= ARITHMETIC_TOLERANCE
          and abs(proposal["gradient"][0] - direct_local["direct_task_gradient"]) <= ARITHMETIC_TOLERANCE,
    }
    return {
        "status": "PASS" if all(row["passed"] for row in rows) and local_comparison["same_information_local_step_equivalent"] else "FAIL",
        "source_report": str(path),
        "source_report_lf_canonical_sha256": hashlib.sha256(report_bytes.replace(b"\r\n", b"\n")).hexdigest(),
        "script_lf_canonical_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "fixed_coordinate_evaluation": independently_evaluated,
        "comparisons": rows, "independent_direct_derivation": direct_local,
        "same_information_finite_step_comparison": local_comparison,
        "incremental_budget": {
            **asdict(ledger), "numerical_optimizer_invocations": 0,
            "analytic_direct_gradient_derivations": 1, "closed_form_scalar_qp_solves": 1,
            "finite_difference_task_evaluations": 0,
            "neural_actor_calls": 0, "environment_steps": 0, "PPO_batches": 0,
            "training_updates": 0, "weight_writes": 0,
        },
        "interpretation": "Same-information direct task differentiation gives the same finite local step; no independent joint-method gain established.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-r", type=float, help="Independently evaluate one joint-core proposal; not optimize it again.")
    parser.add_argument("--accepted-r", type=float, help="Independently evaluate one accepted joint-core coordinate.")
    parser.add_argument("--compare-round-json", type=Path, help="Audit the recorded r=0 to r=0.1 round without rerunning TO.")
    args = parser.parse_args()
    if args.compare_round_json is not None:
        print(json.dumps(compare_saved_round(args.compare_round_json), indent=2, ensure_ascii=False, allow_nan=False))
        return
    result = solve_direct()
    if args.proposal_r is not None or args.accepted_r is not None:
        ledger = EvaluationLedger()
        comparison = {}
        for name, r in (("proposal", args.proposal_r), ("accepted", args.accepted_r)):
            if r is not None:
                comparison[name] = evaluate_family(r, ledger)
        result["supplied_joint_coordinates"] = comparison
        result["supplied_coordinate_evaluation_budget"] = asdict(ledger)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
