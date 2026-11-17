#!/usr/bin/env python3
"""Finite W54 C full-horizon GPU solve with independent CPU recomputation.

Every entry belongs to a frozen manifest. No new physics, native rollout,
derivatives, learning, or online controller installation is performed here.
"""
import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import osqp
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
REL = Path("terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
KEYS = ("stage_A", "stage_B", "stage_c", "x0", "reference", "last_force",
        "contact", "feet", "force_normal_upper")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu_recompute(p, forces, states, cfg):
    """Independent NumPy recurrence, objective and all SRB force inequalities."""
    n = len(forces)
    truth = np.empty_like(states)
    truth[0] = p["x0"]
    stage_residual = np.empty(n)
    for k in range(n):
        truth[k+1] = (p["stage_A"][k] @ truth[k] +
                      p["stage_B"][k] @ forces[k].reshape(-1) + p["stage_c"][k])
        stage_residual[k] = np.max(np.abs(states[k+1] - (
            p["stage_A"][k] @ states[k] + p["stage_B"][k] @ forces[k].reshape(-1) + p["stage_c"][k])))
    delta = truth - p["reference"]
    weights = np.tile(np.asarray(cfg["mpc"]["state_weights"], dtype=float), (n+1, 1))
    weights[-1] *= 2.
    differences = np.diff(np.concatenate((p["last_force"][None], forces)), axis=0)
    terms = dict(state=float(np.sum(weights * delta * delta)),
                 force=float(cfg["mpc"]["force_regularization"] * np.sum(forces * forces)),
                 force_difference=float(cfg["mpc"]["force_difference_weight"] * np.sum(differences * differences)))
    fz = forces[..., 2]
    mu = cfg["friction_coefficient"]
    violations = dict(
        unilateral_N=float(max(0., -np.min(fz))),
        normal_upper_N=float(max(0., np.max(fz - p["force_normal_upper"]))),
        friction_pyramid_N=float(max(0., np.max(np.abs(forces[..., :2]) - mu*fz[..., None]))),
        inactive_force_N=float(np.max(np.abs(forces * (~p["contact"])[..., None]))),
        dynamics=float(max(np.max(np.abs(states[0]-p["x0"])), np.max(stage_residual))),
        independent_trajectory=float(np.max(np.abs(states-truth))))
    return dict(objective=sum(terms.values()), terms=terms, violations=violations,
                SRB_feasible=all(v <= 1e-6 for v in violations.values()),
                articulated_WBC_evaluated=False, material_task_evaluated=False), truth


def cpu_qp(p, cfg, assemble):
    start = time.perf_counter()
    q = assemble(p, cfg)
    assembly_s = time.perf_counter() - start
    solver = osqp.OSQP()
    start = time.perf_counter()
    solver.setup(P=sparse.triu(sparse.csc_matrix(q["P"]), format="csc"), q=q["q"],
                 A=sparse.csc_matrix(q["A"]), l=q["lower"], u=q["upper"],
                 verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=20000,
                 polishing=True, adaptive_rho=True, adaptive_rho_interval=50,
                 check_termination=25)
    result = solver.solve(raise_error=False)
    solve_s = time.perf_counter() - start
    report = dict(status=result.info.status, status_value=int(result.info.status_val),
                  iterations=int(result.info.iter), setup_and_solve_wall_s=solve_s,
                  assembly_wall_s=assembly_s, primal_residual=float(result.info.prim_res),
                  dual_residual=float(result.info.dual_res))
    if result.info.status_val not in (1, 2):
        return report, None, None
    n = len(p["contact"])
    nx = (n+1)*12
    states = result.x[:nx].reshape(n+1, 12)
    forces = result.x[nx:].reshape(n, 4, 3)
    report.update(cpu_recompute(p, forces, states, cfg)[0])
    return report, states, forces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pass-id", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cfg = json.loads(Path(manifest["configuration"]).read_text())
    out = ROOT/"outputs"/REL/"w54-C"/args.pass_id
    out.mkdir(parents=True, exist_ok=False)
    compact = ROOT/"results"/REL/"w54"/("task-c-"+args.pass_id+".json")
    report = dict(schema="w54_complete_gpu_force_horizon_v1", pass_id=args.pass_id,
        status="running", started_unix_s=time.time(), pid=os.getpid(),
        deadline=None, wall_deadline_enabled=False, kill_on_conversation_end=False,
        input_manifest=str(args.manifest), input_manifest_sha256=sha(args.manifest),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/"scripts/s1_w54_gpu_force.py",ROOT/"scripts/w54_gpu_force_horizon.py",
            ROOT/"scripts/s1_w54_f_backend.py"]},
        configuration_sha256=sha(manifest["configuration"]),
        method="JAX FP64 4096 fixed accelerated projected-gradient steps; exact Euclidean force-pyramid projection",
        force_parameterization="every knot x FL/FR/RL/RR xyz; 14x12 independent force variables",
        CPU_reference="same F full-state/per-knot-force convex QP, OSQP FP64, eps_abs/rel 1e-7",
        equal_optimization_budget=False,
        contact_candidate="one corrected causal nominal continuation per saved state",
        missing_records=[r for r in manifest["records"] if r["status"] != "ready"],
        new_physics_steps=0, new_native_rollout=0, new_FD=0,
        new_PPO_samples=0, new_PPO_updates=0, donor=0,
        input_preparation_timing=manifest.get("preparation_timing"),
        online_installed=False, full_articulated_WBC_coverage=False,
        material_dynamics_coverage=False, cases=[])
    write(compact, report)
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    wall0 = time.perf_counter()
    records = [r for r in manifest["records"] if r["status"] == "ready"]
    load0 = time.perf_counter()
    problems = []
    for r in records:
        with np.load(r["input_path"], allow_pickle=False) as z:
            problems.append({k:z[k].copy() for k in KEYS})
    timing = dict(input_read_s=time.perf_counter()-load0)
    from s1_w54_f_backend import assemble_force_horizon
    init0 = time.perf_counter()
    import jax
    import jax.numpy as jnp
    from s1_w54_gpu_force import solve_horizon, ITERATIONS
    device = next(d for d in jax.devices() if d.platform == "gpu")
    report["GPU_device"] = dict(id=device.id, platform=device.platform, kind=device.device_kind)
    timing["GPU_runtime_initialization_s"] = time.perf_counter()-init0
    transfer0 = time.perf_counter()
    arrays = [np.stack([p[k] for p in problems]) for k in
              ("stage_A","stage_B","stage_c","x0","reference","last_force","contact","force_normal_upper")]
    arguments = [jax.device_put(a, device) for a in arrays]
    arguments += [jax.device_put(np.asarray(cfg["mpc"]["state_weights"], dtype=float), device),
                  jax.device_put(np.asarray(cfg["mpc"]["force_regularization"]), device),
                  jax.device_put(np.asarray(cfg["mpc"]["force_difference_weight"]), device),
                  jax.device_put(np.asarray(cfg["friction_coefficient"]), device)]
    jax.block_until_ready(arguments)
    timing["host_to_device_s"] = time.perf_counter()-transfer0
    compile0 = time.perf_counter()
    executable = solve_horizon.lower(*arguments, iterations=ITERATIONS).compile()
    timing["JIT_compile_s"] = time.perf_counter()-compile0
    solve0 = time.perf_counter()
    output = executable(*arguments)
    jax.block_until_ready(output)
    timing["warm_full_horizon_solve_and_synchronize_s"] = time.perf_counter()-solve0
    download0 = time.perf_counter()
    actual = {k:np.asarray(jax.device_get(v)) for k,v in output.items()}
    timing["device_to_host_s"] = time.perf_counter()-download0
    report["GPU_memory_stats"] = device.memory_stats()
    timing["CPU_independent_recompute_and_QP_s"] = 0.
    timing["artifact_serialization_s"] = 0.
    for i,(record,p) in enumerate(zip(records,problems)):
        check0 = time.perf_counter()
        independently, truth = cpu_recompute(p,actual["forces"][i],actual["states"][i],cfg)
        reference, cpu_states, cpu_forces = cpu_qp(p,cfg,assemble_force_horizon)
        independently["GPU_objective_abs_error"] = abs(independently["objective"]-float(actual["objective"][i]))
        row = dict(source=record, GPU=independently, CPU_QP=reference,
                   projected_gradient_inf=float(actual["projected_gradient_inf"][i]),
                   GPU_objective=float(actual["objective"][i]),
                   first_force_N=actual["first_force"][i].tolist())
        if cpu_states is not None:
            row["GPU_minus_CPU_objective"] = independently["objective"]-reference["objective"]
            row["relative_objective_gap"] = row["GPU_minus_CPU_objective"]/max(1.,abs(reference["objective"]))
            row["first_force_max_abs_difference_N"] = float(np.max(np.abs(actual["first_force"][i]-cpu_forces[0])))
        timing["CPU_independent_recompute_and_QP_s"] += time.perf_counter()-check0
        save0 = time.perf_counter()
        path = out/("case-%04d.npz"%i)
        saved = dict(**p, GPU_forces=actual["forces"][i], GPU_states=actual["states"][i],
                     CPU_independent_states=truth, GPU_first_force=actual["first_force"][i],
                     GPU_objective=actual["objective"][i])
        if cpu_states is not None: saved.update(CPU_QP_states=cpu_states,CPU_QP_forces=cpu_forces)
        np.savez_compressed(path, **saved)
        row["artifact"] = dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
        timing["artifact_serialization_s"] += time.perf_counter()-save0
        report["cases"].append(row)
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    timing["total_process_wall_s"] = time.perf_counter()-wall0
    timing["user_cpu_s"] = usage1.ru_utime-usage0.ru_utime
    timing["system_cpu_s"] = usage1.ru_stime-usage0.ru_stime
    timing["GPU_path_including_initialization_compile_transfer_s"] = sum(timing[k] for k in (
        "GPU_runtime_initialization_s","host_to_device_s","JIT_compile_s",
        "warm_full_horizon_solve_and_synchronize_s","device_to_host_s"))
    timing["GPU_path_without_initialization_compile_s"] = sum(timing[k] for k in (
        "host_to_device_s","warm_full_horizon_solve_and_synchronize_s","device_to_host_s"))
    gaps=[r["relative_objective_gap"] for r in report["cases"] if "relative_objective_gap" in r]
    report.update(status="completed",ended_unix_s=time.time(),timing=timing,
        complete_horizons=len(records),force_knots=len(records)*14,
        SRB_feasible_horizons=sum(r["GPU"]["SRB_feasible"] for r in report["cases"]),
        max_relative_objective_gap=max(gaps) if gaps else None,
        max_dynamics_recompute_error=max(r["GPU"]["violations"]["independent_trajectory"] for r in report["cases"]),
        max_objective_recompute_error=max(r["GPU"]["GPU_objective_abs_error"] for r in report["cases"]),
        coverage=dict(GPU=["full_horizon_condensing","full_objective_and_force_difference",
                           "all_per_knot_force_variables","exact_contact_friction_unilateral_projection",
                           "4096_optimization_iterations","full_state_force_output","first_action","objective"],
                      CPU=["causal_input_reconstruction","F_model_assembly","independent_dynamics_cost_constraint_recompute",
                           "same_problem_OSQP_quality_reference"],
                      not_covered=["articulated_WBC_GPU","material_predictor_GPU","real_plant_GPU","online_controller"]))
    report["CPU_reference_timing"]={key:sum(row["CPU_QP"][key] for row in report["cases"])
        for key in ("assembly_wall_s","setup_and_solve_wall_s")}
    report["exercised_contact_patterns"]={str(list(pattern)):int(count) for pattern,count in zip(
        *np.unique(np.concatenate([p["contact"] for p in problems]),axis=0,return_counts=True))}
    report["contact_switches_in_horizons"]=int(sum(np.count_nonzero(np.diff(p["contact"].astype(int),axis=0)) for p in problems))
    report["future_joint_trajectory_available"]=False
    report["speed_interpretation"]="Batched offline quality/time comparison, not equal-iteration budgets or established online speedup; causal input preparation is common CPU work and reported separately."
    write(out/"result.json",report);write(compact,report)
    print(json.dumps({k:report[k] for k in ("status","complete_horizons","SRB_feasible_horizons",
        "max_relative_objective_gap","max_dynamics_recompute_error","max_objective_recompute_error","timing")},indent=2))


if __name__ == "__main__":
    main()
