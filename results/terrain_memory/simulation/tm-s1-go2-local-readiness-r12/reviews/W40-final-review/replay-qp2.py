#!/usr/bin/env python3
"""Reproduce W40 A314's actual qp1 cycling failure and replay the qp2 repair.

Requires NumPy, a C++17 compiler, patch, and the MuJoCo 3.2.3 wheel unpacked at
--mujoco-root. Compiles the real frozen projection headers reconstructed from
the repository patches. No robot rollout, training, GPU task, or server job.

Example:
  python replay-qp2.py --mujoco-root /path/to/site-packages/mujoco --output qp2-replay.json
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PREVIOUS = HERE.parent / "W40-stop-review"
RESULTS = HERE.parents[1]
FAILURE = RESULTS / "train-job-20261106T103028Z-w40-a-PANN_PPO_S5-s1-ep00314.json"
DP = ct.POINTER(ct.c_double)
IP = ct.POINTER(ct.c_int)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ptr(x):
    return x.ctypes.data_as(DP)


def scalar_gradient(H, center, u):
    delta = u - center
    gradient = np.zeros(len(u))
    for i in range(len(u)):
        for j in range(len(u)):
            gradient[i] += H[i, j] * delta[j]
    return gradient


def section(text, relative):
    marker = "--- a/" + relative + "\n"
    return marker + text.split(marker, 1)[1].split("--- a/", 1)[0]


def compile_solver(tmp, header, library, include, revision):
    (tmp / "r29_projection_current.h").write_text(header)
    source = (PREVIOUS / "qp-replay-wrapper.cc").read_text()
    source = source.replace('"r29_projection_qp1.h"', '"r29_projection_current.h"')
    source = source.replace("int* iterations,char* status,int status_size)",
                            "int* iterations,char* status,int status_size,int* detail)")
    extra = ("detail[0]=r.refinement_face_solves;"
             "detail[1]=r.refinement_bound_releases;detail[2]=r.refinement_zero_steps;")
    if revision == "qp1":
        extra = "detail[0]=detail[1]=detail[2]=-1;"
    source = source.replace("*iterations=r.refinement_iterations;", "*iterations=r.refinement_iterations;" + extra)
    (tmp / "wrapper.cc").write_text(source)
    destination = tmp / (revision + ".so")
    subprocess.run(["c++", "-O2", "-std=c++17", "-shared", "-fPIC", str(tmp / "wrapper.cc"),
                    "-I" + str(tmp), "-I" + str(include), str(library),
                    "-Wl,-rpath," + str(library.parent), "-o", str(destination)],
                   check=True, text=True, capture_output=True)
    lib = ct.CDLL(str(destination))
    lib.native_projection.argtypes = [ct.c_int, DP, DP, DP, DP, DP, ct.c_char_p, ct.c_int]
    lib.native_projection.restype = ct.c_int
    lib.refine_projection.argtypes = [ct.c_int, DP, DP, DP, DP, DP, IP, ct.c_char_p, ct.c_int, IP]
    lib.refine_projection.restype = ct.c_int
    return lib


def refinement(lib, H, center, lower, upper, initial, native_nfree, certificate):
    u = initial.copy()
    before = certificate(H, center, lower, upper, initial, native_nfree)
    nfree = native_nfree
    iterations = ct.c_int()
    status = ct.create_string_buffer(128)
    detail = (ct.c_int * 3)(0, 0, 0)
    if not before["converged"]:
        nfree = lib.refine_projection(len(u), *map(ptr, (H, center, lower, upper, u)),
            ct.byref(iterations), status, len(status), detail)
    after = certificate(H, center, lower, upper, u, nfree)
    return u, dict(attempted=not before["converged"], iterations=iterations.value,
                  status=status.value.decode() if not before["converged"] else "not_needed",
                  nfree=nfree, certificate=after,
                  face_solves=detail[0] if detail[0] >= 0 else None,
                  bound_releases=detail[1] if detail[1] >= 0 else None,
                  zero_steps=detail[2] if detail[2] >= 0 else None)


def cycling_witness(H, center, lower, upper, u, certificate, mujoco):
    """The recorded qp1 terminal state itself witnesses release/relock at alpha=0."""
    result = certificate(H, center, lower, upper, u, -1)
    active = np.array(result["active_bounds"])
    free = np.flatnonzero(active == 0)
    gradient = scalar_gradient(H, center, u)
    tolerance = 1e-7 * result["scale"]
    wrong = np.where(active == -1, -gradient, np.where(active == 1, gradient, 0.0))
    released = int(np.argmax(wrong))
    expanded = np.flatnonzero((active == 0) | (np.arange(len(u)) == released))
    factor = np.ascontiguousarray(H[np.ix_(expanded, expanded)])
    rhs = np.ascontiguousarray(-gradient[expanded])
    step = np.zeros(len(expanded))
    mujoco.mju_cholFactor.argtypes = [DP, ct.c_int, ct.c_double]
    mujoco.mju_cholFactor.restype = ct.c_int
    mujoco.mju_cholSolve.argtypes = [DP, DP, DP, ct.c_int]
    rank = mujoco.mju_cholFactor(ptr(factor), len(expanded), 0.0)
    mujoco.mju_cholSolve(ptr(step), ptr(factor), ptr(rhs), len(expanded))
    position = int(np.flatnonzero(expanded == released)[0])
    direction = float(step[position])
    alpha = ((upper[released] - u[released]) / direction if direction > 0 else
             (lower[released] - u[released]) / direction)
    assert rank == len(expanded) and wrong[released] > tolerance
    assert np.max(np.abs(gradient[free])) <= tolerance
    assert direction * active[released] > 0 and alpha == 0.0
    return dict(free_axes=free.tolist(), gradient_tolerance=tolerance,
                max_abs_free_gradient=float(np.max(np.abs(gradient[free]))),
                released_axis_zero_based=released, released_bound=int(active[released]),
                wrong_sign_multiplier=float(wrong[released]),
                enlarged_face_newton_direction_at_released_axis_rad=direction,
                maximum_feasible_step=alpha,
                implication="qp1 labels the unminimized free face stationary, releases this bound, takes a zero step, and reactivates the same bound. The next release/relock returns to the same point.",
                working_set_error="The final KKT acceptance tolerance is too loose to certify minimization of the free face in this ill-conditioned coupled Hessian.")


def replay(mujoco_root):
    library = mujoco_root / "libmujoco.so.3.2.3"
    if not library.is_file():
        raise ValueError("The exact-source reproduction requires MuJoCo 3.2.3.")
    old_module = importlib.util.spec_from_file_location("w40_previous_replay", PREVIOUS / "replay-qp.py")
    old = importlib.util.module_from_spec(old_module)
    old_module.loader.exec_module(old)
    certificate = old.certificate
    base = ROOT / "experiments/terrain_memory/s1-go2-r11-S5-r29.patch"
    patches = {v: ROOT / ("experiments/terrain_memory/s1-go2-r11-S5-r29-" + v + ".patch")
               for v in ("qp1", "qp2")}
    relative = "mjpc/planners/ilqg/r29_projection.h"
    added = base.read_text().split("+++ b/" + relative + "\n", 1)[1].split("--- a/", 1)[0]
    original = "\n".join(line[1:] for line in added.splitlines() if line.startswith("+")) + "\n"
    fixtures = json.loads((PREVIOUS / "qp-replay-fixtures.json").read_text())["cases"]
    for case in fixtures:
        case["recorded_implementation"] = "native"
    failure_result = json.loads(FAILURE.read_text())
    failure = failure_result["r29_failure"]
    query = failure["query"]
    fixtures.append(dict(case_id="A314-q12387", query_id=query["query_id"],
        time_s=query["time"], plan_id=query["plan_id"], H=failure["metric_H"],
        center=query["center_before_projection"], lower=query["instantaneous_lower"],
        upper=query["instantaneous_upper"], recorded_u=query["u_star"],
        recorded_converged=False, recorded_implementation="qp1"))
    results = []
    with tempfile.TemporaryDirectory(prefix="w40-qp2-replay-") as directory:
        tmp = Path(directory)
        solver = tmp / "mjpc/planners/ilqg"
        solver.mkdir(parents=True)
        header = solver / "r29_projection.h"
        header.write_text(original)
        shutil.copyfile(PREVIOUS / "qp-boxqp.h", solver / "boxqp.h")
        libraries = {}
        for revision in ("qp1", "qp2"):
            subprocess.run(["patch", "--batch", "--silent", "-p1"],
                input=section(patches[revision].read_text(), relative), text=True,
                cwd=tmp, check=True, capture_output=True)
            libraries[revision] = compile_solver(tmp, header.read_text(), library, mujoco_root / "include", revision)
        for case in fixtures:
            H, center, lower, upper, recorded = [np.ascontiguousarray(case[k], dtype=np.float64)
                for k in ("H", "center", "lower", "upper", "recorded_u")]
            native = np.zeros(len(center))
            log = ct.create_string_buffer(100000)
            nfree = libraries["qp1"].native_projection(len(center), *map(ptr, (H, center, lower, upper, native)), log, len(log))
            original_certificate = certificate(H, center, lower, upper, native, nfree)
            previous, qp1 = refinement(libraries["qp1"], H, center, lower, upper, native, nfree, certificate)
            repaired, qp2 = refinement(libraries["qp2"], H, center, lower, upper, native, nfree, certificate)
            source_output = native if case["recorded_implementation"] == "native" else previous
            assert np.array_equal(source_output, recorded), case["case_id"] + " recorded output mismatch"
            assert qp2["certificate"]["converged"], case["case_id"] + " still fails original certificate"
            if original_certificate["converged"]:
                assert np.array_equal(native, repaired)
            eigenvalues = np.linalg.eigvalsh(H)
            item = dict(case_id=case["case_id"], query_id=case["query_id"], time_s=case["time_s"],
                plan_id=case["plan_id"], recorded_implementation=case["recorded_implementation"],
                recorded_output_reproduced_bit_for_bit=True,
                metric_min_eigenvalue=float(eigenvalues[0]), metric_max_eigenvalue=float(eigenvalues[-1]),
                metric_condition=float(np.linalg.cond(H)), metric_max_asymmetry=float(np.max(np.abs(H-H.T))),
                native_nfree=nfree, native_certificate=original_certificate,
                native_log_tail=log.value.decode().splitlines()[-3:], qp1=qp1, qp2=qp2,
                native_passing_output_unchanged=(bool(np.array_equal(native, repaired))
                    if original_certificate["converged"] else None),
                prior_qp1_passing_output_unchanged=(bool(np.array_equal(previous, repaired))
                    if qp1["certificate"]["converged"] else None),
                max_abs_qp2_vs_qp1_delta_rad=float(np.max(np.abs(repaired-previous))))
            if case["recorded_implementation"] == "qp1":
                assert qp1["status"] == "iteration_limit" and qp1["iterations"] == 608
                item["cycling_witness"] = cycling_witness(H, center, lower, upper, previous,
                    certificate, ct.CDLL(str(library)))
                item["native_u"] = native.tolist()
                item["qp1_rejected_u"] = previous.tolist()
                item["qp2_u"] = repaired.tolist()
            results.append(item)
    return dict(schema="w40_qp2_actual_failure_replay_v1",
        evidence_commit="c438f5463659ab7c8d3a7f5eced251639e14b2ef",
        scope="Twelve recorded QP inputs, compiled native/qp1/qp2 numerical functions. No physical rollout or PPO/GPU task.",
        new_failure_file=str(FAILURE.relative_to(ROOT)), new_failure_file_sha256=sha256(FAILURE),
        previous_fixtures_sha256=sha256(PREVIOUS / "qp-replay-fixtures.json"),
        base_patch_sha256=sha256(base), patch_sha256={v: sha256(p) for v,p in patches.items()},
        library=library.name, library_sha256=sha256(library),
        original_tolerances=dict(primal_rad=1e-10, KKT_scaled=1e-7, active_bound_rad=1e-12),
        exact_dependency_commit="088079eff0450e32b98ee743141780ed68307506",
        dependency_equivalence="MuJoCo pinned engine_util_solve.c box-QP section is byte-identical to3.2.3, as documented in the previous W40 replay.",
        recorded_cases=len(results), all_recorded_outputs_reproduced=True,
        all_qp2_outputs_pass_original_certificate=True,
        native_passing_outputs_unchanged=sum(x["native_passing_output_unchanged"] is True for x in results),
        prior_qp1_refined_passing_outputs_unchanged=sum(x["qp1"]["attempted"] and
            x["prior_qp1_passing_output_unchanged"] is True for x in results),
        results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.mujoco_root.resolve())
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
