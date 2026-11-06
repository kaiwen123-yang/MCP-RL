#!/usr/bin/env python3
"""Replay the two W40 rejected QPs and adjacent accepted queries, without physics.

Example:
  python replay-qp.py --mujoco-root /path/to/site-packages/mujoco --output replay.json

The fixture consists solely of recorded W40 inputs. The script reconstructs the
immutable r29 projection header, applies the proposed incremental repair, and
compiles the same refinement function against the supplied MuJoCo library.
It does not run the robot, PPO, GPU/JAX, or any server session.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certificate(H, center, lower, upper, u, nfree):
    """The original r29 scalar-loop primal and scaled KKT definition."""
    n = len(u)
    delta = u - center
    gradient = np.zeros(n)
    scale, primal, kkt = 1.0, 0.0, 0.0
    active = []
    finite = True
    for i in range(n):
        primal = max(primal, lower[i] - u[i], u[i] - upper[i])
        hc = 0.0
        for j in range(n):
            gradient[i] += H[i, j] * delta[j]
            hc += abs(H[i, j] * center[j])
        scale = max(scale, hc)
        fixed = lower[i] == upper[i]
        a = 2 if fixed else (-1 if u[i] <= lower[i] + 1e-12 else (1 if u[i] >= upper[i] - 1e-12 else 0))
        active.append(a)
        residual = 0.0 if fixed else (max(0.0, -gradient[i]) if a == -1 else
                    (max(0.0, gradient[i]) if a == 1 else abs(gradient[i])))
        kkt = max(kkt, residual)
        finite = finite and np.isfinite(u[i]) and np.isfinite(gradient[i])
    return dict(primal_rad=float(primal), kkt=float(kkt), kkt_scaled=float(kkt / scale),
                scale=float(scale), active_bounds=active,
                converged=bool(finite and nfree != -1 and primal <= 1e-10 and kkt / scale <= 1e-7),
                centered_objective=float(0.5 * delta @ H @ delta))


def replay(mujoco_root: Path, fixtures: Path):
    base_patch = ROOT / 'experiments/terrain_memory/s1-go2-r11-S5-r29.patch'
    repair_patch = ROOT / 'experiments/terrain_memory/s1-go2-r11-S5-r29-qp1.patch'
    libraries = sorted(mujoco_root.glob('libmujoco.so.*'))
    if len(libraries) != 1:
        raise RuntimeError(f'Expected one MuJoCo shared library under {mujoco_root}: {libraries}')
    library = libraries[0].resolve()
    if library.name != 'libmujoco.so.3.2.3':
        raise RuntimeError('This exact-source replay requires MuJoCo 3.2.3; another version needs a new source comparison.')
    include = mujoco_root / 'include'
    source = base_patch.read_text().split('+++ b/mjpc/planners/ilqg/r29_projection.h\n', 1)[1].split('--- a/', 1)[0]
    original = '\n'.join(line[1:] for line in source.splitlines() if line.startswith('+')) + '\n'
    marker = '--- a/mjpc/planners/ilqg/r29_projection.h\n'
    incremental = marker + repair_patch.read_text().split(marker, 1)[1].split('--- a/', 1)[0]
    fixture = json.loads(fixtures.read_text())

    with tempfile.TemporaryDirectory(prefix='w40-qp-replay-') as tmp:
        tmp = Path(tmp)
        solver = tmp / 'mjpc/planners/ilqg'
        solver.mkdir(parents=True)
        (solver / 'r29_projection.h').write_text(original)
        shutil.copyfile(HERE / 'qp-boxqp.h', solver / 'boxqp.h')
        subprocess.run(['patch', '--batch', '--silent', '-p1'], input=incremental,
                       text=True, cwd=tmp, check=True, capture_output=True)
        shutil.copyfile(solver / 'r29_projection.h', tmp / 'r29_projection_qp1.h')
        shutil.copyfile(HERE / 'qp-replay-wrapper.cc', tmp / 'wrapper.cc')
        subprocess.run(['c++', '-O2', '-std=c++17', '-shared', '-fPIC', str(tmp / 'wrapper.cc'),
                        '-I' + str(tmp), '-I' + str(include), str(library),
                        '-Wl,-rpath,' + str(library.parent), '-o', str(tmp / 'qp.so')],
                       check=True, capture_output=True, text=True)
        lib = ctypes.CDLL(str(tmp / 'qp.so'))
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int)
        lib.native_projection.argtypes = [ctypes.c_int, dp, dp, dp, dp, dp, ctypes.c_char_p, ctypes.c_int]
        lib.native_projection.restype = ctypes.c_int
        lib.refine_projection.argtypes = [ctypes.c_int, dp, dp, dp, dp, dp, ip, ctypes.c_char_p, ctypes.c_int]
        lib.refine_projection.restype = ctypes.c_int
        pointer = lambda x: x.ctypes.data_as(dp)
        results = []
        for case in fixture['cases']:
            H, center, lower, upper, recorded = [np.ascontiguousarray(case[k], dtype=np.float64)
                for k in ('H', 'center', 'lower', 'upper', 'recorded_u')]
            n = len(center)
            native = np.zeros(n)
            log = ctypes.create_string_buffer(100000)
            nfree = lib.native_projection(n, pointer(H), pointer(center), pointer(lower), pointer(upper),
                                         pointer(native), log, len(log))
            before = certificate(H, center, lower, upper, native, nfree)
            result = dict(case_id=case['case_id'], query_id=case['query_id'], time_s=case['time_s'],
                          plan_id=case['plan_id'], recorded_converged=case['recorded_converged'],
                          native_nfree=nfree, native_matches_recorded_exactly=bool(np.array_equal(native, recorded)),
                          native_max_abs_recorded_difference_rad=float(np.max(np.abs(native-recorded))),
                          native=before, native_log_tail=log.value.decode().splitlines()[-3:],
                          metric_min_eigenvalue=float(np.linalg.eigvalsh(H)[0]),
                          metric_max_eigenvalue=float(np.linalg.eigvalsh(H)[-1]),
                          metric_condition_number=float(np.linalg.cond(H)))
            repaired = native.copy()
            iteration = ctypes.c_int()
            status = ctypes.create_string_buffer(128)
            final_nfree = nfree
            if not before['converged']:
                final_nfree = lib.refine_projection(n, pointer(H), pointer(center), pointer(lower), pointer(upper),
                    pointer(repaired), ctypes.byref(iteration), status, len(status))
            after = certificate(H, center, lower, upper, repaired, final_nfree)
            result.update(refinement_attempted=not before['converged'], refinement_iterations=iteration.value,
                          refinement_status=status.value.decode() if not before['converged'] else 'not_needed',
                          final_nfree=final_nfree, final=after,
                          max_correction_rad=float(np.max(np.abs(repaired-native))),
                          passing_native_action_unchanged=(bool(np.array_equal(native,repaired)) if before['converged'] else None))
            if not before['converged']:
                # Distinguish an absolute-coordinate cancellation hypothesis
                # from the observed native active-set iteration limit.
                zero = np.zeros(n)
                shifted_lower = np.ascontiguousarray(lower-center)
                shifted_upper = np.ascontiguousarray(upper-center)
                shifted = np.zeros(n)
                shifted_log = ctypes.create_string_buffer(100000)
                shifted_nfree = lib.native_projection(n, pointer(H), pointer(zero), pointer(shifted_lower),
                    pointer(shifted_upper), pointer(shifted), shifted_log, len(shifted_log))
                result['native_shift_only'] = certificate(H,center,lower,upper,shifted+center,shifted_nfree)
                result['native_shift_only_log_tail'] = shifted_log.value.decode().splitlines()[-3:]
                result['native_u'] = native.tolist()
                result['repaired_u'] = repaired.tolist()
            if not result['native_matches_recorded_exactly']:
                raise AssertionError(f'Native replay differs from recorded input: {case["case_id"]}')
            if before['converged'] != case['recorded_converged'] or not after['converged']:
                raise AssertionError(f'Certificate mismatch: {case["case_id"]}')
            results.append(result)

    return dict(schema='w40_qp_failure_replay_v1', scope='Recorded algebra only; no physical rollout, training, or GPU task.',
                base_patch_sha256=sha256(base_patch), repair_patch_sha256=sha256(repair_patch),
                fixtures_sha256=sha256(fixtures), mujoco_library=library.name, mujoco_library_sha256=sha256(library),
                exact_dependency_commit='088079eff0450e32b98ee743141780ed68307506',
                dependency_equivalence='mju_boxQP through the end of engine_util_solve.c is byte-identical between pinned source and MuJoCo 3.2.3 (10399 bytes).',
                native_failed_cases=sum(not x['native']['converged'] for x in results),
                repaired_cases=sum(x['refinement_attempted'] and x['final']['converged'] for x in results),
                unchanged_passing_cases=sum(x['passing_native_action_unchanged'] is True for x in results),
                results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mujoco-root', required=True, type=Path)
    parser.add_argument('--fixtures', type=Path, default=HERE / 'qp-replay-fixtures.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = replay(args.mujoco_root.resolve(), args.fixtures)
    text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
