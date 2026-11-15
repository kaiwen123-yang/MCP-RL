#!/usr/bin/env python3
"""W50 C: compare saved W49 active GPU rows, without importing a simulator."""
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
NPZ = ROOT / 'outputs' / BASE / 'w49/task-c/all-exposed-substeps.npz'
OLD = ROOT / 'results' / BASE / 'w49/task-c.json'
OUT = ROOT / 'results' / BASE / 'w50/task-c.json'
SITE = Path('/root/autodl-tmp/MCP-RL/work/envs/w41-mjwarp/lib/python3.12/site-packages')
SOURCE = SITE / 'mujoco_warp/_src'
CONSTRAINT_INPUTS = ('J', 'pos', 'margin', 'D', 'vel', 'aref', 'frictionloss')
SMOOTH_INPUTS = ('ctrl', 'actuator_force', 'qfrc_actuator', 'qfrc_passive',
                 'qfrc_applied', 'xfrc_applied', 'qfrc_bias', 'qfrc_smooth',
                 'qacc_smooth', 'M', 'qLD', 'qLDiagInv')
OUTPUTS = ('qfrc_constraint', 'qacc', 'qpos', 'qvel', 'efc_Ma', 'solver_niter')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {'equal': bool(np.array_equal(a, b)),
            'different_elements': int(np.count_nonzero(a != b)),
            'max_abs': float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0}


def snapshot(z, condition, step):
    prefix = f'{condition}_{step:02d}_'
    return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}


def contact_order(s, world):
    # The current saved fixture has one contact per geom pair; global pool id is not identity.
    ids = np.flatnonzero(s['contact_worldid'][:int(s['nacon'][0])] == world)
    return sorted(ids.tolist(), key=lambda i: tuple(s['contact_geom'][i].tolist()))


def constraint_order(s, world):
    rows = []
    for i in range(int(s['nefc'][world])):
        kind, identity = int(s['efc_type'][world, i]), int(s['efc_id'][world, i])
        if kind == 1:  # FRICTION_DOF: id is a physical dof.
            key = (kind, identity, -1, 0)
        elif kind == 7:  # CONTACT_ELLIPTIC: id is a global contact pool address.
            geom = s['contact_geom'][identity]
            component = int(np.flatnonzero(s['contact_efc_address'][identity] == i)[0])
            key = (kind, int(geom[0]), int(geom[1]), component)
        else:
            raise ValueError(f'Unexpected existing constraint type {kind}; no alternate fixture is substituted')
        rows.append((key, i))
    rows.sort()
    return [list(key) for key, _ in rows], [i for _, i in rows]


def compare(a, wa, b, wb):
    ka, ia = constraint_order(a, wa)
    kb, ib = constraint_order(b, wb)
    constraints = {}
    for name in CONSTRAINT_INPUTS + ('state', 'force'):
        aa, bb = a['efc_' + name][wa, ia], b['efc_' + name][wb, ib]
        if name == 'J':
            # nv=18; the saved dense J has 20 storage columns.
            aa, bb = aa[:, :18], bb[:, :18]
        constraints[name] = delta(aa, bb)
    ca, cb = contact_order(a, wa), contact_order(b, wb)
    contacts = {}
    excluded = {'contact_worldid', 'contact_geomcollisionid', 'contact_efc_address'}
    for name in sorted(k for k in a if k.startswith('contact_') and k not in excluded):
        if a[name].size:
            contacts[name.removeprefix('contact_')] = delta(a[name][ca], b[name][cb])
    return {
        'canonical_constraint_keys_equal': ka == kb,
        'constraints': constraints,
        'contact_values_equal': all(v['equal'] for v in contacts.values()),
        'contact_differences': {k: v for k, v in contacts.items() if not v['equal']},
        'smooth_inputs': {k: delta(a[k][wa], b[k][wb]) for k in SMOOTH_INPUTS},
        'outputs': {k: delta(a[k][wa], b[k][wb]) for k in OUTPUTS},
        'active_rows': [int(a['nefc'][wa]), int(b['nefc'][wb])],
        'active_contacts': [len(ca), len(cb)],
    }


def source_record(name, spans, interpretation):
    path = SOURCE / name
    return {'path': str(path), 'sha256': digest(path), 'line_spans': spans,
            'interpretation': interpretation}


def main():
    begin = time.time()
    previous = json.loads(OLD.read_text())
    z = np.load(NPZ)
    series = {c: [snapshot(z, c, i) for i in range(11)] for c in ('F1', 'L', 'F2')}
    first = {}
    later = []
    layouts = {}
    for condition, frames in series.items():
        first[condition] = compare(frames[1], 0, frames[1], 1)
        s = frames[1]
        layouts[condition] = {
            'friction_dof_row_ids': [s['efc_id'][w, :int(s['nf'][w])].tolist() for w in range(2)],
            'dof_frictionadr': s['dof_frictionadr'].tolist(),
            'contact_pool_worldids': s['contact_worldid'][:int(s['nacon'][0])].tolist(),
            'contact_pool_geom_pairs': s['contact_geom'][:int(s['nacon'][0])].tolist(),
            'contact_row_pool_ids': [s['efc_id'][w, int(s['nf'][w]):int(s['nefc'][w])].tolist() for w in range(2)],
            'canonical_constraint_keys': constraint_order(s, 0)[0],
            'unused_constraint_rows_per_world': [int(s['efc_J'].shape[1] - n) for n in s['nefc']],
            'unused_J_columns': int(s['efc_J'].shape[2] - 18),
            'unused_global_contact_slots': int(s['contact_dist'].shape[0] - s['nacon'][0]),
        }
        for step, s in enumerate(frames[1:], 1):
            c = compare(s, 0, s, 1)
            later.append({'condition': condition, 'substep': step,
                'active_rows': c['active_rows'], 'active_contacts': c['active_contacts'],
                'contact_values_equal': c['contact_values_equal'],
                'constraint_input_differences': [k for k in CONSTRAINT_INPUTS if not c['constraints'][k]['equal']],
                'constraint_force_max_abs': c['constraints']['force']['max_abs'],
                'qacc_max_abs': c['outputs']['qacc']['max_abs'],
                'qpos_max_abs': c['outputs']['qpos']['max_abs'],
                'qvel_max_abs': c['outputs']['qvel']['max_abs']})
    same_world = {str(w): compare(series['F1'][1], w, series['F2'][1], w) for w in range(2)}
    code = [
        source_record('constraint.py', [[1953, 2053], [5092, 5095], [6113, 6114]],
            'DOF friction rows use atomic allocation. CONSTRAINT option calls _sort_constraints, which is TODO/pass in this installed source; toggling that flag alone is not a repair.'),
        source_record('solver.py', [[71, 164], [1158, 1210], [2376, 2407], [2908, 3013], [4540, 4627]],
            'Row-indexed tile line-search reductions, Newton grad_dot atomic sum and elliptic contact Hessian atomic sum are candidate rounding paths. A separate SolverContext is allocated per solve and is absent from Data snapshots; these saved boundaries do not identify the first differing kernel.'),
        source_record('forward.py', [[3502, 3560]],
            'Constraint assembly and smooth acceleration precede solver.solve, which precedes implicit integration.'),
        source_record('types.py', [[308, 324]],
            'Determinism flags describe separate contact ordering, constraint ordering and atomic reduction features.'),
        source_record('io.py', [[572, 572]],
            'put_model initializes determinism to NONE; W49 script does not change it.'),
    ]
    callback = ROOT / 'scripts/s1_go2_w41_warp_backend.py'
    source_match = {path: {'saved_sha256': sha, 'current_sha256': digest(Path(path)),
        'matches_W49': digest(Path(path)) == sha} for path, sha in previous['sources_code'].items()}
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions(path=[str(SITE)])
                if d.metadata['Name'].lower() in ('mujoco', 'mujoco-warp', 'warp-lang')}
    result = {
        'schema': 's1_w50_gpu_saved_active_constraint_analysis_v1',
        'status': 'completed_zero_physics_analysis',
        'plan_commit': '9ba97283c13119ec197429b75fce6b139a506a33',
        'command': 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-21 work/envs/d57/bin/python scripts/w50_gpu_analysis.py',
        'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)),
        'started_unix_s': begin, 'ended_unix_s': time.time(), 'wall_s': time.time()-begin,
        'deadline': None, 'wall_deadline_enabled': False,
        'work': {'saved_conditions': 3, 'saved_state_boundaries': 33, 'saved_step_pairs': 30,
            'new_native_solves': 0, 'new_native_full_rollouts': 0, 'new_native_substeps': 0,
            'new_gpu_world_steps': 0, 'warmup_world_steps': 0, 'FD': 0,
            'new_PPO_samples': 0, 'new_PPO_updates': 0, 'GPU_context_created': False},
        'source': {'path': str(NPZ), 'bytes': NPZ.stat().st_size, 'sha256': digest(NPZ),
            'matches_W49_recorded_sha256': digest(NPZ) == previous['artifact']['sha256'],
            'all_saved_array_fields_per_boundary': len(series['F1'][0]),
            'versions_recorded_W49': previous['versions'], 'versions_installed_now': versions,
            'W49_options': previous['options'], 'current_source_matches_recorded_files': source_match,
            'identity_limit': 'W49 recorded io.py and callback hashes; exact historical solver.py/constraint.py hashes were not recorded, so their historical byte identity is not independently established.'},
        'normalization': {
            'constraint_rows': 'Only 0:nefc per world; friction identity=(type,dof), elliptic identity=(type,geom_pair,cone_component).',
            'jacobian': 'Only 18 physical DOF columns of 20-column dense storage.',
            'contacts': 'Only 0:nacon, split by worldid and ordered by unique geom pair. Compare all nonempty numeric contact arrays except allocation/address fields.',
            'not_physical_differences': ['unused constraint rows', 'unused J columns', 'unused contact slots',
                'raw global contact ids', 'contact worldid labels', 'global geomcollisionid', 'raw efc row addresses'],
            'no_first_kernel_claim': 'Snapshots are post-step Data boundaries, not solver iteration/kernel trace.'},
        'first_step_layouts': layouts,
        'first_step_canonical_comparisons': first,
        'F1_vs_F2_first_step_same_world': same_world,
        'ten_step_summary': later,
        'source_paths': code,
        'callback': {'path': str(callback), 'sha256': digest(callback), 'line_spans': [[17, 23], [25, 44]],
            'write_sets': 'filter_contact writes dist[i] per active pooled contact; material_force writes out[world,dof] per distinct launch index. Initial passive/external forces are equal; no direct cross-world shared write found in this callback.'},
        'conclusions': [
            'All three first-step comparisons have identical canonical active constraint inputs and identical smooth dynamics inputs. Physical contact fields are equal. Raw efc_J/D/aref differences are row-layout artifacts, not different physical constraint assembly.',
            'After normalization, contact efc_force already differs at the first stored step, while friction forces and constraint states match. The first observable physical divergence lies in the constraint solve before integration.',
            'DOF friction row permutation and global contact allocation expose different reduction ordering. This is a concrete plausible floating-point mechanism, not proof of a unique kernel or a race.',
            'F1 versus F2 same-world canonical inputs also match at step 1, but force/qacc differ. Therefore friction-row permutation alone is not established as sufficient explanation; solver reduction order and unrecorded local workspace remain unresolved.',
            'The installed canonical-constraint-sort function is a stub. No source/config repair was applied and no new probe was justified from these boundary-only data.',
            'Small FP32 differences do not by themselves invalidate engineering use. Full native J is still absent on GPU, and this analysis provides no equivalent-native-ranking or speedup qualification.'
        ],
        'probe': {'status': 'not_run', 'reason': 'No isolated repair with a uniquely discriminating probe established; preserve conditional <=20-world-step budget and unrelated GPU training.'},
        'qualification': {'full_native_cost_implemented_on_gpu': False, 'equivalent_native_ranking': False,
            'original_task_result': None, 'blocks_CPU_W50': False},
        'server_originals_retained': True, 'user_pc_download': False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'result': str(OUT), 'bytes': OUT.stat().st_size,
        'work': result['work'], 'first_step_contact_force_max_abs': {c: v['constraints']['force']['max_abs'] for c,v in first.items()},
        'same_world_F1_F2_force_max_abs': {w: v['constraints']['force']['max_abs'] for w,v in same_world.items()}}))


if __name__ == '__main__':
    main()
