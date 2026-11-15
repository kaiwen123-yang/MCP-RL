#!/usr/bin/env python3
"""Read W50 published records/binaries; no MuJoCo model, solve, or integration.

Run from any directory with --repo and optionally --originals pointing to the
extracted W50 finite Release. Raw data are deliberately not copied into Git.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def policy(path):
    a = np.fromfile(path, dtype='<f8')
    h = a[:16]
    T, ns, nu, nx, nr = map(int, h[3:8])
    i = 16
    p = {'header': h}
    for key, shape in [('x', (T, ns)), ('u', (T-1, nu)), ('t', (T,)),
                       ('r', (T, nr)), ('K', (T, nu, nx)), ('H', (T, nu, nu)),
                       ('Qu', (T, nu)), ('Qxu', (T, nx, nu)), ('active', (T, nu))]:
        n = int(np.prod(shape)); p[key] = a[i:i+n].reshape(shape); i += n
    ext = a[i:i+6]; i += 6; n = int(ext[2])
    p['extension'] = ext
    p['delta_t'] = a[i:i+n]; i += n
    p['delta_u'] = a[i:i+n*nu].reshape(n, nu); i += n*nu
    if h[0] == 50:
        p['provenance'] = a[i:i+32]; i += 32
    if i != len(a):
        raise ValueError(f'Unknown policy payload: {path}')
    return p


def candidates(path):
    a = np.fromfile(path, dtype='<f8'); i = 0; rows = []
    while i < len(a):
        h = a[i:i+19]; i += 19
        if h[0] != 33:
            raise ValueError(f'Unknown candidate schema at {i-19}')
        T, ns, nu, nx, nr = map(int, h[5:10])
        q = {'header': h}
        for prefix in ['', 'ref_']:
            for key, shape in [('x', (T, ns)), ('u', (T-1, nu)), ('t', (T,)), ('r', (T, nr))]:
                n = int(np.prod(shape)); q[prefix+key] = a[i:i+n].reshape(shape); i += n
        for key, shape in [('K', (T, nu, nx)), ('H', (T, nu, nu)), ('Qu', (T, nu)),
                           ('Qxu', (T, nx, nu)), ('active', (T, nu))]:
            n = int(np.prod(shape)); q[key] = a[i:i+n].reshape(shape); i += n
        for key in ['prefix', 'feet']:
            n = int(a[i]); i += 1; q[key] = a[i:i+n]; i += n
        rows.append(q)
    return rows


def cost_parts(r, definitions):
    out = {}
    for d in definitions:
        x = r[:, d['start']:d['end']]; p = d['norm_parameter']
        if d['norm'] == 0:
            v = .5*np.sum(x*x, axis=1)
        elif d['norm'] == 2:
            v = np.sqrt(np.sum(x*x, axis=1)+p*p)-p
        elif d['norm'] == 6:
            v = np.sum(np.sqrt(x*x+p*p)-p, axis=1)
        else:
            raise ValueError(d['norm'])
        out[d['name']] = float(np.mean(d['weight']*v))
    return out


def quat_product(a, b):
    return np.r_[a[0]*b[0]-np.dot(a[1:], b[1:]),
                 a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])]


def state_diff(a, b):
    """Go2's free joint plus 12 hinges, MuJoCo local rotation convention."""
    qa, qb = a[3:7], b[3:7]
    if np.array_equal(qa, qb) or np.array_equal(qa, -qb):
        rot = np.zeros(3)
    else:
        dq = quat_product(qa*np.array([1., -1., -1., -1.]), qb)
        if dq[0] < 0:
            dq = -dq
        n = np.linalg.norm(dq[1:])
        rot = np.zeros(3) if n == 0 else dq[1:]*(2*np.arctan2(n, dq[0])/n)
    return np.r_[b[:3]-a[:3], rot, b[7:19]-a[7:19], b[19:37]-a[19:37]]


def interp(t, grid, values):
    if t <= grid[0]: return values[0]
    if t >= grid[-1]: return values[-1]
    hi = int(np.searchsorted(grid, t, side='right')); lo = hi-1
    w = (t-grid[lo])/(grid[hi]-grid[lo])
    return (1-w)*values[lo]+w*values[hi]


def raw_center(p, x, t):
    xr = interp(t, p['t'], p['x']).copy(); xr[3:7] /= np.linalg.norm(xr[3:7])
    u = interp(t, p['t'][:-1], p['u']).copy()
    K = interp(t, p['t'][:-1], p['K'][:-1])
    u += p['header'][12]*(K@state_diff(xr, x))
    if len(p['delta_t']): u += interp(t, p['delta_t'], p['delta_u'])
    return u


def maximum(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.


def tangent_perturb(x, j, amount):
    """Array-only free-joint/right quaternion increment plus hinge/qvel step."""
    y = x.copy()
    if 3 <= j < 6:
        dq = np.zeros(4); dq[0] = np.cos(amount/2); dq[1+j-3] = np.sin(amount/2)
        y[3:7] = quat_product(y[3:7], dq)
        y[3:7] /= np.linalg.norm(y[3:7])
    else:
        y[j if j < 3 else j+1] += amount
    return y


def raw_analysis(originals, label):
    d = originals/'task-b'/label
    inp = json.loads((originals/'inputs'/label/'input.json').read_text())
    b = policy(d/'first-baseline.bin'); n = policy(d/'first-backward.bin')
    installed = policy(d/'installed.bin'); rs = candidates(d/'snapshot.bin.r33-candidates')
    call = json.loads((d/'planner-calls.jsonl').read_text().splitlines()[-1])
    cap = np.fromfile(d/'first-direction.bin', '<f8'); count, nu = map(int, cap[3:5])
    ts = cap[11:11+count]; k = cap[11+count:].reshape(count, nu)
    z = np.fromfile(d/'first-centers.bin', '<f8'); nx = int(z[3]); i = 7
    center = z[i:i+count*nu].reshape(count, nu); i += count*nu
    L = z[i:i+count*nu*nx].reshape(count, nu, nx); i += count*nu*nx
    H = z[i:].reshape(count, nu, nu)
    base = next(r for r in rs if r['header'][2] == 0 and r['header'][10] == b['header'][12]
                and np.array_equal(r['ref_x'], b['x']) and np.array_equal(r['ref_u'], b['u']))
    raw_reconstructed = np.array([raw_center(b, x, t) for x, t in zip(base['x'][:-1], ts)])
    eps = float(z[5]); L_reconstructed = np.empty_like(L)
    for it, (x, t) in enumerate(zip(base['x'][:-1], ts)):
        for j in range(nx):
            L_reconstructed[it, :, j] = (raw_center(b, tangent_perturb(x, j, eps), t)
                - raw_center(b, tangent_perturb(x, j, -eps), t))/(2*eps)
    # At each normalized midpoint of the actual B trajectory, the conceptual
    # positive-family alpha->0 reference error is exactly zero. Its center is
    # the interpolation of sampled centers. This measures commutation error,
    # not a rollout or a claim that off-trajectory feedback maps are identical.
    midpoint_errors = []
    for j in range(count-1):
        t = .5*(ts[j]+ts[j+1]); x = .5*(base['x'][j]+base['x'][j+1])
        x[3:7] /= np.linalg.norm(x[3:7])
        midpoint_errors.append(.5*(center[j]+center[j+1])-raw_center(b, x, t))
    t_last = .5*(base['t'][-2]+base['t'][-1])
    x_last = .5*(base['x'][-2]+base['x'][-1]); x_last[3:7] /= np.linalg.norm(x_last[3:7])
    final_control_interval_error = center[-1]-raw_center(b, x_last, t_last)
    terminal_center_error = center[-1]-raw_center(b, base['x'][-1], base['t'][-1])
    out = {'label': label, 'raw_candidate_records': len(rs), 'rows': [],
           'baseline_J': float(base['header'][11]),
           'source_mu': float(cap[6]), 'backward_object_mu': float(n['header'][8]),
           'raw_center_max_abs_error_rad': maximum(raw_reconstructed-center),
           'raw_center_readback_scope': 'Independent NumPy quaternion/control-map algebra; no native library or physics.',
           'old_effective_K_full_grid_max': maximum(b['header'][12]*b['K']),
           'old_effective_K_queried_tail_max': maximum(b['header'][12]*b['K'][-2]),
           'rebased_L_max': maximum(L), 'new_K_max': maximum(n['K']),
           'new_K_absmax_index': [int(v) for v in np.unravel_index(np.argmax(np.abs(n['K'])), n['K'].shape)],
           'new_K_at_first_control_max': maximum(n['K'][0]),
           'raw_center_jacobian_max_abs_error': maximum(L_reconstructed-L),
           'alpha0_limit_midpoint_center_max_abs_error_rad': maximum(midpoint_errors),
           'alpha0_limit_midpoint_scope': 'Normalized B reference midpoints before final control node; does not establish full off-trajectory equivalence or projection equivalence.',
           'alpha0_limit_final_control_interval_midpoint_error_rad': maximum(final_control_interval_error),
           'alpha0_limit_terminal_center_error_rad': maximum(terminal_center_error),
           'alpha0_endpoint_scope': 'Hypothetical positive-family alpha->0 map: action/Jacobian held at node 174 while reference state continues to node 175. This is not the cached exact-B alpha=0 branch and does not modify any deployed object.',
           'k_max': maximum(k), 'captured_grid_equal': bool(np.array_equal(ts, base['t'][:-1])),
           'installed_match_records': [], 'max_cost_recompute_abs_error': 0.}
    formulas = []
    for idx, r in enumerate(rs):
        h = r['header']; parts = cost_parts(r['r'], inp['cost_definitions'])
        err = abs(sum(parts.values())-h[11]); out['max_cost_recompute_abs_error'] = max(out['max_cost_recompute_abs_error'], err)
        p = policy(d/f'complete-candidate-{idx+1}.bin')
        # record_plan_id is legitimately replaced on installation. Semantic
        # header fields, including feedback scale, must still be compared.
        equal = (np.array_equal(p['header'][3:], installed['header'][3:])
                 and all(np.array_equal(p[key], installed[key]) for key in p if key != 'header'))
        if equal: out['installed_match_records'].append(idx+1)
        feet = r['feet'].reshape(-1, 29)[:, 1:].reshape(-1, 4, 7)
        row = {'record': idx+1, 'kind': int(h[2]), 'slot': int(h[3]), 'alpha': float(h[4]),
               'J': float(h[11]), 'risk': float(h[12]), 'failure': bool(h[13]),
               'min_z_m': float(r['x'][:, 2].min()), 'final_z_m': float(r['x'][-1, 2]),
               'dx_m': float(r['x'][-1, 0]-r['x'][0, 0]),
               'mean_vx_m_s': float(r['x'][:, 19].mean()),
               'max_action_change_vs_B_rad': maximum(r['u']-base['u']), 'J_parts': parts,
               'foot_clearance_max_m': feet[:, :, 3].max(axis=0).tolist(),
               'foot_load_fraction': ((feet[:, :, 4]+feet[:, :, 5])>1).mean(axis=0).tolist()}
        out['rows'].append(row)
        if h[2] == 1:
            alpha = h[4]
            formulas.append({'slot': int(h[3]),
              'ff_error': maximum(p['u']-center-alpha*k),
              'K_error': maximum(p['K'][:-1]-((1-alpha)*L+alpha*n['K'][:-1])),
              'H_error': maximum(p['H'][:-1]-((1-alpha)*H+alpha*n['H'][:-1])),
              'reference_x_equal': bool(np.array_equal(p['x'], base['x'])),
              'reference_t_equal': bool(np.array_equal(p['t'], base['t'])),
              'K_terminal_equal': bool(np.array_equal(p['K'][-1], p['K'][-2])),
              'H_terminal_equal': bool(np.array_equal(p['H'][-1], p['H'][-2])),
              'capture_Qu_equal': bool(np.array_equal(p['Qu'], n['Qu'])),
              'capture_Qxu_equal': bool(np.array_equal(p['Qxu'], n['Qxu'])),
              'capture_active_equal': bool(np.array_equal(p['active'], n['active'])),
              'scale': float(p['header'][12]), 'delta_knots': len(p['delta_t'])})
    out['formulas'] = formulas
    out['selected_source'] = call['candidate_source']
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--originals', type=Path); ap.add_argument('--output', type=Path)
    args = ap.parse_args(); d = args.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w50'
    a = json.loads((d/'task-a.json').read_text()); b = json.loads((d/'task-b.json').read_text())
    dep = json.loads((d/'deployment.json').read_text())
    out = {'source_commit': '09bd55e0cb70244938ad82db7dd202a31ac70f77',
           'scope': 'Read-only published JSON arithmetic and optional original binary arrays; no new model/FD/solve/physics/PPO.',
           'lifecycle': [], 'finite': [], 'work': {}}
    for s in a['streams']:
        q = s['actual_held_reference_query']; first = s['support_windows'][0]
        out['lifecycle'].append({'stream': s['stream'], 'saved_objects': len(s['saved_policy_lifecycle']),
            'reference_end_values': sorted({x['reference_end_s'] for x in s['saved_policy_lifecycle']}),
            'array_group_counts': {k: len(v) for k, v in s['installed_policy_array_groups'].items()},
            'expired_query_fraction': q['after_reference_end']/q['count'],
            'expired_queries': q['after_reference_end'], 'query_count': q['count'],
            'early_swing_loaded_fraction': first['actual_loaded_during_target_swing_fraction'],
            'early_all_four_loaded_fraction': next(x['fraction'] for x in first['load_patterns'] if x['loaded_FL_FR_RL_RR']==[1,1,1,1]),
            'early_PD_saturated_fraction': first['actual_held_pd_saturation_any_fraction']})
    for e in b['events']:
        B=e['baseline']; w=e['selected']; g=e['homotopy']
        eligible=[c for c in g['candidates'] if not c['failure'] and
          (c['risk']<g['baseline_risk'] or (c['risk']==g['baseline_risk'] and
           g['baseline_J']-c['J']>g['baseline_budget']+c['budget']))]
        out['finite'].append({'label': e['label'], 'strict_positive_count': len(eligible),
            'recorded_strict_count': len(e['qualified_proposals']), 'baseline_J': B['J'],
            'baseline_min_z': B['minimum_z_m'], 'baseline_dx': B['forward_displacement_m'],
            'selected_source': e['selection_source'],
            'selected_J': w['J'] if w else None,
            'relative_J_decrease': (B['J']-w['J'])/B['J'] if w else None,
            'min_z_improvement_mm': 1000*(w['motion']['minimum_z_m']-B['minimum_z_m']) if w else None,
            'final_z_improvement_mm': 1000*(w['motion']['final_z_m']-B['final_z_m']) if w else None,
            'dx_change_mm': 1000*(w['motion']['forward_displacement_m']-B['forward_displacement_m']) if w else None,
            'selected_cost_delta': {k:w['motion']['J_parts'][k]-v for k,v in B['J_parts'].items()} if w else None})
    out['work']={k:b['work'][k]+dep['work'][k] for k in b['work']}
    out['work'].update(saved_solves=len(b['events']), deployment_solves=dep['native_solves'],
      deployment_steps=dep['physics_substeps'], valid_native_equivalent_deployment_steps=0,
      raw_center_evaluations=b['raw_center_evaluations']+sum(x['midpoint']['call']['raw_center_evaluations'] for x in dep['checks']))
    if args.originals:
        out['raw_binary_analysis']=[raw_analysis(args.originals, label) for label in ['A237','B236','A328','B625']]
    text=json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output: args.output.write_text(text)
    else: print(text,end='')


if __name__ == '__main__':
    main()
