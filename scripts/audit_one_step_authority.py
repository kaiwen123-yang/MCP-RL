"""Exact one-step outer relaxation of all 768 saved native MPC queries.

No policy is evaluated or updated; no environment/planner is run. Independent
checks use the original reference map and a scalar convex minimizer. Freedom
to choose u independently of the actor makes this a control-authority outer
bound, not a stronger solver for the same frozen-policy class.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from response_ppo.env import EnvironmentConfig, ResponseServo

OUT = ROOT/'notes/stage-feasibility-2026-10-17'
SEEDS = (17, 29, 43)
ARMS = ('joint_support', 'fixed_support', 'joint_penalty', 'fixed_penalty')
TICKS = (0, 60, 120, 180)
TOL = 1e-8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_extrema(z, raw, cfg):
    """c is monotone; at c endpoints, eta has a quadratic dependence."""
    omega, zeta, kappa, factor = np.array([1.2, .7, 3., 0.])+np.array([1.8, .7, 11., .5])*expit(raw)
    p, v, a = z
    q = a-np.sin(p)-.2*v-.1*v*abs(v)
    rate = omega/(1+factor*v*v)
    base = a+cfg.tau*(np.cos(p)*v+(.2+.2*abs(v))*q-kappa*q)
    linear = -cfg.tau*kappa*2*zeta*rate*v
    records = []
    for c in (-.45, .45):
        quadratic = cfg.tau*kappa*rate*rate*(c-p)
        etas = [.8, 1.2]
        if quadratic != 0:
            stationary = -linear/(2*quadratic)
            if .8 < stationary < 1.2:
                etas.append(stationary)
        for eta in etas:
            raw_u = base+linear*eta+quadratic*eta*eta
            records.append((float(np.clip(raw_u, -1., 1.)), c, eta))
    return min(r[0] for r in records), max(r[0] for r in records), records


def drift(x, cfg):
    p, v, a = x
    return np.array([p+cfg.dt*v, v+cfg.dt*(a-np.sin(p)-.2*v-.1*v*abs(v)),
                     (1-cfg.dt/cfg.tau)*a])


def scalar_minimax(eo, P, lower, upper):
    """Min max_j [Paa*t^2 + b_j*t + c_j], all same positive curvature."""
    aa = P[2, 2]
    bb = 2*eo@P[:2, 2]
    cc = np.einsum('bi,ij,bj->b', eo, P[:2, :2], eo)
    candidates = [lower, upper, *(-bb/(2*aa))]
    for i in range(len(bb)):
        for j in range(i):
            if bb[i] != bb[j]:
                candidates.append((cc[j]-cc[i])/(bb[i]-bb[j]))
    candidates = np.clip(candidates, lower, upper)
    values = np.max(aa*candidates[:, None]**2+bb*candidates[:, None]+cc, axis=1)
    k = np.argmin(values)
    def objective(t):
        return float(np.max(aa*t*t+bb*t+cc))
    independent = minimize_scalar(objective, bounds=(lower, upper), method='bounded',
                                  options={'xatol': 1e-14, 'maxiter': 200})
    # Include endpoints since bounded minimization generally approaches them.
    independent_value = min(independent.fun, objective(lower), objective(upper))
    return float(values[k]), float(candidates[k]), float(abs(independent_value-values[k])), int(independent.nfev)


def summarize(rows):
    empty = [r for r in rows if r['pool_empty']]
    unresolved = [r for r in empty if not r['nominal_impossible_next_tube']]
    return {'queries':len(rows), 'pool_empty':len(empty),
            'initial_tube_failed':sum(r['initial_tube_failed'] for r in rows),
            'empty_initial_tube_failed':sum(r['initial_tube_failed'] for r in empty),
            'nominal_impossible_next_tube':sum(r['nominal_impossible_next_tube'] for r in rows),
            'empty_nominal_impossible_next_tube':sum(r['nominal_impossible_next_tube'] for r in empty),
            'empty_unbounded_schur_impossible':sum(r['schur_impossible_next_tube'] for r in empty),
            'robust_impossible_next_tube':sum(r['robust_impossible_next_tube'] for r in rows),
            'empty_robust_impossible_next_tube':sum(r['robust_impossible_next_tube'] for r in empty),
            'empty_not_explained_by_nominal_one_step_bound':len(unresolved),
            'unresolved_keys':[r['key'] for r in unresolved],
            'smallest_positive_nominal_failure_gap':min((r['nominal_min_V']-r['gate_budget'] for r in rows
                                                         if r['nominal_impossible_next_tube']),default=None)}


def main():
    started = time.perf_counter()
    protocol = OUT/'protocol.md'
    protocol_hash = sha(protocol)
    rows, inputs, checks = [], [], []
    reference_checks = scalar_evals = 0
    for seed in SEEDS:
        for arm in ARMS:
            folder = ROOT/f'work/ppo-mechanism-2026-10-16/formal/seedrun-{seed}/seed-{seed}'
            config_path = folder/'result.json'
            checkpoint = folder/f'{arm}.pt'
            replay_path = ROOT/f'work/neural-demand-audit-2026-10-17/replays/{seed}-{arm}.npz'
            cfg = EnvironmentConfig(**json.loads(config_path.read_text(encoding='utf-8'))['environment'])
            checkpoint_before = sha(checkpoint)
            for path in (config_path, replay_path):
                inputs.append({'path':str(path.relative_to(ROOT)), 'sha256':sha(path)})
            with np.load(replay_path, allow_pickle=False) as replay:
                P, B, E = [replay[k].astype(np.float64) for k in ('P','B','E')]
                assert np.array_equal(P, P.T) and np.linalg.eigvalsh(P).min() > 0
                assert np.array_equal(B[:2], np.zeros(2)) and E[0] == E[2] == 0
                schur = P[:2, :2]-np.outer(P[:2,2], P[:2,2])/P[2,2]
                native = object.__new__(ResponseServo)
                native.config = cfg
                native.B = torch.tensor(B, dtype=torch.float64)
                for tick in TICKS:
                    pool_path = ROOT/f'work/closed-loop-interface-2026-10-17/pools/{seed}-{arm}-tick{tick}.npz'
                    inputs.append({'path':str(pool_path.relative_to(ROOT)), 'sha256':sha(pool_path)})
                    with np.load(pool_path, allow_pickle=False) as pool:
                        valid = ((pool['reference_domain_margin_max'] <= 0)
                                 & (pool['actual_domain_margin_max'] <= 1e-7)
                                 & (pool['input_margin_max'] <= 0)
                                 & (pool['tube_margin_max'] <= TOL)
                                 & np.isfinite(pool['actual_cost']) & np.isfinite(pool['reference_cost']))
                    for env in range(16):
                        x, z, raw = [replay[k][tick,env].astype(np.float64) for k in ('x','z','raw_design')]
                        lo_ref, hi_ref, ref_candidates = reference_extrema(z,raw,cfg)
                        for value,c,eta in ref_candidates:
                            _, original = native.response(torch.tensor(z),torch.tensor(c,dtype=torch.float64),
                                                          torch.tensor(eta,dtype=torch.float64),torch.tensor(raw))
                            checks.append(abs(float(original)-value))
                            reference_checks += 1
                        center = drift(x,cfg)-drift(z,cfg)
                        lower = center[2]+B[2]*(-cfg.input_bound-hi_ref)
                        upper = center[2]+B[2]*(cfg.input_bound-lo_ref)
                        nominal, action_error, nerr, nevals = scalar_minimax(center[None,:2],P,lower,upper)
                        robust_eo = center[None,:2]+np.array([[-cfg.disturbance_bound],[cfg.disturbance_bound]])*E[None,:2]
                        robust, robust_action_error, rerr, revals = scalar_minimax(robust_eo,P,lower,upper)
                        scalar_evals += nevals+revals
                        unbounded = float(center[:2]@schur@center[:2])
                        initial = float((x-z)@P@(x-z))
                        budget = cfg.tube_budget+TOL
                        # A larger free-input class cannot rule out any known feasible policy candidate.
                        assert not (valid[env].any() and nominal > budget+1e-12)
                        assert nominal+1e-12 >= unbounded and robust+1e-12 >= nominal
                        # Reconstruct a bounded u, uref pair attaining the scalar minimizer.
                        diff = (action_error-center[2])/B[2]
                        uref_witness = max(lo_ref, -cfg.input_bound-diff)
                        u_witness = uref_witness+diff
                        assert uref_witness <= hi_ref+1e-12 and abs(u_witness) <= cfg.input_bound+1e-12
                        rows.append({'key':f'{seed}/{arm}/{tick}/{env}', 'seed':seed,'arm':arm,'tick':tick,'env':env,
                                     'pool_empty':bool(not valid[env].any()),'initial_V':initial,
                                     'initial_tube_failed':bool(initial > budget),'gate_budget':budget,
                                     'uref_range':[lo_ref,hi_ref],'free_actuator_error_range':[float(lower),float(upper)],
                                     'schur_min_V':unbounded,'nominal_min_V':nominal,'robust_min_V':robust,
                                     'nominal_minimizing_ea':action_error,'robust_minimizing_ea':robust_action_error,
                                     'nominal_witness_u':float(u_witness),'nominal_witness_uref':float(uref_witness),
                                     'nominal_impossible_next_tube':bool(nominal>budget),
                                     'schur_impossible_next_tube':bool(unbounded>budget),
                                     'robust_impossible_next_tube':bool(robust>budget),
                                     'independent_nominal_solver_error':nerr,'independent_robust_solver_error':rerr})
            assert checkpoint_before == sha(checkpoint)
            inputs.append({'path':str(checkpoint.relative_to(ROOT)),'sha256':checkpoint_before,'unchanged':True,
                           'loaded_as_policy':False})
    assert protocol_hash == sha(protocol)
    report = {'date':'2026-10-17','scope':'One-step causal free-input outer relaxation, not policy optimization or a complete task proof',
              'protocol_sha256':protocol_hash,'script_sha256':sha(Path(__file__)), 'inputs':inputs,
              'information':'Nominal w=0 and pre-disturbance minmax w in [-.04,.04] reported separately; no hidden phase read',
              'bounds':'All continuous c/eta in original domain; u free within original U. Current u/command chosen before w. Other domain/continuation constraints relaxed.',
              'floating_arithmetic':'Analytic global minima for declared scalar problems evaluated in float64; not an interval-arithmetic certificate',
              'all':summarize(rows),
              'support':summarize([r for r in rows if r['arm'].endswith('_support')]),
              'penalty':summarize([r for r in rows if r['arm'].endswith('_penalty')]),
              'by_case':{f'{seed}/{arm}':summarize([r for r in rows if r['seed']==seed and r['arm']==arm]) for seed in SEEDS for arm in ARMS},
              'validation':{'max_native_reference_value_error':max(checks),
                            'max_independent_scalar_value_error':max(max(r['independent_nominal_solver_error'],r['independent_robust_solver_error']) for r in rows),
                            'positive_definite_saved_metrics':True,'known_feasible_candidates_respected':True},
              'budget':{'queries':len(rows),'analytic_scalar_problems':2*len(rows),
                        'independent_scalar_objective_evaluations':scalar_evals,
                        'original_reference_one_step_checks':reference_checks,
                        'actor_evaluations':0,'environment_transitions':0,'planner_calls':0,'training_updates':0,
                        'checkpoint_writes':0,'wall_seconds':time.perf_counter()-started}, 'queries':rows}
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'one-step-authority.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('all','validation','budget')},indent=2))


if __name__ == '__main__':
    main()
