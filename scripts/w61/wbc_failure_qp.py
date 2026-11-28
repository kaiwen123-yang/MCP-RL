"""W61 D0: replay the frozen linear-quadratic problem and identify feasibility."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import osqp
from scipy import linalg, optimize, sparse
import scipy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def violation(A, x, lower, upper):
    value = A @ x
    return float(max(0., np.max(lower-value,initial=0.), np.max(value-upper,initial=0.)))


def solve(P, q, A, lower, upper, settings):
    solver = osqp.OSQP()
    solver.setup(P=sparse.triu(sparse.csc_matrix(P), format="csc"), q=q,
                 A=sparse.csc_matrix(A), l=lower, u=upper, verbose=False,
                 eps_abs=settings['eps_abs'], eps_rel=settings['eps_rel'],
                 max_iter=settings['max_iter'], polishing=True, adaptive_rho=True,
                 adaptive_rho_interval=50, check_termination=25, warm_starting=True)
    return solver.solve(raise_error=False)


def summary(sol):
    info = sol.info
    return dict(status=info.status, iterations=info.iter, primal=info.prim_res,
                dual=info.dual_res, objective=info.obj_val, solve_s=info.solve_time)


def norm_span(values):
    v = np.asarray(values)
    return [float(v[v>0].min()), float(v.max())]


def analyze(run, source, saved_original):
    directory = run/'w55-f'
    stems = sorted(directory.glob('failed-wbc-*-P.npz'))
    [p_path] = stems
    stem = p_path.name[:-6]
    a_path, b_path = directory/(stem+'-A.npz'), directory/(stem+'.npz')
    P, A = sparse.load_npz(p_path).toarray(), sparse.load_npz(a_path).toarray()
    with np.load(b_path) as data:
        q, low, high = data['q'], data['lower'], data['upper']
    failure = json.loads((directory/'solver-failure.json').read_text())
    config = json.loads((directory/'manifest.json').read_text())['config']['qp']
    # The W59 WBC passes no x or y warm start and creates a fresh solver each tick.
    original = None if saved_original else solve(P,q,A,low,high,config)
    equality = low == high
    E, b = A[equality], high[equality]
    C, lower, upper = A[~equality], low[~equality], high[~equality]
    has_upper, has_lower = np.isfinite(upper), np.isfinite(lower)
    lp = optimize.linprog(np.zeros(len(q)), A_ub=np.vstack((C[has_upper],-C[has_lower])),
        b_ub=np.r_[upper[has_upper],-lower[has_lower]], A_eq=E, b_eq=b,
        bounds=[(None,None)]*len(q), method='highs')
    # Eliminate exactly the original equalities. Scale only nonzero inequality
    # row normals. Both operations preserve the physical feasible region.
    x0 = np.linalg.lstsq(E,b,rcond=None)[0]
    N = linalg.null_space(E)
    rows = C@N
    lengths=np.linalg.norm(rows,axis=1)
    movable=lengths>1e-12
    if violation(C[~movable],x0,lower[~movable],upper[~movable])>1.e-9:
        raise ValueError('An eliminated constant inequality is infeasible')
    rscale=lengths[movable]
    reduced=solve(N.T@P@N,N.T@(P@x0+q),rows[movable]/rscale[:,None],
                  (lower[movable]-(C@x0)[movable])/rscale,
                  (upper[movable]-(C@x0)[movable])/rscale,config)
    x=x0+N@reduced.x
    dual_c=np.zeros(len(C))
    dual_c[movable]=reduced.y/rscale
    grad=P@x+q+C.T@dual_c
    dual_e=np.linalg.lstsq(E.T,-grad,rcond=None)[0]
    values=A@x
    tight=np.flatnonzero((~equality)&(np.minimum(high-values,values-low)<1.e-5))
    latest=sorted(directory.glob('control-*.npz'))[-1]
    with np.load(latest) as c:
        prev={k:c[k][-1].tolist() for k in ('time','planned_contact','observed_contact',
             'active_wbc_contact','confirmed_contact','load_bearing_contact','solver_iterations','solver_primal_residual','solver_dual_residual')}
    return {
      'attempt':run.name,'stored_failure':failure,'source_commit':source,'osqp':osqp.__version__,
      'scipy':scipy.__version__,'frozen_snapshot_sha256':{p.name:sha(p) for p in (p_path,a_path,b_path)},
      'settings':config,'explicit_warm_start':None,'record_dir':str(directory),
      'rows':len(low),'columns':len(q),'equality_rows':int(sum(equality)),'equality_rank':int(np.linalg.matrix_rank(E)),
      'P_eigenvalue_span':norm_span(np.linalg.eigvalsh(P)), 'A_row_norm_span':norm_span(np.linalg.norm(A,axis=1)),
      'previous_successful_tick':prev,'failed_tick_contact_from_force_bounds':[
          bool(high[154+3*i+2]>0) for i in range(4)],
      'original_once':summary(original) if original is not None else None,
      'original_replay_note':('One replay was executed, but following analysis failed before its result was serialized. Stored W59 residuals are the observed baseline.' if saved_original else 'One replay preserved'),
      'original_maximum_physical_violation':violation(A,original.x,low,high) if original is not None else None,
      'linear_feasibility':{'status':str(lp.message),'maximum_physical_violation':violation(A,lp.x,low,high) if lp.success else None,
                            'feasible_witness':lp.x.tolist() if lp.success else None},
      'equality_elimination_same_budget':{**summary(reduced),'constant_rows_eliminated':int(sum(~movable)),
        'mapped_objective':float(.5*x@P@x+q@x),'maximum_physical_violation':violation(A,x,low,high),
        'physical_stationarity_inf':float(np.linalg.norm(grad+E.T@dual_e,np.inf)), 'active_inequality_rows_at_1e-5':tight.tolist(),
        'primal_solution_original_coordinates':x.tolist()},
      'scope':'One saved W59 QP. Modified numerical coordinates require closed-loop evaluation before controller adoption.'
    }


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--frozen-source',required=True)
    p.add_argument('--use-stored-original',action='store_true')
    args=p.parse_args()
    result=analyze(args.run,args.frozen_source,args.use_stored_original)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('original_once','linear_feasibility','equality_elimination_same_budget')}))
