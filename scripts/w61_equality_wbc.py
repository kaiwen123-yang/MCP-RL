"""Solve exactly the frozen WBC in coordinates spanning its affine equality set."""
import json
import numpy as np
from scipy import sparse
from s1_w54_f_r52_backend import ControllerNumericalFailure


class EqualityReducedWBC:
    def _solve(self,P,q,A,lo,hi,kind,t,warm=None):
        if kind!='wbc':
            return super()._solve(P,q,A,lo,hi,kind,t,warm)
        equal=lo==hi
        E,b=A[equal],lo[equal]
        C,lower,upper=A[~equal],lo[~equal],hi[~equal]
        # Full right basis gives the tangent and the normal solution in one SVD.
        U,s,Vt=np.linalg.svd(E,full_matrices=True)
        tol=max(E.shape)*np.finfo(float).eps*s[0]
        rank=int(np.count_nonzero(s>tol))
        origin=Vt[:rank].T@((U[:,:rank].T@b)/s[:rank])
        N=Vt[rank:].T
        projected=C@N;length=np.linalg.norm(projected,axis=1);offset=C@origin
        moving=length>1.e-12
        constant_violation=max(np.max(lower[~moving]-offset[~moving],initial=0.),
                               np.max(offset[~moving]-upper[~moving],initial=0.))
        eps=self.cfg['qp']['eps_abs']
        if constant_violation>eps or np.linalg.norm(E@origin-b,np.inf)>eps:
            raise ControllerNumericalFailure(f'W61 WBC affine equality inconsistency at {t:.6f}')
        scale=length[moving]
        initial=N.T@(warm-origin) if warm is not None else None
        try:
            reduced,stats=super()._solve(N.T@P@N,N.T@(P@origin+q),projected[moving]/scale[:,None],
                (lower[moving]-offset[moving])/scale,(upper[moving]-offset[moving])/scale,kind,t,initial)
        except ControllerNumericalFailure:
            name=self.record_dir/f'original-wbc-{t:.6f}'
            sparse.save_npz(str(name)+'-P.npz',sparse.csc_matrix(P))
            sparse.save_npz(str(name)+'-A.npz',sparse.csc_matrix(A))
            np.savez(str(name)+'.npz',q=q,lower=lo,upper=hi,origin=origin,tangent=N,inequality_scale=scale)
            raise
        x=origin+N@reduced
        y=A@x;z=np.maximum(lo,np.minimum(hi,y))
        violation=float(np.max(np.abs(y-z)))
        tol_primal=eps+self.cfg['qp']['eps_rel']*max(np.max(np.abs(y)),np.max(np.abs(z)))
        factor=10. if stats['status_val']==2 else 1.
        if violation>factor*tol_primal:
            raise ControllerNumericalFailure(f'W61 WBC original-row residual {violation:.6g} at {t:.6f}')
        stats.update(objective=float(.5*x@P@x+q@x),physical_max_row_violation=violation,
                     original_primal_tolerance=tol_primal,equality_rank=rank,tangent_dim=N.shape[1])
        filename=self.record_dir/'w61-equality-solve.jsonl'
        with filename.open('a') as stream:
            stream.write(json.dumps(dict(time=t,**stats))+'\n')
        return x,stats
