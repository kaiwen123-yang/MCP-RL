"""Matched physical MPPI controls returning the weighted mean for each executable topology."""
import numpy as np


def weighted_proposal(warm,evaluate,noise,*,method,sigma=.35,temperature=.12,state=None):
    eps=np.asarray(noise,dtype=float)
    rounds,modes,n,dim=eps.shape
    eye=np.eye(dim)
    if state is None:
        mu=np.asarray(warm,dtype=float).copy()
        scales=np.broadcast_to(sigma*eye,(modes,dim,dim)).copy()
    else:
        mu,scales=map(np.array,state)
    improvements=[]
    for _k in range(rounds):
        offsets=eps[_k]@scales.transpose(0,2,1)
        sampled=mu[:,None,:]+offsets
        at=np.concatenate((mu[:,None,:],sampled),axis=1)
        resp=np.asarray(evaluate(at))
        cost=np.einsum('mnr,mnr->mn',resp,resp)
        # The residual importance correction applies the full target density,
        # including when the quadratic proposal changes both mean and covariance.
        logw=-cost[:,1:]/temperature-.5*(sampled/sigma**2*sampled).sum(-1)+.5*(eps[_k]**2).sum(-1)
        weights=np.exp(logw-logw.max(axis=1,keepdims=True))
        weights/=weights.sum(axis=1,keepdims=True)
        physical_mean=np.einsum('mn,mnd->md',weights,sampled)
        checked=np.asarray(evaluate(physical_mean[:,None,:]))
        checked_cost=np.einsum('mnr,mnr->mn',checked,checked).ravel()
        if not np.isfinite(checked_cost).all():
            raise FloatingPointError('Non-finite MPPI mean rollout')
        improvements.append((checked_cost<cost[:,0]).tolist())
        if method=='mppi_mean':
            mu=physical_mean.copy()
        elif method=='quadratic_mppi':
            for m in range(modes):
                # q proportional to the Gaussian reference times a local quadratic
                # least-squares task model; the next importance weights use the
                # actually sampled bounded-variance Gaussian and the true cost.
                dz=offsets[m]
                dy=resp[m,1:]-resp[m,:1]
                J=np.linalg.solve(dz.T@dz+1.e-4*eye,dz.T@dy)
                precision=eye/sigma**2+2./temperature*(J@J.T)
                eig,vec=np.linalg.eigh(precision)
                cov=(vec*np.clip(1./eig,.15*sigma**2,sigma**2))@vec.T
                scales[m]=np.linalg.cholesky(cov)
                shift=np.linalg.solve(precision,mu[m]/sigma**2+2./temperature*(J@resp[m,0]))
                mu[m]-=shift*min(1.,.55/max(np.linalg.norm(shift),1.e-15))
        else:
            raise ValueError(method)
    best=int(np.argmin(checked_cost))
    return dict(mode=best,z=physical_mean[best],per_mode=physical_mean,per_mode_cost=checked_cost,
                proposal_state=(mu,scales),evaluated_modes=modes,evaluated_candidates=rounds*modes*(n+2),
                refinement_improved_center=improvements)
