"""Finite topology-conditioned sampling and regularized local residual refinement."""
import numpy as np


def finite_proposal(warm, evaluate, noise, *, method='hybrid', sigma=.35,
                    temperature=.12, damping=.04, trust=.55):
    """Search independent executable modes. evaluate receives and returns [mode,candidate,...].

    All real decisions are in unconstrained coordinates; the caller maps them to
    commands before scoring. Ordinary and variance-reduced MPPI use N(0,sigma²I)
    as their reference measure. noise is fixed [iteration,mode,samples,coordinate].
    The callback returns physical, weighted least-squares residual vectors, including
    the common material-scenario risk. Budget is M*(K*(N+2)) vector evaluations.
    """
    means=np.asarray(warm,dtype=float).copy()
    eps=np.asarray(noise,dtype=float)
    rounds,modes,n,dim=eps.shape
    if means.shape!=(modes,dim) or method not in ('mppi','vr_mppi','local','hybrid'):
        raise ValueError('Topology/coordinate shape or search method incompatible')
    eye=np.eye(dim)
    scale=np.broadcast_to(sigma*eye,(modes,dim,dim)).copy()
    best=means.copy()
    best_cost=np.full(modes,np.inf)
    accepted=[]
    for k in range(rounds):
        delta=eps[k] @ scale.transpose(0,2,1)
        cloud=means[:,None,:]+delta
        all_z=np.concatenate((means[:,None,:],cloud),axis=1)
        residual=np.asarray(evaluate(all_z),dtype=float)
        cost=np.einsum('mnr,mnr->mn',residual,residual)
        if not np.isfinite(cost).all():
            raise FloatingPointError('Non-finite sampled response in finite-budget search')
        log_w=-cost[:,1:]/temperature
        # Importance ratio p0(z)/q(z); determinant is constant within one topology.
        log_w+=-.5*((cloud/sigma)**2).sum(-1)+.5*(eps[k]**2).sum(-1)
        log_w-=log_w.max(axis=1,keepdims=True)
        weights=np.exp(log_w)
        weights/=weights.sum(axis=1,keepdims=True)
        candidate=np.einsum('mn,mnd->md',weights,cloud)
        for m in range(modes):
            if method in ('hybrid','local'):
                # Locally fit each physical residual from the finite candidate budget.
                # Damping is expressed in residual/latent units and the proposed
                # step is checked with an actual rollout, including active-set changes.
                anchor=(np.argmin(cost[m]) if method=='hybrid' else 0)
                steps=all_z[m]-all_z[m,anchor]
                changes=residual[m]-residual[m,anchor]
                jac=np.linalg.solve(steps.T@steps+1e-4*eye,steps.T@changes)
                rhs=jac@residual[m,anchor]
                dz=-np.linalg.solve(jac@jac.T+damping*eye,rhs)
                radius=np.linalg.norm(dz)
                candidate[m]=all_z[m,anchor]+dz*min(1.,trust/max(radius,1e-15))
            elif method=='vr_mppi':
                diff=cloud[m]-candidate[m]
                cov=(diff.T*weights[m])@diff+.15*sigma**2*eye
                eig,vec=np.linalg.eigh(cov)
                scale[m]=(vec*np.sqrt(np.clip(eig,.20*sigma**2,2.*sigma**2)))@vec.T
        checked=np.asarray(evaluate(candidate[:,None,:]),dtype=float)
        new_cost=np.einsum('mnr,mnr->mn',checked,checked).ravel()
        if not np.isfinite(new_cost).all():
            raise FloatingPointError('Non-finite refinement in finite-budget search')
        all_cost=np.column_stack((cost,new_cost))
        all_z=np.concatenate((all_z,candidate[:,None,:]),axis=1)
        pick=all_cost.argmin(axis=1)
        value=all_cost[np.arange(modes),pick]
        better=value<best_cost
        best[better]=all_z[np.arange(modes),pick][better]
        best_cost=np.minimum(best_cost,value)
        accepted.append((new_cost<cost[:,0]).tolist())
        means=(candidate if method in ('mppi','vr_mppi') else best).copy()
    which=int(np.argmin(best_cost))
    return dict(mode=which,z=best[which],per_mode=best,per_mode_cost=best_cost,
                evaluated_modes=modes,evaluated_candidates=rounds*modes*(n+2),
                refinement_improved_center=accepted)
