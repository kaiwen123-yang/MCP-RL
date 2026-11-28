"""Batched contact-mode x command x material prediction with finite within-mode refinement."""
import time
import jax
import jax.numpy as jnp
import numpy as np
from .problem import DIM, MODES, SCENARIOS, rollout
from .proposal import finite_proposal
from .weighted import weighted_proposal


@jax.jit
def batch_residual(problem, latent, modes):
    weights=jnp.sqrt(problem['belief'])
    def mode_residual(vectors,mode):
        def one(z):
            r=jax.vmap(lambda scenario:rollout(problem,z,mode,scenario)[0])(jnp.arange(len(SCENARIOS)))
            return (r*weights[:,None,None]).ravel()
        return jax.vmap(one)(vectors)
    return jax.vmap(mode_residual)(latent,modes)


@jax.jit
def output(problem,latent,mode):
    projections=jax.vmap(lambda s:rollout(problem,latent,mode,s))(jnp.arange(len(SCENARIOS)))
    return projections[1:4]


def search(problem,method,key,warm=None,candidates=12,iterations=2,mode_ids=None,proposal_state=None):
    """The prior, proposal, and refinement use the same pre-saturation coordinates."""
    start=time.perf_counter()
    modes=np.asarray(range(len(MODES)) if mode_ids is None else mode_ids,dtype='i4')
    noise=np.random.default_rng(key).standard_normal((iterations,len(modes),candidates,DIM))
    center=np.zeros((len(modes),DIM)) if warm is None else np.asarray(warm)
    p=jax.tree.map(jnp.asarray,problem)
    ids=jnp.asarray(modes)
    search_times=[]
    def observe(z):
        tick=time.perf_counter()
        residual=np.asarray(batch_residual(p,jnp.asarray(z),ids))
        search_times.append(time.perf_counter()-tick)
        return residual
    if method in ('mppi_mean','quadratic_mppi'):
        sol=weighted_proposal(center,observe,noise,method=method,state=proposal_state)
    else:
        sol=finite_proposal(center,observe,noise,method=method)
    sol['mode']=int(modes[sol['mode']])
    states,forces,tau=map(np.asarray,output(p,jnp.asarray(sol['z']),jnp.asarray(sol['mode'])))
    sol.update(states=states,forces=forces,tau=tau,force_reference=np.einsum('s,stlc->tlc',problem['belief'],forces),
               search_wall_s=time.perf_counter()-start,transfer_rollout_refinement_sync_s=search_times,
               maximum_torque_violation_Nm=float(max(0.,(tau-problem['hi']).max(),(problem['lo']-tau).max())))
    return sol
