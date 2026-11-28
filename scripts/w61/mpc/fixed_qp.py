"""Fixed-contact force QP using finite ADMM and an augmented-state Riccati recursion."""
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve


def project_force(force,mask,cap,mu):
    # Analytic Euclidean projection onto a truncated four-sided friction pyramid.
    p=force.reshape(-1,4,3)
    a=jnp.abs(p[...,:2])
    hi,lo=a.max(-1),a.min(-1)
    z=(p[...,2]+mu*(hi+lo))/(1.+2*mu**2)
    z=jnp.where(mu*z>=lo,(p[...,2]+mu*hi)/(1.+mu**2),z)
    z=jnp.where(mu*z>=hi,p[...,2],z)
    z=jnp.minimum(jnp.maximum(z,0.),cap)
    xy=jnp.clip(p[...,:2],-mu*z[...,None],mu*z[...,None])
    return (jnp.concatenate((xy,z[...,None]),-1)*mask[...,None]).reshape(force.shape)


def riccati_factors(p,rho):
    n=len(p['A'])
    identity=jnp.eye(12)
    zero=jnp.zeros((12,12))
    Qa=jnp.block([[jnp.diag(p['qw']),zero],[zero,p['D']*identity]])
    terminal=jnp.block([[2*jnp.diag(p['qw']),zero],[zero,zero]])
    cross=jnp.concatenate((zero,-p['D']*identity),axis=0)
    a=jnp.pad(p['A'],((0,0),(0,12),(0,12)))
    b=jnp.concatenate((p['B'],jnp.broadcast_to(identity,(n,12,12))),axis=1)
    c=jnp.pad(p['c'],((0,0),(0,12)))
    r=(p['R']+p['D']+.5*rho)*identity
    def backward(nextP,st):
        A,B,offset=st
        L=jnp.linalg.cholesky(r+B.T@nextP@B)
        coupling=B.T@nextP@A+cross.T
        K=cho_solve((L,True),coupling)
        P=Qa+A.T@nextP@A-coupling.T@K
        return .5*(P+P.T),(L,K,coupling,nextP@offset)
    _,(L,K,C,affine)=jax.lax.scan(backward,terminal,(a,b,c),reverse=True)
    q=jnp.pad(-p['ref'][:-1]*p['qw'],((0,0),(0,12)))
    final=jnp.pad(-2*p['qw']*p['ref'][-1],(0,12))
    return (L,K,C,affine,a,b,c,q,final)


def fixed_solve(p,n_iterations,rho):
    L,K,C,affine,a,b,c,q,final=riccati_factors(p,rho)
    weight=.5*rho
    initial=jnp.concatenate((p['x0'],p['prev'].ravel()))
    z0=project_force(jnp.broadcast_to(p['prev'].ravel(),(len(c),12)),p['contact'],p['cap'],p['mu'])
    def iteration(_,current):
        z,u,_,_=current
        def backwards(adj,stage):
            chol,cross,A,B,off,g,center=stage
            nxt=off+adj
            k=cho_solve((chol,True),B.T@nxt-weight*center)
            return g+A.T@nxt-cross.T@k,k
        _,ks=jax.lax.scan(backwards,final,(L,C,a,b,affine,q,z-u),reverse=True)
        def forwards(x,stage):
            gain,feed,A,B,off=stage
            f=-gain@x-feed
            return A@x+B@f+off,f
        _,f=jax.lax.scan(forwards,initial,(K,ks,a,b,c))
        projected=project_force(f+u,p['contact'],p['cap'],p['mu'])
        return projected,u+f-projected,jnp.max(jnp.abs(f-projected)),rho*jnp.max(jnp.abs(projected-z))
    z,_,primal,dual=jax.lax.fori_loop(0,n_iterations,iteration,(z0,jnp.zeros_like(z0),jnp.array(0.),jnp.array(0.)))
    def propagate(x,stage):
        A,B,off,f=stage
        nxt=A@x+B@f+off
        return nxt,nxt
    _,x=jax.lax.scan(propagate,p['x0'],(p['A'],p['B'],p['c'],z))
    return jnp.concatenate((p['x0'][None],x)),z.reshape(-1,4,3),primal,dual


@partial(jax.jit, static_argnums=(1,))
def solve_batch(problems,n_iterations,rho):
    return jax.vmap(lambda p: fixed_solve(p,n_iterations,rho))(problems)
