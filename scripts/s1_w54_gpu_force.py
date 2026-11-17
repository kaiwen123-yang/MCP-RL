"""W54 C: a complete FP64, per-knot force horizon solved on JAX GPU.

The sole optimization method is 4096 accelerated projected-gradient steps.
All state transitions, condensing, objective gradients, exact Euclidean
friction-pyramid projections, and final trajectory evaluation run on GPU.
Inputs are F's corrected, causal SRB contract. This is not articulated WBC,
a material simulation, or an online replacement for either physical arm.
"""
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
ITERATIONS = 4096


def project_force(value, contact, upper, mu):
    """Exact projection onto |fx|,|fy|<=mu*fz, 0<=fz<=upper, or zero.

    The scalar normal optimum has three intervals according to how many
    tangential components saturate. Enumerating those intervals gives the
    Euclidean projection, unlike clipping fz before the tangential components.
    Arrays have arbitrary leading batch/horizon dimensions.
    """
    absxy = jnp.abs(value[..., :2])
    bounds = jnp.sort(absxy, axis=-1) / mu
    low = jnp.stack((jnp.zeros_like(upper), bounds[..., 0], bounds[..., 1]), axis=-1)
    high = jnp.minimum(jnp.stack((bounds[..., 0], bounds[..., 1], upper), axis=-1), upper[..., None])
    z0 = value[..., 2]
    raw = jnp.stack(((z0 + mu * jnp.sum(absxy, axis=-1)) / (1 + 2 * mu * mu),
                     (z0 + mu * jnp.max(absxy, axis=-1)) / (1 + mu * mu), z0), axis=-1)
    z = jnp.minimum(jnp.maximum(raw, low), high)
    score = ((z - z0[..., None]) ** 2 +
             jnp.sum(jnp.maximum(absxy[..., None, :] - mu * z[..., :, None], 0.) ** 2, axis=-1))
    score = jnp.where(high >= low, score, jnp.inf)
    best = jnp.argmin(score, axis=-1)
    normal = jnp.take_along_axis(z, best[..., None], axis=-1)[..., 0]
    tangent = jnp.clip(value[..., :2], -mu * normal[..., None], mu * normal[..., None])
    result = jnp.concatenate((tangent, normal[..., None]), axis=-1)
    return jnp.where(contact[..., None], result, 0.)


def condense(A, B, c, x0, reference, last_force, weights, reg, difference):
    """Construct the exact full-horizon force Hessian and affine state map."""
    batch, n, nx, nf = B.shape
    nu = n * nf
    selector = jnp.eye(nu, dtype=B.dtype).reshape(n, nf, nu)
    start_map = jnp.zeros((batch, nx, nu), dtype=B.dtype)

    def transition(carry, args):
        affine, mapping = carry
        ak, bk, ck, select = args
        affine = jnp.einsum("bij,bj->bi", ak, affine) + ck
        mapping = ak @ mapping + bk @ select
        return (affine, mapping), (affine, mapping)

    _, (after, maps) = jax.lax.scan(
        transition, (x0, start_map),
        (jnp.swapaxes(A, 0, 1), jnp.swapaxes(B, 0, 1), jnp.swapaxes(c, 0, 1), selector))
    affine = jnp.concatenate((x0[:, None], jnp.swapaxes(after, 0, 1)), axis=1)
    mapping = jnp.concatenate((start_map[:, None], jnp.swapaxes(maps, 0, 1)), axis=1)
    w = jnp.tile(weights, n + 1).at[-nx:].multiply(2.)
    flat_map = mapping.reshape(batch, (n + 1) * nx, nu)
    flat_delta = (affine - reference).reshape(batch, (n + 1) * nx)
    rate = jnp.eye(nu, dtype=B.dtype) - jnp.eye(nu, k=-nf, dtype=B.dtype)
    H = 2. * (jnp.einsum("bki,k,bkj->bij", flat_map, w, flat_map) +
              reg * jnp.eye(nu, dtype=B.dtype) + difference * (rate.T @ rate))
    g = 2. * jnp.einsum("bki,k,bk->bi", flat_map, w, flat_delta)
    g = g.at[:, :nf].add(-2. * difference * last_force.reshape(batch, nf))
    lipschitz = jnp.max(jnp.sum(jnp.abs(H), axis=-1), axis=-1)
    return H, g, lipschitz, affine, mapping, w


@partial(jax.jit, static_argnames=("iterations",))
def solve_horizon(A, B, c, x0, reference, last_force, contact, upper,
                  weights, reg, difference, mu, iterations=ITERATIONS):
    batch, n, _, nf = B.shape
    H, g, L, affine, mapping, w = condense(
        A, B, c, x0, reference, last_force, weights, reg, difference)
    shape = (batch, n, 4, 3)
    start = project_force(jnp.broadcast_to(last_force[:, None], shape), contact, upper, mu).reshape(batch, -1)

    def iteration(carry, unused):
        previous, accelerated, momentum = carry
        grad = jnp.einsum("bij,bj->bi", H, accelerated) + g
        current = project_force(
            (accelerated - grad / L[:, None]).reshape(shape), contact, upper, mu).reshape(batch, -1)
        next_momentum = (1. + jnp.sqrt(1. + 4. * momentum * momentum)) / 2.
        next_accelerated = current + ((momentum - 1.) / next_momentum) * (current - previous)
        return (current, next_accelerated, next_momentum), None

    (flat, _, _), _ = jax.lax.scan(iteration, (start, start, jnp.asarray(1., dtype=B.dtype)),
                                  xs=None, length=iterations)
    forces = flat.reshape(shape)
    states = affine + jnp.einsum("bkiu,bu->bki", mapping, flat)
    error = (states - reference).reshape(batch, -1)
    differences = jnp.concatenate((forces[:, :1] - last_force[:, None],
                                   forces[:, 1:] - forces[:, :-1]), axis=1)
    objective = jnp.sum(w[None] * error * error, axis=-1)
    objective += reg * jnp.sum(forces * forces, axis=(1, 2, 3))
    objective += difference * jnp.sum(differences * differences, axis=(1, 2, 3))
    grad = jnp.einsum("bij,bj->bi", H, flat) + g
    projected = project_force((flat - grad / L[:, None]).reshape(shape), contact, upper, mu)
    residual = L * jnp.max(jnp.abs(flat - projected.reshape(batch, -1)), axis=-1)
    return dict(forces=forces, first_force=forces[:, 0], states=states,
                objective=objective, projected_gradient_inf=residual,
                lipschitz_upper=L)
