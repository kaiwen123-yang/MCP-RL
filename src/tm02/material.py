"""Rosenkranz et al., arXiv:2401.14270v1, Sections 3/4 and Appendix A.

All six-vectors use physical tensor components (11,22,33,12,13,23),
not engineering shear. Derivatives therefore divide shear entries by two.
"""
import jax
import jax.numpy as jnp
import numpy as np

COMPONENTS = ("11", "22", "33", "12", "13", "23")
METRIC = np.array([1., 1., 1., 2., 2., 2.])


def tensor(x):
    return jnp.array([[x[0], x[3], x[4]],
                      [x[3], x[1], x[5]],
                      [x[4], x[5], x[2]]])


def invariants(x, fourth=True):
    s = tensor(x)
    s2 = s @ s
    return jnp.array([jnp.trace(s), jnp.trace(s2),
                      jnp.sum(s2*s2) if fourth else jnp.sum(s2*s)])


def ficnn(w, x):
    # Appendix A.1, Eqs.31-32: nonnegative invariant and hidden weights.
    z = jax.nn.softplus(x @ jax.nn.softplus(w['wi']) + w['b'])
    return (z @ jax.nn.softplus(w['wo'])
            + x @ jax.nn.softplus(w['skip']) + w['bo']).squeeze()


def picnn(w, x, y):
    # Appendix A.2, Eqs.34-36; u is convex, v is unrestricted.
    v = jnp.tanh(y @ w['v'] + w['bv'])
    gate1 = jax.nn.softplus(y @ w['g1'] + w['bg1'])
    u = jax.nn.softplus((x*gate1) @ jax.nn.softplus(w['u1'])
                        + y @ w['y1'] + w['b1'])
    gateu = jax.nn.softplus(v @ w['gu'] + w['bgu'])
    gatex = jax.nn.softplus(v @ w['gx'] + w['bgx'])
    return ((u*gateu) @ jax.nn.softplus(w['u2'])
            + (x*gatex) @ jax.nn.softplus(w['x2'])
            + v @ w['y2'] + w['b2']).squeeze()


def energy(w, e, norm):
    inv = invariants(e)
    x = (inv-norm['inv_m']) / norm['inv_s']
    zero = -norm['inv_m'] / norm['inv_s']
    raw0 = ficnn(w, zero)
    slope0 = jax.grad(ficnn, 1)(w, zero)[0] / norm['inv_s'][0]
    # Eqs.8-12: constants and trace-linear correction at physical zero.
    return norm['energy_s'] * (ficnn(w, x)-raw0-slope0*inv[0])


def dissipation(w, rate, p, norm):
    x = (invariants(rate)-norm['rate_inv_m']) / norm['rate_inv_s']
    y = (invariants(p, False)-norm['state_inv_m']) / norm['state_inv_s']
    zero = -norm['rate_inv_m'] / norm['rate_inv_s']
    raw0 = picnn(w, zero, y)
    slope0 = jax.grad(picnn, 1)(w, zero, y)[0] / norm['rate_inv_s'][0]
    # Eqs.14-15: correction at zero rate FOR EACH current p.
    return norm['dissipation_s'] * (
        picnn(w, x, y)-raw0-slope0*jnp.sum(rate[:3]))


def forces(w, norm, e, q, rate):
    metric = jnp.asarray(METRIC, dtype=e.dtype)
    p = e-q
    eq = jax.grad(energy, 1)(w['eq'], e, norm)/metric
    tau = jax.grad(energy, 1)(w['ov'], p, norm)/metric
    tauhat = jax.grad(dissipation, 1)(w['phi'], rate, p, norm)/metric
    return eq+tau, tau, tauhat


def auxiliary(w, norm, e, stress, dt):
    # Section 4.3: 13 inputs; standard i,f,g,o gates; 50 -> 6 linear head.
    x = jnp.concatenate(((e-norm['e_m'])/norm['e_s'],
                         (stress-norm['s_m'])/norm['s_s'],
                         ((dt-norm['dt_m'])/norm['dt_s'])[:, None]), axis=1)
    def cell(carry, xt):
        h, c = carry
        gates = xt @ w['wx'] + h @ w['wh'] + w['b']
        i, f, g, o = jnp.split(gates, 4)
        c = jax.nn.sigmoid(f)*c + jax.nn.sigmoid(i)*jnp.tanh(g)
        h = jax.nn.sigmoid(o)*jnp.tanh(c)
        return (h, c), (h @ w['head']+w['bh'])*norm['e_s']+norm['e_m']
    zero = jnp.zeros(50, dtype=e.dtype)
    _, q = jax.lax.scan(cell, (zero, zero), x)
    qprev = jnp.concatenate((jnp.zeros((1, 6), dtype=e.dtype), q[:-1]))
    return q, (q-qprev)/dt[:, None]


def loss(w, norm, e, stress, dt):
    q, rate = auxiliary(w['rnn'], norm, e, stress, dt)
    pred, tau, tauhat = jax.vmap(forces, (None, None, 0, 0, 0))(w, norm, e, q, rate)
    # Eqs.23-25 (MPa): no FNN division by the stress normalization scale.
    terms = jnp.array([jnp.mean(jnp.abs(pred-stress)),
                       jnp.mean(jnp.abs(tau-tauhat))])
    return jnp.sum(terms), terms


def initialize(seed):
    rng = np.random.default_rng(seed)
    def mat(a, b, positive=False):
        limit = np.sqrt(6/(a+b))
        x = rng.uniform(-limit, limit, (a, b)).astype(np.float32)
        if positive:
            x = np.log(np.expm1(np.abs(x)+0.01)).astype(np.float32)
        return x
    def zero(n):
        return np.zeros(n, np.float32)
    def fi():
        return dict(wi=mat(3, 10, True), b=zero(10),
                    wo=mat(10, 1, True), skip=mat(3, 1, True), bo=zero(1))
    phi = dict(v=mat(3, 10), bv=zero(10), g1=mat(3, 3), bg1=zero(3),
               u1=mat(3, 10, True), y1=mat(3, 10), b1=zero(10),
               gu=mat(10, 10), bgu=zero(10), gx=mat(10, 3), bgx=zero(3),
               u2=mat(10, 1, True), x2=mat(3, 1, True), y2=mat(10, 1), b2=zero(1))
    rnn = dict(wx=mat(13, 200), wh=mat(50, 200), b=zero(200),
               head=mat(50, 6), bh=zero(6))
    return dict(eq=fi(), ov=fi(), phi=phi, rnn=rnn)


def normalization(e, stress, dt):
    def bounds(x, axis=None):
        lo, hi = np.min(x, axis=axis), np.max(x, axis=axis)
        return (hi+lo)/2, (hi-lo)/2
    e_m, e_s = bounds(e)
    s_m, s_s = bounds(stress)
    dt_m, dt_s = bounds(dt)
    # Independent histories have their own t0; difference before pooling.
    rate = np.diff(e, axis=-2)/dt[..., None]
    e = e.reshape(-1, 6)
    rate = rate.reshape(-1, 6)
    inv = np.asarray(jax.vmap(invariants)(e))
    rinv = np.asarray(jax.vmap(invariants)(rate))
    sinv = np.asarray(jax.vmap(lambda x: invariants(x, False))(e))
    im, isc = bounds(inv, 0)
    rm, rs = bounds(rinv, 0)
    sm, ss = bounds(sinv, 0)
    _, rate_s = bounds(rate)
    # A.3: q and p scales are estimated solely from epsilon, not true q.
    return dict(e_m=e_m, e_s=e_s, s_m=s_m, s_s=s_s, dt_m=dt_m, dt_s=dt_s,
                inv_m=im, inv_s=isc, rate_inv_m=rm, rate_inv_s=rs,
                state_inv_m=sm, state_inv_s=ss, energy_s=e_s*s_s,
                dissipation_s=rate_s*s_s)


def save_weights(path, w, norm):
    arrays = {f'{net}/{key}': np.asarray(value) for net, par in w.items()
              for key, value in par.items()}
    arrays.update({f'norm/{key}': np.asarray(value) for key, value in norm.items()})
    np.savez_compressed(path, **arrays)


def load_weights(path, include_rnn=False):
    z = np.load(path)
    w = {net: {key.split('/')[1]: z[key].astype(np.float64)
               for key in z.files if key.startswith(net+'/')}
         for net in ('eq', 'ov', 'phi') + (('rnn',) if include_rnn else ())}
    norm = {key.split('/')[1]: z[key].astype(np.float64)
            for key in z.files if key.startswith('norm/')}
    return w, norm
