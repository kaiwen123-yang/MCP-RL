"""Incremental axisymmetric TM02 mixed boundaries and unilateral KV columns.

Isotropy and a virgin isotropic state imply e22=e33, q22=q33 and zero
shear under a normal boundary. The two retained Biot equations, lateral
stress=0 and normal displacement/stress equation are the exact invariant
subspace of the six-component material, not a constrained-lateral modulus.
"""
import numpy as np
from .reference import IntegrationFailure


def reduced_stiffness(k, g):
    return np.array([[k+4*g/3, 2*k-4*g/3], [k-2*g/3, 2*k+2*g/3]])


class ReferenceColumns:
    def __init__(self, parameters):
        p = parameters
        self.ce = reduced_stiffness(p['K_eq_MPa'], p['G_eq_MPa'])
        self.co = reduced_stiffness(p['K_ov_MPa'], p['G_ov_MPa'])
        self.v = reduced_stiffness(p['eta_K_table_value'], p['eta_D_table_value'])
        self.ct = self.ce+self.co
        self.free_map = np.linalg.solve(self.ct, self.co)
        self.mixed_map = np.zeros((2, 2))
        self.mixed_map[1] = self.co[1]/self.ct[1, 1]
        self.a, self.o = p['a_MPa'], p['o']

    def step(self, eprev, qprev, normal, dt, displacement):
        """normal is epsilon11 for displacement, sigma11 [original MPa] otherwise."""
        n = len(normal)
        if n == 0:
            return eprev.copy(), qprev.copy(), np.empty((0, 2)), np.empty(0)
        maps = np.where(displacement[:, None, None], self.mixed_map, self.free_map)
        origin_free = np.linalg.solve(self.ct, np.vstack((normal, np.zeros(n)))).T
        origin_fixed = np.column_stack((normal, -self.ct[1, 0]/self.ct[1, 1]*normal))
        origin = np.where(displacement[:, None], origin_fixed, origin_free)
        derivative = np.einsum('ij,njk->nik', self.co, maps-np.eye(2))
        q = qprev.copy()
        def residual(qi):
            e = origin + np.einsum('nij,nj->ni', maps, qi)
            s = (e-qi)@self.co.T
            g = (1-self.o)*np.exp(-np.sum(s*s*np.array([1., 2.]), axis=1)/self.a**2)
            visc = (qi-qprev)@self.v.T/dt
            return (g+self.o)[:, None]*visc-s, e, s, g, visc
        for iteration in range(31):
            r, e, s, g, visc = residual(q)
            err = np.max(np.abs(r), axis=1)
            if np.max(err) <= 1e-8:
                return e, q, e@self.ce.T+(e-q)@self.co.T, err
            if iteration == 30 or not np.all(np.isfinite(r)):
                raise IntegrationFailure(-1, np.max(err), iteration)
            df = -2*g[:, None]*np.einsum('ni,nij->nj', s*np.array([1., 2.]), derivative)/self.a**2
            jac = (g+self.o)[:, None, None]*self.v[None]/dt + visc[:, :, None]*df[:, None, :] - derivative
            change = np.linalg.solve(jac, -r[..., None])[..., 0]
            alpha = np.ones(n)
            merit = np.sum(r*r, axis=1)
            for backtrack in range(16):
                candidate = q+alpha[:, None]*change
                rc = residual(candidate)[0]
                reject = (np.sum(rc*rc, axis=1) > (1-1e-4*alpha)*merit) & (err>1e-8)
                if not np.any(reject):
                    break
                alpha[reject] *= .5
            q = candidate


class LearnedColumns:
    def __init__(self, weights_path):
        import jax
        import jax.numpy as jnp
        from .material import forces, load_weights
        device = jax.devices('gpu')[0]
        with jax.experimental.enable_x64(), jax.default_device(device):
            w, norm = load_weights(weights_path)
            w, norm = jax.tree.map(jnp.asarray, (w, norm))
        def one(x, qp, normal, dt, displacement):
            e = jnp.array([x[0], x[1], x[1], 0., 0., 0.])
            q = jnp.array([x[2], x[3], x[3], 0., 0., 0.])
            prev = jnp.array([qp[0], qp[1], qp[1], 0., 0., 0.])
            stress, tau, visc = forces(w, norm, e, q, (q-prev)/dt)
            # Scale the displacement residual by the reference instantaneous
            # modulus solely for Newton conditioning; boundary remains exact.
            boundary = jnp.where(displacement, 3000.*(x[0]-normal), stress[0]-normal)
            return jnp.array([boundary, stress[1], visc[0]-tau[0], visc[1]-tau[1]])
        jac = jax.jacfwd(one)
        def solve4(matrix, rhs):
            # Fixed four-dimensional Newton system. Fusing pivoted elimination
            # avoids a separate tiny batched GPU library solve per iteration.
            augmented = jnp.concatenate((matrix, rhs[:,None]), axis=1)
            for column in range(4):
                pivot = jnp.argmax(jnp.where(jnp.arange(4)>=column,jnp.abs(augmented[:,column]),-1.))
                first, second = augmented[column], augmented[pivot]
                augmented = augmented.at[column].set(second).at[pivot].set(first)
                row = augmented[column]/augmented[column,column]
                augmented = augmented-augmented[:,column,None]*row[None,:]
                augmented = augmented.at[column].set(row)
            return augmented[:,4]
        def solve(ep, qp, normal, dt, fixed):
            x = jnp.concatenate((ep, qp))
            def cond(c):
                xx, it = c
                return (jnp.max(jnp.abs(one(xx, qp, normal, dt, fixed)))>1e-8) & (it<30)
            def body(c):
                xx, it = c
                r = one(xx, qp, normal, dt, fixed)
                change = solve4(jac(xx, qp, normal, dt, fixed), -r)
                def bcond(z):
                    a, k = z
                    rr = one(xx+a*change, qp, normal, dt, fixed)
                    return (jnp.dot(rr, rr)>(1-1e-4*a)*jnp.dot(r, r)) & (k<16)
                a, _ = jax.lax.while_loop(bcond, lambda z:(z[0]*.5, z[1]+1), (1., 0))
                return xx+a*change, it+1
            x, _ = jax.lax.while_loop(cond, body, (x, 0))
            e = jnp.array([x[0], x[1], x[1], 0., 0., 0.])
            q = jnp.array([x[2], x[3], x[3], 0., 0., 0.])
            stress = forces(w, norm, e, q, jnp.zeros(6))[0][:2]
            return x[:2], x[2:], stress, jnp.max(jnp.abs(one(x, qp, normal, dt, fixed)))
        self.solve = jax.jit(jax.vmap(solve, in_axes=(0, 0, 0, None, 0)), device=device)
        self.first_call_including_compile_s = None

    def step(self, eprev, qprev, normal, dt, displacement):
        if len(normal) == 0:
            return eprev.copy(), qprev.copy(), np.empty((0, 2)), np.empty(0)
        # Fixed blocks avoid compiling a different Newton solver for every new footprint.
        n = len(normal)
        block = 128
        outputs = [[] for _ in range(4)]
        for start in range(0, n, block):
            count = min(block, n-start)
            pad = lambda a: np.pad(a[start:start+count], ((0, block-count),)+((0, 0),)*(a.ndim-1))
            import jax
            import time
            begin = time.perf_counter()
            with jax.experimental.enable_x64():
                values = self.solve(pad(eprev), pad(qprev), pad(normal), dt, pad(displacement))
            for target, value in zip(outputs, values):
                target.append(np.asarray(value)[:count])
            if self.first_call_including_compile_s is None:
                self.first_call_including_compile_s = time.perf_counter()-begin
        result = tuple(np.concatenate(x) for x in outputs)
        if not np.all(np.isfinite(result[3])) or np.max(result[3])>1e-7:
            raise IntegrationFailure(-1, np.max(result[3]), 30)
        return result
