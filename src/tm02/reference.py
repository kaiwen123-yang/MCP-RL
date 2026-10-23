"""Independent analytical 3D reference, Eqs.26-30 and Table 1 (MPa,s)."""
import numpy as np
from scipy.interpolate import CubicSpline

METRIC = np.array([1., 1., 1., 2., 2., 2.])


class IntegrationFailure(RuntimeError):
    def __init__(self, step, residual, iterations):
        super().__init__(f'Newton failed at step {step}: residual={residual:.6g} MPa after {iterations} iterations')
        self.step, self.residual, self.iterations = step, float(residual), int(iterations)


def newton(fun, jac, initial, settings, step):
    x = initial.copy()
    r = np.asarray(fun(x))
    for iteration in range(settings['max_iterations']+1):
        error = float(np.max(np.abs(r)))
        if np.isfinite(error) and error <= settings['absolute_tolerance_MPa']:
            return x, error, iteration
        if not np.isfinite(error) or iteration == settings['max_iterations']:
            raise IntegrationFailure(step, error, iteration)
        direction = np.linalg.solve(np.asarray(jac(x)), -r)
        merit = np.dot(r, r)
        for backtrack in range(settings['max_backtracks']+1):
            alpha = 0.5**backtrack
            candidate = x+alpha*direction
            rc = np.asarray(fun(candidate))
            if np.dot(rc, rc) <= (1-settings['armijo']*alpha)*merit:
                break
        else:
            raise IntegrationFailure(step, error, iteration)
        x, r = candidate, rc


def stiffness(k, g):
    spherical = np.zeros((6, 6))
    spherical[:3, :3] = 1/3
    return 3*k*spherical + 2*g*(np.eye(6)-spherical)


class Reference:
    def __init__(self, p):
        self.ceq = stiffness(p['K_eq_MPa'], p['G_eq_MPa'])
        self.cov = stiffness(p['K_ov_MPa'], p['G_ov_MPa'])
        self.v0 = stiffness(p['eta_K_table_value'], p['eta_D_table_value'])
        self.a, self.o, self.b = p['a_MPa'], p['o'], p['b']

    def stress(self, e, q):
        return self.ceq @ e + self.cov @ (e-q)

    def residual_jacobian(self, q, e, qprev, dt, de_dq):
        p = e-q
        sov = self.cov @ p
        square = np.dot(sov*METRIC, sov)/self.a**2
        # Fixed Table 1 b=2; avoids a spurious norm derivative at zero.
        g = (1-self.o)*np.exp(-square)
        f = g+self.o
        r = (q-qprev)/dt
        vr = self.v0 @ r
        ds_dq = self.cov @ (de_dq-np.eye(6))
        df_dq = -2*g*(sov*METRIC) @ ds_dq / self.a**2
        return f*vr-sov, f*self.v0/dt+np.outer(vr, df_dq)-ds_dq

    def path(self, time, command, settings, stress_control=False):
        n = len(time)
        e, q, stress = (np.zeros((n, 6)) for _ in range(3))
        residual, iterations = np.zeros(n), np.zeros(n, int)
        de_dq = np.linalg.solve(self.ceq+self.cov, self.cov) if stress_control else np.zeros((6, 6))
        for i in range(1, n):
            dt = time[i]-time[i-1]
            if stress_control:
                origin = np.linalg.solve(self.ceq+self.cov, command[i])
                strain = lambda qi: origin+de_dq @ qi
            else:
                strain = lambda qi: command[i]
            def both(qi):
                return self.residual_jacobian(qi, strain(qi), q[i-1], dt, de_dq)
            q[i], residual[i], iterations[i] = newton(
                lambda qi: both(qi)[0], lambda qi: both(qi)[1], q[i-1], settings, i)
            e[i] = strain(q[i])
            stress[i] = self.stress(e[i], q[i])
        return dict(time=time, strain=e, stress=stress, q=q,
                    residual=residual, newton_iterations=iterations)


def random_path(config, evaluation=False):
    p = config['evaluation'] if evaluation else config['train']
    rng = np.random.default_rng(p['strain_seed'])
    increments = p['increments']
    # Draw sample clock first, then knots. This ordering is a fixed project choice.
    dt = np.full(increments, p['dt_s']) if evaluation else rng.uniform(*p['dt_uniform_s'], increments)
    time = np.r_[0., np.cumsum(dt)]
    kt, ke = [0.], [np.zeros(6)]
    while kt[-1] < time[-1]:
        kt.append(kt[-1]+rng.uniform(*config['spline_knot_dt_s']))
        increment = rng.normal(0, config['spline_knot_strain_increment_normal_std'], 6)
        outside = np.abs(ke[-1]+increment) > config['knot_strain_absolute_bound']
        while np.any(outside):
            increment[outside] = rng.normal(0, config['spline_knot_strain_increment_normal_std'], outside.sum())
            outside = np.abs(ke[-1]+increment) > config['knot_strain_absolute_bound']
        ke.append(ke[-1]+increment)
    kt, ke = np.array(kt), np.array(ke)
    strain = CubicSpline(kt, ke, bc_type=config['spline_boundary_condition'])(time)
    return time, strain, kt, ke
