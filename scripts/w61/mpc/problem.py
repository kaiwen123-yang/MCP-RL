"""Reduced Go2 support-transfer prediction for topology-conditioned MPC.

The observation is a causal Stable F plan input. The three terrain scenarios
share one physical command, and differ only in terrain realization. Distinct
world footprints retain their Kelvin-Voigt unloading state over the transfer.
"""
import numpy as np
import jax
import jax.numpy as jnp

DT = .025
H = 14
DIM = 16
MODES = ('FL_RR', 'FR_RL', 'FL_then_RR', 'FR_then_RL')
SCENARIOS = ('rigid', 'kv8000', 'kv3500')
SLOPE = .5 / np.sqrt(2)
WEIGHTS = np.array([30, 30, 10, 1, 1, 100, 1, 1, 1, 10, 10, 20.])
CONTACT_AREA = .001
CELL = .02


def build_problem(obs, speed, capacity, probabilities, preload, swing_duration=.20):
    """Construct one fixed transfer from four measured contacts at publication.

    x0 = [world RPY, COM m, world omega rad/s, COM velocity m/s]. Input J
    orders feet FL,FR,RL,RR and columns free base then each leg's 3 joints.
    preload is a prescribed initial indentation in metres for the soft patches.
    """
    if not np.asarray(obs['contact'][0]).all():
        raise ValueError('A prospective transfer begins on four confirmed feet')
    if not np.isclose(sum(probabilities), 1.) or min(probabilities) <= 0:
        raise ValueError('The common belief must be an interior probability vector')
    x = np.asarray(obs['x0'], dtype=float)
    feet = np.asarray(obs['feet'][0], dtype=float)
    direction = np.array([np.cos(x[2]), np.sin(x[2])])
    step = direction * speed * .40
    stages = np.arange(H)
    refs = np.broadcast_to(x, (H, 12)).copy()
    refs[:, :2] = 0.
    refs[:, 6:9] = 0.
    refs[:, 9:11] = direction * speed
    refs[:, 11] = 0.
    refs[:, 3:5] = x[None, 3:5] + DT * (stages[:, None]+1) * refs[:, 9:11]
    refs[:, 5] = obs['reference'][0, 5]
    contacts = np.ones((len(MODES), H, 4), dtype=bool)
    points = np.broadcast_to(feet, (len(MODES), H, 4, 3)).copy()
    landed = np.zeros_like(contacts)
    for mode, pair in enumerate(((0, 3), (1, 2), (0, 3), (1, 2))):
        for j, leg in enumerate(pair):
            start = .05 + (.05*j if mode >= 2 else 0.)
            tau = np.clip((stages*DT-start)/swing_duration, 0., 1.)
            moving = (stages*DT >= start-1.e-9) & (tau < 1.-1.e-9)
            blend = tau**3*(10.-15.*tau+6.*tau**2)
            points[mode, :, leg, :2] += blend[:, None]*step
            points[mode, :, leg, 2] += .03*64.*(tau*(1.-tau))**3
            landed[mode, :, leg] = tau >= 1.-1.e-9
            contacts[mode, :, leg] = ~moving
    # With this workspace the four legs have mutually disjoint old/new footprints.
    # The shifted sole can overlap its previous footprint. Retain its shared
    # world cells and exact area fractions for loading and free recovery.
    half = np.sqrt(CONTACT_AREA)/2
    masks = np.zeros((4, 2, 20))
    soft = np.zeros((4, 2), bool)
    for leg in range(4):
        ids = {}
        for endpoint, xy in enumerate((feet[leg, :2], feet[leg, :2]+step)):
            if not (.8+half <= xy[0] <= 2.2-half and abs(xy[1]) <= .6-half):
                continue
            soft[leg, endpoint] = True
            lo, hi = xy-half, xy+half
            for i in range(int(np.floor(lo[0]/CELL)), int(np.floor(hi[0]/CELL))+1):
                for j in range(int(np.floor(lo[1]/CELL)), int(np.floor(hi[1]/CELL))+1):
                    square = CELL*np.array([i, j])
                    a = np.prod(np.maximum(0., np.minimum(hi, square+CELL)-np.maximum(lo, square)))
                    if a > 0:
                        k = ids.setdefault((i,j), len(ids))
                        masks[leg, endpoint, k] = a / CONTACT_AREA
    jac = np.asarray(obs['J_contact_force'])
    jac = np.stack([jac[l, :, 6+3*l:9+3*l] for l in range(4)])
    lo = np.asarray(obs['torque_lower']).reshape(4,3)*np.asarray(capacity)[:,None]
    hi = np.asarray(obs['torque_upper']).reshape(4,3)*np.asarray(capacity)[:,None]
    bias = np.asarray(obs['h'])[6:].reshape(4,3)
    if np.any(bias < lo) or np.any(bias > hi):
        raise ValueError('Posture bias is outside the declared joint authority')
    return dict(x=x, ref=refs, feet=points, contact=contacts, landed=landed,
                area=masks, soft=soft, initial_delta=(masks[:,0]>0)*preload,
                J=jac, bias=bias, lo=lo, hi=hi, I=np.asarray(obs['inertia']),
                invI=np.linalg.inv(obs['inertia']), mass=float(obs['mass']),
                prev=np.asarray(obs['last_force']), belief=np.array(probabilities),
                w=np.sqrt(WEIGHTS/H), height=refs[0,5])


def _world_euler_rate(rpy, omega):
    yaw, pitch = rpy[2], rpy[1]
    a = jnp.cos(yaw)*omega[0] + jnp.sin(yaw)*omega[1]
    return jnp.array([a/jnp.cos(pitch), -jnp.sin(yaw)*omega[0]+jnp.cos(yaw)*omega[1],
                      a*jnp.tan(pitch)+omega[2]])


def rollout(p, z, mode, scenario):
    # z is sampled before tanh; prior and actual proposal densities use these
    # identical coordinates. Clipped actuator outputs never acquire Gaussian density.
    k = jnp.array([8000., 8000., 3500.])[scenario]
    d = 400.
    kleg = 24000.
    side = jnp.array([1., -1., 1., -1.])
    end = jnp.tanh(z.reshape(2, 8))
    def advance(carry, t):
        x, delta, prev = carry
        on = p['contact'][mode, t]
        idx = p['landed'][mode, t].astype(jnp.int32)
        areas = p['area'][jnp.arange(4), idx]
        soft = p['soft'][jnp.arange(4), idx] & (scenario != 0)
        points = p['feet'][mode,t]
        a = end[0] + (end[1]-end[0])*(t/(H-1))
        f0 = p['mass'] * jnp.maximum(0., 9.81 + 45.*(p['height']-x[5])-8.*x[11]) / on.sum()
        desired = f0 * (1.+.8*a[:4])
        # Reference impedance uses a *common*, fixed material belief. This
        # command is causal state feedback, not a different open-loop decision
        # using the hidden scenario. Nominal inverse KV adds command feedforward.
        mapped_soft = p['soft'][jnp.arange(4),idx]
        # KV with a virgin reference state; the rollout itself retains history.
        compliance = (p['belief'][1]/(8000.+d/DT) + p['belief'][2]/(3500.+d/DT))
        command = desired*(1./kleg + mapped_soft*compliance)
        free = delta/(1.+k*DT/d)
        # Per-sole monotone balance with one common displacement at all
        # intersecting cells. Active-set iteration solves the piecewise-linear
        # traction equation, never an average of independently inverted columns.
        stiffness=k+d/DT
        def balance(_, disp):
            active = (disp[:,None] > free) & (areas>0.)
            slopes = stiffness*areas*active
            return (kleg*command + jnp.sum(slopes*free,axis=1))/(kleg+jnp.sum(slopes,axis=1))
        disp = jax.lax.fori_loop(0, 9, balance, command)
        trial_z = jnp.where(soft, (areas*stiffness*jnp.maximum(disp[:,None]-free,0.)).sum(1),
                            kleg*command)
        velerr=p['ref'][t,9:11]-x[9:11]
        fx = jnp.tanh(5.*velerr[0]+2.*a[4]+side*a[6])
        fy = jnp.tanh(5.*velerr[1]+2.*a[5]+side*a[7])
        direction = jnp.stack((SLOPE*fx,SLOPE*fy,jnp.ones(4)),axis=1)
        jt = jnp.einsum('lij,li->lj',p['J'],direction)
        # Torque tau=bias-J^T f. A saturated command reduces requested penetration
        # itself. The final constrained load obeys the same constitutive equation.
        affine_lim = jnp.where(jt>0.,(p['bias']-p['lo'])/jnp.maximum(jt,1.e-12),
                               (p['hi']-p['bias'])/jnp.maximum(-jt,1.e-12))
        fz = jnp.where(on,jnp.minimum(trial_z,affine_lim.min(1)),0.)
        force = fz[:,None]*direction
        def inverse(_, current):
            active=(current[:,None]>=free-1.e-12) & (areas>0)
            slopes=areas*stiffness*active
            denom=slopes.sum(1)
            return jnp.where(denom>0,(fz+(slopes*free).sum(1))/jnp.maximum(denom,1.e-20),current)
        final_disp=jax.lax.fori_loop(0,9,inverse,disp)
        loaded=(areas>0) & soft[:,None] & on[:,None] & (fz[:,None]>0.) & (final_disp[:,None]>free)
        delta=jnp.where(loaded,final_disp[:,None],free)
        application=points.at[:,2].add(-jnp.where(soft & on & (fz>0.),final_disp,0.))
        net=force.sum(0)/p['mass']-jnp.array([0.,0.,9.81])
        moment=jnp.cross(application-x[3:6],force).sum(0)-jnp.cross(x[6:9],p['I']@x[6:9])
        alpha=p['invI']@moment
        angle=x[:3]+DT*_world_euler_rate(x[:3],x[6:9]+.5*DT*alpha)
        xnext=jnp.concatenate((angle,x[3:6]+DT*(x[9:12]+.5*DT*net),x[6:9]+DT*alpha,x[9:12]+DT*net))
        task=(xnext-p['ref'][t])*p['w']*jnp.where(t==H-1,jnp.sqrt(2.),1.)
        residual=jnp.concatenate((task, (force*jnp.sqrt(1.e-5/H)).ravel(),
                                 ((force-prev)*jnp.sqrt(1.e-4/H)).ravel()))
        tau=p['bias']-jt*fz[:,None]
        return (xnext,delta,force),(residual,xnext,force,tau,command,final_disp,delta.max(axis=1))
    initial=(p['x'],p['initial_delta'],p['prev'])
    return jax.lax.scan(advance,initial,jnp.arange(H))[1]


def evaluate(p,z,m,s):
    residual=rollout(p,z,m,s)[0]
    return jnp.vdot(residual,residual)
