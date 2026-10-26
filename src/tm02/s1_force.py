"""One retained unilateral/friction map for MPC and held-force execution."""
import numpy as np


def project_contact_force(force, contact, alpha, grf_max, mu, xp=np):
    force = xp.asarray(force).reshape(4, 3)
    cap = xp.asarray(contact) * xp.asarray(alpha) * grf_max
    z = xp.clip(force[:, 2], 0., cap)
    xy = xp.clip(force[:, :2], -mu*z[:, None], mu*z[:, None])
    return xp.concatenate((xy, z[:, None]), axis=1)


def loading_fraction(age, duration):
    u = np.clip(age/duration, 0., 1.)
    return u**3*(10.-15.*u+6.*u*u)
