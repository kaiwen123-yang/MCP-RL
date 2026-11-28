"""Finite-area, traction-free column response for a published W61-A identity."""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import lsq_linear

from .w61_material import Family, MaterialResponse, features


def adapt_from_coupons(checkpoint, method, seed, strain, dt, stress_MPa):
    """Identify one material from full observed coupons. Does not consume query or ground truth."""
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = data['config']
    if method == 'spectrum_nnls':
        x = features(np.asarray(strain), np.asarray(dt), cfg).reshape(-1, 16).astype(float)
        fit = lsq_linear(x, np.asarray(stress_MPa).ravel()/cfg['stress_scale_MPa'],
                         bounds=(0,np.inf), method='bvls')
        return MaterialResponse(fit.x*np.asarray(cfg['stiffness_scale_MPa']), np.array(cfg['tau_s']),
                                cfg['strain_scale'])
    net = Family(cfg, method)
    net.load_state_dict(data['state_dicts'][f'{method}_{seed}'])
    return net.infer(strain, dt, stress_MPa, data['probe_gram'])


@dataclass
class ColumnHistory:
    strain: np.ndarray
    branch_strain: np.ndarray

    @classmethod
    def virgin(cls, n_cells, n_modes):
        return cls(np.zeros((n_cells,2)), np.zeros((n_cells,n_modes,2)))


class FamilyColumns:
    """Independent world-fixed columns with zero lateral stress and shaft order (11,22=33)."""
    def __init__(self, response):
        self.response = response

    def virgin(self, n_cells):
        return ColumnHistory.virgin(n_cells, len(self.response.tau))

    def step(self, history, normal, dt, displacement):
        """Prescribe epsilon11 for true mask entries and sigma11 [MPa] elsewhere.

        Previous history is read-only. Return a new material state, sigma11/sigma22,
        and maximum boundary residual per column.
        """
        w = self.response.coefficients
        inv_strain2 = 1 / self.response.strain_scale**2
        retain = self.response.tau / (self.response.tau + dt)
        k_mode, g_mode = w[4::2]*retain, w[5::2]*retain
        k = w[0] + k_mode.sum()
        g = 2*(w[1] + g_mode.sum())
        qn, ql = history.branch_strain[...,0],history.branch_strain[...,1]
        pv = (qn+2*ql) @ k_mode
        pd = (qn-ql) @ (2*g_mode)
        k3, g3 = w[2]*inv_strain2, (4/3)*w[3]*inv_strain2

        def response(t, d):
            vol = k*t + k3*t**3 - pv
            shear = g*d + g3*d**3 - pd
            return vol, shear, k + 3*k3*t*t, g + 3*g3*d*d

        # Scalar monotone equations are independent under prescribed normal stress;
        # prescribed normal strain locks t = en+2el, d = en-el.
        en, el = history.strain.T.copy()
        en = np.where(displacement, normal, en)
        for _ in range(20):
            t, d = en+2*el, en-el
            vol,shear,dv,dd = response(t,d)
            sn, sl = vol + (2/3)*shear, vol - shear/3
            residual = np.maximum(abs(sl),np.where(displacement,0.,abs(sn-normal)))
            if residual.max() <= 1.e-9:
                break
            el_fixed = el - sl/(2*dv+dd/3)
            t_free = t - (vol-normal/3)/dv
            d_free = d - (shear-normal)/dd
            el = np.where(displacement,el_fixed,(t_free-d_free)/3)
            en = np.where(displacement,normal,(t_free+2*d_free)/3)
        else:
            raise ArithmeticError(f'W61 family mixed boundary Newton residual {residual.max():.6g} MPa')
        strain = np.column_stack((en,el))
        state = history.branch_strain + (1-retain)[None,:,None]*(strain[:,None,:]-history.branch_strain)
        return ColumnHistory(strain,state),np.column_stack((sn,sl)),residual

    def contact(self, history, sole_bottom_m, areas_m2, dt, thickness_m=.05, gamma=.025):
        """One pressure per covered world cell from the common sole boundary; separation is free.

        Supply one bottom and exact intersected area per world key. Area zero releases a
        cell while its material memory evolves; use separate ColumnHistory per observer.
        """
        n = len(sole_bottom_m)
        zero = np.zeros(n)
        free, _, free_res = self.step(history, zero, dt, np.zeros(n,dtype=bool))
        normal = np.asarray(sole_bottom_m)/thickness_m
        candidates = (areas_m2 > 0) & (normal <= free.strain[:,0])
        new, stress, residual = self.step(history,np.where(candidates,normal,0.),dt,candidates)
        contact = candidates & (stress[:,0] < 0)
        state = ColumnHistory(np.where(contact[:,None],new.strain,free.strain),
                              np.where(contact[:,None,None],new.branch_strain,free.branch_strain))
        force = np.where(contact,-stress[:,0]*1.e6*gamma*areas_m2,0.)
        return state,force,np.maximum(residual,free_res)
