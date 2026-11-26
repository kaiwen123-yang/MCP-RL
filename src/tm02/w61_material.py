"""Isotropic small-strain family PANN with a causal backward-Euler material history.

Tensor component order is (11,22,33,12,13,23), including physical shear strain.
Only the material identity is adapted; every specimen has its own branch strains.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

IDENTITY = np.array([1., 1., 1., 0., 0., 0.])
METRIC = np.array([1., 1., 1., 2., 2., 2.])


def split_strain(e):
    trace = np.sum(e[..., :3], axis=-1, keepdims=True)
    return trace, e - trace * IDENTITY / 3


def stress_basis(e, history, strain_scale):
    """Each column is the tensor derivative of a nonnegative energy basis."""
    volume, dev = split_strain(e)
    ov_volume, ov_dev = split_strain(e[..., None, :] - history)
    squared_dev = np.sum(dev**2 * METRIC, axis=-1, keepdims=True)
    terms = [volume * IDENTITY, 2 * dev, volume**3 / strain_scale**2 * IDENTITY,
             2 * dev * squared_dev / strain_scale**2]
    for i in range(history.shape[-2]):
        terms.extend([ov_volume[..., i, :] * IDENTITY, 2 * ov_dev[..., i, :]])
    return np.stack(terms, axis=-1)


@dataclass
class MaterialResponse:
    """Fixed condition. Coefficients are in MPa and branch state in dimensionless strain."""
    coefficients: np.ndarray
    tau: np.ndarray
    strain_scale: float

    def step(self, strain, previous, dt):
        a = (dt / (self.tau + dt))[:, None]
        state = previous + a * (np.asarray(strain)[..., None, :] - previous)
        return stress_basis(strain, state, self.strain_scale) @ self.coefficients, state

    def virgin(self):
        return np.zeros((len(self.tau), 6))

    def indicators(self):
        # Static tangent at the unloaded, relaxed state. The relaxation excess
        # after a small hydrostatic strain step is measured relative to its equilibrium.
        h = 1.e-6
        hydro = h * IDENTITY / 3
        shear = h * np.array([0., 0., 0., 1., 0., 0.])
        equilibrium = lambda e: stress_basis(e, np.broadcast_to(e, self.virgin().shape),
                                               self.strain_scale) @ self.coefficients
        k = (equilibrium(hydro)[0] - equilibrium(-hydro)[0]) / (2 * h)
        g = (equilibrium(shear)[3] - equilibrium(-shear)[3]) / (4 * h)
        t = np.linspace(0, 12., 2401)
        state = (1 - np.exp(-t[:, None, None] / self.tau[None, :, None])) * hydro
        step = stress_basis(np.broadcast_to(hydro, (len(t), 6)), state,
                            self.strain_scale) @ self.coefficients
        excess = step[:, 0] - equilibrium(hydro)[0]
        return dict(relaxed_bulk_MPa=float(k), relaxed_shear_MPa=float(g),
                    hydro_relaxation_area_time_0_to_12_s=float(np.trapezoid(excess / excess[0], t)))


def features(strain, dt, config):
    """Same branchwise BE integration as MaterialResponse.step, batched over whole histories."""
    history = np.zeros((*strain.shape[:-2], len(config['tau_s']), 6))
    states = []
    for i in range(strain.shape[-2]):
        delta = dt[..., i, None, None]
        history = history + delta / (delta + np.asarray(config['tau_s'])[:, None]) * (
                  strain[..., i, None, :] - history)
        states.append(history)
    raw = stress_basis(strain, np.stack(states, axis=-3), config['strain_scale'])
    return (raw * np.asarray(config['stiffness_scale_MPa']) / config['stress_scale_MPa']).astype('f')


def sufficient_statistics(x, y):
    x = x.flatten(1, -2)
    y = y.flatten(1)
    return (x.transpose(1, 2) @ x) / x.shape[1], (x.transpose(1, 2) @ y[..., None]).squeeze(-1) / x.shape[1]


class Family(nn.Module):
    def __init__(self, config, method):
        super().__init__()
        self.config, self.method = config, method
        size = config['latent_dim']
        self.encoder = nn.Sequential(nn.Linear(32, 24), nn.Tanh(), nn.Linear(24, size))
        center = torch.ones(16)
        center[2:4] = .05
        self.base = nn.Parameter(torch.log(torch.expm1(center)))
        if method == 'embedding_response':
            self.hidden = nn.Linear(size, 10)
            self.readout = nn.Parameter(.18 * torch.randn(10, 16))
        else:
            rank = config['conditional_rank']
            self.gate = nn.Linear(size, rank)
            self.left = nn.Parameter(.25 * torch.randn(4, rank))
            self.right = nn.Parameter(.25 * torch.randn(rank, 4))
        if method == 'lowrank_anil':
            self.log_inner_lr = nn.Parameter(torch.tensor(2.5))

    def weights_and_jacobian(self, code):
        if self.method == 'embedding_response':
            hidden = self.hidden(code).tanh()
            logits = self.base + hidden @ self.readout
            jac = torch.einsum('bh,hi,hc->bic', 1-hidden.square(), self.readout, self.hidden.weight)
        else:
            gates = self.gate(code).tanh()
            factors = torch.einsum('ur,rv->ruv', self.left, self.right).flatten(1)
            logits = self.base + gates @ factors
            jac = torch.einsum('br,ri,rc->bic', 1-gates.square(), factors, self.gate.weight)
        return F.softplus(logits), logits.sigmoid()[..., None] * jac

    def adapt(self, gram, rhs, probe):
        # Normalized contractions of the support histories only.
        code = self.encoder(torch.cat([rhs, gram.diagonal(dim1=-2, dim2=-1)], dim=-1))
        cfg = self.config
        eye = torch.eye(cfg['latent_dim'], device=gram.device)
        for _ in range(cfg['inner_steps']):
            weights, jac = self.weights_and_jacobian(code)
            jt = jac.transpose(1, 2)
            grad = jt @ ((gram @ weights[..., None]) - rhs[..., None])
            if self.method == 'lowrank_anil':
                code = code - F.softplus(self.log_inner_lr) * grad.squeeze(-1)
                continue
            hessian = jt @ gram @ jac + cfg['inner_damping'] * eye
            if self.method == 'lowrank_isotropic_gn':
                hessian = hessian + cfg['probe_metric_lambda'] * .005 * eye
                delta = torch.linalg.solve(hessian, -grad)
            else:
                metric = jt @ probe @ jac
                delta = torch.linalg.solve(hessian + cfg['probe_metric_lambda'] * metric, -grad)
                rms = ((delta.transpose(1, 2) @ metric @ delta).clamp_min(1.e-15)).sqrt()
                delta = delta / (rms / cfg['probe_step_rms_limit_in_10MPa']).clamp_min(1)
            code = code + delta.squeeze(-1)
        return self.weights_and_jacobian(code)[0]

    def infer(self, support_strain, dt, measured_stress, probe_gram):
        """Produce a publication from support histories; future calls use only current input and state."""
        x = torch.from_numpy(features(support_strain, dt, self.config))[None]
        y = torch.from_numpy(np.asarray(measured_stress, dtype='f'))[None] / self.config['stress_scale_MPa']
        with torch.no_grad():
            w = self.adapt(*sufficient_statistics(x, y), torch.as_tensor(probe_gram))[0].numpy()
        return MaterialResponse(w * self.config['stiffness_scale_MPa'], np.array(self.config['tau_s']),
                                self.config['strain_scale'])


class HistoryGRU(nn.Module):
    """Support-conditional GRU with a physics spectrum skip and a causal recurrent stress residual."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 32))
        self.spectrum = nn.Linear(32, 16)
        self.gru = nn.GRU(13, 32, batch_first=True)
        self.out = nn.Linear(32, 6)

    def forward(self, gram, rhs, features_q, inputs):
        identity = self.encoder(torch.cat([rhs, gram.diagonal(dim1=-2, dim2=-1)], -1))
        batch, paths, length, _ = inputs.shape
        start = identity[:, None].expand(-1, paths, -1).reshape(1, batch*paths, 32).contiguous()
        states, _ = self.gru(inputs.flatten(0, 1), start)
        residual = self.out(states).reshape(batch, paths, length, 6)
        base = torch.einsum('bptck,bk->bptc', features_q, F.softplus(self.spectrum(identity)))
        return base + residual - residual[:, :, :1]
