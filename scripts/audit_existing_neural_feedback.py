"""Read-only derivatives of existing trained PPO actors on verified native replays.

No training/optimizer/rollout is called. Functional parameter probes only check
mixed derivatives; checkpoints remain byte-identical. Native interval geometry
is differentiated, including its contribution to the actor's observations.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from scipy import linalg
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from response_ppo.env import EnvironmentConfig, ResponseServo
from response_ppo.realization import feasible_interval

OUT = ROOT / 'notes/neural-demand-audit-2026-10-17'
WORK = ROOT / 'work/neural-demand-audit-2026-10-17'
SEEDS = (17, 29, 43)
ARMS = ('joint_support', 'fixed_support', 'joint_penalty', 'fixed_penalty')
TIMES = np.arange(0, 256, 32)
INDICES = (TIMES[:, None] * 16 + np.arange(16)).reshape(-1)
PARAMETER_EPS = (1e-4, 1e-5)
KEYS = ('mean.0.weight', 'mean.0.bias', 'mean.2.weight',
        'mean.2.bias', 'mean.4.weight', 'mean.4.bias')
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def mean(flat, obs):
    w0 = flat[:1088].reshape(64, 17)
    b0 = flat[1088:1152]
    w1 = flat[1152:5248].reshape(64, 64)
    b1 = flat[5248:5312]
    w2 = flat[5312:5376].reshape(1, 64)
    b2 = flat[5376:5377]
    return F.linear(torch.tanh(F.linear(torch.tanh(F.linear(obs, w0, b0)), w1, b1)), w2, b2).squeeze(-1)


mean_obs_grad = torch.func.grad(mean, argnums=1)


def feedback(flat, obs, obs_x, mid_x, half, half_x):
    raw = mean(flat, obs)
    squashed = torch.tanh(raw)
    return mid_x + half_x * squashed + half * (1-squashed.square()) * (mean_obs_grad(flat, obs) @ obs_x)


batched_feedback = torch.func.vmap(feedback, in_dims=(None, 0, 0, 0, 0, 0))
batched_mixed = torch.func.vmap(torch.func.jacrev(feedback, argnums=0),
                              in_dims=(None, 0, 0, 0, 0, 0), chunk_size=16)
batched_value_jac = torch.func.vmap(torch.func.jacrev(mean, argnums=0), in_dims=(None, 0), chunk_size=32)


class NativeGeometry:
    def __init__(self, archive, support):
        self.cfg = EnvironmentConfig()
        self.support = support
        self.P = torch.tensor(archive['P'], dtype=torch.float64)
        self.scale = torch.tensor([1., 1., self.cfg.input_bound])
        # Do not construct ResponseServo: its constructor would run a planner.
        self.env = object.__new__(ResponseServo)
        self.env.config = self.cfg
        self.env.P = self.P
        self.env.B = torch.tensor(archive['B'], dtype=torch.float64)
        self.env.E = torch.tensor(archive['E'], dtype=torch.float64)

    def base(self, x, z, command, design, bounds):
        return torch.cat((x/self.scale, z/self.scale, (x-z)/self.scale,
                          command[:, :1]/.45, (command[:, 1:]-1.)/.2,
                          2*torch.sigmoid(design)-1, bounds), -1)

    def interval(self, x, z, command, design):
        znext, _ = self.env.response(z, command[:, 0], command[:, 1], design)
        value = feasible_interval(self.env.f0(x), self.env.B, self.env.E,
                                  znext, x-z, self.P, self.cfg.rho, self.cfg.delta,
                                  self.cfg.disturbance_bound, self.cfg.input_bound)
        return value, znext

    def observations(self, x, z, command, design):
        interval, znext = self.interval(x, z, command, design)
        bounds = torch.stack((interval.lo, interval.hi), -1)
        bounds = torch.where(interval.feasible[:, None], bounds/self.cfg.input_bound, 0.)
        return self.base(x, z, command, design, bounds), interval, znext

    def root_gaps(self, x, z, znext):
        offset = self.env.f0(x)-znext
        ep = torch.stack((offset-self.env.E*self.cfg.disturbance_bound,
                          offset+self.env.E*self.cfg.disturbance_bound), 1)
        bvec = self.env.B
        a = bvec @ self.P @ bvec
        b = 2*torch.einsum('i,ij,bkj->bk', bvec, self.P, ep)
        c = torch.einsum('bki,ij,bkj->bk', ep, self.P, ep)
        c -= (self.cfg.rho*torch.einsum('bi,ij,bj->b', x-z, self.P, x-z)+self.cfg.delta)[:, None]
        disc = b.square()-4*a*c
        sq = torch.sqrt(disc.clamp_min(0))
        stable_q = -.5*(b+torch.where(b>=0, sq, -sq))
        r1 = stable_q/a
        r2 = torch.where(stable_q != 0, c/torch.where(stable_q != 0, stable_q, 1.), -b/(2*a))
        lower = torch.cat((torch.minimum(r1, r2), torch.full((len(x), 1), -self.cfg.input_bound)), -1).sort(-1).values
        upper = torch.cat((torch.maximum(r1, r2), torch.full((len(x), 1), self.cfg.input_bound)), -1).sort(-1).values
        return lower[:, -1]-lower[:, -2], upper[:, 1]-upper[:, 0]

    def geometry(self, x, z, command, design):
        obs, interval, znext = self.observations(x, z, command, design)
        lowgap, highgap = self.root_gaps(x, z, znext)
        regular = (interval.feasible & (interval.hi > interval.lo)
                   & (interval.discriminant > 0).all(-1) & (lowgap > 1e-12) & (highgap > 1e-12))
        ox, mx, hs, hx, usable = [], [], [], [], []
        for i in range(len(x)):
            feasible = bool(interval.feasible[i])
            good = bool(regular[i])
            def combined(state):
                point = state[None]
                if feasible:
                    o, limits, _ = self.observations(point, z[i:i+1], command[i:i+1], design[i:i+1])
                    midpoint = (limits.lo+limits.hi)/2 if self.support else torch.zeros(1)
                    halfwidth = (limits.hi-limits.lo)/2 if self.support else torch.full((1,), self.cfg.input_bound)
                else:
                    o = self.base(point, z[i:i+1], command[i:i+1], design[i:i+1], torch.zeros((1, 2)))
                    if self.support:
                        self.env.x, self.env.batch = point, 1
                        midpoint = self.env.fallback(znext[i:i+1])
                        halfwidth = torch.zeros(1)
                    else:
                        midpoint, halfwidth = torch.zeros(1), torch.full((1,), self.cfg.input_bound)
                return torch.cat((o[0], midpoint, halfwidth))
            if good or not feasible:
                values = combined(x[i])
                dx = torch.autograd.functional.jacobian(combined, x[i], vectorize=True)
                if not torch.isfinite(dx).all():
                    good = False
                    dx = torch.zeros((19, 3))
                # Non-feasible support is an exact theta-independent fallback.
                usable.append(good or not feasible)
                ox.append(dx[:17]*self.scale)
                mx.append(dx[17]*self.scale/self.cfg.input_bound)
                hs.append(values[18]/self.cfg.input_bound)
                hx.append(dx[18]*self.scale/self.cfg.input_bound)
            else:
                usable.append(False)
                ox.append(torch.zeros((17, 3))); mx.append(torch.zeros(3))
                hs.append(torch.tensor(0.)); hx.append(torch.zeros(3))
        return obs, tuple(torch.stack(v) for v in (ox, mx, hs, hx)), {
            'interval_feasible': interval.feasible, 'interval_width': interval.hi-interval.lo,
            'lower_branch_gap': lowgap, 'upper_branch_gap': highgap,
            'regular_feasible': regular, 'mixed_derivative_usable': usable,
            'fallback_count': int((~interval.feasible).sum()) if self.support else 0,
            'nonregular_mixed_excluded': sum(not v for v in usable),
        }


def audit_case(seed, arm):
    start = time.perf_counter()
    checkpoint = ROOT/f'work/ppo-mechanism-2026-10-16/formal/seedrun-{seed}/seed-{seed}/{arm}.pt'
    replay = WORK/f'replays/{seed}-{arm}.npz'
    before_hash = sha(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    p32 = torch.cat([saved['actor'][k].reshape(-1) for k in KEYS])
    assert len(p32) == 5377
    p = p32.double()
    with np.load(replay, allow_pickle=False) as data:
        oldobs = torch.tensor(data['obs'].reshape(-1, 17), dtype=torch.float32)
        oldmean = torch.tensor(data['mean'].reshape(-1), dtype=torch.float32)
        native = [torch.tensor(data[name].reshape(-1, size), dtype=torch.float64)
                  for name, size in (('x', 3), ('z', 3), ('command', 2), ('raw_design', 4))]
        geometry = NativeGeometry(data, arm.endswith('support'))
        oldaction = torch.tensor(data['action'].reshape(-1), dtype=torch.float64)
    # Match the original batch width, avoiding a different float32 GEMM reduction.
    matched32 = torch.cat([mean(p32, oldobs[t:t+16]) for t in range(0, len(oldobs), 16)])
    float32_error = float((matched32-oldmean).abs().max())
    if float32_error != 0:
        raise RuntimeError(f'{seed}/{arm}: checkpoint forward differs from verified replay: {float32_error}')
    allobs, allinterval, allznext = geometry.observations(*native)
    anchors = [v[INDICES] for v in native]
    obs, derivative_geometry, branch = geometry.geometry(*anchors)
    assert torch.equal(obs, allobs[INDICES])
    prediction64 = mean(p, allobs)
    normalized = torch.tanh(prediction64)
    if geometry.support:
        mids = torch.where(allinterval.feasible, (allinterval.lo+allinterval.hi)/2, 0.)
        halves = torch.where(allinterval.feasible, (allinterval.hi-allinterval.lo)/2, 0.)
        geometry.env.x, geometry.env.batch = native[0], len(allobs)
        action64 = torch.where(allinterval.feasible, mids+halves*normalized, geometry.env.fallback(allznext))
    else:
        action64 = geometry.cfg.input_bound*normalized
    J = batched_value_jac(p, obs).detach().numpy()
    M = batched_mixed(p, obs, *derivative_geometry).detach().numpy().reshape(-1, len(p))
    if not np.isfinite(J).all() or not np.isfinite(M).all():
        raise RuntimeError('Nonfinite neural derivatives; no silent replacement.')
    _, sj, V = linalg.svd(J, full_matrices=False, lapack_driver='gesdd')
    tolerance = max(J.shape)*np.finfo(np.float64).eps*sj[0]
    rank = int(np.count_nonzero(sj > tolerance))
    Vr = V[:rank]
    projected = M-(M@Vr.T)@Vr
    cc = projected@projected.T
    eig, left = linalg.eigh(cc, subset_by_index=(len(cc)-1, len(cc)-1))
    signal = float(np.sqrt(max(eig[-1], 0.)))
    direction = projected.T@left[:, -1]
    direction /= np.linalg.norm(direction) if np.linalg.norm(direction) else 1.
    if direction[np.argmax(np.abs(direction))] < 0:
        direction *= -1
    v = torch.tensor(direction)
    mixed_norm = float(np.sqrt(max(linalg.eigh(M@M.T, eigvals_only=True, subset_by_index=(len(M)-1, len(M)-1))[0], 0.)))
    jv, mv = J@direction, M@direction
    j_residual = float(np.linalg.norm(jv)/sj[0])
    orthogonality = float(np.max(np.abs(Vr@Vr.T-np.eye(rank)))) if rank else 0.
    rounding_bound = 10*np.finfo(float).eps*(len(p)+rank)*max(mixed_norm, 1e-300)
    checks = []
    for eps in PARAMETER_EPS:
        kp = batched_feedback(p+eps*v, obs, *derivative_geometry).detach().numpy().reshape(-1)
        km = batched_feedback(p-eps*v, obs, *derivative_geometry).detach().numpy().reshape(-1)
        fd = (kp-km)/(2*eps)
        error = float(np.linalg.norm(fd-mv))
        checks.append({'parameter_epsilon': eps, 'max_abs_error': float(np.max(np.abs(fd-mv))),
                       'relative_L2_error': error/max(np.linalg.norm(mv), 1e-300)})
    _, all_mean_jvp = torch.func.jvp(lambda parameters: mean(parameters, allobs), (p,), (v,))
    heldout = np.ones(len(allobs), dtype=bool); heldout[INDICES] = False
    leaks = all_mean_jvp.detach().numpy()[heldout]
    leak_limit = 1e-7*max(1., float(prediction64.abs().max()))
    relative_signal = signal/max(mixed_norm, 1e-300)
    resolved = (relative_signal >= 1e-6 and signal > 100*rounding_bound
                and j_residual <= 1e-9 and all(c['relative_L2_error'] <= 1e-3 for c in checks))
    assert sha(checkpoint) == before_hash
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK/f'direction-{seed}-{arm}.npz', direction=direction, J_singular_values=sj,
                        anchor_mean_jvp=jv, anchor_feedback_jvp=mv.reshape(-1, 3),
                        all_mean_jvp=all_mean_jvp.detach().numpy())
    return {
        'seed': seed, 'arm': arm, 'checkpoint_relative_path': checkpoint.relative_to(ROOT).as_posix(),
        'checkpoint_byte_sha256': before_hash, 'checkpoint_unchanged': True,
        'replay_relative_path': replay.relative_to(ROOT).as_posix(), 'replay_byte_sha256': sha(replay),
        'parameter_count': len(p), 'log_std_frozen': saved['actor']['log_std'],
        'source_policy': 'Actual saved Tanh PPO actor, original native 3-state/17-observation/1-action mapping',
        'precision': {'float32_latent_forward_vs_replay_max': float32_error,
                      'float64_rebuilt_obs_vs_saved_float32_max': float((allobs-oldobs.double()).abs().max()),
                      'float64_latent_vs_saved_float32_max': float((prediction64-oldmean.double()).abs().max()),
                      'float64_physical_action_vs_saved_float32_max': float((action64-oldaction).abs().max())},
        'anchors': {'count': len(INDICES), 'time_indices': TIMES, 'all_16_environments': True, 'branch': branch},
        'linear_operators': {'J_shape': J.shape, 'M_shape': M.shape, 'rank_tolerance': tolerance,
                             'J_rank': rank, 'J_rank_relative_1e_7': int(np.count_nonzero(sj > 1e-7*sj[0])),
                             'J_singular_values': sj, 'right_basis_orthogonality_max': orthogonality,
                             'M_spectral_norm': mixed_norm, 'projected_M_spectral_norm': signal,
                             'projected_M_relative_signal': relative_signal,
                             'projection_roundoff_bound': rounding_bound},
        'direction': {'norm': float(np.linalg.norm(direction)), 'Jv_L2': float(np.linalg.norm(jv)),
                      'Jv_relative_residual': j_residual, 'Mv_L2': float(np.linalg.norm(mv)),
                      'anchor_local_resolved': bool(resolved), 'parameter_FD_checks': checks,
                      'temporary_vector_only': True, 'no_parameter_update': True},
        'heldout': {'count': int(heldout.sum()), 'mean_jvp_max_abs': float(np.max(np.abs(leaks))),
                    'mean_jvp_RMS': float(np.sqrt(np.mean(leaks**2))), 'declared_tolerance': leak_limit,
                    'passes_full_replay_value_preservation': bool(np.max(np.abs(leaks)) <= leak_limit),
                    'scope': 'Derivative at unselected conditions of reconstructed deterministic evaluation; not old PPO training data'},
        'MPC_demand': {'actor_dependency_in_original_reference_predictor': False,
                       'direct_actor_demand_gradient': 0., 'robust_feedback_objective_added': False,
                       'performance_comparison_run': False},
        'wall_seconds': time.perf_counter()-start,
        'budget': {'anchor_value_jacobian_rows': len(J), 'mixed_jacobian_rows': len(M),
                   'parameter_direction_probe_vectors': 4, 'heldout_JVP_conditions': int(heldout.sum()),
                   'new_environment_transitions_in_this_script': 0, 'training_updates': 0,
                   'optimizer_steps': 0, 'checkpoint_writes': 0, 'policy_updates': 0}}


def main():
    results = []
    for seed in SEEDS:
        for arm in ARMS:
            result = audit_case(seed, arm)
            results.append(result)
            print(json.dumps({'seed': seed, 'arm': arm, 'rank': result['linear_operators']['J_rank'],
                              'signal_ratio': result['linear_operators']['projected_M_relative_signal'],
                              'anchor_pass': result['direction']['anchor_local_resolved'],
                              'heldout_max': result['heldout']['mean_jvp_max_abs'],
                              'heldout_pass': result['heldout']['passes_full_replay_value_preservation'],
                              'seconds': result['wall_seconds']}, ensure_ascii=False), flush=True)
    report = {'date': '2026-10-17', 'protocol_content_sha256': text_sha(OUT/'neural-protocol.md'),
              'script_content_sha256': text_sha(Path(__file__)), 'cases': results,
              'scope': 'Actual trained networks, fixed native evaluation conditions, no learning or changed MPC objective',
              'summary': {'case_count': len(results),
                          'resolved_anchor_cases': sum(r['direction']['anchor_local_resolved'] for r in results),
                          'full_replay_value_preserving_cases': sum(r['direction']['anchor_local_resolved'] and r['heldout']['passes_full_replay_value_preservation'] for r in results),
                          'original_MPC_nonzero_feedback_demand_cases': 0,
                          'training_updates': 0, 'policy_updates': 0, 'checkpoint_writes': 0},
              'limitations': ['Float64 derivatives extend existing float32 weights; forward differences are reported.',
                              'Anchor-null directions need not preserve heldout values or a complete training distribution.',
                              'No original training advantages/log-probabilities were reconstructed.',
                              'The original planner does not predict through the actor; adding robustness would be a new planner problem.']}
    (OUT/'neural-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=plain)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
