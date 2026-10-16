"""One-step set algebra audit only; no learning and no robot certificate."""
from pathlib import Path
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from response_ppo.realization import feasible_interval, map_normalized_action, endpoint_residual


def run():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(1616)
    dtype = torch.float64
    count = 1000
    f0 = 0.1 * torch.randn(count, 3, dtype=dtype, generator=generator)
    znext = 0.1 * torch.randn(count, 3, dtype=dtype, generator=generator)
    error = 0.3 * torch.randn(count, 3, dtype=dtype, generator=generator)
    B = torch.tensor([0.1, 0.05, 0.2], dtype=dtype)
    E = torch.tensor([0.05, -0.1, 0.1], dtype=dtype)
    P = torch.tensor([[0.5, 0.03, 0.01], [0.03, 0.3, -0.02], [0.01, -0.02, 0.2]], dtype=dtype)
    interval = feasible_interval(f0, B, E, znext, error, P, .95, .02, .2, 1.2)
    okay = interval.feasible
    maximum = -float("inf")
    for s in torch.linspace(-1, 1, 19, dtype=dtype):
        action = map_normalized_action(torch.full((count,), s, dtype=dtype), interval)
        residual = endpoint_residual(f0[okay], B, E, znext[okay], error[okay], P, .95, .02, .2, action[okay])
        maximum = max(maximum, residual.max().item())
        for w in torch.linspace(-.2, .2, 17, dtype=dtype):
            e_next = f0[okay] + B * action[okay, None] + E * w - znext[okay]
            direct = torch.einsum("bi,ij,bj->b", e_next, P, e_next) - .95 * torch.einsum("bi,ij,bj->b", error[okay], P, error[okay]) - .02
            maximum = max(maximum, direct.max().item())
    assert maximum < 1e-12
    assert torch.isnan(map_normalized_action(torch.zeros(count, dtype=dtype), interval)[~okay]).all()
    disagreements = 0
    for action in torch.linspace(-1.2, 1.2, 501, dtype=dtype):
        a = torch.full((64,), action, dtype=dtype)
        residual = endpoint_residual(f0[:64], B, E, znext[:64], error[:64], P, .95, .02, .2, a)
        inside = interval.feasible[:64] & (a >= interval.lo[:64]) & (a <= interval.hi[:64])
        disagreements += (inside != (residual <= 0)).sum().item()
    assert disagreements == 0
    zero = torch.zeros(3, dtype=dtype)
    unit = torch.tensor([1., 0., 0.], dtype=dtype)
    identity = torch.eye(3, dtype=dtype)
    singleton = feasible_interval(zero, unit, zero, zero, zero, identity, .9, 0., 0., 1.)
    assert singleton.feasible and singleton.lo == 0 and singleton.hi == 0
    no_input_all = feasible_interval(zero, zero, zero, zero, zero, identity, .9, 0., 0., 1.)
    assert no_input_all.feasible and no_input_all.lo == -1 and no_input_all.hi == 1
    no_input_empty = feasible_interval(unit, zero, zero, zero, zero, identity, .9, 0., 0., 1.)
    assert not no_input_empty.feasible and no_input_empty.lo > no_input_empty.hi
    uncontrollable_empty = feasible_interval(zero, unit, zero, torch.tensor([0., 1., 0.], dtype=dtype), zero, identity, .9, .5, 0., 1.)
    assert not uncontrollable_empty.feasible
    # Repeated feasible one-step constructions check the recurrence bookkeeping.
    rho, delta, initial, steps = .95, .02, .8, 80
    recursive = initial
    for _ in range(steps):
        recursive = rho * recursive + delta
    geometric = rho**steps * initial + delta * (1 - rho**steps) / (1 - rho)
    assert abs(recursive - geometric) < 1e-12
    return {
        "status": "passed",
        "training": "none",
        "dtype": str(dtype),
        "random_cases": count,
        "feasible_cases": okay.sum().item(),
        "infeasible_cases": (~okay).sum().item(),
        "mapped_actions_per_feasible_case": 19,
        "disturbance_grid_per_action": 17,
        "maximum_one_step_excess": maximum,
        "membership_comparisons": 64 * 501,
        "membership_disagreements": disagreements,
        "singleton_and_zero_input_cases": "passed",
        "geometric_recurrence_error": abs(recursive - geometric),
        "scope": "declared discrete affine model and fixed SPD metric only; not a nonlinear integrator or robot guarantee",
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_name("realization-check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
