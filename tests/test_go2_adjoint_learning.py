"""Mechanism tests only: estimator identity, closed-loop adjoint and returns.

No optimizer or simulator is used. These are numerical correctness tests, not
learning experiments or evidence of a Go2 performance advantage.
"""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from go2_frozen.task import TaskSpec
from go2_learning.learning import actual_returns, continuation_direction, ppo_objective


class MechanismTests(unittest.TestCase):
    def test_cv_zero_mean_with_ppo_clipping_and_changed_variance(self):
        nodes, weights = np.polynomial.hermite.hermgauss(96)
        action = torch.tensor(np.sqrt(2.0) * nodes, dtype=torch.float64)
        weights = torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float64)
        mean = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
        log_std = torch.tensor(.15, dtype=torch.float64, requires_grad=True)
        old = torch.distributions.Normal(0., 1.)
        new = torch.distributions.Normal(mean, log_std.exp())
        ratio = (new.log_prob(action) - old.log_prob(action)).exp()
        # The ordinary PPO term is intentionally in and out of clipping.
        advantage = action.square() - .5
        ordinary = torch.minimum(ratio * advantage,
                                 ratio.clamp(.8, 1.2) * advantage)
        direction = 1.3
        cv = .25 * direction * (mean - ratio * action)
        self.assertGreater(float(((ratio - 1).abs() > .2).double().mean()), .5)
        objective0 = (weights * ordinary).sum()
        objective1 = (weights * (ordinary + cv)).sum()
        self.assertLess(abs(float((weights * cv).sum().detach())), 1e-11)
        grads0 = torch.autograd.grad(objective0, (mean, log_std), retain_graph=True)
        grads1 = torch.autograd.grad(objective1, (mean, log_std))
        for g0, g1 in zip(grads0, grads1):
            self.assertLess(float((g0 - g1).abs()), 1e-10)
        # Verify the production objective uses exactly this independent CV.
        out, _ = ppo_objective(
            new.log_prob(action), old.log_prob(action), advantage,
            mean.expand(len(action), 1), torch.zeros(len(action), 1, dtype=action.dtype),
            action[:, None], torch.full((len(action), 1), direction, dtype=action.dtype),
            alpha=.25)
        self.assertLess(float((out - (ordinary + cv).mean()).abs().detach()), 1e-12)

    def test_returns_cross_stages_but_not_episode_boundaries(self):
        result = actual_returns(
            torch.tensor([-1., -2., -3., -4., -5.]),
            torch.tensor([99., 0., 99., 99., 10.]),
            torch.tensor([False, True, False, False, False]),
            torch.tensor([False, False, False, False, True]))
        torch.testing.assert_close(result, torch.tensor([-3., -2., -2., 1., 5.]))

    def test_adjoint_includes_future_feedback_jacobian(self):
        class Actor(nn.Module):
            def mean(self, state, context, command):
                out = torch.zeros_like(state[..., :12])
                return torch.cat((-2 * state[..., :1], out[..., 1:]), -1)

        class Dynamics(nn.Module):
            dt = .02

            def forward(self, state, raw):
                return torch.cat((state[..., :1] + self.dt * raw[..., :1],
                                  state[..., 1:]), -1)

        class Value(nn.Module):
            def forward(self, state, context, command):
                return -state[..., 0].square()

        x = torch.zeros(1, 48, dtype=torch.float64)
        x[:, 0] = .3
        x[:, 2] = .34
        memory = torch.tensor([[0., 0., -1.]], dtype=torch.float64)
        command = torch.zeros(1, 3, dtype=torch.float64)
        d, _ = continuation_direction(Actor(), Dynamics(), Value(), x, memory,
                                      command, command, TaskSpec(), 10)
        # First raw is independent; the remaining nine steps have A+B*K=.96.
        expected = -2 * .3 * (.96 ** 10) * .02 * (.96 ** 9)
        self.assertAlmostEqual(float(d[0, 0]), expected, places=12)
        self.assertEqual(float(d[0, 1:].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
