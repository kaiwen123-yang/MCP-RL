"""Three direct-mechanism tests; no simulator, optimizer, or learned checkpoint."""

import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2_frozen.realized_learning import (
    ExecutedMacroBatch, ExecutionIdentity, PredictedMacroBatch,
    factual_response_loss, freeze_response_reference, ppo_realized_cost_loss,
    realized_macro_costs, realized_return_to_go, returns_for_micro_actions,
    score_function_cost_loss,
)


class RealizedLearningMechanisms(unittest.TestCase):
    def setUp(self):
        self.identity = ExecutionIdentity("frozen-policy", "physics/controller/clock-v1", "same-task-history-commands")

    def test_factual_fit_uses_real_labels_and_circular_yaw_only(self):
        actual = torch.zeros(1, 1, 6, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            actual[..., 2] = math.pi - 0.1
        predicted = actual.detach().clone()
        predicted[..., 2] = -math.pi + 0.1
        predicted.requires_grad_()
        actual_resource = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
        predicted_resource = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
        true_task = torch.full((1, 1), 7.0, dtype=torch.float64, requires_grad=True)
        batch = ExecutedMacroBatch(actual, actual_resource, self.identity, true_task, True)
        prediction = PredictedMacroBatch(predicted, predicted_resource, self.identity, "model-v1")
        parts = factual_response_loss(prediction, batch)
        self.assertAlmostEqual(float(parts.feature_loss), 0.2**2 / 6, places=12)
        gradients = torch.autograd.grad(parts.loss, (predicted, predicted_resource, actual, actual_resource, true_task), allow_unused=True)
        self.assertGreater(float(gradients[0].abs().sum()), 0.0)
        self.assertGreater(float(gradients[1].abs().sum()), 0.0)
        self.assertEqual(gradients[2:], (None, None, None))
        incompatible = PredictedMacroBatch(predicted, predicted_resource,
                                          ExecutionIdentity("frozen-policy", "physics/controller/clock-v1", "other-history"), "model-v1")
        with self.assertRaisesRegex(ValueError, "identity"):
            factual_response_loss(incompatible, batch)

    def test_real_terminal_consequence_reaches_all_prior_actions(self):
        costs = torch.tensor([[1., 2.], [0., 1.], [9., 3.]], dtype=torch.float64)
        expected = torch.tensor([[10., 6.], [9., 4.], [9., 3.]], dtype=torch.float64)
        self.assertTrue(torch.equal(realized_return_to_go(costs, True), expected))
        micro = returns_for_micro_actions(costs, 2, complete_task=True)
        self.assertTrue(torch.equal(micro, expected.repeat_interleave(2, dim=0)))
        # Omitting a real suffix cannot silently turn a truncated segment into
        # a complete task. Supplying its measured cost recovers exact credit.
        with self.assertRaisesRegex(ValueError, "truncated"):
            realized_return_to_go(costs[:1], False)
        reconstructed = realized_return_to_go(costs[:1], False, known_realized_suffix_cost=costs[1:].sum(0))
        self.assertTrue(torch.equal(reconstructed[0], expected[0]))

    def test_policy_credit_uses_executed_cost_and_equals_standard_po(self):
        reference_features = torch.zeros(1, 2, 6, dtype=torch.float64, requires_grad=True)
        reference_resources = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
        reference = freeze_response_reference(PredictedMacroBatch(
            reference_features, reference_resources, self.identity, "fixed-reference-before-data"))
        actual = torch.zeros(1, 2, 6, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            actual[0, 1, 0] = 1.0
        batch = ExecutedMacroBatch(actual, torch.zeros(1, 2, dtype=torch.float64),
                                   self.identity, torch.tensor([[1., 4.]], dtype=torch.float64), True)
        costs = realized_macro_costs(batch, reference, predictability_weight=0.6)
        self.assertTrue(torch.allclose(costs.total, torch.tensor([[1., 4.1]], dtype=torch.float64)))
        returns = realized_return_to_go(costs.total, batch.complete_task)
        # Recorded actions are actual samples in this mathematical mechanism
        # test; they are not represented as a real Go2 stochastic rollout.
        raw_actions = torch.tensor([[-1., 1.]], dtype=torch.float64)
        mean = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        logp = torch.distributions.Normal(mean, torch.ones_like(mean)).log_prob(raw_actions)
        old_logp = logp.detach().clone()
        ppo = ppo_realized_cost_loss(logp, old_logp, returns)
        score = score_function_cost_loss(logp, returns)
        gradient_ppo, = torch.autograd.grad(ppo.loss, mean, retain_graph=True)
        gradient_score, = torch.autograd.grad(score, mean, retain_graph=True)
        self.assertAlmostEqual(float(gradient_ppo), (4.1 - 1.0) / 2, places=12)
        self.assertTrue(torch.allclose(gradient_ppo, gradient_score))
        leaked = torch.autograd.grad(ppo.loss, (reference_features, reference_resources, actual), allow_unused=True)
        self.assertEqual(leaked, (None, None, None))
        with self.assertRaisesRegex(ValueError, "No behavior density"):
            ppo_realized_cost_loss(logp, None, returns)


if __name__ == "__main__":
    unittest.main()
