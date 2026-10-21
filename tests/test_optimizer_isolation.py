"""One real Adam-state isolation test; no Go2 rollout or policy training.

The small linear modules exercise the production load_state path with populated
Adam moments. A real optimizer step is necessary to catch same-device aliasing:
equal tensor values or equal fingerprints alone do not establish ownership.
"""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from go2_learning.runner import load_state, optimizer_state_summary, state_dict


def tiny_system():
    modules = {name: torch.nn.Linear(2, 1)
               for name in ("actor", "dynamics", "response", "value")}
    optimizers = {name: torch.optim.Adam(module.parameters(), lr=1e-3)
                  for name, module in modules.items()}
    return {
        **modules,
        "updater": SimpleNamespace(
            actor_optimizer=optimizers["actor"], value_optimizer=optimizers["value"],
            actor_updates=0, value_updates=0),
        "f_optimizer": optimizers["dynamics"],
        "g_optimizer": optimizers["response"],
    }


def all_optimizers(system):
    return (system["updater"].actor_optimizer, system["updater"].value_optimizer,
            system["f_optimizer"], system["g_optimizer"])


def step_all(system, gradient):
    for optimizer in all_optimizers(system):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.full_like(parameter, gradient)
        optimizer.step()


def storages(value):
    if isinstance(value, torch.Tensor):
        return {value.untyped_storage().data_ptr()}
    if isinstance(value, dict):
        return set().union(*(storages(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(storages(item) for item in value))
    return set()


class OptimizerIsolationTest(unittest.TestCase):
    def assert_state_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertEqual(expected.shape, actual.shape)
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_state_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_loaded_branches_own_all_adam_tensors(self):
        torch.manual_seed(191)
        common = tiny_system()
        step_all(common, 0.25)
        common["updater"].actor_updates = common["updater"].value_updates = 1
        template = state_dict(common)
        untouched_template = copy.deepcopy(template)
        first, second = tiny_system(), tiny_system()
        load_state(first, template)
        load_state(second, template)
        self.assert_state_equal(template, state_dict(first))
        self.assert_state_equal(template, state_dict(second))
        expected_summary = optimizer_state_summary(common)
        self.assertEqual(expected_summary, optimizer_state_summary(first))
        self.assertEqual(expected_summary, optimizer_state_summary(second))
        for item in expected_summary.values():
            self.assertEqual(item["step_min"], 1.0)
            self.assertEqual(item["step_max"], 1.0)

        # Examine live states, not state_dict(system), whose deep copy would
        # hide an ownership bug in the loaded optimizer.
        live_first = [optimizer.state_dict() for optimizer in all_optimizers(first)]
        live_second = [optimizer.state_dict() for optimizer in all_optimizers(second)]
        template_optimizers = [template[name] for name in
                               ("actor_optimizer", "value_optimizer", "f_optimizer", "g_optimizer")]
        self.assertTrue(storages(live_first).isdisjoint(storages(live_second)))
        self.assertTrue(storages(live_first).isdisjoint(storages(template_optimizers)))
        self.assertTrue(storages(live_second).isdisjoint(storages(template_optimizers)))

        untouched_second = state_dict(second)
        step_all(first, -0.75)
        self.assert_state_equal(untouched_template, template)
        self.assert_state_equal(untouched_second, state_dict(second))
        self.assertEqual(expected_summary, optimizer_state_summary(second))
        self.assertNotEqual(expected_summary, optimizer_state_summary(first))
        for item in optimizer_state_summary(first).values():
            self.assertEqual(item["step_min"], 2.0)
            self.assertEqual(item["step_max"], 2.0)


if __name__ == "__main__":
    unittest.main()
