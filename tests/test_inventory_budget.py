"""Budget-constrained safety-stock allocation tests."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inventory_budget", ROOT / "python" / "inventory_budget.py")
budget = importlib.util.module_from_spec(spec)
sys.modules["inventory_budget"] = budget
spec.loader.exec_module(budget)


class TestLossAndEoq(unittest.TestCase):
    def test_normal_loss_at_zero(self):
        self.assertAlmostEqual(budget.normal_loss(0.0), 0.398942, places=5)

    def test_normal_loss_decreases(self):
        self.assertGreater(budget.normal_loss(0.0), budget.normal_loss(1.0))
        self.assertGreater(budget.normal_loss(1.0), budget.normal_loss(2.0))

    def test_eoq_formula(self):
        value = budget.eoq(1000, 50, 0.25, 20)
        self.assertAlmostEqual(value, np.sqrt(2 * 1000 * 50 / (0.25 * 20)))


class TestAbc(unittest.TestCase):
    def test_abc_thresholds(self):
        classes = budget.classify_abc(np.array([80.0, 15.0, 5.0]))
        self.assertEqual(classes.tolist(), ["A", "B", "C"])

    def test_abc_is_a_partition(self):
        classes = budget.classify_abc(np.array([50, 30, 10, 5, 5]))
        self.assertEqual(set(classes), {"A", "B", "C"})


class TestAllocation(unittest.TestCase):
    def setUp(self):
        self.unit_cost = np.array([100.0, 200.0, 300.0])
        self.sigma_l = np.array([1.0, 1.0, 1.0])
        self.cycles = np.array([5.0, 5.0, 5.0])
        self.demand = np.array([100.0, 100.0, 100.0])

    def test_greedy_respects_budget(self):
        _, spent = budget.allocate_greedy(
            self.unit_cost, self.sigma_l, self.cycles, budget=100.0, rate=0.25
        )
        self.assertLessEqual(spent, 100.0 + 1e-6)

    def test_greedy_beats_uniform_at_the_same_budget(self):
        z_opt, _ = budget.allocate_greedy(
            self.unit_cost, self.sigma_l, self.cycles, budget=60.0, rate=0.25
        )
        z_uniform = budget.uniform_z_for_budget(self.unit_cost, self.sigma_l, 60.0, 0.25)
        fill_opt = budget.fill_rate(z_opt, self.cycles, self.sigma_l, self.demand)
        fill_uniform = budget.fill_rate(
            np.full_like(z_opt, z_uniform), self.cycles, self.sigma_l, self.demand
        )
        self.assertGreaterEqual(fill_opt, fill_uniform - 1e-9)

    def test_uniform_budget_is_respected(self):
        z = budget.uniform_z_for_budget(self.unit_cost, self.sigma_l, 80.0, 0.25)
        spend = budget.holding_cost(
            np.full_like(self.unit_cost, z), self.unit_cost, self.sigma_l, 0.25
        ).sum()
        self.assertLessEqual(spend, 80.0 + 1e-6)

    def test_fill_rate_is_bounded(self):
        z = np.array([0.0, 2.0, 3.5])
        value = budget.fill_rate(z, self.cycles, self.sigma_l, self.demand)
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
