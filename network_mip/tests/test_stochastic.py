"""随机规划：情景、VSS/EVPI、鲁棒对照。"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import stochastic  # noqa: E402
from src.instance import load_instance  # noqa: E402


def tight_instance(scale: float = 0.35):
    base = load_instance()
    return dataclasses.replace(
        base, dcs=[dataclasses.replace(dc, capacity=dc.capacity * scale) for dc in base.dcs]
    )


class TestScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = load_instance()
        cls.scenarios = stochastic.generate_scenarios(cls.inst, n_scenarios=6, sigma=0.35, seed=3)

    def test_probabilities_sum_to_one(self):
        self.assertAlmostEqual(sum(s.probability for s in self.scenarios), 1.0)

    def test_multipliers_are_mean_preserving(self):
        for zone in self.inst.zone_names:
            mean = sum(s.probability * s.multipliers[zone] for s in self.scenarios)
            self.assertAlmostEqual(mean, 1.0, places=10, msg=zone)

    def test_multipliers_are_positive(self):
        self.assertTrue(all(v > 0 for s in self.scenarios for v in s.multipliers.values()))

    def test_reproducible_with_seed(self):
        again = stochastic.generate_scenarios(self.inst, n_scenarios=6, sigma=0.35, seed=3)
        self.assertEqual(self.scenarios[0].multipliers, again[0].multipliers)


class TestScaling(unittest.TestCase):
    def test_demand_is_scaled(self):
        inst = load_instance()
        multiplier = 2.5
        scaled = stochastic.scaled_instance(inst, {zone: multiplier for zone in inst.zone_names})
        for family in inst.families:
            for zone, demand in family.weekly_demand.items():
                self.assertAlmostEqual(scaled.demand(zone, family.name), demand * multiplier)


class TestStochasticAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = tight_instance(0.35)
        cls.scenarios = stochastic.generate_scenarios(cls.inst, n_scenarios=3, sigma=0.35, seed=5)
        cls.result = stochastic.analyze(
            cls.inst, cls.scenarios, time_limit=60, overflow_cost=5.0
        )

    def test_solvers_reach_optimal(self):
        self.assertEqual(self.result["stochastic"].status, "Optimal")
        self.assertEqual(self.result["robust"].status, "Optimal")

    def test_vss_and_evpi_are_non_negative(self):
        self.assertGreaterEqual(self.result["vss"], -1.0)
        self.assertGreaterEqual(self.result["evpi"], -1.0)

    def test_stochastic_is_not_worse_than_deterministic_on_average(self):
        self.assertLessEqual(
            self.result["stochastic"].expected_cost,
            self.result["deterministic_eval"]["expected"] + 1.0,
        )

    def test_robust_caps_the_worst_case(self):
        self.assertLessEqual(
            self.result["robust"].worst_cost,
            self.result["deterministic_eval"]["worst"] + 1.0,
        )

    def test_wait_and_see_is_the_lower_bound(self):
        self.assertLessEqual(
            self.result["wait_and_see_expected"], self.result["stochastic"].expected_cost + 1.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
