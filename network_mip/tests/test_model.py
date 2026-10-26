"""单元测试：算例、模型可行性、下界关系。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baselines, model  # noqa: E402
from src.instance import haversine_km, load_instance  # noqa: E402


class TestInstance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = load_instance()

    def test_sizes(self):
        self.assertEqual(len(self.inst.zones), 20)
        self.assertEqual(len(self.inst.dcs), 8)
        self.assertEqual(len(self.inst.families), 5)

    def test_demand_is_positive(self):
        total = sum(sum(f.weekly_demand.values()) for f in self.inst.families)
        self.assertGreater(total, 10_000)

    def test_distance_symmetry(self):
        a, b = self.inst.central, self.inst.dcs[0]
        self.assertAlmostEqual(haversine_km(a, b), haversine_km(b, a), places=6)

    def test_known_distance(self):
        """Memphis -> Rotterdam 约 7,100 km（±10%）。"""
        rotterdam = next(d for d in self.inst.dcs if d.name == "Rotterdam")
        d = haversine_km(self.inst.central, rotterdam)
        self.assertTrue(6600 < d < 7800, d)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = load_instance()
        cls.sol = model.solve(cls.inst, time_limit=120)
        cls.lp = model.solve(cls.inst, relax=True, time_limit=120)
        cls.nearest = baselines.all_dcs_nearest(cls.inst)
        cls.single = baselines.single_central(cls.inst)

    def test_solver_optimal(self):
        self.assertEqual(self.sol.status, "Optimal")
        self.assertIsNotNone(self.sol.objective)

    def test_every_zone_is_served_exactly_once(self):
        self.assertEqual(set(self.sol.assignment), set(self.inst.zone_names))

    def test_assignment_only_uses_open_facilities(self):
        open_set = set(self.sol.open_dcs)
        for zone, fac in self.sol.assignment.items():
            if fac == model.DIRECT_NAME:
                continue
            self.assertIn(fac, open_set, f"{zone} served by closed {fac}")

    def test_capacity_is_respected(self):
        caps = {d.name: d.capacity for d in self.inst.dcs}
        for d in self.sol.open_dcs:
            served = sum(
                52.0 * self.inst.zone_weekly_total(z)
                for z, fac in self.sol.assignment.items()
                if fac == d
            )
            self.assertLessEqual(served, caps[d] + 1e-6, f"{d}: {served:,.0f} > {caps[d]:,.0f}")

    def test_milp_not_worse_than_baselines(self):
        self.assertLessEqual(self.sol.objective, self.nearest["total"] + 1.0)
        self.assertLessEqual(self.sol.objective, self.single["total"] + 1.0)

    def test_lp_is_a_lower_bound(self):
        self.assertLessEqual(self.lp.objective, self.sol.objective + 1e-6)

    def test_breakdown_matches_objective(self):
        """独立重算一遍成本，应该和求解器目标值一致。"""
        open_set = set(self.sol.open_dcs)
        if any(v == model.DIRECT_NAME for v in self.sol.assignment.values()):
            open_set.add(model.DIRECT_NAME)
        recomputed = model.cost_terms(
            self.inst, self.sol.assignment, open_set, self.inst.params["service_level"]
        )["total"]
        self.assertAlmostEqual(recomputed, self.sol.objective, delta=max(1.0, 1e-4 * recomputed))


class TestDirectOption(unittest.TestCase):
    def test_no_dc_when_fixed_costs_are_prohibitive(self):
        """固定成本极高时，最优解应该是"全部直发"——说明直发选项真的在可行域里。"""
        inst = load_instance()
        inst.dcs = [dataclasses.replace(d, fixed_cost=d.fixed_cost * 1_000) for d in inst.dcs]
        sol = model.solve(inst, time_limit=120)
        self.assertEqual(sol.open_dcs, [])
        self.assertTrue(all(v == model.DIRECT_NAME for v in sol.assignment.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
