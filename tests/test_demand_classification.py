"""ADI/CV² demand classification tests."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "demand_classification", ROOT / "python" / "demand_classification.py"
)
demand = importlib.util.module_from_spec(spec)
sys.modules["demand_classification"] = demand
spec.loader.exec_module(demand)


class TestClassify(unittest.TestCase):
    def test_quadrants(self):
        self.assertEqual(demand.classify(1.0, 0.2), "smooth")
        self.assertEqual(demand.classify(1.0, 0.8), "erratic")
        self.assertEqual(demand.classify(2.0, 0.2), "intermittent")
        self.assertEqual(demand.classify(2.0, 0.8), "lumpy")

    def test_thresholds_are_inclusive_on_the_low_side(self):
        self.assertEqual(demand.classify(1.31, 0.48), "smooth")
        self.assertEqual(demand.classify(1.33, 0.48), "intermittent")
        self.assertEqual(demand.classify(1.31, 0.50), "erratic")


class TestWeeklyMatrix(unittest.TestCase):
    def test_it_fills_missing_weeks_with_zero(self):
        raw = pd.DataFrame(
            {
                "Order Date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
                "Product ID": ["A", "A"],
                "Quantity": [2, 3],
            }
        )
        matrix = demand.weekly_matrix(raw)
        self.assertEqual(matrix.loc["A"].sum(), 5)
        self.assertGreaterEqual(matrix.shape[1], 3)
        self.assertTrue((matrix.loc["A"] == 0).any())


class TestDemandStats(unittest.TestCase):
    def test_adi_and_cv2(self):
        weeks = pd.period_range("2024-01-01", periods=4, freq="W")
        matrix = pd.DataFrame(
            {"A": [0, 0, 2, 4], "B": [1, 2, 3, 4]},
            index=weeks,
        ).T
        stats = demand.demand_stats(matrix).set_index("Product_ID")
        self.assertAlmostEqual(stats.loc["A", "adi"], 2.0)
        self.assertAlmostEqual(stats.loc["A", "cv2"], (np.std([2, 4], ddof=1) / 3) ** 2)
        self.assertEqual(stats.loc["A", "demand_class"], "intermittent")
        self.assertEqual(stats.loc["B", "demand_class"], "smooth")


if __name__ == "__main__":
    unittest.main(verbosity=2)
