#!/usr/bin/env python3
"""Stochastic vs deterministic vs robust network design: VSS and EVPI.

The deterministic MILP optimises for mean demand.  This script asks whether
that is good enough when zone demand is uncertain:

* **deterministic** — optimise for mean demand, then evaluate on scenarios;
* **stochastic**    — two-stage model, minimise expected cost;
* **robust**        — two-stage model, minimise the worst scenario;
* **wait-and-see**  — perfect information, a separate network per scenario.

Two standard OR metrics:

* **VSS** = deterministic expected cost − stochastic expected cost
  (the cost of ignoring uncertainty);
* **EVPI** = stochastic expected cost − wait-and-see cost
  (what better demand information is worth).

Capacity is soft: units above nameplate pay ``--overflow-cost`` (overtime /
emergency handling), so all policies are evaluated on the same cost function.

Usage
-----
    python scripts/stochastic.py
    python scripts/stochastic.py --scenarios 12 --sigma 0.4 --capacity-scales 1.0,0.5,0.35
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import model, stochastic  # noqa: E402
from src.instance import load_instance  # noqa: E402

RESULTS = ROOT / "results"
IMAGES = ROOT / "images"


def real_dc_count(solution) -> int:
    return len([name for name in solution.open_dcs if name != model.DIRECT_NAME])


def with_capacity(inst, scale: float):
    return dataclasses.replace(
        inst, dcs=[dataclasses.replace(dc, capacity=dc.capacity * scale) for dc in inst.dcs]
    )


def scenario_table(capacity_scale: float, scenarios, result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "capacity_scale": capacity_scale,
            "scenario": [scenario.name for scenario in scenarios],
            "probability": [scenario.probability for scenario in scenarios],
            "deterministic_cost": result["deterministic_eval"]["scenario_costs"],
            "stochastic_cost": result["stochastic"].scenario_costs,
            "robust_cost": result["robust"].scenario_costs,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--capacity-scales", default="1.0,0.5,0.35")
    ap.add_argument("--overflow-cost", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--time-limit", type=int, default=90)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    base = load_instance()
    scales = [float(value) for value in args.capacity_scales.split(",") if value.strip()]
    rows = []
    scenario_rows = []
    result_by_scale = {}

    for scale in scales:
        inst = with_capacity(base, scale)
        scenarios = stochastic.generate_scenarios(
            inst, n_scenarios=args.scenarios, sigma=args.sigma, seed=args.seed, low=0.5, high=1.5
        )
        result = stochastic.analyze(
            inst, scenarios, time_limit=args.time_limit, overflow_cost=args.overflow_cost
        )
        result_by_scale[scale] = (result, scenarios)

        det_expected = result["deterministic_eval"]["expected"]
        rows.append(
            {
                "capacity_scale": scale,
                "sigma": args.sigma,
                "scenarios": len(scenarios),
                "det_expected": det_expected,
                "det_worst": result["deterministic_eval"]["worst"],
                "stoch_expected": result["stochastic"].expected_cost,
                "stoch_worst": result["stochastic"].worst_cost,
                "robust_expected": result["robust"].expected_cost,
                "robust_worst": result["robust"].worst_cost,
                "wait_and_see": result["wait_and_see_expected"],
                "vss": result["vss"],
                "evpi": result["evpi"],
                "vss_pct": result["vss"] / det_expected,
                "evpi_pct": result["evpi"] / det_expected,
                "det_open": real_dc_count(result["deterministic"]),
                "stoch_open": real_dc_count(result["stochastic"]),
                "robust_open": real_dc_count(result["robust"]),
            }
        )
        scenario_rows.append(scenario_table(scale, scenarios, result))

        print(
            f"capacity {scale:.2f}: VSS ${result['vss']:,.0f} ({result['vss']/det_expected:.2%}), "
            f"EVPI ${result['evpi']:,.0f} ({result['evpi']/det_expected:.2%}), "
            f"open det/stoch/robust = {real_dc_count(result['deterministic'])}/"
            f"{real_dc_count(result['stochastic'])}/{real_dc_count(result['robust'])}"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "stochastic_summary.csv", index=False)
    pd.concat(scenario_rows, ignore_index=True).to_csv(
        RESULTS / "stochastic_scenarios.csv", index=False
    )

    # Solution table
    solution_rows = []
    for scale, (result, _) in result_by_scale.items():
        for name, solution in (
            ("deterministic", result["deterministic"]),
            ("stochastic", result["stochastic"]),
            ("robust", result["robust"]),
        ):
            solution_rows.append(
                {
                    "capacity_scale": scale,
                    "policy": name,
                    "open_count": real_dc_count(solution),
                    "open_dcs": ";".join(
                        sorted(name for name in solution.open_dcs if name != model.DIRECT_NAME)
                    ),
                    "uses_direct": model.DIRECT_NAME in solution.open_dcs
                    or any(value == model.DIRECT_NAME for value in solution.assignment.values()),
                    "direct_zones": sum(
                        1 for value in solution.assignment.values() if value == model.DIRECT_NAME
                    ),
                }
            )
    pd.DataFrame(solution_rows).to_csv(RESULTS / "stochastic_solutions.csv", index=False)

    metrics = {
        "params": vars(args),
        "results": summary.to_dict(orient="records"),
        "interpretation": {
            "vss": "cost of ignoring demand uncertainty in network design",
            "evpi": "value of perfect demand information",
        },
    }
    (RESULTS / "stochastic_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------------- charts
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tight_scale = min(scales)
        tight, scenarios = result_by_scale[tight_scale]
        fig = plt.figure(figsize=(13.5, 4.2))
        grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.35], wspace=0.32)

        ax = fig.add_subplot(grid[0, 0])
        labels = ["deterministic", "stochastic", "robust", "wait-and-see"]
        expected = [
            tight["deterministic_eval"]["expected"],
            tight["stochastic"].expected_cost,
            tight["robust"].expected_cost,
            tight["wait_and_see_expected"],
        ]
        worst = [
            tight["deterministic_eval"]["worst"],
            tight["stochastic"].worst_cost,
            tight["robust"].worst_cost,
            float("nan"),
        ]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, np.array(expected) / 1e6, width=0.4, label="expected", color="#1f6f78")
        ax.bar(x + 0.2, np.array(worst) / 1e6, width=0.4, label="worst case", color="#d99a2b")
        ax.set_xticks(x, ["det", "stoch", "robust", "perfect\ninfo"])
        ax.set_ylabel("annual cost (USD M)")
        ax.set_title(f"Tight capacity ({tight_scale:.0%}), {args.scenarios} scenarios")
        ax.legend(fontsize=8)

        ax2 = fig.add_subplot(grid[0, 1])
        ax2.plot(summary["capacity_scale"] * 100, summary["vss"] / 1e3, marker="o",
                 color="#1f6f78", label="VSS")
        ax2.plot(summary["capacity_scale"] * 100, summary["evpi"] / 1e3, marker="o",
                 color="#9b3b3b", label="EVPI")
        ax2.set_xlabel("nameplate capacity (% of base)")
        ax2.set_ylabel("value (USD k)")
        ax2.set_title("When does uncertainty matter?")
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(grid[0, 2])
        det_costs = np.array(tight["deterministic_eval"]["scenario_costs"])
        stoch_costs = np.array(tight["stochastic"].scenario_costs)
        robust_costs = np.array(tight["robust"].scenario_costs)
        order = np.argsort(-det_costs)
        xscenario = np.arange(len(order))
        ax3.plot(xscenario, det_costs[order] / 1e6, marker="o", label="deterministic", color="#9b3b3b")
        ax3.plot(xscenario, stoch_costs[order] / 1e6, marker="o", label="stochastic", color="#1f6f78")
        ax3.plot(xscenario, robust_costs[order] / 1e6, marker="s", label="robust", color="#d99a2b")
        ax3.set_xticks(xscenario, [scenarios[i].name for i in order], rotation=45, fontsize=7)
        ax3.set_xlabel("demand scenario (sorted by deterministic cost)")
        ax3.set_ylabel("annual cost (USD M)")
        ax3.set_title(f"Scenario costs at {tight_scale:.0%} capacity")
        ax3.legend(fontsize=8)

        fig.savefig(IMAGES / "stochastic.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {IMAGES / 'stochastic.png'}")
    except Exception as exc:  # pragma: no cover - optional chart
        print(f"chart skipped: {exc}")

    print(f"wrote {RESULTS / 'stochastic_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
