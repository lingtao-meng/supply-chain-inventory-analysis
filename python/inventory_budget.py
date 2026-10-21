#!/usr/bin/env python3
"""Budget-constrained safety-stock allocation across SKUs.

ABC tells you *which* SKUs matter; it does not tell you how much safety stock
each one should get.  This script treats the safety-stock budget as a resource
and allocates it by **marginal shortage reduction per dollar**:

* demand: annual demand ``D`` and weekly standard deviation from the
  ADI/CV² classification step;
* lead time ``L`` = 4 weeks, so ``σ_L = √L · σ_week``;
* service level ``z`` → safety stock ``z·σ_L``, holding cost ``h·c·z·σ_L``;
* expected shortage per cycle ``σ_L · L(z)``, cycles per year ``D/Q`` with EOQ;
* fill rate = ``1 − shortage units / D``.

Policies compared at the same holding budget:

* **ABC**: A 99%, B 95%, C 90% (the common practice);
* **uniform**: one z for every SKU, chosen to hit the budget;
* **optimized**: greedy marginal benefit per dollar, z grid 0–3.5.

Usage
-----
    python python/demand_classification.py     # builds results/demand_classification.csv
    python python/inventory_budget.py
    python python/inventory_budget.py --shortage-cost 75 --lead-time 6
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
MASTER = ROOT / "data" / "superstore_products.csv"
CLASSIFICATION = RESULTS / "demand_classification.csv"


def normal_loss(z: float) -> float:
    """Standard normal loss function L(z) = φ(z) − z·(1 − Φ(z))."""
    return float(stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))


def eoq(demand: float, order_cost: float, holding_rate: float, unit_cost: float) -> float:
    return float(np.sqrt(2 * demand * order_cost / (holding_rate * unit_cost)))


def classify_abc(values: np.ndarray) -> np.ndarray:
    """A ≤ 80% cumulative value, B ≤ 95%, C the rest."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values)
    cumulative = np.cumsum(values[order]) / values.sum()
    classes = np.empty(len(values), dtype=object)
    for rank, idx in enumerate(order):
        classes[idx] = "A" if cumulative[rank] <= 0.80 else "B" if cumulative[rank] <= 0.95 else "C"
    return classes


def holding_cost(z: np.ndarray, unit_cost: np.ndarray, sigma_l: np.ndarray, rate: float) -> np.ndarray:
    return rate * unit_cost * z * sigma_l


def shortage_units(z: np.ndarray, cycles: np.ndarray, sigma_l: np.ndarray) -> np.ndarray:
    losses = np.array([normal_loss(value) for value in z])
    return cycles * sigma_l * losses


def fill_rate(z: np.ndarray, cycles: np.ndarray, sigma_l: np.ndarray, demand: np.ndarray) -> float:
    return float(1 - shortage_units(z, cycles, sigma_l).sum() / demand.sum())


def uniform_z_for_budget(
    unit_cost: np.ndarray,
    sigma_l: np.ndarray,
    budget: float,
    rate: float,
    upper: float = 5.0,
) -> float:
    lower = 0.0
    for _ in range(60):
        mid = (lower + upper) / 2
        if holding_cost(np.full_like(unit_cost, mid), unit_cost, sigma_l, rate).sum() < budget:
            lower = mid
        else:
            upper = mid
    return (lower + upper) / 2


def allocate_greedy(
    unit_cost: np.ndarray,
    sigma_l: np.ndarray,
    cycles: np.ndarray,
    budget: float,
    rate: float,
    step: float = 0.05,
    z_max: float = 3.5,
) -> tuple[np.ndarray, float]:
    """Marginal benefit per dollar, with a precomputed z grid and a max-heap."""
    grid = np.arange(0, z_max + step / 2, step)
    losses = np.array([normal_loss(value) for value in grid])
    holding = rate * unit_cost[:, None] * sigma_l[:, None] * grid[None, :]
    units = cycles[:, None] * sigma_l[:, None] * losses[None, :]
    delta_holding = np.diff(holding, axis=1)
    delta_units = np.diff(units, axis=1)
    ratio = -delta_units / delta_holding

    level = np.zeros(len(unit_cost), dtype=int)
    heap = [(-ratio[i, 0], i) for i in range(len(unit_cost))]
    heapq.heapify(heap)
    spent = 0.0
    while heap:
        _, i = heapq.heappop(heap)
        k = level[i]
        if k >= len(grid) - 1:
            continue
        cost = delta_holding[i, k]
        if spent + cost > budget + 1e-9:
            continue  # never becomes affordable: budget only decreases
        spent += cost
        level[i] += 1
        if level[i] < len(grid) - 1:
            heapq.heappush(heap, (-ratio[i, level[i]], i))
    return grid[level], spent


def plot(summary: pd.DataFrame, frontier: pd.DataFrame, detail: pd.DataFrame, images: Path, rate: float = 0.25) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))

    ax = axes[0]
    ax.plot(frontier["budget"] / 1e3, frontier["uniform_fill"] * 100, marker="o",
            color="#d99a2b", label="uniform z")
    ax.plot(frontier["budget"] / 1e3, frontier["optimized_fill"] * 100, marker="o",
            color="#1f6f78", label="marginal allocation")
    abc = summary.loc[summary["policy"] == "ABC"].iloc[0]
    ax.scatter(abc["budget"] / 1e3, abc["fill_rate"] * 100, color="#9b3b3b", s=60, zorder=5,
               label="ABC policy")
    ax.set_xlabel("safety-stock holding budget (USD k)")
    ax.set_ylabel("demand-weighted fill rate (%)")
    ax.set_title("Fill rate at the same budget")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    abc_detail = detail.set_index("Product_ID")["abc"]
    classes = ["A", "B", "C"]
    abc_spend = [
        float(holding_cost(detail["z_abc"].to_numpy(), detail["unit_cost"].to_numpy(),
                           detail["sigma_l"].to_numpy(), rate)[abc_detail.to_numpy() == c].sum())
        for c in classes
    ]
    opt_spend = [
        float(holding_cost(detail["z_opt"].to_numpy(), detail["unit_cost"].to_numpy(),
                           detail["sigma_l"].to_numpy(), rate)[abc_detail.to_numpy() == c].sum())
        for c in classes
    ]
    x = np.arange(2)
    bottom = np.zeros(2)
    colors = {"A": "#1f6f78", "B": "#d99a2b", "C": "#9b3b3b"}
    for c, abc_value, opt_value in zip(classes, abc_spend, opt_spend):
        values = np.array([abc_value, opt_value]) / 1e3
        ax2.bar(x, values, bottom=bottom, label=c, color=colors[c])
        bottom += values
    ax2.set_xticks(x, ["ABC policy", "optimized"])
    ax2.set_ylabel("holding budget by class (USD k)")
    ax2.set_title("Where the safety-stock budget goes")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(images / "inventory_budget.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lead-time", type=float, default=4.0, help="weeks")
    ap.add_argument("--order-cost", type=float, default=50.0)
    ap.add_argument("--holding-rate", type=float, default=0.25)
    ap.add_argument("--shortage-cost", type=float, default=50.0, help="USD per unit short")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    if not CLASSIFICATION.exists():
        raise SystemExit("run python/demand_classification.py first")

    master = pd.read_csv(MASTER).drop_duplicates("Product_ID")
    classification = pd.read_csv(CLASSIFICATION)
    df = master.merge(
        classification[["Product_ID", "std_weekly", "mean_weekly", "cv_weekly"]],
        on="Product_ID",
        how="left",
    )
    df["std_weekly"] = df["std_weekly"].fillna(df["Annual_Demand"] / 52 * 0.8).clip(lower=0.05)
    df = df.sort_values("Annual_Demand", ascending=False).reset_index(drop=True)

    demand = df["Annual_Demand"].to_numpy(dtype=float)
    unit_cost = df["Unit_Cost"].to_numpy(dtype=float)
    sigma_week = df["std_weekly"].to_numpy(dtype=float)
    sigma_l = np.sqrt(args.lead_time) * sigma_week
    order_qty = np.array(
        [eoq(d, args.order_cost, args.holding_rate, c) for d, c in zip(demand, unit_cost)]  # noqa: B905
    )
    cycles = demand / order_qty
    abc = classify_abc(demand * unit_cost)

    z_map = {"A": stats.norm.ppf(0.99), "B": stats.norm.ppf(0.95), "C": stats.norm.ppf(0.90)}
    z_abc = np.array([z_map[c] for c in abc])
    budget = float(holding_cost(z_abc, unit_cost, sigma_l, args.holding_rate).sum())

    z_uniform = uniform_z_for_budget(unit_cost, sigma_l, budget, args.holding_rate)
    z_opt, spent = allocate_greedy(
        unit_cost, sigma_l, cycles, budget, args.holding_rate, z_max=3.5
    )

    def metrics(z: np.ndarray) -> dict:
        units = shortage_units(z, cycles, sigma_l).sum()
        return {
            "budget": float(holding_cost(z, unit_cost, sigma_l, args.holding_rate).sum()),
            "shortage_units": float(units),
            "shortage_cost": float(units * args.shortage_cost),
            "fill_rate": float(1 - units / demand.sum()),
            "avg_z": float(np.mean(z)),
        }

    summary = pd.DataFrame(
        [
            {"policy": "ABC", **metrics(z_abc)},
            {"policy": "uniform", **metrics(np.full_like(z_abc, z_uniform))},
            {"policy": "optimized", **metrics(z_opt)},
        ]
    )
    summary["budget"] = [budget, budget, spent]
    summary.to_csv(RESULTS / "inventory_budget_summary.csv", index=False)

    detail = pd.DataFrame(
        {
            "Product_ID": df["Product_ID"],
            "abc": abc,
            "annual_demand": demand,
            "unit_cost": unit_cost,
            "sigma_l": sigma_l,
            "eoq": order_qty,
            "z_abc": z_abc,
            "z_opt": z_opt,
            "ss_abc": z_abc * sigma_l,
            "ss_opt": z_opt * sigma_l,
        }
    )
    detail.to_csv(RESULTS / "inventory_budget_detail.csv", index=False)

    frontier_rows = []
    for fraction in (0.25, 0.50, 0.75, 1.00, 1.25):
        target = budget * fraction
        zu = uniform_z_for_budget(unit_cost, sigma_l, target, args.holding_rate)
        zo, sp = allocate_greedy(unit_cost, sigma_l, cycles, target, args.holding_rate)
        frontier_rows.append(
            {
                "budget_multiplier": fraction,
                "budget": target,
                "uniform_z": zu,
                "uniform_fill": fill_rate(np.full_like(z_abc, zu), cycles, sigma_l, demand),
                "optimized_fill": fill_rate(zo, cycles, sigma_l, demand),
                "optimized_spend": sp,
            }
        )
    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(RESULTS / "inventory_budget_frontier.csv", index=False)

    payload = {
        "assumptions": {
            "lead_time_weeks": args.lead_time,
            "order_cost": args.order_cost,
            "holding_rate": args.holding_rate,
            "shortage_cost_per_unit": args.shortage_cost,
        },
        "abc_counts": {c: int((abc == c).sum()) for c in "ABC"},
        "summary": summary.to_dict(orient="records"),
        "frontier": frontier.to_dict(orient="records"),
    }
    (RESULTS / "inventory_budget.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"ABC mix: {payload['abc_counts']}, ABC budget ${budget:,.0f}")
    print(summary.to_string(index=False))
    print("\nfrontier:")
    print(frontier.to_string(index=False))

    plot(summary, frontier, detail, IMAGES, args.holding_rate)
    print(f"wrote {IMAGES / 'inventory_budget.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
