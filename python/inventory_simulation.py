#!/usr/bin/env python3
"""Inventory policy simulation — the numbers behind the ``(s, Q)`` notebooks.

The notebooks compute EOQ / safety stock / reorder points analytically and run a
90-day simulation on a *single* A-class SKU.  Two problems with that:

1.  90 days produced only 2 replenishment cycles — far too few to say anything
    about a stockout rate.
2.  A single SKU cannot show whether the ABC tiering actually pays off.

This script fixes both: every SKU is simulated over 365 days with 100
independent replications, and three policies are compared on the same demand
draws (common random numbers), so cost differences are not noise.

It also adds the two things the analysis was missing:

*   a **cost–service frontier** and the cost-optimal service level;
*   lead-time variability in the safety-stock formula
    (``σ_LT² ≈ L·σ_d² + d̄²·σ_L²``), which dominates for slow movers.

Usage
-----
    python python/inventory_simulation.py
    python python/inventory_simulation.py --reps 200 --shortage-cost 50

Outputs go to ``results/`` and ``images/``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"

ABC_TARGETS = {"A": 0.99, "B": 0.95, "C": 0.90}


# ---------------------------------------------------------------- analytics


def abc_classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Annual_Volume"] = df["Annual_Demand"] * df["Unit_Cost"]
    df = df.sort_values("Annual_Volume", ascending=False).reset_index(drop=True)
    cum = df["Annual_Volume"].cumsum() / df["Annual_Volume"].sum() * 100
    df["ABC_Class"] = np.where(cum <= 80, "A", np.where(cum <= 95, "B", "C"))
    return df


def eoq(demand: float, unit_cost: float, ordering_cost: float, holding_rate: float) -> float:
    h = unit_cost * holding_rate
    if h <= 0:
        return 0.0
    return float(np.sqrt(2 * demand * ordering_cost / h))


def lead_time_demand_sigma(d_bar: float, d_sd: float, l_bar: float, l_sd: float) -> float:
    """σ of demand over the lead time — includes lead-time variability.

    ``σ_LT² ≈ L·σ_d² + d̄²·σ_L²``.  Most textbook EOQ/ROP examples drop the
    second term; for slow movers (d̄ small, few units per day) it is the term
    that actually drives the safety stock.
    """
    return float(np.sqrt(l_bar * d_sd**2 + d_bar**2 * l_sd**2))


def safety_stock(sigma_lt: float, service_level: float) -> float:
    return float(stats.norm.ppf(service_level) * sigma_lt)


# ---------------------------------------------------------------- simulation


@dataclass
class Policy:
    name: str
    reorder_point: float
    order_qty: float


def simulate_policy(
    sku: pd.Series,
    policy: Policy,
    demand_draws: np.ndarray,
    lead_draws: np.ndarray,
    horizon: int,
    reps: int,
    holding_rate: float,
    ordering_cost: float,
    shortage_cost: float,
) -> dict:
    """Daily-review ``(s, Q)`` simulation, vectorised across replications.

    ``demand_draws`` / ``lead_draws`` have shape ``(reps, horizon)`` and are
    shared across the three policies, so every policy faces the same world.
    """
    max_lt = int(lead_draws.max()) + 2
    sched = np.zeros((reps, horizon + max_lt + 1))
    on_hand = np.full(reps, policy.order_qty, dtype=float)
    pipeline = np.zeros(reps, dtype=float)  # outstanding orders, O(1) lookup

    h_daily = sku["Unit_Cost"] * holding_rate / 365.0
    rows = np.arange(reps)

    total_demand = lost_units = holding_cost = 0.0
    order_cost = stockout_cost = 0.0
    inventory_area = 0.0
    stockout_days = 0
    n_orders = 0

    for t in range(horizon):
        arrivals = sched[:, t]
        on_hand += arrivals
        pipeline -= arrivals

        demand = demand_draws[:, t]
        sales = np.minimum(on_hand, demand)
        lost = demand - sales
        on_hand -= sales

        total_demand += demand.sum()
        lost_units += lost.sum()
        stockout_cost += lost.sum() * shortage_cost
        stockout_days += int((lost > 0).sum())
        holding_cost += on_hand.sum() * h_daily
        inventory_area += on_hand.sum()

        reorder = (on_hand + pipeline) <= policy.reorder_point
        if reorder.any():
            idx = rows[reorder]
            arrive = t + lead_draws[reorder, t].astype(int)
            np.add.at(sched, (idx, arrive), policy.order_qty)
            pipeline[reorder] += policy.order_qty
            order_cost += reorder.sum() * ordering_cost
            n_orders += int(reorder.sum())

    # costs are averaged over replications -> "one average year"
    holding_cost /= reps
    order_cost /= reps
    stockout_cost /= reps
    total_cost = holding_cost + order_cost + stockout_cost
    return {
        "sku": sku["Product_ID"],
        "abc": sku["ABC_Class"],
        "policy": policy.name,
        "fill_rate": 1 - lost_units / max(total_demand, 1e-9),
        "stockout_day_share": stockout_days / (reps * horizon),
        "avg_inventory": inventory_area / (reps * horizon),
        "orders_per_rep": n_orders / reps,
        "holding_cost": holding_cost,
        "ordering_cost": order_cost,
        "stockout_cost": stockout_cost,
        "total_cost": total_cost,
    }


# ---------------------------------------------------------------- frontier


def optimal_z(
    q: float,
    unit_cost: float,
    sigma_lt: float,
    annual_demand: float,
    holding_rate: float,
    shortage_cost: float,
) -> tuple[float, float]:
    """Pick z minimising holding + expected-shortage cost for one SKU."""
    best = (0.0, np.inf)
    for z in np.arange(0.0, 3.4, 0.05):
        loss = stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))
        cost = (
            (q / 2 + z * sigma_lt) * unit_cost * holding_rate
            + loss * sigma_lt * annual_demand / max(q, 1) * shortage_cost
        )
        if cost < best[1]:
            best = (float(z), float(cost))
    return best


def service_frontier(
    df: pd.DataFrame,
    levels: np.ndarray,
    holding_rate: float,
    ordering_cost: float,
    shortage_cost: float,
) -> pd.DataFrame:
    """Annual cost vs. service level (holding + ordering + expected shortage)."""
    rows = []
    for level in levels:
        z = stats.norm.ppf(level)
        ss = z * df["sigma_lt"]
        q = df["EOQ"].clip(lower=1)
        holding = (q / 2 + ss) * df["Unit_Cost"] * holding_rate
        ordering = df["Annual_Demand"] / q * ordering_cost
        # standard normal loss function
        loss = stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))
        shortage = loss * df["sigma_lt"] * df["Annual_Demand"] / q * shortage_cost
        rows.append(
            {
                "service_level": level,
                "z": z,
                "safety_stock_units": ss.sum(),
                "holding_cost": holding.sum(),
                "ordering_cost": ordering.sum(),
                "expected_shortage_cost": shortage.sum(),
                "total_cost": (holding + ordering + shortage).sum(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=365)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--ordering-cost", type=float, default=50.0)
    ap.add_argument("--holding-rate", type=float, default=0.25)
    ap.add_argument("--lead-time-mean", type=float, default=7.0)
    ap.add_argument("--lead-time-cv", type=float, default=0.30)
    ap.add_argument("--shortage-cost", type=float, default=25.0, help="$ per unit short")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = abc_classify(pd.read_csv(ROOT / "data" / "superstore_products.csv"))
    df["daily_mean"] = df["Annual_Demand"] / 365.0
    df["EOQ"] = [
        eoq(d, c, args.ordering_cost, args.holding_rate)
        for d, c in zip(df["Annual_Demand"], df["Unit_Cost"])
    ]

    l_bar = args.lead_time_mean
    l_sd = l_bar * args.lead_time_cv
    # unit demand -> Poisson: var = mean, so sigma_d = sqrt(mean).
    # Slow movers are therefore high-CV by construction (CV ~ 1/sqrt(mean)).
    df["sigma_lt"] = [
        lead_time_demand_sigma(d, np.sqrt(d), l_bar, l_sd) for d in df["daily_mean"]
    ]

    n_a = int((df["ABC_Class"] == "A").sum())
    print(f"SKUs: {len(df)} | A={n_a} B={(df.ABC_Class=='B').sum()} C={(df.ABC_Class=='C').sum()}")
    print(f"A-class share of dollar volume: "
          f"{df[df.ABC_Class=='A'].Annual_Volume.sum() / df.Annual_Volume.sum():.1%}")
    print(f"median EOQ: {df.EOQ.median():.1f} units | median sigma_LT: {df.sigma_lt.median():.2f}")

    # ---------------- policies
    policies = []
    for _, row in df.iterrows():
        q = max(row["EOQ"], 1.0)
        base_rop = row["daily_mean"] * l_bar
        z_opt, _ = optimal_z(
            q, row["Unit_Cost"], row["sigma_lt"], row["Annual_Demand"],
            args.holding_rate, args.shortage_cost,
        )
        policies.append(
            (
                row,
                Policy("no safety stock", base_rop, q),
                Policy("uniform 95% SL", base_rop + safety_stock(row["sigma_lt"], 0.95), q),
                Policy(
                    "ABC tiered (99/95/90)",
                    base_rop + safety_stock(row["sigma_lt"], ABC_TARGETS[row["ABC_Class"]]),
                    q,
                ),
                Policy("cost-optimal per SKU", base_rop + z_opt * row["sigma_lt"], q),
            )
        )

    # ---------------- simulate (common random numbers per SKU)
    results: dict[str, list[dict]] = {}
    for row, *pols in policies:
        lam = row["daily_mean"]
        demand = rng.poisson(lam, size=(args.reps, args.horizon)).astype(float)
        lead = np.clip(
            rng.lognormal(np.log(l_bar), args.lead_time_cv, size=(args.reps, args.horizon)),
            1, None,
        ).round()
        for pol in pols:
            m = simulate_policy(
                row, pol, demand, lead, args.horizon, args.reps,
                args.holding_rate, args.ordering_cost, args.shortage_cost,
            )
            results.setdefault(pol.name, []).append(m)

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    detail = pd.concat([pd.DataFrame(v) for v in results.values()], ignore_index=True)
    detail.to_csv(RESULTS / "simulation_detail.csv", index=False)

    summary = []
    for name, rows in results.items():
        d = pd.DataFrame(rows)
        summary.append(
            {
                "policy": name,
                "skus": len(d),
                "fill_rate": d["fill_rate"].mean(),
                "stockout_day_share": d["stockout_day_share"].mean(),
                "orders_per_year": d["orders_per_rep"].mean(),
                "avg_inventory_units": d["avg_inventory"].mean(),
                "holding_cost": d["holding_cost"].sum(),
                "ordering_cost": d["ordering_cost"].sum(),
                "stockout_cost": d["stockout_cost"].sum(),
                "total_cost": d["holding_cost"].sum() + d["ordering_cost"].sum() + d["stockout_cost"].sum(),
            }
        )
    comp = pd.DataFrame(summary)
    base = comp.loc[comp.policy == "no safety stock", "total_cost"].iloc[0]
    comp["saving_vs_no_ss"] = 1 - comp["total_cost"] / base
    comp.to_csv(RESULTS / "simulation_summary.csv", index=False)
    print("\n" + comp.drop(columns=["skus"]).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ---------------- frontier
    levels = np.array([0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995])
    front = service_frontier(df, levels, args.holding_rate, args.ordering_cost, args.shortage_cost)
    front_out = front.copy()
    front_out["service_level"] = (front_out["service_level"] * 100).round(1)
    front_out.to_csv(RESULTS / "service_level_frontier.csv", index=False)
    best = front.loc[front["total_cost"].idxmin()]
    print(f"\ncost-optimal service level: {best['service_level']:.1%} "
          f"(total ${best['total_cost']:,.0f}/yr)")

    # ---------------- optimal service level vs shortage cost
    sens_rows = []
    for sc in (10, 25, 50, 100, 250, 500):
        f = service_frontier(df, levels, args.holding_rate, args.ordering_cost, sc)
        b = f.loc[f["total_cost"].idxmin()]
        sens_rows.append(
            {
                "shortage_cost": sc,
                "optimal_service_level": b["service_level"],
                "min_total_cost": b["total_cost"],
            }
        )
    sens = pd.DataFrame(sens_rows)
    sens_out = sens.copy()
    sens_out["optimal_service_level"] = (sens_out["optimal_service_level"] * 100).round(1)
    sens_out.to_csv(RESULTS / "optimal_service_vs_shortage_cost.csv", index=False)
    print("\noptimal service level vs shortage cost:")
    print(sens_out.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    # ---------------- chart
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sl = front["service_level"] * 100
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        ax.plot(sl, front["total_cost"] / 1e3, "o-", color="#2f5d8c", label="total cost")
        ax.plot(sl, front["holding_cost"] / 1e3, "s--", color="#9aa5b1", label="holding")
        ax.plot(sl, front["expected_shortage_cost"] / 1e3, "^--", color="#c0504d", label="expected shortage")
        ax.plot(sl, front["ordering_cost"] / 1e3, "d--", color="#8a6d3b", label="ordering")
        b = best["service_level"] * 100
        ax.axvline(b, color="green", ls=":", lw=1.2)
        ax.annotate(f"optimum ≈ {b:.1f}%", (b, best["total_cost"] / 1e3),
                    textcoords="offset points", xytext=(-64, 16), fontsize=8, color="green")
        ax.set_xlabel("service level (%)")
        ax.set_ylabel("annual cost ($k)")
        ax.set_title("Inventory cost–service frontier — 200 SKUs")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(IMAGES / "cost_service_frontier.png", dpi=150)
        print(f"wrote {IMAGES / 'cost_service_frontier.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"chart skipped: {exc}")

    # ---------------- markdown
    tiered_cost = comp.loc[comp.policy == "ABC tiered (99/95/90)", "total_cost"].iloc[0]
    uniform_cost = comp.loc[comp.policy == "uniform 95% SL", "total_cost"].iloc[0]
    lines = [
        "# Inventory simulation — refreshed numbers",
        "",
        f"SKUs = {len(df)} · horizon = {args.horizon} days · replications = {args.reps} · "
        "common random numbers across policies",
        "",
        f"- ordering cost ${args.ordering_cost:.0f}/order · holding rate {args.holding_rate:.0%} · "
        f"shortage cost ${args.shortage_cost:.0f}/unit",
        f"- lead time lognormal, mean {args.lead_time_mean:.0f} d, CV {args.lead_time_cv:.0%}; "
        "daily demand Poisson (unit demand)",
        "",
        "## Policy comparison (365 days × 100 replications, all 200 SKUs)",
        "",
        comp.drop(columns=["skus"]).to_string(index=False, float_format=lambda v: f"{v:,.2f}"),
        "",
        "## Cost–service frontier",
        "",
        front_out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"),
        "",
        "## Optimal service level vs. shortage cost",
        "",
        sens_out.to_string(index=False, float_format=lambda v: f"{v:,.1f}"),
        "",
        f"Cost-optimal service level: **{best['service_level']:.1%}** "
        f"(total ${best['total_cost']:,.0f}/yr).",
        "",
        f"ABC tiering vs uniform 95%: ${tiered_cost:,.0f} vs ${uniform_cost:,.0f} "
        f"({1 - tiered_cost / uniform_cost:+.1%}).",
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
