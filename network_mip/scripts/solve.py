#!/usr/bin/env python3
"""求解两级库存网络：基线对比 + MILP + LP 下界 + 灵敏度。

Usage
-----
    python scripts/solve.py
    python scripts/solve.py --service-level 0.99 --fixed-mult 1.5

输出
----
    results/summary.md          结论与表格
    results/policy_comparison.csv
    results/network_solution.csv  开哪些仓、谁服务谁
    results/sensitivity.csv
    images/network_map.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baselines, model  # noqa: E402
from src.instance import load_instance  # noqa: E402

RESULTS = ROOT / "results"
IMAGES = ROOT / "images"


def row(name: str, open_dcs: int, terms: dict) -> dict:
    return {
        "policy": name,
        "open_dcs": open_dcs,
        "fixed": terms.get("fixed", 0.0),
        "inbound": terms.get("inbound", 0.0),
        "outbound": terms.get("outbound", 0.0),
        "holding_cycle": terms.get("holding_cycle", 0.0),
        "holding_safety": terms.get("holding_safety", 0.0),
        "total": terms["total"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-level", type=float, default=None)
    ap.add_argument("--fixed-mult", type=float, default=1.0)
    ap.add_argument("--time-limit", type=int, default=120)
    args = ap.parse_args()

    inst = load_instance()
    sl = args.service_level if args.service_level else inst.params["service_level"]
    print(f"zones={len(inst.zones)} dcs={len(inst.dcs)} families={len(inst.families)} "
          f"service level={sl:.0%}")

    # ---------------- 基线
    base_single = baselines.single_central(inst, sl)
    base_all = baselines.all_dcs_nearest(inst, sl)

    # ---------------- MILP
    sol = model.solve(inst, service_level=sl, fixed_cost_mult=args.fixed_mult,
                      time_limit=args.time_limit)
    print(f"MILP status={sol.status} objective=${sol.objective:,.0f} "
          f"open={len(sol.open_dcs)}")

    # ---------------- LP 下界
    lp = model.solve(inst, service_level=sl, fixed_cost_mult=args.fixed_mult, relax=True)
    gap = (sol.objective - lp.objective) / sol.objective if sol.objective else float("nan")
    print(f"LP bound=${lp.objective:,.0f}  optimality gap={gap:.2%}")

    rows = [
        row("single central (status quo)", 0, base_single),
        row("all 8 DCs, nearest assignment", len(inst.dcs), base_all),
        row("MILP optimum", len(sol.open_dcs), sol.breakdown),
    ]
    comp = pd.DataFrame(rows)
    comp["vs_status_quo"] = 1 - comp["total"] / comp.loc[0, "total"]
    comp.to_csv(RESULTS / "policy_comparison.csv", index=False)
    view = comp.copy()
    for col in ("fixed", "inbound", "outbound", "holding_cycle", "holding_safety", "total"):
        view[col] = view[col].map(lambda v: f"${v:,.0f}")
    view["vs_status_quo"] = comp["vs_status_quo"].map(lambda v: f"{v:.1%}")
    print("\n" + view.to_string(index=False))

    # ---------------- 网络方案
    pretty = lambda n: "central direct" if n == model.DIRECT_NAME else n  # noqa: E731
    net = pd.DataFrame(
        [
            {"zone": z, "served_by": pretty(d), "is_direct": d == model.DIRECT_NAME}
            for z, d in sorted(sol.assignment.items())
        ]
    )
    net.to_csv(RESULTS / "network_solution.csv", index=False)

    # ---------------- 灵敏度
    sens = []
    for fm in (0.5, 1.0, 1.5, 2.0):
        s = model.solve(inst, service_level=sl, fixed_cost_mult=fm, time_limit=args.time_limit)
        sens.append({
            "fixed_cost_multiplier": fm,
            "open_dcs": len(s.open_dcs),
            "total_cost": s.objective,
            "vs_status_quo": 1 - s.objective / base_single["total"],
        })
    for level in (0.90, 0.95, 0.99):
        s = model.solve(inst, service_level=level, fixed_cost_mult=args.fixed_mult,
                        time_limit=args.time_limit)
        sens.append({
            "fixed_cost_multiplier": args.fixed_mult,
            "service_level": level,
            "open_dcs": len(s.open_dcs),
            "total_cost": s.objective,
            "vs_status_quo": 1 - s.objective / base_single["total"],
        })
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(RESULTS / "sensitivity.csv", index=False)

    # ---------------- 地图
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        facs = {f.name: f for f in model.facilities(inst)}
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for zc in inst.zones:
            ax.scatter(zc.lon, zc.lat, s=18, c="#9aa5b1", zorder=2)
            ax.annotate(zc.name, (zc.lon, zc.lat), fontsize=6, color="#6b7280",
                        xytext=(3, 3), textcoords="offset points")
        for d in inst.dcs:
            opened = d.name in sol.open_dcs
            ax.scatter(d.lon, d.lat, s=110 if opened else 45,
                       c="#2f5d8c" if opened else "#d0d5dd",
                       marker="s", zorder=3, edgecolor="white")
            ax.annotate(d.name, (d.lon, d.lat), fontsize=7, fontweight="bold" if opened else "normal",
                        color="#1f3a5f" if opened else "#9aa5b1",
                        xytext=(4, -9), textcoords="offset points")
        for zc_name, fac_name in sol.assignment.items():
            zc = next(z for z in inst.zones if z.name == zc_name)
            fac = facs[fac_name]
            ax.plot([zc.lon, fac.lon], [zc.lat, fac.lat], lw=0.5,
                    color="#c0504d" if fac_name == model.DIRECT_NAME else "#2f5d8c",
                    alpha=0.35, ls=":" if fac_name == model.DIRECT_NAME else "-")
        ax.scatter(inst.central.lon, inst.central.lat, s=140, marker="*",
                   c="#8a6d3b", zorder=4, edgecolor="white")
        ax.annotate("central DC", (inst.central.lon, inst.central.lat), fontsize=7,
                    color="#8a6d3b", xytext=(4, 6), textcoords="offset points")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(f"Two-echelon network — {len(sol.open_dcs)} of {len(inst.dcs)} DCs open")
        fig.tight_layout()
        fig.savefig(IMAGES / "network_map.png", dpi=150)
        print(f"wrote {IMAGES / 'network_map.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"map skipped: {exc}")

    # ---------------- summary.md
    lines = [
        "# 两级库存网络优化 — 结果",
        "",
        f"20 个需求区 · 8 个候选区域仓 · 5 个品类 · 服务水平 {sl:.0%} · "
        f"固定成本系数 {args.fixed_mult:g}",
        "",
        "## 三种做法的年化成本",
        "",
        *["| " + " | ".join(view.columns) + " |", "|" + "|".join(["---"] * len(view.columns)) + "|"],
        *["| " + " | ".join(map(str, r)) + " |" for r in view.values],
        "",
        f"- 相对现状（单中心直发）节省 **{1 - sol.objective / base_single['total']:.1%}**",
        f"- 相对「全开 8 仓」节省 **{1 - sol.objective / base_all['total']:.1%}**",
        f"- LP 下界 ${lp.objective:,.0f}，**最优性 gap {gap:.2%}**"
        + ("（本例 LP 松弛直接取到整数解，所以 gap 为 0——说明这个规模和成本结构下"
           "连续松弛已经足够紧，规模再大才需要 branch-and-bound）" if gap < 1e-6 else ""),
        "",
        "## 选址与分配",
        "",
        f"开仓：{'、'.join(sol.open_dcs)}",
        "",
        "| 需求区 | 服务的仓 |",
        "|---|---|",
        *[f"| {z} | {pretty(d)} |" for z, d in sorted(sol.assignment.items())],
        "",
        "## 灵敏度",
        "",
        sens_df.to_string(index=False, float_format=lambda v: f"{v:,.3f}"),
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
