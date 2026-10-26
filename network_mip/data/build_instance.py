#!/usr/bin/env python3
"""从 DataCo 订单数据生成两级库存网络的算例。

为什么用真实数据建算例
----------------------
纯随机的算例没有说服力（"为什么这个需求是 37 件？"）。这里的需求、季节波动、
单价全部来自 DataCo 供应链数据集：
  - 需求 = 各国 × 各品类的实际销量（2015-01 → 2017-09）
  - CV   = 该品类周度销量的实际变异系数
  - 单价 = 该品类的销售额 / 销量

地理坐标和成本参数是公开的近似值，写在下面，便于复核。

Usage
-----
    python data/build_instance.py --dataco ../data/DataCoSupplyChainDataset.csv

输出 data/instance.json（已提交到仓库，CI 不需要原始 CSV）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# 需求区域：DataCo 销量前 20 的国家（西语国名 → 英文 + 地理中心）
ZONES = {
    "Estados Unidos": ("United States", 39.8, -98.6),
    "Francia": ("France", 46.2, 2.2),
    "México": ("Mexico", 23.6, -102.5),
    "Alemania": ("Germany", 51.2, 10.4),
    "Australia": ("Australia", -25.3, 133.8),
    "Reino Unido": ("United Kingdom", 55.4, -3.4),
    "Brasil": ("Brazil", -14.2, -51.9),
    "China": ("China", 35.9, 104.2),
    "Italia": ("Italy", 41.9, 12.6),
    "India": ("India", 20.6, 78.9),
    "España": ("Spain", 40.5, -3.7),
    "Indonesia": ("Indonesia", -0.8, 113.9),
    "El Salvador": ("El Salvador", 13.8, -88.9),
    "Honduras": ("Honduras", 15.2, -86.2),
    "República Dominicana": ("Dominican Republic", 18.7, -70.2),
    "Cuba": ("Cuba", 21.5, -77.8),
    "Turquía": ("Turkey", 38.9, 35.2),
    "Nicaragua": ("Nicaragua", 12.9, -85.2),
    "Guatemala": ("Guatemala", 15.8, -90.2),
    "Nigeria": ("Nigeria", 9.1, 8.7),
}

# 区域仓候选点（8 个枢纽城市）
DCS = [
    # name, lat, lon, 年化固定成本, 年吞吐能力(件)
    ("Los Angeles", 34.05, -118.24, 100_000, 900_000),
    ("Rotterdam", 51.92, 4.48, 105_000, 1_200_000),
    ("São Paulo", -23.55, -46.63, 80_000, 600_000),
    ("Dubai", 25.20, 55.27, 75_000, 600_000),
    ("Singapore", 1.35, 103.82, 90_000, 800_000),
    ("Shanghai", 31.23, 121.47, 95_000, 1_000_000),
    ("Lagos", 6.52, 3.38, 55_000, 400_000),
    ("Johannesburg", -26.20, 28.05, 60_000, 450_000),
]

CENTRAL = ("Memphis (central DC)", 35.15, -90.05)

FAMILIES = ["Fan Shop", "Apparel", "Golf", "Footwear", "Outdoors"]

PARAMS = {
    "weeks_per_year": 52,
    "demand_scale": 20,              # DataCo 给的是样本量级，放大到中型零售商规模
    "lead_time_central_dc": 3.0,     # weeks, central -> regional DC
    "lead_time_dc_zone": 1.0,        # weeks, regional DC -> zone
    "lead_time_direct": 4.0,         # weeks, central -> zone (direct ship)
    "holding_rate": 0.25,            # per year, on unit value
    "inbound_rate": 0.10,            # $ per unit per 1000 km（海运/整柜，最便宜）
    "outbound_rate": 0.55,           # $ per unit per 1000 km（区域配送，最贵）
    "direct_rate": 0.45,             # $ per unit per 1000 km（国际直发）
    "review_weeks": 2.0,             # cycle stock = review/2 weeks of demand
    "service_level": 0.95,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataco", default=str(ROOT.parent / "data" / "DataCoSupplyChainDataset.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "instance.json"))
    args = ap.parse_args()

    src = Path(args.dataco)
    if not src.exists():
        raise SystemExit(f"DataCo CSV not found: {src}")

    df = pd.read_csv(src, encoding="latin1", usecols=[
        "order date (DateOrders)", "Order Country", "Department Name",
        "Order Item Quantity", "Order Item Total",
    ])
    df["date"] = pd.to_datetime(df["order date (DateOrders)"])
    df = df[df["Order Country"].isin(ZONES) & df["Department Name"].isin(FAMILIES)]
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    weeks = (df["date"].max() - df["date"].min()).days / 7.0

    families = []
    for fam in FAMILIES:
        sub = df[df["Department Name"] == fam]
        weekly = sub.groupby("week")["Order Item Quantity"].sum()
        weekly = weekly.reindex(
            pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-MON"),
            fill_value=0,
        )
        cv = float(weekly.std(ddof=1) / weekly.mean()) if weekly.mean() > 0 else 0.4
        units = float(sub["Order Item Quantity"].sum())
        value = float(sub["Order Item Total"].sum())
        demand = (
            sub.groupby("Order Country")["Order Item Quantity"].sum()
            .reindex(list(ZONES)).fillna(0.0) / max(weeks, 1.0)
        ) * PARAMS["demand_scale"]
        families.append({
            "name": fam,
            "unit_value": round(value / units, 2) if units else 0.0,
            "cv": round(cv, 3),
            "weekly_demand": {ZONES[k][0]: round(float(v), 3) for k, v in demand.items()},
        })

    instance = {
        "meta": {
            "source": "DataCo SupplyChainDataset",
            "period_weeks": round(weeks, 1),
            "dates": [str(df["date"].min().date()), str(df["date"].max().date())],
            "note": "demand/CV/unit value from real orders; coordinates and cost rates are stated approximations",
        },
        "central": {"name": CENTRAL[0], "lat": CENTRAL[1], "lon": CENTRAL[2]},
        "dcs": [
            {"name": n, "lat": la, "lon": lo, "fixed_cost": fc, "capacity": cap}
            for n, la, lo, fc, cap in DCS
        ],
        "zones": [{"name": v[0], "lat": v[1], "lon": v[2]} for v in ZONES.values()],
        "families": families,
        "params": PARAMS,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(instance, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(sum(f["weekly_demand"].values()) for f in families)
    print(f"zones={len(instance['zones'])} dcs={len(instance['dcs'])} families={len(families)}")
    print(f"total weekly demand = {total:,.0f} units/week, period = {weeks:.0f} weeks")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
