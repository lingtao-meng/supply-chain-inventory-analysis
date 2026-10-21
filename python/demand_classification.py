#!/usr/bin/env python3
"""Demand classification (ADI / CV²) for the Superstore SKUs.

Not every SKU should be forecast the same way.  Syntetos–Boylan classify demand
along two axes:

* **ADI** (average demand interval) = periods / non-zero periods;
* **CV²** = squared coefficient of variation of the *non-zero* demand sizes.

| | CV² < 0.49 | CV² ≥ 0.49 |
|---|---|---|
| ADI < 1.32 | smooth | erratic |
| ADI ≥ 1.32 | intermittent | lumpy |

The class determines the method: exponential smoothing for smooth, a quantile
buffer for erratic, Croston/SBA for intermittent, and aggregate + manual review
for lumpy.  The output is joined to the 200-SKU master used by the inventory
simulation, so the budget allocation can use a realistic demand CV per SKU.

Usage
-----
    python python/demand_classification.py
    python python/demand_classification.py --data "/path/Sample - Superstore.csv"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
MASTER = ROOT / "data" / "superstore_products.csv"
RAW_NAME = "Sample - Superstore.csv"

ADI_CUT, CV2_CUT = 1.32, 0.49

METHOD = {
    "smooth": "指数平滑 / ARIMA",
    "erratic": "指数平滑 + 分位数缓冲",
    "intermittent": "Croston / SBA",
    "lumpy": "聚合预测 + 人工复核",
}


def find_data(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [ROOT / "data" / RAW_NAME]
    candidates += sorted(Path.home().glob(f".cache/kagglehub/datasets/**/{RAW_NAME}"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Superstore transaction CSV not found; pass --data")


def load_transactions(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin1", usecols=["Order Date", "Product ID", "Quantity"])
    raw["Order Date"] = pd.to_datetime(raw["Order Date"])
    return raw


def weekly_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """Product × week demand matrix, zero-filled over the full horizon."""
    raw = raw.assign(week=raw["Order Date"].dt.to_period("W"))
    matrix = raw.groupby(["Product ID", "week"])["Quantity"].sum().unstack(fill_value=0)
    full_weeks = pd.period_range(raw["week"].min(), raw["week"].max(), freq="W")
    return matrix.reindex(columns=full_weeks, fill_value=0)


def classify(adi: float, cv2: float) -> str:
    if adi < ADI_CUT and cv2 < CV2_CUT:
        return "smooth"
    if adi < ADI_CUT:
        return "erratic"
    if cv2 < CV2_CUT:
        return "intermittent"
    return "lumpy"


def demand_stats(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, series in matrix.iterrows():
        values = series.to_numpy(dtype=float)
        non_zero = values[values > 0]
        if len(non_zero) == 0:
            continue
        adi = len(values) / len(non_zero)
        cv2 = float((non_zero.std(ddof=1) / non_zero.mean()) ** 2) if len(non_zero) > 1 else 0.0
        rows.append(
            {
                "Product_ID": product,
                "weeks": len(values),
                "non_zero_weeks": int(len(non_zero)),
                "adi": adi,
                "cv2": cv2,
                "mean_weekly": float(values.mean()),
                "std_weekly": float(values.std(ddof=1)),
                "cv_weekly": float(values.std(ddof=1) / values.mean()) if values.mean() > 0 else 0.0,
                "avg_nonzero_demand": float(non_zero.mean()),
                "demand_class": classify(adi, cv2),
            }
        )
    return pd.DataFrame(rows)


def plot(stats: pd.DataFrame, summary: pd.DataFrame, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "smooth": "#1f6f78",
        "erratic": "#d99a2b",
        "intermittent": "#2f5d8c",
        "lumpy": "#9b3b3b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))

    ax = axes[0]
    for name, group in stats.groupby("demand_class"):
        ax.scatter(group["adi"], group["cv2"].clip(lower=1e-4), s=14, alpha=0.65,
                   color=colors[name], label=name)
    ax.axvline(ADI_CUT, color="#333", linestyle="--", linewidth=1)
    ax.axhline(CV2_CUT, color="#333", linestyle="--", linewidth=1)
    ax.set_xlabel("ADI (average demand interval)")
    ax.set_ylabel("CV² of non-zero demand")
    ax.set_yscale("log")
    ax.set_title("Demand classification (Syntetos–Boylan)")
    for label, x, y in [
        ("smooth", 1.45, 1e-2),
        ("erratic", 1.45, 2.0),
        ("intermittent", 16, 1e-2),
        ("lumpy", 16, 2.0),
    ]:
        ax.text(x, y, label, fontsize=8, color="#666", ha="left", va="center")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    view = summary.sort_values("products")
    ax2.bar(view["demand_class"], view["products"], color=[colors[c] for c in view["demand_class"]])
    for i, value in enumerate(view["products"]):
        ax2.text(i, value + 1, f"{int(value)}", ha="center", fontsize=8)
    ax2.set_ylabel("SKUs")
    ax2.set_title("How many SKUs fall in each class")

    fig.tight_layout()
    fig.savefig(images / "demand_classification.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    path = find_data(args.data)
    raw = load_transactions(path)
    matrix = weekly_matrix(raw)
    stats = demand_stats(matrix)

    master = pd.read_csv(MASTER)
    stats = stats.merge(
        master[["Product_ID", "Category", "Annual_Demand", "Unit_Cost", "Annual_Revenue"]],
        on="Product_ID",
        how="left",
    )
    stats["in_master"] = stats["Annual_Demand"].notna()

    summary = (
        stats.groupby("demand_class")
        .agg(
            products=("Product_ID", "size"),
            share=("Product_ID", lambda s: len(s) / len(stats)),
            avg_nonzero_weeks=("non_zero_weeks", "mean"),
            avg_adi=("adi", "mean"),
            avg_cv2=("cv2", "mean"),
        )
        .reset_index()
        .sort_values("products", ascending=False)
    )
    summary["method"] = summary["demand_class"].map(METHOD)

    stats.to_csv(RESULTS / "demand_classification.csv", index=False)
    summary.to_csv(RESULTS / "demand_class_summary.csv", index=False)
    (RESULTS / "demand_classification.json").write_text(
        json.dumps(
            {
                "data": str(path),
                "weeks": int(matrix.shape[1]),
                "products": int(len(stats)),
                "master_products": int(stats["in_master"].sum()),
                "thresholds": {"adi": ADI_CUT, "cv2": CV2_CUT},
                "summary": summary.to_dict(orient="records"),
                "methods": METHOD,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"data: {path}")
    print(f"{matrix.shape[1]} weeks, {len(stats):,} products, {int(stats['in_master'].sum())} in the 200-SKU master")
    print(summary.to_string(index=False))

    plot(stats, summary, IMAGES)
    print(f"wrote {IMAGES / 'demand_classification.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
