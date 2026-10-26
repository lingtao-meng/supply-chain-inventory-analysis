"""对比基线：不优化会怎么干。"""

from __future__ import annotations

from scipy import stats

from src.instance import Instance, haversine_km


def single_central(inst: Instance, service_level: float | None = None) -> dict:
    """现状：只用一个中心仓直发所有客户区，不设区域仓。

    这是对比 MIP 的关键基线——它没有固定成本，但运输距离长、补货提前期长，
    安全库存也因此更高。
    """
    p = inst.params
    sl = service_level if service_level is not None else p["service_level"]
    z = stats.norm.ppf(sl)
    lt = p["lead_time_direct"]
    review = p["review_weeks"]

    dist = {zc.name: haversine_km(inst.central, zc) for zc in inst.zones}
    transport = sum(
        p["direct_rate"] * dist[c] / 1000.0 * 52.0 * inst.zone_weekly_total(c)
        for c in inst.zone_names
    )

    cycle = safety = 0.0
    for fam in inst.families:
        units = sum(fam.weekly_demand.values())
        cycle += fam.unit_value * (review / 2.0) * units
        safety += fam.unit_value * z * fam.cv * (lt ** 0.5) * units

    holding_c = p["holding_rate"] * cycle
    holding_s = p["holding_rate"] * safety
    total = transport + holding_c + holding_s
    return {
        "policy": "single central (status quo)",
        "open_dcs": 0,
        "fixed": 0.0,
        "inbound": 0.0,
        "outbound": transport,
        "holding_cycle": holding_c,
        "holding_safety": holding_s,
        "total": total,
    }


def nearest_assignment(inst: Instance) -> dict[str, str]:
    """每个需求区交给直线距离最近的仓（不考虑容量和固定成本）。"""
    return {
        zc.name: min(inst.dcs, key=lambda d: haversine_km(zc, d)).name for zc in inst.zones
    }


def all_dcs_nearest(inst: Instance, service_level: float | None = None) -> dict:
    """把 8 个仓全开，按距离就近分配——"什么都要"的做法。"""
    from src.model import cost_terms

    assignment = nearest_assignment(inst)
    open_set = {d.name for d in inst.dcs}
    terms = cost_terms(
        inst,
        assignment,
        open_set,
        service_level if service_level is not None else inst.params["service_level"],
    )
    return {"policy": "all 8 DCs, nearest assignment", "open_dcs": len(open_set), **terms}
