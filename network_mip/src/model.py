"""两级库存网络的混合整数规划模型。

决策
----
1. 开哪些区域仓        y_d ∈ {0,1}
2. 每个需求区由谁服务   x_{c,d} ∈ {0,1}（含一个"中心仓直发"的虚拟仓，fixed cost = 0）
3. 每个仓每个品类的安全库存  SS_{d,f} ≥ 0

目标（年化成本）
----------------
固定成本 + 干线运输 + 支线运输 + 持有成本（周转库存 + 安全库存）

关键设计
--------
- **直发是选项之一**：如果不设虚拟仓，"不建区域仓"这个方案根本不在可行域里，
  优化器只能被迫开仓，结论就会偏。
- **安全库存线性化**：按 σ 与需求同比例近似，σ_LT = CV_f · D_{d,f} · √L，
  于是 ``SS ≥ z · CV_f · √L · Σ_c μ_{c,f} x_{c,d}`` 是线性的，模型保持纯 MILP。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pulp
from scipy import stats

from src.instance import Instance, Node, distance_matrix, haversine_km

DIRECT_NAME = "__direct__"


@dataclass
class Solution:
    status: str
    objective: float | None
    open_dcs: list[str] = field(default_factory=list)
    assignment: dict[str, str] = field(default_factory=dict)
    safety_stock: dict[tuple[str, str], float] = field(default_factory=dict)
    breakdown: dict[str, float] = field(default_factory=dict)
    is_relaxation: bool = False

    @property
    def n_open(self) -> int:
        return len(self.open_dcs)


def direct_node(inst: Instance) -> Node:
    """虚拟仓：从中心仓直发，没有固定成本、没有干线成本、提前期最长。"""
    return Node(
        name=DIRECT_NAME,
        lat=inst.central.lat,
        lon=inst.central.lon,
        fixed_cost=0.0,
        capacity=float("inf"),
    )


def facilities(inst: Instance) -> list[Node]:
    """候选设施 = 8 个区域仓 + 1 个直发选项。"""
    return list(inst.dcs) + [direct_node(inst)]


def facility_economics(inst: Instance, fac: Node) -> dict:
    """每个设施的路由参数：距离、费率、补货提前期、是否有干线。"""
    p = inst.params
    if fac.name == DIRECT_NAME:
        return {"is_direct": True, "lt": p["lead_time_direct"], "inbound_km": 0.0}
    return {
        "is_direct": False,
        "lt": p["lead_time_central_dc"],
        "inbound_km": haversine_km(inst.central, fac),
    }


def _leg_distance(inst: Instance, zone: Node, fac: Node, zone_dc: dict) -> float:
    if fac.name == DIRECT_NAME:
        return haversine_km(inst.central, zone)
    return zone_dc[(zone.name, fac.name)]


def _leg_rate(inst: Instance, fac: Node) -> float:
    p = inst.params
    return p["direct_rate"] if fac.name == DIRECT_NAME else p["outbound_rate"]


def cost_terms(
    inst: Instance,
    assignment: dict[str, str],
    open_dcs: set[str],
    service_level: float,
    fixed_cost_mult: float = 1.0,
) -> dict[str, float]:
    """给定"谁服务谁"，算一版年化成本（MIP 与各基线共用，保证可比）。"""
    p = inst.params
    z = stats.norm.ppf(service_level)
    _, zone_dc = distance_matrix(inst)
    review = p["review_weeks"]

    facs = {f.name: f for f in facilities(inst)}
    served = [f for f in open_dcs if f in facs]

    fixed = sum(facs[d].fixed_cost * fixed_cost_mult for d in served)
    inbound = 0.0
    outbound = 0.0
    cycle = safety = 0.0

    for d in served:
        fac = facs[d]
        econ = facility_economics(inst, fac)
        weekly_total = sum(
            inst.demand(c, f.name) for c in inst.zone_names if assignment.get(c) == d
            for f in inst.families
        )
        inbound += (
            p["inbound_rate"] * econ["inbound_km"] / 1000.0 * 52.0 * weekly_total
            if not econ["is_direct"]
            else 0.0
        )
        for fam in inst.families:
            units = sum(
                inst.demand(c, fam.name)
                for c in inst.zone_names
                if assignment.get(c) == d
            )
            cycle += fam.unit_value * (review / 2.0) * units
            safety += fam.unit_value * z * fam.cv * (econ["lt"] ** 0.5) * units

    for c, d in assignment.items():
        zone = next(x for x in inst.zones if x.name == c)
        fac = facs[d]
        outbound += (
            _leg_rate(inst, fac) * _leg_distance(inst, zone, fac, zone_dc) / 1000.0
            * 52.0 * inst.zone_weekly_total(c)
        )

    holding_cycle = p["holding_rate"] * cycle
    holding_safety = p["holding_rate"] * safety
    total = fixed + inbound + outbound + holding_cycle + holding_safety
    return {
        "fixed": fixed,
        "inbound": inbound,
        "outbound": outbound,
        "holding_cycle": holding_cycle,
        "holding_safety": holding_safety,
        "total": total,
    }


def build_model(
    inst: Instance,
    service_level: float | None = None,
    fixed_cost_mult: float = 1.0,
    relax: bool = False,
):
    """构造 MILP；``relax=True`` 时把 0-1 变量放松成连续（求下界）。"""
    p = inst.params
    sl = service_level if service_level is not None else p["service_level"]
    z = stats.norm.ppf(sl)
    _, zone_dc = distance_matrix(inst)
    facs = facilities(inst)
    review = p["review_weeks"]

    cat = "Continuous" if relax else "Binary"
    prob = pulp.LpProblem("multi_echelon_inventory", pulp.LpMinimize)

    y = {f.name: pulp.LpVariable(f"open_{i}", 0, 1, cat=cat) for i, f in enumerate(facs)}
    x = {
        (c.name, f.name): pulp.LpVariable(f"assign_{i}_{j}", 0, 1, cat=cat)
        for i, c in enumerate(inst.zones)
        for j, f in enumerate(facs)
    }
    ss = {
        (f.name, fam.name): pulp.LpVariable(f"ss_{i}_{j}", 0, None)
        for i, f in enumerate(facs)
        for j, fam in enumerate(inst.families)
    }

    # 每个需求区必须且只能被一个设施服务
    for c in inst.zones:
        prob += pulp.lpSum(x[(c.name, f.name)] for f in facs) == 1, f"serve_{c.name}"

    # 没开的仓不能接活
    for f in facs:
        for c in inst.zones:
            prob += x[(c.name, f.name)] <= y[f.name], f"link_{c.name}_{f.name}"

    # 吞吐能力（件/年）；直发虚拟仓不限
    for f in facs:
        if f.name == DIRECT_NAME:
            continue
        throughput = pulp.lpSum(
            52.0 * inst.demand(c.name, fam.name) * x[(c.name, f.name)]
            for c in inst.zones
            for fam in inst.families
        )
        prob += throughput <= f.capacity * y[f.name], f"capacity_{f.name}"

    # 安全库存下界
    for f in facs:
        lt = facility_economics(inst, f)["lt"]
        for fam in inst.families:
            served = pulp.lpSum(
                inst.demand(c.name, fam.name) * x[(c.name, f.name)] for c in inst.zones
            )
            prob += ss[(f.name, fam.name)] >= z * fam.cv * (lt ** 0.5) * served, (
                f"ss_{f.name}_{fam.name}"
            )

    # ---- 目标
    fixed = pulp.lpSum(f.fixed_cost * fixed_cost_mult * y[f.name] for f in facs)

    inbound = pulp.lpSum(
        p["inbound_rate"] * facility_economics(inst, f)["inbound_km"] / 1000.0
        * pulp.lpSum(
            52.0 * inst.demand(c.name, fam.name) * x[(c.name, f.name)]
            for c in inst.zones
            for fam in inst.families
        )
        for f in facs
        if f.name != DIRECT_NAME
    )

    outbound = pulp.lpSum(
        _leg_rate(inst, f) * _leg_distance(inst, c, f, zone_dc) / 1000.0
        * 52.0 * inst.zone_weekly_total(c.name)
        * x[(c.name, f.name)]
        for c in inst.zones
        for f in facs
    )

    holding = pulp.lpSum(
        p["holding_rate"] * fam.unit_value
        * (
            (review / 2.0)
            * pulp.lpSum(inst.demand(c.name, fam.name) * x[(c.name, f.name)] for c in inst.zones)
            + ss[(f.name, fam.name)]
        )
        for f in facs
        for fam in inst.families
    )

    prob += fixed + inbound + outbound + holding
    return prob, {"y": y, "x": x, "ss": ss}, {"service_level": sl, "fixed_cost_mult": fixed_cost_mult}


def solve(
    inst: Instance,
    service_level: float | None = None,
    fixed_cost_mult: float = 1.0,
    relax: bool = False,
    time_limit: int = 120,
) -> Solution:
    prob, v, info = build_model(inst, service_level, fixed_cost_mult, relax)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    if prob.objective.value() is None:  # pragma: no cover
        return Solution(status=status, objective=None, is_relaxation=relax)

    if relax:
        return Solution(
            status=status,
            objective=prob.objective.value(),
            breakdown={"total": prob.objective.value()},
            is_relaxation=True,
        )

    facs = facilities(inst)
    assignment = {
        c.name: max(facs, key=lambda f: v["x"][(c.name, f.name)].value() or 0.0).name
        for c in inst.zones
    }
    open_dcs = sorted(
        f.name for f in inst.dcs if (v["y"][f.name].value() or 0.0) > 0.5
    )
    open_set = set(open_dcs)
    if any(dc == DIRECT_NAME for dc in assignment.values()):
        open_set.add(DIRECT_NAME)

    breakdown = cost_terms(inst, assignment, open_set, info["service_level"], fixed_cost_mult)
    return Solution(
        status=status,
        objective=prob.objective.value(),
        open_dcs=open_dcs,
        assignment=assignment,
        safety_stock={
            (f.name, fam.name): (v["ss"][(f.name, fam.name)].value() or 0.0)
            for f in facs
            for fam in inst.families
        },
        breakdown=breakdown,
    )
