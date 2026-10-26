"""Two-stage stochastic network design, plus VSS / EVPI / robust comparison.

The deterministic MILP in :mod:`src.model` optimises for *mean* demand.  This
module asks the OR question that follows:

* demand by zone is uncertain; first-stage decisions (which DCs to open, who
  serves whom) are made before demand is known;
* second-stage safety stock adapts to each demand scenario;
* compare three policies under the same scenario set:
  deterministic (mean demand), stochastic (expected cost), robust (min-max);
* report **VSS** (value of the stochastic solution) and **EVPI** (expected
  value of perfect information).

A positive VSS means ignoring demand uncertainty is expensive; a large EVPI
means better forecasting has real option value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp
from scipy import stats

from src import model
from src.instance import Family, Instance, distance_matrix


@dataclass(frozen=True)
class Scenario:
    name: str
    multipliers: dict[str, float]
    probability: float


@dataclass
class StochasticSolution:
    status: str
    objective: float | None
    open_dcs: list[str]
    assignment: dict[str, str]
    expected_cost: float | None = None
    worst_cost: float | None = None
    scenario_costs: list[float] | None = None
    is_robust: bool = False


def generate_scenarios(
    inst: Instance,
    n_scenarios: int = 10,
    sigma: float = 0.15,
    seed: int = 42,
    low: float = 0.6,
    high: float = 1.4,
) -> list[Scenario]:
    """Log-normal demand multipliers per zone, clipped to a plausible band."""
    rng = np.random.default_rng(seed)
    names = inst.zone_names
    raw = np.clip(
        rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=(n_scenarios, len(names))),
        low,
        high,
    )
    # Mean-preserving: each zone's expected multiplier is exactly 1, so the
    # deterministic model (mean demand) is the expectation of the scenario set.
    raw = raw / raw.mean(axis=0, keepdims=True)
    return [
        Scenario(
            name=f"S{index + 1}",
            multipliers=dict(zip(names, map(float, raw[index]))),
            probability=1.0 / n_scenarios,
        )
        for index in range(n_scenarios)
    ]


def scaled_instance(inst: Instance, multipliers: dict[str, float]) -> Instance:
    """A copy of the instance with zone demand multiplied by the scenario."""
    families = [
        Family(
            name=fam.name,
            unit_value=fam.unit_value,
            cv=fam.cv,
            weekly_demand={
                zone: demand * multipliers.get(zone, 1.0)
                for zone, demand in fam.weekly_demand.items()
            },
        )
        for fam in inst.families
    ]
    return Instance(
        central=inst.central,
        dcs=inst.dcs,
        zones=inst.zones,
        families=families,
        params=inst.params,
    )


def _overflow_units(inst: Instance, assignment: dict[str, str], multipliers: dict, open_set: set) -> float:
    """Annual units above nameplate capacity, given a fixed assignment."""
    overflow = 0.0
    for facility in inst.dcs:
        if facility.name not in open_set:
            continue
        throughput = sum(
            52.0 * inst.demand(zone.name, family.name) * multipliers.get(zone.name, 1.0)
            for zone in inst.zones
            if assignment.get(zone.name) == facility.name
            for family in inst.families
        )
        overflow += max(0.0, throughput - facility.capacity)
    return overflow


def evaluate_policy(
    inst: Instance,
    assignment: dict[str, str],
    open_dcs: list[str],
    scenarios: list[Scenario],
    service_level: float,
    overflow_cost: float = 5.0,
) -> dict:
    """Expected and worst-case cost of a fixed first-stage policy.

    Capacity is soft: units above nameplate capacity pay ``overflow_cost``
    (overtime / emergency handling).  This keeps the deterministic and
    stochastic policies on the same cost function, which is what VSS requires.
    """
    open_set = set(open_dcs)
    if any(facility == model.DIRECT_NAME for facility in assignment.values()):
        open_set.add(model.DIRECT_NAME)

    costs = []
    for scenario in scenarios:
        terms = model.cost_terms(
            scaled_instance(inst, scenario.multipliers),
            assignment,
            open_set,
            service_level,
        )
        overflow = _overflow_units(inst, assignment, scenario.multipliers, open_set)
        costs.append(terms["total"] + overflow_cost * overflow)
    probabilities = np.array([scenario.probability for scenario in scenarios])
    return {
        "scenario_costs": costs,
        "expected": float(np.dot(probabilities, costs)),
        "worst": float(max(costs)),
    }


def build_stochastic_model(
    inst: Instance,
    scenarios: list[Scenario],
    service_level: float | None = None,
    robust: bool = False,
    overflow_cost: float = 5.0,
):
    """Scenario-based MILP.  First stage: y, x.  Second stage: safety stock."""
    p = inst.params
    service_level = service_level if service_level is not None else p["service_level"]
    z = stats.norm.ppf(service_level)
    _, zone_dc = distance_matrix(inst)
    facilities = model.facilities(inst)
    review = p["review_weeks"]

    problem = pulp.LpProblem("stochastic_network", pulp.LpMinimize)
    y = {
        facility.name: pulp.LpVariable(f"open_{i}", 0, 1, cat="Binary")
        for i, facility in enumerate(facilities)
    }
    x = {
        (zone.name, facility.name): pulp.LpVariable(
            f"assign_{i}_{j}", 0, 1, cat="Binary"
        )
        for i, zone in enumerate(inst.zones)
        for j, facility in enumerate(facilities)
    }
    ss = {
        (facility.name, family.name, scenario.name): pulp.LpVariable(
            f"ss_{i}_{j}_{k}", 0, None
        )
        for i, facility in enumerate(facilities)
        for j, family in enumerate(inst.families)
        for k, scenario in enumerate(scenarios)
    }
    overflow = {
        (facility.name, scenario.name): pulp.LpVariable(f"overflow_{i}_{k}", 0, None)
        for i, facility in enumerate(facilities)
        if facility.name != model.DIRECT_NAME
        for k, scenario in enumerate(scenarios)
    }

    for zone in inst.zones:
        problem += (
            pulp.lpSum(x[(zone.name, facility.name)] for facility in facilities) == 1,
            f"serve_{zone.name}",
        )
    for facility in facilities:
        for zone in inst.zones:
            problem += (
                x[(zone.name, facility.name)] <= y[facility.name],
                f"link_{zone.name}_{facility.name}",
            )

    for scenario in scenarios:
        mult = scenario.multipliers
        for facility in facilities:
            if facility.name == model.DIRECT_NAME:
                continue
            throughput = pulp.lpSum(
                52.0
                * inst.demand(zone.name, family.name)
                * mult.get(zone.name, 1.0)
                * x[(zone.name, facility.name)]
                for zone in inst.zones
                for family in inst.families
            )
            problem += (
                overflow[(facility.name, scenario.name)]
                >= throughput - facility.capacity * y[facility.name],
                f"capacity_{facility.name}_{scenario.name}",
            )

        for facility in facilities:
            lead_time = model.facility_economics(inst, facility)["lt"]
            for family in inst.families:
                served = pulp.lpSum(
                    inst.demand(zone.name, family.name)
                    * mult.get(zone.name, 1.0)
                    * x[(zone.name, facility.name)]
                    for zone in inst.zones
                )
                problem += (
                    ss[(facility.name, family.name, scenario.name)]
                    >= z * family.cv * (lead_time**0.5) * served,
                    f"ss_{facility.name}_{family.name}_{scenario.name}",
                )

    fixed = pulp.lpSum(
        facility.fixed_cost * y[facility.name] for facility in facilities
    )
    scenario_costs = []
    for scenario in scenarios:
        mult = scenario.multipliers
        inbound = pulp.lpSum(
            p["inbound_rate"]
            * model.facility_economics(inst, facility)["inbound_km"]
            / 1000.0
            * 52.0
            * inst.demand(zone.name, family.name)
            * mult.get(zone.name, 1.0)
            * x[(zone.name, facility.name)]
            for facility in facilities
            if facility.name != model.DIRECT_NAME
            for zone in inst.zones
            for family in inst.families
        )
        outbound = pulp.lpSum(
            model._leg_rate(inst, facility)
            * model._leg_distance(inst, zone, facility, zone_dc)
            / 1000.0
            * 52.0
            * inst.zone_weekly_total(zone.name)
            * mult.get(zone.name, 1.0)
            * x[(zone.name, facility.name)]
            for zone in inst.zones
            for facility in facilities
        )
        holding = pulp.lpSum(
            p["holding_rate"]
            * family.unit_value
            * (
                (review / 2.0)
                * pulp.lpSum(
                    inst.demand(zone.name, family.name)
                    * mult.get(zone.name, 1.0)
                    * x[(zone.name, facility.name)]
                    for zone in inst.zones
                )
                + ss[(facility.name, family.name, scenario.name)]
            )
            for facility in facilities
            for family in inst.families
        )
        overflow_penalty = pulp.lpSum(
            overflow[(facility.name, scenario.name)] for facility in facilities
            if facility.name != model.DIRECT_NAME
        )
        scenario_costs.append(inbound + outbound + holding + overflow_cost * overflow_penalty)

    if robust:
        worst = pulp.LpVariable("worst_scenario_cost", 0, None)
        for cost in scenario_costs:
            problem += worst >= cost
        problem += fixed + worst
    else:
        problem += fixed + pulp.lpSum(
            scenario.probability * cost
            for scenario, cost in zip(scenarios, scenario_costs)  # noqa: B905
        )

    return problem, {"y": y, "x": x, "ss": ss}, scenario_costs


def _extract(problem, variables, inst, scenarios, service_level, robust, overflow_cost=5.0) -> StochasticSolution:
    status = pulp.LpStatus[problem.status]
    if problem.objective.value() is None:  # pragma: no cover
        return StochasticSolution(status=status, objective=None, open_dcs=[], assignment={})

    open_dcs = [
        name for name, variable in variables["y"].items() if (variable.value() or 0) > 0.5
    ]
    assignment = {}
    for zone in inst.zones:
        options = [
            (facility.name, variables["x"][(zone.name, facility.name)].value() or 0.0)
            for facility in model.facilities(inst)
        ]
        assignment[zone.name] = max(options, key=lambda pair: pair[1])[0]

    evaluated = evaluate_policy(inst, assignment, open_dcs, scenarios, service_level, overflow_cost)
    return StochasticSolution(
        status=status,
        objective=float(problem.objective.value()),
        open_dcs=open_dcs,
        assignment=assignment,
        expected_cost=evaluated["expected"],
        worst_cost=evaluated["worst"],
        scenario_costs=evaluated["scenario_costs"],
        is_robust=robust,
    )


def solve_stochastic(
    inst: Instance,
    scenarios: list[Scenario],
    service_level: float | None = None,
    robust: bool = False,
    time_limit: int = 120,
    overflow_cost: float = 5.0,
) -> StochasticSolution:
    service_level = (
        service_level if service_level is not None else inst.params["service_level"]
    )
    problem, variables, _ = build_stochastic_model(
        inst, scenarios, service_level, robust=robust, overflow_cost=overflow_cost
    )
    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    return _extract(problem, variables, inst, scenarios, service_level, robust, overflow_cost)


def analyze(
    inst: Instance,
    scenarios: list[Scenario],
    service_level: float | None = None,
    time_limit: int = 120,
    overflow_cost: float = 5.0,
) -> dict:
    """Run deterministic / stochastic / robust / wait-and-see and compute VSS, EVPI."""
    service_level = (
        service_level if service_level is not None else inst.params["service_level"]
    )
    deterministic = model.solve(inst, service_level=service_level, time_limit=time_limit)
    deterministic_eval = evaluate_policy(
        inst, deterministic.assignment, deterministic.open_dcs, scenarios,
        service_level, overflow_cost,
    )

    stochastic = solve_stochastic(
        inst, scenarios, service_level, robust=False,
        time_limit=time_limit, overflow_cost=overflow_cost,
    )
    robust = solve_stochastic(
        inst, scenarios, service_level, robust=True,
        time_limit=time_limit, overflow_cost=overflow_cost,
    )

    # Wait-and-see: one first-stage network per scenario, soft capacity included.
    wait_and_see = []
    for scenario in scenarios:
        single = Scenario(name=scenario.name, multipliers=scenario.multipliers, probability=1.0)
        solution = solve_stochastic(
            inst, [single], service_level, robust=False,
            time_limit=time_limit, overflow_cost=overflow_cost,
        )
        wait_and_see.append(float(solution.objective))
    probabilities = np.array([scenario.probability for scenario in scenarios])
    wait_and_see_expected = float(np.dot(probabilities, wait_and_see))

    vss = float(deterministic_eval["expected"] - stochastic.expected_cost)
    evpi = float(stochastic.expected_cost - wait_and_see_expected)
    return {
        "deterministic": deterministic,
        "deterministic_eval": deterministic_eval,
        "stochastic": stochastic,
        "robust": robust,
        "wait_and_see": wait_and_see,
        "wait_and_see_expected": wait_and_see_expected,
        "vss": vss,
        "evpi": evpi,
        "service_level": service_level,
    }
