"""算例加载与地理距离。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE = ROOT / "data" / "instance.json"


@dataclass(frozen=True)
class Node:
    name: str
    lat: float
    lon: float
    fixed_cost: float = 0.0
    capacity: float = 0.0


@dataclass(frozen=True)
class Family:
    name: str
    unit_value: float
    cv: float
    weekly_demand: dict[str, float]


@dataclass
class Instance:
    central: Node
    dcs: list[Node]
    zones: list[Node]
    families: list[Family]
    params: dict

    # ---- 便捷访问
    @property
    def zone_names(self) -> list[str]:
        return [z.name for z in self.zones]

    @property
    def dc_names(self) -> list[str]:
        return [d.name for d in self.dcs]

    def demand(self, zone: str, family: str) -> float:
        fam = next(f for f in self.families if f.name == family)
        return fam.weekly_demand.get(zone, 0.0)

    def zone_weekly_total(self, zone: str) -> float:
        return sum(f.weekly_demand.get(zone, 0.0) for f in self.families)


def haversine_km(a: Node, b: Node) -> float:
    """大圆距离（km）——用来近似运输距离。"""
    r = 6371.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dphi = p2 - p1
    dlambda = math.radians(b.lon - a.lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def load_instance(path: str | Path = DEFAULT_INSTANCE) -> Instance:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    central = Node(**raw["central"])
    dcs = [Node(**d) for d in raw["dcs"]]
    zones = [Node(**z) for z in raw["zones"]]
    families = [Family(**f) for f in raw["families"]]
    return Instance(central=central, dcs=dcs, zones=zones, families=families, params=raw["params"])


def distance_matrix(inst: Instance) -> tuple[dict, dict]:
    """((central, dc) 距离, (zone, dc) 距离)"""
    from_central = {d.name: haversine_km(inst.central, d) for d in inst.dcs}
    zone_dc = {
        (z.name, d.name): haversine_km(z, d) for z in inst.zones for d in inst.dcs
    }
    return from_central, zone_dc
