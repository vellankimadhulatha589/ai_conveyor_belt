"""Physical layout of the belt: where the sensor suites are."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from .. import config


@dataclass(frozen=True)
class SensorPoint:
    point_id: str
    name: str
    km: float
    kind: str          # "drive" (head station), "idler" or "tail"
    elevation_m: float
    curve: bool        # sits in a horizontal curve
    channels: tuple

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d


def elevation_at(km: float) -> float:
    """Synthetic terrain: a long incline with a bump and a steep section."""
    return 25.0 * km + 55.0 * math.sin(2 * math.pi * km / 5.0) + 18.0 * math.sin(2 * math.pi * km / 1.3)


def is_curve(km: float) -> bool:
    return any(abs(km - c) <= config.CURVE_RADIUS_KM for c in config.CURVE_KM)


def build_points() -> list[SensorPoint]:
    n = int(config.ROUTE_LENGTH_M / config.POINT_SPACING_M) + 1
    points = []
    for i in range(n):
        km = round(i * config.POINT_SPACING_M / 1000.0, 3)
        if i == 0:
            kind, name = "drive", "Head drive station"
            channels = tuple(config.IDLER_CHANNELS + config.DRIVE_ONLY_CHANNELS)
        elif i == n - 1:
            kind, name = "tail", "Tail pulley"
            channels = tuple(config.IDLER_CHANNELS)
        else:
            kind, name = "idler", f"Idler station km {km:.1f}"
            channels = tuple(config.IDLER_CHANNELS)
        points.append(
            SensorPoint(
                point_id=f"P{i:02d}",
                name=name,
                km=km,
                kind=kind,
                elevation_m=round(elevation_at(km), 1),
                curve=is_curve(km),
                channels=channels,
            )
        )
    return points


POINTS = build_points()
POINT_INDEX = {p.point_id: i for i, p in enumerate(POINTS)}
POINT_BY_ID = {p.point_id: p for p in POINTS}
DRIVE_ID = POINTS[0].point_id
