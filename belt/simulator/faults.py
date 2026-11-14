"""Fault schedules for the simulator.

Two kinds of events are scheduled:

* **Equipment faults** (bearing, mistrack, slip, overload, undervoltage): a gradual
  degradation that starts ``lead_hours`` before a failure that trips the belt.
* **Sensor faults** (outage, stuck, drift): the *data* is wrong although the belt is fine.
  The backend has to tell these apart from real equipment faults.

Everything is derived from a seeded RNG so that the history generator and the live
generator reproduce exactly the same schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import numpy as np

from .layout import POINTS, DRIVE_ID

# rate = expected faults per day (whole belt), lead = hours of visible degradation,
# repair = hours the belt stays down after the failure trips it.
MODE_PARAMS = {
    "bearing": dict(rate=0.13, lead=240.0, repair=6.0),
    "mistrack": dict(rate=0.07, lead=72.0, repair=4.0),
    "slip": dict(rate=0.02, lead=96.0, repair=8.0),
    "overload": dict(rate=0.035, lead=8.0, repair=3.0),
    "undervoltage": dict(rate=0.03, lead=120.0, repair=5.0),
}
DRIVE_MODES = {"slip", "overload", "undervoltage"}

# typical healthy value of every channel, used as the value of a stuck sensor
TYPICAL = {
    "vibration": 2.4, "temperature": 40.0, "lateral_offset": 3.0, "ambient_temp": 18.0,
    "belt_speed": 3.99, "drive_speed": 4.0, "belt_load": 3000.0, "motor_current": 300.0, "voltage": 690.0,
}


@dataclass
class Fault:
    fault_id: str
    mode: str
    point_id: str
    failure_time: datetime
    lead_hours: float
    repair_hours: float
    direction: int = 1  # sign of the lateral drift for mistracking

    @property
    def onset(self) -> datetime:
        return self.failure_time - timedelta(hours=self.lead_hours)

    @property
    def repair_end(self) -> datetime:
        return self.failure_time + timedelta(hours=self.repair_hours)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failure_time"] = self.failure_time.isoformat()
        d["onset"] = self.onset.isoformat()
        d["repair_end"] = self.repair_end.isoformat()
        return d


@dataclass
class SensorFault:
    kind: str                  # "outage" | "stuck" | "drift"
    point_id: str
    channel: str | None
    start: datetime
    end: datetime
    value: float | None = None  # stuck value, or drift in units/day

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d


def _conflicts(cand: Fault, existing: list[Fault], margin_h: float = 12.0) -> bool:
    for f in existing:
        if f.point_id != cand.point_id:
            continue
        a0, a1 = cand.onset - timedelta(hours=margin_h), cand.repair_end + timedelta(hours=margin_h)
        if a0 < f.repair_end and f.onset < a1:
            return True
    return False


def _choose_point(rng: np.random.Generator, mode: str) -> str:
    if mode in DRIVE_MODES:
        return DRIVE_ID
    candidates = [p for p in POINTS if p.kind != "drive"]
    if mode == "mistrack":
        w = np.array([3.0 if p.curve else 1.0 for p in candidates])
    else:
        w = np.ones(len(candidates))
    return candidates[int(rng.choice(len(candidates), p=w / w.sum()))].point_id


def demo_pins(demo_now: datetime) -> list[Fault]:
    """Faults that are 'in progress' when the live demo starts, so the dashboard has
    something to show immediately."""
    h = lambda x: demo_now + timedelta(hours=x)  # noqa: E731
    return [
        Fault("DEMO-1", "bearing", "P12", h(60), 240.0, 6.0),
        Fault("DEMO-2", "bearing", "P05", h(130), 250.0, 6.0),
        Fault("DEMO-3", "mistrack", "P16", h(30), 72.0, 4.0, direction=1),
        Fault("DEMO-4", "slip", DRIVE_ID, h(150), 96.0, 8.0),
    ]


def build_schedule(rng: np.random.Generator, start: datetime, end: datetime, demo_now: datetime) -> list[Fault]:
    days = (end - start).total_seconds() / 86400.0
    pins = demo_pins(demo_now)
    faults: list[Fault] = []
    n_id = 0
    for mode, p in MODE_PARAMS.items():
        n = int(rng.poisson(p["rate"] * days))
        for _ in range(n):
            for _attempt in range(40):
                lead = p["lead"] * float(rng.uniform(0.7, 1.3))
                ft = start + timedelta(hours=float(rng.uniform(lead + 1.0, days * 24.0)))
                cand = Fault(
                    fault_id=f"F{n_id:03d}",
                    mode=mode,
                    point_id=_choose_point(rng, mode),
                    failure_time=ft,
                    lead_hours=lead,
                    repair_hours=p["repair"] * float(rng.uniform(0.8, 1.4)),
                    direction=int(rng.choice([-1, 1])),
                )
                if _conflicts(cand, faults) or _conflicts(cand, pins):
                    continue
                faults.append(cand)
                n_id += 1
                break
    faults.extend(pins)
    faults.sort(key=lambda f: f.failure_time)
    return faults


def build_sensor_faults(rng: np.random.Generator, start: datetime, end: datetime) -> list[SensorFault]:
    days = (end - start).total_seconds() / 86400.0
    out: list[SensorFault] = []
    # outages: all channels of one point silent for a while (rows missing entirely)
    for _ in range(int(rng.poisson(0.25 * days))):
        p = POINTS[int(rng.integers(len(POINTS)))]
        s = start + timedelta(hours=float(rng.uniform(0, days * 24)))
        dur = float(rng.uniform(1, 6)) if rng.random() > 0.1 else float(rng.uniform(6, 12))
        out.append(SensorFault("outage", p.point_id, None, s, s + timedelta(hours=dur)))
    # stuck sensors: one channel repeats a constant value
    for _ in range(int(rng.poisson(0.12 * days))):
        p = POINTS[int(rng.integers(len(POINTS)))]
        ch = str(rng.choice([c for c in p.channels if c != "ambient_temp"]))
        s = start + timedelta(hours=float(rng.uniform(0, days * 24)))
        dur = float(rng.uniform(8, 30))
        out.append(
            SensorFault("stuck", p.point_id, ch, s, s + timedelta(hours=dur), value=round(TYPICAL[ch] * float(rng.uniform(0.85, 1.15)), 3))
        )
    # slow calibration drift of two temperature sensors (deliberately NOT handled, see README)
    for pid in rng.choice([p.point_id for p in POINTS if p.kind == "idler"], size=2, replace=False):
        out.append(SensorFault("drift", str(pid), "temperature", start, end, value=0.03))
    return out


def fault_from_dict(d: dict) -> Fault:  # pragma: no cover - convenience
    return Fault(d["fault_id"], d["mode"], d["point_id"], datetime.fromisoformat(d["failure_time"]),
                 d["lead_hours"], d["repair_hours"], d.get("direction", 1))
