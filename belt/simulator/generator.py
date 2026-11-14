"""The sensor simulator: the *only* data source of the sensor pipeline.

``BeltSimulator.generate(timestamps)`` returns one row per (sensor point, timestamp) with
the channels each sensor suite measures. It models

* a healthy baseline that depends on context (belt load, ambient temperature, time of day,
  an 18 h/day duty cycle),
* equipment faults with failure-specific degradation curves (see ``faults.py`` and
  ``docs/DESIGN.md`` section 4),
* messy data: missing readings, outages, stuck sensors, spikes, impossible values,
  slow calibration drift, duplicated rows.

The output is deterministic for a given (seed, start, history_days), which lets the live
streamer continue exactly where the history ended.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .faults import Fault, SensorFault, build_schedule, build_sensor_faults
from .layout import POINTS, POINT_INDEX

TWO_PI = 2.0 * math.pi
IMPOSSIBLE = {
    "vibration": -1.0, "temperature": -99.0, "lateral_offset": 999.0, "ambient_temp": 99.0,
    "belt_speed": -5.0, "drive_speed": -5.0, "belt_load": -100.0, "motor_current": -50.0, "voltage": 0.0,
}

DEFAULT_START = datetime(2026, 1, 1)


class BeltSimulator:
    def __init__(self, seed: int = 7, start: datetime = DEFAULT_START, history_days: int = 90,
                 extra_days: int = 40):
        self.seed = seed
        self.start = start
        self.history_days = history_days
        self.extra_days = extra_days
        self.demo_now = start + timedelta(days=history_days)
        self.end = self.demo_now + timedelta(days=extra_days)
        rng = np.random.default_rng(seed)
        n = len(POINTS)
        self.vib_bias = rng.normal(0.0, 0.15, n)
        self.temp_bias = rng.normal(0.0, 1.0, n)
        self.off_bias = rng.normal(0.0, 3.0, n)
        self.elev = np.array([p.elevation_m for p in POINTS])
        self.curve = np.array([1.0 if p.curve else 0.0 for p in POINTS])
        self.faults: list[Fault] = build_schedule(rng, start, self.end, self.demo_now)
        self.sensor_faults: list[SensorFault] = build_sensor_faults(rng, start, self.end)

    # ------------------------------------------------------------------ config persistence
    def config_dict(self) -> dict:
        return {"seed": self.seed, "start": self.start.isoformat(), "history_days": self.history_days,
                "extra_days": self.extra_days}

    def save_config(self, path: Path = config.SIM_CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.config_dict(), indent=2))

    @classmethod
    def from_config(cls, path: Path = config.SIM_CONFIG_PATH) -> "BeltSimulator":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        return cls(seed=d["seed"], start=datetime.fromisoformat(d["start"]), history_days=d["history_days"],
                   extra_days=d.get("extra_days", 40))

    def save_ground_truth(self, path: Path = config.GROUND_TRUTH_PATH) -> None:
        """Answer key used ONLY by evaluation scripts. The backend never reads it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        gt = {"sim": self.config_dict(), "demo_now": self.demo_now.isoformat(),
              "faults": [f.to_dict() for f in self.faults],
              "sensor_faults": [s.to_dict() for s in self.sensor_faults]}
        path.write_text(json.dumps(gt, indent=2))

    # ------------------------------------------------------------------ helpers
    def _h(self, t: datetime) -> float:
        return (t - self.start).total_seconds() / 3600.0

    def timestamps(self, t0: datetime, t1: datetime) -> pd.DatetimeIndex:
        return pd.date_range(t0, t1, freq=f"{config.SAMPLE_MINUTES}min", inclusive="left")

    def maintenance_between(self, t0: datetime, t1: datetime) -> list[dict]:
        """Work-order records (what a CMMS would hold): failure repaired in (t0, t1]."""
        out = []
        for f in self.faults:
            if t0 < f.repair_end <= t1:
                out.append({"point_id": f.point_id, "mode": f.mode,
                            "failure_time": f.failure_time.isoformat(timespec="seconds"),
                            "repaired_ts": f.repair_end.isoformat(timespec="seconds")})
        return out

    # ------------------------------------------------------------------ generation
    def generate(self, ts: pd.DatetimeIndex, messy: bool = True) -> pd.DataFrame:
        n_t, n_p = len(ts), len(POINTS)
        cols = ["point_id", "ts"] + config.ALL_CHANNELS
        if n_t == 0:
            return pd.DataFrame(columns=cols)
        rng = np.random.default_rng([self.seed, int(ts[0].value // 10 ** 9) % (2 ** 31)])
        hours = np.asarray((ts - pd.Timestamp(self.start)) / pd.Timedelta(hours=1), dtype=float)
        hod = np.asarray(ts.hour, dtype=float) + np.asarray(ts.minute, dtype=float) / 60.0

        # ---- operating state: 18 h/day duty cycle (stopped 04:00-10:00) + trips after failures
        sched_stop = (hod >= 4.0) & (hod < 10.0)
        down = np.zeros(n_t, dtype=bool)
        for f in self.faults:
            fh = self._h(f.failure_time)
            down |= (hours >= fh) & (hours < fh + f.repair_hours)
        running = ~(sched_stop | down)
        r = running.astype(float)

        # ---- context
        ambient_base = 18.0 + 10.0 * np.sin(TWO_PI * (hod - 9.0) / 24.0) + 3.0 * np.sin(TWO_PI * hours / (24 * 7) + 1.0)
        ambient = ambient_base[:, None] - 0.006 * self.elev[None, :] + rng.normal(0, 0.3, (n_t, n_p))
        load = np.clip(
            3000.0 + 500.0 * np.sin(TWO_PI * hours / 3.7 + 0.5) + 300.0 * np.sin(TWO_PI * hours / 1.3 + 2.0)
            + rng.normal(0, 120.0, n_t),
            800.0, None,
        ) * r
        imbalance = np.sin(TWO_PI * hours / 9.1)
        R = r[:, None]
        L = load[:, None]

        # ---- healthy baselines
        vib = R * (1.8 + 0.0003 * L + self.vib_bias[None, :] + rng.normal(0, 0.2, (n_t, n_p))) \
            + (1 - R) * (0.15 + np.abs(rng.normal(0, 0.03, (n_t, n_p))))
        temp = R * (ambient + 14.0 + 0.0025 * L + self.temp_bias[None, :] + rng.normal(0, 0.6, (n_t, n_p))) \
            + (1 - R) * (ambient + 2.0 + rng.normal(0, 0.3, (n_t, n_p)))
        phase = np.cos(np.arange(n_p) * 0.7)[None, :]
        off = R * (rng.normal(0, 1.0, (n_t, n_p)) * (10.0 + 6.0 * self.curve[None, :])
                   + 12.0 * imbalance[:, None] * phase + self.off_bias[None, :]) \
            + (1 - R) * rng.normal(0, 2.0, (n_t, n_p))

        drive_speed = r * (config.NOMINAL_SPEED_MS + rng.normal(0, 0.02, n_t))
        slip = 0.003 + np.abs(rng.normal(0, 0.001, n_t))
        current = r * (80.0 + 0.075 * load) * (1.0 + rng.normal(0, 0.015, n_t))
        voltage = 690.0 + rng.normal(0, 4.0, n_t)
        load_meas = load + r * rng.normal(0, 25.0, n_t)

        cur_mult = np.ones(n_t)
        speed_mult = np.ones(n_t)
        volt_mult = np.ones(n_t)
        slip_extra = np.zeros(n_t)

        # ---- equipment faults
        for f in self.faults:
            tau = self._h(f.failure_time) - hours  # hours until failure
            act = (tau >= 0.0) & (tau <= f.lead_hours) & running
            if not act.any():
                continue
            i = POINT_INDEX[f.point_id]
            s = np.clip(1.0 - tau / f.lead_hours, 0.0, 1.0)
            if f.mode == "bearing":
                # vibration rises first (exponentially), temperature much later (late indicator)
                g = np.exp(-tau / (0.2 * f.lead_hours))
                vib[:, i] += act * (12.0 * g + rng.normal(0, 0.6, n_t) * g)
                lt = 0.4 * f.lead_hours
                st = np.clip(1.0 - tau / lt, 0.0, 1.0)
                actt = act & (tau <= lt)
                temp[:, i] += actt * (52.0 * st ** 2 + rng.normal(0, 1.2, n_t) * st)
                cur_mult *= 1.0 + 0.02 * st ** 2 * actt
            elif f.mode == "mistrack":
                off[:, i] += act * f.direction * 160.0 * s ** 1.5
                vib[:, i] += act * 1.2 * s
            elif f.mode == "slip":
                thr = 3300.0 - 2200.0 * s          # slips only under peak load at first
                extra = np.where(act & (load > thr), 0.07 * s, 0.0)
                slip_extra += extra
                temp[:, 0] += act * 30.0 * s * (0.3 + 0.7 * (extra > 0))
                vib[:, 0] += act * 0.5 * s
                cur_mult *= 1.0 + 0.03 * s * act
            elif f.mode == "overload":
                cur_mult *= 1.0 + 0.45 * s * act
                speed_mult *= 1.0 - 0.10 * s * act
                temp[:, 0] += act * 28.0 * s
            elif f.mode == "undervoltage":
                depth = act * 0.14 * s * (0.4 + 0.6 * (load > 2800.0))
                volt_mult *= 1.0 - depth
                cur_mult *= 1.0 + 0.5 * depth
                speed_mult *= 1.0 - 0.3 * depth
                temp[:, 0] += 100.0 * depth

        drive_speed_f = drive_speed * speed_mult
        belt_speed = drive_speed_f * (1.0 - slip - slip_extra) + r * rng.normal(0, 0.01, n_t)
        arrays = {
            "vibration": np.clip(vib, 0.0, None),
            "temperature": temp,
            "lateral_offset": off,
            "ambient_temp": ambient,
        }

        def drive_only(v: np.ndarray) -> np.ndarray:
            a = np.full((n_t, n_p), np.nan)
            a[:, 0] = v
            return a

        arrays["belt_speed"] = drive_only(np.clip(belt_speed, 0.0, None))
        arrays["drive_speed"] = drive_only(drive_speed_f)
        arrays["belt_load"] = drive_only(load_meas)
        arrays["motor_current"] = drive_only(current * cur_mult)
        arrays["voltage"] = drive_only(voltage * volt_mult)

        present = np.ones((n_t, n_p), dtype=bool)
        if messy:
            self._apply_messy(arrays, present, hours, rng)

        ids = np.array([p.point_id for p in POINTS])
        data = {"point_id": np.tile(ids, n_t), "ts": np.repeat(ts.values, n_p)}
        for ch in config.ALL_CHANNELS:
            data[ch] = arrays[ch].reshape(-1)
        df = pd.DataFrame(data)
        df = df[present.reshape(-1)]
        if messy and len(df):
            dup = rng.random(len(df)) < 0.003  # duplicated messages
            if dup.any():
                df = pd.concat([df, df[dup]], ignore_index=True)
        return df.reset_index(drop=True)

    def _apply_messy(self, arrays: dict, present: np.ndarray, hours: np.ndarray, rng: np.random.Generator) -> None:
        for sf in self.sensor_faults:
            i = POINT_INDEX[sf.point_id]
            sh, eh = self._h(sf.start), self._h(sf.end)
            mask = (hours >= sh) & (hours < eh)
            if not mask.any():
                continue
            if sf.kind == "outage":
                present[mask, i] = False
            elif sf.kind == "stuck":
                arr = arrays[sf.channel]
                valid = ~np.isnan(arr[:, i])
                arr[mask & valid, i] = sf.value
            elif sf.kind == "drift":
                arr = arrays[sf.channel]
                arr[:, i] += sf.value * np.clip((hours - sh) / 24.0, 0.0, None)
        for ch, arr in arrays.items():
            valid = ~np.isnan(arr)
            # spikes
            m = (rng.random(arr.shape) < 0.0008) & valid
            k = int(m.sum())
            if k:
                if ch == "lateral_offset":
                    arr[m] = arr[m] + rng.choice([-1.0, 1.0], k) * rng.uniform(60, 150, k)
                else:
                    arr[m] = arr[m] * rng.uniform(4.0, 9.0, k)
            # impossible values
            m = (rng.random(arr.shape) < 0.0004) & valid
            if m.any():
                arr[m] = IMPOSSIBLE[ch]
            # isolated missing readings
            m = (rng.random(arr.shape) < 0.012) & valid
            arr[m] = np.nan
