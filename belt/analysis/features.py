"""Feature engineering for the failure-prediction model.

The model never sees raw sensor values. It sees **how far each channel is from its
context-aware expectation, in the physically 'bad' direction**, summarised over several
look-back windows (6 h / 24 h / 72 h): mean, max and trend. That makes one model usable for
every sensor point (a bearing at km 0.6 and one at km 4.4 look alike once load, ambient
temperature and per-point offsets are removed).

The feature history of a component is cut at its last repair (maintenance reset), otherwise
the degradation of the *old* part would haunt the *new* one for days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..simulator.layout import POINTS, DRIVE_ID

CH_FEATURES = ["vibration", "temperature", "lateral_offset", "motor_current", "voltage", "slip", "belt_speed"]
# +1: a positive residual is bad, -1: a negative residual is bad, 0: either direction is bad
BAD_SIGN = {"vibration": 1, "temperature": 1, "lateral_offset": 0, "motor_current": 1,
            "voltage": -1, "slip": 1, "belt_speed": -1}
WINDOWS = config.FEATURE_WINDOWS_H

FEATURE_NAMES = (
    [f"{ch}_{stat}{w}" for ch in CH_FEATURES for w in WINDOWS for stat in ("mean", "max")]
    + [f"{ch}_trend{t}" for ch in CH_FEATURES for t in (24, 72)]
    + ["is_drive"]
)


def hourly_badness(z: dict) -> dict:
    """Residual z-scores -> 'badness' (positive = worse) averaged per hour (NaN while stopped)."""
    out = {}
    for ch in CH_FEATURES:
        if ch not in z:
            continue
        zz = z[ch].clip(-30.0, 30.0)
        sign = BAD_SIGN[ch]
        bad = zz.abs() if sign == 0 else zz * sign
        out[ch] = bad.resample("1h").mean()
    return out


def _vals(a: np.ndarray | None, lo: int, hi: int) -> np.ndarray:
    if a is None or hi <= lo:
        return np.empty(0)
    seg = a[max(lo, 0):hi]
    return seg[np.isfinite(seg)]


def _point_features(arrs: dict, e: int, r0: int, is_drive: bool) -> dict:
    row: dict = {}
    for ch in CH_FEATURES:
        a = arrs.get(ch)
        for w in WINDOWS:
            s = _vals(a, max(e - w + 1, r0), e + 1)
            row[f"{ch}_mean{w}"] = float(s.mean()) if len(s) else np.nan
            row[f"{ch}_max{w}"] = float(s.max()) if len(s) else np.nan
        for t in (24, 72):
            half = t // 2
            recent = _vals(a, max(e - half + 1, r0), e + 1)
            prior = _vals(a, max(e - 2 * half + 1, r0), max(e - half + 1, r0))
            row[f"{ch}_trend{t}"] = float(recent.mean() - prior.mean()) if len(recent) and len(prior) else np.nan
    row["is_drive"] = 1.0 if is_drive else 0.0
    return row


def build_features(hourly: dict, snapshots: pd.DatetimeIndex, resets: dict | None = None) -> pd.DataFrame:
    """One feature row per (sensor point, snapshot time).

    ``resets`` maps point_id -> list of timestamps of completed repairs (history before the last
    repair at or before the snapshot time is ignored)."""
    resets = resets or {}
    any_ch = next(iter(hourly.values()))
    hidx = any_ch.index
    rows = []
    for p in POINTS:
        arrs = {ch: hourly[ch][p.point_id].to_numpy(dtype=float) for ch in hourly
                if np.isfinite(hourly[ch][p.point_id].to_numpy(dtype=float)).any()}
        rs = sorted(pd.Timestamp(t) for t in resets.get(p.point_id, []))
        for t in snapshots:
            e = int(hidx.searchsorted(t, side="right")) - 1
            if e < 0:
                continue
            past = [r for r in rs if r <= t]
            r0 = int(hidx.searchsorted(past[-1], side="left")) if past else 0
            row = _point_features(arrs, e, r0, p.point_id == DRIVE_ID)
            row["point_id"] = p.point_id
            row["ts"] = t
            rows.append(row)
    df = pd.DataFrame(rows)
    return df[["point_id", "ts"] + FEATURE_NAMES]


def label_snapshots(feat: pd.DataFrame, failures: list[dict]) -> pd.DataFrame:
    """Add ``y`` (class index), ``tth`` (hours to failure, NaN if none), ``ttf_next`` and ``valid``.

    A snapshot is positive when a failure of mode M at that point happens within
    ``PREDICTION_WINDOW_H[M]`` hours (the period in which its degradation signature exists).
    Snapshots during the repair downtime itself are invalid (nothing meaningful to predict)."""
    by_point: dict = {}
    for f in failures:
        by_point.setdefault(f["point_id"], []).append(
            (pd.Timestamp(f["failure_time"]), pd.Timestamp(f["repaired_ts"]), f.get("mode")))
    y = np.zeros(len(feat), dtype=int)
    tth = np.full(len(feat), np.nan)
    ttf_next = np.full(len(feat), np.nan)   # hours to the next failure at this point, whatever the window
    valid = np.ones(len(feat), dtype=bool)
    ts = feat["ts"].to_numpy()
    pids = feat["point_id"].to_numpy()
    for i in range(len(feat)):
        t = pd.Timestamp(ts[i])
        for ft, rep, mode in sorted(by_point.get(pids[i], []), key=lambda x: x[0]):
            if ft <= t < rep:
                valid[i] = False
                break
            if ft > t:
                hrs = (ft - t) / pd.Timedelta(hours=1)
                ttf_next[i] = hrs
                if mode in config.PREDICTION_WINDOW_H and hrs <= config.PREDICTION_WINDOW_H[mode]:
                    y[i] = config.CLASSES.index(mode)
                    tth[i] = hrs
                break
    out = feat.copy()
    out["y"], out["tth"], out["ttf_next"], out["valid"] = y, tth, ttf_next, valid
    return out
