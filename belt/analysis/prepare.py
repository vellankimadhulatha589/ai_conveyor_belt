"""Turn stored readings into analysis-ready, context-aware residuals.

Pipeline for one analysis window::

    raw rows -> wide tables (time x point, regular 5 min grid, flagged values -> NaN)
             -> running mask (is the belt moving?)
             -> expected value per channel from a *learned baseline* (depends on load, ambient)
             -> residual (observed - expected) and robust z-score, only while running
             -> stuck-sensor samples removed

A "high vibration" reading is only abnormal *given the current load*; comparing against the
learned expectation instead of a fixed number avoids false alarms at peak load.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..simulator.layout import POINTS, DRIVE_ID

POINT_IDS = [p.point_id for p in POINTS]
GRID = f"{config.SAMPLE_MINUTES}min"

# channel -> context features used to compute its expected value
CONTEXT_FEATURES = {
    "vibration": ["belt_load"],
    "temperature": ["belt_load", "ambient_temp"],
    "lateral_offset": [],
    "slip": [],
    "motor_current": ["belt_load"],
    "voltage": [],
    "belt_speed": [],
}
DRIVE_ONLY = {"slip", "motor_current", "voltage", "belt_speed"}


@dataclass
class Wide:
    idx: pd.DatetimeIndex
    ch: dict                      # channel -> DataFrame(time x point)
    running: np.ndarray           # bool per timestamp
    n_flagged: int = 0


@dataclass
class Analysis:
    wide: Wide
    delta: dict = field(default_factory=dict)      # channel -> DataFrame
    z: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)
    stuck: dict = field(default_factory=dict)      # (channel, point) -> bool np.ndarray


# ----------------------------------------------------------------------------- loading
def load_readings(conn: sqlite3.Connection, since: str | None = None, until: str | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM readings"
    cond, params = [], []
    if since:
        cond.append("ts >= ?")
        params.append(since)
    if until:
        cond.append("ts <= ?")
        params.append(until)
    if cond:
        sql += " WHERE " + " AND ".join(cond)
    sql += " ORDER BY ts"
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    frames = []
    while True:  # chunked: keeps peak memory low for ~1 M rows; plain tuples are robust to row_factory
        rows = cur.fetchmany(200_000)
        if not rows:
            break
        frames.append(pd.DataFrame.from_records([tuple(r) for r in rows], columns=cols))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    df["ts"] = pd.to_datetime(df["ts"])
    for ch in config.ALL_CHANNELS:
        df[ch] = pd.to_numeric(df[ch], errors="coerce")
    return df


def to_wide(df: pd.DataFrame) -> Wide | None:
    if df.empty:
        return None
    df = df.copy()
    n_flagged = 0
    flagged = df["flags"].fillna("") != ""
    if flagged.any():
        n_flagged = int(flagged.sum())
        sub = df.loc[flagged, "flags"]
        for ch in config.ALL_CHANNELS:
            m = sub.str.contains(f":{ch};", regex=False)
            if m.any():
                df.loc[m[m].index, ch] = np.nan  # impossible / non-numeric values are ignored
    ch: dict = {}
    for name in config.ALL_CHANNELS:
        w = df.pivot(index="ts", columns="point_id", values=name)
        w = w.reindex(columns=POINT_IDS).resample(GRID).mean()
        ch[name] = w
    idx = ch["vibration"].index
    for name in config.ALL_CHANNELS:
        ch[name] = ch[name].reindex(idx)
    # derived: slip = 1 - belt speed / drive speed (drive station only)
    slip = pd.DataFrame(np.nan, index=idx, columns=POINT_IDS)
    bs, ds = ch["belt_speed"][DRIVE_ID], ch["drive_speed"][DRIVE_ID]
    ok = (ds > 1.0) & bs.notna()
    slip.loc[ok, DRIVE_ID] = 1.0 - bs[ok] / ds[ok]
    ch["slip"] = slip
    return Wide(idx=idx, ch=ch, running=infer_running(ch), n_flagged=n_flagged)


def infer_running(ch: dict) -> np.ndarray:
    """Belt is running when the drive station reports speed. Uses the *lower* of the two speed
    readings (so one sensor stuck at a high value cannot fake a running belt) and falls back to
    the median vibration of all points if both speed sensors are missing."""
    spd = pd.concat([ch["belt_speed"][DRIVE_ID], ch["drive_speed"][DRIVE_ID]], axis=1).min(axis=1, skipna=True)
    fallback = (ch["vibration"].median(axis=1) > 1.0).to_numpy()
    run = (spd > config.RUNNING_SPEED_THRESHOLD).to_numpy()
    return np.where(spd.notna().to_numpy(), run, fallback).astype(bool)


# ----------------------------------------------------------------------------- baseline
def _context(wide: Wide) -> dict:
    load = wide.ch["belt_load"][DRIVE_ID].ffill(limit=3).to_numpy(dtype=float)
    ambient = wide.ch["ambient_temp"].ffill(limit=3)
    return {"belt_load": load, "ambient": ambient}


def _design(ctx: dict, feats: list[str], pid: str, n: int) -> np.ndarray:
    cols = []
    for f in feats:
        if f == "belt_load":
            cols.append(ctx["belt_load"])
        elif f == "ambient_temp":
            cols.append(ctx["ambient"][pid].to_numpy(dtype=float))
    if not cols:
        return np.zeros((n, 0))
    return np.column_stack(cols)


def _fit_line(X: np.ndarray, y: np.ndarray, iters: int = 3, k: float = 3.0):
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    if len(y) < 200:
        return None
    keep = np.ones(len(y), dtype=bool)
    beta = None
    sigma = 1.0
    for _ in range(iters):
        A = np.column_stack([np.ones(int(keep.sum())), X[keep]])
        beta, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        resid = y - np.column_stack([np.ones(len(y)), X]) @ beta
        med = np.median(resid[keep])
        sigma = max(1.4826 * float(np.median(np.abs(resid[keep] - med))), 1e-6)
        keep = np.abs(resid - med) < k * sigma
        if keep.sum() < 100:
            break
    return {"intercept": float(beta[0]), "coef": [float(c) for c in beta[1:]], "sigma": float(sigma)}


def fit_baseline(wide: Wide) -> dict:
    """Robust (iteratively trimmed) least squares per channel and sensor point, fitted on the
    samples taken while the belt was running. Faults in the history are a small minority and are
    trimmed away, so the baseline describes *healthy* behaviour."""
    ctx = _context(wide)
    n = len(wide.idx)
    out: dict = {}
    for chn, feats in CONTEXT_FEATURES.items():
        out[chn] = {}
        for pid in POINT_IDS:
            if chn in DRIVE_ONLY and pid != DRIVE_ID:
                continue
            y = wide.ch[chn][pid].to_numpy(dtype=float).copy()
            y[~wide.running] = np.nan
            fit = _fit_line(_design(ctx, feats, pid, n), y)
            if fit is not None:
                fit["features"] = feats
                out[chn][pid] = fit
    return out


# ----------------------------------------------------------------------------- stuck sensors
def stuck_mask(x: np.ndarray, running: np.ndarray) -> np.ndarray:
    """True where a sensor repeats the exact same value for >= STUCK_MIN_SAMPLES consecutive
    running samples. Healthy sensors always carry noise, so exact repetition is a sensor fault."""
    mask = np.zeros(len(x), dtype=bool)
    sel = np.flatnonzero(running & np.isfinite(x))
    if len(sel) < config.STUCK_MIN_SAMPLES:
        return mask
    v = x[sel]
    same = np.concatenate([[False], v[1:] == v[:-1]])
    rid = np.cumsum(~same)
    counts = np.bincount(rid)
    mask[sel[counts[rid] >= config.STUCK_MIN_SAMPLES]] = True
    return mask


# ----------------------------------------------------------------------------- residuals
def analyse(wide: Wide, baseline: dict) -> Analysis:
    an = Analysis(wide=wide)
    ctx = _context(wide)
    n = len(wide.idx)
    for chn, per_point in baseline.items():
        d = pd.DataFrame(np.nan, index=wide.idx, columns=POINT_IDS)
        z = d.copy()
        e = d.copy()
        for pid, fit in per_point.items():
            y = wide.ch[chn][pid].to_numpy(dtype=float).copy()
            sm = stuck_mask(y, wide.running)
            an.stuck[(chn, pid)] = sm
            X = _design(ctx, fit["features"], pid, n)
            exp = fit["intercept"] + (X @ np.asarray(fit["coef"]) if X.shape[1] else 0.0)
            exp = np.broadcast_to(exp, (n,)).astype(float)
            delta = y - exp
            valid = wide.running & ~sm & np.isfinite(delta)
            delta = np.where(valid, delta, np.nan)
            d[pid] = delta
            e[pid] = np.where(valid, exp, np.nan)
            z[pid] = delta / fit["sigma"]
        an.delta[chn], an.z[chn], an.expected[chn] = d, z, e
    # stuck masks for channels without a baseline (e.g. ambient temperature, belt load)
    for chn in config.ALL_CHANNELS:
        for pid in POINT_IDS:
            if (chn, pid) in an.stuck:
                continue
            y = wide.ch[chn][pid].to_numpy(dtype=float)
            if np.isfinite(y).any():
                an.stuck[(chn, pid)] = stuck_mask(y, wide.running)
    return an
