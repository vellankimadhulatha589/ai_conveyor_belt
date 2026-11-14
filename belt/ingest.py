"""Ingestion: validation, de-duplication and storage of incoming readings.

Rules (all deliberately conservative - we never silently drop *data*, only *garbage*):

* rows without a parseable timestamp or with an unknown sensor point are **rejected** (counted);
* values that are not numeric or outside the physical limits are **stored raw but flagged**;
* missing values are stored as NULL;
* the primary key (point_id, ts) makes the endpoint **idempotent**: duplicates are ignored
  and counted, out-of-order arrival is harmless.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import config

_COLUMNS = ["point_id", "ts"] + config.ALL_CHANNELS + ["flags"]
_INSERT = (
    "INSERT OR IGNORE INTO readings(" + ",".join(_COLUMNS) + ") VALUES (" + ",".join("?" * len(_COLUMNS)) + ")"
)


def _totals_update(conn: sqlite3.Connection, result: dict) -> None:
    from .db import get_meta, set_meta

    tot = get_meta(conn, "ingest_totals", {"received": 0, "inserted": 0, "duplicates": 0, "rejected": 0, "flagged": 0})
    for k in tot:
        tot[k] = tot.get(k, 0) + result.get(k, 0)
    set_meta(conn, "ingest_totals", tot)


def ingest_records(conn: sqlite3.Connection, records: list[dict]) -> dict:
    return ingest_frame(conn, pd.DataFrame.from_records(records) if records else pd.DataFrame())


def ingest_frame(conn: sqlite3.Connection, df: pd.DataFrame, track_totals: bool = True) -> dict:
    result = {"received": int(len(df)), "inserted": 0, "duplicates": 0, "rejected": 0, "flagged": 0}
    if df.empty or "point_id" not in df.columns or "ts" not in df.columns:
        result["rejected"] = result["received"]
        return result
    df = df.copy()
    for ch in config.ALL_CHANNELS:
        if ch not in df.columns:
            df[ch] = np.nan

    known = {r[0] for r in conn.execute("SELECT point_id FROM points")}
    ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    valid = (ts.notna() & df["point_id"].isin(known)).to_numpy()
    result["rejected"] = int((~valid).sum())
    df = df[valid].copy()
    ts = ts[valid]
    if df.empty:
        return result
    df["ts"] = ts.dt.tz_convert(None).dt.strftime("%Y-%m-%dT%H:%M:%S")

    flags = np.full(len(df), "", dtype=object)
    for ch in config.ALL_CHANNELS:
        raw = df[ch]
        num = pd.to_numeric(raw, errors="coerce")
        bad_type = (raw.notna() & num.isna()).to_numpy()
        lo, hi = config.LIMITS[ch]
        out_of_range = ((num < lo) | (num > hi)).to_numpy()
        for idx in np.flatnonzero(bad_type | out_of_range):
            flags[idx] += ("bad_type:" if bad_type[idx] else "range:") + ch + ";"
        df[ch] = num
    df["flags"] = flags
    result["flagged"] = int((flags != "").sum())

    sub = df[_COLUMNS]
    rows = sub.astype(object).where(sub.notna(), None).values.tolist()
    before = conn.total_changes
    conn.executemany(_INSERT, rows)
    conn.commit()
    result["inserted"] = conn.total_changes - before
    result["duplicates"] = len(rows) - result["inserted"]
    if track_totals:
        _totals_update(conn, result)
    return result


def add_maintenance(conn: sqlite3.Connection, records: list[dict]) -> int:
    n = 0
    for r in records:
        cur = conn.execute(
            "INSERT OR IGNORE INTO maintenance_log(point_id, mode, failure_time, repaired_ts) VALUES (?,?,?,?)",
            (r["point_id"], r.get("mode"), r["failure_time"], r["repaired_ts"]),
        )
        n += cur.rowcount
    conn.commit()
    return n
