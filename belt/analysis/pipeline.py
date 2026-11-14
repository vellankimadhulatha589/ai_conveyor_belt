"""Live analysis cycle executed by the backend whenever new data has arrived.

    stored readings (last ~100 h)
        -> residuals vs learned baseline          (prepare.py)
        -> anomalies + sensor faults              (anomaly.py)   -> table ``anomalies``
        -> features + failure prediction          (features.py, predictor.py) -> table ``predictions``
        -> one status per sensor point            (normal / abnormal / predicted_failure)

"Now" is the timestamp of the newest reading, not the wall clock: the simulator runs faster than
real time, and a real deployment would behave the same when a data feed is delayed.
"""
from __future__ import annotations

import os
import sqlite3
import threading

import numpy as np
import pandas as pd

from .. import config
from ..db import get_meta, set_meta
from ..simulator.layout import POINTS
from .anomaly import detect_equipment, detect_sensor_faults
from .features import FEATURE_NAMES, build_features, hourly_badness
from .predictor import FailurePredictor
from .prepare import analyse, load_readings, to_wide

_lock = threading.Lock()
_model_cache: dict = {"mtime": None, "obj": None}


def _predictor() -> FailurePredictor | None:
    path = config.PREDICTOR_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if _model_cache["mtime"] != mtime:
        _model_cache["obj"] = FailurePredictor.load(path)
        _model_cache["mtime"] = mtime
    return _model_cache["obj"]


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


# ----------------------------------------------------------------------------- anomalies
def apply_anomalies(conn: sqlite3.Connection, results: list[dict], now: str) -> None:
    active = {(r["point_id"], r["type"]): r for r in
              conn.execute("SELECT * FROM anomalies WHERE status='active'").fetchall()}
    for r in results:
        key = (r["point_id"], r["type"])
        if r["state"] == "active":
            if key in active:
                conn.execute(
                    "UPDATE anomalies SET severity=?, last_seen=?, value=?, expected=?, delta=?, detail=? WHERE id=?",
                    (r["severity"], now, r.get("value"), r.get("expected"), r.get("delta"), r.get("detail"),
                     active[key]["id"]))
            else:
                conn.execute(
                    "INSERT INTO anomalies(point_id, channel, type, severity, status, first_seen, last_seen, "
                    "value, expected, delta, detail) VALUES (?,?,?,?, 'active', ?,?,?,?,?,?)",
                    (r["point_id"], r.get("channel"), r["type"], r["severity"], now, now, r.get("value"),
                     r.get("expected"), r.get("delta"), r.get("detail")))
        elif r["state"] == "normal" and key in active:
            conn.execute("UPDATE anomalies SET status='resolved', resolved_ts=? WHERE id=?", (now, active[key]["id"]))
    conn.commit()


def apply_maintenance(conn: sqlite3.Connection, now: str) -> None:
    """A completed repair resolves the equipment anomalies that existed before it."""
    last_id = get_meta(conn, "maintenance_applied_id", 0)
    rows = conn.execute("SELECT * FROM maintenance_log WHERE id > ? AND repaired_ts <= ? ORDER BY id",
                        (last_id, now)).fetchall()
    for m in rows:
        conn.execute(
            "UPDATE anomalies SET status='resolved', resolved_ts=? WHERE status='active' AND point_id=? "
            "AND type NOT LIKE 'sensor_%' AND first_seen <= ?",
            (m["repaired_ts"], m["point_id"], m["repaired_ts"]))
        last_id = max(last_id, m["id"])
    if rows:
        set_meta(conn, "maintenance_applied_id", last_id)
    conn.commit()


def resets_from_db(conn: sqlite3.Connection, now: str | None = None) -> dict:
    sql = "SELECT point_id, repaired_ts FROM maintenance_log"
    params: tuple = ()
    if now:
        sql += " WHERE repaired_ts <= ?"
        params = (now,)
    resets: dict = {}
    for r in conn.execute(sql, params):
        resets.setdefault(r["point_id"], []).append(pd.Timestamp(r["repaired_ts"]))
    return resets


# ----------------------------------------------------------------------------- predictions
def mask_before_reset(an, resets: dict) -> None:
    """Ignore everything measured before the last completed repair of a point: the *old* part's
    degraded samples must not trigger anomalies for the *new* part."""
    idx = an.wide.idx
    for pid, times in resets.items():
        if not times:
            continue
        m = np.asarray(idx < max(times))
        if not m.any():
            continue
        for table in (an.z, an.delta, an.expected):
            for chn in table:
                table[chn].loc[m, pid] = np.nan


def predict_now(an, predictor: FailurePredictor, now: str, resets: dict) -> list[dict]:
    hourly = hourly_badness(an.z)
    t = an.wide.idx[-1]
    feat = build_features(hourly, pd.DatetimeIndex([t]), resets)
    pred = predictor.predict(feat)
    rows = []
    for k in range(len(feat)):
        f = feat.iloc[k]
        p = pred.iloc[k]
        no_data = bool(f[FEATURE_NAMES].isna().all())
        p_fail = 0.0 if no_data else float(p["p_fail"])
        level = "predicted_failure" if p_fail >= config.P_PREDICTED_FAILURE else \
            "watch" if p_fail >= config.P_WATCH else "none"
        mode = str(p["mode"])
        eta_h = float(p["eta_h"])
        rows.append({
            "point_id": f["point_id"], "failure_mode": mode if level != "none" else None,
            "p_fail": p_fail, "level": level,
            "eta_hours": eta_h if level != "none" else None,
            "eta_low_h": float(p["eta_low_h"]) if level != "none" else None,
            "eta_high_h": float(p["eta_high_h"]) if level != "none" else None,
            "eta_ts": _iso(pd.Timestamp(now) + pd.Timedelta(hours=eta_h)) if level != "none" else None,
            "evidence": predictor.evidence(f, mode) if level != "none" else None,
        })
    return rows


def store_predictions(conn: sqlite3.Connection, rows: list[dict], now: str) -> None:
    conn.execute("DELETE FROM predictions WHERE run_ts=?", (now,))
    conn.executemany(
        "INSERT INTO predictions(run_ts, point_id, failure_mode, p_fail, level, eta_hours, eta_low_h, eta_high_h, "
        "eta_ts, evidence) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(now, r["point_id"], r["failure_mode"], r["p_fail"], r["level"], r["eta_hours"], r["eta_low_h"],
          r["eta_high_h"], r["eta_ts"], r["evidence"]) for r in rows])
    cutoff = _iso(pd.Timestamp(now) - pd.Timedelta(days=3))
    conn.execute("DELETE FROM predictions WHERE run_ts < ?", (cutoff,))
    conn.commit()


# ----------------------------------------------------------------------------- status
def update_status(conn: sqlite3.Connection, pred_rows: list[dict], now: str) -> None:
    sev_rank = {s: i for i, s in enumerate(config.SEVERITY_ORDER)}
    anomalies: dict = {}
    for a in conn.execute("SELECT * FROM anomalies WHERE status='active'"):
        anomalies.setdefault(a["point_id"], []).append(a)
    preds = {r["point_id"]: r for r in pred_rows}
    for p in POINTS:
        status, reason, max_sev = "normal", "", None
        acts = sorted(anomalies.get(p.point_id, []), key=lambda a: -sev_rank.get(a["severity"], 0))
        if acts:
            status = "abnormal"
            max_sev = acts[0]["severity"]
            reason = ", ".join(f"{a['type']} ({a['severity']})" for a in acts[:3])
        pr = preds.get(p.point_id)
        if pr and pr["level"] == "predicted_failure":
            status = "predicted_failure"
            reason = f"{pr['failure_mode']} failure in ~{pr['eta_hours']:.0f} h (p={pr['p_fail']:.0%})" + \
                (f"; {reason}" if reason else "")
        conn.execute("INSERT OR REPLACE INTO point_status(point_id, status, reason, max_severity, updated_ts) "
                     "VALUES (?,?,?,?,?)", (p.point_id, status, reason, max_sev, now))
    conn.commit()


# ----------------------------------------------------------------------------- orchestration
def run_analysis(conn: sqlite3.Connection, force: bool = False) -> dict:
    with _lock:
        last = conn.execute("SELECT MAX(ts) AS m FROM readings").fetchone()["m"]
        if last is None:
            return {"status": "no_data"}
        if not force and get_meta(conn, "analysis_last_ts") == last:
            return {"status": "up_to_date", "data_ts": last}
        baseline = get_meta(conn, "baseline")
        predictor = _predictor()
        if baseline is None or predictor is None:
            return {"status": "not_trained", "data_ts": last}
        since = _iso(pd.Timestamp(last) - pd.Timedelta(hours=config.ANALYSIS_LOOKBACK_H))
        wide = to_wide(load_readings(conn, since=since))
        if wide is None:
            return {"status": "no_data"}
        an = analyse(wide, baseline)
        resets = resets_from_db(conn, last)
        mask_before_reset(an, resets)
        results = detect_equipment(an) + detect_sensor_faults(an)
        apply_anomalies(conn, results, last)
        apply_maintenance(conn, last)
        rows = predict_now(an, predictor, last, resets)
        store_predictions(conn, rows, last)
        update_status(conn, rows, last)
        set_meta(conn, "analysis_last_ts", last)
        set_meta(conn, "belt_running", bool(wide.running[-1]))
        return {"status": "ok", "data_ts": last, "anomaly_checks": len(results),
                "predicted_failures": sum(r["level"] == "predicted_failure" for r in rows)}
