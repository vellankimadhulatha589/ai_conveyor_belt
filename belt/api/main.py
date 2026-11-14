"""FastAPI backend: receive, store and analyse.

Run:  python -m uvicorn belt.api.main:app --port 8000     (interactive docs at /docs)

Responsibilities
* ``POST /readings``     lenient ingestion of simulator data (validation, flagging, de-duplication)
* ``POST /maintenance``  work orders (labels for the model, resets of component history)
* a background thread runs the analysis cycle (anomalies, predictions, point status) whenever new
  data has arrived
* read-only endpoints feed the dashboard
* ``/vision/*``          frame-level detections in, loop-aware defect tracks out
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import config
from ..analysis.pipeline import run_analysis
from ..db import connect, get_meta, init_db, upsert_points
from ..ingest import add_maintenance, ingest_records
from ..simulator.layout import POINTS, POINT_BY_ID
from ..vision import tracker

log = logging.getLogger("belt.api")


class ReadingsIn(BaseModel):
    readings: list[dict[str, Any]]   # deliberately lenient: validation happens in ingest, row by row


class MaintenanceIn(BaseModel):
    records: list[dict[str, Any]]


class ObservationsIn(BaseModel):
    observations: list[dict[str, Any]]


def get_db():
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def _analysis_loop(stop: threading.Event) -> None:
    while not stop.wait(config.ANALYSIS_INTERVAL_S):
        conn = None
        try:
            conn = connect()
            run_analysis(conn)
        except Exception:  # keep the loop alive whatever happens
            log.exception("analysis cycle failed")
        finally:
            if conn is not None:
                conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = connect()
    init_db(conn)
    upsert_points(conn, POINTS)
    conn.close()
    stop = threading.Event()
    t = threading.Thread(target=_analysis_loop, args=(stop,), daemon=True, name="analysis")
    t.start()
    yield
    stop.set()


app = FastAPI(title="Conveyor belt health API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _last_ts(conn) -> str | None:
    return conn.execute("SELECT MAX(ts) AS m FROM readings").fetchone()["m"]


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


# ------------------------------------------------------------------------------------ ingestion
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/readings")
def post_readings(body: ReadingsIn, conn=Depends(get_db)):
    return ingest_records(conn, body.readings)


@app.post("/maintenance")
def post_maintenance(body: MaintenanceIn, conn=Depends(get_db)):
    ok = [r for r in body.records if all(k in r for k in ("point_id", "failure_time", "repaired_ts"))]
    return {"stored": add_maintenance(conn, ok), "rejected": len(body.records) - len(ok)}


@app.post("/analysis/run")
def post_analysis(force: bool = True, conn=Depends(get_db)):
    return run_analysis(conn, force=force)


# ------------------------------------------------------------------------------------ read models
@app.get("/status/summary")
def status_summary(conn=Depends(get_db)):
    counts = {"normal": 0, "abnormal": 0, "predicted_failure": 0}
    for r in conn.execute("SELECT status, COUNT(*) AS n FROM point_status GROUP BY status"):
        counts[r["status"]] = r["n"]
    sev = {s: 0 for s in config.SEVERITY_ORDER}
    for r in conn.execute("SELECT severity, COUNT(*) AS n FROM anomalies WHERE status='active' GROUP BY severity"):
        sev[r["severity"]] = r["n"]
    return {
        "last_data_ts": _last_ts(conn),
        "analysis_last_ts": get_meta(conn, "analysis_last_ts"),
        "belt_running": get_meta(conn, "belt_running"),
        "model_trained": bool(get_meta(conn, "baseline")) and config.PREDICTOR_PATH.exists(),
        "n_points": len(POINTS),
        "status_counts": counts,
        "active_anomalies": sum(sev.values()),
        "anomaly_severity_counts": sev,
        "ingest_totals": get_meta(conn, "ingest_totals", {}),
        "model_metrics": get_meta(conn, "model_metrics", {}),
        "vision": tracker.summary(conn),
    }


@app.get("/points")
def get_points(conn=Depends(get_db)):
    status = {r["point_id"]: dict(r) for r in conn.execute("SELECT * FROM point_status")}
    last_run = conn.execute("SELECT MAX(run_ts) AS m FROM predictions").fetchone()["m"]
    preds = {}
    if last_run:
        preds = {r["point_id"]: dict(r) for r in conn.execute("SELECT * FROM predictions WHERE run_ts=?", (last_run,))}
    out = []
    for p in conn.execute("SELECT * FROM points ORDER BY km"):
        d = dict(p)
        d["channels"] = d["channels"].split(",")
        st = status.get(p["point_id"], {})
        d["status"] = st.get("status", "normal")
        d["reason"] = st.get("reason", "")
        d["max_severity"] = st.get("max_severity")
        pr = preds.get(p["point_id"], {})
        d["p_fail"] = pr.get("p_fail")
        d["failure_mode"] = pr.get("failure_mode")
        d["eta_hours"] = pr.get("eta_hours")
        latest = conn.execute("SELECT * FROM readings WHERE point_id=? ORDER BY ts DESC LIMIT 1",
                              (p["point_id"],)).fetchone()
        d["latest"] = {c: latest[c] for c in config.ALL_CHANNELS if latest is not None and latest[c] is not None}
        d["latest_ts"] = latest["ts"] if latest is not None else None
        out.append(d)
    return out


@app.get("/points/{point_id}/series")
def get_series(point_id: str, hours: int = Query(48, ge=1, le=720),
               channels: str = "vibration,temperature,lateral_offset", conn=Depends(get_db)):
    if point_id not in POINT_BY_ID:
        raise HTTPException(404, "unknown point")
    chs = [c for c in channels.split(",") if c in config.ALL_CHANNELS]
    last = _last_ts(conn)
    if not chs or last is None:
        return {}
    since = _iso(pd.Timestamp(last) - pd.Timedelta(hours=hours))
    rows = conn.execute(f"SELECT ts, flags, {','.join(chs)} FROM readings WHERE point_id=? AND ts>=? ORDER BY ts",
                        (point_id, since)).fetchall()
    stride = max(1, len(rows) // 700)
    out: dict = {c: [] for c in chs}
    for r in rows[::stride]:
        fl = r["flags"] or ""
        for c in chs:
            if r[c] is not None and f":{c};" not in fl:
                out[c].append([r["ts"], r[c]])
    return out


@app.get("/anomalies")
def get_anomalies(status: str = "active", conn=Depends(get_db)):
    q = "SELECT a.*, p.km, p.name FROM anomalies a JOIN points p ON p.point_id=a.point_id"
    params: tuple = ()
    if status != "all":
        q += " WHERE a.status=?"
        params = (status,)
    q += " ORDER BY a.id DESC LIMIT 500"
    rows = [dict(r) for r in conn.execute(q, params)]
    rank = {s: i for i, s in enumerate(config.SEVERITY_ORDER)}
    rows.sort(key=lambda r: (-rank.get(r["severity"], 0), r["km"]))
    return rows


@app.get("/predictions/latest")
def get_predictions(include_all: bool = False, conn=Depends(get_db)):
    last_run = conn.execute("SELECT MAX(run_ts) AS m FROM predictions").fetchone()["m"]
    if not last_run:
        return {"run_ts": None, "predictions": []}
    q = "SELECT r.*, p.km, p.name FROM predictions r JOIN points p ON p.point_id=r.point_id WHERE r.run_ts=?"
    if not include_all:
        q += " AND r.level != 'none'"
    q += " ORDER BY r.p_fail DESC"
    return {"run_ts": last_run, "predictions": [dict(r) for r in conn.execute(q, (last_run,))]}


@app.get("/data-quality")
def data_quality(hours: int = 24, conn=Depends(get_db)):
    last = _last_ts(conn)
    if last is None:
        return {}
    since = _iso(pd.Timestamp(last) - pd.Timedelta(hours=hours))
    n_rows = conn.execute("SELECT COUNT(*) AS n FROM readings WHERE ts>=?", (since,)).fetchone()["n"]
    expected = len(POINTS) * (hours * 60 // config.SAMPLE_MINUTES)
    flags: dict = {}
    for r in conn.execute("SELECT flags FROM readings WHERE ts>=? AND flags != ''", (since,)):
        for f in (r["flags"] or "").split(";"):
            if f:
                flags[f] = flags.get(f, 0) + 1
    nulls = {}
    for c in ("vibration", "temperature", "lateral_offset"):
        nulls[c] = conn.execute(f"SELECT COUNT(*) AS n FROM readings WHERE ts>=? AND {c} IS NULL", (since,)).fetchone()["n"]
    sensor = [dict(r) for r in conn.execute(
        "SELECT a.point_id, a.type, a.detail, a.first_seen, p.km FROM anomalies a JOIN points p ON p.point_id=a.point_id "
        "WHERE a.status='active' AND a.type LIKE 'sensor_%' ORDER BY p.km")]
    return {
        "window_hours": hours, "rows_received": n_rows, "rows_expected": expected,
        "rows_missing_pct": round(100.0 * max(expected - n_rows, 0) / max(expected, 1), 2),
        "flagged_values": flags, "null_values": nulls, "active_sensor_faults": sensor,
        "ingest_totals": get_meta(conn, "ingest_totals", {}),
    }


# ------------------------------------------------------------------------------------ vision
@app.post("/vision/observations")
def post_observations(body: ObservationsIn, conn=Depends(get_db)):
    return {"counts": tracker.record_batch(conn, body.observations)}


@app.get("/vision/tracks")
def get_tracks(statuses: str = "confirmed,suspected", conn=Depends(get_db)):
    return tracker.list_tracks(conn, tuple(s for s in statuses.split(",") if s))


@app.get("/vision/summary")
def get_vision_summary(conn=Depends(get_db)):
    return tracker.summary(conn)


@app.post("/vision/reset")
def post_vision_reset(conn=Depends(get_db)):
    tracker.reset_vision(conn)
    return {"status": "cleared"}


@app.post("/vision/tracks/{track_id}/repair")
def post_track_repair(track_id: int, conn=Depends(get_db)):
    if not tracker.mark_repaired(conn, track_id):
        raise HTTPException(404, "unknown track")
    return {"status": "repaired"}


config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/vision/frames", StaticFiles(directory=str(config.FRAMES_DIR)), name="frames")
