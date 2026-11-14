"""SQLite storage layer (standard library only).

SQLite was chosen deliberately: zero set-up for a reviewer, one file, and the data volume
of this project (~0.7 M readings for 90 days of 26 sensor suites) is well within its limits.
The access code is isolated here so swapping in PostgreSQL/TimescaleDB later only touches
this module and ``ingest.py``.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS points (
    point_id TEXT PRIMARY KEY, name TEXT, km REAL, kind TEXT,
    elevation_m REAL, curve INTEGER, channels TEXT
);

-- Raw readings, one row per (sensor point, timestamp). NULL = missing / not applicable.
-- Raw values are never modified; suspicious values are flagged in `flags`.
CREATE TABLE IF NOT EXISTS readings (
    point_id TEXT NOT NULL, ts TEXT NOT NULL,
    vibration REAL, temperature REAL, lateral_offset REAL, ambient_temp REAL,
    belt_speed REAL, drive_speed REAL, belt_load REAL, motor_current REAL, voltage REAL,
    flags TEXT DEFAULT '',
    PRIMARY KEY (point_id, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts);

-- Work orders: what a maintenance system (CMMS) would know. Used for model labels and to
-- reset the feature history of a component after it was repaired/replaced.
CREATE TABLE IF NOT EXISTS maintenance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, point_id TEXT NOT NULL, mode TEXT,
    failure_time TEXT NOT NULL, repaired_ts TEXT NOT NULL,
    UNIQUE (point_id, failure_time)
);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT, point_id TEXT NOT NULL, channel TEXT, type TEXT NOT NULL,
    severity TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
    first_seen TEXT, last_seen TEXT, resolved_ts TEXT,
    value REAL, expected REAL, delta REAL, detail TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_anomaly ON anomalies(point_id, type) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_ts TEXT NOT NULL, point_id TEXT NOT NULL,
    failure_mode TEXT, p_fail REAL, level TEXT,
    eta_hours REAL, eta_low_h REAL, eta_high_h REAL, eta_ts TEXT, evidence TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_ts);

CREATE TABLE IF NOT EXISTS point_status (
    point_id TEXT PRIMARY KEY, status TEXT NOT NULL, reason TEXT, max_severity TEXT, updated_ts TEXT
);

-- Vision: every analysed frame is kept (audit trail); tracks are the de-duplicated defects.
CREATE TABLE IF NOT EXISTS vision_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, frame_id TEXT UNIQUE, ts TEXT, lap INTEGER,
    position_m REAL, segment INTEGER, label TEXT, score REAL, severity REAL,
    quality_ok INTEGER, quality_note TEXT, bbox TEXT, side TEXT, image_file TEXT,
    outcome TEXT, track_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_vobs_lap ON vision_observations(lap);

CREATE TABLE IF NOT EXISTS vision_tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, segment INTEGER NOT NULL, position_m REAL,
    label TEXT, label_votes TEXT, status TEXT NOT NULL,
    first_lap INTEGER, last_lap INTEGER, laps_seen INTEGER DEFAULT 0,
    last_checked_lap INTEGER, consecutive_misses INTEGER DEFAULT 0,
    n_defect_obs INTEGER DEFAULT 0, n_clean_obs INTEGER DEFAULT 0,
    max_severity REAL DEFAULT 0, last_severity REAL DEFAULT 0, severity_history TEXT DEFAULT '[]',
    image_file TEXT, first_seen_ts TEXT, last_seen_ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_vtracks_segment ON vision_tracks(segment);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path) if path else config.DB_PATH
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:  # pragma: no cover
        pass
    return conn


@contextmanager
def session(path: Path | str | None = None):
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_points(conn: sqlite3.Connection, points) -> None:
    rows = [(p.point_id, p.name, p.km, p.kind, p.elevation_m, int(p.curve), ",".join(p.channels)) for p in points]
    conn.executemany(
        "INSERT OR REPLACE INTO points(point_id,name,km,kind,elevation_m,curve,channels) VALUES (?,?,?,?,?,?,?)", rows
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (TypeError, ValueError):
        return default


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", (key, json.dumps(value)))
    conn.commit()


def reset_db(path: Path | str | None = None) -> None:
    path = Path(path) if path else config.DB_PATH
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()
