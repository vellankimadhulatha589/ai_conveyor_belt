"""Defect tracking on a *looping* belt (standard library only).

The classifier looks at single frames. This module turns a stream of frames into a list of
physical defects, handling what the loop causes:

* **Same defect, many sightings.** Every lap the camera sees the same stretch of belt again. A
  defect is identified by *where it sits on the belt loop* (position -> 10 m segment), not by
  when it was photographed. Sightings within +-1 segment merge into one **track**.
* **Partial coverage / flicker.** One frame per segment per lap with some positional jitter means a
  small defect is only seen in some laps. A track therefore needs ``CONFIRM_LAPS`` sightings in
  *different* laps to become *confirmed*; a missed lap does not clear a confirmed defect (damage
  does not heal by itself).
* **Transient false positives** (a lump of ore stuck to the belt, dust, a bad frame) are seen once
  and then disappear: unconfirmed tracks are *dismissed* after ``DISMISS_AFTER_MISSES`` clean laps.
* **Known belt features.** Vulcanised splices look like defects, so detections near a known splice
  position are logged but ignored.
* **Bad frames** (blur, too dark/bright) are logged but never update a track - a bad frame is
  not evidence of a clean belt.
* **Growth.** The severity per lap is kept per track so wear progression can be shown.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .. import config

OPEN = ("suspected", "confirmed")


def seg_of(position_m: float) -> int:
    return int((position_m % config.LOOP_LENGTH_M) // config.SEGMENT_M) % config.N_SEGMENTS


def circ_seg_diff(a: int, b: int) -> int:
    d = abs(a - b)
    return min(d, config.N_SEGMENTS - d)


def near_splice(position_m: float, margin_m: float = 0.0) -> bool:
    """True if the position is on a known splice. The tracker passes ``margin_m`` because the reported
    position carries encoder error: a frame that really shows the stripe can be reported just outside it."""
    pos = position_m % config.LOOP_LENGTH_M
    for s in config.KNOWN_SPLICES_M:
        d = abs(pos - s)
        if min(d, config.LOOP_LENGTH_M - d) <= config.SPLICE_MASK_M + margin_m:
            return True
    return False


def severity_level(sev: float) -> str:
    return "low" if sev < 0.33 else "medium" if sev < 0.66 else "high"


def is_normal_label(label: str) -> bool:
    return str(label).strip().lower() in config.NORMAL_CLASS_NAMES


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def record_observation(conn: sqlite3.Connection, obs: dict) -> dict:
    """Store one analysed frame and update the defect tracks. Returns {'outcome', 'track_id'}."""
    for k in ("frame_id", "lap", "position_m", "label", "score"):
        if k not in obs or obs[k] is None:
            return {"outcome": "rejected", "track_id": None, "reason": f"missing field {k}"}
    if conn.execute("SELECT 1 FROM vision_observations WHERE frame_id=?", (obs["frame_id"],)).fetchone():
        return {"outcome": "duplicate", "track_id": None}
    lap = int(obs["lap"])
    pos = float(obs["position_m"]) % config.LOOP_LENGTH_M
    seg = seg_of(pos)
    score = float(obs["score"])
    sev = float(obs.get("severity") or 0.0)
    quality_ok = bool(obs.get("quality_ok", True))
    ts = obs.get("ts") or _now()
    label = str(obs["label"])

    track_id = None
    if near_splice(pos, config.POSITION_ERROR_M):
        outcome = "ignored_splice"
    elif not quality_ok:
        outcome = "low_quality"
    elif (not is_normal_label(label)) and score >= config.DETECTION_THRESHOLD:
        outcome, track_id = _on_defect(conn, obs, lap, pos, seg, label, score, sev, ts)
    else:
        outcome, track_id = _on_clean(conn, lap, seg)

    conn.execute(
        "INSERT INTO vision_observations(frame_id, ts, lap, position_m, segment, label, score, severity, quality_ok, "
        "quality_note, bbox, side, image_file, outcome, track_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (obs["frame_id"], ts, lap, pos, seg, label, score, sev, int(quality_ok), obs.get("quality_note", ""),
         json.dumps(obs.get("bbox")) if obs.get("bbox") is not None else None, obs.get("side", "carry"),
         obs.get("image_file"), outcome, track_id))
    return {"outcome": outcome, "track_id": track_id}


def _find_track(conn: sqlite3.Connection, seg: int):
    r = config.MERGE_RADIUS_SEGMENTS
    rows = conn.execute(
        "SELECT * FROM vision_tracks WHERE status IN ('suspected','confirmed') "
        "AND (ABS(segment-?) <= ? OR ABS(segment-?) >= ?)", (seg, r, seg, config.N_SEGMENTS - r)).fetchall()
    if not rows:
        return None
    return min(rows, key=lambda t: circ_seg_diff(t["segment"], seg))


def _on_defect(conn, obs, lap, pos, seg, label, score, sev, ts):
    t = _find_track(conn, seg)
    image = obs.get("image_file")
    if t is None:
        status = "confirmed" if config.CONFIRM_LAPS <= 1 else "suspected"
        cur = conn.execute(
            "INSERT INTO vision_tracks(segment, position_m, label, label_votes, status, first_lap, last_lap, laps_seen, "
            "last_checked_lap, consecutive_misses, n_defect_obs, n_clean_obs, max_severity, last_severity, "
            "severity_history, image_file, first_seen_ts, last_seen_ts) "
            "VALUES (?,?,?,?,?,?,?,1,?,0,1,0,?,?,?,?,?,?)",
            (seg, pos, label, json.dumps({label: score}), status, lap, lap, lap, sev, sev,
             json.dumps([[lap, sev]]), image, ts, ts))
        return "new_track", cur.lastrowid

    votes = json.loads(t["label_votes"] or "{}")
    votes[label] = votes.get(label, 0.0) + score
    best = max(votes, key=votes.get)
    hist = json.loads(t["severity_history"] or "[]")
    laps = {h[0]: i for i, h in enumerate(hist)}
    laps_seen = t["laps_seen"]
    if lap in laps:
        hist[laps[lap]][1] = max(hist[laps[lap]][1], sev)   # same lap seen twice: not a new confirmation
    else:
        hist.append([lap, sev])
        laps_seen += 1
    hist = hist[-60:]
    n = t["n_defect_obs"]
    new_pos = t["position_m"]
    if abs(pos - t["position_m"]) < config.LOOP_LENGTH_M / 2:  # do not average across the loop seam
        new_pos = (t["position_m"] * n + pos) / (n + 1)
    status = "confirmed" if (t["status"] == "confirmed" or laps_seen >= config.CONFIRM_LAPS) else "suspected"
    keep_image = image if (image and (not t["image_file"] or sev >= t["max_severity"])) else t["image_file"]
    conn.execute(
        "UPDATE vision_tracks SET position_m=?, segment=?, label=?, label_votes=?, status=?, last_lap=?, laps_seen=?, "
        "last_checked_lap=?, consecutive_misses=0, n_defect_obs=n_defect_obs+1, max_severity=?, last_severity=?, "
        "severity_history=?, image_file=?, last_seen_ts=? WHERE id=?",
        (new_pos, seg_of(new_pos), best, json.dumps(votes), status, max(lap, t["last_lap"] or lap), laps_seen,
         max(lap, t["last_checked_lap"] or lap), max(sev, t["max_severity"] or 0.0), sev, json.dumps(hist),
         keep_image, ts, t["id"]))
    became = status == "confirmed" and t["status"] != "confirmed"
    return ("confirmed_now" if became else "merged"), t["id"]


def _on_clean(conn, lap, seg):
    t = conn.execute("SELECT * FROM vision_tracks WHERE status IN ('suspected','confirmed') AND segment=?",
                     (seg,)).fetchone()
    if t is None:
        return "clean", None
    if t["last_checked_lap"] == lap:      # this lap already produced a sighting (or a miss) for the track
        return "clean", t["id"]
    misses = (t["consecutive_misses"] or 0) + 1
    status = t["status"]
    outcome = "clean"
    if status == "suspected" and misses >= config.DISMISS_AFTER_MISSES:
        status, outcome = "dismissed", "dismissed_track"
    conn.execute("UPDATE vision_tracks SET consecutive_misses=?, last_checked_lap=?, n_clean_obs=n_clean_obs+1, "
                 "status=? WHERE id=?", (misses, lap, status, t["id"]))
    return outcome, t["id"]


def record_batch(conn: sqlite3.Connection, observations: list[dict]) -> dict:
    counts: dict = {}
    for o in observations:
        r = record_observation(conn, o)
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    conn.commit()
    return counts


def mark_repaired(conn: sqlite3.Connection, track_id: int) -> bool:
    cur = conn.execute("UPDATE vision_tracks SET status='repaired' WHERE id=?", (track_id,))
    conn.commit()
    return cur.rowcount > 0


def list_tracks(conn: sqlite3.Connection, statuses=("confirmed", "suspected")) -> list[dict]:
    q = ",".join("?" * len(statuses))
    rows = conn.execute(f"SELECT * FROM vision_tracks WHERE status IN ({q}) ORDER BY max_severity DESC, id", tuple(statuses))
    out = []
    for r in rows:
        d = dict(r)
        d["label_votes"] = json.loads(d["label_votes"] or "{}")
        d["severity_history"] = json.loads(d["severity_history"] or "[]")
        d["severity_level"] = severity_level(d["max_severity"] or 0.0)
        d["km_on_carry_side"] = round(d["position_m"] / 1000.0, 3)
        out.append(d)
    return out


def summary(conn: sqlite3.Connection) -> dict:
    c = conn.execute(
        "SELECT COUNT(*) AS n, "
        "SUM(CASE WHEN outcome IN ('new_track','merged','confirmed_now') THEN 1 ELSE 0 END) AS defect_frames, "
        "SUM(CASE WHEN outcome='ignored_splice' THEN 1 ELSE 0 END) AS splice_frames, "
        "SUM(CASE WHEN outcome='low_quality' THEN 1 ELSE 0 END) AS low_quality_frames, "
        "MAX(lap) AS last_lap FROM vision_observations").fetchone()
    tracks = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) AS n FROM vision_tracks GROUP BY status")}
    last_lap = c["last_lap"]
    coverage = None
    if last_lap is not None:
        r = conn.execute("SELECT COUNT(DISTINCT segment) AS s FROM vision_observations WHERE lap=? AND quality_ok=1",
                         (last_lap,)).fetchone()
        coverage = round(r["s"] / config.N_SEGMENTS, 3)
    unique = tracks.get("confirmed", 0) + tracks.get("suspected", 0)
    return {
        "frames": c["n"] or 0, "defect_frames": c["defect_frames"] or 0, "splice_frames_ignored": c["splice_frames"] or 0,
        "low_quality_frames": c["low_quality_frames"] or 0, "last_lap": last_lap, "coverage_last_lap": coverage,
        # the camera only sees CAMERA_FRAME_M of each SEGMENT_M, so imaged surface << segments sampled
        "surface_imaged_last_lap": None if coverage is None else round(coverage * config.CAMERA_FRAME_M / config.SEGMENT_M, 3),
        "tracks": tracks, "unique_open_defects": unique,
        "dedup_ratio": round((c["defect_frames"] or 0) / unique, 2) if unique else None,
        "loop_length_m": config.LOOP_LENGTH_M, "segment_m": config.SEGMENT_M,
    }


def reset_vision(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM vision_observations")
    conn.execute("DELETE FROM vision_tracks")
    conn.commit()