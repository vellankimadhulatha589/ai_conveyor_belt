"""Train the baseline and the failure predictor from the stored history, and evaluate honestly.

    python -m belt.analysis.train

Evaluation protocol (time-based, no random shuffling - neighbouring snapshots of one failure are
almost identical, so a random split would leak):

* the last ``HOLDOUT_TAIL_DAYS`` days are excluded from training *and* evaluation: faults still in
  progress there have no work order yet, so their snapshots would be labelled "healthy" wrongly;
* the remaining period is split 70/30 by time with a 7-day purge gap in between;
* the model fitted on the first part is scored on the second part;
* a naive rule ("any channel's 6 h mean residual >= 4 sigma") is scored the same way as a reference;
* the deployed model is then re-fitted on everything.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .. import config
from ..db import connect, init_db, set_meta
from .features import build_features, hourly_badness, label_snapshots
from .pipeline import resets_from_db
from .predictor import FailurePredictor
from .prepare import analyse, fit_baseline, load_readings, to_wide

HOLDOUT_TAIL_DAYS = 10
PURGE_H = 168
GRACE_H = 288  # a healthy-labelled snapshot this close to a failure is "early warning", not a false alarm


def fetch_failures(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT point_id, mode, failure_time, repaired_ts FROM maintenance_log")]


def _episodes(ts: pd.Series, step_h: float) -> int:
    ts = ts.sort_values()
    if ts.empty:
        return 0
    gaps = ts.diff().dt.total_seconds().fillna(1e12) / 3600.0
    return int((gaps > 1.5 * step_h).sum())


def score_alerts(te: pd.DataFrame, flagged: np.ndarray, failures: list[dict], t_split: pd.Timestamp) -> dict:
    """Snapshot-level and fault-level quality of a boolean alert signal."""
    y = (te["y"].to_numpy() > 0)
    tp = int((flagged & y).sum())
    fp = int((flagged & ~y).sum())
    fn = int((~flagged & y).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    # false alarms, ignoring early warnings inside the grace period before a failure
    healthy = (~y) & ~(te["ttf_next"].to_numpy() <= GRACE_H)
    fa = flagged & healthy
    te = te.assign(_flag=flagged, _fa=fa)
    episodes = 0
    for _pid, g in te[te["_fa"]].groupby("point_id"):
        episodes += _episodes(g["ts"], config.SNAPSHOT_STEP_H)
    # fault-level: was every failure announced, and how early?
    leads, detected, total = [], 0, 0
    for f in failures:
        ft = pd.Timestamp(f["failure_time"])
        if ft < t_split or f["mode"] not in config.PREDICTION_WINDOW_H:
            continue
        window = pd.Timedelta(hours=config.PREDICTION_WINDOW_H[f["mode"]])
        sel = te[(te["point_id"] == f["point_id"]) & (te["ts"] < ft) & (te["ts"] >= ft - window) & te["_flag"]]
        total += 1
        if len(sel):
            detected += 1
            leads.append((ft - sel["ts"].min()) / pd.Timedelta(hours=1))
    n_days = max((te["ts"].max() - te["ts"].min()).total_seconds() / 86400.0, 1.0)
    return {
        "snapshots": int(len(te)), "positive_snapshots": int(y.sum()),
        "precision": round(precision, 3), "recall": round(recall, 3),
        "false_alarm_episodes": episodes,
        "false_alarm_episodes_per_100_point_days": round(100.0 * episodes / (n_days * len(te["point_id"].unique())), 2),
        "faults_in_test": total, "faults_detected": detected,
        "median_lead_time_h": round(float(np.median(leads)), 1) if leads else None,
        "min_lead_time_h": round(float(np.min(leads)), 1) if leads else None,
    }


def evaluate(te: pd.DataFrame, pred: pd.DataFrame, failures: list[dict], t_split: pd.Timestamp) -> dict:
    thr = config.P_PREDICTED_FAILURE
    score = pred["p_fail"].to_numpy()
    y = te["y"].to_numpy() > 0
    ml = score_alerts(te, score >= thr, failures, t_split)
    ml["threshold"] = thr
    ml["roc_auc"] = round(float(roc_auc_score(y, score)), 3) if 0 < y.sum() < len(y) else None
    pos = y & (score >= thr)
    if pos.any():
        true_mode = te["y"].to_numpy()[pos]
        pred_mode = np.array([config.CLASSES.index(m) for m in pred["mode"].to_numpy()[pos]])
        ml["mode_accuracy_on_detected"] = round(float((true_mode == pred_mode).mean()), 3)
        err = np.abs(pred["eta_h"].to_numpy()[pos] - te["tth"].to_numpy()[pos])
        ml["eta_median_abs_error_h"] = round(float(np.median(err)), 1)
        lo, hi = pred["eta_low_h"].to_numpy()[pos], pred["eta_high_h"].to_numpy()[pos]
        tt = te["tth"].to_numpy()[pos]
        ml["eta_band_coverage"] = round(float(((tt >= lo) & (tt <= hi)).mean()), 3)
    cols = [f"{c}_mean6" for c in ("vibration", "temperature", "lateral_offset", "motor_current", "voltage", "slip", "belt_speed")]
    rule = np.nan_to_num(te[cols].to_numpy(dtype=float), nan=0.0).max(axis=1) >= 4.0
    return {"model": ml, "rule_baseline": score_alerts(te, rule, failures, t_split)}


def train_all(conn: sqlite3.Connection, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    log("loading readings ...")
    wide = to_wide(load_readings(conn))
    if wide is None:
        raise RuntimeError("no readings in the database - run scripts/bootstrap.py first")
    log(f"  {len(wide.idx)} timestamps x {len(wide.ch['vibration'].columns)} points, "
        f"{wide.n_flagged} flagged raw values ignored")

    log("fitting context-aware baselines ...")
    baseline = fit_baseline(wide)
    set_meta(conn, "baseline", baseline)

    log("computing residuals and features ...")
    an = analyse(wide, baseline)
    hourly = hourly_badness(an.z)
    failures = fetch_failures(conn)
    resets = resets_from_db(conn)
    t_end = (wide.idx[-1] - pd.Timedelta(days=HOLDOUT_TAIL_DAYS)).floor("h")
    t0 = (wide.idx[0] + pd.Timedelta(hours=24)).ceil("h")
    snaps = pd.date_range(t0, t_end, freq=f"{config.SNAPSHOT_STEP_H}h")
    feat = build_features(hourly, snaps, resets)
    lab = label_snapshots(feat, failures)
    lab = lab[lab["valid"]].reset_index(drop=True)
    n_pos = int((lab["y"] > 0).sum())
    log(f"  {len(lab)} labelled snapshots, {n_pos} positive, {len(failures)} work orders")
    if n_pos < 20:
        raise RuntimeError("too few failures in the history to train a predictor - simulate more days")

    metrics: dict = {"snapshots": int(len(lab)), "positive_snapshots": n_pos, "work_orders": len(failures)}
    t_split = snaps[int(len(snaps) * 0.7)]
    tr = lab[lab["ts"] < t_split - pd.Timedelta(hours=PURGE_H)]
    te = lab[lab["ts"] >= t_split].reset_index(drop=True)
    if (tr["y"] > 0).sum() >= 40 and (te["y"] > 0).sum() >= 5:
        log(f"holdout evaluation (train < {t_split - pd.Timedelta(hours=PURGE_H):%Y-%m-%d}, test >= {t_split:%Y-%m-%d}) ...")
        m = FailurePredictor().fit(tr, tr["y"].to_numpy(), tr["tth"].to_numpy())
        metrics["holdout"] = evaluate(te, m.predict(te), failures, t_split)
        metrics["holdout"]["train_span"] = [str(tr["ts"].min()), str(tr["ts"].max())]
        metrics["holdout"]["test_span"] = [str(te["ts"].min()), str(te["ts"].max())]
        log(json.dumps(metrics["holdout"], indent=2))
    else:
        log("not enough positives for a holdout split - skipping evaluation")

    log("fitting final model on all labelled snapshots ...")
    final = FailurePredictor().fit(lab, lab["y"].to_numpy(), lab["tth"].to_numpy())
    final.save()
    set_meta(conn, "model_metrics", metrics)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"saved {config.PREDICTOR_PATH}")
    return metrics


if __name__ == "__main__":  # pragma: no cover
    c = connect()
    init_db(c)
    train_all(c)
