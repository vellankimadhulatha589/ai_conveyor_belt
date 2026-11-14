"""Anomaly detection on residuals.

An anomaly is raised for (sensor point, type) when the **median** of the last ~1 h of running
samples deviates from the learned expectation by

* a robust z-score >= ``ANOMALY_Z``  (statistically unusual), **and**
* an absolute amount >= the first severity threshold of that channel (practically relevant),

in the direction that is physically bad for that channel. Using the median over a window means a
single spike or impossible reading can never raise an alarm; a sustained shift does.

Every check returns one of three states:

* ``active``  - condition present (with severity),
* ``normal``  - enough fresh data and no deviation  -> an open anomaly may be resolved,
* ``unknown`` - not enough fresh data (belt stopped, sensor silent) -> leave things as they are.

Sensor faults (stuck, offline) are reported as their own anomaly types so that they are never
mistaken for equipment faults.
"""
from __future__ import annotations

import numpy as np

from .. import config
from .prepare import Analysis, POINT_IDS, DRIVE_ONLY
from ..simulator.layout import DRIVE_ID


def severity_for(channel: str, amount: float) -> str | None:
    t1, t2, t3 = config.ANOMALY_RULES[channel][2]
    if amount >= 1.5 * t3:
        return "critical"
    if amount >= t3:
        return "high"
    if amount >= t2:
        return "medium"
    if amount >= t1:
        return "low"
    return None


def _max_sev(a: str | None, b: str | None) -> str | None:
    order = config.SEVERITY_ORDER
    if a is None:
        return b
    if b is None:
        return a
    return a if order.index(a) >= order.index(b) else b


def detect_equipment(an: Analysis) -> list[dict]:
    out: list[dict] = []
    n = len(an.wide.idx)
    for chn, (atype, direction, _thr) in config.ANOMALY_RULES.items():
        if chn not in an.z:
            continue
        for pid in POINT_IDS:
            if chn in DRIVE_ONLY and pid != DRIVE_ID:
                continue
            zs = an.z[chn][pid].to_numpy(dtype=float)
            valid = np.flatnonzero(np.isfinite(zs))
            raw = an.wide.ch[chn][pid].to_numpy(dtype=float)
            rule_sev = None
            if chn in config.ABSOLUTE_RULES:
                limit, sev = config.ABSOLUTE_RULES[chn]
                tail = raw[np.isfinite(raw)][-3:]
                if len(tail) == 3 and np.all(np.abs(tail) > limit):
                    rule_sev = sev
            fresh = len(valid) >= config.ANOMALY_MIN_VALID and \
                (n - 1 - valid[-1]) * config.SAMPLE_MINUTES / 60.0 <= config.ANOMALY_STALE_HOURS
            if not fresh and rule_sev is None:
                out.append({"point_id": pid, "type": atype, "channel": chn, "state": "unknown"})
                continue
            sev = rule_sev
            value = expected = delta = None
            detail = ""
            if fresh:
                last = valid[-config.ANOMALY_WINDOW:]
                med_z = float(np.median(zs[last]))
                med_d = float(np.median(an.delta[chn][pid].to_numpy(dtype=float)[last]))
                expected = float(np.median(an.expected[chn][pid].to_numpy(dtype=float)[last]))
                value = expected + med_d
                delta = med_d
                signed_z = {1: med_z, -1: -med_z, 0: abs(med_z)}[direction]
                amount = {1: med_d, -1: -med_d, 0: abs(med_d)}[direction]
                if signed_z >= config.ANOMALY_Z:
                    sev = _max_sev(sev, severity_for(chn, amount))
                detail = f"median of last {len(last)} samples: {value:.3g} vs expected {expected:.3g} ({med_z:+.1f} sigma)"
            if sev is None:
                out.append({"point_id": pid, "type": atype, "channel": chn, "state": "normal"})
            else:
                out.append({"point_id": pid, "type": atype, "channel": chn, "state": "active", "severity": sev,
                            "value": value, "expected": expected, "delta": delta,
                            "detail": detail or f"absolute safety rule: |{chn}| > {config.ABSOLUTE_RULES[chn][0]}"})
    return out


def detect_sensor_faults(an: Analysis) -> list[dict]:
    """Stuck and silent sensors, reported separately from equipment anomalies."""
    out: list[dict] = []
    n = len(an.wide.idx)
    for pid in POINT_IDS:
        # silent sensor suite: no data in most of the last samples
        tail = an.wide.ch["vibration"][pid].iloc[-config.OFFLINE_WINDOW:]
        tail_t = an.wide.ch["temperature"][pid].iloc[-config.OFFLINE_WINDOW:]
        offline = (tail.isna().mean() > 0.5) and (tail_t.isna().mean() > 0.5)
        out.append({"point_id": pid, "type": "sensor_offline", "channel": None,
                    "state": "active" if offline else "normal", "severity": "medium",
                    "detail": f"no data in {tail.isna().sum()} of the last {len(tail)} samples"})
        # stuck channels
        for chn in ("vibration", "temperature", "lateral_offset"):
            sm = an.stuck.get((chn, pid))
            if sm is None:
                continue
            raw = an.wide.ch[chn][pid].to_numpy(dtype=float)
            sel = np.flatnonzero(an.wide.running & np.isfinite(raw))
            if len(sel) == 0:
                continue
            last = sel[-1]
            recent = (n - 1 - last) * config.SAMPLE_MINUTES / 60.0 <= 12.0
            if not recent:
                continue
            out.append({"point_id": pid, "type": f"sensor_stuck_{chn}", "channel": chn,
                        "state": "active" if sm[last] else "normal", "severity": "medium",
                        "value": float(raw[last]),
                        "detail": f"{chn} sensor repeats exactly {raw[last]:.3g} (>= {config.STUCK_MIN_SAMPLES} samples)"})
    return out
