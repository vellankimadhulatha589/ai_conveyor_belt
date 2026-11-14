"""Live sensor stream: the simulator keeps going where the stored history ended and POSTs the data
to the backend, faster than real time.

    python -m belt.simulator.live                        # 30 simulated minutes per real second
    python -m belt.simulator.live --minutes-per-second 60
    python -m belt.simulator.live --seconds-per-sample 3     # one new reading per sensor every 3 s

Messy delivery on purpose: rows inside a batch are shuffled (out of order), some are duplicated, some
values are missing/impossible (see generator.py). When a repaired failure falls inside the streamed
period, the simulator also posts the work order (what the maintenance system would send).
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

from .. import config
from .generator import BeltSimulator


def _records(df: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).copy()   # out-of-order delivery
    df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df.astype(object).where(df.notna(), None)
    return df.to_dict(orient="records")


def wait_for_api(api: str, timeout: float = 120.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(api + "/health", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise SystemExit(f"API at {api} did not come up")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=config.API_URL)
    ap.add_argument("--minutes-per-second", type=float, default=30.0)
    ap.add_argument("--seconds-per-sample", type=float, default=0.0,
                    help="slow 'live' mode: send ONE new reading per sensor every N real seconds "
                         "(e.g. 3 -> a new data point for all 26 points every 3 s; 300 = true real time, "
                         "because one sample stands for 5 simulated minutes). Overrides --minutes-per-second.")
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop after this many simulated hours (0 = never)")
    args = ap.parse_args()
    if args.seconds_per_sample > 0:
        loop_seconds = args.seconds_per_sample
        step = config.SAMPLE_MINUTES
        args.minutes_per_second = step / loop_seconds
    else:
        loop_seconds = 1.0
        step = max(config.SAMPLE_MINUTES,
                   int(round(args.minutes_per_second / config.SAMPLE_MINUTES)) * config.SAMPLE_MINUTES)

    sim = BeltSimulator.from_config()
    wait_for_api(args.api)
    summary = requests.get(args.api + "/status/summary", timeout=30).json()
    last = summary.get("last_data_ts")
    cursor = pd.Timestamp(last).to_pydatetime() + timedelta(minutes=config.SAMPLE_MINUTES) if last else sim.demo_now
    started = cursor
    rng = np.random.default_rng(99)
    print(f"live stream starts at simulated time {cursor} (history ended at {last}); "
          f"{args.minutes_per_second:g} simulated min / real second")

    while cursor < sim.end:
        t_loop = time.time()
        nxt = cursor + timedelta(minutes=step)
        ts = sim.timestamps(cursor, nxt)
        df = sim.generate(ts)
        res = requests.post(args.api + "/readings", json={"readings": _records(df, rng)}, timeout=60).json()
        due = sim.maintenance_between(cursor - timedelta(minutes=config.SAMPLE_MINUTES), nxt - timedelta(minutes=config.SAMPLE_MINUTES))
        if due:
            requests.post(args.api + "/maintenance", json={"records": due}, timeout=30)
            for d in due:
                print(f"  work order: {d['mode']} at {d['point_id']} repaired {d['repaired_ts']}")
        print(f"[{ts[0]:%Y-%m-%d %H:%M}] sent {res['received']} rows: inserted={res['inserted']} "
              f"duplicates={res['duplicates']} flagged={res['flagged']} rejected={res['rejected']}")
        cursor = nxt
        if args.max_hours and (cursor - started) >= timedelta(hours=args.max_hours):
            break
        time.sleep(max(0.0, loop_seconds - (time.time() - t_loop)))


if __name__ == "__main__":
    main()