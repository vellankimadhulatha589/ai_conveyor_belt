"""One-time set-up: simulate history, store it, train the models, run a first analysis.

    python scripts/bootstrap.py                 # 120 simulated days, ~1-3 minutes
    python scripts/bootstrap.py --days 60       # smaller / faster

Steps
 1. simulator -> 26 sensor suites x N days of 5-minute readings (messy on purpose),
    delivered in shuffled chunks through the same ingestion code the API uses
 2. work orders of the failures that were repaired during the history (labels for the model)
 3. baseline + failure predictor training with a time-based hold-out evaluation
 4. first analysis cycle so the dashboard has something to show immediately
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from belt import config  # noqa: E402
from belt.analysis.pipeline import run_analysis  # noqa: E402
from belt.analysis.train import train_all  # noqa: E402
from belt.db import connect, init_db, reset_db, upsert_points  # noqa: E402
from belt.ingest import add_maintenance, ingest_frame  # noqa: E402
from belt.simulator.generator import BeltSimulator  # noqa: E402
from belt.simulator.layout import POINTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120, help="simulated history length")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep-db", action="store_true", help="do not delete an existing database")
    args = ap.parse_args()

    t_start = time.time()
    sim = BeltSimulator(seed=args.seed, history_days=args.days)
    sim.save_config()
    sim.save_ground_truth()
    print(f"simulator: seed={args.seed}, {args.days} days from {sim.start:%Y-%m-%d}, "
          f"{len(sim.faults)} scheduled equipment faults, {len(sim.sensor_faults)} sensor faults/outages")

    if not args.keep_db:
        reset_db()
    conn = connect()
    init_db(conn)
    upsert_points(conn, POINTS)

    print("generating and ingesting history ...")
    totals = {"received": 0, "inserted": 0, "duplicates": 0, "rejected": 0, "flagged": 0}
    chunk = timedelta(days=5)
    t = sim.start
    while t < sim.demo_now:
        t1 = min(t + chunk, sim.demo_now)
        df = sim.generate(sim.timestamps(t, t1))
        df = df.sample(frac=1.0, random_state=int(t.timestamp()) % (2 ** 31))  # out-of-order delivery
        res = ingest_frame(conn, df)
        for k in totals:
            totals[k] += res[k]
        print(f"  {t:%Y-%m-%d} .. {t1:%Y-%m-%d}: {res['inserted']} rows stored")
        t = t1
    print(f"ingest totals: {totals}")

    last_hist = sim.demo_now - timedelta(minutes=config.SAMPLE_MINUTES)
    wo = sim.maintenance_between(sim.start, last_hist)
    print(f"storing {add_maintenance(conn, wo)} work orders (repaired failures in the history)")

    print("\ntraining models ...")
    train_all(conn)

    print("\nfirst analysis cycle ...")
    print(run_analysis(conn, force=True))
    n_pf = conn.execute("SELECT COUNT(*) FROM point_status WHERE status='predicted_failure'").fetchone()[0]
    n_ab = conn.execute("SELECT COUNT(*) FROM point_status WHERE status='abnormal'").fetchone()[0]
    n_an = conn.execute("SELECT COUNT(*) FROM anomalies WHERE status='active'").fetchone()[0]
    print(f"status: {n_pf} point(s) predicted failure, {n_ab} abnormal, {n_an} active anomalies")
    conn.close()
    print(f"\nbootstrap finished in {time.time() - t_start:.0f}s. Database: {config.DB_PATH}")
    print("Next:\n  python -m belt.vision.train            (or add --synthetic for the offline stand-in)"
          "\n  python scripts/start_all.py            (API + live stream + dashboard)")


if __name__ == "__main__":
    main()
