"""Analysis building blocks (need numpy/pandas/scikit-learn; skipped if missing).

The slow end-to-end test (60 simulated days, trains the model, checks that a bearing fault is
announced before it happens) only runs with  RUN_SLOW=1 python -m unittest tests.test_analysis -v
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    import pandas as pd
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover
    np = pd = None

from belt import config  # noqa: E402

if pd is not None:
    from belt.analysis import anomaly, features, prepare  # noqa: E402


@unittest.skipIf(pd is None, "numpy/pandas/scikit-learn not installed")
class BuildingBlockTests(unittest.TestCase):
    def test_stuck_mask_detects_exact_repetition_only_while_running(self):
        x = np.concatenate([np.random.default_rng(0).normal(5, 1, 50), np.full(30, 3.3), np.random.default_rng(1).normal(5, 1, 20)])
        running = np.ones(len(x), dtype=bool)
        m = prepare.stuck_mask(x, running)
        self.assertTrue(m[50:80].all())
        self.assertFalse(m[:50].any() or m[80:].any())
        running[50:80] = False                      # belt stopped: constant values are expected
        self.assertFalse(prepare.stuck_mask(x, running).any())

    def test_trimmed_fit_recovers_baseline_despite_outliers(self):
        rng = np.random.default_rng(2)
        load = rng.uniform(1000, 4000, 3000)
        y = 1.8 + 0.0003 * load + rng.normal(0, 0.2, 3000)
        y[:300] += 10.0                              # 10 % faulty samples
        fit = prepare._fit_line(load[:, None], y)
        self.assertAlmostEqual(fit["coef"][0], 0.0003, delta=0.00005)
        self.assertAlmostEqual(fit["sigma"], 0.2, delta=0.06)

    def test_severity_levels(self):
        self.assertIsNone(anomaly.severity_for("vibration", 0.5))
        self.assertEqual(anomaly.severity_for("vibration", 2.0), "low")
        self.assertEqual(anomaly.severity_for("vibration", 8.0), "high")
        self.assertEqual(anomaly.severity_for("vibration", 20.0), "critical")

    def test_features_ignore_history_before_a_repair(self):
        idx = pd.date_range("2026-01-01", periods=200, freq="1h")
        bad = pd.DataFrame(10.0, index=idx, columns=[p for p in prepare.POINT_IDS])  # old part: very bad
        bad.iloc[100:] = 0.0                                                          # new part from hour 100
        hourly = {"vibration": bad}
        snap = pd.DatetimeIndex([idx[150]])
        with_reset = features.build_features(hourly, snap, {"P03": [idx[100]]})
        without = features.build_features(hourly, snap, {})
        row_r = with_reset[with_reset["point_id"] == "P03"].iloc[0]
        row_n = without[without["point_id"] == "P03"].iloc[0]
        self.assertAlmostEqual(row_r["vibration_mean72"], 0.0)
        self.assertGreater(row_n["vibration_mean72"], 1.0)

    def test_labels_mark_window_before_failure_and_skip_downtime(self):
        ts = pd.to_datetime(["2026-01-10 00:00", "2026-01-18 00:00", "2026-01-19 03:00", "2026-01-19 20:00"])
        feat = pd.DataFrame({"point_id": ["P05"] * 4, "ts": ts})
        fail = [{"point_id": "P05", "mode": "bearing", "failure_time": "2026-01-19 00:00:00",
                 "repaired_ts": "2026-01-19 06:00:00"}]
        lab = features.label_snapshots(feat, fail)
        self.assertEqual(list(lab["y"]), [0, config.CLASSES.index("bearing"), 0, 0])
        self.assertEqual(list(lab["valid"]), [True, True, False, True])
        self.assertAlmostEqual(lab["tth"].iloc[1], 24.0)


@unittest.skipUnless(os.environ.get("RUN_SLOW") and pd is not None, "set RUN_SLOW=1 to run the end-to-end test")
class EndToEndTest(unittest.TestCase):
    def test_bearing_failure_is_predicted_before_it_happens(self):
        import tempfile
        from datetime import timedelta

        from belt import db
        from belt.analysis.pipeline import run_analysis
        from belt.analysis.train import train_all
        from belt.ingest import add_maintenance, ingest_frame
        from belt.simulator.generator import BeltSimulator
        from belt.simulator.layout import POINTS

        tmp = Path(tempfile.mkdtemp())
        config.MODELS_DIR = tmp
        config.PREDICTOR_PATH = tmp / "predictor.joblib"
        import belt.analysis.pipeline as pipe_mod
        pipe_mod._model_cache.update(mtime=None, obj=None)

        sim = BeltSimulator(seed=3, history_days=60)
        conn = db.connect(tmp / "t.db")
        db.init_db(conn)
        db.upsert_points(conn, POINTS)
        t = sim.start
        while t < sim.demo_now:
            t1 = min(t + timedelta(days=5), sim.demo_now)
            ingest_frame(conn, sim.generate(sim.timestamps(t, t1)))
            t = t1
        add_maintenance(conn, sim.maintenance_between(sim.start, sim.demo_now - timedelta(minutes=5)))
        metrics = train_all(conn, verbose=False)
        self.assertGreater(metrics["positive_snapshots"], 20)
        res = run_analysis(conn, force=True)
        self.assertEqual(res["status"], "ok")
        # DEMO-1 (bearing at P12) fails 60 h after the history ends and is already degrading
        row = conn.execute("SELECT * FROM point_status WHERE point_id='P12'").fetchone()
        self.assertIn(row["status"], ("abnormal", "predicted_failure"))


if __name__ == "__main__":
    unittest.main()
