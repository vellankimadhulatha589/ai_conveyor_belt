"""Simulator + ingestion tests (need numpy/pandas; skipped automatically if they are missing)."""
import sys
import unittest
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = pd = None

from belt import config, db  # noqa: E402

if pd is not None:
    from belt.ingest import ingest_frame, ingest_records  # noqa: E402
    from belt.simulator.generator import BeltSimulator  # noqa: E402
    from belt.simulator.layout import POINTS, DRIVE_ID  # noqa: E402


@unittest.skipIf(pd is None, "numpy/pandas not installed")
class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = BeltSimulator(seed=11, history_days=30)

    def test_layout_has_26_points_and_one_drive(self):
        self.assertEqual(len(POINTS), 26)
        self.assertEqual(sum(p.kind == "drive" for p in POINTS), 1)

    def test_deterministic_for_same_seed(self):
        a = BeltSimulator(seed=5, history_days=10)
        b = BeltSimulator(seed=5, history_days=10)
        ts = a.timestamps(a.start, a.start + timedelta(hours=6))
        pd.testing.assert_frame_equal(a.generate(ts), b.generate(ts))

    def test_drive_only_channels_are_empty_elsewhere(self):
        ts = self.sim.timestamps(self.sim.start, self.sim.start + timedelta(hours=3))
        df = self.sim.generate(ts, messy=False)
        other = df[df["point_id"] != DRIVE_ID]
        for ch in config.DRIVE_ONLY_CHANNELS:
            self.assertTrue(other[ch].isna().all(), ch)
        self.assertTrue(df[df["point_id"] == DRIVE_ID]["voltage"].notna().all())

    def test_duty_cycle_belt_stops_six_hours_a_day(self):
        ts = self.sim.timestamps(self.sim.start, self.sim.start + timedelta(days=1))
        df = self.sim.generate(ts, messy=False)
        drive = df[df["point_id"] == DRIVE_ID].set_index("ts")
        stopped = (drive["belt_speed"] < 0.5).mean()
        self.assertTrue(0.2 < stopped < 0.45, stopped)

    def test_messy_mode_injects_garbage_clean_mode_does_not(self):
        ts = self.sim.timestamps(self.sim.start, self.sim.start + timedelta(days=10))
        clean = self.sim.generate(ts, messy=False)
        messy = self.sim.generate(ts, messy=True)
        self.assertFalse(clean["vibration"].isna().any())
        self.assertTrue(messy["vibration"].isna().any())          # missing readings
        self.assertLess(len(messy[~messy.duplicated(["point_id", "ts"])]), len(messy))  # duplicates
        self.assertTrue(((messy["temperature"] < -30) | (messy["temperature"] > 150)).any())  # impossible values

    def test_bearing_fault_vibration_rises_before_temperature(self):
        f = next(x for x in self.sim.faults if x.fault_id == "DEMO-1")
        i_before = f.failure_time - timedelta(hours=124)   # 100-124 h before failure
        ts = self.sim.timestamps(i_before, i_before + timedelta(hours=24))
        df = self.sim.generate(ts, messy=False)
        run = df[df["point_id"] == DRIVE_ID].set_index("ts")["belt_speed"] > 0.5
        p = df[df["point_id"] == f.point_id].set_index("ts")
        p = p[run.reindex(p.index).fillna(False)]
        ts_early = self.sim.timestamps(f.failure_time - timedelta(hours=300), f.failure_time - timedelta(hours=276))
        dfe = self.sim.generate(ts_early, messy=False)
        rune = dfe[dfe["point_id"] == DRIVE_ID].set_index("ts")["belt_speed"] > 0.5
        pe = dfe[dfe["point_id"] == f.point_id].set_index("ts")
        pe = pe[rune.reindex(pe.index).fillna(False)]
        # 100-124 h before failure: vibration already rising, temperature (late indicator) not yet
        self.assertGreater(p["vibration"].mean(), pe["vibration"].mean() + 0.6)
        self.assertLess(abs(p["temperature"].mean() - pe["temperature"].mean()), 4.0)

    def test_maintenance_orders_follow_failures(self):
        for wo in self.sim.maintenance_between(self.sim.start, self.sim.end):
            self.assertLess(wo["failure_time"], wo["repaired_ts"])


@unittest.skipIf(pd is None, "numpy/pandas not installed")
class IngestTests(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")
        db.init_db(self.conn)
        db.upsert_points(self.conn, POINTS)

    def _row(self, pid="P01", ts="2026-01-01T10:00:00", **kw):
        r = {"point_id": pid, "ts": ts, "vibration": 2.5, "temperature": 45.0, "lateral_offset": 1.0, "ambient_temp": 20.0}
        r.update(kw)
        return r

    def test_stores_and_is_idempotent(self):
        r1 = ingest_records(self.conn, [self._row()])
        r2 = ingest_records(self.conn, [self._row()])
        self.assertEqual((r1["inserted"], r2["inserted"], r2["duplicates"]), (1, 0, 1))

    def test_unknown_point_and_bad_timestamp_are_rejected(self):
        r = ingest_records(self.conn, [self._row(pid="P99"), self._row(ts="not-a-date"), self._row()])
        self.assertEqual((r["rejected"], r["inserted"]), (2, 1))

    def test_impossible_and_non_numeric_values_are_flagged_but_kept(self):
        r = ingest_records(self.conn, [self._row(temperature=999.0, vibration="abc")])
        self.assertEqual(r["flagged"], 1)
        row = self.conn.execute("SELECT * FROM readings").fetchone()
        self.assertEqual(row["temperature"], 999.0)           # raw value preserved
        self.assertIn("range:temperature;", row["flags"])
        self.assertIn("bad_type:vibration;", row["flags"])
        self.assertIsNone(row["vibration"])

    def test_missing_values_become_null_and_out_of_order_is_fine(self):
        ingest_records(self.conn, [self._row(ts="2026-01-01T10:05:00", vibration=None),
                                   self._row(ts="2026-01-01T10:00:00")])
        rows = self.conn.execute("SELECT ts, vibration FROM readings ORDER BY ts").fetchall()
        self.assertEqual([r["ts"] for r in rows], ["2026-01-01T10:00:00", "2026-01-01T10:05:00"])
        self.assertIsNone(rows[1]["vibration"])

    def test_timezone_aware_timestamps_are_normalised_to_utc(self):
        ingest_records(self.conn, [self._row(ts="2026-01-01T12:00:00+02:00")])
        self.assertEqual(self.conn.execute("SELECT ts FROM readings").fetchone()["ts"], "2026-01-01T10:00:00")

    def test_frame_ingest_matches_record_ingest(self):
        df = pd.DataFrame([self._row(), self._row(pid="P02")])
        self.assertEqual(ingest_frame(self.conn, df)["inserted"], 2)


if __name__ == "__main__":
    unittest.main()
