"""Tests for the loop-aware defect tracker. Standard library only: run with

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from belt import config, db  # noqa: E402
from belt.vision import tracker  # noqa: E402


def make_conn():
    conn = db.connect(":memory:")
    db.init_db(conn)
    return conn


_n = 0


def obs(lap, pos, label="tear", score=0.9, sev=0.5, ok=True):
    global _n
    _n += 1
    return {"frame_id": f"f{_n}", "lap": lap, "position_m": pos, "label": label, "score": score,
            "severity": sev, "quality_ok": ok}


def clean(lap, pos):
    return obs(lap, pos, label="normal", score=0.05, sev=0.0)


class TrackerTests(unittest.TestCase):
    def test_same_defect_every_lap_is_one_track(self):
        conn = make_conn()
        for lap in range(1, 6):
            tracker.record_observation(conn, obs(lap, 1234.0 + lap))  # +-1 m of positional noise
        tracks = tracker.list_tracks(conn)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["status"], "confirmed")
        self.assertEqual(tracks[0]["laps_seen"], 5)
        self.assertEqual(tracks[0]["n_defect_obs"], 5)

    def test_first_sighting_is_only_suspected(self):
        conn = make_conn()
        r = tracker.record_observation(conn, obs(1, 500.0))
        self.assertEqual(r["outcome"], "new_track")
        self.assertEqual(tracker.list_tracks(conn)[0]["status"], "suspected")

    def test_two_sightings_in_same_lap_do_not_confirm(self):
        conn = make_conn()
        tracker.record_observation(conn, obs(1, 500.0))
        tracker.record_observation(conn, obs(1, 501.0))
        t = tracker.list_tracks(conn)[0]
        self.assertEqual(t["laps_seen"], 1)
        self.assertEqual(t["status"], "suspected")

    def test_splice_is_ignored(self):
        conn = make_conn()
        r = tracker.record_observation(conn, obs(1, config.KNOWN_SPLICES_M[0] + 1.0))
        self.assertEqual(r["outcome"], "ignored_splice")
        self.assertEqual(tracker.list_tracks(conn), [])

    def test_splice_mask_covers_encoder_error(self):
        # a frame that shows the stripe (<= 3 m) can be reported up to a few metres further out
        conn = make_conn()
        s = config.KNOWN_SPLICES_M[0]
        edge = tracker.record_observation(conn, obs(1, s + config.SPLICE_MASK_M + 1.5))
        self.assertEqual(edge["outcome"], "ignored_splice")
        far = tracker.record_observation(conn, obs(1, s + config.SPLICE_MASK_M + config.POSITION_ERROR_M + 2.0))
        self.assertEqual(far["outcome"], "new_track")

    def test_bad_frame_is_not_evidence(self):
        conn = make_conn()
        r = tracker.record_observation(conn, obs(1, 800.0, ok=False))
        self.assertEqual(r["outcome"], "low_quality")
        self.assertEqual(tracker.list_tracks(conn), [])

    def test_transient_false_positive_is_dismissed(self):
        conn = make_conn()
        tracker.record_observation(conn, obs(1, 900.0))
        for lap in range(2, 2 + config.DISMISS_AFTER_MISSES):
            tracker.record_observation(conn, clean(lap, 900.0))
        self.assertEqual(tracker.list_tracks(conn), [])
        self.assertEqual(tracker.summary(conn)["tracks"].get("dismissed"), 1)

    def test_confirmed_defect_survives_missed_laps(self):
        conn = make_conn()
        tracker.record_observation(conn, obs(1, 900.0))
        tracker.record_observation(conn, obs(2, 900.0))
        for lap in range(3, 15):
            tracker.record_observation(conn, clean(lap, 900.0))
        t = tracker.list_tracks(conn)[0]
        self.assertEqual(t["status"], "confirmed")
        self.assertGreaterEqual(t["consecutive_misses"], 10)

    def test_neighbouring_segments_merge_but_distant_do_not(self):
        conn = make_conn()
        tracker.record_observation(conn, obs(1, 1000.0))
        tracker.record_observation(conn, obs(2, 1000.0 + config.SEGMENT_M))        # one segment away: same defect
        tracker.record_observation(conn, obs(2, 1000.0 + 5 * config.SEGMENT_M))    # five segments away: new defect
        self.assertEqual(len(tracker.list_tracks(conn)), 2)

    def test_wrap_around_the_loop_seam(self):
        conn = make_conn()
        tracker.record_observation(conn, obs(1, config.LOOP_LENGTH_M - 4.0))
        tracker.record_observation(conn, obs(2, 3.0))
        self.assertEqual(len(tracker.list_tracks(conn)), 1)

    def test_frame_ids_are_idempotent(self):
        conn = make_conn()
        o = obs(1, 500.0)
        tracker.record_observation(conn, o)
        self.assertEqual(tracker.record_observation(conn, o)["outcome"], "duplicate")
        self.assertEqual(tracker.summary(conn)["frames"], 1)

    def test_low_score_detection_is_treated_as_clean(self):
        conn = make_conn()
        r = tracker.record_observation(conn, obs(1, 500.0, score=0.2))
        self.assertEqual(r["outcome"], "clean")

    def test_label_vote_and_growth_history(self):
        conn = make_conn()
        tracker.record_observation(conn, obs(1, 700.0, label="wear", score=0.6, sev=0.2))
        tracker.record_observation(conn, obs(2, 700.0, label="tear", score=0.95, sev=0.5))
        tracker.record_observation(conn, obs(3, 700.0, label="tear", score=0.9, sev=0.7))
        t = tracker.list_tracks(conn)[0]
        self.assertEqual(t["label"], "tear")
        self.assertEqual([h[0] for h in t["severity_history"]], [1, 2, 3])
        self.assertAlmostEqual(t["max_severity"], 0.7)
        self.assertEqual(t["severity_level"], "high")


if __name__ == "__main__":
    unittest.main()