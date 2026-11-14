"""Central configuration: paths, belt geometry, channel definitions, thresholds.

Everything that is a *tunable assumption* lives here so that the design document
can point at one file.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BELT_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.environ.get("BELT_DB_PATH", DATA_DIR / "belt.db"))
MODELS_DIR = DATA_DIR / "models"
SIM_CONFIG_PATH = DATA_DIR / "sim_config.json"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"  # evaluation only, never read by the backend
VISION_DATASET_DIR = Path(os.environ.get("BELT_VISION_DATASET", DATA_DIR / "vision_dataset"))
FRAMES_DIR = DATA_DIR / "vision_frames"
VISION_MODEL_PATH = MODELS_DIR / "vision_model.joblib"
PREDICTOR_PATH = MODELS_DIR / "predictor.joblib"
API_URL = os.environ.get("BELT_API_URL", "http://127.0.0.1:8000")

# --------------------------------------------------------------------------- belt geometry
ROUTE_LENGTH_M = 5000.0            # head (drive) to tail pulley
LOOP_LENGTH_M = 2 * ROUTE_LENGTH_M  # the belt is a closed loop: carry side + return side
POINT_SPACING_M = 200.0            # one sensor suite every 200 m -> 26 points
CURVE_KM = (1.8, 3.1)              # horizontal curves (higher mistracking risk)
CURVE_RADIUS_KM = 0.25
NOMINAL_SPEED_MS = 4.0
SAMPLE_MINUTES = 5                 # simulated sampling period of every sensor suite

# --------------------------------------------------------------------------- channels
IDLER_CHANNELS = ["vibration", "temperature", "lateral_offset", "ambient_temp"]
DRIVE_ONLY_CHANNELS = ["belt_speed", "drive_speed", "belt_load", "motor_current", "voltage"]
ALL_CHANNELS = IDLER_CHANNELS + DRIVE_ONLY_CHANNELS

# Hard physical limits: anything outside is an impossible reading (sensor fault),
# it is stored (raw data is never thrown away) but flagged and ignored by analysis.
LIMITS = {
    "vibration": (0.0, 60.0),
    "temperature": (-30.0, 150.0),
    "lateral_offset": (-400.0, 400.0),
    "ambient_temp": (-40.0, 60.0),
    "belt_speed": (0.0, 8.0),
    "drive_speed": (0.0, 8.0),
    "belt_load": (0.0, 8000.0),
    "motor_current": (0.0, 1200.0),
    "voltage": (300.0, 900.0),
}

UNITS = {
    "vibration": "mm/s",
    "temperature": "degC",
    "lateral_offset": "mm",
    "ambient_temp": "degC",
    "belt_speed": "m/s",
    "drive_speed": "m/s",
    "belt_load": "t/h",
    "motor_current": "A",
    "voltage": "V",
    "slip": "fraction",
}

# --------------------------------------------------------------------------- analysis
RUNNING_SPEED_THRESHOLD = 0.5      # m/s on the drive station => belt is running
STUCK_MIN_SAMPLES = 18             # identical consecutive running samples => stuck sensor (1.5 h)
ANOMALY_WINDOW = 12                # last 12 running samples (~1 h) judged by their median
ANOMALY_MIN_VALID = 8
ANOMALY_Z = 4.0                    # robust z-score a channel must exceed (together with MIN_ABS)
ANOMALY_STALE_HOURS = 3.0          # newest valid sample must be this recent to judge a channel
OFFLINE_WINDOW = 12                # samples; > 50 % missing => sensor offline

# channel -> (anomaly type, direction (+1 high is bad, -1 low is bad, 0 both), severity thresholds
# on |observed - expected| for low / medium / high; the 4th level "critical" is 1.5 x the 3rd)
ANOMALY_RULES = {
    "vibration":      ("vibration_high",   +1, (1.5, 3.5, 7.0)),
    "temperature":    ("overheating",      +1, (6.0, 15.0, 30.0)),
    "lateral_offset": ("belt_drift",        0, (40.0, 80.0, 120.0)),
    "slip":           ("belt_slip",        +1, (0.01, 0.03, 0.05)),
    "motor_current":  ("motor_overcurrent", +1, (40.0, 90.0, 130.0)),
    "voltage":        ("undervoltage",     -1, (25.0, 55.0, 85.0)),
    "belt_speed":     ("speed_drop",       -1, (0.2, 0.5, 1.0)),
}
# absolute safety rules independent of the learned baseline: channel -> (limit, severity)
ABSOLUTE_RULES = {"temperature": (85.0, "critical"), "lateral_offset": (140.0, "high")}

SEVERITY_ORDER = ["low", "medium", "high", "critical"]

# --------------------------------------------------------------------------- prediction
FAILURE_MODES = ["bearing", "mistrack", "slip", "overload", "undervoltage"]
CLASSES = ["none"] + FAILURE_MODES
# How long before the failure the degradation signature is physically observable.
# Used for labelling training snapshots (documented assumption, see docs/DESIGN.md).
PREDICTION_WINDOW_H = {"bearing": 168, "mistrack": 72, "slip": 96, "overload": 12, "undervoltage": 120}
SNAPSHOT_STEP_H = 3
P_WATCH = 0.30                     # probability that makes a prediction visible as "watch"
P_PREDICTED_FAILURE = 0.50         # probability that marks a point as "predicted failure"
FEATURE_WINDOWS_H = (6, 24, 72)
ANALYSIS_INTERVAL_S = 8            # background analysis cadence of the API
ANALYSIS_LOOKBACK_H = 100          # hours of data loaded for each analysis cycle

# --------------------------------------------------------------------------- vision
SEGMENT_M = 10.0                   # belt is divided into 10 m segments for defect bookkeeping
N_SEGMENTS = int(LOOP_LENGTH_M / SEGMENT_M)
CAMERA_FRAME_M = 2.0               # belt length visible in one frame
MERGE_RADIUS_SEGMENTS = 1          # detections within +-1 segment belong to the same defect
CONFIRM_LAPS = 2                   # a defect needs sightings in this many *different* laps
DISMISS_AFTER_MISSES = 5           # unconfirmed track dismissed after this many clean laps
DETECTION_THRESHOLD = 0.5
KNOWN_SPLICES_M = (3333.0, 7777.0)  # vulcanised splices: normal belt features, not defects
SPLICE_MASK_M = 3.0          # physical half-width of the splice stripe
POSITION_ERROR_M = 3.0       # margin for encoder noise (~1 m sigma): the tracker masks width + this
NORMAL_CLASS_NAMES = {"normal", "good", "ok", "healthy", "no_defect", "normal condition", "normal_condition"}