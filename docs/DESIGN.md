# Design document

> What we monitor, how, and why. Every number quoted here that is a *tunable assumption* lives in
> [`belt/config.py`](../belt/config.py) or [`belt/simulator/faults.py`](../belt/simulator/faults.py).

## 1. Problem and scope

A 5 km mining conveyor (10 km of belt, because it is a loop) runs 18 h/day in dust, vibration and
temperature swings. When it stops, the mine stops. The system has to turn sensor streams and
camera images into **a picture of belt health and early warning of failures**.

Scope decisions (built vs. skipped are listed in section 10):

* Five failure modes are modelled end to end. The others are named, argued about and skipped.
* One sensor suite every 200 m (26 points) plus a camera looking at the belt surface.
* "Predict" means: for each sensor point, *which* failure mode, *how likely*, and *when* (with an
  uncertainty band) - not a generic health score.

## 2. What can go wrong (research summary)

Searched sources are linked in the README. Failure modes of a mining belt conveyor, grouped by
where they happen:

| Area | Failure | Typical causes | First symptom we can measure |
|---|---|---|---|
| Idlers / rollers | bearing wear, seizure, shell wear-through | contamination, lubrication loss, overload, dust | vibration up, **later** temperature up |
| Belt | longitudinal tear, transverse tear, cover wear/gouges, splice failure | sharp rock in chute, seized idler shaft, fatigue, impact | **vision**; sometimes sudden speed/current events |
| Tracking | belt wanders sideways (mistracking) | off-centre load, misaligned structure, material build-up, wind, bad splice | lateral offset drifts |
| Drive / pulley | slip, overload/blockage, bearing failure, worn lagging | low tension, wet pulley, jammed chute, seized rollers | drive vs belt speed gap, motor current, pulley temperature |
| Electrical | under/over-voltage | weak grid, long cable run | voltage, current up and speed down |
| Material handling | blocked chute, spillage, carry-back | overload, worn skirts | motor current, tracking |
| Safety | fire / overheating | seized roller, friction | temperature |

**Failures cascade** (a bad splice -> carry-back -> build-up on idlers -> mistracking -> edge damage ->
more spillage). This is why the dashboard shows *all* active anomalies, not just the worst one, and why
the model gets per-point features of several channels at once.

**Symptoms are not failures.** "Temperature up", "vibration up", "speed down", "voltage low", "load
high", "misaligned" are *signals*. The system measures signals, detects abnormal ones (anomalies) and
infers likely root causes (predictions). The same symptom can have several causes - e.g. speed down
could be slip, overload or low voltage - which is why the *combination* of channels matters:

| Pattern | Most likely cause |
|---|---|
| vibration up, then temperature up at one idler | bearing degradation |
| speed gap (drive vs belt) + drive pulley temperature up | belt slip |
| speed down + current up (voltage normal) | overload / blocked chute |
| speed down + current up + **voltage down** | electrical supply problem |
| speed up, current down, load down | empty belt (normal) |
| offset drifting, vibration slightly up | mistracking |
| flat-lined or impossible values on one channel | **sensor fault, not equipment** |

## 3. What we monitor, and why

| Channel | Where | Why |
|---|---|---|
| `vibration` (mm/s) | every point (idler frame) | **early** indicator of bearing/roller faults |
| `temperature` (degC) | every point (bearing) | **late** indicator, fire safety, drive friction |
| `lateral_offset` (mm) | every point (belt edge) | mistracking, edge damage |
| `ambient_temp` (degC) | every point | context: separates a hot afternoon from a hot bearing |
| `belt_speed`, `drive_speed` (m/s) | drive station | slip = difference of the two; also tells us whether the belt is running |
| `belt_load` (t/h) | drive station (belt scale) | context: normal vibration/current/temperature depend on load |
| `motor_current` (A), `voltage` (V) | drive station | overload, electrical problems |
| camera frames | one camera on the loop | surface damage: tears, gouges, wear |

Placement: every 200 m along 5 km (26 points), the head station is the *drive* point with the
drive-only channels, the last point is the tail pulley. Two horizontal curves are marked
(km 1.8 and 3.1): mistracking is 3x more likely there and offset noise is higher. The terrain profile
(incline, steep section) is synthetic and used for the belt map.

**Context matters.** A vibration of 3.5 mm/s is normal at peak load and alarming at half load. So the
analysis never compares raw values to fixed limits (except a few safety rules); it compares them to the
*expected value for the current load and ambient temperature* (section 6).

## 4. Simulator (the only source of sensor data)

`belt/simulator/` - `generator.py` produces, for any set of timestamps, one row per sensor point.
Deterministic for a given `(seed, start, history_days)`, so the live stream continues exactly where the
history ended.

**Healthy behaviour**

* 18 h/day duty cycle: stopped 04:00-10:00. Stops also happen after failures (the belt trips and
  stays down for the repair time).
* Load: slow and fast oscillations + noise, around 3000 t/h.
* Vibration = 1.8 + 0.0003 * load + per-point bias + noise. Bearing temperature = ambient + 14 + 0.0025 *
  load + bias + noise. Ambient follows a daily and a weekly cycle and the terrain elevation.
* Lateral offset: noise (larger in curves) + slow load-imbalance wobble.
* Motor current ~ 80 + 0.075 * load; voltage 690 V +- noise; slip about 0.3 %.

**Fault curves** (`tau` = hours before the failure that trips the belt, `L` = lead time):

| Mode | Where | Lead L | What changes and how |
|---|---|---|---|
| bearing | any non-drive point | ~240 h (+-30 %) | vibration += 12*exp(-tau/(0.2 L)) (early, accelerating); temperature += 52*s^2 only in the last 0.4 L (late); drive current +2 % at the end |
| mistrack | any non-drive point, 3x weight in curves | ~72 h | offset drifts +-160 mm * s^1.5 (s = 1 - tau/L); vibration up slightly |
| slip | drive | ~96 h | slip only under peak load at first (threshold falls over time), then continuous; drive temperature up; current +3 % |
| overload | drive | ~8 h | current +45 %, speed -10 %, drive temperature +28 |
| undervoltage | drive | ~120 h | voltage sags (deeper at peak load, up to 14 %), current +50 % of sag, speed -30 % of sag |

Correlation is built in where physics says so (a late-stage bearing raises motor current; mistracking
raises vibration nearby). **Four faults are pinned to be "in progress" when the live demo starts** so the
dashboard has something to show immediately; the rest of the schedule is random (Poisson rates in
`faults.py`).

**Messy data** (graded in the brief, and a large part of the backend's job):

| Problem | Simulated as | Handling |
|---|---|---|
| missing readings | 1.2 % random NaN | stored as NULL, ignored, no alarm |
| sensor outages | all channels of a point silent for 1-12 h (rows absent) | wide table is re-gridded to 5 min, `sensor_offline` anomaly if > 50 % missing |
| stuck sensor | channel repeats one constant for 8-30 h | exact repetition >= 1.5 h while running -> `sensor_stuck_*` anomaly, samples excluded |
| spikes | 0.08 % of values x4-9 | median over ~1 h window cannot be moved by a spike |
| impossible values | 0.04 % (-99 degC, 999 mm ...) | outside physical limits -> flagged on ingest, stored raw, ignored |
| calibration drift | 2 temperature sensors drift +0.03 degC/day | **not handled** (documented) |
| duplicated messages | 0.3 % of rows | primary key (point, ts) makes ingest idempotent, counted |
| out-of-order arrival | rows shuffled per batch | sorted by key, harmless |

The answer key (`data/ground_truth.json`) is written for evaluation only; the backend never reads it.

## 5. Backend

FastAPI + SQLite (standard library `sqlite3`, WAL mode). Why: zero set-up for a reviewer, one file, and
~0.9 M readings for 120 days is far below SQLite's limits; all SQL sits in `belt/db.py`, `belt/ingest.py`
and the analysis modules, so PostgreSQL/TimescaleDB would be a contained change.

```
simulator -> POST /readings -> validate / flag / de-duplicate -> readings table
                                          |
                    background thread (every ~8 s if new data) -> analysis cycle
                                          |
              anomalies, predictions, point_status tables  -> GET endpoints -> dashboard
camera frames -> vision detector -> POST /vision/observations -> tracker -> vision_tracks table
```

Tables: `points`, `readings` (raw, flags column), `maintenance_log` (work orders), `anomalies`,
`predictions`, `point_status`, `vision_observations` (every frame, audit trail), `vision_tracks`
(de-duplicated defects), `meta` (baseline, metrics, counters).

Ingestion rules: unparsable timestamp / unknown point -> rejected and counted; non-numeric or impossible
value -> kept raw, flagged, ignored by analysis; missing -> NULL; duplicates ignored and counted. The
endpoint accepts a lenient JSON body on purpose - validation happens per row so one bad row never
rejects a batch.

"Now" is the newest data timestamp, not the wall clock (the simulator runs faster than real time and a
delayed feed should not look like a healthy belt).

## 6. Analysis: anomalies

1. Readings -> wide tables on a regular 5-minute grid; flagged values become NaN.
2. **Running mask**: belt is running if the lower of belt/drive speed is > 0.5 m/s (the lower one so a
   stuck-high sensor cannot fake a running belt); fallback = median vibration of all points.
3. **Learned baseline** per channel and point: iteratively trimmed least squares on running samples,
   e.g. `vibration ~ load`, `temperature ~ load + ambient`. Faults in the training history are a minority
   and are trimmed, so the baseline describes healthy behaviour. Residual scale = robust sigma (MAD).
4. **Anomaly** for (point, type) = median of the last ~1 h of running samples deviates in the bad
   direction by `>= 4 robust sigma` **and** by at least the channel's first severity threshold
   (statistically unusual *and* practically relevant). Severity low/medium/high/critical from the amount
   (e.g. temperature +6/+15/+30 degC). A few absolute safety rules (temperature > 85 degC) bypass the baseline.
5. Each check returns `active`, `normal` or `unknown` (not enough fresh data). Only `normal` resolves an
   open anomaly, so a belt that stopped for repair does not silently "heal" its alarms. A completed repair
   (work order) resolves earlier equipment anomalies and cuts the point's history.
6. **Sensor faults are separate anomaly types** (`sensor_stuck_*`, `sensor_offline`) and are removed
   from the data before equipment checks - a flat-lined thermometer is not an overheating bearing.

## 7. Prediction

**Target.** For every sensor point at time t: which failure mode (if any) will happen, how likely, and when.
A snapshot is positive for mode M if a failure of M at that point follows within M's *observable window*
(bearing 168 h, mistrack 72 h, slip 96 h, overload 12 h, undervoltage 120 h) - the period in which the
degradation signature physically exists. This is an assumption (documented in `config.py`); in reality
the window would come from reliability engineers and maintenance history. Labels come from the
**work-order log** (`maintenance_log`), like a real CMMS, not from the simulator's hidden schedule.

**Features** (57): for each of 7 channels (vibration, temperature, lateral offset, motor current, voltage,
slip, belt speed) the residual in the bad direction, summarised over 6/24/72 h as mean and max, plus 24 h
and 72 h trend. Because residuals remove load, ambient and per-point offsets, **one model serves all
points**. History is cut at the last repair (maintenance reset): otherwise the old part's degradation
would haunt the new one.

**Models.** scikit-learn `HistGradientBoosting`: a multi-class classifier (none + 5 modes) for "what/how likely"
and three quantile regressors (20/50/80 %) for "when" -> ETA with a band. Reasons: handles missing values
natively (belt stopped, sensor silent), no scaling, fast on CPU, robust for small imbalanced tabular
data; and it is explainable enough - the evidence column shows the channel residuals that drove the call.

**Evaluation (honest protocol).** Time-based split (70/30 with a 7-day purge gap), never random:
neighbouring snapshots of one failure are almost identical, so shuffling would leak. The last 10 days are
excluded because faults still in progress have no work order yet (their snapshots would be mislabelled).
Reported: precision/recall of the alert, ROC-AUC, **fault-level recall and lead time** (what an operator
cares about), false-alarm episodes per 100 point-days (early warnings inside a 12-day grace period before a
failure are not counted as false alarms), ETA error and the coverage of the 20-80 % band. A naive rule
("any channel's 6 h mean >= 4 sigma") is scored the same way as a reference, to show what ML adds.
Numbers are written to `data/models/metrics.json` and shown on the dashboard.

**Limitations.** Trained and tested on simulated data: it proves the pipeline, not real-world accuracy.
Drive-station modes have few examples (one drive, ~10 faults). The probability is not calibrated.
Predictions are re-computed each cycle without smoothing, so a point can flicker around the threshold.

## 8. Vision

**Dataset.** See the README ("Vision dataset") - the source and the reasons for the choice are documented
there because it is the part you have to confirm and fill in. The pipeline needs images grouped into a
*normal* class and one or more *defect* classes.

**Model.** Pillow + numpy features (colour stats, grey and gradient histograms, local-binary-pattern
texture, a 6x6 grid of intensity/gradient) -> RandomForest. Why not a CNN: runs on a laptop CPU with no
PyTorch/GPU, public belt datasets are small, belt damage is mostly an abrupt texture/intensity change on a
uniform surface, and the `DefectDetector` interface lets a CNN replace it later. Severity is a heuristic
(half model confidence, half the share of the frame deviating from the healthy reference); the bounding
box is the union of deviating grid cells.

### The loop problem

The belt is a closed loop, so the camera sees *the same stretch of belt again and again*. Consequences and
what the system does about each:

| # | Consequence of the loop | Why it breaks naive image-wear detection | Handling | Status |
|---|---|---|---|---|
| 1 | Same defect is photographed every lap | one 3 m tear becomes 50 "defects" a day; counts and alarms explode | defects are identified by **position on the loop** (10 m segments, merge radius +-1), not by time. One physical defect = one *track* | solved |
| 2 | A frame has no inherent location | "image at 14:03" does not say which belt metres it shows | every frame carries `lap` and `position_m` (encoder); `position mod loop length` -> segment | solved in simulation, encoder drift not simulated |
| 3 | Position uncertainty (encoder noise, slip, stretch) | the same defect lands in neighbouring segments | merge radius +-1 segment, running-mean position; encoder re-sync on a known splice is **designed, not implemented** | partly |
| 4 | Small defects are only seen in some laps (frame covers ~2 m of a 10 m segment) | detection flickers; "not seen" does not mean "gone" | confirmation needs sightings in **>= 2 different laps**; a confirmed defect is never cleared by missed laps (damage does not heal) | solved |
| 5 | Transient things look like defects (ore lump stuck to the belt, dust, water) | permanent false alarms | unconfirmed track is **dismissed after 5 clean laps** | solved (cost: a real small defect can be dismissed and then re-discovered as a new track) |
| 6 | Splices pass the camera every lap and look odd | a permanent "defect" at the same spot forever | known splice positions are masked (+-3 m): logged, never tracked | solved for *known* splices; unknown ones would be confirmed as defects |
| 7 | Bad frames (blur, glare, dark, dust) | a blurred frame is not evidence of a clean belt | quality gate derived from the training data; bad frames are logged but never update a track | solved |
| 8 | Wear **grows** lap after lap | a single score per image hides progression | severity per lap is kept per track and shown as a sparkline | partly (growth is shown, not yet forecast) |
| 9 | Only one belt side is visible | the pulley side, where idlers touch, is never inspected | `side` is stored per frame; a second camera is out of scope | **not solved** |
| 10 | Train/test leakage | frames of the same stretch are near-identical; random splits inflate accuracy | reported accuracy is called an upper bound; the tracker is evaluated end to end in the loop simulation (defects found vs placed) | partly |
| 11 | Lighting, dust and belt speed change over the day and lap | domain shift between dataset and site | photometric augmentation in the simulation; per-site calibration of the references is needed | not solved |

## 9. Dashboard

Streamlit; it talks only to the HTTP API. Four required views are numbered on the page: status of every
sensor point on a terrain/belt map (green/orange/red), active anomalies (location, type, severity),
predictions (what, where, when with band and evidence), vision detections (loop chart, track table with
severity-per-lap sparkline, thumbnails). Extras: KPIs, per-point time series, data-quality panel
(duplicates, impossible values, sensor faults), hold-out metrics.

## 10. Build vs. skip

| Built | Skipped (and why) |
|---|---|
| simulator with 5 fault modes, duty cycle, 8 kinds of messy data | wind, ore-type effects, thermal inertia of bearings (temperature reacts instantly when the belt stops) - not needed to demonstrate the pipeline |
| lenient, idempotent ingestion | auth, TLS, rate limiting, message queue |
| context-aware anomaly detection + sensor-fault separation | automatic root-cause chains across points (cascades are shown, not modelled) |
| failure-mode + ETA model with honest hold-out and a rule baseline | calibration, per-mode thresholds, smoothing of predictions, retraining schedule |
| loop-aware vision tracker, tested | CNN, real encoder/splice re-sync, second camera, defect growth forecast |
| dashboard with all required views | alert delivery (SMS/e-mail), acknowledgement workflow, multi-user |
| SQLite | PostgreSQL/Timescale, containers, cloud deployment |
