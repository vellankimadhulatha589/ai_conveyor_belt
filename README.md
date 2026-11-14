# Mining conveyor belt - health monitoring and failure prediction

A small but complete system: **simulated sensors -> backend (store + analyse) -> prediction model ->
dashboard**, plus a **vision component** that finds wear and tear on the belt and copes with the fact that
the belt is a *loop*.

* Design document: **[docs/DESIGN.md](docs/DESIGN.md)** (what is monitored, how, why; read it second)
* Plain-language walkthrough for newcomers (examples, libraries, model, run steps, how to present it):
  **[docs/EXPLAIN_PROJECT.txt](docs/EXPLAIN_PROJECT.txt)**
* Code flow - which file calls which, starting from `scripts/start_all.py`:
  **[docs/CODE_FLOW.md](docs/CODE_FLOW.md)**
* AI chat logs: [ai_usage/](ai_usage/)

```
 simulator ──POST /readings──▶ FastAPI backend ──▶ SQLite ──▶ analysis (anomalies, predictions) ─┐
 (26 sensor suites, messy)       validate / flag /                                               ├─▶ Streamlit dashboard
                                 de-duplicate                                                    │
 public belt images ─▶ vision detector ─▶ POST /vision/observations ─▶ loop-aware tracker ───────┘
```

> **Status.** The whole pipeline has been run end to end (simulate -> ingest -> train -> API -> dashboard ->
> vision loop) and the 32 unit tests pass. The vision model is trained on a public Kaggle dataset (see
> [section 4](#4-vision-dataset)) and the dashboard screenshots are in [section 8](#8-dashboard-screenshots).
> All sensor data comes from my own simulator, so the sensor results in section 7 show that the pipeline and
> the evaluation logic work, not real-world accuracy.

---

## 1. Quick start (VS Code, step by step)

Everything runs from the folder `ai_conveyor_belt/` with the Python environment that is already there
(`.venv`). No Python version is installed by anything in this repo.

**Step 0 - open and select the interpreter.** In VS Code: *File -> Open Folder* -> `ai_conveyor_belt`.
`Ctrl+Shift+P` -> *Python: Select Interpreter* -> `.venv/bin/python`. Open a terminal (`Ctrl+\``) - it
activates the environment. (WSL: use the *WSL* remote window; ports 8000/8501 are forwarded automatically.)

**Step 1 - install the libraries** (numpy, pandas, scikit-learn, fastapi, uvicorn, streamlit, plotly, pillow, ...):

```bash
pip install -r requirements.txt
```

**Step 2 - create the history and train the models** (about 1-3 minutes):

```bash
python scripts/bootstrap.py            # or: python scripts/bootstrap.py --days 60   (faster)
```

What happens: the simulator generates 120 days of 5-minute readings for 26 sensor points (with missing
values, stuck sensors, impossible values, duplicates, shuffled order), the data goes through the same
ingestion code the API uses into `data/belt.db`, work orders of repaired failures are stored, the baseline
and the failure predictor are trained with a time-based hold-out evaluation (printed, and saved to
`data/models/metrics.json`), and a first analysis runs. Four faults are deliberately "in progress" at the
end of the history, so the dashboard shows something immediately.

**Step 3 - vision model.** Put your public dataset in `data/vision_dataset/<class>/*.jpg` (see
[Vision dataset](#4-vision-dataset)) and train:

```bash
python -m belt.vision.train
```

For an offline smoke test without the Kaggle download you can generate procedural images instead
(`python -m belt.vision.train --synthetic`); they are only meant for tests and are not the project's dataset.

**Step 4 - start everything:**

```bash
python scripts/start_all.py            # API + live sensor stream + dashboard, Ctrl+C stops all
python scripts/start_all.py --seconds-per-sample 3   # same, but 'live': new readings every 3 s (see docs/CODE_FLOW.md, section 5)
```

or in three terminals (also available as VS Code *Run and Debug* entries, see `.vscode/launch.json`):

```bash
python -m uvicorn belt.api.main:app --port 8000     # backend, docs at http://localhost:8000/docs
python -m belt.simulator.live                       # live stream, 30 simulated minutes per second
python -m streamlit run belt/dashboard/app.py       # dashboard at http://localhost:8501
```

**Step 5 - watch the belt loop** (in another terminal; needs step 3):

```bash
python -m belt.vision.run_loop --laps 6 --lap-delay 2
```

It prints, per lap, how many frames were analysed and what the tracker did with them, and at the end the
score: *raw defect detections vs. unique defects vs. defects actually placed on the belt*.

**Step 6 - tests:**

```bash
python -m unittest discover -s tests -t . -v        # fast; slow end-to-end:  RUN_SLOW=1 python -m unittest tests.test_analysis -v
```

Order matters only for steps 2 -> 4; the API can be started before bootstrap, it just shows "database empty".

---

## 2. How it works, step by step

1. **Simulator** (`belt/simulator`) - for any timestamps it returns one row per sensor point: vibration,
   bearing temperature, lateral belt offset, ambient temperature, and at the drive station belt/drive speed,
   load, motor current, voltage. A healthy belt depends on load, ambient temperature and an 18 h/day duty
   cycle. Faults (bearing, mistracking, slip, overload, undervoltage) follow degradation curves in which
   *vibration rises first and temperature last*. Garbage is injected on purpose.
2. **Ingestion** (`belt/ingest.py`, `POST /readings`) - rows with unknown point or unparsable time are
   rejected and counted; impossible/non-numeric values are kept raw but flagged; duplicates are ignored
   (primary key = point + time); out-of-order is harmless.
3. **Baseline** (`analysis/prepare.py`) - for each channel and point the *expected* value given load and
   ambient temperature is learned from history (robust regression). Residual = observed - expected.
4. **Anomalies** (`analysis/anomaly.py`) - the median of the last hour of residuals must exceed 4 robust sigma
   **and** a practical minimum in the bad direction. Stuck or silent sensors are reported as *sensor* faults,
   not equipment faults. Output: location, type, severity.
5. **Prediction** (`analysis/features.py`, `predictor.py`) - residual features over 6/24/72 h per point ->
   gradient boosting: *what* may fail (5 modes), *how likely*, and *when* (median ETA + 20-80 % band).
6. **Status** - each point is `normal`, `abnormal` (active anomaly) or `predicted_failure` (model probability
   >= 50 %).
7. **Vision** (`belt/vision`) - features -> classifier per frame -> tracker that turns frames into *physical
   defects on the loop* (section 5).
8. **Dashboard** (`belt/dashboard/app.py`) - the four required views plus point detail, data quality, metrics.

---

## 3. Design decisions

| Decision | Why | Alternative I did not take |
|---|---|---|
| Sensor point every 200 m, drive-only channels at the head | realistic: drive electricals exist once; idlers carry vibration/temperature/offset | identical sensors everywhere (simpler, less realistic) |
| Compare against a **learned load-aware baseline**, not fixed limits | 3.5 mm/s is normal at peak load and alarming at half load; avoids false alarms | static thresholds (kept only for a few safety rules) |
| Median over ~1 h + both a z-score and a practical minimum | spikes can't alarm; tiny but "significant" shifts can't alarm | single-sample thresholds |
| Sensor faults are their own anomaly type | a flat-lined thermometer is not an overheating bearing | treating everything as equipment |
| Labels from a **work-order log**, not the simulator's hidden schedule | mirrors reality (maintenance records); keeps the backend honest about what it knows | reading the answer key |
| Per-mode "observable window" labels | a failure is only predictable while its signature exists (overload: ~hours, bearing: days) | one fixed horizon for all modes |
| Time-based evaluation + purge gap + naive-rule baseline | random splits leak between near-identical snapshots; a baseline shows what ML adds | random 80/20 |
| HistGradientBoosting + quantile regressors | NaN-tolerant, no scaling, fast, gives an ETA band | deep nets (no data), single regressor (no uncertainty) |
| Classical image features + RandomForest | laptop CPU, no PyTorch, small datasets, swappable behind `DefectDetector` | fine-tuned CNN / YOLO (better, heavier - listed under "differently") |
| SQLite, FastAPI, Streamlit | one-file DB and three commands for a reviewer | PostgreSQL/Timescale, React |
| Everything tunable in `belt/config.py` | assumptions are visible and challengeable | constants scattered in code |

---

## 4. Vision dataset

**Dataset used:** [Industrial Machinery Fault Image Dataset (Kaggle, zara2099)](https://www.kaggle.com/datasets/zara2099/industrial-machinery-fault-image-dataset).
Licence: as stated on that Kaggle page (check it before redistributing the images; they are not committed to
this repository, `data/vision_dataset/` is git-ignored).

| Class folder | Images | Role |
|---|---|---|
| `Normal condition` | 48 | normal (healthy) |
| `Crack` | 47 | defect |
| `Misalignment` | 53 | defect |
| `Surface defects` | 43 | defect |
| `Wear` | 43 | defect |

234 images in total, already one folder per class, so `scripts/yolo_to_folders.py` was not needed.

**Why this dataset:** it is public, small enough to train on a laptop CPU in seconds, and it has a normal class
plus several defect classes, which is what the detector and the loop tracker need.

**Limitations (stated up front):**

* The images show **conveyor rollers and idlers**, not the belt surface. It is a loose match for belt-surface
  tear/wear; the defect names in the dashboard are the dataset's class names.
* About 45 images per class: the hold-out set has only 10-15 images per class, so the accuracy figure is
  noisy (one image moves it by several points).
* Frames may come from the same scenes, so a random split can overestimate accuracy.
* A dataset with real belt-surface damage (for example the larger iron-ore conveyor dataset on Kaggle, or the
  4-class set of [Applied Sciences 11(16):7282](https://www.mdpi.com/2076-3417/11/16/7282)) would be the next
  step; the code does not change, only the folder content.

**How to plug a dataset in**

* Folder-per-class: `data/vision_dataset/normal/*.jpg`, `.../tear/*.jpg`, `.../wear/*.jpg` ... Class names
  `normal`, `good`, `ok`, `healthy`, `no_defect`, `normal condition` count as normal (matching ignores case); everything else is a defect type.
* YOLO-format download (images + `labels/*.txt`)? Convert with
  `python scripts/yolo_to_folders.py --src <download folder> [--names tear,damage,wear]` (empty label file =
  normal, otherwise the class of the largest box).
* Then `python -m belt.vision.train` (prints accuracy, per-class precision/recall, confusion matrix; saved to
  `data/models/vision_metrics.json`).

---

## 5. The looping belt (vision) - my reasoning

Full table in [docs/DESIGN.md section 8](docs/DESIGN.md#8-vision). The short version:

* The camera sees **the same belt metres every lap**. If every sighting were a "defect", one tear would
  become dozens a day. So a defect is identified by **where it sits on the loop** (position mod loop length ->
  10 m segment), not by when the picture was taken. Frames from different laps at the same place merge into one
  *track*.
* One frame covers ~2 m of a 10 m segment, so small defects are **only caught in some laps** (detections
  flicker). A track becomes *confirmed* after sightings in >= 2 *different* laps; a confirmed defect is never
  cleared by missed laps. An unconfirmed one is *dismissed* after 5 clean laps (ore lump, dust).
* **Splices** pass the camera every lap and look like defects: known splice positions are masked.
* **Bad frames** (blur, dark, glare) are logged but never count as evidence of a clean belt.
* Severity per lap is kept per track, so wear **growth** is visible.
* **Not solved, on purpose:** the underside (pulley side) is not inspected - one camera only; encoder drift
  and re-synchronisation on a splice are designed but not implemented; unknown splices would be confirmed
  as defects; site-specific lighting calibration; reported classifier accuracy is an upper bound because
  nearby frames leak.

`belt/vision/run_loop.py` simulates exactly these effects (laps, jittered frames, defects re-appearing,
splices, bad frames, encoder noise) against the real API, then scores the result (defects found vs placed).

---

## 6. Messy input - what the system does

| Mess | Handling | Where |
|---|---|---|
| missing readings / silent sensor | NULL, ignored; `sensor_offline` anomaly if > 50 % missing | `ingest.py`, `anomaly.py` |
| duplicates, out-of-order | primary key + sorted re-gridding; counted on the dashboard | `ingest.py`, `prepare.py` |
| impossible values | stored raw, flagged, excluded from analysis | `ingest.py` |
| stuck sensor | exact repetition >= 1.5 h while running -> sensor fault, samples excluded | `prepare.py`, `anomaly.py` |
| spikes | median over the last hour; features use clipped hourly means | `anomaly.py`, `features.py` |
| belt stopped (duty cycle, trips) | residuals only while running; "unknown" instead of "normal" | `prepare.py`, `anomaly.py` |
| component repaired | history of that point is cut at the repair (no ghost alarms) | `pipeline.py`, `features.py` |
| calibration drift | **not handled** (two simulated sensors drift +0.03 degC/day) | - |

---

## 7. Results

Produced by `python scripts/bootstrap.py` (seed 7, 120 days of history, 49 scheduled faults). Hold-out =
last part of the timeline, with a 7-day purge between train and test so a fault cannot straddle the split
(10 faults in the test span). Full report: `data/models/metrics.json`, also in the dashboard's *Model
evaluation* expander.

| | ML model | Naive rule (any channel >= 4 sigma) |
|---|---|---|
| Faults announced before failing | 10 / 10 | 10 / 10 |
| Median / minimum lead time before failure | 76.5 h / 26.5 h | 59.7 h / 26.5 h |
| Snapshot precision / recall | 0.95 / 0.92 | 0.98 / 0.63 |
| False-alarm episodes (per 100 point-days) | 5 (0.59) | 1 (0.12) |
| Failure mode correct (of detected) | 97.6 % | not applicable |
| ETA median absolute error | 8.8 h | not applicable |

How to read this, without flattering it:

* Both approaches catch every fault in the test span, so the model's real gain is *earlier and steadier*
  warnings (about 17 h more lead time, recall 0.92 vs 0.63) - and it pays for that with roughly five times
  more false-alarm episodes. Which side of that trade a mine wants is an operations decision, not a modelling one.
* **Ten test faults is a small sample.** Treat every digit as indicative. The model is trained and tested on
  data from my own simulator, so this proves the pipeline and the evaluation logic, not real-world accuracy.
* **The ETA uncertainty band is over-confident:** the 20-80 % quantile band should contain the true time
  about 60 % of the time and contains it 39 % of the time. The point ETA is usable (8.8 h median error),
  the band is not calibrated yet. Not fixed - listed in section 9.

**Vision loop logic, measured with the procedural test images** (`python -m belt.vision.run_loop`, 12 hidden
defects on a 10 km loop, 8 laps, 8 000 frames, 4 % blurred/dark frames, 2 splices, ~1 m encoder noise):

| | before fixes | after fixes |
|---|---|---|
| real defects confirmed | 12 / 12 | 11 / 12 |
| confirmed tracks with no real defect | 19 | 0 |

Two defects of the first version were found by reading this output and are fixed: (1) the classifier read
dark dust specks as "wear", so ~2 % of healthy frames were flagged; with 1 000 frames per lap, random
flags in neighbouring segments in two different laps look like a defect. Fix: train with lighting / noise /
dust augmentation (ranges wider than the simulator's, so it is not tuned to the test). (2) A frame showing
a splice could be reported just outside the splice mask because of encoder error; the mask now includes the
position error (`config.POSITION_ERROR_M`, with a unit test).

The remaining miss is a 1.5 m "wear" defect: the camera sees ~2 m of every 10 m, so a defect that short is
caught in only ~35 % of laps and was seen once in 8 laps. That is the coverage limit of the setup, not a
bug - the dashboard shows it ("Belt surface imaged" about 18 %). More laps or a denser camera would fix it.

Those test images are trivially separable (hold-out accuracy 1.0), so the table above tests the **loop
logic** (de-duplication, splice masking, confirmation over laps), not classification quality.

**Vision on the Kaggle dataset.** `python -m belt.vision.train` prints overall accuracy, per-class
precision/recall and the confusion matrix and saves them to `data/models/vision_metrics.json`;
`run_loop` prints the loop result (confirmed and matched, confirmed with no real defect). With 234 images
and rollers instead of belt surface, treat those numbers as a demonstration that the real-data path works.
The dashboard on this dataset (screenshots in section 8) shows more confirmed tracks than the 12 hidden
defects, i.e. the classifier raises false alarms on these hard-to-separate images; the loop confirmation
logic reduces but does not remove them.

---

## 8. Dashboard screenshots

Overview: belt map (red diamonds are points predicted to fail), active anomalies and model predictions.

![Overview: belt map, anomalies, predictions](docs/screenshots/01_overview.png)

Vision: loop chart, defect tracks and example frames from the Kaggle dataset.

![Vision: loop chart, tracks, thumbnails](docs/screenshots/02_vision_loop.png)

Point detail: sensor series of a point with a predicted failure (vibration rises before temperature).

![Point detail: vibration rising before temperature](docs/screenshots/03_point_detail.png)

Data quality panel: rejected, flagged and duplicate readings.

![Data quality](docs/screenshots/04_data_quality.png)

---

## 9. What I would do differently / next

* **Real data first.** Everything is validated on simulated data. With real data the first week would go
  to the baseline (seasonality, ore types), to correcting the observable-window assumptions with reliability
  engineers, and to a labelled failure history instead of simulated work orders.
* **Vision:** use real belt-surface images instead of rollers; fine-tune a CNN/YOLO with segmentation masks, calibrate severity against repair decisions,
  add a splice detector and encoder re-sync, a second camera for the pulley side, and forecast defect growth
  per track (the severity history is already stored).
* **Prediction:** calibrate probabilities, smooth over cycles (hysteresis), per-mode thresholds, model the
  cascades between points (a seized idler raises neighbours' vibration), retraining and drift monitoring.
* **Platform:** PostgreSQL/Timescale, a queue between simulator/edge and backend, authentication, Docker
  and a cloud deployment, alert delivery with acknowledgement.
* **Calibrate the prediction band** (conformal prediction on the hold-out) and report precision/recall per
  failure mode, not only overall; with 10 test faults the evaluation itself needs far more history.
* **Camera coverage:** a line-scan camera that images the whole belt instead of 2 m every 10 m.

---

## 10. Repository layout

```
README.md  docs/DESIGN.md  ai_usage/  requirements.txt
belt/config.py                    all tunable assumptions
belt/simulator/                   layout, fault schedule, generator, live streamer
belt/db.py  belt/ingest.py        SQLite schema, validation + storage
belt/analysis/                    prepare (baseline, residuals), anomaly, features, predictor, train, pipeline
belt/api/main.py                  FastAPI backend
belt/vision/                      features, train, detector, tracker (loop logic), run_loop, synth (test images only)
belt/dashboard/app.py             Streamlit dashboard
scripts/                          bootstrap, start_all, yolo_to_folders
tests/                            tracker (stdlib), simulator/ingest, analysis, optional slow end-to-end
.vscode/                          launch configurations
data/                             database, models, frames, dataset (generated / downloaded, git-ignored)
```

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: numpy` (or pandas, fastapi ...) | `pip install -r requirements.txt` in the `.venv` selected in VS Code |
| Dashboard says "Cannot reach the backend" | start the API (`python -m uvicorn belt.api.main:app --port 8000`) |
| Dashboard says "database is empty" | `python scripts/bootstrap.py` |
| `too few failures in the history` | use more days: `bootstrap.py --days 120` |
| Vision loop: `No such file ... vision_model.joblib` | `python -m belt.vision.train` first (dataset in `data/vision_dataset/`) |
| `pip` fails with `CERTIFICATE_VERIFY_FAILED` (corporate proxy) | configure your company's CA / index mirror for pip; do not disable TLS verification |
| Port already in use | `start_all.py --api-port 8001 --dashboard-port 8502` |
| Start over | stop everything, delete `data/belt.db*`, run `bootstrap.py` again (it recreates the database) |

## 12. Sources used for the failure-mode research

[Canadian Mining Journal - belt tracking](https://www.canadianminingjournal.com/featured-article/feeling-centered-the-zen-of-conveyor-belt-tracking/) ·
[Australian Bulk Handling Review - problems with belts in operation](https://www.bulkhandlingreview.com.au/problems-with-belts-in-operation/) ·
[Idler failure types (Vale data, ~98 k rollers)](https://www.metallum.com.br/24cbecimat/anais/PDF/IVd18-001.pdf) ·
[TMKS - tear causes](https://www.t-mks.com/en/solutions/conveyor-belt-longitudinal-tear-prevention-solutions/) ·
[Idler condition-based monitoring review](https://itecsu.com/resources/idler-condition-based-monitoring-systems-review/) ·
[MDPI - NDT for conveyor belt monitoring](https://www.mdpi.com/2076-3417/15/24/13272) ·
[Applied Sciences 11(16):7282 - belt damage detection](https://www.mdpi.com/2076-3417/11/16/7282)
