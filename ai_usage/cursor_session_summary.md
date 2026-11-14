# Cursor session summary (condensed, in order)

## 1. Understanding the problem before writing code
**Human:** pasted the assignment and asked for an analysis with examples, *before* building anything.
**Assistant:** split it into the five components, explained sensors / simulator / prediction / vision /
dashboard with examples, and pointed out the "looping belt" trap in the vision part (same defect seen every
lap, position, one visible side, splices, leakage).
**Why it mattered:** decided early that the loop problem is a *tracking* problem, not a classification one.

## 2. "What can go wrong on a conveyor belt? Can you search?"
**Human:** did not accept a from-memory answer and asked for research.
**Assistant:** ran web searches (belt-tracking article, bulk-handling review, idler failure study of ~98 k
rollers, belt-tear causes, condition-monitoring reviews) and produced a failure-mode table with sources
and the observation that *failures cascade*.
**Outcome:** the failure list in `docs/DESIGN.md` section 2 is source-backed; the idea that temperature is a
**late** indicator and vibration an **early** one came from the search results and drove the simulator curves.

## 3. Human pushback: symptoms vs failures
**Human:** pointed out that temperature rising, vibration, speed up/down, voltage low/high, load and
alignment also have to be considered, and asked how each is detected and whether to include them.
**Assistant:** explained they are *signals*, not failures, and mapped symptom patterns to causes (e.g. speed
down + current up + voltage down = electrical problem, not overload). Added "sensor faults" as a separate
class so a flat-lined sensor is never reported as a hot bearing.
**Outcome:** the channel list (vibration, temperature, offset, ambient, two speeds, load, current, voltage)
and the symptom-to-cause table in the design doc.

## 4. Failure mode -> signal -> time-shape table
**Assistant** proposed concrete degradation curves and noise/messy-data models; **human** approved
("yes") and asked for the full project.
**Open questions handed back and resolved by defaults:** 26 sensor points / 200 m, 5-minute sampling,
120 simulated days, both history backfill and a live stream.

## 5. Building the project
Constraints given by the human: write everything into `ai_conveyor_belt/`, **do not install any Python
version**, the human will run it from VS Code, include a database and a step-by-step README.

Decisions the assistant made and why (all documented in the design doc):
- SQLite + FastAPI + Streamlit: zero set-up for a reviewer.
- Classical image features + RandomForest instead of a CNN: no PyTorch/GPU needed, interface allows swapping.
- Anomaly detection against a *learned, load-aware baseline* rather than fixed thresholds.
- Time-based evaluation with a purge gap and a naive-rule baseline, instead of a random split.
- Labels from a work-order log, not from the simulator's hidden schedule.

## 6. Corrections and caveats the assistant raised against its own work
- Package installation first failed (corporate TLS interception by Zscaler), so the code was initially written
  and only statically checked. After the human trusted the proxy certificate in WSL and installed the
  requirements, the assistant ran everything: 32 unit tests (incl. the slow end-to-end one), bootstrap, API,
  dashboard (page text checked, no exceptions) and the vision loop.
- Running it exposed two real vision bugs that static checking could not: dust specks classified as "wear"
  (19 false confirmed tracks over 8 laps) and a splice mask that ignored encoder error. Both were fixed (training
  augmentation, mask margin + unit test) and the loop result was re-measured: 0 false tracks, 11/12 real defects.
- The "belt coverage" KPI counted sampled segments (89 %), which overstated what a 2 m camera view per 10 m
  segment really images (~18 %); the dashboard now shows both.
- Reported honestly in the README: the ETA band is over-confident (39 % coverage for a nominal 60 % band), the
  model trades 5x more false-alarm episodes for earlier warnings, and 10 test faults is a small sample.
- The vision dataset was **not verified** (no download possible). Candidates are listed with their caveats;
  the human has to choose, check the licence and fill in the README section.
- An offline synthetic image generator exists only so the pipeline can be exercised; it is labelled as a
  stand-in everywhere because the brief requires a public dataset.
- Dashboard screenshots could not be captured by the assistant's browser tool (it only grabs the small visible
  panel); the README section for them is left for the human.

## 7. Things the human should still do / challenge
- Run it, read the printed hold-out metrics, and check whether the claims in the design doc hold; adjust the
  doc to the real numbers.
- Question the assumptions listed in `belt/config.py` (observable windows, thresholds): they are plausible
  engineering guesses, not measured facts.
