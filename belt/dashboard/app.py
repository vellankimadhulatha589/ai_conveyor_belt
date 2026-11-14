"""Belt-health dashboard (Streamlit). Talks to the backend only through its HTTP API.

    streamlit run belt/dashboard/app.py

The four required views are numbered in the page:
  1. status of every sensor point (belt map)      3. model predictions (what / where / when)
  2. active anomalies (location, type, severity)  4. vision detections (loop-aware defect tracks)
"""
from __future__ import annotations

import inspect
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("BELT_API_URL", "http://127.0.0.1:8000")
LOOP_M = 10000.0
ROUTE_M = 5000.0
SPLICES_M = (3333.0, 7777.0)
CURVES_KM = (1.8, 3.1)

STATUS_COLOR = {"normal": "#2ca02c", "abnormal": "#ff9f1c", "predicted_failure": "#d62728"}
STATUS_LABEL = {"normal": "Normal", "abnormal": "Abnormal", "predicted_failure": "Predicted failure"}
SEV_ICON = {"critical": "🔴 critical", "high": "🟠 high", "medium": "🟡 medium", "low": "🔵 low"}
UNITS = {"vibration": "mm/s", "temperature": "°C", "lateral_offset": "mm", "ambient_temp": "°C",
         "belt_speed": "m/s", "drive_speed": "m/s", "belt_load": "t/h", "motor_current": "A", "voltage": "V"}

st.set_page_config(page_title="Conveyor belt health", page_icon="🛠️", layout="wide")

# Newer Streamlit versions deprecate ``use_container_width=True`` in favour of ``width="stretch"`` (and will
# remove the old form); older versions only know the old one. Pick whichever this installation supports.
STRETCH = ({"width": "stretch"}
           if all("stretch" in str(inspect.signature(f)) for f in (st.plotly_chart, st.dataframe))
           else {"use_container_width": True})


# ----------------------------------------------------------------------------- data access
def api_get(path: str, **params):
    try:
        r = requests.get(API + path, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001 - show the problem instead of crashing the page
        st.session_state["api_error"] = f"{type(exc).__name__}: {exc}"
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_image(url: str):
    try:
        r = requests.get(url, timeout=10)
        return r.content if r.ok else None
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------- views
def kpis(summary: dict) -> None:
    c = summary["status_counts"]
    sev = summary["anomaly_severity_counts"]
    vis = summary["vision"]["tracks"]
    cols = st.columns(6)
    cols[0].metric("Data time (simulated)", (summary["last_data_ts"] or "-").replace("T", " "))
    running = summary.get("belt_running")
    cols[1].metric("Belt", "running" if running else "stopped" if running is not None else "-")
    cols[2].metric("Points normal", c.get("normal", 0), help=f"of {summary['n_points']} sensor points")
    cols[3].metric("Abnormal", c.get("abnormal", 0))
    cols[4].metric("Predicted failure", c.get("predicted_failure", 0))
    cols[5].metric("Confirmed belt defects", vis.get("confirmed", 0))
    st.caption(f"Active anomalies: {summary['active_anomalies']}  "
               f"(critical {sev.get('critical', 0)}, high {sev.get('high', 0)}, medium {sev.get('medium', 0)}, "
               f"low {sev.get('low', 0)})")


def belt_map(points: list[dict]) -> None:
    km = [p["km"] for p in points]
    elev = [p["elevation_m"] for p in points]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km, y=elev, mode="lines", line=dict(color="#9aa0a6", width=3),
                             hoverinfo="skip", showlegend=False))
    for status in ("normal", "abnormal", "predicted_failure"):
        sel = [p for p in points if p["status"] == status]
        if not sel:
            continue
        hover = []
        for p in sel:
            lv = ", ".join(f"{k}={v:.3g}" for k, v in p["latest"].items() if k in ("vibration", "temperature", "lateral_offset"))
            risk = f"<br>failure risk {p['p_fail']:.0%}" if p.get("p_fail") is not None else ""
            hover.append(f"<b>{p['point_id']} - {p['name']}</b><br>km {p['km']:.1f}<br>{STATUS_LABEL[status]}"
                         f"{('<br>' + p['reason']) if p['reason'] else ''}{risk}<br>{lv}")
        fig.add_trace(go.Scatter(
            x=[p["km"] for p in sel], y=[p["elevation_m"] for p in sel], mode="markers+text",
            text=[p["point_id"] for p in sel], textposition="top center", textfont=dict(size=9),
            marker=dict(size=20 if status == "predicted_failure" else 16,
                        color=STATUS_COLOR[status], line=dict(width=1, color="#222"),
                        symbol="diamond" if status == "predicted_failure" else "circle"),
            name=STATUS_LABEL[status], hovertext=hover, hoverinfo="text"))
    for c in CURVES_KM:
        fig.add_vrect(x0=c - 0.25, x1=c + 0.25, fillcolor="#8ab4f8", opacity=0.15, line_width=0,
                      annotation_text="curve", annotation_position="bottom left")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="distance from head drive (km)",
                      yaxis_title="terrain elevation (m)", legend=dict(orientation="h", y=1.12),
                      yaxis=dict(range=[min(elev) - 25, max(elev) + 45]))
    st.plotly_chart(fig, **STRETCH, key="belt_map")


def anomalies_table() -> None:
    rows = api_get("/anomalies", status="active") or []
    if not rows:
        st.success("No active anomalies.")
        return
    df = pd.DataFrame([{
        "Severity": SEV_ICON.get(r["severity"], r["severity"]),
        "Location": f"km {r['km']:.1f} ({r['point_id']})",
        "Type": r["type"],
        "Observed vs expected": (f"{r['value']:.3g} vs {r['expected']:.3g}" if r.get("value") is not None and r.get("expected") is not None
                                 else (r.get("detail") or "")),
        "Since": (r.get("first_seen") or "").replace("T", " "),
    } for r in rows])
    st.dataframe(df, hide_index=True, **STRETCH, height=min(420, 38 + 35 * len(df)))


def predictions_table() -> None:
    data = api_get("/predictions/latest") or {}
    preds = data.get("predictions", [])
    if not preds:
        st.success("No failure predicted within the model horizon.")
        return
    df = pd.DataFrame([{
        "Level": "Predicted failure" if p["level"] == "predicted_failure" else "Watch",
        "Where": f"km {p['km']:.1f} ({p['point_id']})",
        "What may fail": p["failure_mode"],
        "Probability": round(100 * p["p_fail"]),
        "When (est.)": f"~{p['eta_hours']:.0f} h  ({p['eta_low_h']:.0f}-{p['eta_high_h']:.0f} h)",
        "Expected at": (p["eta_ts"] or "").replace("T", " "),
        "Evidence": p["evidence"],
    } for p in preds])
    st.dataframe(df, hide_index=True, **STRETCH, height=min(420, 38 + 35 * len(df)),
                 column_config={"Probability": st.column_config.ProgressColumn("Probability", min_value=0, max_value=100, format="%d%%")})
    st.caption("ETA = median of three quantile models, in brackets the 20 %-80 % band. "
               "Time is measured on the simulated data clock.")


def loop_chart(tracks: list[dict]) -> None:
    fig = go.Figure()
    th = [i for i in range(361)]
    fig.add_trace(go.Scatterpolar(theta=th, r=[1] * 361, mode="lines", line=dict(color="#9aa0a6", width=6),
                                  hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatterpolar(theta=[0, 180], r=[1, 1], mode="markers+text", text=["head", "tail"],
                                  textposition="top center", marker=dict(size=9, color="#5f6368"),
                                  hoverinfo="skip", name="pulleys"))
    fig.add_trace(go.Scatterpolar(theta=[s / LOOP_M * 360 for s in SPLICES_M], r=[1, 1], mode="markers",
                                  marker=dict(size=10, color="#80868b", symbol="diamond"),
                                  name="known splice (ignored)", hovertemplate="splice at %{theta:.0f} deg<extra></extra>"))
    for status, color in (("confirmed", "#d62728"), ("suspected", "#ff9f1c")):
        sel = [t for t in tracks if t["status"] == status]
        if sel:
            fig.add_trace(go.Scatterpolar(
                theta=[t["position_m"] / LOOP_M * 360 for t in sel], r=[1] * len(sel), mode="markers",
                marker=dict(size=[10 + 22 * (t["max_severity"] or 0) for t in sel], color=color, line=dict(width=1, color="#222")),
                name=status, hovertext=[f"#{t['id']} {t['label']} at {t['position_m']:.0f} m, severity {t['severity_level']}, "
                                        f"seen in {t['laps_seen']} lap(s)" for t in sel], hoverinfo="text"))
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=-0.05),
                      polar=dict(angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                                                  tickvals=[0, 90, 180, 270],
                                                  ticktext=["0 m", "2.5 km", "5 km (tail)", "7.5 km"]),
                                 radialaxis=dict(visible=False, range=[0, 1.15])))
    st.plotly_chart(fig, **STRETCH, key="loop_chart")
    st.caption("The belt is a closed loop of 10 km: 0-5 km is the carry side, 5-10 km the return side. "
               "One dot = one physical defect, however many laps it was seen in.")


def vision_panel(summary: dict) -> None:
    v = summary["vision"]
    if not v["frames"]:
        st.info("No vision data yet. Run `python -m belt.vision.run_loop` (see README).")
        return
    c = st.columns(6)
    c[0].metric("Laps inspected", v["last_lap"])
    c[1].metric("Frames analysed", v["frames"])
    c[2].metric("Raw defect detections", v["defect_frames"])
    c[3].metric("Unique defects", v["unique_open_defects"], help="after de-duplication across laps")
    c[4].metric("Segments sampled (last lap)", f"{(v['coverage_last_lap'] or 0):.0%}",
                help="share of the 10 m segments that produced at least one usable frame")
    c[5].metric("Belt surface imaged", f"{(v.get('surface_imaged_last_lap') or 0):.0%}",
                help="the camera sees ~2 m of every 10 m segment per lap, so small defects are only caught in "
                     "some laps - this is the coverage gap the loop tracker has to live with")
    st.caption(f"Ignored on purpose: {v['splice_frames_ignored']} frames of known splices, "
               f"{v['low_quality_frames']} bad frames (blur/dark). "
               f"Tracks: {v['tracks']}")
    tracks = api_get("/vision/tracks", statuses="confirmed,suspected") or []
    left, right = st.columns([1, 1.2])
    with left:
        loop_chart(tracks)
    with right:
        if tracks:
            df = pd.DataFrame([{
                "ID": t["id"], "Position (m on loop)": round(t["position_m"]),
                "Side": "carry" if t["position_m"] < ROUTE_M else "return",
                "Type": t["label"], "Severity": t["severity_level"], "Status": t["status"],
                "Laps seen": t["laps_seen"], "Last lap": t["last_lap"],
                "Severity per lap": [h[1] for h in t["severity_history"]],
            } for t in tracks])
            st.dataframe(df, hide_index=True, **STRETCH, height=380,
                         column_config={"Severity per lap": st.column_config.LineChartColumn("Severity per lap", y_min=0, y_max=1)})
    shown = [t for t in tracks if t.get("image_file")][:6]
    if shown:
        st.markdown("**Most severe detections**")
        cols = st.columns(len(shown))
        for col, t in zip(cols, shown):
            img = fetch_image(f"{API}/vision/frames/{t['image_file']}")
            if img:
                col.image(img, caption=f"#{t['id']} {t['label']} @ {t['position_m']:.0f} m ({t['severity_level']})")


def point_detail(points: list[dict]) -> None:
    labels = {p["point_id"]: f"{p['point_id']} - km {p['km']:.1f} - {STATUS_LABEL[p['status']]}" for p in points}
    ids = list(labels)
    default = next((p["point_id"] for p in points if p["status"] == "predicted_failure"), ids[0])
    c1, c2 = st.columns([2, 1])
    pid = c1.selectbox("Sensor point", ids, index=ids.index(default), format_func=lambda x: labels[x], key="pt_select")
    hours = c2.select_slider("Window (hours)", options=[24, 48, 72, 168, 336], value=72, key="pt_hours")
    p = next(p for p in points if p["point_id"] == pid)
    if p["reason"]:
        st.warning(p["reason"])
    chans = ["vibration", "temperature", "lateral_offset"]
    if p["kind"] == "drive":
        chans += ["belt_speed", "motor_current", "voltage", "belt_load"]
    series = api_get(f"/points/{pid}/series", hours=hours, channels=",".join(chans)) or {}
    cols = st.columns(3)
    for i, ch in enumerate(chans):
        pts = series.get(ch, [])
        fig = go.Figure(go.Scatter(x=[a for a, _ in pts], y=[b for _, b in pts], mode="lines", line=dict(width=1.5)))
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), title=f"{ch} [{UNITS.get(ch, '')}]")
        cols[i % 3].plotly_chart(fig, **STRETCH, key=f"series_{pid}_{ch}")
    st.caption("Gaps = belt stopped (18 h/day duty cycle, trips) or missing data. Impossible values are not drawn.")


def data_quality_panel(summary: dict) -> None:
    dq = api_get("/data-quality") or {}
    if not dq:
        return
    c = st.columns(4)
    c[0].metric("Rows received (24 h)", dq["rows_received"], help=f"expected {dq['rows_expected']}")
    c[1].metric("Rows missing", f"{dq['rows_missing_pct']}%", help="sensor outages / lost messages")
    tot = dq.get("ingest_totals", {})
    c[2].metric("Duplicates dropped", tot.get("duplicates", 0))
    c[3].metric("Values flagged impossible", tot.get("flagged", 0))
    if dq["flagged_values"]:
        st.write("Flagged values in the last 24 h:", dq["flagged_values"])
    sf = dq.get("active_sensor_faults", [])
    if sf:
        st.write("**Active sensor faults** (data problems, not equipment problems)")
        st.dataframe(pd.DataFrame(sf), hide_index=True, **STRETCH)
    else:
        st.success("No active sensor faults.")


def metrics_panel(summary: dict) -> None:
    m = (summary.get("model_metrics") or {}).get("holdout")
    if not m:
        st.info("No evaluation metrics stored.")
        return
    st.caption(f"Time-based hold-out: train {m['train_span'][0][:10]} .. {m['train_span'][1][:10]}, "
               f"test {m['test_span'][0][:10]} .. {m['test_span'][1][:10]}")
    st.dataframe(pd.DataFrame({"ML model": m["model"], "Naive rule (any channel >= 4 sigma)": m["rule_baseline"]}),
                 **STRETCH)


# ----------------------------------------------------------------------------- page
st.title("Mining conveyor belt - health monitor")
with st.sidebar:
    st.header("Controls")
    refresh = st.select_slider("Auto-refresh (seconds)", options=[0, 2, 5, 10, 30], value=5,
                               help="0 = off. Reload the page for a manual refresh.")
    st.caption(f"Backend: {API}")
    if st.button("Run analysis now"):
        try:
            st.write(requests.post(API + "/analysis/run", timeout=120).json())
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    st.markdown("---")
    st.caption("Colours: green normal, orange abnormal (active anomaly), red predicted failure.")


def render() -> None:
    st.session_state.pop("api_error", None)
    summary = api_get("/status/summary")
    if summary is None:
        st.error(f"Cannot reach the backend at {API}. Start it with "
                 f"`python -m uvicorn belt.api.main:app --port 8000`.\n\n{st.session_state.get('api_error', '')}")
        return
    if not summary["last_data_ts"]:
        st.info("The database is empty. Run `python scripts/bootstrap.py` first (see README).")
        return
    if not summary["model_trained"]:
        st.warning("Models are not trained yet - run `python scripts/bootstrap.py`.")
    kpis(summary)

    st.subheader("1. Status of every sensor point")
    points = api_get("/points") or []
    if points:
        belt_map(points)

    a, b = st.columns(2)
    with a:
        st.subheader("2. Active anomalies")
        anomalies_table()
    with b:
        st.subheader("3. Model predictions")
        predictions_table()

    st.subheader("4. Vision detections (belt wear and tear)")
    vision_panel(summary)

    st.subheader("Sensor point detail")
    if points:
        point_detail(points)

    with st.expander("Data quality (messy input handling)"):
        data_quality_panel(summary)
    with st.expander("Model evaluation (hold-out)"):
        metrics_panel(summary)


frag = getattr(st, "fragment", None)
if frag is not None and refresh:
    frag(run_every=refresh)(render)()
else:
    render()