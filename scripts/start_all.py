"""Start the whole system with one command (API + live sensor stream + dashboard).

    python scripts/start_all.py                       # sensors only
    python scripts/start_all.py --vision              # also run the vision loop demo
    python scripts/start_all.py --speed 60            # faster simulation
    python scripts/start_all.py --seconds-per-sample 3   # slow live mode: new data every 3 s

Press Ctrl+C to stop everything. Equivalent manual commands are listed in the README.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=30.0, help="simulated minutes per real second")
    ap.add_argument("--seconds-per-sample", type=float, default=0.0,
                    help="slow live mode: one new reading per sensor every N real seconds (e.g. 3); overrides --speed")
    ap.add_argument("--vision", action="store_true", help="also run the vision loop simulation")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--dashboard-port", type=int, default=8501)
    args = ap.parse_args()

    api = f"http://127.0.0.1:{args.api_port}"
    env = dict(os.environ, BELT_API_URL=api, PYTHONPATH=str(ROOT))
    procs: list[subprocess.Popen] = []

    def spawn(name: str, cmd: list[str]) -> subprocess.Popen:
        print(f"starting {name}: {' '.join(cmd)}")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env)
        procs.append(p)
        return p

    try:
        spawn("API", [PY, "-m", "uvicorn", "belt.api.main:app", "--port", str(args.api_port)])
        for _ in range(60):
            try:
                if requests.get(api + "/health", timeout=2).ok:
                    break
            except requests.RequestException:
                time.sleep(1)
        spawn("dashboard", [PY, "-m", "streamlit", "run", "belt/dashboard/app.py",
                            "--server.port", str(args.dashboard_port), "--server.headless", "true"])
        live_cmd = [PY, "-m", "belt.simulator.live", "--minutes-per-second", str(args.speed)]
        if args.seconds_per_sample > 0:
            live_cmd += ["--seconds-per-sample", str(args.seconds_per_sample)]
        spawn("sensor stream", live_cmd)
        if args.vision:
            spawn("vision loop", [PY, "-m", "belt.vision.run_loop", "--laps", "8", "--lap-delay", "3"])
        print(f"\nDashboard: http://localhost:{args.dashboard_port}    API docs: {api}/docs\n(Ctrl+C to stop)\n")
        while True:
            time.sleep(1)
            for p in procs:
                if p.poll() not in (None, 0) and p is procs[0]:
                    raise SystemExit("API process exited")
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()