"""Vision stream simulator: a camera watching a belt that loops past it, lap after lap.

    python -m belt.vision.run_loop --laps 6 --lap-delay 2

What it models (and therefore what the tracker has to cope with):

* the loop is divided into 10 m segments; per lap the camera takes ONE frame per segment at a random
  position inside the segment (the camera sees ~2 m of belt, so a short defect is only caught in some
  laps -> detections flicker);
* every physical defect is put at a fixed place on the loop and re-appears every lap, with a fresh
  photometric variation (lighting, sensor noise, dust specks);
* a few % of frames are bad (blurred / dark): the quality gate must reject them;
* two vulcanised splices look like a stripe across the belt and are visible every lap;
* the encoder reports the belt position with ~1 m noise.

Defect images are drawn from the public dataset (or the synthetic stand-in); the *ground truth* of
where defects were placed is written to ``data/vision_truth.json`` so the run can be scored.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .. import config
from .detector import DefectDetector
from .synth import add_splice_band
from .train import list_dataset
from .tracker import near_splice

TRUTH_PATH = config.DATA_DIR / "vision_truth.json"


@lru_cache(maxsize=512)
def _load(path: str) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((256, 256))
        return im.copy()


def circ_dist(a: float, b: float) -> float:
    d = abs(a - b) % config.LOOP_LENGTH_M
    return min(d, config.LOOP_LENGTH_M - d)


def build_belt(rng: np.random.Generator, by_class: dict, normal_classes: list[str], defect_classes: list[str],
               n_defects: int) -> list[dict]:
    defects: list[dict] = []
    tries = 0
    while len(defects) < n_defects and tries < 5000:
        tries += 1
        pos = float(rng.uniform(0, config.LOOP_LENGTH_M))
        if any(circ_dist(pos, s) < 40 for s in config.KNOWN_SPLICES_M):
            continue
        if any(circ_dist(pos, d["position_m"]) < 80 for d in defects):
            continue
        cls = str(rng.choice(defect_classes))
        files = by_class.get(cls, [])
        if not files:
            continue
        defects.append({"position_m": pos, "length_m": float(rng.uniform(1.5, 5.0)), "class": cls,
                        "file": str(files[int(rng.integers(len(files)))])})
    return defects


def photometric(img: Image.Image, rng: np.random.Generator, bad_rate: float) -> tuple[Image.Image, str]:
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.85, 1.15)))
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 3.0, (img.size[1], img.size[0], 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.15:  # dust specks
        d = ImageDraw.Draw(img)
        for _ in range(int(rng.integers(2, 8))):
            x, y = rng.uniform(0, img.size[0]), rng.uniform(0, img.size[1])
            r = rng.uniform(0.5, 2.0)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(15, 15, 15))
    bad = ""
    if rng.random() < bad_rate:
        if rng.random() < 0.6:
            img, bad = img.filter(ImageFilter.GaussianBlur(radius=6)), "blur"
        else:
            img, bad = ImageEnhance.Brightness(img).enhance(0.04), "dark"
    return img, bad


def _post(api: str, path: str, payload: dict, timeout: float = 120.0) -> dict:
    r = requests.post(api + path, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def run(args: argparse.Namespace) -> None:
    detector = DefectDetector()
    rng = np.random.default_rng(args.seed)
    by_class: dict = {}
    for p, c in list_dataset(args.dataset):
        by_class.setdefault(c, []).append(p)
    defects = build_belt(rng, by_class, detector.normal_classes, detector.defect_classes, args.defects)
    normal_files = [str(p) for c in detector.normal_classes for p in by_class.get(c, [])]
    if not normal_files:
        raise SystemExit("dataset has no normal images")
    TRUTH_PATH.write_text(json.dumps({"defects": defects, "splices_m": list(config.KNOWN_SPLICES_M)}, indent=2))
    config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    start_lap = 1
    if args.reset:
        _post(args.api, "/vision/reset", {})
    else:
        s = requests.get(args.api + "/vision/summary", timeout=30).json()
        start_lap = (s.get("last_lap") or 0) + 1
    run_id = datetime.now(timezone.utc).strftime("%H%M%S")
    print(f"belt loop {config.LOOP_LENGTH_M:.0f} m, {config.N_SEGMENTS} segments of {config.SEGMENT_M:.0f} m, "
          f"{len(defects)} true defects, splices at {config.KNOWN_SPLICES_M}")

    for lap in range(start_lap, start_lap + args.laps):
        t0 = time.time()
        obs: list[dict] = []
        n_seg = config.N_SEGMENTS if not args.max_frames else min(config.N_SEGMENTS, args.max_frames)
        seg_ids = random.Random(lap).sample(range(config.N_SEGMENTS), n_seg) if args.max_frames else range(n_seg)
        for seg in sorted(seg_ids):
            center = seg * config.SEGMENT_M + float(rng.uniform(0, config.SEGMENT_M))
            hit = None
            for d in defects:
                if circ_dist(center, d["position_m"]) <= (d["length_m"] + config.CAMERA_FRAME_M) / 2:
                    hit = d
                    break
            if hit is not None:
                content = _load(hit["file"])
            else:
                content = _load(normal_files[int(rng.integers(len(normal_files)))])
            if near_splice(center):
                content = add_splice_band(content, rng)
            img, bad = photometric(content, rng, args.bad_frame_rate)
            obs.append({"seg": seg, "center": center, "img": img, "bad": bad, "truth": hit is not None})

        results = []
        for i in range(0, len(obs), 128):
            results.extend(detector.analyse_batch([o["img"] for o in obs[i:i + 128]]))

        payload = []
        for o, res in zip(obs, results):
            frame_id = f"{run_id}-L{lap:04d}-S{o['seg']:04d}"
            image_file = None
            if res["label"] in detector.defect_classes and res["score"] >= config.DETECTION_THRESHOLD and res["quality_ok"]:
                thumb = o["img"].copy()
                thumb.thumbnail((224, 224))
                image_file = f"{frame_id}.jpg"
                thumb.save(config.FRAMES_DIR / image_file, quality=85)
            payload.append({
                "frame_id": frame_id, "lap": lap,
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "position_m": o["center"] + float(rng.normal(0, 1.0)),   # encoder noise
                "label": res["label"], "score": res["score"], "severity": res["severity"],
                "quality_ok": res["quality_ok"], "quality_note": res["quality_note"] or o["bad"],
                "bbox": res["bbox"], "side": "carry", "image_file": image_file,
            })
        counts: dict = {}
        for i in range(0, len(payload), 250):
            c = _post(args.api, "/vision/observations", {"observations": payload[i:i + 250]})["counts"]
            for k, v in c.items():
                counts[k] = counts.get(k, 0) + v
        print(f"lap {lap}: {len(payload)} frames in {time.time() - t0:.1f}s -> {counts}")
        if args.lap_delay > 0:
            time.sleep(args.lap_delay)

    score_run(args.api, defects)


def score_run(api: str, defects: list[dict]) -> None:
    s = requests.get(api + "/vision/summary", timeout=30).json()
    tracks = requests.get(api + "/vision/tracks", params={"statuses": "confirmed,suspected"}, timeout=30).json()
    confirmed = [t for t in tracks if t["status"] == "confirmed"]
    matched = set()
    false_tracks = 0
    for t in confirmed:
        hits = [i for i, d in enumerate(defects) if circ_dist(t["position_m"], d["position_m"]) <= 25.0]
        if hits:
            matched.add(hits[0])
        else:
            false_tracks += 1
    print("\n=== loop handling result ===")
    print(f"frames analysed            : {s['frames']}")
    print(f"raw defect detections      : {s['defect_frames']}")
    print(f"unique open defect tracks  : {s['unique_open_defects']} (dedup ratio {s['dedup_ratio']})")
    print(f"true defects on the belt   : {len(defects)}")
    print(f"confirmed & matched        : {len(matched)} / {len(defects)}")
    print(f"confirmed but no real defect: {false_tracks}")
    print(f"ignored splice frames      : {s['splice_frames_ignored']}, rejected bad frames: {s['low_quality_frames']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=config.API_URL)
    ap.add_argument("--dataset", type=Path, default=config.VISION_DATASET_DIR)
    ap.add_argument("--laps", type=int, default=6)
    ap.add_argument("--lap-delay", type=float, default=2.0, help="seconds to wait between laps")
    ap.add_argument("--defects", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=0, help="subsample: frames per lap (0 = every segment)")
    ap.add_argument("--bad-frame-rate", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--keep", dest="reset", action="store_false", help="do not clear earlier vision results")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
