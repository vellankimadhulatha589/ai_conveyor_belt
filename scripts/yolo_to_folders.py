"""Convert a YOLO-format detection dataset into the image-folder layout used by belt.vision.train.

    python scripts/yolo_to_folders.py --src ~/Downloads/belt_dataset --names tear,damage,wear
    python scripts/yolo_to_folders.py --src PATH            # class names are read from data.yaml if present

Many belt datasets (Kaggle, Roboflow, Ultralytics) ship as images + YOLO label files with bounding
boxes. The vision classifier here works per image, so each image gets ONE label:

* label file missing or empty            -> "normal"
* otherwise                              -> the class of the largest box in the file

Output: data/vision_dataset/<class>/<image>.jpg (images are copied, nothing is modified in --src).
This is a simplification (an image with a tear AND wear is labelled by the larger box); it is
listed under "what I would do differently" in the README.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from belt import config  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_names(src: Path) -> list[str] | None:
    for y in list(src.rglob("data.yaml")) + list(src.rglob("*.yaml")):
        text = y.read_text(errors="ignore")
        m = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
        if m:
            return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
        m = re.search(r"names\s*:\s*\n((?:\s+\d+\s*:\s*.+\n?)+)", text)
        if m:
            return [line.split(":", 1)[1].strip().strip("'\"") for line in m.group(1).strip().splitlines()]
    return None


def label_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, default=config.VISION_DATASET_DIR)
    ap.add_argument("--names", help="comma separated class names in YOLO class-id order")
    ap.add_argument("--normal-name", default="normal")
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",")] if args.names else read_names(args.src)
    if not names:
        raise SystemExit("could not read class names; pass --names a,b,c (YOLO class-id order)")
    print("classes:", names)
    counts: dict = {}
    for img in sorted(args.src.rglob("*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lab = label_for(img)
        cls = args.normal_name
        if lab.exists():
            best, best_area = None, 0.0
            for line in lab.read_text().splitlines():
                p = line.split()
                if len(p) == 5:                      # class cx cy w h
                    area = float(p[3]) * float(p[4])
                elif len(p) > 5:                     # segmentation polygon: class x1 y1 x2 y2 ...
                    xs, ys = [float(v) for v in p[1::2]], [float(v) for v in p[2::2]]
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                else:
                    continue
                if area >= best_area:
                    best, best_area = int(float(p[0])), area
            if best is not None and best < len(names):
                cls = names[best]
        out = args.dst / cls
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / f"{img.parent.name}_{img.name}")
        counts[cls] = counts.get(cls, 0) + 1
    print("copied:", counts, "->", args.dst)


if __name__ == "__main__":
    main()
