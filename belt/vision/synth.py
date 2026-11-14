"""Procedural belt images - a **stand-in** so the whole system runs before you download a dataset.

!!  These images are NOT the vision dataset of the submission.  The brief requires images from a
!!  public dataset; see the README ("Vision dataset") for where to get one and how to place it in
!!  ``data/vision_dataset/<class_name>/*.jpg``.  This module only exists to test the pipeline and the
!!  loop-handling logic end to end (and to make the unit/integration run possible offline).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 128
CLASSES = ["normal", "wear", "damage", "tear"]


def _base(rng: np.random.Generator) -> np.ndarray:
    level = rng.uniform(45, 75)
    noise = rng.normal(0, 7, (SIZE, SIZE))
    stripes = 3.0 * np.sin(np.linspace(0, rng.uniform(20, 40), SIZE))[None, :]   # woven-cord texture
    arr = level + noise + stripes
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(img, dtype=np.float32)
    rgb = np.stack([a * 0.96, a * 0.99, a * 1.06], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_image(kind: str, rng: np.random.Generator) -> Image.Image:
    img = Image.fromarray(_base(rng))
    d = ImageDraw.Draw(img, "RGBA")
    if kind == "wear":      # broad, low-contrast, lighter abraded area with scratches
        cx, cy = rng.uniform(30, 98, 2)
        rx, ry = rng.uniform(22, 45), rng.uniform(15, 35)
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(130, 130, 135, 90))
        for _ in range(rng.integers(4, 9)):
            x0, y0 = rng.uniform(0, SIZE, 2)
            d.line([x0, y0, x0 + rng.uniform(-40, 40), y0 + rng.uniform(-40, 40)], fill=(170, 170, 175, 110), width=1)
        img = img.filter(ImageFilter.GaussianBlur(0.6))
    elif kind == "damage":  # dark irregular gouge with a bright rim
        cx, cy = rng.uniform(30, 98, 2)
        pts = []
        for ang in np.linspace(0, 2 * np.pi, 11)[:-1]:
            r = rng.uniform(8, 24)
            pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
        d.polygon(pts, fill=(12, 12, 14, 255), outline=(200, 200, 200, 255))
    elif kind == "tear":    # long jagged crack across the belt
        y = rng.uniform(30, 98)
        pts = [(0, y)]
        for x in np.linspace(10, SIZE, 12):
            y = float(np.clip(y + rng.uniform(-9, 9), 5, SIZE - 5))
            pts.append((x, y))
        d.line(pts, fill=(8, 8, 8, 255), width=int(rng.integers(3, 6)))
        d.line([(x, yy - 3) for x, yy in pts], fill=(190, 190, 190, 200), width=1)
    return img


def add_splice_band(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """A vulcanised splice: a straight lighter band across the belt (normal, but looks odd)."""
    img = img.copy()
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    x = int(rng.uniform(0.3, 0.6) * w)
    d.rectangle([x, 0, x + max(3, int(0.055 * w)), h], fill=(150, 150, 150, 120))
    return img


def generate_dataset(root: Path, n_per_class: int = 150, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    n = 0
    for kind in CLASSES:
        out = root / kind
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            make_image(kind, rng).save(out / f"synthetic_{kind}_{i:04d}.jpg", quality=92)
            n += 1
    return n
