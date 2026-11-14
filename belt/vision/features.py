"""Hand-crafted image features (Pillow + numpy only).

Why not a CNN? The project must run on a laptop CPU without a GPU or a 2 GB PyTorch install, the
public datasets for belt damage are small (hundreds to a few thousand images), and a classical
texture/gradient pipeline is fast, explainable and surprisingly strong on this kind of problem
(belt damage = abrupt texture/intensity change on an otherwise uniform rubber surface). The
``DefectDetector`` interface hides the model, so a fine-tuned CNN can replace it later.

Feature groups: colour statistics, grey histogram, gradient-magnitude histogram and edge density,
local-binary-pattern texture histogram, and a 6 x 6 grid of mean intensity / mean gradient that gives
the model (and the anomaly map) a coarse notion of *where* things are.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = 96
GRID = 6
_SHIFTS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def to_array(img: Image.Image, size: int = SIZE) -> np.ndarray:
    img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def gradient_magnitude(g: np.ndarray) -> np.ndarray:
    gx = np.diff(g, axis=1, append=g[:, -1:])
    gy = np.diff(g, axis=0, append=g[-1:, :])
    return np.sqrt(gx * gx + gy * gy)


def lbp_hist(g: np.ndarray) -> np.ndarray:
    c = g[1:-1, 1:-1]
    code = np.zeros(c.shape, dtype=np.int32)
    h, w = g.shape
    for k, (dy, dx) in enumerate(_SHIFTS):
        nb = g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        code += (nb >= c).astype(np.int32) << k
    hist = np.bincount((code >> 3).ravel(), minlength=32).astype(np.float32)
    return hist / max(code.size, 1)


def compute(img: Image.Image) -> dict:
    """Return the feature vector, the per-cell grids and quality indicators of one frame."""
    a = to_array(img)
    g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    mag = gradient_magnitude(g)
    cs = SIZE // GRID
    cell_int = g.reshape(GRID, cs, GRID, cs).mean(axis=(1, 3))
    cell_grad = mag.reshape(GRID, cs, GRID, cs).mean(axis=(1, 3))
    color = np.concatenate([a.mean(axis=(0, 1)), a.std(axis=(0, 1))])
    ghist = np.histogram(g, bins=16, range=(0, 1))[0] / g.size
    mhist = np.histogram(np.clip(mag, 0, 0.5), bins=12, range=(0, 0.5))[0] / mag.size
    grad_stats = np.array([mag.mean(), np.percentile(mag, 95), (mag > 0.15).mean()])
    pct = np.percentile(g, [1, 99])
    feat = np.concatenate([color, ghist, mhist, grad_stats, lbp_hist(g), cell_int.ravel(), cell_grad.ravel(), pct]).astype(np.float32)
    return {
        "features": feat, "cell_int": cell_int, "cell_grad": cell_grad,
        "brightness": float(g.mean()), "sharpness": float(mag.mean()),
    }
