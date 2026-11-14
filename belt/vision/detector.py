"""Frame-level defect detector: classification + coarse localisation + frame-quality gate.

For every frame it returns

* ``label`` / ``score``  - most likely defect class and P(defect) = 1 - P(normal classes);
* ``severity`` (0..1)    - a *heuristic*: half the model confidence, half the share of the frame whose
                           texture/intensity deviates from the healthy-belt reference. It is not
                           calibrated against real repair decisions (see README, "what I would do
                           differently");
* ``bbox``               - bounding box of the deviating cells, normalised to [0, 1];
* ``quality_ok``         - False for frames that are too dark/bright or too blurry (dust, motion).
                           The tracker never treats such a frame as evidence.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from .. import config
from . import features


class DefectDetector:
    def __init__(self, bundle_path: Path = config.VISION_MODEL_PATH):
        b = joblib.load(bundle_path)
        self.clf = b["model"]
        self.clf.n_jobs = 1
        self.classes: list[str] = b["classes"]
        self.normal_classes: list[str] = b["normal_classes"]
        self.defect_classes: list[str] = b["defect_classes"]
        self.ref = b["ref"]
        self.quality = b["quality"]
        self.metrics = b.get("metrics", {})
        cls = list(self.clf.classes_)
        self._normal_cols = [cls.index(self.classes.index(c)) for c in self.normal_classes
                             if self.classes.index(c) in cls]
        self._defect_cols = [(cls.index(self.classes.index(c)), c) for c in self.defect_classes
                             if self.classes.index(c) in cls]

    def _quality(self, brightness: float, sharpness: float) -> tuple[bool, str]:
        if brightness < self.quality["bright_lo"]:
            return False, "too dark"
        if brightness > self.quality["bright_hi"]:
            return False, "too bright / glare"
        if sharpness < 0.4 * self.quality["sharp_p05"]:
            return False, "blurred (dust or motion)"
        return True, ""

    def _anomaly_map(self, cell_grad: np.ndarray, cell_int: np.ndarray) -> tuple[float, list | None]:
        ref = self.ref
        sg = np.maximum(ref["grad_mad"], 0.25 * np.abs(ref["grad_med"]) + 1e-3)
        si = np.maximum(ref["int_mad"], 0.25 * np.abs(ref["int_med"]) + 1e-3)
        z = np.maximum(np.abs(cell_grad - ref["grad_med"]) / sg, np.abs(cell_int - ref["int_med"]) / si)
        flagged = z > 3.5
        area = float(flagged.mean())
        if not flagged.any():
            return area, None
        rows, cols = np.where(flagged)
        g = features.GRID
        return area, [round(cols.min() / g, 3), round(rows.min() / g, 3),
                      round((cols.max() + 1) / g, 3), round((rows.max() + 1) / g, 3)]

    def analyse_batch(self, images: list[Image.Image]) -> list[dict]:
        if not images:
            return []
        comp = [features.compute(im) for im in images]
        X = np.vstack([c["features"] for c in comp])
        proba = self.clf.predict_proba(X)
        out = []
        for k, c in enumerate(comp):
            p_norm = float(proba[k, self._normal_cols].sum()) if self._normal_cols else 0.0
            score = 1.0 - p_norm
            best_col, best_label = max(self._defect_cols, key=lambda cc: proba[k, cc[0]])
            area, bbox = self._anomaly_map(c["cell_grad"], c["cell_int"])
            is_defect = score >= config.DETECTION_THRESHOLD
            severity = min(1.0, 0.5 * score + 0.5 * min(1.0, area * 2.5)) if is_defect else 0.0
            ok, note = self._quality(c["brightness"], c["sharpness"])
            out.append({
                "label": best_label if is_defect else self.normal_classes[0],
                "score": round(score, 4), "severity": round(float(severity), 3),
                "bbox": bbox if is_defect else None, "quality_ok": ok, "quality_note": note,
            })
        return out

    def analyse(self, img: Image.Image) -> dict:
        return self.analyse_batch([img])[0]
