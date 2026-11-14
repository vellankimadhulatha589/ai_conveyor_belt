"""Failure prediction model.

* **What fails?**  a multi-class gradient-boosted classifier (``none`` + 5 failure modes) that
  outputs the probability that a failure of each mode will happen within its observable window.
* **When?**  three quantile regressors (20 % / 50 % / 80 %) on log(hours-to-failure), trained on
  positive snapshots only, give an ETA with an uncertainty band.

Gradient boosting (scikit-learn ``HistGradientBoosting*``) was chosen because it handles missing
values natively (belt stopped, sensor silent), needs no feature scaling, trains in seconds on CPU
and is robust on small, tabular, imbalanced data. Evidence for the prediction is derived from the
feature values themselves so that an operator can see *why* the model fired.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .. import config
from .features import FEATURE_NAMES

MODE_EVIDENCE = {
    "bearing": ["vibration", "temperature"],
    "mistrack": ["lateral_offset"],
    "slip": ["slip", "temperature", "motor_current"],
    "overload": ["motor_current", "belt_speed", "temperature"],
    "undervoltage": ["voltage", "motor_current", "belt_speed"],
}
QUANTILES = (0.2, 0.5, 0.8)


class FailurePredictor:
    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.clf: HistGradientBoostingClassifier | None = None
        self.regs: dict = {}
        self.eta_const: float = 48.0
        self.feature_names = list(FEATURE_NAMES)

    # ------------------------------------------------------------------ training
    def fit(self, X: pd.DataFrame, y: np.ndarray, tth: np.ndarray) -> "FailurePredictor":
        Xv = X[self.feature_names].to_numpy(dtype=float)
        self.clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=200, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=self.random_state,
        )
        self.clf.fit(Xv, y)
        pos = y > 0
        self.regs = {}
        if pos.sum() >= 40:
            target = np.log1p(tth[pos])
            for q in QUANTILES:
                reg = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_depth=3, learning_rate=0.08, max_iter=150,
                    early_stopping=False, random_state=self.random_state,
                )
                reg.fit(Xv[pos], target)
                self.regs[q] = reg
        if pos.any():
            self.eta_const = float(np.nanmedian(tth[pos]))
        return self

    # ------------------------------------------------------------------ inference
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xv = X[self.feature_names].to_numpy(dtype=float)
        proba = self.clf.predict_proba(Xv)
        full = np.zeros((len(Xv), len(config.CLASSES)))
        for j, c in enumerate(self.clf.classes_):
            full[:, int(c)] = proba[:, j]
        out = pd.DataFrame(full, columns=[f"p_{c}" for c in config.CLASSES], index=X.index)
        out["p_fail"] = 1.0 - out["p_none"]
        modes = full[:, 1:]
        out["mode"] = [config.CLASSES[1 + int(k)] for k in modes.argmax(axis=1)]
        if self.regs:
            qs = np.column_stack([np.expm1(self.regs[q].predict(Xv)) for q in QUANTILES])
            qs = np.sort(qs, axis=1)
        else:
            qs = np.tile(self.eta_const, (len(Xv), 3))
        qs = np.clip(qs, 0.5, float(max(config.PREDICTION_WINDOW_H.values())))
        out["eta_low_h"], out["eta_h"], out["eta_high_h"] = qs[:, 0], qs[:, 1], qs[:, 2]
        return out

    def evidence(self, row: pd.Series, mode: str) -> str:
        parts = []
        for ch in MODE_EVIDENCE.get(mode, []):
            v = row.get(f"{ch}_mean6")
            win = "6h"
            if v is None or not np.isfinite(v):
                v, win = row.get(f"{ch}_mean24"), "24h"
            if v is not None and np.isfinite(v):
                parts.append(f"{ch} {v:+.1f} sigma ({win} mean)")
        return "; ".join(parts) if parts else "insufficient recent data"

    # ------------------------------------------------------------------ persistence
    def save(self, path=None) -> None:
        path = path or config.PREDICTOR_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=None) -> "FailurePredictor | None":
        try:
            return joblib.load(path or config.PREDICTOR_PATH)
        except (FileNotFoundError, OSError):
            return None
