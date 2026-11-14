"""Train the vision defect classifier from an image-folder dataset.

    python -m belt.vision.train                 # uses data/vision_dataset/<class>/*.jpg
    python -m belt.vision.train --dataset PATH
    python -m belt.vision.train --synthetic     # generate the offline STAND-IN dataset first

Folder name = class label. Classes named normal/good/ok/healthy/... (see config.NORMAL_CLASS_NAMES)
are "normal", every other class is a defect type.

Honest evaluation note: many public belt datasets are frames cut from videos, so near-identical
images exist. A random split then overestimates accuracy. If your dataset provides a video/scene id,
split by it; the number printed here is an upper bound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .. import config
from . import features, synth

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_dataset(root: Path) -> list[tuple[Path, str]]:
    items = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in IMAGE_EXT:
                items.append((f, d.name))
    return items


def extract(paths: list[Path]) -> dict:
    feats, cg, ci, sharp, bright = [], [], [], [], []
    for p in paths:
        with Image.open(p) as im:
            r = features.compute(im)
        feats.append(r["features"])
        cg.append(r["cell_grad"])
        ci.append(r["cell_int"])
        sharp.append(r["sharpness"])
        bright.append(r["brightness"])
    return {"X": np.vstack(feats), "cell_grad": np.stack(cg), "cell_int": np.stack(ci),
            "sharp": np.array(sharp), "bright": np.array(bright)}


def augment(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Nuisance variation a real camera sees (lighting, sensor noise, dust on the lens/belt).

    Without it the classifier reads dark dust specks as "wear" and flags ~2 % of healthy frames, which
    over many laps creates false defect tracks. Ranges are deliberately wider than what the loop
    simulator (run_loop.photometric) applies, so the fix is not tuned to the test.
    """
    im = ImageEnhance.Brightness(im.convert("RGB")).enhance(float(rng.uniform(0.75, 1.25)))
    arr = np.asarray(im, dtype=np.float32)
    arr = arr + rng.normal(0, float(rng.uniform(0, 5.0)), arr.shape)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.4:
        d = ImageDraw.Draw(im)
        w, h = im.size
        for _ in range(int(rng.integers(2, 12))):
            x, y, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.5, 2.5) * max(1.0, w / 256)
            g = int(rng.integers(5, 40))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(g, g, g))
    return im


def extract_augmented(paths: list[Path], rng: np.random.Generator, copies: int) -> np.ndarray:
    rows = []
    for p in paths:
        with Image.open(p) as im:
            base = im.convert("RGB")
        for _ in range(copies):
            rows.append(features.compute(augment(base, rng))["features"])
    return np.vstack(rows) if rows else np.empty((0, 0))


def _defect_vs_normal(clf, X: np.ndarray, y_true: np.ndarray, normal_ids: set) -> dict:
    cols = [list(clf.classes_).index(i) for i in normal_ids if i in clf.classes_]
    p_def = 1.0 - clf.predict_proba(X)[:, cols].sum(axis=1)
    flagged = p_def >= config.DETECTION_THRESHOLD
    is_def = np.array([k not in normal_ids for k in y_true])
    tp, fp, fn = int((flagged & is_def).sum()), int((flagged & ~is_def).sum()), int((~flagged & is_def).sum())
    return {"precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
            "false_alarm_rate_on_normal": round(fp / max(int((~is_def).sum()), 1), 3)}


def _robust_ref(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(cells, axis=0)
    mad = 1.4826 * np.median(np.abs(cells - med), axis=0)
    return med, mad


def train(root: Path, out_path: Path = config.VISION_MODEL_PATH, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    items = list_dataset(root)
    if not items:
        raise SystemExit(f"No images found under {root}. Put images in {root}/<class_name>/*.jpg "
                         f"or run with --synthetic.")
    classes = sorted({c for _, c in items})
    normal = [c for c in classes if c.lower() in config.NORMAL_CLASS_NAMES]
    defects = [c for c in classes if c not in normal]
    if not normal or not defects:
        raise SystemExit(f"Need at least one normal class ({sorted(config.NORMAL_CLASS_NAMES)}) and one defect "
                         f"class, found {classes}")
    log(f"{len(items)} images, classes: " + ", ".join(f"{c}={sum(1 for _, k in items if k == c)}" for c in classes))
    log("extracting features ...")
    data = extract([p for p, _ in items])
    y = np.array([classes.index(c) for _, c in items])

    idx = np.arange(len(y))
    strat = y if np.bincount(y).min() >= 2 else None
    tr, te = train_test_split(idx, test_size=0.25, stratify=strat, random_state=0)
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)
    rng = np.random.default_rng(0)
    paths = [p for p, _ in items]
    log("augmenting (lighting / noise / dust) ...")
    aug_tr = extract_augmented([paths[i] for i in tr], rng, copies=2)
    X_fit = np.vstack([data["X"][tr], aug_tr])
    y_fit = np.concatenate([y[tr], np.repeat(y[tr], 2)])
    clf.fit(X_fit, y_fit)
    X_te_aug = extract_augmented([paths[i] for i in te], np.random.default_rng(1), copies=1)
    pred = clf.predict(data["X"][te])
    rep = classification_report(y[te], pred, labels=list(range(len(classes))), target_names=classes,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y[te], pred, labels=list(range(len(classes)))).tolist()
    normal_ids = {classes.index(c) for c in normal}
    clean = _defect_vs_normal(clf, data["X"][te], y[te], normal_ids)
    noisy = _defect_vs_normal(clf, X_te_aug, y[te], normal_ids)
    acc_aug = float((clf.predict(X_te_aug) == y[te]).mean())
    metrics = {
        "n_images": len(items), "classes": classes, "normal_classes": normal, "defect_classes": defects,
        "holdout_images": int(len(te)),
        "accuracy": round(float(rep["accuracy"]), 3),
        "per_class": {c: {k: round(float(rep[c][k]), 3) for k in ("precision", "recall", "f1-score")} for c in classes},
        "confusion_matrix(rows=true, cols=pred)": cm,
        "defect_vs_normal": clean,
        "accuracy_on_augmented_holdout": round(acc_aug, 3),
        "defect_vs_normal_on_augmented_holdout": noisy,
    }
    log(json.dumps(metrics, indent=2))

    # final model on everything + references for the anomaly map and the frame-quality gate
    aug_all = extract_augmented(paths, np.random.default_rng(2), copies=2)
    clf.fit(np.vstack([data["X"], aug_all]), np.concatenate([y, np.repeat(y, 2)]))
    is_normal_img = np.array([classes[k] in normal for k in y])
    g_med, g_mad = _robust_ref(data["cell_grad"][is_normal_img])
    i_med, i_mad = _robust_ref(data["cell_int"][is_normal_img])
    bundle = {
        "model": clf, "classes": classes, "normal_classes": normal, "defect_classes": defects,
        "ref": {"grad_med": g_med, "grad_mad": g_mad, "int_med": i_med, "int_mad": i_mad},
        # quality gate is derived from the dataset itself (a dark mine belt is not "too dark")
        "quality": {"sharp_p05": float(np.percentile(data["sharp"], 5)),
                    "bright_lo": float(max(0.02, 0.5 * np.percentile(data["bright"], 1))),
                    "bright_hi": float(min(0.98, 1.0 - 0.5 * (1.0 - np.percentile(data["bright"], 99))))},
        "metrics": metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    (out_path.parent / "vision_metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"saved {out_path}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.VISION_DATASET_DIR)
    ap.add_argument("--synthetic", action="store_true", help="generate the offline stand-in dataset first")
    ap.add_argument("--n-per-class", type=int, default=150)
    args = ap.parse_args()
    if args.synthetic:
        n = synth.generate_dataset(args.dataset, args.n_per_class)
        print(f"generated {n} SYNTHETIC stand-in images in {args.dataset} (not the submission dataset)")
    train(args.dataset)


if __name__ == "__main__":
    main()