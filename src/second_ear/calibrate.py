"""Pair-level calibrator: logistic regression over judge, holistic, and verifier signals.

Classical ML is the right tool here: a dozen features, a few hundred labeled pairs, and the goal is a
calibrated probability plus one threshold. Plain numpy, no extra dependency. Trained on dev pairs and
the T1 calibration split; the locked test split never trains it.

The threshold is chosen for precision at a realistic error rate, not on the label mix of the training
sets. Dev and T1 are about 30-50% errors; a real hearing has far fewer erroneous renditions, so the
same false-flag rate costs much more precision there:
    precision(prevalence) = TPR * prev / (TPR * prev + FPR * (1 - prev))
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Eight features. A 17-feature version (with one indicator per error family) fit the dev set well but
# transferred worse to the human-labeled T1 calibration split (Nov 2, out-of-fold on dev + T1 calib).
FEATURES = ["bias", "holistic_flag", "base_max_share", "judge_max_share", "judge_max_major", "judge_says_major",
            "typed_check", "n_families"]


def _shares(flags: list[dict], key: str) -> list[float]:
    return [f.get(key, 0) / f["votes"] for f in flags if f.get("votes")]


def features(row: dict) -> list[float]:
    """row: one pair from flags.jsonl. Judge and holistic flags carry verifier votes when --verify ran."""
    flags, base = row.get("flags", []), row.get("baseline", [])
    return [1.0, float(bool(base)), max(_shares(base, "yes"), default=0.0), max(_shares(flags, "yes"), default=0.0),
            max(_shares(flags, "major"), default=0.0), float(any(f.get("severity") == "major" for f in flags)),
            float(any(f.get("typed_check") for f in flags)), float(len({f["family"] for f in flags}))]


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 4000, lr: float = 0.3) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]] / len(y)
        w -= lr * g
    return w


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X @ w))


def rates(p: np.ndarray, y: np.ndarray, t: float) -> dict:
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {"threshold": round(float(t), 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "fpr": fp / max(1, fp + tn)}


def precision_at(tpr: float, fpr: float, prevalence: float) -> float:
    num = tpr * prevalence
    den = num + fpr * (1 - prevalence)
    return num / den if den else 0.0


def pr_curve(p: np.ndarray, y: np.ndarray) -> list[dict]:
    return [rates(p, y, t) for t in np.unique(np.round(p, 3))[::-1]]


def cross_val(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0, l2: float = 1.0) -> np.ndarray:
    """Out-of-fold probabilities, so the threshold is chosen on pairs the model did not train on."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    p = np.zeros(len(y))
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        w = fit(X[train], y[train], l2=l2)
        p[test] = predict(w, X[test])
    return p


def choose_threshold(p: np.ndarray, y: np.ndarray, target_precision: float, prevalence: float | None = None,
                     min_flagged: int = 5) -> tuple[float, dict]:
    """Lowest threshold (highest recall) whose out-of-fold precision meets the target.

    With prevalence set, precision is the expected precision at that error rate, from TPR and FPR.
    Falls back to the threshold with the best such precision when none meets the target.
    """
    best, best_r, fallback, fallback_r = None, None, 1.0, None
    for r in pr_curve(p, y):
        if r["tp"] + r["fp"] < min_flagged:
            continue
        prec = precision_at(r["recall"], r["fpr"], prevalence) if prevalence else r["precision"]
        r["adj_precision"] = prec
        if prec >= target_precision:
            best, best_r = r["threshold"], r
        if fallback_r is None or prec > fallback_r["adj_precision"]:
            fallback, fallback_r = r["threshold"], r
    return (best, best_r) if best is not None else (fallback, fallback_r or {})


def save(path: Path, w: np.ndarray, threshold: float, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"features": FEATURES, "weights": w.round(5).tolist(), "threshold": threshold,
                                **meta}, indent=1), encoding="utf-8")


def load(path: Path) -> tuple[np.ndarray, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if d["features"] != FEATURES:
        raise ValueError("calibrator features changed; refit with scripts/calibrate.py")
    return np.array(d["weights"]), float(d["threshold"])
