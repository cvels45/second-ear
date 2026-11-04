"""Fit the pair-level calibrator on verified judge runs and choose its threshold.

Reads flags.jsonl from the latest --verify run of each training set (or the run dirs you pass),
fits a logistic regression on judge, verifier, and holistic-pass features, and picks the threshold
from 5-fold out-of-fold probabilities. Writes eval/calibrator.json. Never reads the locked test split.

Usage (from second-ear/):
  uv run python scripts/calibrate.py
  uv run python scripts/calibrate.py --target-precision 0.6 --prevalence 0.1
  uv run python scripts/calibrate.py --runs data/runs/judges/dev/<run_id> data/runs/judges/t1-calib/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from second_ear import calibrate as C  # noqa: E402
from second_ear.settings import PKG_ROOT, load_settings  # noqa: E402

TRAIN_SETS = ["dev", "t1-calib"]


def latest_verified(set_dir: Path, holistic_model: str = "muse-spark-1.3") -> Path | None:
    """Latest run with verifier votes whose holistic pass used `holistic_model` (the pipeline default)."""
    for d in sorted((x for x in set_dir.glob("*") if x.is_dir()), reverse=True):
        m = d / "metrics.json"
        if not m.exists():
            continue
        meta = json.loads(m.read_text(encoding="utf-8"))
        if "verifier_version" in meta and meta.get("holistic_model", meta.get("model")) == holistic_model:
            return d
    return None


def rule_rates(rows: list[dict], y: np.ndarray, pred) -> dict:
    p = np.array([float(pred(r)) for r in rows])
    return C.rates(p, y, 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="*", type=Path, help="run dirs (default: latest verified run per set)")
    ap.add_argument("--target-precision", type=float, default=0.6)
    ap.add_argument("--threshold", choices=["p50", "prevalence"], default="p50",
                    help="p50: flag at calibrated probability >= 0.5; prevalence: the precision-at-error-rate rule")
    ap.add_argument("--prevalence", type=float, default=0.10,
                    help="share of renditions with a meaning error assumed for a real hearing")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--holistic-model", default="muse-spark-1.3", help="use runs whose holistic pass used this model")
    ap.add_argument("--out", type=Path, default=PKG_ROOT / "eval" / "calibrator.json")
    args = ap.parse_args()
    s = load_settings()

    runs = args.runs or [r for r in (latest_verified(s.runs_dir / "judges" / k, args.holistic_model) for k in TRAIN_SETS) if r]
    if not runs:
        print("No verified runs found. Run scripts/run_text_eval.py --set dev --verify (and t1-calib) first.")
        return 2
    rows, sets = [], []
    for r in runs:
        if "t1-test" in str(r):
            print(f"Refusing {r}: the locked test split never trains the calibrator.")
            return 2
        for line in (r / "flags.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.append(row)
            sets.append(r.parent.name)
        print(f"  {r.parent.name}/{r.name}: {sum(1 for x in sets if x == r.parent.name)} pairs")
    X = np.array([C.features(r) for r in rows])
    y = np.array([1.0 if r["gold"]["label"] == "error" else 0.0 for r in rows])
    sets = np.array(sets)
    print(f"{len(y)} pairs, {int(y.sum())} errors, {len(C.FEATURES)} features")

    oof = C.cross_val(X, y, l2=args.l2)
    thr_prev, at_prev = C.choose_threshold(oof, y, args.target_precision, args.prevalence)
    print(f"\nprevalence rule: threshold {thr_prev:.3f} for precision >= {args.target_precision} at "
          f"{args.prevalence:.0%} error rate (out of fold)")
    if at_prev.get("adj_precision", 0) < args.target_precision:
        print(f"  target not met: best expected precision is {at_prev.get('adj_precision', 0):.3f}")
    # Default rule (Nov 2): flag when the calibrated probability is at least 0.5. The prevalence rule, fit on
    # the pooled sets, landed at 0.90: T1's noisy "clean" labels pushed it up, and on dev audio it cut recall
    # from 0.90 to 0.71 for 3 fewer false alarms in 10 sessions.
    thr = thr_prev if args.threshold == "prevalence" else 0.5
    at = C.rates(oof, y, thr)
    print(f"saved threshold: {thr:.3f} ({args.threshold} rule)")

    def line(name: str, r: dict) -> str:
        adj = C.precision_at(r["recall"], r["fpr"], args.prevalence)
        return (f"  {name:24} P {r['precision']:.3f}  R {r['recall']:.3f}  clean flagged {r['fpr']:.3f}  "
                f"P@{args.prevalence:.0%} {adj:.3f}")

    print("\nout-of-fold, all training pairs:")
    rules = {
        "judges (any flag)": lambda r: bool(r["flags"]),
        "baseline (any flag)": lambda r: bool(r["baseline"]),
        "verified majority": lambda r: any(f.get("votes") and f["yes"] / f["votes"] >= 0.5
                                           for f in r["flags"] + r["baseline"]),
        "verified unanimous": lambda r: any(f.get("votes") and f["yes"] == f["votes"]
                                            for f in r["flags"] + r["baseline"]),
    }
    for name, pred in rules.items():
        print(line(name, rule_rates(rows, y, pred)))
    print(line("calibrated", C.rates(oof, y, thr)))
    per_set = {}
    for k in sorted(set(sets)):
        m = sets == k
        per_set[k] = C.rates(oof[m], y[m], thr)
        print(line(f"calibrated [{k}]", per_set[k]))

    print("\nPR curve (out of fold):")
    for r in C.pr_curve(oof, y)[:: max(1, len(np.unique(np.round(oof, 3))) // 12)]:
        print(f"  t {r['threshold']:.3f}  P {r['precision']:.3f}  R {r['recall']:.3f}  clean flagged {r['fpr']:.3f}")

    w = C.fit(X, y, l2=args.l2)
    print("\nweights:")
    for f, v in sorted(zip(C.FEATURES, w), key=lambda t: -abs(t[1])):
        print(f"  {f:18} {v:+.3f}")
    C.save(args.out, w, thr, {
        "trained_on": sorted(set(sets.tolist())), "runs": [f"{r.parent.name}/{r.name}" for r in runs],
        "n": int(len(y)), "n_errors": int(y.sum()), "target_precision": args.target_precision,
        "prevalence": args.prevalence, "l2": args.l2, "oof_at_threshold": at, "threshold_rule": args.threshold,
        "prevalence_rule_threshold": thr_prev,
        "oof_per_set": per_set,
    })
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
