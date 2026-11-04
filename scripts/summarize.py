"""Summarize runs with 95% bootstrap intervals: text sets (resampling pairs) and audio sessions
(resampling 5-minute blocks, so one long hearing still gives an interval).

Usage (from second-ear/):
  uv run python scripts/summarize.py --text data/runs/judges/t1-calib/<run> --e2e data/runs/e2e/dev/asr/<run> \
      data/runs/e2e/dev/audio-muse-spark-1.2/<run>
Writes a Markdown table to stdout and, with --out, to a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from second_ear import calibrate  # noqa: E402
from second_ear.settings import PKG_ROOT  # noqa: E402

B = 2000
RNG = np.random.default_rng(2026)


def prf(tp, fp, fn):
    p = tp / np.maximum(1, tp + fp)
    r = tp / np.maximum(1, tp + fn)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(1e-9, p + r), 0.0)
    return p, r, f


def ci(x) -> str:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"{lo:.2f}-{hi:.2f}"


def text_systems(rows: list[dict]) -> dict[str, np.ndarray]:
    def share(f):
        return f["yes"] / f["votes"] if f.get("votes") else 0.0
    out = {"one-prompt text baseline": np.array([bool(r["baseline"]) for r in rows]),
           "family judges": np.array([bool(r["flags"]) for r in rows])}
    if any(f.get("votes") for r in rows for f in r["flags"] + r["baseline"]):
        out["+ verifier (majority)"] = np.array([any(share(f) >= 0.5 for f in r["flags"] + r["baseline"]) for r in rows])
        cal = PKG_ROOT / "eval" / "calibrator.json"
        if cal.exists():
            try:
                w, thr = calibrate.load(cal)
                p = calibrate.predict(w, np.array([calibrate.features(r) for r in rows]))
                out["+ calibrator"] = p >= thr
            except ValueError:
                pass
    return out


def summarize_text(run: Path) -> list[str]:
    rows = [json.loads(x) for x in (run / "flags.jsonl").read_text(encoding="utf-8").splitlines()]
    y = np.array([r["gold"]["label"] == "error" for r in rows])
    systems = text_systems(rows)
    n = len(rows)
    idx = RNG.integers(0, n, size=(B, n))
    base = systems["one-prompt text baseline"]
    lines = [f"\n**{run.parent.name}** ({n} pairs, {int(y.sum())} errors; run {run.name})\n",
             "| System | Precision | Recall | F1 | Clean flagged | F1 vs one-prompt |", "|---|---|---|---|---|---|"]
    for name, pred in systems.items():
        tp, fp, fn = (pred & y)[idx].sum(1), (pred & ~y)[idx].sum(1), (~pred & y)[idx].sum(1)
        p, r, f = prf(tp, fp, fn)
        bp, br, bf = prf((base & y)[idx].sum(1), (base & ~y)[idx].sum(1), (~base & y)[idx].sum(1))
        P, R, F = prf(np.array((pred & y).sum()), np.array((pred & ~y).sum()), np.array((~pred & y).sum()))
        fpr = (pred & ~y).sum() / max(1, (~y).sum())
        d = "" if name.startswith("one-prompt") else f"{float(F - prf(np.array((base & y).sum()), np.array((base & ~y).sum()), np.array((~base & y).sum()))[2]):+.2f} ({ci(f - bf)})"
        lines.append(f"| {name} | {float(P):.2f} ({ci(p)}) | {float(R):.2f} ({ci(r)}) | {float(F):.2f} ({ci(f)}) | {fpr:.1%} | {d} |")
    trained = []
    cal = PKG_ROOT / "eval" / "calibrator.json"
    if cal.exists():
        trained = json.loads(cal.read_text(encoding="utf-8")).get("trained_on", [])
    if run.parent.name in trained:
        lines.append(f"\nThe calibrator was fit on {run.parent.name}: its row is in-sample.")
    return lines


def summarize_e2e(runs: list[Path], block_s: float = 300.0) -> list[str]:
    lines = ["\n| System | Set | Hours | Precision | Recall | F1 | False alarms / h | Recall by third |",
             "|---|---|---|---|---|---|---|---|"]
    for run in runs:
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        blocks = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
        for s in m["sessions_scored"]:
            for e in s.get("error_items", []):
                blocks[(s["session_id"], int(e["start_s"] // block_s))][0 if e["hit"] else 2] += 1
            for f in s.get("flagged_items", []):
                if not f["tp"]:
                    blocks[(s["session_id"], int(f["start_s"] // block_s))][1] += 1
        arr = np.array(list(blocks.values())) if blocks else np.zeros((1, 3))
        idx = RNG.integers(0, len(arr), size=(B, len(arr)))
        tp, fp, fn = arr[idx, 0].sum(1), arr[idx, 1].sum(1), arr[idx, 2].sum(1)
        p, r, f = prf(tp, fp, fn)
        o = m["overall"]
        name = m.get("model") or ("pipeline" + (f" ({m['transcript']} transcript)" if "transcript" in m else ""))
        if m.get("concat"):
            name += ", whole recording"
        lines.append(f"| {name} | {m.get('set', '?')} | {m['hours']:.2f} | {o['precision']:.2f} ({ci(p)}) | "
                     f"{o['recall']:.2f} ({ci(r)}) | {o['f1']:.2f} ({ci(f)}) | {m['false_alarms_per_hour']} | "
                     f"{m.get('recall_by_third', '')} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text", nargs="*", type=Path, default=[])
    ap.add_argument("--e2e", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    lines = []
    if args.text:
        lines.append("## Text pairs (95% bootstrap intervals over pairs)")
        for r in args.text:
            lines += summarize_text(r)
    if args.e2e:
        lines.append("\n## Audio sessions (95% bootstrap intervals over 5-minute blocks)")
        lines += summarize_e2e(args.e2e)
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
