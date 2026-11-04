"""Grid-search turn-pairing settings on cached ASR (dev and the T4 calibration split). No API calls.

Prints, per setting: gold pairs, predicted pairs, gold pairs hit, and exact pairs (one predicted pair
covering exactly one gold original and its rendition). Never reads the locked test sets.

Writes the best setting to eval/pairing.json, which the pipeline, live mode, and run_asr.py then use.

Usage (from second-ear/):  uv run python scripts/tune_pairing.py
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_ear.data.sessions import asr_dir, load_set  # noqa: E402
from second_ear.eval.asr_eval import map_turns  # noqa: E402
from second_ear.pairing import PARAMS_FILE, merge_splits, pair_turns  # noqa: E402
from second_ear.pipeline import asr_to_pturns  # noqa: E402
from second_ear.settings import load_settings  # noqa: E402


def main() -> int:
    s = load_settings()
    from second_ear.sim import LabseSim

    sim = LabseSim()
    data = []
    for name in ("dev", "t4-calib"):
        adir = asr_dir(s.runs_dir, name)
        for ref in load_set(s.data_dir, name):
            cp = adir / f"{ref.session_id}.json"
            if cp.exists():
                turns = merge_splits(asr_to_pturns(json.loads(cp.read_text(encoding="utf-8")), ref.gold["lang"]))
                data.append((name, turns, ref.gold))
    print(f"{len(data)} sessions with cached ASR")
    grid = list(itertools.product((3, 8), (1.0, 0.7, 0.4, 0.2, 0.0), (0.9, 0.6, 0.3)))
    results = []
    for mg, base, skip in grid:
        tot = {"dev": Counter(), "t4-calib": Counter()}
        for name, turns, gold in data:
            pr = pair_turns(turns, sim=sim, max_group=mg, base=base, skip_cost=skip)
            m = map_turns(turns, gold["turns"])
            gp = {(t["renders"], t["turn_id"]) for t in gold["turns"] if t["kind"] == "rendition" and t["renders"]}
            c = tot[name]
            c["gold"] += len(gp)
            c["pred"] += len(pr.pairs)
            hit = set()
            for p in pr.pairs:
                og = {m.get(i) for i in p.orig} - {None}
                rg = {m.get(i) for i in p.rend} - {None}
                hit |= {(o, r) for o in og for r in rg} & gp
                c["exact"] += len(og) == 1 and len(rg) == 1 and (next(iter(og)), next(iter(rg))) in gp
            c["hit"] += len(hit)
        row = " | ".join(f"{k}: gold {c['gold']} pred {c['pred']} hit {c['hit']} exact {c['exact']}" for k, c in tot.items())
        print(f"max_group {mg} base {base} skip {skip} | {row}", flush=True)
        exact = tot["dev"]["exact"] + tot["t4-calib"]["exact"]
        extra = tot["dev"]["pred"] + tot["t4-calib"]["pred"] - exact  # pairs that match no gold pair: cost and false alarms
        results.append(((exact - 0.5 * extra, exact),
                        {"max_group": mg, "base": base, "skip_cost": skip,
                         "exact": {k: c["exact"] for k, c in tot.items()}, "gold": {k: c["gold"] for k, c in tot.items()}}))
    best = max(results, key=lambda r: r[0])[1]
    PARAMS_FILE.write_text(json.dumps(best, indent=1), encoding="utf-8")
    print(f"\nBest (exact pairs minus half the extra pairs): {best}. Saved {PARAMS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
