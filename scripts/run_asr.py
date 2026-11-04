"""Transcribe dev sessions (file endpoint + coverage check), pair turns, and score both against gold.

Results are cached per session in data/runs/asr/dev/<dialogue>.json; re-runs only transcribe new
sessions. Cost: $0.18 per audio hour (80 minutes of dev audio is about $0.25).

Usage (from second-ear/):
  uv run python scripts/run_asr.py                   # all dev sessions
  uv run python scripts/run_asr.py --only es-S01     # one session
  uv run python scripts/run_asr.py --rescore         # no API calls: re-pair and re-score cached ASR
  uv run python scripts/run_asr.py --set t4-calib    # voiced T1 sessions (also t4-test, t3)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_ear.asr.session import transcribe_session  # noqa: E402
from second_ear.data.sessions import SETS, asr_dir, load_set  # noqa: E402
from second_ear.eval.asr_eval import score_dialogue  # noqa: E402
from second_ear.llm import Ledger, SpendGuard  # noqa: E402
from second_ear.pairing import (  # noqa: E402
    PTurn, is_meta, lang_of, lexical_sim, merge_splits, pair_turns, tuned_params, uninterpreted_runs,
)
from second_ear.settings import load_settings  # noqa: E402

LANG_NAME = {"es": "Spanish", "pt": "Portuguese", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic", "ta": "Tamil"}


def to_pturns(asr: dict, src: str) -> list[PTurn]:
    out = []
    for i, t in enumerate(asr["turns"]):
        lang = lang_of(t["text"], src) if t["text"] else "unk"
        out.append(PTurn(i, t["start_s"], t["end_s"], t["text"], lang, is_meta(t["text"], lang), t["source"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="dev", choices=SETS)
    ap.add_argument("--only", nargs="*", help="dialogue ids")
    ap.add_argument("--rescore", action="store_true", help="use cached ASR only")
    ap.add_argument("--fresh", action="store_true", help="ignore cached ASR and transcribe again")
    ap.add_argument("--strategy", default="utterance", choices=["utterance", "chunk"])
    ap.add_argument("--sim", choices=["labse", "lexical"], default="labse",
                    help="pairing similarity (labse needs: uv sync --group gpu)")
    args = ap.parse_args()
    s = load_settings()
    # chunk results from the first run (Nov 1) stay in asr/dev for comparison; utterance results go to asr/<set>-utt
    out_dir = asr_dir(s.runs_dir, args.set) if args.strategy == "utterance" else s.runs_dir / "asr" / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(s.ledger_path)
    guard = SpendGuard(ledger, s.spend_cap_total, s.spend_cap_per_run)

    sim = lexical_sim
    if args.sim == "labse":
        try:
            from second_ear.sim import LabseSim

            sim = LabseSim()
            print(f"LaBSE on {sim.device}")
        except ImportError as exc:
            print(f"LaBSE unavailable ({exc}); using lexical similarity. Run: uv sync --group gpu")
    refs = load_set(s.data_dir, args.set)
    if args.only:
        refs = [r for r in refs if r.session_id in args.only]
    rows = []
    for ref in refs:
        did, gold = ref.session_id, ref.gold
        cache = out_dir / f"{did}.json"
        if cache.exists() and not args.fresh:
            asr = json.loads(cache.read_text(encoding="utf-8"))
        elif args.rescore:
            continue
        else:
            try:
                res = transcribe_session(ref.wav, settings=s, ledger=ledger, guard=guard,
                                         language_bias=[LANG_NAME[gold["lang"]], "English"],
                                         strategy=args.strategy)
            except Exception as exc:  # one bad session must not stop the batch
                print(f"{did}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                continue
            asr = res.to_dict()
            cache.write_text(json.dumps(asr, ensure_ascii=False, indent=1), encoding="utf-8")
        turns = merge_splits(to_pturns(asr, gold["lang"]))
        pairing = pair_turns(turns, sim=sim, **tuned_params())
        sc = score_dialogue(turns, pairing, gold)
        sc["uninterpreted_candidates"] = len(uninterpreted_runs(pairing))
        sc["rechecked"], sc["gaps"] = asr["rechecked"], asr["gaps"]
        rows.append(sc)
        print(f"{did}: coverage {sc['coverage']} | error {sc['error_rate']} | pairs {sc['pairs_hit']}/"
              f"{sc['pairs_gold']} (+{sc['pairs_wrong']} wrong) | rechecked {asr['rechecked']}, gaps {asr['gaps']}",
              flush=True)

    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    print("\nper language: pair recall, src/en error rate")
    for lang, rs in sorted(by_lang.items()):
        pr = sum(r["pairs_hit"] for r in rs) / max(1, sum(r["pairs_gold"] for r in rs))
        es = sum(r["error_rate"].get("src", 0) for r in rs) / len(rs)
        ee = sum(r["error_rate"].get("en", 0) for r in rs) / len(rs)
        print(f"  {lang}: pairs {pr:.3f} | src {es:.3f} | en {ee:.3f} | dialogues {len(rs)}")
    (out_dir / "_scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRun cost ${ledger.run_spent():.4f}. Scores in {out_dir / '_scores.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
