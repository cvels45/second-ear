"""Run the full pipeline on dev sessions and score findings against gold turns by time.

Transcripts:
  asr   cached Voice Transcribe turns from scripts/run_asr.py (data/runs/asr/dev-utt)
  gold  the scripted text with the TTS turn times (ablation: what ASR costs us)

Each session's findings go to data/runs/e2e/<transcript>/<run_id>/<session>.json, metrics to metrics.json.
Judge, holistic, and verifier calls are cached, so a re-run of unchanged pairs is free.

Usage (from second-ear/):
  uv run python scripts/run_e2e.py --per-lang 2                 # 10 sessions, about $6
  uv run python scripts/run_e2e.py --transcript gold --per-lang 2
  uv run python scripts/run_e2e.py --only es-S01 --yes
  uv run python scripts/run_e2e.py --per-lang 2 --adjudicate     # + audio check of every finding
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_ear.data.sessions import LOCKED, SETS, asr_dir, load_set  # noqa: E402
from second_ear.eval.e2e_eval import aggregate, score_links, score_session  # noqa: E402
from second_ear.hearing_pass import hearing_review  # noqa: E402
from second_ear.llm import BudgetExceeded, LLMClient  # noqa: E402
from second_ear.pairing import lexical_sim  # noqa: E402
from second_ear.pipeline import (  # noqa: E402
    Calibrator, adjudicate_findings, asr_to_pturns, gold_to_pturns, run_session, save,
)
from second_ear.settings import PKG_ROOT, load_settings  # noqa: E402

EST_PER_PAIR = 0.026  # 6 judges + holistic at minimal, ~55% of pairs verified with 3 votes at low
EST_PER_ADJ = 0.004  # one Spark 1.2 call with ~10 s of audio per finding


def pick_sessions(refs: list, args) -> list:
    if args.only:
        return [r for r in refs if r.session_id in args.only]
    if args.langs:
        refs = [r for r in refs if r.gold["lang"] in args.langs]
    if args.per_lang:
        by = defaultdict(list)
        for r in refs:
            by[r.gold["lang"]].append(r)
        refs = [r for rs in by.values() for r in rs[: args.per_lang]]
    return refs


def print_summary(name: str, m: dict) -> None:
    o = m["overall"]
    print(f"{name:12} P {o['precision']:.3f}  R {o['recall']:.3f}  F1 {o['f1']:.3f}  (tp {o['tp']}, fp {o['fp']}, "
          f"fn {o['fn']}) | false alarms/h {m['false_alarms_per_hour']} | per 100 clean {m['false_alarms_per_100_clean']}"
          f" | number/date R {m['number_date_recall']} | ts err {m['timestamp_error_median_s']} s"
          + (f" | reachable {m['reachable_share']}" if "reachable_share" in m else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="dev", choices=SETS)
    ap.add_argument("--transcript", default="asr", choices=["asr", "gold"])
    ap.add_argument("--locked-test-run", action="store_true", help="required for t4-test and t3 (run once, at the freeze)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--per-lang", type=int, help="first N sessions per language")
    ap.add_argument("--model", default="muse-spark-1.3")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument("--holistic-model", help="model for the holistic pass (default: --model)")
    ap.add_argument("--holistic-reasoning")
    ap.add_argument("--sim", choices=["labse", "lexical"], default="labse")
    ap.add_argument("--calibrator", type=Path, default=PKG_ROOT / "eval" / "calibrator.json")
    ap.add_argument("--provider", default="meta", choices=["meta", "local", "mock"])
    ap.add_argument("--adjudicate", action="store_true",
                    help="Spark 1.2 listens to both clips of each finding (interpreter error vs transcription)")
    ap.add_argument("--dual-asr", action="store_true", help="with --adjudicate: Omnilingual ASR transcribes each clip too")
    ap.add_argument("--hearing-pass", action="store_true",
                    help="Muse Wonder whole-hearing pass; scores decision links against the dev decisions")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--cap", type=float, help="per-run spend cap in USD (default 10, or SE_SPEND_CAP_PER_RUN)")
    args = ap.parse_args()
    s = load_settings()
    if args.cap:
        s.spend_cap_per_run = args.cap
    if args.set in LOCKED and not args.locked_test_run:
        print(f"{args.set} is a locked test set. Run it once, at the freeze, with --locked-test-run.")
        return 2
    adir = asr_dir(s.runs_dir, args.set)
    refs = {r.session_id: r for r in pick_sessions(load_set(s.data_dir, args.set), args)}
    sessions = []
    for did, ref in refs.items():
        gold = ref.gold
        if args.transcript == "asr":
            cp = adir / f"{did}.json"
            if not cp.exists():
                print(f"{did}: no cached ASR (run scripts/run_asr.py --set {args.set}); skipped")
                continue
            turns = asr_to_pturns(json.loads(cp.read_text(encoding="utf-8")), gold["lang"])
        else:
            turns = gold_to_pturns(gold)
        sessions.append((did, gold, turns))
    n_pairs = sum(len(t) // 2 for _, _, t in sessions)
    est = n_pairs * (EST_PER_PAIR + (0.3 * EST_PER_ADJ if args.adjudicate else 0)) if args.provider == "meta" else 0.0
    cal = Calibrator(args.calibrator)
    print(f"{len(sessions)} sessions ({args.transcript} transcripts), about {n_pairs} pairs, about ${est:.2f} before "
          f"cache hits | calibrator: {'threshold %.3f' % cal.threshold if cal.w is not None else 'none (verifier majority)'}")
    if args.provider == "meta" and not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return 1

    sim = lexical_sim
    if args.sim == "labse":
        try:
            from second_ear.sim import LabseSim

            sim = LabseSim()
        except ImportError as exc:
            print(f"LaBSE unavailable ({exc}); lexical similarity")
    client = LLMClient(provider=args.provider, settings=s)
    omni = None
    if args.adjudicate and args.dual_asr:
        from second_ear.asr.omnilingual import load as load_omni

        omni = load_omni()
    out_dir = s.runs_dir / "e2e" / args.set / args.transcript / client.ledger.run_id
    rows, t0 = [], time.time()
    for did, gold, turns in sessions:
        try:
            res = run_session(did, gold["lang"], turns, client=client, sim=sim, calibrator=cal,
                              duration_s=gold.get("duration_s", 0.0), model=args.model, reasoning=args.reasoning,
                              holistic_model=args.holistic_model, holistic_reasoning=args.holistic_reasoning)
        except BudgetExceeded as exc:
            print(f"Stopped: {exc}")
            break
        save(res, out_dir / f"{did}.json")  # saved before the audio check, so a stop keeps the judging
        adj_failed, stop = 0, False
        if args.adjudicate:
            try:
                adj_failed = adjudicate_findings(res, refs[did].wav, out_dir / "clips" / did, client, omni=omni)
            except BudgetExceeded as exc:
                print(f"Stopped during the audio check: {exc}")
                stop = True
            save(res, out_dir / f"{did}.json")
        d = res.to_dict()
        sc = score_session(d["findings"], gold, pairs=d["pairs"])
        if args.adjudicate:
            kept = [f for f in d["findings"] if f["check"] != "uncertain_transcription"]
            sc["adjudicated"] = score_session(kept, gold, pairs=d["pairs"])
            sc["adjudicated"]["demoted"] = len(d["findings"]) - len(kept)
            sc["adjudicated"]["failed"] = adj_failed
        if args.hearing_pass and not stop:
            dec = (gold.get("decision") or {}).get("text")
            try:
                rv = hearing_review(client, d["pairs"], d["findings"], gold["lang"], dec).model_dump()
                (out_dir / f"{did}.review.json").write_text(json.dumps(rv, ensure_ascii=False, indent=1), encoding="utf-8")
                sc["links"] = score_links(d["findings"], rv, gold)
                sc["consistency_findings"] = len(rv["consistency"])
            except Exception as exc:  # noqa: BLE001
                print(f"  hearing pass failed: {type(exc).__name__}: {str(exc)[:160]}")
        rows.append(sc)
        if stop:
            break
        print(f"{did}: {len(res.pairs)} pairs, {len(res.findings)} findings | errors {sc['errors']}, hit {sc['hits']}, "
              f"fp {sc['fp']} {sc['fp_by_status']} | failed {len(res.failed_pairs)} | ${res.cost_usd:.3f} | "
              f"{time.time() - t0:.0f} s", flush=True)
    if not rows:
        return 1
    m = aggregate(rows)
    if args.adjudicate:
        m["adjudicated"] = aggregate([r["adjudicated"] for r in rows])
        m["adjudicated"]["demoted"] = sum(r["adjudicated"]["demoted"] for r in rows)
    links = [r["links"] for r in rows if r.get("links")]
    if links:
        tot = {k: sum(x[k] for x in links) for k in links[0]}
        m["decision_links"] = tot | {
            "link_precision": round(tot["links_correct"] / max(1, tot["links"]), 3),
            "link_recall_of_detected": round(tot["linked_cited_errors"] / max(1, tot["detected_cited"]), 3),
            "top1_cited_share": round(tot["top1_cited"] / len(links), 3), "sessions_with_decision": len(links)}
    m.update(set=args.set, transcript=args.transcript, calibrator=str(args.calibrator) if cal.w is not None else None,
             threshold=cal.threshold, cost_usd=round(client.ledger.run_spent(), 4), sessions_scored=rows)
    (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print_summary("pipeline", m)
    if args.adjudicate:
        print_summary("adjudicated", m["adjudicated"])
        print(f"  (demoted to 'uncertain transcription': {m['adjudicated']['demoted']} findings)")
    if "decision_links" in m:
        dl = m["decision_links"]
        print(f"decision links: precision {dl['link_precision']} | cited errors detected {dl['detected_cited']}/"
              f"{dl['cited_errors']}, linked {dl['linked_cited_errors']} | top-ranked finding is a cited error in "
              f"{dl['top1_cited']}/{dl['sessions_with_decision']} sessions")
    for lg, v in m["by_lang"].items():
        print(f"  {lg}: P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}")
    print(f"\nCost ${m['cost_usd']:.4f}. Saved {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
