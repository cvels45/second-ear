"""Run the Muse Spark judges (and optionally the one-prompt baseline) on labeled text pairs.

Sets:
  dev        gold text of the 36 dev dialogues (Claude-written labels)
  t1-calib   the 208-pair calibration split of the human-labeled T1 set (tuning allowed)
  t1-test    the locked 832-pair test split: refused unless --locked-test-run (run once, at the freeze)

Always does a dry run first and asks before spending. Responses are cached, so re-runs of unchanged
prompts are free.

Usage (from second-ear/):
  uv run python scripts/run_text_eval.py --set dev --limit 120
  uv run python scripts/run_text_eval.py --set t1-calib --baseline
  uv run python scripts/run_text_eval.py --set dev --provider local      # Glimmer on the PC, free
  uv run python scripts/run_text_eval.py --set t1-calib --verify         # + verifier votes and calibrator

--verify sends every claim (family judges and the one-prompt baseline) to the verifier, three votes each,
and saves the votes in flags.jsonl for scripts/calibrate.py. If eval/calibrator.json exists, it also
reports the calibrated pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from second_ear import calibrate  # noqa: E402
from second_ear.eval.text_eval import dev_pairs, sample, score, t1_pairs  # noqa: E402
from second_ear.judges import baseline_pair, families_for, judge_pair  # noqa: E402
from second_ear.llm import BudgetExceeded, LLMClient  # noqa: E402
from second_ear.settings import PKG_ROOT, load_settings  # noqa: E402
from second_ear.verifier import VERIFIER_VERSION, verify_pair  # noqa: E402

EST_PER_CALL = {"minimal": 0.0025, "low": 0.0050}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, choices=["dev", "t1-calib", "t1-test"])
    ap.add_argument("--limit", type=int, help="stratified sample of N pairs")
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--provider", default="meta", choices=["meta", "local", "mock"])
    ap.add_argument("--model", default="muse-spark-1.3")
    ap.add_argument("--reasoning", default="minimal", choices=["minimal", "low", "medium"])
    ap.add_argument("--baseline", action="store_true", help="also run the one-prompt text baseline")
    ap.add_argument("--verify", action="store_true",
                    help="verify every claim (3 votes); implies --baseline, whose claims are verified too")
    ap.add_argument("--holistic-model", help="model for the one-prompt holistic pass (default: --model)")
    ap.add_argument("--holistic-reasoning", help="reasoning effort for the holistic pass (default: --reasoning)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    ap.add_argument("--locked-test-run", action="store_true")
    ap.add_argument("--cap", type=float, help="per-run spend cap in USD (default 10, or SE_SPEND_CAP_PER_RUN)")
    args = ap.parse_args()
    s = load_settings()
    if args.cap:
        s.spend_cap_per_run = args.cap

    if args.set == "dev":
        items = dev_pairs(s.data_dir / "dev" / "dialogues.jsonl", set(args.langs) if args.langs else None)
    else:
        split = "calib" if args.set == "t1-calib" else "test"
        if split == "test" and not args.locked_test_run:
            print("t1-test is the locked test split. Run it once, at the freeze, with --locked-test-run.")
            return 2
        items = t1_pairs(s.data_dir / "test_human" / "pairs.jsonl", split)
    items = sample(items, args.limit)
    if args.verify:
        args.baseline = True
    model = args.model if args.provider == "meta" else s.local_llm_model

    calls = sum(len(families_for(p)) for p, _ in items) + (len(items) if args.baseline else 0)
    if args.verify:
        calls += int(0.55 * len(items)) * 3 * 2  # ~55% of pairs carry a claim; 3 votes at low reasoning (~2x cost)
    est = calls * EST_PER_CALL.get(args.reasoning, 0.01) if args.provider == "meta" else 0.0
    n_err = sum(g["label"] == "error" for _, g in items)
    print(f"{args.set}: {len(items)} pairs ({n_err} errors), {calls} calls, about ${est:.2f} before cache hits "
          f"({args.provider}, {model}, reasoning {args.reasoning})")
    if args.provider == "meta" and not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return 1

    client = LLMClient(provider=args.provider, settings=s)
    out_dir = s.runs_dir / "judges" / args.set / client.ledger.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    flags_by, base_by, errors = {}, {}, []
    t0 = time.time()

    votes_by = {}

    def work(p):
        fl, info = judge_pair(client, p, model=model, reasoning=args.reasoning)
        bl = (baseline_pair(client, p, model=args.holistic_model or model,
                            reasoning=args.holistic_reasoning or args.reasoning) if args.baseline else [])
        if args.verify and (fl or bl):
            votes_by[p.pair_id] = verify_pair(client, p, fl + bl, model=model)
        return p.pair_id, fl, bl, info

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, p) for p, _ in items]
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                pid, fl, bl, info = fut.result()
                flags_by[pid], base_by[pid] = fl, bl
            except BudgetExceeded as exc:
                print(f"Stopped: {exc}")
                ex.shutdown(cancel_futures=True)
                break
            except Exception as exc:  # keep going; report at the end
                errors.append(repr(exc)[:300])
            if k % 25 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} pairs, ${client.ledger.run_spent():.3f}, {time.time() - t0:.0f} s", flush=True)

    done = [(p, g) for p, g in items if p.pair_id in flags_by]
    metrics = {"set": args.set, "provider": args.provider, "model": model, "reasoning": args.reasoning,
               "holistic_model": args.holistic_model or model, "holistic_reasoning": args.holistic_reasoning or args.reasoning,
               "pairs": len(done), "errors": errors[:20], "cost_usd": round(client.ledger.run_spent(), 4),
               "pipeline_judges": score(done, flags_by)}
    if args.baseline:
        metrics["one_prompt_baseline"] = score(done, base_by)
    rows = []
    for p, g in done:
        fl = [asdict(f) for f in flags_by[p.pair_id]]
        bl = [asdict(f) for f in base_by.get(p.pair_id, [])]
        for f, v in zip(fl + bl, votes_by.get(p.pair_id, [])):
            f.update(votes=v.votes, yes=v.yes, major=v.major)
        rows.append({"pair_id": p.pair_id, "gold": g, "orig": p.orig, "rend": p.rend, "flags": fl, "baseline": bl})
    with (out_dir / "flags.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.verify:
        metrics["verifier_version"] = VERIFIER_VERSION

        def surviving(key: str, which: str) -> dict[str, list]:
            out = {}
            for p, _ in done:
                fs = list(flags_by[p.pair_id]) if which != "baseline" else []
                fs += list(base_by.get(p.pair_id, [])) if which != "judges" else []
                vs = votes_by.get(p.pair_id, [])
                out[p.pair_id] = [v.flag for v in vs if v.flag in fs and getattr(v, key) / max(1, v.votes) >= 0.5]
            return out
        metrics["verified_judges"] = score(done, surviving("yes", "judges"))
        metrics["verified_all"] = score(done, surviving("yes", "all"))
        metrics["verified_all_major"] = score(done, surviving("major", "all"))
        cal = PKG_ROOT / "eval" / "calibrator.json"
        if cal.exists():
            try:
                w, thr = calibrate.load(cal)
                pr = calibrate.predict(w, np.array([calibrate.features(r) for r in rows]))
                keep = {p.pair_id: (flags_by[p.pair_id] + base_by.get(p.pair_id, [])) if q >= thr else []
                        for (p, _), q in zip(done, pr)}
                metrics["calibrated"] = score(done, keep)
                metrics["calibrated"]["threshold"] = thr
                if args.set in json.loads(cal.read_text(encoding="utf-8")).get("trained_on", []):
                    print(f"note: the calibrator was fit on {args.set}; its numbers here are in-sample "
                          "(scripts/calibrate.py reports out-of-fold numbers)")
            except ValueError as exc:
                print(f"calibrator skipped: {exc}")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    m = metrics["pipeline_judges"]["overall"]
    print(f"\njudges:   P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  (tp {m['tp']}, fp {m['fp']}, fn {m['fn']})")
    for key, name in (("one_prompt_baseline", "baseline"), ("verified_judges", "verified judges"),
                      ("verified_all", "verified all"), ("verified_all_major", "verified major"),
                      ("calibrated", "calibrated")):
        if key in metrics:
            b = metrics[key]["overall"]
            print(f"{name + ':':16}P {b['precision']:.3f}  R {b['recall']:.3f}  F1 {b['f1']:.3f}  "
                  f"(clean flagged: {sum(v for k, v in metrics[key]['negatives'].items() if k.endswith('_flagged'))})")
    print(f"negatives flagged: {metrics['pipeline_judges']['negatives']}")
    print(f"{len(errors)} failed pairs. Cost ${metrics['cost_usd']:.4f}. Saved {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
