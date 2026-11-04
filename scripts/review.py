"""Review an interpreted recording and write the HTML report: Second Ear's review mode, end to end.

  audio -> Voice Transcribe (one request per utterance) -> language ID and turn pairing (LaBSE)
  -> Muse Spark family judges + holistic pass -> verifier votes -> calibrated findings
  -> Muse Spark 1.2 listens to both clips of each finding -> Muse Wonder whole-hearing pass -> report.html

Usage (from second-ear/):
  uv run python scripts/review.py ..\\data\\dev\\sessions\\es-S02.mp3 --lang es --title "Mock merits hearing"
  uv run python scripts/review.py hearing.mp3 --lang zh --decision decision.txt --out ..\\data\\reviews\\case1

Every model call is cached, so re-running a reviewed recording only re-renders the report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_ear import audio, report  # noqa: E402
from second_ear.asr.session import transcribe_session  # noqa: E402
from second_ear.hearing_pass import hearing_review  # noqa: E402
from second_ear.llm import LLMClient  # noqa: E402
from second_ear.pairing import lexical_sim  # noqa: E402
from second_ear.pipeline import Calibrator, adjudicate_findings, asr_to_pturns, run_session, save  # noqa: E402
from second_ear.settings import PKG_ROOT, load_settings  # noqa: E402

VT_LANG = {"es": "Spanish", "pt": "Portuguese", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic", "ta": "Tamil"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio", type=Path)
    ap.add_argument("--lang", required=True, choices=sorted(VT_LANG))
    ap.add_argument("--title")
    ap.add_argument("--decision", type=Path, help="text file with the decision (optional)")
    ap.add_argument("--out", type=Path, help="output folder (default: data/reviews/<audio name>)")
    ap.add_argument("--no-adjudicate", action="store_true")
    ap.add_argument("--no-hearing-pass", action="store_true")
    ap.add_argument("--no-dual-asr", action="store_true", help="skip the second transcription (Omnilingual)")
    ap.add_argument("--calibrator", type=Path, default=PKG_ROOT / "eval" / "calibrator.json")
    ap.add_argument("--holistic-model", help="model for the holistic pass (default: Muse Spark 1.3)")
    ap.add_argument("--holistic-reasoning")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--cap", type=float, help="per-run spend cap in USD (default 10, or SE_SPEND_CAP_PER_RUN)")
    args = ap.parse_args()
    s = load_settings()
    if args.cap:
        s.spend_cap_per_run = args.cap
    out = args.out or s.data_dir / "reviews" / args.audio.stem
    out.mkdir(parents=True, exist_ok=True)

    wav = out / "audio16k.wav"
    if not wav.exists():
        audio.to_wav(args.audio, wav)
    secs = audio.duration_s(wav)
    pairs_est = max(1, int(secs / 7))
    est = secs / 3600 * 0.18 + pairs_est * 0.028 + 0.2
    print(f"{args.audio.name}: {secs / 60:.1f} min, about {pairs_est} exchanges, about ${est:.2f} before cache hits")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return 1

    client = LLMClient(settings=s)
    t0 = time.time()
    asr_path = out / "asr.json"
    if asr_path.exists():
        asr = json.loads(asr_path.read_text(encoding="utf-8"))
    else:
        res = transcribe_session(wav, settings=s, ledger=client.ledger, guard=client.guard,
                                 language_bias=[VT_LANG[args.lang], "English"])
        asr = res.to_dict()
        asr_path.write_text(json.dumps(asr, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  transcribed: {len(asr['turns'])} turns ({time.time() - t0:.0f} s)")

    try:
        from second_ear.sim import LabseSim

        sim = LabseSim()
    except ImportError:
        sim = lexical_sim
        print("  LaBSE unavailable; lexical pairing (uv sync --group gpu)")
    cal = Calibrator(args.calibrator)
    result = run_session(args.audio.stem, args.lang, asr_to_pturns(asr, args.lang), client=client, sim=sim,
                         calibrator=cal, duration_s=secs, holistic_model=args.holistic_model,
                         holistic_reasoning=args.holistic_reasoning)
    print(f"  judged: {len(result.pairs)} exchanges, {len(result.findings)} findings, "
          f"{len(result.failed_pairs)} failed ({time.time() - t0:.0f} s)")
    if not args.no_adjudicate:
        omni = None
        if not args.no_dual_asr:
            from second_ear.asr.omnilingual import load as load_omni

            omni = load_omni()
        failed = adjudicate_findings(result, wav, out / "clips", client, omni=omni)
        print(f"  audio-checked: {sum(f.check == 'interpreter_error' for f in result.findings)} confirmed, "
              f"{sum(f.check == 'uncertain_transcription' for f in result.findings)} to check, {failed} failed")
    decision = args.decision.read_text(encoding="utf-8") if args.decision else None
    review = None
    if not args.no_hearing_pass:
        d = result.to_dict()
        try:
            review = hearing_review(client, d["pairs"], d["findings"], args.lang, decision).model_dump()
            (out / "hearing_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  hearing pass: {len(review['consistency'])} consistency findings, "
                  f"{len(review['decision_links'])} decision links")
        except Exception as exc:  # noqa: BLE001 - the report still renders without it
            print(f"  hearing pass failed: {type(exc).__name__}: {str(exc)[:200]}")
    save(result, out / "findings.json")
    path = report.write(out / "report.html", result.to_dict(), title=args.title or args.audio.stem, review=review,
                        decision=decision, models={"judges": "Muse Spark 1.3", "audio check": "Muse Spark 1.2",
                                                   "hearing pass": "Muse Wonder 1.0", "speech": "Voice Transcribe",
                                                   "second transcription": "Omnilingual ASR"})
    print(f"\nReport: {path}\nCost ${client.ledger.run_spent():.3f}, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
