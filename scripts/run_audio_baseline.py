"""One-prompt audio baseline: give an audio LLM the whole recording and ask for every interpreting error.

This is what a lawyer could do today with one prompt. It is scored with the same time-based scorer
as the pipeline, with a generous tolerance (default 3 s around the time the model gives).

--concat joins the selected sessions of each language into one long recording (about 20 minutes for
6 dev sessions), to measure how the baseline degrades with hearing length.

Usage (from second-ear/):
  uv run python scripts/run_audio_baseline.py --per-lang 2                      # Spark 1.2, ~3-minute sessions
  uv run python scripts/run_audio_baseline.py --model muse-wonder-1.0 --per-lang 2
  uv run python scripts/run_audio_baseline.py --concat --langs es zh            # ~20-minute recordings
  uv run python scripts/run_audio_baseline.py --concat --chunk-min 5            # chunked baseline: 5-minute pieces
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel, Field  # noqa: E402

from second_ear import audio  # noqa: E402
from second_ear.data.sessions import LOCKED, SETS, load_set  # noqa: E402
from second_ear.eval.e2e_eval import aggregate, score_session  # noqa: E402
from second_ear.eval.text_eval import family_of  # noqa: E402
from second_ear.judges import LANG  # noqa: E402
from second_ear.llm import LLMClient, audio_message  # noqa: E402
from second_ear.settings import load_settings  # noqa: E402
from second_ear.taxonomy import ACCEPTABLE, ErrorType  # noqa: E402

TYPES = tuple(t.value for t in ErrorType)


class AudioFinding(BaseModel):
    time: str = Field(description="MM:SS (or H:MM:SS) where the interpreter's faulty rendition starts")
    type: Literal[TYPES]  # type: ignore[valid-type]
    speaker_said: str
    interpreter_said: str
    explanation: str


class AudioReview(BaseModel):
    findings: list[AudioFinding]


def prompt_a2(lang: str) -> str:
    """Revision 2 (Nov 2): walk the hearing exchange by exchange, and spell out what counts."""
    name = LANG.get(lang, lang)
    return (
        f"This recording is an interpreted immigration hearing in English and {name}. Questions are asked in English "
        f"and the interpreter renders each one into {name}; the respondent answers in {name} and the interpreter "
        "renders each answer into English. The English renditions become the official record.\n\n"
        "Work through the recording exchange by exchange. For every exchange, compare what was said with the "
        "interpreter's rendition and report each meaning change: a number, count, date, age, or duration changed; a "
        "name changed; anything left out or added; yes and no reversed; certainty changed (for example 'I think' "
        "dropped); force softened or intensified; pronouns or who did what changed; a legal term changed; a question "
        "rendered as a different question; a correction the interpreter made mid-sentence; and any exchange in "
        f"{name} that was never rendered into English.\n\nFor each error give the time (MM:SS) where the "
        "interpreter's rendition starts, the speaker's words, the interpreter's words, and why the meaning changed. "
        f"Do not report paraphrase, word order, or politeness changes. Not errors: {ACCEPTABLE}"
    )


def prompt(lang: str) -> str:
    name = LANG.get(lang, lang)
    return (
        f"This is a recording of an interpreted immigration hearing in English and {name}. The judge and the "
        f"attorneys speak English; the respondent speaks {name}; a court interpreter renders each question into "
        f"{name} and each answer into English, consecutively.\n\nYou are a check interpreter. Find every "
        "interpreting error that changes the meaning: omissions, additions, changed numbers, dates, ages, names, "
        "polarity, hedges, intensity, pronouns, legal terms, a question rendered as a different question, and "
        "exchanges the interpreter never rendered. For each, give the time where the interpreter's faulty "
        f"rendition starts, what the speaker said, what the interpreter said, and why it matters. Not errors: "
        f"{ACCEPTABLE} Return an empty list if the interpretation is faithful."
    )


def to_seconds(t: str) -> float | None:
    parts = re.findall(r"\d+(?:\.\d+)?", t)
    if not parts:
        return None
    vals = [float(x) for x in parts[-3:]]
    secs = 0.0
    for v in vals:
        secs = secs * 60 + v
    return secs


def concat_sessions(refs: list, out_dir: Path) -> tuple[str, dict, Path]:
    golds = [(r.session_id, r.gold) for r in refs]
    lang = golds[0][1]["lang"]
    sid = f"{lang}-concat{len(golds)}"
    wav = out_dir / f"{sid}.wav"
    segs = audio.concat([r.wav for r in refs], wav, gap_s=1.5)
    turns = []
    for (did, g), seg in zip(golds, segs):
        for t in g["turns"]:
            turns.append({**t, "start_s": round(t["start_s"] + seg.start_s, 3), "end_s": round(t["end_s"] + seg.start_s, 3)})
    gold = {"dialogue_id": sid, "lang": lang, "turns": turns, "duration_s": round(segs[-1].end_s, 3),
            "parts": [did for did, _ in golds]}
    mp3 = audio.to_mp3(wav, wav.with_suffix(".mp3"))
    (out_dir / f"{sid}.gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
    return sid, gold, mp3


def fit_inline(mp3: Path, total_s: float, out_dir: Path) -> Path:
    """Re-encode at a lower bitrate if base64 of the file would pass the 50 MB inline limit (a 3-hour
    hearing at 64 kbps is about 86 MB). Speech stays intelligible at 24 kbps; below 16 kbps we stop and
    let the call fail, which counts as a baseline result."""
    if mp3.stat().st_size * 4 / 3 <= 48_000_000:
        return mp3
    kbps = max(16, min(48, int(48_000_000 * 3 / 4 * 8 / 1000 / total_s)))
    out = out_dir / f"{mp3.stem}-{kbps}k.mp3"
    if not out.exists():
        audio.to_mp3(mp3, out, bitrate=f"{kbps}k")
    return out


def chunk(mp3: Path, total_s: float, minutes: float, out_dir: Path) -> list[tuple[float, Path, float]]:
    """Cut a recording at silences into pieces of at most `minutes`; returns (offset, path, duration)."""
    import soundfile as sf

    from second_ear import vad

    wav = out_dir / "full.wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not wav.exists():
        audio.to_wav(mp3, wav)
    x, sr = sf.read(str(wav), dtype="int16")
    cuts = vad.cut_points(vad.speech_regions(x, sr), total_s, max_chunk_s=minutes * 60, search_s=min(90.0, minutes * 20))
    bounds = list(zip([0.0, *cuts], [*cuts, total_s]))
    pieces = []
    for k, (a, b) in enumerate(bounds):
        pw = out_dir / f"c{k}.wav"
        sf.write(str(pw), x[int(a * sr): int(b * sr)], sr, subtype="PCM_16")
        pieces.append((a, audio.to_mp3(pw, pw.with_suffix(".mp3")), b - a))
    return pieces


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="muse-spark-1.2", choices=["muse-spark-1.2", "muse-wonder-1.0", "muse-spark-1.3"])
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--set", default="dev", choices=SETS)
    ap.add_argument("--locked-test-run", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--per-lang", type=int)
    ap.add_argument("--concat", action="store_true", help="one long recording per language")
    ap.add_argument("--tol", type=float, default=3.0, help="seconds of tolerance around the model's time")
    ap.add_argument("--prompt", default="a1", choices=["a1", "a2"], help="baseline prompt revision")
    ap.add_argument("--chunk-min", type=float, help="chunked baseline: split at silences into pieces of this many minutes")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--cap", type=float, help="per-run spend cap in USD (default 10, or SE_SPEND_CAP_PER_RUN)")
    args = ap.parse_args()
    s = load_settings()
    if args.cap:
        s.spend_cap_per_run = args.cap
    if args.set in LOCKED and not args.locked_test_run:
        print(f"{args.set} is a locked test set. Run it once, at the freeze, with --locked-test-run.")
        return 2
    refs = load_set(s.data_dir, args.set)
    if args.only:
        refs = [r for r in refs if r.session_id in args.only]
    if args.langs:
        refs = [r for r in refs if r.gold["lang"] in args.langs]
    by = defaultdict(list)
    for r in refs:
        by[r.gold["lang"]].append(r)
    if args.per_lang:
        by = {k: v[: args.per_lang] for k, v in by.items()}

    client = LLMClient(settings=s)
    tag = (f"audio-{args.model}{'-concat' if args.concat else ''}{f'-chunk{args.chunk_min:g}' if args.chunk_min else ''}"
           f"{'' if args.prompt == 'a1' else '-' + args.prompt}{'' if args.reasoning == 'low' else '-' + args.reasoning}")
    out_dir = s.runs_dir / "e2e" / args.set / tag / client.ledger.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for lang, rs in sorted(by.items()):
        if args.concat:
            jobs.append(concat_sessions(rs, out_dir))
        else:
            jobs += [(r.session_id, r.gold, r.mp3) for r in rs]
    secs = sum(g["duration_s"] for _, g, _ in jobs)
    price_in = 4.0 if "wonder" in args.model else 1.25
    est = secs * s.audio_tokens_per_sec * price_in / 1e6 + len(jobs) * 0.03
    print(f"{len(jobs)} recordings, {secs / 60:.1f} min of audio, {args.model}, about ${est:.2f}")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return 1

    rows, rows_wide = [], []
    for sid, gold, mp3 in jobs:
        t0 = time.time()
        pieces = (chunk(mp3, gold["duration_s"], args.chunk_min, out_dir / "chunks" / sid) if args.chunk_min
                  else [(0.0, fit_inline(mp3, gold["duration_s"], out_dir / "inline"), gold["duration_s"])])
        findings, cost, failed = [], 0.0, None
        for off, path, dur in pieces:
            try:
                text = (prompt_a2 if args.prompt == "a2" else prompt)(gold["lang"])
                r = client.chat(model=args.model, messages=audio_message(text, path), schema=AudioReview,
                                reasoning_effort=args.reasoning, max_completion_tokens=12000, audio_seconds=dur,
                                prompt_cache_key=f"audio-baseline-{args.prompt}", purpose=f"baseline:audio:{args.model}")
            except Exception as exc:  # noqa: BLE001 - a refusal or a length limit is a result
                failed = f"{type(exc).__name__}: {str(exc)[:200]}"
                break
            cost += r.cost_usd
            for f in r.parsed.findings:
                t = to_seconds(f.time)
                if t is None:
                    continue
                findings.append({"finding_id": f"{sid}-a{len(findings) + 1}", "pair_id": "", "rend_start_s": off + t,
                                 "rend_end_s": off + t + 1.0, "type": f.type, "family": family_of(f.type),
                                 "source_words": f.speaker_said, "rendition_words": f.interpreter_said,
                                 "explanation": f.explanation, "time": f.time, "chunk_offset_s": off})
        if failed:
            print(f"{sid}: FAILED {failed}")
            rows.append({"session_id": sid, "failed": True, "error": failed})
            continue
        sc = score_session(findings, gold, tol_s=args.tol)
        wide = score_session(findings, gold, tol_s=10.0)
        rows.append(sc)
        rows_wide.append(wide)
        (out_dir / f"{sid}.json").write_text(json.dumps({"session_id": sid, "model": args.model, "findings": findings,
                                                          "chunks": len(pieces)},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{sid}: {gold['duration_s'] / 60:.1f} min | {len(findings)} findings | errors {sc['errors']}, hit "
              f"{sc['hits']} (+-{args.tol:.0f} s), {wide['hits']} (+-10 s) | fp {sc['fp']} | ${cost:.3f} | "
              f"{time.time() - t0:.0f} s", flush=True)
    ok = [r for r in rows if not r.get("failed")]
    if not ok:
        (out_dir / "metrics.json").write_text(json.dumps({"set": args.set, "model": args.model, "failed": len(rows),
                                                           "errors": [r.get("error") for r in rows]}, indent=1),
                                              encoding="utf-8")
        print("Every recording failed; errors saved in metrics.json")
        return 1
    m = aggregate(ok)
    m.update(set=args.set, model=args.model, tol_s=args.tol, concat=args.concat, failed=len(rows) - len(ok),
             within_10s=aggregate(rows_wide)["overall"], cost_usd=round(client.ledger.run_spent(), 4), sessions_scored=ok)
    (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    o, w = m["overall"], m["within_10s"]
    print(f"\naudio baseline ({args.model}): P {o['precision']:.3f}  R {o['recall']:.3f}  F1 {o['f1']:.3f} (+-{args.tol:.0f} s)"
          f" | within 10 s: P {w['precision']:.3f}  R {w['recall']:.3f} | false alarms/h {m['false_alarms_per_hour']}"
          f" | number/date R {m['number_date_recall']} | ts err {m['timestamp_error_median_s']} s")
    print(f"Cost ${m['cost_usd']:.4f}. Saved {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
