"""Smoke test: answer the open API questions from docs/technical-review.md before building on them.

Checks (default run costs about $0.25; --long adds about $2):
  vt_file      Voice Transcribe file mode: diarized turns, speakers, numbers as digits or words
  vt_cap       Does the file endpoint reject audio over 10 minutes? (rejected requests aren't billed)
  vt_realtime  Can a file stream faster than real time through the realtime WebSocket?
  audio_llm    Spark 1.2 and Wonder audio input: accepted length, audio tokens per second
  determinism  Spark 1.3, temperature 0 + seed + minimal reasoning: identical output twice?
  structured   Spark 1.3 strict JSON schema with low reasoning: parses? tokens per judge call?
  cache        Prompt-prefix cache: cached_tokens > 0 on the second call?

Usage (from second-ear/):
  uv run python scripts/smoke_test.py            # all default checks, asks before spending
  uv run python scripts/smoke_test.py --long     # also 30- and 60-minute audio-LLM inputs
  uv run python scripts/smoke_test.py --only vt_file,cache --yes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_ear import audio  # noqa: E402
from second_ear.asr import VoiceTranscribeError, stream_file, transcribe_file  # noqa: E402
from second_ear.data import fleurs  # noqa: E402
from second_ear.llm import LLMClient, audio_message  # noqa: E402
from second_ear.settings import load_settings  # noqa: E402

ALL_CHECKS = ["vt_file", "vt_cap", "vt_realtime", "audio_llm", "determinism", "structured", "cache"]


# ---------------------------------------------------------------- audio fixtures


def build_sessions(data_dir: Path, long: bool) -> dict[str, Path]:
    """Alternate Spanish and English FLEURS readings of the same sentences (a crude 'interpreted' session)."""
    work = data_dir / "smoke"
    clips_dir = work / "clips"
    es = fleurs.fetch("es", data_dir / "fleurs", n=60)
    en = fleurs.fetch("en", data_dir / "fleurs", n=60)
    en_by_id = {c.id: c for c in en}
    pairs = [(c, en_by_id[c.id]) for c in es if c.id in en_by_id] or list(zip(es, en))
    ordered: list[Path] = []
    for a, b in pairs:
        for c in (a, b):
            out = clips_dir / f"{c.lang}_{c.id}.wav"
            if not out.exists():
                audio.to_wav(c.path, out)
            ordered.append(out)
    if not ordered:
        raise RuntimeError("No FLEURS clips downloaded")

    def build(name: str, min_seconds: float) -> Path:
        dst = work / f"{name}.wav"
        if dst.exists():
            return dst
        seq, total, i = [], 0.0, 0
        while total < min_seconds:
            p = ordered[i % len(ordered)]
            seq.append(p)
            total += audio.duration_s(p) + 0.6
            i += 1
        audio.concat(seq, dst)
        return dst

    sessions = {"s60": build("session_60s", 60), "s5m": build("session_5min", 300), "s11m": build("session_11min", 660)}
    sessions["s5m_mp3"] = audio.to_mp3(sessions["s5m"], work / "session_5min.mp3")
    if long:
        for name, secs in (("s30m", 1800), ("s60m", 3600)):
            wav = build(f"session_{secs // 60}min", secs)
            sessions[f"{name}_mp3"] = audio.to_mp3(wav, work / f"session_{secs // 60}min.mp3")
    return sessions


# ---------------------------------------------------------------- judge fixtures


class NumberCheck(BaseModel):
    error_present: bool
    error_type: Literal["none", "count", "ordinal", "date", "quantity", "omission"]
    source_span: str
    rendered_span: str
    proposed_correction: str
    explanation: str


JUDGE_SYSTEM = (
    "You check a court interpreter's rendering for meaning-level errors in numbers. "
    "Decide whether the English interpretation changes, drops, or adds any count, ordinal, date, or "
    "quantity from the source. Wording differences that keep the meaning are not errors. "
    "Keep explanation under 25 words."
)
JUDGE_USER = (
    'Source (Spanish): "Arrestaron a mi padre dos veces: la primera en 2019 y la segunda en marzo de 2021."\n'
    'Interpretation (English): "They arrested my father for the second time in March 2021."'
)

TAXONOMY_PREFIX = "\n".join(
    f"Rule {i}: " + line
    for i, line in enumerate(
        [
            "Omission: source content with legal or factual weight is missing from the rendering.",
            "Addition: the rendering adds content the speaker did not say.",
            "Distortion: the rendering changes what the speaker meant.",
            "Count vs ordinal: 'two times' and 'for the second time' are different facts.",
            "Referent change: same number, different thing ('fourth year of high school' vs 'fourth grade').",
            "Hedge change: dropped or added uncertainty ('I think', 'maybe', 'about').",
            "Pronoun shift: 'I' becomes 'we', 'he' becomes 'they', or similar.",
            "Polarity: yes/no or negation reversed.",
            "Softening: 'kidnapped' rendered as 'taken'; 'beaten' as 'hit'.",
            "Wrong question: the question was rendered differently, so the answer responds to something else.",
            "Self-correction: the interpreter corrects mid-rendering; both versions are on the record.",
            "Acceptable variation: different words, same meaning; never flag these.",
        ]
        * 12,
        start=1,
    )
)  # about 2,000+ tokens: long enough to test prefix caching


def judge_messages(user: str = JUDGE_USER, prefix: str = "") -> list[dict]:
    system = (prefix + "\n\n" + JUDGE_SYSTEM) if prefix else JUDGE_SYSTEM
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class AudioProbe(BaseModel):
    languages: list[str]
    first_sentence: str
    approx_minutes: float


# ---------------------------------------------------------------- checks


def check_vt_file(ctx: dict) -> dict:
    out = transcribe_file(
        ctx["sessions"]["s60"], settings=ctx["s"], ledger=ctx["ledger"], guard=ctx["guard"],
        mode="DIARIZATION", language_bias=["English", "Spanish"],
    )
    turns = out.get("turns", [])
    speakers = sorted({t.get("speaker") for t in turns if t.get("speaker")})
    text = " ".join(t.get("transcript", "") for t in turns) or out.get("transcript", "")
    (ctx["run_dir"] / "vt_file.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "turns": len(turns),
        "speakers": speakers,
        "has_digits": any(ch.isdigit() for ch in text),
        "first_turns": [(t.get("speaker"), t.get("startMs"), t.get("transcript", "")[:80]) for t in turns[:3]],
    }


def check_vt_cap(ctx: dict) -> dict:
    try:
        out = transcribe_file(
            ctx["sessions"]["s11m"], settings=ctx["s"], ledger=ctx["ledger"], guard=ctx["guard"],
            mode="ENDPOINTING", enforce_limits=False,
        )
        return {"rejected": False, "note": "11-minute file accepted (no 10-minute cap?)", "turns": len(out.get("turns", []))}
    except VoiceTranscribeError as exc:
        return {"rejected": True, "status": exc.status, "detail": exc.detail[:160]}


def check_vt_realtime(ctx: dict) -> dict:
    out = stream_file(
        ctx["sessions"]["s60"], settings=ctx["s"], ledger=ctx["ledger"], guard=ctx["guard"],
        mode="DIARIZATION", language_bias=["English", "Spanish"], pace=None,
    )
    with (ctx["run_dir"] / "vt_realtime_events.jsonl").open("w", encoding="utf-8") as f:
        for ev in out["events"]:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    types: dict[str, int] = {}
    for ev in out["events"]:
        types[ev.get("type", "?")] = types.get(ev.get("type", "?"), 0) + 1
    return {
        "audio_s": round(out["audio_s"], 1),
        "wall_s": round(out["wall_s"], 1),
        "speedup_vs_realtime": round(out["audio_s"] / out["wall_s"], 2) if out["wall_s"] else None,
        "final_turns": len(out["turns"]),
        "event_types": types,
    }


def check_audio_llm(ctx: dict) -> dict:
    client: LLMClient = ctx["client"]
    inputs = [("5min", ctx["sessions"]["s5m_mp3"])]
    if ctx["long"]:
        inputs += [("30min", ctx["sessions"]["s30m_mp3"]), ("60min", ctx["sessions"]["s60m_mp3"])]
    prompt = "List the languages spoken, quote the first sentence, and estimate the recording length in minutes."
    results = {}
    for label, path in inputs:
        secs = audio.duration_s(path)
        for model in ("muse-spark-1.2", "muse-wonder-1.0"):
            key = f"{model}@{label}"
            try:
                r = client.chat(
                    model=model, messages=audio_message(prompt, path), schema=AudioProbe,
                    reasoning_effort="low", max_completion_tokens=4000, purpose=f"smoke audio {key}",
                    audio_seconds=secs, use_cache=False,
                )
                text_tokens = 120  # prompt + schema, rough
                results[key] = {
                    "ok": True,
                    "prompt_tokens": r.prompt_tokens,
                    "audio_tokens_per_sec": round((r.prompt_tokens - text_tokens) / secs, 1),
                    "output_tokens": r.completion_tokens,
                    "cost_usd": round(r.cost_usd, 4),
                    "languages": r.parsed.languages,
                    "approx_minutes": r.parsed.approx_minutes,
                }
            except Exception as exc:  # record refusals and limits as results
                results[key] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return results


def check_determinism(ctx: dict) -> dict:
    client: LLMClient = ctx["client"]
    outs = []
    for _ in range(2):
        r = client.chat(
            model="muse-spark-1.3", messages=judge_messages(), schema=NumberCheck,
            reasoning_effort="minimal", temperature=0, seed=7, max_completion_tokens=1500,
            purpose="smoke determinism", use_cache=False,
        )
        outs.append(r)
    a, b = outs[0].parsed, outs[1].parsed
    return {
        "identical_json": a.model_dump() == b.model_dump(),
        "same_decision": (a.error_present, a.error_type) == (b.error_present, b.error_type),
        "decision": [a.error_present, a.error_type],
        "output_tokens": [r.completion_tokens for r in outs],
        "reasoning_tokens": [r.reasoning_tokens for r in outs],
        "cost_per_call_usd": round(outs[0].cost_usd, 5),
    }


def check_structured(ctx: dict) -> dict:
    client: LLMClient = ctx["client"]
    r = client.chat(
        model="muse-spark-1.3", messages=judge_messages(), schema=NumberCheck,
        reasoning_effort="low", max_completion_tokens=2000, purpose="smoke structured", use_cache=False,
    )
    return {
        "parsed": True,
        "decision": [r.parsed.error_present, r.parsed.error_type],
        "proposed_correction": r.parsed.proposed_correction,
        "prompt_tokens": r.prompt_tokens,
        "output_tokens": r.completion_tokens,
        "reasoning_tokens": r.reasoning_tokens,
        "cost_per_call_usd": round(r.cost_usd, 5),
    }


def check_cache(ctx: dict) -> dict:
    client: LLMClient = ctx["client"]
    users = [JUDGE_USER, JUDGE_USER.replace("dos veces", "tres veces")]
    rs = [
        client.chat(
            model="muse-spark-1.3", messages=judge_messages(u, prefix=TAXONOMY_PREFIX), schema=NumberCheck,
            reasoning_effort="minimal", max_completion_tokens=1500, prompt_cache_key="smoke-taxonomy-v1",
            purpose="smoke cache", use_cache=False,
        )
        for u in users
    ]
    return {
        "prompt_tokens": [r.prompt_tokens for r in rs],
        "cached_tokens": [r.cached_tokens for r in rs],
        "cache_hit_on_second": rs[1].cached_tokens > 0,
        "cost_usd": [round(r.cost_usd, 5) for r in rs],
    }


CHECKS = {
    "vt_file": check_vt_file,
    "vt_cap": check_vt_cap,
    "vt_realtime": check_vt_realtime,
    "audio_llm": check_audio_llm,
    "determinism": check_determinism,
    "structured": check_structured,
    "cache": check_cache,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(ALL_CHECKS))
    ap.add_argument("--long", action="store_true", help="also test 30- and 60-minute audio-LLM inputs (~$2)")
    ap.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    args = ap.parse_args()
    chosen = [c.strip() for c in args.only.split(",") if c.strip()] or ALL_CHECKS

    s = load_settings()
    if not s.model_api_key:
        print("MODEL_API_KEY is missing in second-ear/.env")
        return 2
    client = LLMClient(provider="meta", settings=s)
    ledger = client.ledger
    print(f"Run {ledger.run_id}. Spent so far (all runs): ${ledger.total_spent():.2f} of ${s.spend_cap_total:.2f}")
    estimate = 0.25 + (2.0 if args.long else 0.0)
    if not args.yes:
        reply = input(f"This run should cost about ${estimate:.2f}. Continue? [y/N] ").strip().lower()
        if reply != "y":
            return 1

    run_dir = s.runs_dir / "smoke" / ledger.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Preparing audio from FLEURS (first run downloads about 1 GB)...")
    sessions = build_sessions(s.data_dir, args.long)
    ctx = {"s": s, "client": client, "ledger": ledger, "guard": client.guard, "sessions": sessions,
           "run_dir": run_dir, "long": args.long}

    results: dict[str, dict] = {}
    for name in chosen:
        t0 = time.time()
        print(f"\n== {name} ==")
        try:
            results[name] = {"status": "ok", **CHECKS[name](ctx)}
        except Exception as exc:
            results[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:400]}
            traceback.print_exc(limit=2)
        results[name]["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(results[name], ensure_ascii=False, indent=2, default=str))

    summary = {"run_id": ledger.run_id, "results": results, "spend_this_run": ledger.summary(),
               "run_cost_usd": round(ledger.run_spent(), 4), "total_spent_usd": round(ledger.total_spent(), 4)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nRun cost ${summary['run_cost_usd']:.4f}; total spent ${summary['total_spent_usd']:.2f}")
    print(f"Saved {run_dir / 'summary.json'}")
    return 0 if all(r["status"] == "ok" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
