"""Diagnose why Voice Transcribe dropped the English turns in the smoke test's mixed session.

The smoke test sent 62 s of alternating Spanish/English FLEURS readings and got back only the
4 Spanish turns. This tries mode and languageBias variants, plus English-only audio, to find a
setting that keeps both languages. Cost: about 8 minutes of audio, roughly $0.03.

Usage (from second-ear/):  uv run python scripts/diag_vt_languages.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_ear import audio  # noqa: E402
from second_ear.asr import VoiceTranscribeError, transcribe_file  # noqa: E402
from second_ear.llm import Ledger, SpendGuard  # noqa: E402
from second_ear.settings import load_settings  # noqa: E402

EN = {"the", "and", "of", "to", "is", "in", "that", "was", "for", "with", "are", "it", "on", "as", "by", "this"}
ES = {"el", "la", "los", "las", "de", "que", "y", "en", "un", "una", "es", "por", "con", "para", "del", "se"}


def lang_mix(text: str) -> dict:
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    en = sum(w in EN for w in words)
    es = sum(w in ES for w in words)
    return {"words": len(words), "en_hits": en, "es_hits": es}


def main() -> int:
    s = load_settings()
    ledger = Ledger(s.ledger_path)
    guard = SpendGuard(ledger, s.spend_cap_total, s.spend_cap_per_run)
    smoke = s.data_dir / "smoke"
    mixed = smoke / "session_60s.wav"
    en_clips = sorted((smoke / "clips").glob("en_*.wav"))[:4]
    if not mixed.exists() or not en_clips:
        print("Run scripts/smoke_test.py first (it builds data/smoke/).")
        return 2
    en_only = smoke / "diag_english_only.wav"
    audio.concat(en_clips, en_only)

    variants = [
        ("mixed | DIARIZATION | no bias", mixed, "DIARIZATION", None),
        ("mixed | DIARIZATION | bias Spanish,English", mixed, "DIARIZATION", ["Spanish", "English"]),
        ("mixed | ENDPOINTING | no bias", mixed, "ENDPOINTING", None),
        ("mixed | ENDPOINTING | bias English,Spanish", mixed, "ENDPOINTING", ["English", "Spanish"]),
        ("mixed | PUSH_TO_TALK | no bias", mixed, "PUSH_TO_TALK", None),
        ("english-only | DIARIZATION | no bias", en_only, "DIARIZATION", None),
        ("english-only | DIARIZATION | bias English", en_only, "DIARIZATION", ["English"]),
        ("first English clip alone | ENDPOINTING | no bias", en_clips[0], "ENDPOINTING", None),
    ]
    out_dir = s.runs_dir / "diag_vt" / ledger.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, path, mode, bias in variants:
        wav = path if path.suffix == ".wav" else audio.to_wav(path, out_dir / (path.stem + ".wav"))
        try:
            res = transcribe_file(wav, settings=s, ledger=ledger, guard=guard, mode=mode, language_bias=bias)
            turns = res.get("turns", [])
            text = res.get("transcript", "") or " ".join(t.get("transcript", "") for t in turns)
            row = {
                "variant": label,
                "audio_s": round(audio.duration_s(wav), 1),
                "turns": len(turns),
                "spans_ms": [(t.get("startMs"), t.get("endMs"), t.get("speaker")) for t in turns],
                **lang_mix(text),
                "text_head": text[:160],
            }
        except VoiceTranscribeError as exc:
            row = {"variant": label, "error": f"{exc.status}: {exc.detail[:160]}"}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRun cost ${ledger.run_spent():.4f}. Saved {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
