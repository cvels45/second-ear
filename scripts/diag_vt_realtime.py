"""Check whether the realtime endpoint keeps both languages, as the file endpoint does in ENDPOINTING mode.

Run 1 (Nov 1, 20261101-110757): realtime ENDPOINTING returned 6 of the 7 turns (it missed the first
English clip, 8.7-14.1 s); DIARIZATION returned the 4 Spanish turns only. The file endpoint in
ENDPOINTING mode returned all 7 in two runs. This run tests whether pacing or a language bias fixes
realtime, which only live mode needs. Final text comes from speechComplete events.
Cost: 3 x 62 s of audio, under $0.01.

Usage (from second-ear/):  uv run python scripts/diag_vt_realtime.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diag_vt_languages import lang_mix  # noqa: E402

from second_ear.asr import VoiceTranscribeError, stream_file  # noqa: E402
from second_ear.llm import Ledger, SpendGuard  # noqa: E402
from second_ear.settings import load_settings  # noqa: E402

# Clip spans of data/smoke/session_60s.wav (from the file-endpoint ENDPOINTING runs), in ms
CLIPS = [(460, 5660, "es"), (8700, 14140, "en"), (17260, 24860, "es"), (26780, 33740, "en"),
         (35900, 42700, "es"), (45020, 49580, "en"), (52140, 60620, "es")]


def completed(events: list[dict]) -> list[dict]:
    """One row per speechComplete, with its start time and the last speaker label seen."""
    starts = {e["turnId"]: e.get("audioProcessedMs") for e in events if e.get("type") == "speechStart"}
    out, speaker = [], None
    for e in events:
        if e.get("type") == "speaker":
            speaker = e.get("label")
        elif e.get("type") == "speechComplete":
            out.append({"start_ms": starts.get(e.get("turnId")), "end_ms": e.get("audioProcessedMs"),
                        "speaker": speaker, "text": (e.get("transcript") or "").strip()})
    return out


def covered(turns: list[dict]) -> list[str]:
    """Which of the 7 known clips overlap some returned turn."""
    hits = []
    for a, b, lang in CLIPS:
        ok = any(t["start_ms"] is not None and t["start_ms"] < b and t["end_ms"] > a for t in turns)
        hits.append(f"{lang}{'+' if ok else '-'}")
    return hits


def main() -> int:
    s = load_settings()
    ledger = Ledger(s.ledger_path)
    guard = SpendGuard(ledger, s.spend_cap_total, s.spend_cap_per_run)
    mixed = s.data_dir / "smoke" / "session_60s.wav"
    if not mixed.exists():
        print("Run scripts/smoke_test.py first (it builds data/smoke/).")
        return 2
    out_dir = s.runs_dir / "diag_vt_realtime" / ledger.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("ENDPOINTING | as fast as possible | repeat", None, None),
        ("ENDPOINTING | paced at 1.2x real time", 1.2, None),
        ("ENDPOINTING | fast | bias Spanish,English", None, ["Spanish", "English"]),
    ]
    rows = []
    for i, (label, pace, bias) in enumerate(variants):
        try:
            res = stream_file(mixed, settings=s, ledger=ledger, guard=guard, mode="ENDPOINTING", pace=pace,
                              language_bias=bias, idle_timeout_s=20)
            turns = completed(res["events"])
            text = " ".join(t["text"] for t in turns)
            row = {"variant": label, "wall_s": round(res["wall_s"], 1), "turns": len(turns),
                   "clips (7)": " ".join(covered(turns)), **lang_mix(text),
                   "errors": [e.get("message") for e in res["events"] if e.get("type") == "error"],
                   "turn_heads": [t["text"][:50] for t in turns]}
            (out_dir / f"events_{i}.json").write_text(json.dumps(res["events"], ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
        except VoiceTranscribeError as exc:
            row = {"variant": label, "error": f"{exc.status}: {exc.detail[:160]}"}
        except Exception as exc:  # e.g. websockets.ConnectionClosedError
            row = {"variant": label, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRun cost ${ledger.run_spent():.4f}. Saved {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
