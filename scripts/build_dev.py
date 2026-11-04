"""Build the dev set from the committed scripts in data_gen/dev/.

Two stages, because speech synthesis runs in its own Python 3.12 environment (second-ear/tts/):

  text   assemble every dialogue (labels, turn ids) and write the TTS input
           data/dev/dialogues.jsonl         labeled turns, one dialogue per line
           data/dev/utterances_kokoro.jsonl en, es, pt, hi, zh lines for tts/synth.py
           data/dev/utterances_other.jsonl  ta and ar lines (other engines)
  audio  after TTS: one session per dialogue with silences between turns, plus timings
           data/dev/sessions/<dialogue>.wav (16 kHz), .mp3, .phone.wav, and gold JSON with times

Usage (from second-ear/):
  uv run python scripts/build_dev.py text
  cd tts; uv run python synth.py --in ../../data/dev/utterances_kokoro.jsonl --out ../../data/dev/tts; cd ..
  uv run python scripts/build_dev.py audio
No API calls, no cost.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_ear import audio  # noqa: E402
from second_ear.data.dialogue import build_all, stats, utterances, write_jsonl  # noqa: E402
from second_ear.settings import PKG_ROOT, load_settings  # noqa: E402

KOKORO_LANGS = {"en", "es", "pt", "hi", "zh"}
SEED = 2026


def is_kokoro(u: dict) -> bool:
    """Tamil and Arabic voices ("ta:f1", "ar:m1") go to other engines, including their English lines."""
    return u["lang"] in KOKORO_LANGS and not str(u.get("voice") or "").startswith(("ta:", "ar:"))


def gap_after(turn: dict, nxt: dict | None, rng: random.Random) -> float:
    """Silence after a turn: short before a rendition, longer after one, as in consecutive interpreting."""
    if nxt is None:
        return 1.0
    if nxt["kind"] == "rendition":
        return rng.uniform(0.45, 0.9)
    if turn["kind"] == "rendition":
        return rng.uniform(0.8, 1.5)
    return rng.uniform(0.5, 1.1)


def stage_text(dev_src: Path, out: Path) -> None:
    dialogues = build_all(dev_src)
    write_jsonl(dialogues, out / "dialogues.jsonl")
    utts = [u for d in dialogues for u in utterances(d)]
    write_jsonl([u for u in utts if is_kokoro(u)], out / "utterances_kokoro.jsonl")
    write_jsonl([u for u in utts if not is_kokoro(u)], out / "utterances_other.jsonl")
    st = stats(dialogues)
    print(json.dumps(st, indent=1))
    by_lang: dict[str, int] = {}
    for d in dialogues:
        by_lang[d["lang"]] = by_lang.get(d["lang"], 0) + 1
    print(f"dialogues per language: {by_lang}")
    print(f"utterances: {len(utts)} ({sum(is_kokoro(u) for u in utts)} for Kokoro)")
    print(f"Wrote {out / 'dialogues.jsonl'}")


def stage_audio(out: Path, tts_dir: Path) -> None:
    dialogues = [json.loads(line) for line in (out / "dialogues.jsonl").read_text(encoding="utf-8").splitlines()]
    sess_dir, wav16 = out / "sessions", out / "tts16"
    done = skipped = 0
    for d in dialogues:
        clips = [tts_dir / f"{t['turn_id']}.wav" for t in d["turns"]]
        if not all(c.exists() for c in clips):
            skipped += 1
            continue
        rng = random.Random(f"{SEED}-{d['dialogue_id']}")
        paths, gaps = [], []
        for i, (t, c) in enumerate(zip(d["turns"], clips)):
            p16 = wav16 / c.name
            if not p16.exists():
                audio.to_wav(c, p16)
            paths.append(p16)
            gaps.append(gap_after(t, d["turns"][i + 1] if i + 1 < len(d["turns"]) else None, rng))
        wav = sess_dir / f"{d['dialogue_id']}.wav"
        segs = audio.concat(paths, wav, gap_s=gaps)
        audio.to_mp3(wav, wav.with_suffix(".mp3"))
        audio.phone_copy(wav, sess_dir / f"{d['dialogue_id']}.phone.wav")
        for t, seg in zip(d["turns"], segs):
            t["start_s"], t["end_s"] = round(seg.start_s, 3), round(seg.end_s, 3)
        d["duration_s"] = round(audio.duration_s(wav), 2)
        (sess_dir / f"{d['dialogue_id']}.gold.json").write_text(json.dumps(d, ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")
        done += 1
    total = sum(json.loads(p.read_text(encoding="utf-8"))["duration_s"] for p in sess_dir.glob("*.gold.json"))
    print(f"{done} sessions built, {skipped} skipped (TTS missing), {total / 60:.1f} min of audio in {sess_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["text", "audio"])
    args = ap.parse_args()
    s = load_settings()
    out = s.data_dir / "dev"
    if args.stage == "text":
        stage_text(PKG_ROOT / "data_gen" / "dev", out)
    else:
        stage_audio(out, out / "tts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
