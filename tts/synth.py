"""Synthesize utterances with Kokoro (CPU, ONNX Runtime). Covers en, es, pt, hi, and zh.

Input JSONL, one utterance per line:
  {"id": "d01-t03-int", "lang": "es", "text": "...", "voice": "ef_dora", "speed": 1.0}
`voice` is optional (defaults below). Any voice can speak any language; we use that on purpose so one
interpreter voice speaks both languages, with an accent, as a real interpreter does. Mandarin lines use
the v1.1-zh model, whose voices differ, so a Mandarin session's interpreter changes voice across
languages (it doesn't affect the pipeline, which pairs turns by language and content, not by speaker).

Output: <out>/<id>.wav, 24 kHz mono PCM16, plus <out>/manifest.jsonl with durations.

Usage (from second-ear/tts/):
  uv run python synth.py --selftest
  uv run python synth.py --in ../../data/dev/utterances.jsonl --out ../../data/dev/tts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "kokoro"
ESPEAK_LANG = {"en": "en-us", "es": "es", "pt": "pt-br", "hi": "hi"}
DEFAULT_VOICE = {"en": "am_michael", "es": "ef_dora", "pt": "pf_dora", "hi": "hf_alpha", "zh": None}
SELFTEST = {
    "en": "On the third of March, they arrested me twice.",
    "es": "Me arrestaron tres veces en marzo de dos mil veintiuno.",
    "pt": "Fui preso duas vezes em março de dois mil e vinte e um.",
    "hi": "मार्च दो हज़ार इक्कीस में मुझे दो बार गिरफ़्तार किया गया।",
    "zh": "二零二一年三月，我被逮捕了两次。",
}


class Engine:
    """Lazy-loads the two Kokoro models; v1.0 for espeak languages, v1.1-zh for Mandarin."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.dir = models_dir
        self._v10 = None
        self._zh = None
        self._g2p_zh = None

    def _need(self, *names: str) -> None:
        missing = [n for n in names if not (self.dir / n).exists()]
        if missing:
            raise SystemExit(f"Missing {missing} in {self.dir}. Run: uv run python download_models.py")

    @property
    def v10(self):
        if self._v10 is None:
            from kokoro_onnx import Kokoro

            self._need("kokoro-v1.0.onnx", "voices-v1.0.bin")
            self._v10 = Kokoro(str(self.dir / "kokoro-v1.0.onnx"), str(self.dir / "voices-v1.0.bin"))
        return self._v10

    @property
    def zh(self):
        if self._zh is None:
            from kokoro_onnx import Kokoro
            from misaki import zh

            self._need("kokoro-v1.1-zh.onnx", "voices-v1.1-zh.bin", "config-v1.1-zh.json")
            self._zh = Kokoro(str(self.dir / "kokoro-v1.1-zh.onnx"), str(self.dir / "voices-v1.1-zh.bin"),
                              vocab_config=str(self.dir / "config-v1.1-zh.json"))
            self._g2p_zh = zh.ZHG2P(version="1.1")
        return self._zh

    def default_zh_voice(self, female: bool = True) -> str:
        prefix = "zf_" if female else "zm_"
        voices = [v for v in self.zh.get_voices() if v.startswith(prefix)]
        if not voices:
            raise SystemExit(f"No {prefix}* voices in voices-v1.1-zh.bin")
        return voices[0]

    def voices(self, lang: str) -> list[str]:
        return self.zh.get_voices() if lang == "zh" else self.v10.get_voices()

    def resolve(self, lang: str, voice: str | None) -> str:
        """Concrete voice id. "zh:f1" / "zh:m2" pick the 1st female / 2nd male voice of the v1.1-zh pack."""
        if lang == "zh":
            pack = self.zh.get_voices()
            if voice is None:
                return self.default_zh_voice()
            if voice.startswith("zh:"):
                kind, n = voice[3], int(voice[4:] or 1)
                cands = [v for v in pack if v.startswith("zf_" if kind == "f" else "zm_")]
                if len(cands) < n:
                    raise ValueError(f"only {len(cands)} {kind} voices in the v1.1-zh pack")
                return cands[n - 1]
            if voice not in pack:
                raise ValueError(f"voice {voice!r} is not in the v1.1-zh voice pack")
            return voice
        if lang not in ESPEAK_LANG:
            raise ValueError(f"Kokoro path has no language {lang!r} (ta and ar use other engines)")
        voice = voice or DEFAULT_VOICE[lang]
        if voice not in self.v10.get_voices():
            raise ValueError(f"voice {voice!r} is not in the v1.0 voice pack")
        return voice

    def say(self, text: str, lang: str, voice: str | None = None, speed: float = 1.0) -> tuple[np.ndarray, int]:
        voice = self.resolve(lang, voice)
        if lang == "zh":
            phonemes, _ = self._g2p_zh(text)
            return self.zh.create(phonemes, voice=voice, speed=speed, is_phonemes=True)
        return self.v10.create(text, voice=voice, speed=speed, lang=ESPEAK_LANG[lang])


def write_pcm16(path: Path, samples: np.ndarray, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), (pcm * 32767).astype(np.int16), sr, subtype="PCM_16")
    return len(pcm) / sr


def run(utterances: list[dict], out_dir: Path, engine: Engine, overwrite: bool = False) -> list[dict]:
    rows, t0 = [], time.time()
    bad = set()
    for u in utterances:  # fail fast on a bad voice or language, before hours of synthesis
        try:
            engine.resolve(u["lang"], u.get("voice"))
        except ValueError as exc:
            bad.add(f"{u['lang']}/{u.get('voice')}: {exc}")
    if bad:
        raise SystemExit("Fix these voices first:\n  " + "\n  ".join(sorted(bad)))
    for i, u in enumerate(utterances, 1):
        dst = out_dir / f"{u['id']}.wav"
        if dst.exists() and not overwrite:
            dur = sf.info(str(dst)).duration
        else:
            samples, sr = engine.say(u["text"], u["lang"], u.get("voice"), float(u.get("speed", 1.0)))
            dur = write_pcm16(dst, samples, sr)
        rows.append({"id": u["id"], "lang": u["lang"], "voice": engine.resolve(u["lang"], u.get("voice")), "file": dst.name,
                     "duration_s": round(dur, 3)})
        if i % 25 == 0 or i == len(utterances):
            print(f"  {i}/{len(utterances)} utterances, {time.time() - t0:.0f} s elapsed", flush=True)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", type=Path, help="utterances JSONL")
    ap.add_argument("--out", type=Path, help="output folder")
    ap.add_argument("--selftest", action="store_true", help="one sentence per language, plus an accented-English check")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    engine = Engine()
    if args.selftest:
        out = args.out or MODELS_DIR.parents[1] / "tts_selftest"
        utts = [{"id": f"selftest_{lang}", "lang": lang, "text": text} for lang, text in SELFTEST.items()]
        utts.append({"id": "selftest_en_spanish_voice", "lang": "en", "text": SELFTEST["en"], "voice": "ef_dora"})
        t0 = time.time()
        rows = run(utts, out, engine, overwrite=True)
        audio_s = sum(r["duration_s"] for r in rows)
        print(f"\n{len(rows)} files, {audio_s:.1f} s of audio in {time.time() - t0:.1f} s, in {out}")
        print(f"zh voices available: {len(engine.voices('zh'))}; v1.0 voices: {len(engine.voices('en'))}")
        print("Listen to two or three files, then tell Claude.")
        return 0
    if not args.inp or not args.out:
        ap.error("--in and --out are required (or use --selftest)")
    utts = [json.loads(line) for line in args.inp.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = run(utts, args.out, engine, overwrite=args.overwrite)
    print(f"{len(rows)} utterances, {sum(r['duration_s'] for r in rows) / 60:.1f} min of audio in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
