"""Synthesize the Tamil dialogues' lines with Indic Parler-TTS (Apache 2.0) on the GPU.

Reads the same utterance JSONL as tts/synth.py and handles rows whose voice starts with "ta:"
(Tamil lines plus the Tamil interpreters' English lines, so each interpreter keeps one voice).
Rows with "ar:" voices are skipped and counted (Arabic has no engine yet).

Named speakers keep a voice consistent across turns. Indic Parler's Tamil speakers are female
(Jaya, Kavitha), so the male applicants borrow South Indian male speakers (Prakash, Harish); listen to
the self-test and remap with --voices if one sounds wrong.

Output: <out>/<id>.wav (PCM16 at the model's rate) and <out>/manifest_parler.jsonl.

Usage (from second-ear/tts_parler/):
  uv run python synth_parler.py --selftest
  uv run python synth_parler.py --in ../../data/dev/utterances_other.jsonl --out ../../data/dev/tts
Needs the gated model ai4bharat/indic-parler-tts accepted on Hugging Face and `hf auth login`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_ID = "ai4bharat/indic-parler-tts"
VOICES = {"ta:f1": "Jaya", "ta:f2": "Kavitha", "ta:m1": "Prakash", "ta:m2": "Harish"}
PITCH = {"ta:f1": "moderate", "ta:f2": "slightly high", "ta:m1": "low", "ta:m2": "moderate"}
SELFTEST = [
    ("ta:f1", "ta", "மொத்தம் எத்தனை முறை நீங்கள் கைது செய்யப்பட்டீர்கள்?"),
    ("ta:f1", "en", "Three times. The second time was in June 2021."),
    ("ta:f2", "ta", "அவர்கள் என்னையும் என் தம்பி குமரனையும் கைது செய்தார்கள்."),
    ("ta:m1", "ta", "மூன்று முறை. இரண்டாவது முறை இரண்டாயிரத்து இருபத்தொன்று ஜூனில்."),
    ("ta:m2", "ta", "நான் திருகோணமலையில் ஒரு பேருந்து நிறுவனத்தில் ஆறு ஆண்டுகள் ஓட்டுநராக இருந்தேன்."),
]


def description(voice: str) -> str:
    name = VOICES[voice]
    return (f"{name} speaks at a moderate pace with a {PITCH.get(voice, 'moderate')} pitch. "
            "The recording is of very high quality, with the speaker's voice sounding clear and very close up, "
            "very clear audio.")


class Parler:
    def __init__(self) -> None:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("WARNING: no CUDA device; Parler on CPU is very slow.", flush=True)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(self.device)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sr = self.model.config.sampling_rate

    def say(self, text: str, voice: str, seed: int) -> np.ndarray:
        torch = self.torch
        torch.manual_seed(seed)
        d = self.desc_tok(description(voice), return_tensors="pt").to(self.device)
        p = self.tok(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            gen = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                      prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        return gen.cpu().numpy().squeeze().astype(np.float32)


def seed_for(uid: str) -> int:
    return int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)


def write(path: Path, audio: np.ndarray, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), (np.clip(audio, -1, 1) * 32767).astype(np.int16), sr, subtype="PCM_16")
    return len(audio) / sr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--voices", help='JSON remap, e.g. {"ta:m1": "Suresh"}')
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.voices:
        VOICES.update(json.loads(args.voices))

    root = Path(__file__).resolve().parents[2] / "data"
    if args.selftest:
        utts = [{"id": f"parler_selftest_{i}_{v.replace(':', '')}_{lang}", "lang": lang, "text": t, "voice": v}
                for i, (v, lang, t) in enumerate(SELFTEST)]
        out = args.out or root / "tts_selftest"
    else:
        if not args.inp or not args.out:
            ap.error("--in and --out are required (or use --selftest)")
        rows = [json.loads(x) for x in args.inp.read_text(encoding="utf-8").splitlines() if x.strip()]
        utts = [u for u in rows if str(u.get("voice") or "").startswith("ta:")]
        skipped = len(rows) - len(utts)
        print(f"{len(utts)} Tamil-voice lines; {skipped} other lines skipped (Arabic has no engine yet)")
        out = args.out
    unknown = sorted({u["voice"] for u in utts if u["voice"] not in VOICES})
    if unknown:
        raise SystemExit(f"No speaker mapped for {unknown}; pass --voices")

    t0 = time.time()
    engine = Parler()
    print(f"Model loaded on {engine.device} in {time.time() - t0:.0f} s", flush=True)
    manifest, t0 = [], time.time()
    for i, u in enumerate(utts, 1):
        dst = out / f"{u['id']}.wav"
        if dst.exists() and not args.overwrite and not args.selftest:
            dur = sf.info(str(dst)).duration
        else:
            dur = write(dst, engine.say(u["text"], u["voice"], seed_for(u["id"])), engine.sr)
        manifest.append({"id": u["id"], "lang": u["lang"], "voice": f"{u['voice']}={VOICES[u['voice']]}",
                         "file": dst.name, "duration_s": round(dur, 3)})
        if i % 10 == 0 or i == len(utts):
            print(f"  {i}/{len(utts)} lines, {time.time() - t0:.0f} s elapsed", flush=True)
    with (out / "manifest_parler.jsonl").open("w", encoding="utf-8") as fh:
        for r in manifest:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    audio_s = sum(r["duration_s"] for r in manifest)
    print(f"{len(manifest)} files, {audio_s / 60:.1f} min of audio in {time.time() - t0:.0f} s, in {out}")
    if args.selftest:
        print("Listen to each file: Tamil should sound natural, and the male voices male. Tell Claude.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
