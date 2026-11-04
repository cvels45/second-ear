"""Audio adjudicator: listen to both clips and decide whether a finding is the interpreter's or the transcript's.

A text judge only sees the ASR transcript, so a misheard word can look like an interpreting error.
Muse Spark 1.2 (Meta's recommended model for audio input) hears the speaker's clip and the
interpreter's clip, says what it hears at the quoted words, and labels the finding:

  interpreter_error         the audio confirms the claim
  uncertain_transcription   the audio contradicts the transcript at the quoted words, or can't settle it

Clips are cut once and kept, so the report can play the same audio the adjudicator heard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .judges import LANG
from .llm import LLMClient, LLMOutputError, audio_message

ADJ_VERSION = "j2"  # j2: the second transcription (Omnilingual) is shown when available
PAD_S = 0.25
GAP_S = 1.0


class Hearing(BaseModel):
    heard_source: str = Field(description="what the speaker actually says at the quoted source words")
    heard_rendition: str = Field(description="what the interpreter actually says at the quoted rendition words")
    transcript_matches_audio: bool
    verdict: Literal["interpreter_error", "transcription_error", "uncertain"]
    reason: str = Field(description="one short sentence")


@dataclass
class Clips:
    orig: Path
    rend: Path
    both: Path
    orig_len_s: float


def cut_clips(wav: Path, out_dir: Path, key: str, orig: tuple[float, float], rend: tuple[float, float]) -> Clips:
    import numpy as np
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = Clips(out_dir / f"{key}.orig.wav", out_dir / f"{key}.rend.wav", out_dir / f"{key}.both.wav", 0.0)
    info = sf.info(str(wav))
    sr = info.samplerate

    def seg(a: float, b: float):
        start = max(0, int((a - PAD_S) * sr))
        stop = min(info.frames, int((b + PAD_S) * sr))
        data, _ = sf.read(str(wav), start=start, stop=stop, dtype="int16")
        return data if data.ndim == 1 else data[:, 0]

    o, r = seg(*orig), seg(*rend)
    paths.orig_len_s = len(o) / sr
    if not paths.both.exists():
        sf.write(str(paths.orig), o, sr, subtype="PCM_16")
        sf.write(str(paths.rend), r, sr, subtype="PCM_16")
        sf.write(str(paths.both), np.concatenate([o, np.zeros(int(GAP_S * sr), dtype=np.int16), r]), sr,
                 subtype="PCM_16")
    return paths


SYSTEM = """You are a court interpreter checking a transcript against the audio.
You hear two clips joined by one second of silence: first the SPEAKER, then the INTERPRETER's rendition.
A reviewer flagged an interpreting error based on the transcript. Listen carefully to the quoted words in each clip.

verdict:
- interpreter_error: the audio matches the transcript at the quoted words, and the claim holds.
- transcription_error: the audio differs from the transcript at the quoted words, and with the words actually spoken the claim does not hold.
- uncertain: you cannot hear the quoted words clearly enough to decide.
Judge only what is audible; do not re-judge the interpretation style."""


def second_opinion(omni, finding: dict, clips: Clips) -> dict:
    """Omnilingual's transcription of both clips, and how much of each quoted span it supports."""
    from .asr.omnilingual import supports

    o, r = omni.transcribe_file(clips.orig), omni.transcribe_file(clips.rend)
    return {"orig": o, "rend": r, "source_support": round(supports(finding["source_words"], o), 2),
            "rendition_support": round(supports(finding["rendition_words"], r), 2)}


def adjudicate(client: LLMClient, finding: dict, clips: Clips, src_lang: str, *, model: str = "muse-spark-1.2",
               reasoning: str = "low", second: dict | None = None) -> Hearing:
    orig_lang = LANG.get(src_lang, src_lang) if finding["direction"] == "src->en" else "English"
    rend_lang = "English" if finding["direction"] == "src->en" else LANG.get(src_lang, src_lang)
    prompt = (
        f"First clip (0:00 to {clips.orig_len_s:.1f} s): the speaker, in {orig_lang}.\n"
        f"Second clip (after the silence): the interpreter, in {rend_lang}.\n\n"
        f"Transcript of the speaker: {finding['orig_text']}\n"
        f"Transcript of the interpreter: {finding['rend_text']}\n\n"
        f"Flagged error ({finding['type']}): the speaker said \"{finding['source_words']}\"; the interpreter said "
        f"\"{finding['rendition_words']}\". {finding['explanation']}"
    )
    if second:
        prompt += ("\n\nA second, independent speech recognizer (no punctuation) heard:\n"
                   f"speaker: {second['orig']}\ninterpreter: {second['rend']}")
    msgs = audio_message(prompt, clips.both, system=SYSTEM)
    for budget in (4000, 12000):  # Nov 2: a third of calls at 2,000 tokens were cut off; retry with more room
        try:
            res = client.chat(model=model, messages=msgs, schema=Hearing, reasoning_effort=reasoning,
                              max_completion_tokens=budget, audio_seconds=clips.orig_len_s + GAP_S + 10,
                              prompt_cache_key=f"adjudicator-{ADJ_VERSION}", purpose="adjudicator")
            return res.parsed
        except LLMOutputError:
            continue
    raise LLMOutputError("adjudicator answer cut off twice")


def check_label(h: Hearing) -> str:
    return "interpreter_error" if h.verdict == "interpreter_error" else "uncertain_transcription"
