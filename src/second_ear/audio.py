"""Audio helpers built on ffmpeg (conversion) and soundfile (read/write)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _ffmpeg(*args: str) -> None:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH (see SETUP.md)")
    subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def to_wav(src: Path, dst: Path, sample_rate: int = 16000) -> Path:
    """Mono 16-bit PCM WAV at the given rate (Voice Transcribe accepts 16 kHz or 24 kHz)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-i", str(src), "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst))
    return dst


def to_mp3(src: Path, dst: Path, bitrate: str = "64k") -> Path:
    """Compact copy for audio-LLM input (60 min at 64 kbps is about 29 MB, under the 50 MB inline cap)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-i", str(src), "-ac", "1", "-c:a", "libmp3lame", "-b:a", bitrate, str(dst))
    return dst


def phone_copy(src: Path, dst: Path) -> Path:
    """Simulate a phone line: G.711 mu-law at 8 kHz, then back to 16 kHz PCM."""
    tmp = dst.with_suffix(".mulaw.wav")
    _ffmpeg("-i", str(src), "-ac", "1", "-ar", "8000", "-c:a", "pcm_mulaw", str(tmp))
    _ffmpeg("-i", str(tmp), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst))
    tmp.unlink(missing_ok=True)
    return dst


def duration_s(path: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(path)).duration)


@dataclass
class Segment:
    source: str
    start_s: float
    end_s: float


def concat(paths: list[Path], dst: Path, sample_rate: int = 16000, gap_s: float | list[float] = 0.6) -> list[Segment]:
    """Concatenate mono WAVs (already at sample_rate) with silence after each; return segment times.

    gap_s is one gap for all clips, or one gap per clip (the silence that follows it).
    """
    import numpy as np
    import soundfile as sf

    gaps = gap_s if isinstance(gap_s, list) else [gap_s] * len(paths)
    if len(gaps) != len(paths):
        raise ValueError("need one gap per clip")
    pieces, segments, t = [], [], 0.0
    for p, g in zip(paths, gaps):
        data, sr = sf.read(str(p), dtype="int16")
        if sr != sample_rate or data.ndim != 1:
            raise ValueError(f"{p} must be mono {sample_rate} Hz (convert with to_wav first)")
        gap = np.zeros(int(g * sample_rate), dtype=np.int16)
        pieces += [data, gap]
        segments.append(Segment(str(p), t, t + len(data) / sample_rate))
        t += (len(data) + len(gap)) / sample_rate
    dst.parent.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(0.6 * sample_rate), dtype=np.int16)
    sf.write(str(dst), np.concatenate(pieces) if pieces else silence, sample_rate, subtype="PCM_16")
    return segments
