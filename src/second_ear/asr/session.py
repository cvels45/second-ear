"""Transcribe a whole recording with the Voice Transcribe file endpoint, plus a coverage check.

1. Cut the recording into chunks of 9.5 minutes or less at silences (the file endpoint's limit is 10).
2. Transcribe each chunk in ENDPOINTING mode, the only mode that kept every turn of mixed
   Spanish/English audio in our diagnostics, and shift turn times to the recording's clock.
3. Coverage check: any speech region of 1 s or more that no returned turn overlaps is sent again on
   its own. If it is still empty, it becomes a "gap" turn (text filled later by the second ASR) so
   that no stretch of speech silently disappears from the record.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import vad
from ..llm.ledger import Ledger, SpendGuard
from ..llm.pricing import asr_cost
from ..settings import Settings
from .voice_transcribe import transcribe_file


@dataclass
class Turn:
    start_s: float
    end_s: float
    text: str
    source: str  # "vt" (first pass), "recheck" (coverage re-send), "gap" (still untranscribed)
    chunk: int = 0


@dataclass
class SessionASR:
    audio: str
    duration_s: float
    chunks: list[tuple[float, float]]
    turns: list[Turn]
    speech_regions: int
    rechecked: int
    gaps: int
    cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["turns"] = [asdict(t) for t in self.turns]
        return d


def _read(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="int16")
    if x.ndim != 1:
        x = x.mean(axis=1).astype(np.int16)
    return x, sr


def _write(path: Path, x: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), x, sr, subtype="PCM_16")


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _vt_turns(resp: dict, offset_s: float, chunk: int, source: str) -> list[Turn]:
    out = []
    for t in resp.get("turns") or []:
        text = (t.get("transcript") or "").strip()
        if not text:
            continue
        out.append(Turn(round(offset_s + t["startMs"] / 1000, 3), round(offset_s + t["endMs"] / 1000, 3), text,
                        source, chunk))
    if not out and (resp.get("transcript") or "").strip():  # no turn list: keep the text as one turn
        out.append(Turn(offset_s, offset_s + resp.get("audioDurationMs", 0) / 1000,
                        resp["transcript"].strip(), source, chunk))
    return out


def transcribe_session(path: Path, *, settings: Settings, ledger: Ledger, guard: SpendGuard,
                       language_bias: list[str] | None = None, keywords: list[str] | None = None,
                       strategy: str = "utterance", min_region_s: float = 1.0, workers: int = 8,
                       transcribe=transcribe_file) -> SessionASR:
    """Transcribe a recording. `transcribe` is injectable so tests can run without the API.

    strategy="utterance" (default) sends each VAD utterance as its own request. On Nov 1 the chunked
    strategy merged a Hindi answer and its English rendition into one turn and returned only the Hindi
    (2 of 21 English turns lost in hi-S09), the same first-language lock seen in diarization mode. One
    request per utterance removes the cross-turn context that causes it, bills only speech, and is
    the same path live mode uses. strategy="chunk" keeps the 9.5-minute chunks for comparison.
    """
    x, sr = _read(path)
    total = len(x) / sr
    regions = vad.speech_regions(x, sr)
    est = asr_cost(total) + asr_cost(sum(r.dur for r in regions) * 0.2)
    guard.check(est, what=f"session ASR {path.name}")
    spent0 = ledger.run_spent()
    notes: list[str] = []

    def clip_and_send(td: Path, name: str, a: float, b: float, source: str, chunk: int) -> list[Turn]:
        a, b = max(0.0, a), min(total, b)
        cp = td / f"{name}.wav"
        _write(cp, x[int(a * sr): int(b * sr)], sr)
        resp = transcribe(cp, settings=settings, ledger=ledger, guard=guard, mode="ENDPOINTING",
                          language_bias=language_bias, keywords=keywords)
        if resp.get("_decode_repaired"):
            notes.append(f"invalid UTF-8 repaired in {name} ({a:.1f}-{b:.1f} s)")
        return _vt_turns(resp, a, chunk, source)

    with tempfile.TemporaryDirectory() as tdn:
        td = Path(tdn)
        if strategy == "utterance":
            utts = [r for r in vad.merge(regions, gap_s=0.35) if r.dur >= 0.3]
            bounds = [(r.start_s, r.end_s) for r in utts]
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(clip_and_send, td, f"utt{i}", r.start_s - 0.15, r.end_s + 0.15, "vt", i)
                        for i, r in enumerate(utts)]
                results = [f.result() for f in futs]
            turns = [t for res in results for t in res]
            speech = [r for r in utts if r.dur >= min_region_s]
            empty = [r for r, res in zip(utts, results) if not res and r.dur >= min_region_s]
        else:
            cuts = vad.cut_points(regions, total)
            bounds = list(zip([0.0, *cuts], [*cuts, total]))
            turns = []
            for ci, (a, b) in enumerate(bounds):
                turns += clip_and_send(td, f"chunk{ci}", a, b, "vt", ci)
            speech = [r for r in vad.merge(regions, gap_s=0.5) if r.dur >= min_region_s]
            empty = [r for r in speech
                     if sum(_overlap(r.start_s, r.end_s, t.start_s, t.end_s) for t in turns) < 0.5 * r.dur]

        # coverage check: re-send every untranscribed speech region once, with more padding
        rechecked = gaps = 0
        for r in empty:
            rechecked += 1
            new = clip_and_send(td, f"recheck{rechecked}", r.start_s - 0.4, r.end_s + 0.4, "recheck", -1)
            if new:
                turns += new
            else:
                gaps += 1
                turns.append(Turn(round(r.start_s, 3), round(r.end_s, 3), "", "gap", -1))
                notes.append(f"untranscribed speech {r.start_s:.1f}-{r.end_s:.1f} s")
    turns.sort(key=lambda t: t.start_s)
    return SessionASR(str(path), round(total, 3), [(round(a, 3), round(b, 3)) for a, b in bounds], turns,
                      len(speech), rechecked, gaps, round(ledger.run_spent() - spent0, 5), notes)
