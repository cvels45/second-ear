"""Energy-based voice activity detection (numpy only).

Classical signal processing is the right tool here: we only need speech regions to cut long audio at
silences and to check that every stretch of speech got a transcript. It needs no model download and
behaves the same on every machine. (Silero VAD is the upgrade path for noisy real-world recordings.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Region:
    start_s: float
    end_s: float

    @property
    def dur(self) -> float:
        return self.end_s - self.start_s


def frame_db(x: np.ndarray, sr: int, win_s: float = 0.03, hop_s: float = 0.01) -> np.ndarray:
    """RMS level per frame in dBFS (int16 or float input)."""
    x = x.astype(np.float32)
    if np.abs(x).max(initial=0) > 1.5:  # int16 scale
        x = x / 32768.0
    win, hop = int(win_s * sr), int(hop_s * sr)
    if len(x) < win:
        return np.array([-120.0])
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def speech_regions(x: np.ndarray, sr: int, *, min_speech_s: float = 0.2, fill_gap_s: float = 0.25,
                   margin_db: float = 12.0, floor_db: float = -50.0, hop_s: float = 0.01) -> list[Region]:
    db = frame_db(x, sr, hop_s=hop_s)
    noise = float(np.percentile(db, 10))
    thr = max(noise + margin_db, floor_db)
    on = db > thr
    regions: list[Region] = []
    i, n = 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            regions.append(Region(i * hop_s, j * hop_s + 0.03))
            i = j
        else:
            i += 1
    merged: list[Region] = []
    for r in regions:  # fill short gaps (between words)
        if merged and r.start_s - merged[-1].end_s <= fill_gap_s:
            merged[-1].end_s = r.end_s
        else:
            merged.append(Region(r.start_s, r.end_s))
    return [r for r in merged if r.dur >= min_speech_s]


def merge(regions: list[Region], gap_s: float) -> list[Region]:
    out: list[Region] = []
    for r in sorted(regions, key=lambda r: r.start_s):
        if out and r.start_s - out[-1].end_s <= gap_s:
            out[-1].end_s = max(out[-1].end_s, r.end_s)
        else:
            out.append(Region(r.start_s, r.end_s))
    return out


def cut_points(regions: list[Region], total_s: float, max_chunk_s: float = 570.0, search_s: float = 90.0) -> list[float]:
    """Chunk boundaries at the middle of the longest silence before each max_chunk_s limit."""
    cuts, start = [], 0.0
    while total_s - start > max_chunk_s:
        lo, hi = start + max_chunk_s - search_s, start + max_chunk_s
        gaps = [(b.start_s - a.end_s, (a.end_s + b.start_s) / 2) for a, b in zip(regions, regions[1:])
                if lo <= (a.end_s + b.start_s) / 2 <= hi]
        cut = max(gaps)[1] if gaps else hi  # no silence found: hard cut at the limit
        cuts.append(round(cut, 3))
        start = cut
    return cuts
