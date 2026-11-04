"""Offline tests: the streaming segmenter and live-mode event flow (no API)."""

from __future__ import annotations

import numpy as np

from second_ear.live import EventBus, Segmenter


def test_segmenter_closes_utterances_after_silence():
    sr = 16000
    x = np.zeros(sr * 10, dtype=np.int16)
    tone = (np.sin(np.arange(int(1.5 * sr)) * 0.1) * 8000).astype(np.int16)
    for start in (1.0, 4.0, 7.0):
        x[int(start * sr): int(start * sr) + len(tone)] = tone
    seg, out = Segmenter(sr), []
    for i in range(0, len(x), 1600):
        out += seg.push(x[i: i + 1600])
    assert len(out) == 3
    for (a, b, samples), start in zip(out, (1.0, 4.0, 7.0)):
        assert abs(a - (start - 0.15)) < 0.1 and abs(b - (start + 1.5 + 0.15)) < 0.1
        assert abs(len(samples) / sr - (b - a)) < 0.02


def test_event_bus_orders_events():
    bus = EventBus()
    bus.emit("turn", text="a")
    bus.emit("finding", type="omission")
    evs = bus.wait_from(0, timeout=0.1)
    assert [e["kind"] for e in evs] == ["turn", "finding"] and [e["id"] for e in evs] == [0, 1]
    assert bus.wait_from(2, timeout=0.05) == []
