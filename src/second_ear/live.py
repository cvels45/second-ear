"""Near-live mode: flag interpreting errors a few seconds after each rendition, during a prep session.

Audio arrives in 100 ms blocks (microphone, or a recording played at real-time speed). An energy
segmenter closes an utterance after 0.5 s of silence; each utterance goes to Voice Transcribe as its
own file request (the same path as review mode), so a turn's text lands 1-2 s after it ends. Turns
are re-paired as they arrive; a pair is judged once the next turn starts or 2 s of silence follow
it. Judges, holistic pass, verifier, and calibrator are the same as in review mode.

Only for settings where recording is lawful and agreed: client prep sessions with every participant's
consent, and depositions noticed for recording. Not for immigration court or USCIS interviews.
"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .asr.session import _vt_turns, _write
from .asr.voice_transcribe import transcribe_file
from .judges import baseline_pair, judge_pair
from .llm import LLMClient
from .pairing import PTurn, Sim, is_meta, lang_of, merge_splits, pair_turns, tuned_params, uninterpreted_runs
from .pipeline import Calibrator, Finding, PairRecord, build_pairs, select_claims, _family_of
from .vad import frame_db
from .verifier import verify_pair

BLOCK_S = 0.1


class Segmenter:
    """Streaming energy segmenter: emits (start_s, end_s, samples) per utterance."""

    def __init__(self, sr: int, end_silence_s: float = 0.5, min_speech_s: float = 0.3, margin_db: float = 12.0,
                 floor_db: float = -50.0, pad_s: float = 0.15):
        self.sr, self.end_sil, self.min_speech, self.margin, self.floor, self.pad = (
            sr, end_silence_s, min_speech_s, margin_db, floor_db, pad_s)
        self.buf = np.zeros(0, dtype=np.int16)
        self.t0 = 0.0  # time of buf[0]
        self.levels: list[float] = []
        self.start: float | None = None
        self.last_voice = 0.0

    def threshold(self) -> float:
        noise = float(np.percentile(self.levels[-3000:], 10)) if self.levels else -60.0
        return max(noise + self.margin, self.floor)

    def push(self, block: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
        t_block = self.t0 + len(self.buf) / self.sr
        self.buf = np.concatenate([self.buf, block])
        db = frame_db(block, self.sr)
        self.levels.extend(db.tolist())
        thr = self.threshold()
        out = []
        for k, level in enumerate(db):
            t = t_block + k * 0.01
            if level > thr:
                if self.start is None:
                    self.start = t
                self.last_voice = t + 0.03
            elif self.start is not None and t - self.last_voice >= self.end_sil:
                if self.last_voice - self.start >= self.min_speech:
                    a, b = max(self.t0, self.start - self.pad), self.last_voice + self.pad
                    seg = self.buf[int((a - self.t0) * self.sr): int((b - self.t0) * self.sr)]
                    out.append((a, b, seg.copy()))
                self.start = None
        keep_from = (self.start - self.pad - 1.0) if self.start is not None else (t_block + len(block) / self.sr - 1.0)
        cut = int(max(0.0, keep_from - self.t0) * self.sr)
        if cut > 0:
            self.buf, self.t0 = self.buf[cut:], self.t0 + cut / self.sr
        return out


class EventBus:
    def __init__(self):
        self.events: list[dict] = []
        self.cv = threading.Condition()

    def emit(self, kind: str, **data) -> None:
        with self.cv:
            self.events.append({"id": len(self.events), "kind": kind, "t": round(time.time(), 2), **data})
            self.cv.notify_all()

    def wait_from(self, k: int, timeout: float = 15.0) -> list[dict]:
        with self.cv:
            if len(self.events) <= k:
                self.cv.wait(timeout)
            return self.events[k:]


class LiveSession:
    def __init__(self, session_id: str, src_lang: str, *, client: LLMClient, sim: Sim, calibrator: Calibrator,
                 bus: EventBus, language_bias: list[str], model: str = "muse-spark-1.3", workers: int = 6):
        self.sid, self.src, self.client, self.sim, self.cal, self.bus = session_id, src_lang, client, sim, calibrator, bus
        self.bias, self.model = language_bias, model
        self.turns: list[PTurn] = []
        self.lock = threading.Lock()
        self.judged: set[tuple] = set()
        self.reported_runs: set[int] = set()
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.tmp = Path(tempfile.mkdtemp(prefix="second-ear-live-"))
        self.n_utt = 0
        self.findings: list[Finding] = []
        self.clock = 0.0

    # --- audio in -> turns ---
    def on_utterance(self, a: float, b: float, samples: np.ndarray, sr: int) -> None:
        self.n_utt += 1
        k = self.n_utt
        self.pool.submit(self._transcribe, k, a, b, samples, sr)

    def _transcribe(self, k: int, a: float, b: float, samples: np.ndarray, sr: int) -> None:
        try:
            p = self.tmp / f"u{k}.wav"
            _write(p, samples, sr)
            resp = transcribe_file(p, settings=self.client.settings, ledger=self.client.ledger, guard=self.client.guard,
                                   mode="ENDPOINTING", language_bias=self.bias)
            new = _vt_turns(resp, a, k, "vt")
        except Exception as exc:  # noqa: BLE001
            self.bus.emit("status", text=f"transcription failed at {a:.0f} s: {type(exc).__name__}")
            return
        with self.lock:
            for t in new:
                lang = lang_of(t.text, self.src) if t.text else "unk"
                pt = PTurn(len(self.turns), t.start_s, t.end_s, t.text, lang, is_meta(t.text, lang), "vt")
                self.turns.append(pt)
                self.bus.emit("turn", idx=pt.idx, start_s=pt.start_s, end_s=pt.end_s, text=pt.text, lang=pt.lang)
            self.turns.sort(key=lambda t: t.start_s)
        self.tick()

    # --- turns -> pairs -> findings ---
    def result(self, duration_s: float) -> dict:
        """Everything so far, in the review-mode result format (for the end-of-session report)."""
        with self.lock:
            turns = merge_splits([PTurn(t.idx, t.start_s, t.end_s, t.text, t.lang, t.meta, t.source) for t in self.turns])
            pairing = pair_turns(turns, sim=self.sim, **tuned_params())
            items = build_pairs(self.sid, self.src, turns, pairing)
        pairs = []
        for _, rec in items:
            rec.pair_id = f"{self.sid}-p{rec.orig_turns[0]:04d}"
            pairs.append(asdict(rec))
        return {"session_id": self.sid, "src_lang": self.src, "duration_s": duration_s, "pairs": pairs,
                "findings": [asdict(f) for f in sorted(self.findings, key=lambda f: f.rend_start_s)],
                "unpaired": pairing.unpaired, "threshold": self.cal.threshold}

    def tick(self, now: float | None = None) -> None:
        now = self.clock if now is None else now
        with self.lock:
            turns = merge_splits([PTurn(t.idx, t.start_s, t.end_s, t.text, t.lang, t.meta, t.source) for t in self.turns])
            pairing = pair_turns(turns, sim=self.sim, **tuned_params())
            items = build_pairs(self.sid, self.src, turns, pairing)
            last_start = turns[-1].start_s if turns else 0.0
            todo = []
            for p, rec in items:
                p.pair_id = rec.pair_id = f"{self.sid}-p{rec.orig_turns[0]:04d}"  # stable as more turns arrive
                key = (tuple(rec.orig_turns), tuple(rec.rend_turns))
                done = rec.rend_start_s < last_start or now - rec.rend_end_s >= 2.0
                if key not in self.judged and done:
                    self.judged.add(key)
                    todo.append((p, rec))
            runs = [r for r in uninterpreted_runs(pairing, min_single_s=2.0) if r[0]["idx"] not in self.reported_runs
                    and r[-1].get("end_s", 0) < last_start]
            for r in runs:
                self.reported_runs.add(r[0]["idx"])
        for p, rec in todo:
            self.bus.emit("pair", pair_id=rec.pair_id, start_s=rec.orig_start_s, orig=rec.orig_text, rend=rec.rend_text,
                          direction=rec.direction)
            self.pool.submit(self._judge, p, rec)
        for r in runs:
            text = " / ".join(u["text"] for u in r)
            f = Finding(f"{self.sid}-u{r[0]['idx']}", "", "src->en", r[0]["start_s"], r[-1].get("end_s", r[0]["start_s"]),
                        r[0]["start_s"], r[-1].get("end_s", r[0]["start_s"]), text, "", "uninterpreted_exchange",
                        "structure", text[:200], "", f"{len(r)} turn(s) with no English rendition.", "", "major",
                        1.0, 1.0, False, "structure")
            self.findings.append(f)
            self.bus.emit("finding", **asdict(f))

    def _judge(self, p, rec: PairRecord) -> None:
        try:
            fl, _ = judge_pair(self.client, p, model=self.model)
            bl = baseline_pair(self.client, p, model=self.model)
            ver = verify_pair(self.client, p, fl + bl, model=self.model) if (fl or bl) else []
        except Exception as exc:  # noqa: BLE001
            self.bus.emit("status", text=f"judging failed for {rec.pair_id}: {type(exc).__name__}")
            return
        flags = [asdict(f) for f in fl]
        base = [asdict(f) for f in bl]
        for f, v in zip(flags + base, ver):
            f.update(votes=v.votes, yes=v.yes, major=v.major)
        prob = self.cal.probability({"flags": flags, "baseline": base})
        self.bus.emit("judged", pair_id=rec.pair_id, probability=round(prob, 3))
        if prob < self.cal.cutoff:
            return
        for k, (v, origin) in enumerate(select_claims(ver, len(fl)), 1):
            f = v.flag
            fd = Finding(f"{rec.pair_id}-f{k}", rec.pair_id, rec.direction, rec.orig_start_s, rec.orig_end_s,
                         rec.rend_start_s, rec.rend_end_s, rec.orig_text, rec.rend_text, f.type,
                         f.family if origin == "judge" else _family_of(f.type), f.source_words, f.rendition_words,
                         f.explanation, f.corrected_rendition, "major" if v.major * 2 >= v.votes else "minor",
                         round(prob, 3), round(v.share, 3), f.typed_check, origin)
            self.findings.append(fd)
            self.bus.emit("finding", **asdict(fd), latency_s=round(self.clock - rec.rend_end_s, 1))


def run_audio(source, sr: int, seg: Segmenter, live: LiveSession, *, realtime: bool = True, player=None) -> None:
    """Feed int16 blocks from `source` (an iterator), pacing to real time when reading a file."""
    t_start = time.monotonic()
    pos = 0
    for block in source:
        if player is not None:
            player.write(block.reshape(-1, 1))
        for a, b, samples in seg.push(block):
            live.on_utterance(a, b, samples, sr)
        pos += len(block)
        live.clock = pos / sr
        if realtime and player is None:
            delay = t_start + pos / sr - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        if int(live.clock * 10) % 10 == 0:
            live.tick()
    for a, b, samples in seg.push(np.zeros(int(sr * 1.0), dtype=np.int16)):
        live.on_utterance(a, b, samples, sr)
    end = time.monotonic() + 8
    while time.monotonic() < end:
        live.clock += 0.5
        live.tick()
        time.sleep(0.5)


def file_blocks(path: Path, sr_expected: int = 16000):
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="int16")
    if sr != sr_expected or data.ndim != 1:
        raise ValueError(f"{path} must be mono {sr_expected} Hz")
    n = int(BLOCK_S * sr)
    for i in range(0, len(data), n):
        yield data[i: i + n]


def mic_blocks(sr: int = 16000):
    import sounddevice as sd

    q: queue.Queue = queue.Queue()
    with sd.InputStream(samplerate=sr, channels=1, dtype="int16", blocksize=int(BLOCK_S * sr),
                        callback=lambda d, *_: q.put(d[:, 0].copy())):
        while True:
            yield q.get()


def sse_server(bus: EventBus, page_html: str, port: int):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                k = 0
                try:
                    while True:
                        evs = bus.wait_from(k)
                        for ev in evs:
                            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8"))
                        if not evs:
                            self.wfile.write(b": ping\n\n")
                        k += len(evs)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
            else:
                body = page_html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    except OSError:  # port reserved or in use (on Windows, Hyper-V/WSL reserve port ranges): let the OS pick one
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
