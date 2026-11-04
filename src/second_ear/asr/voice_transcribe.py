"""Muse Voice Transcribe: file endpoint (10 min max per request) and realtime WebSocket.

File:     POST {base}/asr/transcribe, multipart parts "request" (JSON) and "audio" (WAV).
Realtime: wss://.../asr/realtime?sessionId=..., first frame = JSON handshake with
          authorization.accessToken; audio as raw PCM frames; {"type": "endStream"} to flush.
Response turns: {turnId, startMs, endMs, transcript, speaker?}. No word timestamps, no
language field, no confidence (source: dev.meta.ai/docs/speech-to-text).

Default mode is ENDPOINTING. Measured Oct 31 and Nov 1 on 62 s of alternating Spanish/English
(scripts/diag_vt_languages.py, two runs): DIARIZATION and PUSH_TO_TALK returned only the Spanish turns
(0 English words, with or without languageBias), while ENDPOINTING returned all 7 turns in both
languages. English-only audio works in every mode, so the model appears to lock onto the first
language it hears. ENDPOINTING has no speaker labels; roles come from language and turn pairing.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from ..llm.ledger import Ledger, LedgerEntry, SpendGuard
from ..llm.pricing import asr_cost
from ..settings import Settings

MODEL = "muse-voice-transcribe-1.0"
MAX_FILE_SECONDS = 600
MAX_FILE_BYTES = 32_000_000


class VoiceTranscribeError(RuntimeError):
    def __init__(self, status: int | None, detail: str):
        super().__init__(f"Voice Transcribe error {status}: {detail[:300]}")
        self.status = status
        self.detail = detail


def _wav_info(path: Path) -> tuple[int, int, str, float]:
    import soundfile as sf

    info = sf.info(str(path))
    return info.samplerate, info.channels, info.subtype, float(info.duration)


def _config(mode: str, language_bias: list[str] | None, keywords: list[str] | None) -> dict:
    cfg: dict = {"model": MODEL, "mode": mode}
    if language_bias:
        cfg["languageBias"] = language_bias  # English language names, e.g. ["English", "Spanish"]
    if keywords:
        cfg["keywords"] = keywords
    return cfg


def transcribe_file(
    path: Path,
    *,
    settings: Settings,
    ledger: Ledger,
    guard: SpendGuard,
    mode: str = "ENDPOINTING",
    language_bias: list[str] | None = None,
    keywords: list[str] | None = None,
    enforce_limits: bool = True,
    timeout_s: float = 600.0,
) -> dict:
    import httpx

    path = Path(path)
    sr, ch, subtype, dur = _wav_info(path)
    if enforce_limits:
        if sr not in (16000, 24000) or ch != 1 or subtype != "PCM_16":
            raise ValueError(f"{path.name}: need mono PCM_16 WAV at 16/24 kHz, got {ch}ch {subtype} {sr} Hz")
        if dur > MAX_FILE_SECONDS or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{path.name}: {dur:.0f}s / {path.stat().st_size / 1e6:.1f} MB exceeds 10 min / 32 MB")
    guard.check(asr_cost(dur), what=f"voice-transcribe file {path.name}")

    cfg = _config(mode, language_bias, keywords) | {"audioEncoding": "WAV"}
    url = f"{settings.meta_asr_file_url}?sessionId=file-{uuid.uuid4()}"
    t0 = time.time()
    with path.open("rb") as fh:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {settings.model_api_key}", "Accept": "application/json"},
            files={"request": (None, json.dumps(cfg), "application/json"), "audio": (path.name, fh, "audio/wav")},
            timeout=timeout_s,
        )
    if resp.status_code != 200:
        raise VoiceTranscribeError(resp.status_code, resp.text)  # failed requests are not billed
    try:
        data = resp.json()
    except UnicodeDecodeError:
        # Seen once on Tamil audio (Nov 1): an invalid UTF-8 byte in the response. Keep the rest of the
        # text and mark the bad bytes with U+FFFD instead of losing the whole chunk.
        data = json.loads(resp.content.decode("utf-8", errors="replace"))
        data["_decode_repaired"] = True
    billed_s = data.get("audioDurationMs", dur * 1000) / 1000
    ledger.record(
        LedgerEntry(
            run_id=ledger.run_id,
            provider="meta",
            model=MODEL,
            purpose=f"file:{path.name}",
            cost_usd=asr_cost(billed_s),
            audio_seconds=billed_s,
            latency_s=time.time() - t0,
        )
    )
    return data


async def _stream(
    path: Path,
    settings: Settings,
    mode: str,
    language_bias: list[str] | None,
    keywords: list[str] | None,
    pace: float | None,
    idle_timeout_s: float,
) -> dict:
    import numpy as np  # noqa: F401  (soundfile returns numpy arrays)
    import soundfile as sf
    import websockets

    data, sr = sf.read(str(path), dtype="int16")
    if sr not in (16000, 24000) or data.ndim != 1:
        raise ValueError("realtime streaming needs mono 16 or 24 kHz audio")
    pcm = data.tobytes()
    chunk_bytes = int(0.08 * sr) * 2  # 80 ms frames, matching the model's chunking
    handshake = _config(mode, language_bias, keywords) | {
        "authorization": {"accessToken": f"Bearer {settings.model_api_key}"},
        "audioEncoding": f"PCM_{sr // 1000}KHZ",
        "partialMode": "CUMULATIVE",
        "emitAudioProgress": False,
    }
    url = f"{settings.meta_ws_url}?sessionId=stream-{uuid.uuid4()}"
    events: list[dict] = []
    t0 = time.monotonic()
    async with websockets.connect(url, max_size=None, open_timeout=20) as ws:
        await ws.send(json.dumps(handshake))
        ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if "sessionId" not in ack:
            raise VoiceTranscribeError(None, f"handshake rejected: {ack.get('message', ack)}")

        async def sender() -> None:
            # The server closes the stream (1008 "Ingress below real-time") if audio arrives slower than
            # real time, so pacing is deadline-based (no drift from send overhead) and never below 1.0x.
            start = time.monotonic()
            for k, i in enumerate(range(0, len(pcm), chunk_bytes)):
                await ws.send(pcm[i : i + chunk_bytes])
                if pace:
                    delay = start + (k + 1) * 0.08 / max(pace, 1.05) - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
            await ws.send(json.dumps({"type": "endStream"}))
            events.append({"type": "_client_end_stream", "_t": time.monotonic() - t0})

        async def receiver() -> None:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=idle_timeout_s)
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    return
                if isinstance(msg, bytes):
                    continue
                try:
                    ev = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                ev["_t"] = time.monotonic() - t0
                events.append(ev)

        await asyncio.gather(sender(), receiver())
    return {
        "session_id": ack.get("sessionId"),
        "events": events,
        "wall_s": time.monotonic() - t0,
        "audio_s": len(data) / sr,
        "turns": [e for e in events if e.get("type") == "speechComplete"],
    }


def stream_file(
    path: Path,
    *,
    settings: Settings,
    ledger: Ledger,
    guard: SpendGuard,
    mode: str = "ENDPOINTING",
    language_bias: list[str] | None = None,
    keywords: list[str] | None = None,
    pace: float | None = None,
    idle_timeout_s: float = 20.0,
) -> dict:
    """Stream a WAV through the realtime endpoint. pace=None sends as fast as possible; 1.0 = real time."""
    path = Path(path)
    _, _, _, dur = _wav_info(path)
    guard.check(asr_cost(dur), what=f"voice-transcribe realtime {path.name}")
    out = asyncio.run(_stream(path, settings, mode, language_bias, keywords, pace, idle_timeout_s))
    ledger.record(
        LedgerEntry(
            run_id=ledger.run_id,
            provider="meta",
            model=MODEL,
            purpose=f"realtime:{path.name}",
            cost_usd=asr_cost(out["audio_s"]),
            audio_seconds=out["audio_s"],
            latency_s=out["wall_s"],
        )
    )
    return out
