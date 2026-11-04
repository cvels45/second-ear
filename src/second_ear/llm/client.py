"""One client for every LLM call: Meta Model API, local Glimmer (llama.cpp), or a mock.

All three speak the OpenAI Chat Completions format. Every call goes through:
response cache -> spend guard (worst-case estimate) -> API -> cost ledger.
"""

from __future__ import annotations

import base64
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel

from ..settings import Settings, load_settings
from .cache import ResponseCache, request_key
from .ledger import Ledger, LedgerEntry, SpendGuard
from .pricing import estimate_tokens_from_text, token_cost

Provider = Literal["meta", "local", "mock"]
INLINE_AUDIO_LIMIT_BYTES = 50_000_000  # Meta inline file limit


class LLMOutputError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    parsed: Any
    model: str
    provider: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    latency_s: float
    from_cache: bool
    finish_reason: str | None = None


# ---------- JSON schema helpers (strict subset accepted by Meta and llama.cpp) ----------


def _strictify(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)


def strict_json_schema(schema: type[BaseModel] | dict) -> tuple[str, dict]:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        name, js = schema.__name__, schema.model_json_schema()
    else:
        js = copy.deepcopy(schema)
        name = js.get("title", "Response")
    _strictify(js)
    return name, js


def _example_from_schema(js: dict, defs: dict | None = None) -> Any:
    """Minimal instance of a schema, for the mock provider."""
    defs = defs if defs is not None else js.get("$defs", {})
    if "$ref" in js:
        return _example_from_schema(defs[js["$ref"].split("/")[-1]], defs)
    if "anyOf" in js:
        return _example_from_schema(js["anyOf"][0], defs)
    if "enum" in js:
        return js["enum"][0]
    t = js.get("type")
    if t == "object":
        return {k: _example_from_schema(v, defs) for k, v in js.get("properties", {}).items()}
    return {"array": [], "string": "", "integer": 0, "number": 0.0, "boolean": False, "null": None}.get(t)


# ---------- messages ----------


def audio_message(prompt: str, audio_path: Path, system: str | None = None) -> list[dict]:
    """User message with a text prompt plus inline audio (WAV or MP3)."""
    audio_path = Path(audio_path)
    fmt = audio_path.suffix.lower().lstrip(".")
    if fmt not in ("wav", "mp3"):
        raise ValueError("Meta audio input accepts WAV or MP3 only")
    size = audio_path.stat().st_size
    if size * 4 / 3 > INLINE_AUDIO_LIMIT_BYTES:
        raise ValueError(f"{audio_path.name} is {size / 1e6:.1f} MB; over the 50 MB inline limit (use MP3 or the Files API)")
    b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    msgs: list[dict] = [{"role": "system", "content": system}] if system else []
    msgs.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
            ],
        }
    )
    return msgs


def _text_only(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if p.get("type") == "text")
    return "\n".join(parts)


# ---------- client ----------


class LLMClient:
    def __init__(
        self,
        provider: Provider = "meta",
        settings: Settings | None = None,
        ledger: Ledger | None = None,
        use_cache: bool = True,
        mock_responder: Callable[[dict], dict] | None = None,
        timeout_s: float = 600.0,
    ):
        self.provider = provider
        self.settings = settings or load_settings()
        self.ledger = ledger or Ledger(self.settings.ledger_path)
        self.guard = SpendGuard(self.ledger, self.settings.spend_cap_total, self.settings.spend_cap_per_run)
        self.cache = ResponseCache(self.settings.cache_path)
        self.use_cache = use_cache
        self.mock_responder = mock_responder
        self._client = None
        self._timeout = timeout_s

    def _sdk(self):
        if self._client is None:
            from openai import OpenAI  # lazy: not needed for mock/offline tests

            if self.provider == "meta":
                if not self.settings.model_api_key:
                    raise RuntimeError("MODEL_API_KEY is not set (second-ear/.env)")
                self._client = OpenAI(
                    api_key=self.settings.model_api_key,
                    base_url=self.settings.meta_api_base,
                    timeout=self._timeout,
                    max_retries=3,
                )
            else:
                self._client = OpenAI(
                    api_key="local", base_url=self.settings.local_llm_url, timeout=self._timeout, max_retries=1
                )
        return self._client

    def chat(
        self,
        *,
        messages: list[dict],
        model: str | None = None,
        schema: type[BaseModel] | dict | None = None,
        reasoning_effort: str | None = None,
        max_completion_tokens: int = 2048,
        temperature: float | None = None,
        seed: int | None = None,
        prompt_cache_key: str | None = None,
        purpose: str = "",
        use_cache: bool | None = None,
        audio_seconds: float = 0.0,
    ) -> LLMResult:
        model = model or (self.settings.local_llm_model if self.provider == "local" else "muse-spark-1.3")
        body: dict[str, Any] = {"model": model, "messages": messages}
        if schema is not None:
            name, js = strict_json_schema(schema)
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": js, "strict": True},
            }
        if temperature is not None:
            body["temperature"] = temperature
        if seed is not None:
            body["seed"] = seed
        extra: dict[str, Any] = {}
        if self.provider == "meta":
            body["max_completion_tokens"] = max_completion_tokens
            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort
            if prompt_cache_key:
                extra["prompt_cache_key"] = prompt_cache_key
        else:
            body["max_tokens"] = max_completion_tokens

        use_cache = self.use_cache if use_cache is None else use_cache
        key = request_key({"provider": self.provider, **body, **extra})
        if use_cache and (hit := self.cache.get(key)) is not None:
            return self._finish(hit, model, schema, purpose, latency=0.0, from_cache=True)

        est_prompt = estimate_tokens_from_text(_text_only(messages)) + int(
            audio_seconds * self.settings.audio_tokens_per_sec
        )
        estimate = token_cost(model, est_prompt, 0, max_completion_tokens, self.provider)
        self.guard.check(estimate, what=purpose or model)

        t0 = time.time()
        if self.provider == "mock":
            raw = (self.mock_responder or self._default_mock)(body)
        else:
            resp = self._sdk().chat.completions.create(**body, extra_body=extra or None)
            raw = self._raw_from_response(resp)
        latency = time.time() - t0
        if use_cache and raw.get("finish_reason") != "length":
            self.cache.put(key, raw)
        return self._finish(raw, model, schema, purpose, latency=latency, from_cache=False)

    # ---- internals ----

    def _default_mock(self, body: dict) -> dict:
        rf = body.get("response_format")
        text = json.dumps(_example_from_schema(rf["json_schema"]["schema"])) if rf else "mock"
        return {"text": text, "finish_reason": "stop", "usage": {}}

    @staticmethod
    def _raw_from_response(resp: Any) -> dict:
        choice = resp.choices[0]
        u = resp.usage
        ptd = getattr(u, "prompt_tokens_details", None) if u else None
        ctd = getattr(u, "completion_tokens_details", None) if u else None
        return {
            "text": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "cached_tokens": (getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0,
                "reasoning_tokens": (getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0,
            },
        }

    def _finish(self, raw: dict, model: str, schema, purpose: str, latency: float, from_cache: bool) -> LLMResult:
        u = raw.get("usage", {})
        pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        cached, reasoning = u.get("cached_tokens", 0), u.get("reasoning_tokens", 0)
        cost = 0.0 if from_cache else token_cost(model, pt, cached, ct, self.provider)
        self.ledger.record(
            LedgerEntry(
                run_id=self.ledger.run_id,
                provider=self.provider,
                model=model,
                purpose=purpose,
                cost_usd=cost,
                prompt_tokens=pt,
                cached_tokens=cached,
                completion_tokens=ct,
                reasoning_tokens=reasoning,
                cache_hit=from_cache,
                latency_s=latency,
            )
        )
        text = raw.get("text", "")
        parsed: Any = None
        if schema is not None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMOutputError(
                    f"Model returned non-JSON (finish_reason={raw.get('finish_reason')}): {text[:200]!r}"
                ) from exc
            parsed = schema.model_validate(data) if isinstance(schema, type) else data
        return LLMResult(
            text=text,
            parsed=parsed,
            model=model,
            provider=self.provider,
            prompt_tokens=pt,
            cached_tokens=cached,
            completion_tokens=ct,
            reasoning_tokens=reasoning,
            cost_usd=cost,
            latency_s=latency,
            from_cache=from_cache,
            finish_reason=raw.get("finish_reason"),
        )
