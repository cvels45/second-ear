"""Meta Model API prices (Standard mode), USD per 1M tokens. Source: dev.meta.ai/docs/pricing-rate-limits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    cached_input: float
    output: float  # includes reasoning tokens


PRICES: dict[str, Price] = {
    "muse-spark-1.3": Price(1.25, 0.15, 4.25),
    "muse-spark-1.2": Price(1.25, 0.15, 4.25),
    "muse-spark-1.1": Price(1.25, 0.15, 4.25),
    "muse-wonder-1.0": Price(4.00, 0.20, 20.00),
}
FREE = Price(0.0, 0.0, 0.0)  # local models (Glimmer via llama.cpp) and mocks

VOICE_TRANSCRIBE_PER_HOUR = 0.18  # billed on whole seconds, rounded down


def price_for(model: str, provider: str = "meta") -> Price:
    if provider != "meta":
        return FREE
    if model not in PRICES:
        raise KeyError(f"No price for model {model!r}; add it to PRICES before spending on it")
    return PRICES[model]


def token_cost(
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    provider: str = "meta",
) -> float:
    """Cost of one call. cached_tokens is a subset of prompt_tokens (Meta's usage semantics)."""
    p = price_for(model, provider)
    cached = min(cached_tokens, prompt_tokens)
    fresh = prompt_tokens - cached
    return (fresh * p.input + cached * p.cached_input + completion_tokens * p.output) / 1_000_000


def asr_cost(audio_seconds: float) -> float:
    return int(audio_seconds) / 3600 * VOICE_TRANSCRIBE_PER_HOUR


def estimate_tokens_from_text(text: str) -> int:
    """Conservative token estimate (about 3 characters per token, rounded up)."""
    return len(text) // 3 + 1
