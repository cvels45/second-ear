"""Offline tests: pricing, ledger and spend guard, response cache, schemas, mock client."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from second_ear.llm import BudgetExceeded, LLMClient, Ledger, LedgerEntry, SpendGuard, strict_json_schema
from second_ear.llm.cache import ResponseCache, request_key
from second_ear.llm.pricing import asr_cost, token_cost
from second_ear.settings import Settings


class Flag(BaseModel):
    error_present: bool
    error_type: Literal["none", "count", "ordinal"]
    note: str | None = None


def make_settings(tmp: Path, cap_total: float = 150, cap_run: float = 10) -> Settings:
    s = Settings(spend_cap_total=cap_total, spend_cap_per_run=cap_run, data_dir=tmp / "data", private_dir=tmp / "private")
    for d in (s.data_dir, s.private_dir, s.cache_path.parent):
        d.mkdir(parents=True, exist_ok=True)
    return s


def test_token_cost_uses_cached_subset():
    # 1,800 prompt tokens of which 1,200 cached, 450 output, Spark Standard
    cost = token_cost("muse-spark-1.3", 1800, 1200, 450)
    expected = (600 * 1.25 + 1200 * 0.15 + 450 * 4.25) / 1e6
    assert cost == pytest.approx(expected)
    assert token_cost("glimmer", 10_000, 0, 10_000, provider="local") == 0.0


def test_asr_cost_rounds_down_seconds():
    assert asr_cost(3600.9) == pytest.approx(0.18)
    assert asr_cost(59.99) == pytest.approx(59 / 3600 * 0.18)


def test_unknown_model_price_raises():
    with pytest.raises(KeyError):
        token_cost("muse-unknown", 1, 0, 1)


def test_spend_guard_blocks_over_caps(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", run_id="r1")
    ledger.record(LedgerEntry(run_id="r0", provider="meta", model="m", purpose="", cost_usd=149.0))
    guard = SpendGuard(ledger, cap_total=150, cap_per_run=10)
    guard.check(0.5)
    with pytest.raises(BudgetExceeded):
        guard.check(1.5)  # total cap
    ledger2 = Ledger(tmp_path / "l2.jsonl", run_id="r1")
    ledger2.record(LedgerEntry(run_id="r1", provider="meta", model="m", purpose="", cost_usd=9.5))
    with pytest.raises(BudgetExceeded):
        SpendGuard(ledger2, cap_total=150, cap_per_run=10).check(1.0)  # per-run cap


def test_cache_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    key = request_key({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert cache.get(key) is None
    cache.put(key, {"text": "ok", "usage": {}})
    assert cache.get(key) == {"text": "ok", "usage": {}}
    assert key == request_key({"messages": [{"content": "hi", "role": "user"}], "model": "m"})


def test_strict_schema_requires_all_and_forbids_extra():
    name, js = strict_json_schema(Flag)
    assert name == "Flag"
    assert js["additionalProperties"] is False
    assert set(js["required"]) == {"error_present", "error_type", "note"}
    assert "default" not in js["properties"]["note"]


def test_mock_client_parses_schema_and_caches(tmp_path):
    s = make_settings(tmp_path)
    client = LLMClient(provider="mock", settings=s)
    msgs = [{"role": "user", "content": "check this"}]
    r1 = client.chat(messages=msgs, schema=Flag, model="muse-spark-1.3", purpose="t")
    assert isinstance(r1.parsed, Flag) and r1.parsed.error_type == "none"
    r2 = client.chat(messages=msgs, schema=Flag, model="muse-spark-1.3", purpose="t")
    assert r2.from_cache and r2.cost_usd == 0.0
    assert client.ledger.summary()["muse-spark-1.3"]["calls"] == 2


def test_guard_runs_before_call(tmp_path):
    s = make_settings(tmp_path, cap_total=0.0001)
    client = LLMClient(provider="meta", settings=s)  # no network: guard must refuse first
    with pytest.raises(BudgetExceeded):
        client.chat(messages=[{"role": "user", "content": "x" * 3000}], model="muse-wonder-1.0", max_completion_tokens=4000)
