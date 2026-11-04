"""Append-only cost ledger and a hard spend guard.

Every billable call is written to private/ledger.jsonl. Before a call, the guard checks
that spent-so-far plus the call's worst-case estimate stays under both caps.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LedgerEntry:
    run_id: str
    provider: str
    model: str
    purpose: str
    cost_usd: float
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    audio_seconds: float = 0.0
    cache_hit: bool = False
    latency_s: float = 0.0
    ts: float = field(default_factory=time.time)


class Ledger:
    def __init__(self, path: Path, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self._lock = threading.Lock()
        self._totals: tuple[float, float] | None = None  # (all runs, this run), kept in memory after one read

    def _load_totals(self) -> tuple[float, float]:
        if self._totals is None:
            es = self.entries()
            self._totals = (sum(e["cost_usd"] for e in es), sum(e["cost_usd"] for e in es if e["run_id"] == self.run_id))
        return self._totals

    def record(self, entry: LedgerEntry) -> None:
        with self._lock:
            total, run = self._load_totals()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
            self._totals = (total + entry.cost_usd, run + (entry.cost_usd if entry.run_id == self.run_id else 0.0))

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def total_spent(self) -> float:
        with self._lock:
            return self._load_totals()[0]

    def run_spent(self, run_id: str | None = None) -> float:
        if run_id is None or run_id == self.run_id:
            with self._lock:
                return self._load_totals()[1]
        return sum(e["cost_usd"] for e in self.entries() if e["run_id"] == run_id)

    def summary(self, run_id: str | None = None) -> dict[str, dict[str, float]]:
        rid = run_id or self.run_id
        by_model: dict[str, dict[str, float]] = {}
        for e in self.entries():
            if e["run_id"] != rid:
                continue
            m = by_model.setdefault(e["model"], {"calls": 0, "cost_usd": 0.0, "cache_hits": 0})
            m["calls"] += 1
            m["cost_usd"] += e["cost_usd"]
            m["cache_hits"] += int(e["cache_hit"])
        return by_model


class SpendGuard:
    def __init__(self, ledger: Ledger, cap_total: float, cap_per_run: float):
        self.ledger = ledger
        self.cap_total = cap_total
        self.cap_per_run = cap_per_run

    def check(self, estimate_usd: float, what: str = "") -> None:
        total = self.ledger.total_spent()
        run = self.ledger.run_spent()
        if total + estimate_usd > self.cap_total:
            raise BudgetExceeded(
                f"{what}: total spend ${total:.2f} + estimate ${estimate_usd:.2f} "
                f"would exceed SE_SPEND_CAP_TOTAL=${self.cap_total:.2f}"
            )
        if run + estimate_usd > self.cap_per_run:
            raise BudgetExceeded(
                f"{what}: this run's spend ${run:.2f} + estimate ${estimate_usd:.2f} "
                f"would exceed SE_SPEND_CAP_PER_RUN=${self.cap_per_run:.2f}"
            )
