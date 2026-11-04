"""SQLite response cache: identical requests are answered from disk at no cost."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def request_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL, "
            "created REAL DEFAULT (julianday('now')))"
        )

    def _execute(self, sql: str, params: tuple = (), fetch: bool = False):
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            try:
                row = conn.execute(sql, params).fetchone() if fetch else conn.execute(sql, params)
                conn.commit()
                return row if fetch else None
            finally:
                conn.close()

    def get(self, key: str) -> dict | None:
        row = self._execute("SELECT response FROM responses WHERE key = ?", (key,), fetch=True)
        return json.loads(row[0]) if row else None

    def put(self, key: str, response: dict) -> None:
        self._execute(
            "INSERT OR REPLACE INTO responses (key, response) VALUES (?, ?)",
            (key, json.dumps(response, ensure_ascii=False)),
        )
