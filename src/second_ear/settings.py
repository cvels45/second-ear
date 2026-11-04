"""Runtime settings, loaded from second-ear/.env and the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]  # second-ear/
REPO_ROOT = PKG_ROOT.parent  # meta_ai_hackathon_2026/


def _load_dotenv() -> None:
    env_path = PKG_ROOT / ".env"
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    if env_path.exists():  # minimal fallback parser
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _path(value: str | None, default: Path) -> Path:
    if not value:
        return default.resolve()
    p = Path(value)
    return (p if p.is_absolute() else PKG_ROOT / p).resolve()


@dataclass
class Settings:
    model_api_key: str = ""
    meta_api_base: str = "https://api.meta.ai/v1"
    local_llm_url: str = "http://127.0.0.1:8001/v1"
    local_llm_model: str = "glimmer"
    spend_cap_total: float = 150.0
    spend_cap_per_run: float = 10.0
    audio_tokens_per_sec: float = 15.0  # measured 12.2 tok/s on Spark 1.2 and Wonder (smoke test, Oct 31)
    data_dir: Path = field(default_factory=lambda: (REPO_ROOT / "data").resolve())
    private_dir: Path = field(default_factory=lambda: (REPO_ROOT / "private").resolve())

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "cache" / "llm_cache.sqlite"

    @property
    def ledger_path(self) -> Path:
        return self.private_dir / "ledger.jsonl"

    @property
    def meta_ws_url(self) -> str:
        return self.meta_api_base.replace("https://", "wss://").rstrip("/") + "/asr/realtime"

    @property
    def meta_asr_file_url(self) -> str:
        return self.meta_api_base.rstrip("/") + "/asr/transcribe"


def load_settings() -> Settings:
    _load_dotenv()
    env = os.environ
    s = Settings(
        model_api_key=env.get("MODEL_API_KEY", ""),
        meta_api_base=env.get("META_API_BASE", "https://api.meta.ai/v1"),
        local_llm_url=env.get("SE_LOCAL_LLM_URL", "http://127.0.0.1:8001/v1"),
        local_llm_model=env.get("SE_LOCAL_LLM_MODEL", "glimmer"),
        spend_cap_total=float(env.get("SE_SPEND_CAP_TOTAL", "150")),
        spend_cap_per_run=float(env.get("SE_SPEND_CAP_PER_RUN", "10")),
        audio_tokens_per_sec=float(env.get("SE_AUDIO_TOKENS_PER_SEC", "15")),
        data_dir=_path(env.get("SE_DATA_DIR"), REPO_ROOT / "data"),
        private_dir=_path(env.get("SE_PRIVATE_DIR"), REPO_ROOT / "private"),
    )
    for d in (s.data_dir, s.private_dir, s.runs_dir, s.cache_path.parent):
        d.mkdir(parents=True, exist_ok=True)
    return s
