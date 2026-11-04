"""Download the Kokoro model files (Apache 2.0) into data/models/kokoro/. About 700 MB, once.

Usage (from second-ear/tts/):  uv run python download_models.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

GH = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
FILES = {
    "kokoro-v1.0.onnx": f"{GH}/kokoro-v1.0.onnx",      # en, es, pt, hi (espeak-ng phonemes)
    "voices-v1.0.bin": f"{GH}/voices-v1.0.bin",
    "kokoro-v1.1-zh.onnx": f"{GH}/kokoro-v1.1-zh.onnx",  # Mandarin (misaki G2P)
    "voices-v1.1-zh.bin": f"{GH}/voices-v1.1-zh.bin",
    "config-v1.1-zh.json": "https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/raw/main/config.json",
}
MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "kokoro"


def fetch(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:  # noqa: S310 (fixed URLs)
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dst.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    tmp.replace(dst)
    print(f"\r  {dst.name}: {dst.stat().st_size / 1e6:.1f} MB" + " " * 20)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = MODELS_DIR / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  {name}: present")
            continue
        fetch(url, dst)
    print(f"Models in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
