"""Check that the build PC is ready: GPU, PyTorch, tools, local Glimmer server, API key, disk.

Usage (from second-ear/):  uv run python scripts/check_env.py
Standard library only; PyTorch is checked if installed (uv sync --group gpu).
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # second-ear/
RESULTS: list[tuple[str, str, str]] = []  # (status, item, detail)


def record(status: str, item: str, detail: str) -> None:
    RESULTS.append((status, item, detail))


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def run(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, str(exc)


def check_python() -> None:
    v = sys.version_info
    detail = f"{platform.python_version()} on {platform.system()} {platform.machine()}"
    record("PASS" if (v.major, v.minor) == (3, 14) else "WARN", "Python", detail)


def check_nvidia() -> None:
    if not shutil.which("nvidia-smi"):
        record("FAIL", "NVIDIA driver", "nvidia-smi not found on PATH")
        return
    code, out = run(["nvidia-smi"])
    if code != 0:
        record("FAIL", "NVIDIA driver", out.strip()[:200])
        return
    driver = re.search(r"Driver Version:\s*([\d.]+)", out)
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", out)
    gpu = re.search(r"(RTX\s*\d{4}[^|]*)", out)
    cuda_v = tuple(int(x) for x in cuda.group(1).split(".")) if cuda else (0,)
    ok = cuda_v >= (13, 1)
    record(
        "PASS" if ok else "FAIL",
        "NVIDIA driver",
        f"GPU {gpu.group(1).strip() if gpu else '?'} | driver {driver.group(1) if driver else '?'} | "
        f"CUDA {cuda.group(1) if cuda else '?'}" + ("" if ok else " (need 13.1+: update the driver)"),
    )


def check_torch() -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        record("WARN", "PyTorch", "not installed; run: uv sync --group gpu")
        return
    if not torch.cuda.is_available():
        record("FAIL", "PyTorch", f"{torch.__version__} installed but CUDA not available (CPU build?)")
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    t0 = time.time()
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    (a @ a).sum().item()
    ms = (time.time() - t0) * 1000
    record(
        "PASS",
        "PyTorch",
        f"{torch.__version__} | CUDA {torch.version.cuda} | {name} | sm_{cap[0]}{cap[1]} | matmul {ms:.0f} ms",
    )


def check_tool(name: str, args: list[str], required: bool, extra_paths: list[Path] | None = None) -> None:
    exe = shutil.which(name)
    if not exe and extra_paths:
        exe = next((str(p) for p in extra_paths if p.exists()), None)
    if not exe:
        record("FAIL" if required else "WARN", name, "not found (see SETUP.md)")
        return
    code, out = run([exe, *args])
    first = out.strip().splitlines()[0] if out.strip() else ""
    record("PASS" if code == 0 else "WARN", name, first[:120])


def http_json(url: str, payload: dict | None = None, timeout: int = 180) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local URL)
        return json.loads(resp.read().decode())


def check_llama_server(base: str, model: str) -> None:
    root = base.rsplit("/v1", 1)[0]
    try:
        http_json(f"{root}/health", timeout=10)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        record("WARN", "llama-server", f"not reachable at {root} ({exc}); start it per SETUP.md step 4")
        return
    schema = {
        "type": "object",
        "properties": {
            "error_present": {"type": "boolean"},
            "source_number": {"type": "string"},
            "rendered_number": {"type": "string"},
        },
        "required": ["error_present", "source_number", "rendered_number"],
        "additionalProperties": False,
    }
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": "You compare a statement with its interpretation. Reply in JSON."},
            {
                "role": "user",
                "content": 'Source (Spanish): "Me golpearon tres veces."\n'
                'Interpretation (English): "They hit me twice."\n'
                "Did the interpreter change the number?",
            },
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "check", "schema": schema}},
    }
    t0 = time.time()
    try:
        out = http_json(f"{base}/chat/completions", payload)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        record("FAIL", "llama-server", f"chat request failed: {exc}")
        return
    secs = time.time() - t0
    msg = out["choices"][0]["message"].get("content") or ""
    usage = out.get("usage", {})
    try:
        parsed = json.loads(msg)
        ok = parsed.get("error_present") is True
        tok = usage.get("completion_tokens", 0)
        rate = f"{tok / secs:.0f} tok/s" if secs > 0 and tok else "?"
        record(
            "PASS" if ok else "WARN",
            "llama-server",
            f"JSON ok, error_present={parsed.get('error_present')} | {tok} tokens in {secs:.1f}s ({rate})",
        )
    except json.JSONDecodeError:
        record("FAIL", "llama-server", f"response was not valid JSON: {msg[:120]!r}")


def check_api_key(env: dict[str, str]) -> None:
    key = env.get("MODEL_API_KEY") or os.environ.get("MODEL_API_KEY", "")
    if not key:
        record("FAIL", "Meta API key", "MODEL_API_KEY missing in second-ear/.env")
    else:
        record("PASS", "Meta API key", f"set ({len(key)} chars; not printed)")


def check_disk(env: dict[str, str]) -> None:
    data_dir = (ROOT / env.get("SE_DATA_DIR", "../data")).resolve()
    probe = data_dir if data_dir.exists() else data_dir.anchor or ROOT
    free_gb = shutil.disk_usage(probe).free / 1e9
    status = "PASS" if free_gb >= 80 else "WARN"
    record(status, "Disk", f"{free_gb:.0f} GB free at {probe} (want 80+ GB for data and models)")


def main() -> int:
    env = load_env(ROOT / ".env")
    check_python()
    check_nvidia()
    check_torch()
    check_tool("ffmpeg", ["-version"], required=True)
    check_tool("git", ["--version"], required=True)
    check_llama_server(
        env.get("SE_LOCAL_LLM_URL", "http://127.0.0.1:8001/v1"), env.get("SE_LOCAL_LLM_MODEL", "glimmer")
    )
    check_api_key(env)
    check_disk(env)

    width = max(len(item) for _, item, _ in RESULTS)
    for status, item, detail in RESULTS:
        print(f"[{status:4}] {item:<{width}}  {detail}")
    fails = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    print(f"\n{len(RESULTS)} checks, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
