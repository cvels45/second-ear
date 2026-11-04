"""Replay a recorded live session (events.jsonl from scripts/live.py) at its original pace, for the demo video.

Usage (from second-ear/):
  uv run python scripts/replay_live.py ..\\data\\live\\es-S02 --lang es --title "Prep session, Spanish"
Then open http://127.0.0.1:8766 and play the session audio alongside.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live import PAGE  # noqa: E402

from second_ear import report  # noqa: E402
from second_ear.live import EventBus, sse_server  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--title", default="Prep session")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--wait", type=float, default=3.0, help="seconds before the replay starts")
    args = ap.parse_args()
    events = [json.loads(x) for x in (args.folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    bus = EventBus()
    page = (PAGE.replace("__CSS__", report.CSS).replace("__TITLE__", args.title)
            .replace("__LANG__", report.LANG.get(args.lang, args.lang)).replace("__TYPES__", json.dumps(report.TYPE_LABEL)))
    srv = sse_server(bus, page, args.port)
    print(f"Replay page: http://127.0.0.1:{srv.server_address[1]} (starts in {args.wait:.0f} s)", flush=True)
    time.sleep(args.wait)
    t0 = time.time()
    for ev in events:
        delay = t0 + ev.get("t_rel", 0) - time.time()
        if delay > 0:
            time.sleep(delay)
        data = {k: v for k, v in ev.items() if k not in ("id", "kind", "t", "t_rel")}
        bus.emit(ev["kind"], **data)
    print("Replay finished.")
    time.sleep(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
