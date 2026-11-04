"""Audio session sets in one format: a WAV, an MP3, and gold turns with times (the dev gold format).

  dev        data/dev/sessions/<id>.gold.json             36 scripted dialogues, Claude-written labels
  t4-calib   data/test_audio/calib/sessions/<id>.gold.json  T1 calibration pairs, voiced (tuning allowed)
  t4-test    data/test_audio/test/sessions/<id>.gold.json   T1 locked test pairs, voiced (run once)
  t3         data/test_clean/<lang>/gold.json             FLEURS clean pairs, real speech (false alarms)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SETS = ["dev", "t4-calib", "t4-test", "t3"]
LOCKED = {"t4-test", "t3"}


@dataclass
class SessionRef:
    session_id: str
    wav: Path
    mp3: Path
    gold: dict


def _t3_to_gold(g: dict) -> dict:
    turns = []
    for k, p in enumerate(g["pairs"]):
        o, r = p["original"], p["rendition"]
        oid, rid = f"{g['session_id']}-t{2 * k + 1:04d}", f"{g['session_id']}-t{2 * k + 2:04d}"
        turns.append({"turn_id": oid, "kind": "original", "lang": o["lang"], "text": o["text"], "renders": None,
                      "label": None, "start_s": o["start_s"], "end_s": o["end_s"]})
        turns.append({"turn_id": rid, "kind": "rendition", "lang": r["lang"], "text": r["text"], "renders": oid,
                      "label": {"status": "faithful"}, "start_s": r["start_s"], "end_s": r["end_s"]})
    return {"dialogue_id": g["session_id"], "lang": g["lang"], "turns": turns, "duration_s": g["duration_s"]}


def load_set(data_dir: Path, name: str) -> list[SessionRef]:
    if name == "dev" or name.startswith("t4-"):
        d = data_dir / "dev" / "sessions" if name == "dev" else data_dir / "test_audio" / name[3:] / "sessions"
        out = []
        for gp in sorted(d.glob("*.gold.json")):
            sid = gp.name.split(".")[0]
            out.append(SessionRef(sid, d / f"{sid}.wav", d / f"{sid}.mp3", json.loads(gp.read_text(encoding="utf-8"))))
        return out
    if name == "t3":
        out = []
        for gp in sorted((data_dir / "test_clean").glob("*/gold.json")):
            g = _t3_to_gold(json.loads(gp.read_text(encoding="utf-8")))
            out.append(SessionRef(g["dialogue_id"], gp.parent / "session.wav", gp.parent / "session.mp3", g))
        return out
    raise ValueError(f"unknown set {name}; one of {SETS}")


def asr_dir(runs_dir: Path, name: str) -> Path:
    return runs_dir / "asr" / ("dev-utt" if name == "dev" else f"{name}-utt")
