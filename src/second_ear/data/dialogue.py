"""Scripted dev dialogues: master scenarios (English) + per-language specs -> labeled turn lists.

A master scenario holds the English side of a fictional hearing as numbered exchanges:
a question by the judge or an attorney, and the applicant's answer (one string, or a list of parts
when a long answer is interpreted in pieces). A per-language spec holds the source-language lines
(the interpreter's faithful rendition of each question and the applicant's answer), the slot values
(names, places, dates), the voices, and the planted renditions:

  plant = what the interpreter actually says at one point, with its label (an error type, or
          "acceptable" for a meaning-preserving paraphrase).

Every rendition without a plant is faithful by construction: the English answer and its
source-language version were written together. Claude writes and labels all dev data (a different
model family from the Muse judges). Test data never comes from here (see docs/test-data.md).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..taxonomy import ErrorType

Side = Literal["q", "a"]
Text = str | list[str]


class Exchange(BaseModel):
    x: int
    q_by: str
    q: str
    a: Text


class Master(BaseModel):
    scenario_id: str
    setting: Literal["merits_hearing", "prep_session", "deposition", "bond_hearing"]
    title: str
    speakers: dict[str, str]  # code -> role
    exchanges: list[Exchange]

    @model_validator(mode="after")
    def _check(self) -> Master:
        xs = [e.x for e in self.exchanges]
        if xs != sorted(set(xs)):
            raise ValueError(f"{self.scenario_id}: exchange numbers must be unique and increasing")
        for e in self.exchanges:
            if e.q_by not in self.speakers:
                raise ValueError(f"{self.scenario_id} x{e.x}: unknown speaker {e.q_by}")
        return self


class Line(BaseModel):
    q: str
    a: Text


class Plant(BaseModel):
    x: int
    side: Side
    part: int = 0
    type: ErrorType | Literal["acceptable"]
    say: str  # what the interpreter actually says
    src_span: str = ""  # the words in the source that were mistranslated (or "" for additions)
    say_span: str = ""  # the words in the rendition that carry the change (or "" for omissions)
    severity: Literal["major", "minor"] = "major"
    dispositive: bool = False  # the mock decision relies on this error
    note: str = ""


class ExtraTurn(BaseModel):
    by: str
    lang: str
    text: str


class Extra(BaseModel):
    after_x: int
    kind: Literal["interpreter_meta", "uninterpreted"]
    turns: list[ExtraTurn]
    note: str = ""


class Decision(BaseModel):
    text: str
    cites_x: list[int] = Field(default_factory=list)  # exchanges whose rendered answers the decision relies on


class Spec(BaseModel):
    dialogue_id: str
    lang: str
    scenario_id: str
    slots: dict[str, str]
    voices: dict[str, str | dict[str, str]]  # speaker -> voice, or lang -> voice for the interpreter
    lines: dict[int, Line]
    plants: list[Plant] = Field(default_factory=list)
    extras: list[Extra] = Field(default_factory=list)
    decision: Decision | None = None


def _parts(t: Text) -> list[str]:
    return [t] if isinstance(t, str) else list(t)


def _fill(text: str, slots: dict[str, str]) -> str:
    try:
        return text.format_map(slots)
    except KeyError as exc:
        raise ValueError(f"missing slot {exc} in: {text[:80]}") from exc


def assemble(master: Master, spec: Spec) -> dict:
    """Build the labeled turn list for one dialogue."""
    if spec.scenario_id != master.scenario_id:
        raise ValueError(f"{spec.dialogue_id}: scenario mismatch")
    missing = [e.x for e in master.exchanges if e.x not in spec.lines]
    if missing:
        raise ValueError(f"{spec.dialogue_id}: no lines for exchanges {missing}")
    plants: dict[tuple[int, Side, int], Plant] = {}
    for p in spec.plants:
        key = (p.x, p.side, p.part)
        if key in plants:
            raise ValueError(f"{spec.dialogue_id}: two plants at {key}")
        plants[key] = p
    extras_after: dict[int, list[Extra]] = {}
    for ex in spec.extras:
        extras_after.setdefault(ex.after_x, []).append(ex)

    s, turns = spec.slots, []

    def add(speaker: str, lang: str, text: str, kind: str, renders: str | None = None,
            faithful: str | None = None, label: dict | None = None, x: int | None = None) -> str:
        tid = f"{spec.dialogue_id}-t{len(turns) + 1:03d}"
        turns.append({"turn_id": tid, "speaker": speaker, "role": master.speakers.get(speaker, speaker),
                      "lang": lang, "text": text, "kind": kind, "x": x, "renders": renders,
                      "faithful": faithful, "label": label})
        return tid

    def rendition(src_id: str, lang: str, faithful: str, key: tuple[int, Side, int]) -> None:
        p = plants.pop(key, None)
        if p is None:
            add("I", lang, faithful, "rendition", src_id, faithful, {"status": "faithful"}, key[0])
            return
        status = "acceptable" if p.type == "acceptable" else "error"
        label = {"status": status, "type": None if status == "acceptable" else str(p.type),
                 "src_span": _fill(p.src_span, s), "say_span": _fill(p.say_span, s), "severity": p.severity,
                 "dispositive": p.dispositive, "note": p.note}
        add("I", lang, _fill(p.say, s), "rendition", src_id, faithful, label, key[0])

    for e in master.exchanges:
        line = spec.lines[e.x]
        q_id = add(e.q_by, "en", _fill(e.q, s), "original", x=e.x)
        rendition(q_id, spec.lang, _fill(line.q, s), (e.x, "q", 0))
        en_parts, src_parts = _parts(e.a), _parts(line.a)
        if len(en_parts) != len(src_parts):
            raise ValueError(f"{spec.dialogue_id} x{e.x}: {len(en_parts)} English parts, {len(src_parts)} source parts")
        for i, (en, src) in enumerate(zip(en_parts, src_parts)):
            a_id = add("A", spec.lang, _fill(src, s), "original", x=e.x)
            rendition(a_id, "en", _fill(en, s), (e.x, "a", i))
        for ex in extras_after.pop(e.x, []):
            for j, t in enumerate(ex.turns):
                if ex.kind == "uninterpreted":
                    label = {"status": "error", "type": str(ErrorType.UNINTERPRETED), "src_span": "", "say_span": "",
                             "severity": "major", "dispositive": False, "note": ex.note} if j == 0 else None
                    add(t.by, t.lang, _fill(t.text, s), "side", label=label, x=e.x)
                else:
                    add(t.by, t.lang, _fill(t.text, s), "meta", label={"status": "meta"}, x=e.x)
    if plants:
        raise ValueError(f"{spec.dialogue_id}: plants point at missing turns {sorted(plants)}")
    if extras_after:
        raise ValueError(f"{spec.dialogue_id}: extras after missing exchanges {sorted(extras_after)}")

    decision = None
    if spec.decision:
        cites = [t["turn_id"] for t in turns
                 if t["kind"] == "rendition" and t["lang"] == "en" and t["x"] in spec.decision.cites_x]
        decision = {"text": _fill(spec.decision.text, s), "cites": cites}
    return {"dialogue_id": spec.dialogue_id, "lang": spec.lang, "scenario_id": master.scenario_id,
            "setting": master.setting, "title": master.title, "voices": spec.voices, "turns": turns,
            "decision": decision}


def voice_for(dialogue: dict, speaker: str, lang: str) -> str | None:
    v = dialogue["voices"].get(speaker)
    if isinstance(v, dict):
        return v.get(lang) or v.get("default")
    return v


def utterances(dialogue: dict) -> list[dict]:
    """TTS input rows for one dialogue."""
    return [{"id": t["turn_id"], "lang": t["lang"], "text": t["text"],
             "voice": voice_for(dialogue, t["speaker"], t["lang"])} for t in dialogue["turns"]]


def stats(dialogues: list[dict]) -> dict:
    types, status, renditions = Counter(), Counter(), 0
    for d in dialogues:
        for t in d["turns"]:
            lab = t["label"] or {}
            if t["kind"] == "rendition":
                renditions += 1
            status[lab.get("status", "none")] += 1
            if lab.get("status") == "error":
                types[lab["type"]] += 1
    return {"dialogues": len(dialogues), "renditions": renditions, "status": dict(status), "error_types": dict(types)}


def load_master(path: Path) -> Master:
    return Master.model_validate_json(path.read_text(encoding="utf-8"))


def load_spec(path: Path) -> Spec:
    return Spec.model_validate_json(path.read_text(encoding="utf-8"))


def build_all(dev_dir: Path) -> list[dict]:
    """dev_dir/scenarios/S01.json ..., dev_dir/<lang>/<lang>-S01.json ... -> assembled dialogues."""
    masters = {m.scenario_id: m for m in (load_master(p) for p in sorted((dev_dir / "scenarios").glob("S*.json")))}
    out = []
    for p in sorted(dev_dir.glob("*/*-S*.json")):
        spec = load_spec(p)
        out.append(assemble(masters[spec.scenario_id], spec))
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
