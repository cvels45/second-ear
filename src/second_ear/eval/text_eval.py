"""Text-level evaluation: load labeled pairs (dev gold, T1 human-labeled) and score flags.

Detection is scored per pair: a pair is flagged if any finding survives. Positives are gold errors;
negatives are faithful and acceptable renditions (dev) or clean segments (T1). Type recall counts a
gold error as typed correctly when a flag of its family is present.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..judges import PairInput
from ..taxonomy import TYPES, ErrorType

T1_TYPE_FAMILY = {"distortion": "meaning", "omission": "omission_addition", "addition": "omission_addition"}


def dev_pairs(dialogues_jsonl: Path, langs: set[str] | None = None) -> list[tuple[PairInput, dict]]:
    out = []
    for line in dialogues_jsonl.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if langs and d["lang"] not in langs:
            continue
        by_id = {t["turn_id"]: t for t in d["turns"]}
        history: list[tuple[str, str]] = []
        for t in d["turns"]:
            if t["kind"] != "rendition" or not t["renders"]:
                continue
            o = by_id[t["renders"]]
            direction = "src->en" if o["lang"] != "en" else "en->src"
            p = PairInput(t["turn_id"], d["lang"], direction, o["text"], t["text"], list(history))
            lab = t["label"] or {"status": "faithful"}
            gold = {"label": "error" if lab["status"] == "error" else "clean", "status": lab["status"],
                    "types": [lab["type"]] if lab.get("type") else [], "lang": d["lang"], "set": "dev",
                    "severity": lab.get("severity"), "dispositive": lab.get("dispositive", False)}
            out.append((p, gold))
            history.append((o["text"], t["text"]))
    return out


def t1_pairs(pairs_jsonl: Path, split: str) -> list[tuple[PairInput, dict]]:
    out = []
    for line in pairs_jsonl.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] != split:
            continue
        if r["src_lang"] == "en":
            src_lang, direction = r["tgt_lang"], "en->src"
        else:
            src_lang, direction = r["src_lang"], "src->en"
        p = PairInput(r["id"], src_lang, direction, r["src"], r["tgt"], [tuple(c) for c in r.get("context", [])])
        out.append((p, {"label": r["label"], "status": r["label"], "types": r["types"], "lang": src_lang,
                        "set": r["source_ds"], "origin": r["origin"]}))
    return out


def sample(items: list, n: int | None, seed: int = 7) -> list:
    if not n or n >= len(items):
        return items
    rng = random.Random(seed)
    pos = [x for x in items if x[1]["label"] == "error"]
    neg = [x for x in items if x[1]["label"] != "error"]
    k = round(n * len(pos) / len(items))
    return rng.sample(pos, min(k, len(pos))) + rng.sample(neg, min(n - k, len(neg)))


def family_of(t: str) -> str:
    if t in T1_TYPE_FAMILY:
        return T1_TYPE_FAMILY[t]
    try:
        return TYPES[ErrorType(t)].family.value
    except ValueError:
        return "?"


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def score(items: list[tuple[PairInput, dict]], flags_by_pair: dict[str, list]) -> dict:
    tp = fp = fn = tn = 0
    by_type, by_lang, by_status = defaultdict(Counter), defaultdict(Counter), Counter()
    for p, g in items:
        flagged = bool(flags_by_pair.get(p.pair_id))
        fams = {f.family for f in flags_by_pair.get(p.pair_id, [])} | {family_of(f.type) for f in flags_by_pair.get(p.pair_id, [])}
        if g["label"] == "error":
            tp += flagged
            fn += not flagged
            for t in g["types"]:
                by_type[t]["n"] += 1
                by_type[t]["detected"] += flagged
                by_type[t]["typed"] += flagged and family_of(t) in fams
        else:
            fp += flagged
            tn += not flagged
            by_status[g["status"] + "_flagged"] += flagged
            by_status[g["status"]] += 1
        by_lang[g["lang"]]["tp" if (flagged and g["label"] == "error") else
                           "fn" if g["label"] == "error" else "fp" if flagged else "tn"] += 1
    out = {"overall": prf(tp, fp, fn), "n": len(items), "negatives": dict(by_status)}
    out["by_type"] = {t: {"n": c["n"], "recall": round(c["detected"] / c["n"], 3),
                          "typed_recall": round(c["typed"] / c["n"], 3)} for t, c in sorted(by_type.items())}
    out["by_lang"] = {lg: prf(c["tp"], c["fp"], c["fn"]) for lg, c in sorted(by_lang.items())}
    return out
