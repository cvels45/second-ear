"""Score ASR and pairing against a dev dialogue's gold turns (times come from the TTS session builder)."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from ..pairing import Pairing, PTurn


def norm_tokens(text: str, lang: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    if lang == "zh":
        return [c for c in t if not c.isspace()]  # character error rate for Mandarin
    return t.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def map_turns(asr: list[PTurn], gold: list[dict]) -> dict[int, str]:
    """ASR turn idx -> gold turn_id with the largest time overlap."""
    out = {}
    for t in asr:
        best, bid = 0.0, None
        for g in gold:
            ov = _overlap(t.start_s, t.end_s, g["start_s"], g["end_s"])
            if ov > best:
                best, bid = ov, g["turn_id"]
        if bid:
            out[t.idx] = bid
    return out


def score_dialogue(asr: list[PTurn], pairing: Pairing, dialogue: dict) -> dict:
    gold = dialogue["turns"]
    src = dialogue["lang"]
    by_id = {g["turn_id"]: g for g in gold}
    m = map_turns(asr, gold)

    # coverage and error rate per language (ASR text grouped by the gold turn it overlaps)
    texts: dict[str, list[str]] = defaultdict(list)
    for t in asr:
        if t.idx in m:
            texts[m[t.idx]].append(t.text)
    err, ref_len, covered = defaultdict(int), defaultdict(int), defaultdict(lambda: [0, 0])
    for g in gold:
        lang = g["lang"]
        key = "src" if lang == src else "en"
        covered[key][1] += 1
        hyp = " ".join(texts.get(g["turn_id"], []))
        if hyp:
            covered[key][0] += 1
        r, h = norm_tokens(g["text"], lang), norm_tokens(hyp, lang)
        err[key] += edit_distance(r, h)
        ref_len[key] += len(r)

    # pairing: each gold rendition should be paired with the turn it renders
    gold_pairs = {(g["renders"], g["turn_id"]) for g in gold if g["kind"] == "rendition" and g["renders"]}
    pred_pairs = set()
    for p in pairing.pairs:
        og = {m.get(i) for i in p.orig} - {None}
        rg = {m.get(i) for i in p.rend} - {None}
        for o in og:
            for r in rg:
                pred_pairs.add((o, r))
    hits = gold_pairs & pred_pairs
    wrong = {pp for pp in pred_pairs - gold_pairs if by_id.get(pp[1], {}).get("kind") == "rendition"
             or by_id.get(pp[0], {}).get("kind") == "original"}
    return {
        "dialogue_id": dialogue["dialogue_id"], "lang": src,
        "coverage": {k: f"{v[0]}/{v[1]}" for k, v in covered.items()},
        "missed_turns": [g["turn_id"] for g in gold if not texts.get(g["turn_id"])],
        "error_rate": {k: round(err[k] / max(1, ref_len[k]), 4) for k in err},
        "pairs_gold": len(gold_pairs), "pairs_hit": len(hits), "pairs_wrong": len(wrong),
        "pair_recall": round(len(hits) / max(1, len(gold_pairs)), 4),
        "unpaired": [{"reason": u["reason"], "gold": m.get(u["idx"])} for u in pairing.unpaired],
    }
