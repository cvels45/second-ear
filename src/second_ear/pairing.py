"""Language per turn, interpreter-meta detection, and monotonic turn pairing.

Consecutive interpreting is a monotonic sequence: each original is followed by its rendition in the
other language. Pairing is a small dynamic program over the turn sequence, which handles a rendition
split across two ASR turns (1:2), an original split across two turns (2:1), and turns that pair with
nothing (side talk in one language, interpreter remarks, gaps). Code and a small similarity model are
the right tools here; no LLM is needed to pair turns.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

SCRIPT = {"zh": r"[一-鿿]", "hi": r"[ऀ-ॿ]", "ar": r"[؀-ۿ]", "ta": r"[஀-௿]"}
# Words shared by English and Spanish/Portuguese ("no", "me", "a", "e") are left out of both lists.
STOP = {
    "en": set("the and of to is in that was for with are it on as by this you i my he she they we did not "
              "what when where how why do your have has had been were at from yes nobody never anyone someone "
              "after before because but if so then there their them his her him our us were would could will "
              "can said told about who which one two three times first second last".split()),
    "es": set("el la los las de que y en un una es por con para del se mi lo le su qué cuándo dónde cómo fue era "
              "estaba usted yo nos al como pero más sí nadie nunca también porque cuando dos tres veces llamo "
              "tenía había años después antes".split()),
    "pt": set("o os as de que e em um uma é por com para do da se meu lhe seu não quando onde como foi era "
              "estava senhor senhora eu nos ao mas mais sim ninguém nunca também porque duas três vezes "
              "tinha anos depois antes".split()),
}
EN_MORPH = re.compile(r"\b\w+(ing|ed|n't|'s|'re|'ve|'ll)\b|\bI\b")
META = re.compile(r"\b(the |this )?interpreter\b|\bfor the record, the\b", re.I)


def lang_of(text: str, src: str) -> str:
    """'src', 'en', or 'unk', restricted to the hearing's language pair."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unk"
    if src in SCRIPT:
        share = len(re.findall(SCRIPT[src], text)) / len(letters)
        return "src" if share > 0.3 else "en"
    words = re.findall(r"[a-záéíóúñüãõâêôçà]+", text.lower())
    s = sum(w in STOP.get(src, set()) for w in words) + 0.75 * len(re.findall(r"[áéíóúñãõâêôçà¿¡]", text.lower()))
    e = sum(w in STOP["en"] for w in words) + 0.75 * len(EN_MORPH.findall(text))
    if s == e:
        return "unk"
    return "src" if s > e else "en"


def is_question(text: str) -> bool:
    return text.strip().endswith(("?", "？", "؟"))


def is_meta(text: str, lang: str) -> bool:
    """Court interpreters refer to themselves in the third person ("the interpreter requests...")."""
    return lang == "en" and bool(META.search(text))


@dataclass
class PTurn:
    idx: int
    start_s: float
    end_s: float
    text: str
    lang: str  # src | en | unk
    meta: bool = False
    source: str = "vt"

    @property
    def dur(self) -> float:
        return max(0.01, self.end_s - self.start_s)


@dataclass
class Pair:
    orig: list[int]
    rend: list[int]
    direction: str  # "en->src" or "src->en"
    orig_text: str
    rend_text: str
    start_s: float
    end_s: float
    score: float


@dataclass
class Pairing:
    pairs: list[Pair]
    unpaired: list[dict] = field(default_factory=list)  # {"idx", "lang", "reason"}


Sim = Callable[[str, str], float]


def lexical_sim(a: str, b: str) -> float:
    """Language-independent fallback: shared digits and capitalized tokens (names, places). 0.5 when neither side has any."""
    def toks(s: str) -> set[str]:
        return set(re.findall(r"\d+", s)) | {w.lower() for w in re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}\b", s)}
    ta, tb = toks(a), toks(b)
    if not ta and not tb:
        return 0.5
    return 0.3 + 0.7 * len(ta & tb) / max(1, len(ta | tb))


def merge_splits(turns: list[PTurn], max_gap_s: float = 0.45) -> list[PTurn]:
    """Join consecutive same-language turns the ASR split mid-utterance.

    A short fragment whose language can't be told ("first call?") takes the language of the turn it
    touches (the previous one if close enough, else the next), so it merges instead of breaking a pair.
    """
    turns = list(turns)
    for k, t in enumerate(turns):
        if t.lang != "unk" or not t.text:
            continue
        prev = turns[k - 1] if k > 0 else None
        nxt = turns[k + 1] if k + 1 < len(turns) else None
        if prev and prev.lang != "unk" and t.start_s - prev.end_s <= 0.8:
            t.lang = prev.lang
        elif nxt and nxt.lang != "unk" and nxt.start_s - t.end_s <= 0.8:
            t.lang = nxt.lang
    out: list[PTurn] = []
    for t in turns:
        p = out[-1] if out else None
        if p and p.lang == t.lang and not p.meta and not t.meta and t.start_s - p.end_s <= max_gap_s:
            out[-1] = PTurn(p.idx, p.start_s, t.end_s, f"{p.text} {t.text}".strip(), p.lang, False, p.source)
        else:
            out.append(t)
    return out


PARAMS_FILE = Path(__file__).resolve().parents[2] / "eval" / "pairing.json"


@lru_cache(maxsize=1)
def tuned_params() -> dict:
    """Pairing settings chosen by scripts/tune_pairing.py on dev and the T4 calibration split (if run)."""
    if PARAMS_FILE.exists():
        d = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
        return {k: d[k] for k in ("max_group", "base", "skip_cost") if k in d}
    return {}


def _group_penalty(k: int) -> float:
    """0.15 for each of the 2nd and 3rd pieces (as before Nov 2), 0.05 for each piece after that."""
    return 0.15 * min(k - 1, 2) + 0.05 * max(0, k - 3)


def pair_turns(turns: list[PTurn], sim: Sim = lexical_sim, *, skip_cost: float = 0.6, w_sim: float = 1.5,
               w_dur: float = 0.4, max_group: int = 8, max_group_gap_s: float = 1.5,
               by_duration: bool = False, base: float = 1.0, skip_per_s: float = 0.25) -> Pairing:
    """Monotonic DP: each original (1-8 consecutive same-language turns) pairs with the next 1-8 turns
    in the other language, or a turn is skipped. Pieces of one group must follow each other within
    max_group_gap_s. Long answers come back from per-utterance ASR as many short turns (a 26-second
    answer as 6 turns on T4, Nov 2), so groups go up to 8; extra pieces beyond 3 cost little.
    """
    n = len(turns)

    def score(orig: list[int], rend: list[int]) -> float:
        ot = " ".join(turns[k].text for k in orig)
        rt = " ".join(turns[k].text for k in rend)
        od = sum(turns[k].dur for k in orig)
        rd = sum(turns[k].dur for k in rend)
        ratio = min(od, rd) / max(od, rd)
        # a rendition keeps the sentence type: a question renders as a question, an answer as an answer
        mood = 0.3 if is_question(ot) == is_question(rt) else -0.6
        if by_duration:  # quality times seconds covered: splitting a pair into pieces gains nothing
            return (od + rd) * (base + w_sim * (sim(ot, rt) - 0.5) + w_dur * (ratio - 0.5) + mood)
        return base + w_sim * (sim(ot, rt) - 0.5) + w_dur * (ratio - 0.5) + mood

    def usable(k: int) -> bool:
        return turns[k].lang in ("src", "en") and not turns[k].meta

    best = [0.0] * (n + 1)
    choice: list[tuple] = [()] * (n + 1)
    for i in range(n - 1, -1, -1):
        opts = [(best[i + 1] - (skip_per_s * turns[i].dur if by_duration else skip_cost), ("skip",))]
        for a in range(1, max_group + 1):  # original made of turns i..i+a-1
            orig = list(range(i, i + a))
            if orig[-1] >= n or not all(usable(k) and turns[k].lang == turns[i].lang for k in orig):
                break
            if a > 1 and turns[orig[-1]].start_s - turns[orig[-2]].end_s > max_group_gap_s:
                break
            if i + a >= n or turns[i + a].lang == turns[i].lang:
                continue  # the rendition must start with the other language
            for b in range(1, max_group + 1):  # rendition made of the next b turns
                rend = list(range(i + a, i + a + b))
                if rend[-1] >= n or not all(usable(k) and turns[k].lang == turns[i + a].lang for k in rend):
                    break
                if b > 1 and turns[rend[-1]].start_s - turns[rend[-2]].end_s > max_group_gap_s:
                    break
                opts.append((best[i + a + b] + score(orig, rend) - _group_penalty(a) - _group_penalty(b),
                             ("pair", orig, rend)))
        best[i], choice[i] = max(opts, key=lambda o: o[0])

    pairs, unpaired, i = [], [], 0
    while i < n:
        c = choice[i]
        if c[0] == "skip":
            t = turns[i]
            reason = "meta" if t.meta else ("gap" if t.source == "gap" else ("unknown_language" if t.lang == "unk"
                                                                             else "no_rendition"))
            unpaired.append({"idx": t.idx, "pos": i, "lang": t.lang, "reason": reason, "start_s": t.start_s,
                             "end_s": t.end_s, "text": t.text})
            i += 1
            continue
        _, orig, rend = c
        o = turns[orig[0]]
        pairs.append(Pair([turns[k].idx for k in orig], [turns[k].idx for k in rend],
                          "src->en" if o.lang == "src" else "en->src",
                          " ".join(turns[k].text for k in orig), " ".join(turns[k].text for k in rend),
                          o.start_s, turns[rend[-1]].end_s, round(score(orig, rend), 3)))
        i += len(orig) + len(rend)
    return Pairing(pairs, unpaired)


def uninterpreted_runs(p: Pairing, min_turns: int = 2, min_single_s: float | None = None) -> list[list[dict]]:
    """Runs of adjacent unpaired source-language turns: candidate uninterpreted exchanges.

    A run qualifies with at least min_turns turns, or, when min_single_s is set, with a single turn of
    at least that many seconds (an answer that never reached the English record at all).
    """
    def ok(run: list[dict]) -> bool:
        if len(run) >= min_turns:
            return True
        dur = run[-1].get("end_s", run[-1]["start_s"]) - run[0]["start_s"] if run else 0.0
        return bool(run) and min_single_s is not None and dur >= min_single_s

    runs, cur = [], []
    for u in sorted(p.unpaired, key=lambda u: u["pos"]):
        good = u["lang"] == "src" and u["reason"] == "no_rendition"
        if good and cur and u["pos"] == cur[-1]["pos"] + 1:
            cur.append(u)
            continue
        if ok(cur):
            runs.append(cur)
        cur = [u] if good else []
    if ok(cur):
        runs.append(cur)
    return runs
