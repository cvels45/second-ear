"""Muse Spark judges: one error family per call, strict JSON, a fixed cached prefix per family.

Each call sees one original/rendition pair plus up to two earlier pairs as context, and returns
findings for its own family only. Splitting by family keeps each prompt short and specific (and
cacheable), and lets the verifier and calibrator treat families differently. The numbers judge also
extracts typed facts from both sides, which code compares as an independent check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, create_model

from .llm import LLMClient, LLMOutputError
from .taxonomy import ACCEPTABLE, TYPES, ErrorType, Family, family_types

PROMPT_VERSION = "v2"
LANG = {"en": "English", "es": "Spanish", "pt": "Portuguese", "zh": "Mandarin Chinese", "hi": "Hindi",
        "ar": "Arabic", "ta": "Tamil", "cs": "Czech", "de": "German", "fr": "French", "ja": "Japanese"}
JUDGED = [Family.NUMBERS, Family.NAMES_TERMS, Family.OMISSION_ADDITION, Family.MEANING, Family.STANCE,
          Family.PRONOUN, Family.QUESTION]

NUM_WORDS = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|"
    r"hundred|thousand|once|twice|first|second|third|half|dozen|"
    r"uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|veinte|cien|mil|primera?|segund[ao]|"
    r"tercer[ao]?|um|uma|duas|três|quatro|nove|dez|vinte|cem|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|"
    r"janeiro|fevereiro|março|maio|junho|julho|setembro|outubro|novembro|dezembro|"
    r"monday|friday|sunday|week|month|year|años?|semanas?|mes(es)?|dias?|días?|anos?|meses|horas?|hours?|days?)\b"
    r"|[〇一二两三四五六七八九十百千万年月日岁次]"
    r"|(एक|दो|तीन|चार|पाँच|छह|सात|आठ|नौ|दस|बारह|पंद्रह|बीस|सौ|हज़ार|साल|महीने|हफ़्ते|दिन|बार)"
    r"|(واحد|اثن|ثلاث|أربع|خمس|ست|سبع|ثمان|تسع|عشر|مئ|ألف|عام|سنة|أشهر|أسبوع|أيام|يوم|مرة)"
    r"|(ஒன்று|இரண்டு|மூன்று|நான்கு|ஐந்து|ஆறு|ஏழு|எட்டு|ஒன்பது|பத்து|ஆயிரம்|ஆண்டு|மாத|வார|நாள்|முறை)",
    re.I)


def has_number(text: str) -> bool:
    return bool(NUM_WORDS.search(text))


@dataclass
class PairInput:
    pair_id: str
    src_lang: str  # the non-English language of the hearing
    direction: str  # "src->en" (answer interpreted into English) or "en->src" (question interpreted)
    orig: str
    rend: str
    context: list[tuple[str, str]] = field(default_factory=list)  # earlier (original, rendition) pairs

    @property
    def orig_lang(self) -> str:
        return self.src_lang if self.direction == "src->en" else "en"

    @property
    def rend_lang(self) -> str:
        return "en" if self.direction == "src->en" else self.src_lang


class Fact(BaseModel):
    kind: Literal["count", "ordinal", "amount", "date", "time", "duration", "frequency", "age", "measure"]
    value: str = Field(description="normalized: digits, ISO dates, units spelled out, e.g. '3', '2021-03', '200 USD per week'")
    refers_to: str = Field(description="what the number is about, in English, 1-4 words")


def finding_model(family: Family) -> type[BaseModel]:
    types = tuple(t.value for t in family_types(family))
    return create_model(
        f"Finding_{family.value}",
        type=(Literal[types], ...),  # type: ignore[valid-type]
        source_words=(str, Field(description="exact words from the ORIGINAL, or '' for an addition")),
        rendition_words=(str, Field(description="exact words from the RENDITION, or '' for an omission")),
        explanation=(str, Field(description="one sentence, in English, on how the meaning changed")),
        correction=(str, Field(description="only the corrected words for this error (a few words), not the whole sentence")),
        severity=(Literal["major", "minor"], ...),
    )


def judgment_model(family: Family) -> type[BaseModel]:
    F = finding_model(family)
    if family == Family.NUMBERS:
        return create_model(f"Judgment_{family.value}", source_facts=(list[Fact], ...),
                            rendition_facts=(list[Fact], ...), findings=(list[F], ...))  # type: ignore[valid-type]
    return create_model(f"Judgment_{family.value}", findings=(list[F], ...))  # type: ignore[valid-type]


FAMILY_FOCUS = {
    Family.NUMBERS: ("numbers, dates, times, durations, frequencies, ages, amounts, and measurements",
                     "First list every number-like fact on each side (source_facts, rendition_facts). Then report "
                     "a finding only where a fact's value, unit, or referent differs. 'For the second time' is an "
                     "ordinal, not a count of two."),
    Family.NAMES_TERMS: ("names of people, places, groups, and organizations, and legal or procedural terms",
                         "A transliteration or spelling variant of the same name is not an error. A different "
                         "legal concept is an error even when the words look close."),
    Family.OMISSION_ADDITION: ("missing or added content",
                               "Report a fact, qualifier, actor, or clause that was dropped or added. Do not report "
                               "dropped fillers, repetitions, or politeness formulas."),
    Family.MEANING: ("changed meaning: distortion, polarity reversal, and self-corrected renditions",
                     "Report what the speaker said versus what the record now says. A self-correction ('... "
                     "correction, three times') is still reported, as self_corrected."),
    Family.STANCE: ("certainty and intensity",
                    "Report dropped or added hedges (I think, maybe, about, I am not sure) and softened or "
                    "intensified actions (kidnapped vs taken, beat vs pushed, kill vs hurt)."),
    Family.PRONOUN: ("who did or experienced something",
                     "Report pronoun or person shifts that change the actor or the people affected (I vs we, he vs "
                     "they, my son vs me). Rendering in the first person on the speaker's behalf is correct."),
    Family.QUESTION: ("whether a question was interpreted as the same question",
                      "The ORIGINAL is a question from the court or a lawyer and the RENDITION is that question in "
                      "the witness's language. Report a rendition that asks something different: another scope, "
                      "time, actor, object, or polarity, so the answer would respond to a different question."),
}


def system_prompt(family: Family) -> str:
    focus, rule = FAMILY_FOCUS[family]
    defs = "\n".join(f"- {t.value}: {TYPES[t].definition} Example: {TYPES[t].example}" for t in family_types(family))
    return f"""You are a check interpreter reviewing a recorded, interpreted hearing for a lawyer. You compare what a speaker said (ORIGINAL) with what the interpreter said for the record (RENDITION), one pair at a time.

Your task in this call: {focus} only. Other reviewers check other kinds of error, so ignore everything else.

Error types you may report:
{defs}

{rule}

Not errors (never report these): {ACCEPTABLE}

Rules:
- Report an error only if the RENDITION changes what the speaker meant in a way a judge could rely on.
- Quote exact words from each side in source_words and rendition_words. Keep every field short.
- One finding per distinct error. If there is no error of your kinds, return an empty findings list.
- severity is major when the change could affect a finding of fact (who, what, when, how many, how sure); otherwise minor.
- The earlier pairs are context only; judge only the pair marked JUDGE."""


def user_prompt(p: PairInput) -> str:
    ctx = "\n".join(f"- ORIGINAL: {o}\n  RENDITION: {r}" for o, r in p.context[-2:]) or "- (none)"
    return f"""Language pair: {LANG.get(p.src_lang, p.src_lang)} and English.

Earlier pairs (context only):
{ctx}

JUDGE this pair:
ORIGINAL ({LANG.get(p.orig_lang, p.orig_lang)}, said by the speaker): {p.orig}
RENDITION ({LANG.get(p.rend_lang, p.rend_lang)}, said by the interpreter): {p.rend}"""


def families_for(p: PairInput) -> list[Family]:
    fams = [Family.NAMES_TERMS, Family.OMISSION_ADDITION, Family.MEANING, Family.STANCE, Family.PRONOUN]
    if has_number(p.orig) or has_number(p.rend):
        fams.insert(0, Family.NUMBERS)
    if p.direction == "en->src" and p.orig.strip().endswith("?"):
        fams.append(Family.QUESTION)
    return fams


def _norm(v: str) -> str:
    return re.sub(r"[^\w]", "", v.lower())


def facts_mismatch(src: list, rend: list) -> bool:
    """Code-level check: the multiset of (kind, value) differs between the two sides."""
    a = sorted((f.kind, _norm(f.value)) for f in src)
    b = sorted((f.kind, _norm(f.value)) for f in rend)
    return a != b


@dataclass
class Flag:
    pair_id: str
    family: str
    type: str
    source_words: str
    rendition_words: str
    explanation: str
    corrected_rendition: str
    severity: str
    typed_check: bool = False


def judge_pair(client: LLMClient, p: PairInput, *, model: str = "muse-spark-1.3", reasoning: str = "minimal",
               families: list[Family] | None = None, max_tokens: int = 3000) -> tuple[list[Flag], dict]:
    flags: list[Flag] = []
    info: dict = {"calls": 0, "cost": 0.0, "typed_check": None, "skipped": []}
    for fam in families or families_for(p):
        schema = judgment_model(fam)
        res = None
        for budget in (max_tokens, max_tokens * 3):  # retry once if the answer was cut off
            try:
                res = client.chat(
                    messages=[{"role": "system", "content": system_prompt(fam)},
                              {"role": "user", "content": user_prompt(p)}],
                    model=model, schema=schema, reasoning_effort=reasoning, max_completion_tokens=budget,
                    prompt_cache_key=f"judge-{fam.value}-{PROMPT_VERSION}", purpose=f"judge:{fam.value}",
                )
                break
            except LLMOutputError:
                continue
            except Exception as exc:  # noqa: BLE001 - e.g. a content-filter refusal: skip this family, keep the pair
                if type(exc).__name__ == "BudgetExceeded":
                    raise
                break
        if res is None:
            info["skipped"].append(fam.value)
            continue
        info["calls"] += 1
        info["cost"] += res.cost_usd
        out = res.parsed
        typed = False
        if fam == Family.NUMBERS:
            typed = facts_mismatch(out.source_facts, out.rendition_facts)
            info["typed_check"] = typed
        for f in out.findings:
            flags.append(Flag(p.pair_id, fam.value, f.type, f.source_words, f.rendition_words, f.explanation,
                              f.correction, f.severity, typed))
    return flags, info


# ---------------- one-prompt text baseline (same model, all types at once) ----------------

class BaselineFinding(BaseModel):
    type: Literal[tuple(t.value for t in ErrorType if t != ErrorType.UNINTERPRETED)]  # type: ignore[valid-type]
    source_words: str
    rendition_words: str
    explanation: str


class BaselineJudgment(BaseModel):
    findings: list[BaselineFinding]


BASELINE_SYSTEM = (
    "You are a check interpreter. Compare the ORIGINAL with the interpreter's RENDITION and list every error that "
    "changes the meaning (omissions, additions, changed numbers, dates, names, polarity, hedges, intensity, "
    f"pronouns, legal terms, or a changed question). Not errors: {ACCEPTABLE} Return an empty list if the rendition "
    "is faithful."
)


def baseline_pair(client: LLMClient, p: PairInput, *, model: str = "muse-spark-1.3", reasoning: str = "minimal") -> list[Flag]:
    res = None
    for budget in (3000, 9000):  # retry once if the answer was cut off
        try:
            res = client.chat(
                messages=[{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": user_prompt(p)}],
                model=model, schema=BaselineJudgment, reasoning_effort=reasoning, max_completion_tokens=budget,
                prompt_cache_key=f"baseline-text-{PROMPT_VERSION}", purpose="baseline:text",
            )
            break
        except LLMOutputError:
            continue
        except Exception as exc:  # noqa: BLE001 - a content-filter refusal counts as no findings
            if type(exc).__name__ == "BudgetExceeded":
                raise
            break
    if res is None:
        return []
    return [Flag(p.pair_id, "baseline", f.type, f.source_words, f.rendition_words, f.explanation, "", "major")
            for f in res.parsed.findings]
