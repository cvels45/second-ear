"""Whole-hearing pass with Muse Wonder: what no single-pair judge can see.

One call reads the full comparison transcript (every pair with its time), the findings so far, and,
when available, the decision. It returns:
  - consistency findings: a name, place, date, or term rendered differently across the hearing, and
    inconsistencies that exist in the English record but not in what the respondent said (the kind an
    adverse credibility finding can rest on);
  - decision links: which findings touch testimony the decision relies on, with the sentence;
  - a short summary for the attorney, with the findings ranked by how much they could matter.

Wonder's 1M-token context holds a full hearing transcript (about 40K tokens for 60 minutes).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .judges import LANG
from .llm import LLMClient

HP_VERSION = "h1"


class Consistency(BaseModel):
    kind: Literal["inconsistent_rendering", "record_only_inconsistency"]
    pair_ids: list[str] = Field(description="the pairs involved, by id")
    english_record: str = Field(description="what the English record says, quoted")
    source_says: str = Field(description="what the speaker actually said, in English")
    explanation: str


class DecisionLink(BaseModel):
    finding_id: str
    decision_sentence: str = Field(description="the sentence of the decision that relies on this testimony, quoted")
    how: str = Field(description="one short sentence: how the error could affect that finding")


class HearingReview(BaseModel):
    consistency: list[Consistency]
    decision_links: list[DecisionLink]
    ranking: list[str] = Field(description="finding ids, most consequential first")
    summary: str = Field(description="3 to 5 sentences for the attorney")


SYSTEM = """You are a senior check interpreter preparing a review of an interpreted immigration hearing for the respondent's attorney.
You receive the full comparison transcript (each pair: what the speaker said, then the interpreter's rendition), the errors already flagged pair by pair, and sometimes the decision.

1. consistency: find problems that only show across the hearing.
   - inconsistent_rendering: the same name, place, date, number, or term rendered differently in different pairs, where the speaker was consistent.
   - record_only_inconsistency: the English record shows the respondent contradicting themselves, but the source shows they did not. These matter most: credibility findings rest on them.
   Do not repeat an already flagged pair-level error unless the inconsistency adds something.
2. decision_links: for each flagged finding whose testimony the decision relies on, quote that decision sentence. Skip findings the decision does not touch. Empty if there is no decision.
3. ranking: all flagged finding ids, most consequential first (could it change a finding of fact or credibility?).
4. summary: plain words, no legal conclusions, and never call this a certified translation."""


def _mmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def transcript_text(pairs: list[dict], src_lang: str) -> str:
    name = LANG.get(src_lang, src_lang)
    out = []
    for p in pairs:
        o, r = (name, "English") if p["direction"] == "src->en" else ("English", name)
        out.append(f"[{p['pair_id']} {_mmss(p['orig_start_s'])}] SPEAKER ({o}): {p['orig_text']}\n"
                   f"    INTERPRETER ({r}): {p['rend_text']}")
    return "\n".join(out)


def findings_text(findings: list[dict]) -> str:
    return "\n".join(f"- {f['finding_id']} [{f['type']}] at {_mmss(f['rend_start_s'])} (pair {f['pair_id'] or 'none'}): "
                     f"speaker \"{f['source_words']}\" -> interpreter \"{f['rendition_words']}\". {f['explanation']}"
                     for f in findings) or "- (none)"


def hearing_review(client: LLMClient, pairs: list[dict], findings: list[dict], src_lang: str,
                   decision: str | None = None, *, model: str = "muse-wonder-1.0", reasoning: str = "low") -> HearingReview:
    msg = (f"Language pair: {LANG.get(src_lang, src_lang)} and English.\n\nCOMPARISON TRANSCRIPT\n"
           f"{transcript_text(pairs, src_lang)}\n\nFLAGGED ERRORS\n{findings_text(findings)}\n\nDECISION\n"
           f"{decision or '(not available)'}")
    res = client.chat(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
                      schema=HearingReview, reasoning_effort=reasoning, max_completion_tokens=8000,
                      prompt_cache_key=f"hearing-{HP_VERSION}", purpose="hearing_pass")
    return res.parsed
