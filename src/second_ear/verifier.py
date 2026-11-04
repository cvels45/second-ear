"""Verifier: re-check every claimed error on a pair with independent votes.

The family judges are tuned for recall, so they over-flag (Nov 2: 15-23% of clean pairs flagged on
dev and T1 calib). The verifier sees the pair and the list of claims and decides, claim by claim,
whether each one changes the meaning in a way a fact-finder could rely on. It runs three times at
low reasoning with different seeds; the share of "yes" votes is the calibrator's main feature.
One call per pair covers all its claims, so the cost scales with flagged pairs, not with flags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from .judges import LANG, Flag, PairInput, user_prompt
from .llm import LLMClient, LLMOutputError
from .taxonomy import ACCEPTABLE

VERIFIER_VERSION = "v2"  # v2 (Nov 2): self-corrections stay errors (minor); ambiguity resolved by context is not
_SEED_OK = True  # set False if the API rejects `seed`


class Verdict(BaseModel):
    claim: int = Field(description="the claim number")
    is_error: bool
    severity: Literal["major", "minor", "none"]
    reason: str = Field(description="one short sentence")


class Verification(BaseModel):
    verdicts: list[Verdict]


SYSTEM = f"""You are a senior court interpreter auditing another reviewer's notes on one interpreted exchange.
For each numbered claim, decide whether the RENDITION really changes what the speaker meant, in a way a judge deciding the case could rely on.

Answer is_error = false when the claim is wrong, too picky, or describes a meaning-preserving change.
Not errors: {ACCEPTABLE} Also not an error: making explicit a gender, number, or possessor the source leaves ambiguous, when nothing in the context contradicts it.
A self-correction ("... correction, three times") leaves the wrong words on the record: is_error = true, severity minor.
severity: major if it could change a finding of fact (who, what, when, how many, how sure); minor if real but unlikely to matter; none if not an error.
Judge each claim independently, and only from the words of the ORIGINAL and the RENDITION."""


def _claims_text(flags: list[Flag]) -> str:
    return "\n".join(f"{i}. [{f.type}] source: \"{f.source_words}\" | rendition: \"{f.rendition_words}\" | "
                     f"{f.explanation}" for i, f in enumerate(flags, 1))


@dataclass
class Verified:
    flag: Flag
    votes: int
    yes: int
    major: int

    @property
    def share(self) -> float:
        return self.yes / self.votes if self.votes else 0.0


def _words(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def same_claim(a: Flag, b: Flag) -> bool:
    """Two claims about the same words (judge and holistic pass often report the same error)."""
    wa, wb = _words(a.source_words) | _words(a.rendition_words), _words(b.source_words) | _words(b.rendition_words)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5


def _vote(client: LLMClient, msg: str, v: int, kw: dict):
    """One verifier vote; retries once with a larger budget if the answer was cut off."""
    global _SEED_OK
    system = {"role": "system", "content": SYSTEM}
    for budget in (6000, 12000):
        kw = kw | {"max_completion_tokens": budget}
        try:
            if _SEED_OK:
                try:
                    return client.chat(messages=[system, {"role": "user", "content": msg}], seed=1000 + v, **kw)
                except Exception as exc:  # API without `seed`: make each vote a distinct request instead
                    if "seed" not in str(exc).lower() or isinstance(exc, LLMOutputError):
                        raise
                    _SEED_OK = False
            return client.chat(messages=[system, {"role": "user", "content": f"{msg}\n\n(Audit pass {v + 1}.)"}], **kw)
        except LLMOutputError:
            continue  # cut off (finish_reason=length): retry with more room
    return None


def verify_pair(client: LLMClient, p: PairInput, flags: list[Flag], *, votes: int = 3,
                model: str = "muse-spark-1.3", reasoning: str = "low") -> list[Verified]:
    """Verifier votes for each claim. Duplicate claims are verified once and share their votes.

    A vote that fails (cut off twice, or refused by the API's content filter) is an abstention, so
    a claim can end with fewer than `votes` votes, or none (unverified).
    """
    if not flags:
        return []
    reps: list[int] = []  # index of the representative claim per group
    group_of: list[int] = []
    for i, f in enumerate(flags):
        for g, r in enumerate(reps):
            if same_claim(f, flags[r]):
                group_of.append(g)
                break
        else:
            reps.append(i)
            group_of.append(len(reps) - 1)
    uniq = [flags[r] for r in reps]
    tally = [[0, 0, 0] for _ in uniq]  # votes, yes, major
    msg = f"{user_prompt(p)}\n\nClaims to audit ({LANG.get(p.src_lang, p.src_lang)} hearing):\n{_claims_text(uniq)}"
    kw = dict(model=model, schema=Verification, reasoning_effort=reasoning,
              prompt_cache_key=f"verifier-{VERIFIER_VERSION}", purpose="verifier")
    for v in range(votes):
        try:
            res = _vote(client, msg, v, kw)
        except Exception as exc:  # noqa: BLE001 - e.g. a content-filter refusal: abstain
            if type(exc).__name__ == "BudgetExceeded":
                raise
            res = None
        if res is None:
            continue
        seen = set()
        for d in res.parsed.verdicts:
            k = d.claim - 1
            if 0 <= k < len(uniq) and k not in seen:
                seen.add(k)
                tally[k][1] += int(d.is_error)
                tally[k][2] += int(d.is_error and d.severity == "major")
        for k in range(len(uniq)):  # a claim the verifier skipped counts as a "no" vote
            tally[k][0] += 1
    return [Verified(f, *tally[g]) for f, g in zip(flags, group_of)]
