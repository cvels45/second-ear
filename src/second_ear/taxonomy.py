"""Error taxonomy: one source of truth for dev labels, judge prompts, and the report.

A flag is an *interpreter error* only if the rendition changes what the speaker meant in a way a
fact-finder could rely on. Style, register, word order, fillers, and repeated words are not errors.
Judges work one family at a time (fixed cached prompt prefix per family); a family groups types that
need the same evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ErrorType(StrEnum):
    OMISSION = "omission"
    ADDITION = "addition"
    DISTORTION = "distortion"
    COUNT_ORDINAL = "count_ordinal"
    REFERENT_UNIT = "referent_unit"
    DATE_DURATION_AGE = "date_duration_age"
    NAME = "name"
    YES_NO_REVERSAL = "yes_no_reversal"
    HEDGE = "hedge"
    PRONOUN_SHIFT = "pronoun_shift"
    SOFTENING = "softening"
    LEGAL_TERM = "legal_term"
    WRONG_QUESTION = "wrong_question"
    SELF_CORRECTED = "self_corrected"
    UNINTERPRETED = "uninterpreted_exchange"


class Family(StrEnum):
    NUMBERS = "numbers"            # typed checks propose; the judge confirms
    NAMES_TERMS = "names_terms"
    OMISSION_ADDITION = "omission_addition"
    MEANING = "meaning"            # distortion, polarity, self-correction
    STANCE = "stance"              # hedges and softening
    PRONOUN = "pronoun"
    QUESTION = "question"          # English question rendered into the source language
    STRUCTURE = "structure"        # found by pairing, not by a judge


@dataclass(frozen=True)
class TypeInfo:
    family: Family
    definition: str
    example: str


TYPES: dict[ErrorType, TypeInfo] = {
    ErrorType.OMISSION: TypeInfo(
        Family.OMISSION_ADDITION,
        "A fact, qualifier, actor, or clause the speaker said is missing from the rendition.",
        '"They beat me and my brother" -> "They beat me."'),
    ErrorType.ADDITION: TypeInfo(
        Family.OMISSION_ADDITION,
        "The rendition adds a fact, detail, or claim the speaker did not say.",
        '"They came at night" -> "They came at night with guns."'),
    ErrorType.DISTORTION: TypeInfo(
        Family.MEANING,
        "The rendition says something different from the source (wrong action, actor, cause, place, or "
        "relationship) without fitting a narrower type.",
        '"They released me" -> "I escaped."'),
    ErrorType.COUNT_ORDINAL: TypeInfo(
        Family.NUMBERS,
        "A count, amount, or ordinal changes, including count vs ordinal confusion.",
        '"for the second time" -> "two times"; "tres veces" -> "twice"; "$200" -> "$2,000"'),
    ErrorType.REFERENT_UNIT: TypeInfo(
        Family.NUMBERS,
        "The number is right but what it counts or measures changes (unit, object, school year vs grade).",
        '"cuarto año" (fourth year of secondary) -> "fourth grade"; "20 kilometers" -> "20 miles"'),
    ErrorType.DATE_DURATION_AGE: TypeInfo(
        Family.NUMBERS,
        "A date, time, duration, frequency, or age changes.",
        '"in March 2019" -> "in May 2019"; "three months" -> "three weeks"'),
    ErrorType.NAME: TypeInfo(
        Family.NAMES_TERMS,
        "A person, place, group, or organization name changes or is replaced by another.",
        '"the Mara Salvatrucha" -> "the police"'),
    ErrorType.YES_NO_REVERSAL: TypeInfo(
        Family.MEANING,
        "The polarity of an answer or statement flips (yes/no, did/did not, could/could not).",
        '"No, I did not report it" -> "Yes, I reported it."'),
    ErrorType.HEDGE: TypeInfo(
        Family.STANCE,
        "Certainty changes: a hedge is dropped or added (I think, maybe, about, I am not sure).",
        '"I think it was about ten people" -> "It was ten people."'),
    ErrorType.PRONOUN_SHIFT: TypeInfo(
        Family.PRONOUN,
        "Who did or experienced something changes through a pronoun or person shift (I vs we, he vs they, "
        "first person vs third person when it changes the actor).",
        '"They took us" -> "They took me"; "I wrote it" -> "We wrote it."'),
    ErrorType.SOFTENING: TypeInfo(
        Family.STANCE,
        "Severity or intensity is weakened or strengthened (kidnapped -> taken; hit -> pushed; threatened "
        "to kill -> threatened).",
        '"Me secuestraron" -> "They took me."'),
    ErrorType.LEGAL_TERM: TypeInfo(
        Family.NAMES_TERMS,
        "A legal or procedural term is rendered as a different legal concept (asylum vs refuge vs residency; "
        "arrested vs detained vs charged; deported vs returned).",
        '"I was detained" -> "I was convicted."'),
    ErrorType.WRONG_QUESTION: TypeInfo(
        Family.QUESTION,
        "An English question rendered into the source language asks something different (scope, time, "
        "actor, or polarity), so the answer responds to a different question.",
        '"Did anyone in your family report it?" -> "¿Usted lo denunció?" ("Did you report it?")'),
    ErrorType.SELF_CORRECTED: TypeInfo(
        Family.MEANING,
        "The interpreter renders an error and then corrects it. Flagged anyway: the record keeps both "
        "versions, and a decision can rely on the first.",
        '"He hit me twice... correction, three times."'),
    ErrorType.UNINTERPRETED: TypeInfo(
        Family.STRUCTURE,
        "Source-language talk between the interpreter and the speaker that is never rendered for the record.",
        "Interpreter and applicant discuss an answer in Spanish; the English record shows only the result."),
}

# Not errors: the judges' "acceptable" class
ACCEPTABLE = (
    "Meaning-preserving changes: paraphrase, reordering, register or politeness changes, fillers, "
    "false starts, repeated words, idioms rendered by sense, first person kept or restored, and "
    "clarifications the interpreter announces as their own."
)


def family_types(family: Family) -> list[ErrorType]:
    return [t for t, info in TYPES.items() if info.family == family]
