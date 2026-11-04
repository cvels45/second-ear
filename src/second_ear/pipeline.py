"""Session pipeline: ASR turns -> pairs -> family judges + holistic pass -> verifier -> calibrated findings.

Every finding carries the times of the original and of the rendition, so the report can cut both
audio clips. Uninterpreted exchanges (runs of source-language turns the interpreter never rendered)
come from the pairing structure, not from a model.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import calibrate
from .judges import Flag, PairInput, baseline_pair, judge_pair
from .llm import LLMClient
from .pairing import Pairing, PTurn, Sim, is_meta, lang_of, merge_splits, pair_turns, tuned_params, uninterpreted_runs
from .verifier import Verified, same_claim, verify_pair


@dataclass
class Finding:
    finding_id: str
    pair_id: str
    direction: str
    orig_start_s: float
    orig_end_s: float
    rend_start_s: float
    rend_end_s: float
    orig_text: str
    rend_text: str
    type: str
    family: str
    source_words: str
    rendition_words: str
    explanation: str
    correction: str
    severity: str
    confidence: float  # calibrated probability that the pair holds a meaning error
    verifier_share: float  # share of verifier votes that kept this claim
    typed_check: bool = False
    origin: str = "judge"  # judge | holistic | structure
    check: str = "unchecked"  # set by the audio adjudicator: interpreter_error | uncertain_transcription
    adjudication: dict | None = None  # what the adjudicator heard, and why
    clips: dict | None = None  # paths of the speaker and interpreter clips


@dataclass
class PairRecord:
    pair_id: str
    direction: str
    orig_turns: list[int]
    rend_turns: list[int]
    orig_start_s: float
    orig_end_s: float
    rend_start_s: float
    rend_end_s: float
    orig_text: str
    rend_text: str
    probability: float = 0.0
    flags: list[dict] = field(default_factory=list)
    baseline: list[dict] = field(default_factory=list)


@dataclass
class SessionResult:
    session_id: str
    src_lang: str
    duration_s: float
    pairs: list[PairRecord]
    findings: list[Finding]
    unpaired: list[dict]
    threshold: float | None
    cost_usd: float = 0.0
    failed_pairs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def asr_to_pturns(asr: dict, src: str) -> list[PTurn]:
    out = []
    for i, t in enumerate(asr["turns"]):
        lang = lang_of(t["text"], src) if t["text"] else "unk"
        out.append(PTurn(i, t["start_s"], t["end_s"], t["text"], lang, is_meta(t["text"], lang), t.get("source", "vt")))
    return out


def gold_to_pturns(gold: dict) -> list[PTurn]:
    """Perfect-transcript turns from a dev gold file (for the gold-transcript ablation)."""
    src = gold["lang"]
    return [PTurn(i, t["start_s"], t["end_s"], t["text"], "en" if t["lang"] == "en" else "src",
                  is_meta(t["text"], "en" if t["lang"] == "en" else "src"), "gold")
            for i, t in enumerate(gold["turns"]) if t["lang"] in ("en", src)]


def build_pairs(session_id: str, src_lang: str, turns: list[PTurn], pairing: Pairing) -> list[tuple[PairInput, PairRecord]]:
    by_idx = {t.idx: t for t in turns}
    out, history = [], []
    for k, p in enumerate(pairing.pairs):
        o = [by_idx[i] for i in p.orig]
        r = [by_idx[i] for i in p.rend]
        pid = f"{session_id}-p{k + 1:03d}"
        rec = PairRecord(pid, p.direction, p.orig, p.rend, o[0].start_s, o[-1].end_s, r[0].start_s, r[-1].end_s,
                         p.orig_text, p.rend_text)
        out.append((PairInput(pid, src_lang, p.direction, p.orig_text, p.rend_text, list(history[-2:])), rec))
        history.append((p.orig_text, p.rend_text))
    return out


def select_claims(verified: list[Verified], n_judge: int) -> list[tuple[Verified, str]]:
    """Claims to report for a pair that passed the threshold: verified claims, best first, duplicates merged.

    Judge claims come first; a holistic claim is kept only if it is not a duplicate of a kept judge claim.
    If no claim won a verifier majority, the strongest single claim is kept so the reviewer sees why.
    """
    ranked = sorted(enumerate(verified), key=lambda kv: (-kv[1].share, -kv[1].major, kv[0]))
    kept: list[tuple[Verified, str]] = []
    for k, v in ranked:
        if v.share < 0.5:
            continue
        origin = "judge" if k < n_judge else "holistic"
        if any(same_claim(v.flag, w.flag) for w, _ in kept):
            continue
        kept.append((v, origin))
    if not kept and ranked:
        k, v = ranked[0]
        kept.append((v, "judge" if k < n_judge else "holistic"))
    return kept


class Calibrator:
    def __init__(self, path: Path | None):
        self.w, self.threshold = None, None
        if path and path.exists():
            try:
                self.w, self.threshold = calibrate.load(path)
            except ValueError as exc:  # features changed since the fit: fall back to the verifier majority
                print(f"Calibrator not used: {exc}")

    def probability(self, row: dict) -> float:
        if self.w is None:  # fallback: best verifier share across claims
            shares = [f["yes"] / f["votes"] for f in row["flags"] + row["baseline"] if f.get("votes")]
            return max(shares, default=0.0)
        return float(calibrate.predict(self.w, np.array([calibrate.features(row)]))[0])

    @property
    def cutoff(self) -> float:
        return self.threshold if self.threshold is not None else 0.5


def run_session(session_id: str, src_lang: str, turns: list[PTurn], *, client: LLMClient, sim: Sim,
                calibrator: Calibrator, duration_s: float, model: str = "muse-spark-1.3",
                reasoning: str = "minimal", workers: int = 8, holistic_model: str | None = None,
                holistic_reasoning: str | None = None) -> SessionResult:
    turns = merge_splits(turns)
    pairing = pair_turns(turns, sim=sim, **tuned_params())
    items = build_pairs(session_id, src_lang, turns, pairing)
    spent0 = client.ledger.run_spent()

    def work(item):
        p, _ = item
        fl, _info = judge_pair(client, p, model=model, reasoning=reasoning)
        bl = baseline_pair(client, p, model=holistic_model or model, reasoning=holistic_reasoning or reasoning)
        ver = verify_pair(client, p, fl + bl, model=model) if (fl or bl) else []
        return fl, bl, ver

    findings: list[Finding] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda it: _safe(work, it), items))
    for (p, rec), res in zip(items, results):
        if isinstance(res, Exception):
            failed.append(f"{p.pair_id}: {type(res).__name__}: {str(res)[:160]}")
            continue
        fl, bl, ver = res
        rec.flags = [asdict(f) for f in fl]
        rec.baseline = [asdict(f) for f in bl]
        for f, v in zip(rec.flags + rec.baseline, ver):
            f.update(votes=v.votes, yes=v.yes, major=v.major)
        rec.probability = round(calibrator.probability({"flags": rec.flags, "baseline": rec.baseline}), 4)
        if rec.probability < calibrator.cutoff:
            continue
        for v, origin in select_claims(ver, len(fl)):
            f = v.flag
            findings.append(Finding(
                f"{rec.pair_id}-f{len([x for x in findings if x.pair_id == rec.pair_id]) + 1}", rec.pair_id,
                rec.direction, rec.orig_start_s, rec.orig_end_s, rec.rend_start_s, rec.rend_end_s, rec.orig_text,
                rec.rend_text, f.type, f.family if origin == "judge" else _family_of(f.type), f.source_words,
                f.rendition_words, f.explanation, f.corrected_rendition, "major" if v.major * 2 >= v.votes else "minor",
                rec.probability, round(v.share, 3), f.typed_check, origin))

    by_idx = {t.idx: t for t in turns}
    for run in uninterpreted_runs(pairing, min_single_s=2.0):
        t0, t1 = by_idx[run[0]["idx"]], by_idx[run[-1]["idx"]]
        text = " / ".join(u["text"] for u in run)
        findings.append(Finding(f"{session_id}-u{t0.idx}", "", "src->en", t0.start_s, t1.end_s, t0.start_s, t1.end_s,
                                text, "", "uninterpreted_exchange", "structure", text[:200], "",
                                f"{len(run)} {src_lang} turn(s) with no English rendition on the record.", "",
                                "major", 1.0, 1.0, False, "structure"))
    findings.sort(key=lambda f: f.rend_start_s)
    return SessionResult(session_id, src_lang, duration_s, [rec for _, rec in items], findings, pairing.unpaired,
                         calibrator.threshold, round(client.ledger.run_spent() - spent0, 4), failed)


def _safe(fn, arg):
    try:
        return fn(arg)
    except Exception as exc:  # noqa: BLE001 - reported per pair
        if type(exc).__name__ == "BudgetExceeded":
            raise
        return exc


def _family_of(t: str) -> str:
    from .eval.text_eval import family_of

    return family_of(t)


def adjudicate_findings(result: SessionResult, wav: Path, clips_dir: Path, client: LLMClient, *,
                        model: str = "muse-spark-1.2", workers: int = 6, omni=None) -> int:
    """Run the audio adjudicator on every judge finding; returns the number that failed.

    With `omni` (an Omnilingual model), each clip is also transcribed by a second, independent ASR,
    and both hypotheses go to the adjudicator.
    """
    from .adjudicate import adjudicate, check_label, cut_clips, second_opinion

    todo = [f for f in result.findings if f.origin != "structure"]
    clips = {}
    for f in todo:  # cut clips and run the local ASR first, in one thread (one GPU)
        c = cut_clips(wav, clips_dir, f.finding_id, (f.orig_start_s, f.orig_end_s), (f.rend_start_s, f.rend_end_s))
        f.clips = {"orig": str(c.orig), "rend": str(c.rend)}
        second = None
        if omni is not None:
            try:
                second = second_opinion(omni, asdict(f), c)
            except Exception:  # noqa: BLE001 - adjudicate without it
                second = None
        clips[f.finding_id] = (c, second)

    def work(f: Finding):
        c, second = clips[f.finding_id]
        try:
            h = adjudicate(client, asdict(f), c, result.src_lang, model=model, second=second)
        except Exception as exc:  # keep the finding, unchecked, and say why
            if type(exc).__name__ == "BudgetExceeded":
                raise
            f.adjudication = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"} | ({"second_asr": second} if second else {})
            raise
        f.adjudication = h.model_dump() | ({"second_asr": second} if second else {})
        f.check = check_label(h)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(lambda f: _safe(work, f), todo))
    return sum(isinstance(o, Exception) for o in outs)


def save(result: SessionResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
