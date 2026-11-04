"""Offline tests: VAD, chunking, coverage check (fake ASR), language ID, and turn pairing."""

from __future__ import annotations

import numpy as np

from second_ear import vad
from second_ear.asr.session import transcribe_session
from second_ear.llm import Ledger, SpendGuard
from second_ear.pairing import PTurn, is_meta, lang_of, merge_splits, pair_turns, uninterpreted_runs
from second_ear.settings import Settings

SR = 16000


def tone(sec: float) -> np.ndarray:
    return (np.sin(np.arange(int(SR * sec)) / 5) * 8000).astype(np.int16)


def silence(sec: float) -> np.ndarray:
    return np.zeros(int(SR * sec), dtype=np.int16)


def test_vad_finds_regions_and_cuts_at_silence():
    x = np.concatenate([tone(2), silence(1), tone(3), silence(0.1), tone(1)])
    regs = vad.speech_regions(x, SR)
    assert len(regs) == 2  # the 0.1 s gap is filled
    assert abs(regs[0].start_s - 0) < 0.05 and abs(regs[1].end_s - 7.1) < 0.1
    big = [vad.Region(i * 10.0, i * 10.0 + 8.0) for i in range(120)]  # 20 minutes, 2 s gaps
    cuts = vad.cut_points(big, 1200.0)
    assert cuts and all(c - p <= 570 for p, c in zip([0.0, *cuts], cuts))
    assert all(any(abs(c - (r.end_s + 1.0)) < 1e-6 for r in big) for c in cuts)  # mid-gap


def test_coverage_check_rechecks_and_marks_gaps(tmp_path):
    import soundfile as sf

    x = np.concatenate([tone(2), silence(1), tone(2), silence(1), tone(2), silence(0.5)])
    wav = tmp_path / "s.wav"
    sf.write(str(wav), x, SR, subtype="PCM_16")
    calls = []

    def fake(path, **kw):
        calls.append(path.name)
        if path.name.startswith("chunk"):  # first pass "drops" the 2nd and 3rd regions
            return {"turns": [{"startMs": 0, "endMs": 2000, "transcript": "Hola."}]}
        if len(calls) == 2:  # recheck of region 2 succeeds
            return {"turns": [{"startMs": 400, "endMs": 2400, "transcript": "Hello."}]}
        return {"turns": []}  # region 3 stays empty -> gap

    s = Settings(data_dir=tmp_path / "d", private_dir=tmp_path / "p")
    ledger = Ledger(tmp_path / "l.jsonl")
    out = transcribe_session(wav, settings=s, ledger=ledger, guard=SpendGuard(ledger, 10, 10), transcribe=fake,
                             strategy="chunk")
    assert [t.source for t in out.turns] == ["vt", "recheck", "gap"]
    assert out.rechecked == 2 and out.gaps == 1
    assert abs(out.turns[1].start_s - 3.0) < 0.1  # recheck times are on the session clock


def test_utterance_strategy_sends_each_turn(tmp_path):
    import soundfile as sf

    x = np.concatenate([tone(2), silence(1), tone(2), silence(1), tone(2), silence(0.5)])
    wav = tmp_path / "s.wav"
    sf.write(str(wav), x, SR, subtype="PCM_16")
    seen = []

    def fake(path, **kw):
        seen.append(path.name)
        if path.name == "utt1.wav":
            return {"turns": []}  # one utterance comes back empty, then succeeds on the recheck
        return {"turns": [{"startMs": 150, "endMs": 2150, "transcript": path.stem}]}

    s = Settings(data_dir=tmp_path / "d", private_dir=tmp_path / "p")
    ledger = Ledger(tmp_path / "l.jsonl")
    out = transcribe_session(wav, settings=s, ledger=ledger, guard=SpendGuard(ledger, 10, 10), transcribe=fake,
                             workers=2)
    assert sorted(seen) == ["recheck1.wav", "utt0.wav", "utt1.wav", "utt2.wav"]
    assert [t.text for t in out.turns] == ["utt0", "recheck1", "utt2"]
    assert out.rechecked == 1 and out.gaps == 0
    assert abs(out.turns[2].start_s - 6.0) < 0.1


def test_lang_of_and_meta():
    assert lang_of("¿Cuántas veces lo arrestaron en total?", "es") == "src"
    assert lang_of("How many times were you arrested in total?", "es") == "en"
    assert lang_of("我被逮捕了三次。", "zh") == "src" and lang_of("Three times.", "zh") == "en"
    assert lang_of("Fui preso duas vezes em março.", "pt") == "src"
    assert is_meta("Your Honor, the interpreter needs to ask the witness to repeat.", "en")
    assert not is_meta("They arrested me and my brother.", "en")


def T(i, a, b, text, lang, meta=False):
    return PTurn(i, a, b, text, lang, meta)


def test_pairing_handles_splits_meta_and_side_talk():
    turns = [
        T(0, 0, 2, "How many times were you arrested?", "en"),
        T(1, 3, 5, "¿Cuántas veces lo arrestaron?", "src"),
        T(2, 6, 7, "Tres veces.", "src"),
        T(3, 7.1, 8, "La última en 2022.", "src"),          # ASR split, merged
        T(4, 9, 11, "Three times. The last in 2022.", "en"),
        T(5, 12, 14, "Your Honor, the interpreter requests a repetition.", "en", True),
        T(6, 15, 16, "¿Tengo que repetirlo?", "src"),        # side talk, never interpreted
        T(7, 16.5, 17, "No, siga.", "src"),
        T(8, 18, 20, "When did you leave?", "en"),
        T(9, 21, 23, "¿Cuándo salió?", "src"),
    ]
    merged = merge_splits(turns)
    assert len(merged) == 9
    true_pairs = {(0, 1), (2, 4), (8, 9)}
    text_idx = {t.text: t.idx for t in merged}

    def toy_sim(a: str, b: str) -> float:  # stands in for LaBSE: high only for real translations
        return 0.95 if (text_idx.get(a), text_idx.get(b)) in true_pairs else 0.2

    p = pair_turns(merged, sim=toy_sim)
    got = [(x.orig, x.rend) for x in p.pairs]
    assert ([0], [1]) in got and ([2], [4]) in got and ([8], [9]) in got
    reasons = {u["idx"]: u["reason"] for u in p.unpaired}
    assert reasons[5] == "meta"
    runs = uninterpreted_runs(p)
    assert len(runs) == 1 and [u["idx"] for u in runs[0]] == [6, 7]
