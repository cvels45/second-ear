"""Label rules for the human-labeled test set (offline, tiny fixtures)."""

from __future__ import annotations

import json
import random

from second_ear.data.human_test import assign_splits, load_indicmt, load_mqm, manifest, sample

HDR = "system\tdoc\tdoc_id\tseg_id\trater\tsource\ttarget\tcategory\tseverity\n"


def _row(sys, seg, rater, tgt, cat, sev, doc="d1", src="源"):
    return f"{sys}\t{doc}\t{seg}\t{seg}\t{rater}\t{src}\t{tgt}\t{cat}\t{sev}\n"


def test_mqm_majority_rule_and_clean(tmp_path):
    rows = [
        # seg 1: two of three raters mark a major accuracy error -> error
        _row("H", 1, "r1", "He was <v>twice</v> arrested.", "Accuracy/Mistranslation", "Major"),
        _row("H", 1, "r2", "He was <v>twice</v> arrested.", "Accuracy/Omission", "Major"),
        _row("H", 1, "r3", "He was twice arrested.", "No-error", "no-error"),
        # seg 2: only one of three -> left out
        _row("H", 2, "r1", "She left.", "Accuracy/Mistranslation", "Major"),
        _row("H", 2, "r2", "She left.", "No-error", "no-error"),
        _row("H", 2, "r3", "She left.", "No-error", "no-error"),
        # seg 3: fluency minor only -> clean
        _row("H", 3, "r1", "They came.", "Fluency/Grammar", "Minor"),
        _row("H", 3, "r2", "They came.", "No-error", "no-error"),
        _row("H", 3, "r3", "They came.", "No-error", "no-error"),
        # seg 4: minor accuracy error -> left out
        _row("H", 4, "r1", "It rained.", "Accuracy/Mistranslation", "Minor"),
        _row("H", 4, "r2", "It rained.", "No-error", "no-error"),
        _row("H", 4, "r3", "It rained.", "No-error", "no-error"),
    ]
    path = tmp_path / "m.tsv"
    path.write_text(HDR + "".join(rows), encoding="utf-8")
    pairs = load_mqm(path, source_ds="t", src_lang="zh", tgt_lang="en", systems=None, human_systems={"H"},
                     seg_col="seg_id", label_source="test")
    by_seg = {p.id.split("|")[-1]: p for p in pairs}
    assert set(by_seg) == {"1", "3"}
    assert by_seg["1"].label == "error" and by_seg["1"].types == ["distortion", "omission"]
    assert by_seg["1"].tgt == "He was twice arrested." and by_seg["1"].spans == ["twice"]
    assert by_seg["1"].agree == 2 and by_seg["1"].raters == 3 and by_seg["1"].origin == "human_translation"
    # context keeps the 2 preceding segments of the document, labeled or not
    assert by_seg["3"].label == "clean" and by_seg["3"].context == [["源", "He was twice arrested."], ["源", "She left."]]


def test_indicmt_rules(tmp_path):
    recs = [
        {"src": "a", "translation": "A", "completion": [
            {"span_type": "Accuracy_Omission", "span_severity": "Very High", "span_text": "x"}]},
        {"src": "b", "translation": "B", "completion": [
            {"span_type": "Fluency_Grammar", "span_severity": "Low", "span_text": "y"}]},
        {"src": "c", "translation": "C", "completion": [
            {"span_type": "Accuracy_Mistranslation", "span_severity": "Medium", "span_text": "z"}]},
        {"src": "d", "translation": "D", "completion": [
            {"span_type": "Fluency_Grammar", "span_severity": "High", "span_text": "w"}]},
    ]
    path = tmp_path / "Hin_test.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    pairs = load_indicmt([path], tgt_lang="hi")
    assert [(p.src, p.label, p.types) for p in pairs] == [("a", "error", ["omission"]), ("b", "clean", [])]


def test_sampling_and_lock_are_deterministic(tmp_path):
    from second_ear.data.human_test import Pair

    def pool():
        return [Pair(id=f"s|{i}", source_ds="s", license="x", label_source="x", src_lang="zh", tgt_lang="en",
                     src=f"s{i}", tgt=f"t{i}", origin="mt", label="error" if i % 2 else "clean",
                     types=["omission"] if i % 6 == 1 else (["distortion"] if i % 2 else []))
                for i in range(60)]

    runs = []
    for _ in range(2):
        rng = random.Random(7)
        picked = sample(pool(), n_error=10, n_clean=10, rng=rng)
        assign_splits(picked, calib_share=0.2, rng=rng)
        runs.append(manifest(picked, {})["items_sha256"])
        assert sum(p.split == "calib" for p in picked) == 4
        assert sum("omission" in p.types for p in picked) >= 3
    assert runs[0] == runs[1]
