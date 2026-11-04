"""Offline tests: time-based scoring of session findings, and uninterpreted-run detection."""

from __future__ import annotations

from second_ear.eval.e2e_eval import aggregate, gold_units, score_session
from second_ear.pairing import Pairing, uninterpreted_runs


def _gold():
    def t(i, kind, lang, status=None, typ=None):
        return {"turn_id": f"t{i}", "kind": kind, "lang": lang, "start_s": 3.0 * i, "end_s": 3.0 * i + 2.5,
                "label": {"status": status, "type": typ} if status else None}
    turns = [t(0, "original", "en"), t(1, "rendition", "es", "faithful"),
             t(2, "original", "es"), t(3, "rendition", "en", "error", "count_ordinal"),
             t(4, "side", "es", "error", "uninterpreted_exchange"), t(5, "side", "es"),
             t(6, "original", "es"), t(7, "rendition", "en", "acceptable")]
    return {"dialogue_id": "d", "lang": "es", "turns": turns, "duration_s": 24.0}


def _f(fid, pid, a, b, fam="numbers", typ="count_ordinal"):
    return {"finding_id": fid, "pair_id": pid, "rend_start_s": a, "rend_end_s": b, "family": fam, "type": typ}


def test_gold_units_merge_side_runs():
    u = gold_units(_gold())
    assert [x["status"] for x in u] == ["faithful", "error", "error", "acceptable"]
    side = [x for x in u if x["kind"] == "side"][0]
    assert (side["start_s"], side["end_s"]) == (12.0, 17.5)


def test_score_session_counts_items_once():
    fs = [_f("a", "p2", 9.1, 11.4), _f("b", "p2", 9.1, 11.4, "meaning", "distortion"),  # two findings, one pair
          _f("u", "", 12.0, 17.4, "structure", "uninterpreted_exchange"),
          _f("c", "p4", 21.0, 23.5, "stance", "hedge")]  # on the acceptable rendition
    sc = score_session(fs, _gold(), pairs=[{"rend_start_s": 9.0, "rend_end_s": 11.5}])
    assert (sc["errors"], sc["hits"], sc["fp"]) == (2, 2, 1)
    assert sc["fp_by_status"] == {"acceptable": 1} and sc["reachable"] == 2
    m = aggregate([sc])
    assert m["overall"]["recall"] == 1.0 and m["number_date_recall"] == 1.0
    assert m["false_alarms_per_hour"] == 150.0


def test_single_long_unpaired_turn_is_a_run():
    p = Pairing([], [{"idx": 5, "pos": 5, "lang": "src", "reason": "no_rendition", "start_s": 10.0, "end_s": 13.0,
                      "text": "x"},
                     {"idx": 9, "pos": 9, "lang": "src", "reason": "no_rendition", "start_s": 30.0, "end_s": 31.0,
                      "text": "y"}])
    assert uninterpreted_runs(p) == []
    assert [r[0]["idx"] for r in uninterpreted_runs(p, min_single_s=2.0)] == [5]
