"""Offline tests: verifier vote counting and the calibrator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from second_ear import calibrate as C
from second_ear.judges import Flag, PairInput
from second_ear.verifier import Verdict, Verification, verify_pair


class FakeClient:
    """Returns scripted verdicts per seed; claim 3 is skipped once."""

    def __init__(self):
        self.seeds = []

    def chat(self, *, seed, **kw):
        self.seeds.append(seed)
        v = [Verdict(claim=1, is_error=True, severity="major", reason=""),
             Verdict(claim=2, is_error=seed == 1000, severity="minor", reason="")]
        if seed != 1001:
            v.append(Verdict(claim=3, is_error=True, severity="minor", reason=""))
        v.append(Verdict(claim=9, is_error=True, severity="major", reason=""))  # out of range: ignored
        return SimpleNamespace(parsed=Verification(verdicts=v))


def _flag(i: int) -> Flag:
    return Flag("p1", "meaning", "polarity", f"source{i}", f"rendition{i}", "", "", "major")


def test_duplicate_claims_share_votes():
    p = PairInput("p1", "es", "src->en", "No fue él.", "It was him.", [])
    a, b = _flag(1), Flag("p1", "baseline", "distortion", "source1", "rendition1", "", "", "major")
    out = verify_pair(FakeClient(), p, [a, _flag(2), b])
    assert [(v.votes, v.yes) for v in out] == [(3, 3), (3, 1), (3, 3)]


def test_verify_pair_counts_votes():
    p = PairInput("p1", "es", "src->en", "No fue él.", "It was him.", [])
    c = FakeClient()
    out = verify_pair(c, p, [_flag(1), _flag(2), _flag(3)])
    assert c.seeds == [1000, 1001, 1002]
    assert [(v.votes, v.yes, v.major) for v in out] == [(3, 3, 3), (3, 1, 0), (3, 2, 0)]
    assert verify_pair(c, p, []) == []


def test_features_and_threshold():
    row = {"flags": [{"family": "numbers", "votes": 3, "yes": 2, "major": 1, "typed_check": True, "severity": "major"}],
           "baseline": [{"family": "baseline", "votes": 3, "yes": 3, "major": 3}]}
    x = dict(zip(C.FEATURES, C.features(row)))
    assert x["judge_max_share"] == 2 / 3 and x["base_max_share"] == 1.0 and x["typed_check"] == 1.0
    assert x["holistic_flag"] == 1.0 and x["judge_says_major"] == 1.0 and x["n_families"] == 1.0
    assert C.features({"flags": [], "baseline": []})[1:] == [0.0] * (len(C.FEATURES) - 1)

    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.4).astype(float)
    share = np.clip(y * 0.7 + rng.normal(0.15, 0.15, 300), 0, 1)
    X = np.c_[np.ones(300), share]
    oof = C.cross_val(X, y)
    thr, at = C.choose_threshold(oof, y, target_precision=0.6, prevalence=0.1)
    assert C.precision_at(at["recall"], at["fpr"], 0.1) >= 0.6
    assert at["recall"] > 0.5
    # a lower base rate needs a stricter threshold for the same precision
    thr_bal, _ = C.choose_threshold(oof, y, target_precision=0.6)
    assert thr >= thr_bal
