"""Score session findings against dev gold turns by time.

The unit is a flagged item a reviewer opens: one pair (or one uninterpreted run). A flagged item is
a hit when its rendition span overlaps a gold error turn, so the reviewer lands on the right place.
Several findings on one pair count once. Gold errors no pair's rendition overlaps are "unreachable":
lost to ASR or pairing before any judge saw them, reported apart from judge misses.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

from .text_eval import family_of, prf


def _ov(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def gold_units(gold: dict) -> list[dict]:
    """Rendition turns and uninterpreted runs, with their label."""
    out, turns = [], gold["turns"]
    k = 0
    while k < len(turns):
        t = turns[k]
        lab = t.get("label") or {}
        if t["kind"] == "rendition":
            out.append({"turn_id": t["turn_id"], "start_s": t["start_s"], "end_s": t["end_s"],
                        "status": lab.get("status", "faithful"), "type": lab.get("type"), "kind": "rendition"})
        elif t["kind"] == "side" and lab.get("status") == "error":
            j = k
            while j + 1 < len(turns) and turns[j + 1]["kind"] == "side" and not turns[j + 1].get("label"):
                j += 1
            out.append({"turn_id": t["turn_id"], "start_s": t["start_s"], "end_s": turns[j]["end_s"],
                        "status": "error", "type": lab.get("type"), "kind": "side"})
            k = j
        k += 1
    return out


def flagged_items(findings: list[dict]) -> list[dict]:
    items: dict[str, dict] = {}
    for f in findings:
        key = f["pair_id"] or f["finding_id"]
        it = items.setdefault(key, {"key": key, "start_s": f["rend_start_s"], "end_s": f["rend_end_s"],
                                    "families": set(), "types": set(), "confidence": f.get("confidence", 1.0)})
        it["families"].add(f["family"])
        it["types"].add(f["type"])
    return list(items.values())


def match(item: dict, units: list[dict], tol_s: float) -> dict | None:
    """Prefer an error unit within the tolerance window; else the unit with the largest overlap."""
    a0, a1 = item["start_s"] - tol_s, item["end_s"] + tol_s
    cands = [(u, _ov(a0, a1, u["start_s"], u["end_s"])) for u in units]
    cands = [(u, o) for u, o in cands if o > 0]
    if not cands:
        return None
    errs = [(u, o) for u, o in cands if u["status"] == "error"]
    return max(errs or cands, key=lambda c: c[1])[0]


def score_session(findings: list[dict], gold: dict, pairs: list[dict] | None = None, tol_s: float = 0.5) -> dict:
    units = gold_units(gold)
    errors = [u for u in units if u["status"] == "error"]
    hit: dict[str, dict] = {}
    fp = Counter()
    ts_err, flagged = [], []
    for it in flagged_items(findings):
        u = match(it, units, tol_s)
        if u and u["status"] == "error":
            if u["turn_id"] not in hit:
                hit[u["turn_id"]] = it
                ts_err.append(abs(it["start_s"] - u["start_s"]))
                flagged.append({"start_s": round(it["start_s"], 2), "tp": True})
            continue  # extra items on an already-hit error are duplicates, not false alarms
        fp[u["status"] if u else "no_turn"] += 1
        flagged.append({"start_s": round(it["start_s"], 2), "tp": False})
    reachable = None
    if pairs is not None:  # gold errors some pair's rendition overlaps (or a run of unpaired turns for side talk)
        spans = [(p["rend_start_s"], p["rend_end_s"]) for p in pairs]
        reachable = {u["turn_id"] for u in errors
                     if u["kind"] == "side" or any(_ov(a, b, u["start_s"], u["end_s"]) > 0 for a, b in spans)}
    by_type = defaultdict(Counter)
    for u in errors:
        c = by_type[u["type"]]
        c["n"] += 1
        c["hit"] += u["turn_id"] in hit
        c["typed"] += u["turn_id"] in hit and family_of(u["type"]) in (hit[u["turn_id"]]["families"] |
                                                                      {family_of(t) for t in hit[u["turn_id"]]["types"]})
        if reachable is not None:
            c["reachable"] += u["turn_id"] in reachable
    n_fp = sum(fp.values())
    return {"session_id": gold["dialogue_id"], "lang": gold["lang"], "duration_s": gold.get("duration_s", 0.0),
            "errors": len(errors), "hits": len(hit), "fp": n_fp, "fp_by_status": dict(fp),
            "reachable": len(reachable) if reachable is not None else None,
            "clean_renditions": sum(u["status"] in ("faithful", "acceptable") for u in units),
            "ts_err_s": ts_err, "by_type": {k: dict(v) for k, v in by_type.items()},
            # item-level outcomes, for bootstrap intervals over time blocks and recall by position
            "error_items": [{"start_s": u["start_s"], "hit": u["turn_id"] in hit, "type": u["type"]} for u in errors],
            "clean_items": [{"start_s": u["start_s"]} for u in units if u["status"] in ("faithful", "acceptable")],
            "flagged_items": flagged}


def aggregate(rows: list[dict]) -> dict:
    tp = sum(r["hits"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    n = sum(r["errors"] for r in rows)
    hours = sum(r["duration_s"] for r in rows) / 3600
    clean = sum(r["clean_renditions"] for r in rows)
    by_type = defaultdict(Counter)
    for r in rows:
        for t, c in r["by_type"].items():
            by_type[t].update(c)
    num = [c for t, c in by_type.items() if family_of(t) == "numbers"]
    ts = [x for r in rows for x in r["ts_err_s"]]
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    out = {
        "sessions": len(rows), "hours": round(hours, 3), "overall": prf(tp, fp, n - tp),
        "false_alarms_per_hour": round(fp / hours, 2) if hours else None,
        "false_alarms_per_100_clean": round(100 * fp / clean, 2) if clean else None,
        "fp_by_status": dict(sum((Counter(r["fp_by_status"]) for r in rows), Counter())),
        "number_date_recall": round(sum(c["hit"] for c in num) / max(1, sum(c["n"] for c in num)), 3),
        "timestamp_error_median_s": round(median(ts), 2) if ts else None,
        "by_type": {t: {"n": c["n"], "recall": round(c["hit"] / c["n"], 3),
                        "typed_recall": round(c["typed"] / c["n"], 3),
                        **({"reachable": round(c["reachable"] / c["n"], 3)} if "reachable" in c else {})}
                    for t, c in sorted(by_type.items())},
        "by_lang": {lg: prf(sum(r["hits"] for r in rs), sum(r["fp"] for r in rs),
                            sum(r["errors"] - r["hits"] for r in rs)) for lg, rs in sorted(by_lang.items())},
    }
    thirds = Counter()
    for r in rows:
        dur = r["duration_s"] or max([e["start_s"] for e in r.get("error_items", [])] + [1.0])
        for e in r.get("error_items", []):
            k = min(2, int(3 * e["start_s"] / dur))
            thirds[(k, "n")] += 1
            thirds[(k, "hit")] += e["hit"]
    if thirds:
        out["recall_by_third"] = [round(thirds[(k, "hit")] / thirds[(k, "n")], 3) if thirds[(k, "n")] else None
                                  for k in range(3)]
    if all(r["reachable"] is not None for r in rows):
        out["reachable_share"] = round(sum(r["reachable"] for r in rows) / max(1, n), 3)
    return out


def score_links(findings: list[dict], review: dict, gold: dict, tol_s: float = 0.5) -> dict | None:
    """Decision linking and ranking from the hearing pass, against the dev decision's cited exchanges.

    cited errors: gold error renditions in the exchanges the decision relies on. A link is correct when
    the linked finding lands on a cited rendition. top1_cited: the first-ranked finding is a cited error.
    """
    dec = gold.get("decision") or {}
    cites = set(dec.get("cites", []))
    if not cites:
        return None
    units = gold_units(gold)
    by_id = {f["finding_id"]: f for f in findings}

    def unit_of(fid: str) -> dict | None:
        f = by_id.get(fid)
        if not f:
            return None
        return match({"start_s": f["rend_start_s"], "end_s": f["rend_end_s"]}, units, tol_s)

    cited_err = {u["turn_id"] for u in units if u["status"] == "error" and u["turn_id"] in cites}
    detected_cited = {u["turn_id"] for f in findings if (u := unit_of(f["finding_id"])) and u["turn_id"] in cited_err}
    linked = [unit_of(d["finding_id"]) for d in review.get("decision_links", [])]
    good = [u for u in linked if u and u["turn_id"] in cites]
    linked_cited_err = {u["turn_id"] for u in good if u["turn_id"] in cited_err}
    ranking = [r for r in review.get("ranking", []) if r in by_id]
    top = unit_of(ranking[0]) if ranking else None
    return {"cited_errors": len(cited_err), "detected_cited": len(detected_cited), "links": len(linked),
            "links_correct": len(good), "linked_cited_errors": len(linked_cited_err),
            "top1_cited": bool(top and top["turn_id"] in cited_err)}
