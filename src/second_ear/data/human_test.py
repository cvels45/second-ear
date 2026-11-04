"""Human-labeled test pairs from public MQM error annotations.

Every label here was made by people (professional translators or language experts), never by a model:

- Google MQM, WMT newstest2020 zh->en: two human translations (Human-A, Human-B), 3 professional raters per
  segment (Freitag et al. 2021). Apache-2.0.
- Google MQM, WMT24 en->es: 15 MT systems plus a human reference, 1 professional rater per segment. Apache-2.0.
- IndicMT Eval en->hi and en->ta: 7 MT systems, MQM spans by language experts (Sai B et al., ACL 2023).
  CC BY-SA 4.0.

Label rules (fixed before any run):
- error: a Major (MQM) or High/Very High (IndicMT) accuracy error of type mistranslation, omission, or addition.
  With 3+ raters, at least 2 must mark one.
- clean: no rater marked any accuracy error, and nobody marked a Major error of any kind.
- Everything else (minor accuracy errors, rater disagreement, untranslated text) is left out of both classes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

MARK = re.compile(r"</?v>")
TYPE_MAP = {"mistranslation": "distortion", "omission": "omission", "addition": "addition"}
ACCURACY = {"mistranslation", "omission", "addition", "untranslated", "untranslated text", "source language fragment"}
MAX_CHARS = 600


@dataclass
class Pair:
    id: str
    source_ds: str
    license: str
    label_source: str
    src_lang: str
    tgt_lang: str
    src: str
    tgt: str
    origin: str  # "human_translation" or "mt"
    label: str  # "error" or "clean"
    types: list[str] = field(default_factory=list)
    spans: list[str] = field(default_factory=list)
    raters: int = 1
    agree: int = 0
    context: list[list[str]] = field(default_factory=list)  # up to 2 preceding [src, tgt] pairs
    split: str = ""

    def text_hash(self) -> str:
        return hashlib.sha256(f"{self.src}\n{self.tgt}".encode()).hexdigest()


def _norm_cat(cat: str) -> str:
    c = cat.strip().lower()
    return c.split("/", 1)[1] if c.startswith("accuracy/") else c


def _mode(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0] if values else ""


def load_mqm(path: Path, *, source_ds: str, src_lang: str, tgt_lang: str, systems: set[str] | None,
             human_systems: set[str], seg_col: str, label_source: str) -> list[Pair]:
    """Group MQM TSV rows (one row per error per rater) into labeled segments."""
    csv.field_size_limit(1 << 30)
    segs: dict[tuple[str, str, str], dict] = {}
    order: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        ix = {h: i for i, h in enumerate(header)}
        for row in reader:
            if len(row) < len(ix) - 1:
                continue
            system, doc, seg = row[ix["system"]], row[ix["doc"]], row[ix[seg_col]]
            if doc.startswith("canary") or (systems is not None and system not in systems):
                continue
            key = (system, doc, seg)
            s = segs.get(key)
            if s is None:
                s = segs[key] = {"src": [], "tgt": [], "raters": set(), "errs": []}
                order[(system, doc)].append((seg, key))
            s["src"].append(MARK.sub("", row[ix["source"]]))
            s["tgt"].append(MARK.sub("", row[ix["target"]]))
            s["raters"].add(row[ix["rater"]])
            cat, sev = _norm_cat(row[ix["category"]]), row[ix["severity"]].strip().lower()
            if cat not in ("no-error", "no error", "") and sev not in ("no-error", "neutral"):
                span = re.search(r"<v>(.*?)</v>", row[ix["target"]]) or re.search(r"<v>(.*?)</v>", row[ix["source"]])
                s["errs"].append((row[ix["rater"]], cat, sev, span.group(1) if span else ""))

    texts = {k: (_mode(v["src"]), _mode(v["tgt"])) for k, v in segs.items()}
    out: list[Pair] = []
    for (system, doc), seglist in order.items():
        seglist.sort(key=lambda t: int(t[0]) if t[0].isdigit() else t[0])
        for i, (seg, key) in enumerate(seglist):
            s, (src, tgt) = segs[key], texts[key]
            if not src or not tgt or len(src) > MAX_CHARS or len(tgt) > MAX_CHARS:
                continue
            n_raters = len(s["raters"])
            major_by_type: dict[str, set[str]] = defaultdict(set)
            spans: list[str] = []
            for rater, cat, sev, span in s["errs"]:
                if sev == "major" and cat in TYPE_MAP:
                    major_by_type[TYPE_MAP[cat]].add(rater)
                    if span:
                        spans.append(span)
            need = 2 if n_raters >= 3 else 1
            agreeing = set().union(*major_by_type.values()) if major_by_type else set()
            any_accuracy = any(cat in ACCURACY for _, cat, _, _ in s["errs"])
            any_major = any(sev == "major" for _, _, sev, _ in s["errs"])
            if len(agreeing) >= need:
                label, types = "error", sorted(t for t, r in major_by_type.items() if r)
            elif not any_accuracy and not any_major:
                label, types = "clean", []
            else:
                continue
            ctx = [list(texts[k]) for _, k in seglist[max(0, i - 2): i]]
            out.append(Pair(
                id=f"{source_ds}|{system}|{doc}|{seg}", source_ds=source_ds, license="Apache-2.0",
                label_source=label_source, src_lang=src_lang, tgt_lang=tgt_lang, src=src, tgt=tgt,
                origin="human_translation" if system in human_systems else "mt", label=label, types=types,
                spans=sorted(set(spans)), raters=n_raters, agree=len(agreeing), context=ctx,
            ))
    return out


def load_indicmt(paths: list[Path], *, tgt_lang: str) -> list[Pair]:
    """IndicMT Eval JSONL: one MT output per line with expert MQM spans in `completion`."""
    out: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(paths):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            d = json.loads(line)
            src, tgt = d["src"].strip(), d["translation"].strip()
            if (src, tgt) in seen or not src or not tgt or len(src) > MAX_CHARS or len(tgt) > MAX_CHARS:
                continue
            seen.add((src, tgt))
            spans = d.get("completion") or []
            high = [s for s in spans if s.get("span_severity") in ("High", "Very High")]
            acc_high = [s for s in high if s["span_type"].removeprefix("Accuracy_").lower() in TYPE_MAP
                        and s["span_type"].startswith("Accuracy_")]
            any_accuracy = any(s["span_type"].startswith("Accuracy") for s in spans)
            if acc_high:
                label = "error"
                types = sorted({TYPE_MAP[s["span_type"].removeprefix("Accuracy_").lower()] for s in acc_high})
            elif not any_accuracy and not high:
                label, types = "clean", []
            else:
                continue
            out.append(Pair(
                id=f"indicmt-en{tgt_lang}|{path.name}|{n}", source_ds=f"indicmt-en{tgt_lang}", license="CC-BY-SA-4.0",
                label_source="IndicMT Eval MQM spans by language experts (Sai B et al., ACL 2023)",
                src_lang="en", tgt_lang=tgt_lang, src=src, tgt=tgt, origin="mt", label=label, types=types,
                spans=sorted({s.get("span_text", "") for s in acc_high} - {""}), raters=1, agree=1 if acc_high else 0,
            ))
    return out


def sample(pairs: list[Pair], *, n_error: int, n_clean: int, rng: random.Random,
           prefer: str | None = None) -> list[Pair]:
    """Pick errors stratified by type (rarer types first) and an equal-sized clean set."""
    errors = [p for p in pairs if p.label == "error"]
    clean = [p for p in pairs if p.label == "clean"]
    rng.shuffle(errors)
    rng.shuffle(clean)
    if prefer:  # e.g. speech-domain segments first, stable within the shuffle
        errors.sort(key=lambda p: prefer not in p.id)
        clean.sort(key=lambda p: prefer not in p.id)
    picked: list[Pair] = []
    quota = {"addition": 0.15, "omission": 0.30}
    for t, share in quota.items():
        want = int(n_error * share)
        for p in errors:
            if want <= 0:
                break
            if t in p.types and p not in picked:
                picked.append(p)
                want -= 1
    for p in errors:
        if len(picked) >= n_error:
            break
        if p not in picked:
            picked.append(p)
    return picked[:n_error] + clean[:n_clean]


def assign_splits(pairs: list[Pair], *, calib_share: float, rng: random.Random) -> None:
    """Stratified by (source, label): calib_share goes to 'calib' (tuning allowed), the rest is the locked 'test'."""
    groups: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    for p in pairs:
        groups[(p.source_ds, p.label)].append(p)
    for group in groups.values():
        rng.shuffle(group)
        k = round(len(group) * calib_share)
        for i, p in enumerate(group):
            p.split = "calib" if i < k else "test"


def manifest(pairs: list[Pair], sources: dict) -> dict:
    rows = [{"id": p.id, "split": p.split, "label": p.label, "types": p.types, "sha256": p.text_hash()}
            for p in sorted(pairs, key=lambda p: p.id)]
    body = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    counts = Counter((p.source_ds, p.split, p.label) for p in pairs)
    return {
        "sources": sources,
        "counts": {f"{s}|{sp}|{lab}": n for (s, sp, lab), n in sorted(counts.items())},
        "items_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "items": rows,
    }


def to_jsonl(pairs: list[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in sorted(pairs, key=lambda p: p.id):
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
