"""Build T3: clean interpreted sessions with human voices on both sides, and their hash lock.

Each pair is one FLORES sentence read aloud by two FLEURS speakers (CC BY 4.0): one in the source
language and one in English. FLORES translations are professional human translations (CC BY-SA 4.0),
so no pair contains a planted error. Every flag the pipeline raises on these sessions counts as a
false alarm (target: 2 or fewer per clean hour). Directions alternate: source -> English, then
English -> source, as in a hearing.

Uses the FLEURS test split (dev is used for the FLEURS baseline sessions), 20 minutes per language.
First run downloads the test audio for 7 languages, a few GB. No API calls, no cost.

Writes:
  data/test_clean/<lang>/session.wav, session.mp3, gold.json   (gitignored)
  second-ear/eval/locks/clean_sessions.json                    (committed: IDs and hashes only)

Usage (from second-ear/):  uv run python scripts/build_clean_sessions.py [--minutes 20] [--langs es zh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_ear import audio  # noqa: E402
from second_ear.data import fleurs  # noqa: E402
from second_ear.settings import PKG_ROOT, load_settings  # noqa: E402

SEED = 2026
LANGS = ["es", "pt", "zh", "hi", "ar", "ta"]
GAP_AFTER_ORIGINAL = 0.7
GAP_AFTER_RENDITION = 1.3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_language(lang: str, minutes: float, data_dir: Path) -> tuple[dict, list[dict]]:
    rng = random.Random(f"{SEED}-{lang}")  # per language, so --langs subsets give the same pairs
    src_ids = {rid for rid, _, _ in fleurs.rows(lang, "test")}
    en_ids = {rid for rid, _, _ in fleurs.rows("en", "test")}
    common = sorted(src_ids & en_ids, key=lambda x: int(x) if x.isdigit() else x)
    rng.shuffle(common)
    candidates = common[: int(minutes * 6) + 20]  # about 10 s per clip, so 3 pairs per minute, plus spares
    src = {c.id: c for c in fleurs.fetch(lang, data_dir / "fleurs", split="test", ids=candidates)}
    en = {c.id: c for c in fleurs.fetch("en", data_dir / "fleurs", split="test", ids=candidates)}

    work = data_dir / "test_clean" / lang
    clips_dir = work / "clips"
    seq, gaps, meta, total = [], [], [], 0.0
    for i, rid in enumerate(rid for rid in candidates if rid in src and rid in en):
        if total >= minutes * 60:
            break
        a, b = (src[rid], en[rid]) if i % 2 == 0 else (en[rid], src[rid])
        wavs = []
        for c in (a, b):
            out = clips_dir / f"{c.lang}_{c.id}.wav"
            if not out.exists():
                audio.to_wav(c.path, out)
            wavs.append(out)
        seq += wavs
        gaps += [GAP_AFTER_ORIGINAL, GAP_AFTER_RENDITION]
        total += sum(audio.duration_s(w) for w in wavs) + GAP_AFTER_ORIGINAL + GAP_AFTER_RENDITION
        meta.append((rid, a, b))

    wav = work / "session.wav"
    segs = audio.concat(seq, wav, gap_s=gaps)
    audio.to_mp3(wav, work / "session.mp3")
    pairs = []
    for k, (rid, a, b) in enumerate(meta):
        so, sr = segs[2 * k], segs[2 * k + 1]
        pairs.append({
            "pair_id": f"t3-{lang}-{k:03d}", "fleurs_id": rid, "direction": f"{a.lang}->{b.lang}", "label": "clean",
            "original": {"lang": a.lang, "text": a.text, "start_s": round(so.start_s, 3), "end_s": round(so.end_s, 3)},
            "rendition": {"lang": b.lang, "text": b.text, "start_s": round(sr.start_s, 3), "end_s": round(sr.end_s, 3)},
        })
    gold = {
        "session_id": f"t3-{lang}", "lang": lang, "split": "test", "duration_s": round(audio.duration_s(wav), 2),
        "label_source": "FLORES professional human translations read by FLEURS speakers; clean by construction",
        "licenses": {"audio": "FLEURS, CC BY 4.0", "text": "FLORES-200, CC BY-SA 4.0"},
        "fleurs_revision": fleurs.revision(lang, "test"),
        "pairs": pairs,
    }
    (work / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
    items = [{"id": p["pair_id"], "fleurs_id": p["fleurs_id"], "direction": p["direction"], "label": "clean",
              "sha256": hashlib.sha256(f"{p['original']['text']}\n{p['rendition']['text']}".encode()).hexdigest()}
             for p in pairs]
    session = {"session_id": gold["session_id"], "duration_s": gold["duration_s"], "pairs": len(pairs),
               "wav_sha256": sha256_file(wav), "fleurs_revision": gold["fleurs_revision"]}
    return session, items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=float, default=20.0, help="minutes per language (default 20)")
    ap.add_argument("--langs", nargs="+", default=LANGS)
    args = ap.parse_args()
    s = load_settings()
    sessions, items = [], []
    for lang in args.langs:
        session, its = build_language(lang, args.minutes, s.data_dir)
        sessions.append(session)
        items += its
        print(f"{lang}: {session['pairs']} pairs, {session['duration_s'] / 60:.1f} min")
    body = json.dumps(sorted(items, key=lambda r: r["id"]), ensure_ascii=False, sort_keys=True)
    lock = {
        "sources": {"fleurs": "google/fleurs test split, CC BY 4.0", "flores": "FLORES-200 text, CC BY-SA 4.0",
                    "rules": {"seed": SEED, "minutes_per_language": args.minutes, "langs": args.langs}},
        "sessions": sessions,
        "items_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "items": sorted(items, key=lambda r: r["id"]),
    }
    path = PKG_ROOT / "eval" / "locks" / "clean_sessions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(x["duration_s"] for x in sessions) / 3600
    print(f"\n{len(items)} clean pairs, {total:.2f} h. Wrote {path}")
    print(f"LOCK items_sha256 = {lock['items_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
