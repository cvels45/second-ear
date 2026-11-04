"""Build T4: the human-labeled T1 pairs, voiced as long interpreted sessions (one per language).

T1 labels come from professional MQM raters and IndicMT annotators (no AI labels). T4 only adds speech:
each pair becomes an original turn and an interpreter rendition turn, voiced with open TTS, so the full
audio pipeline and the audio baselines are scored on human labels, and on hearings 45-70 minutes long.

  zh  speaker Mandarin -> interpreter English     (MQM newstest2020 zh-en)
  es  speaker English  -> interpreter Spanish     (MQM WMT24 en-es)
  hi  speaker English  -> interpreter Hindi       (IndicMT en-hi)
  ta  speaker English  -> interpreter Tamil       (IndicMT en-ta; Tamil voiced with Indic Parler-TTS)

Stages (from second-ear/), for --split calib (tuning) or test (locked; the build writes a lock):
  uv run python scripts/build_t4.py text --split calib
  cd tts; uv run python synth.py --in ../../data/test_audio/calib/utterances_kokoro.jsonl --out ../../data/test_audio/tts; cd ..
  cd tts_parler; uv run python synth_parler.py --in ../../data/test_audio/calib/utterances_other.jsonl --out ../../data/test_audio/tts; cd ..
  uv run python scripts/build_t4.py audio --split calib
No API calls, no cost.
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
from second_ear.settings import PKG_ROOT, load_settings  # noqa: E402

SEED = 2026
# speaker voice, interpreter voice (both Kokoro, except Tamil through Parler)
VOICES = {"zh": ("zh:m1", "zf_xiaoxiao"), "es": ("am_michael", "ef_dora"), "hi": ("am_fenrir", "hm_omega"),
          "ta": ("af_bella", "ta:f1")}


def sessions_from_pairs(pairs: list[dict], split: str) -> list[dict]:
    by_lang: dict[str, list[dict]] = {}
    for p in pairs:
        if p["split"] != split:
            continue
        lang = p["tgt_lang"] if p["src_lang"] == "en" else p["src_lang"]
        by_lang.setdefault(lang, []).append(p)
    out = []
    for lang, ps in sorted(by_lang.items()):
        ps.sort(key=lambda p: p["id"])  # document order keeps neighboring segments together
        did = f"t4{split}-{lang}"
        spk, itp = VOICES[lang]
        turns = []
        for k, p in enumerate(ps):
            oid, rid = f"{did}-t{2 * k + 1:04d}", f"{did}-t{2 * k + 2:04d}"
            lab = {"status": "error", "type": p["types"][0] if p["types"] else "distortion", "types": p["types"],
                   "pair_id": p["id"]} if p["label"] == "error" else {"status": "faithful", "pair_id": p["id"]}
            turns.append({"turn_id": oid, "speaker": "S", "lang": p["src_lang"], "text": p["src"], "kind": "original",
                          "renders": None, "label": None, "voice": spk})
            turns.append({"turn_id": rid, "speaker": "I", "lang": p["tgt_lang"], "text": p["tgt"], "kind": "rendition",
                          "renders": oid, "label": lab, "voice": itp})
        out.append({"dialogue_id": did, "lang": lang, "split": split, "turns": turns,
                    "label_source": "T1 human labels (MQM raters, IndicMT annotators); speech is TTS"})
    return out


def is_kokoro(voice: str) -> bool:
    return not voice.startswith(("ta:", "ar:"))


def stage_text(data_dir: Path, split: str) -> None:
    pairs = [json.loads(x) for x in (data_dir / "test_human" / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    sessions = sessions_from_pairs(pairs, split)
    out = data_dir / "test_audio" / split
    out.mkdir(parents=True, exist_ok=True)
    with (out / "dialogues.jsonl").open("w", encoding="utf-8") as fh:
        for d in sessions:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    utts = [{"id": t["turn_id"], "lang": t["lang"], "text": t["text"], "voice": t["voice"]} for d in sessions for t in d["turns"]]
    for name, keep in (("utterances_kokoro.jsonl", True), ("utterances_other.jsonl", False)):
        with (out / name).open("w", encoding="utf-8") as fh:
            for u in utts:
                if is_kokoro(u["voice"]) == keep:
                    fh.write(json.dumps(u, ensure_ascii=False) + "\n")
    for d in sessions:
        n = len(d["turns"]) // 2
        e = sum((t["label"] or {}).get("status") == "error" for t in d["turns"])
        print(f"{d['dialogue_id']}: {n} pairs, {e} errors, {sum(len(t['text']) for t in d['turns'])} characters")
    print(f"{len(utts)} utterances ({sum(is_kokoro(u['voice']) for u in utts)} for Kokoro). Wrote {out}")


def gap_after(turn: dict, rng: random.Random) -> float:
    return rng.uniform(0.45, 0.9) if turn["kind"] == "original" else rng.uniform(0.9, 1.6)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage_audio(data_dir: Path, split: str) -> None:
    out = data_dir / "test_audio" / split
    tts_dir, wav16, sess_dir = data_dir / "test_audio" / "tts", data_dir / "test_audio" / "tts16", out / "sessions"
    sessions = [json.loads(x) for x in (out / "dialogues.jsonl").read_text(encoding="utf-8").splitlines()]
    lock_sessions = []
    for d in sessions:
        clips = [tts_dir / f"{t['turn_id']}.wav" for t in d["turns"]]
        missing = [c.name for c in clips if not c.exists()]
        if missing:
            print(f"{d['dialogue_id']}: {len(missing)} clips missing (first: {missing[0]}); skipped")
            continue
        rng = random.Random(f"{SEED}-{d['dialogue_id']}")
        paths, gaps = [], []
        for t, c in zip(d["turns"], clips):
            p16 = wav16 / c.name
            if not p16.exists():
                audio.to_wav(c, p16)
            paths.append(p16)
            gaps.append(gap_after(t, rng))
        wav = sess_dir / f"{d['dialogue_id']}.wav"
        segs = audio.concat(paths, wav, gap_s=gaps)
        audio.to_mp3(wav, wav.with_suffix(".mp3"))
        for t, seg in zip(d["turns"], segs):
            t["start_s"], t["end_s"] = round(seg.start_s, 3), round(seg.end_s, 3)
        d["duration_s"] = round(audio.duration_s(wav), 2)
        (sess_dir / f"{d['dialogue_id']}.gold.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        labels = [{"turn_id": t["turn_id"], "text_sha256": hashlib.sha256(t["text"].encode()).hexdigest(),
                   "label": t["label"]} for t in d["turns"]]
        lock_sessions.append({"session_id": d["dialogue_id"], "duration_s": d["duration_s"], "wav_sha256": sha256_file(wav),
                              "labels_sha256": hashlib.sha256(json.dumps(labels, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
        print(f"{d['dialogue_id']}: {d['duration_s'] / 60:.1f} min, {len(d['turns']) // 2} pairs")
    if split == "test" and lock_sessions:
        lock = {"set": "t4-test", "built_from": "eval/locks/human_test.json", "sessions": lock_sessions,
                "items_sha256": hashlib.sha256(json.dumps(lock_sessions, sort_keys=True).encode()).hexdigest()}
        path = PKG_ROOT / "eval" / "locks" / "t4_test.json"
        path.write_text(json.dumps(lock, indent=1), encoding="utf-8")
        print(f"LOCK items_sha256 = {lock['items_sha256']}  ({path})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["text", "audio"])
    ap.add_argument("--split", required=True, choices=["calib", "test"])
    args = ap.parse_args()
    s = load_settings()
    (stage_text if args.stage == "text" else stage_audio)(s.data_dir, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
