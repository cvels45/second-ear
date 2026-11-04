"""Fetch FLEURS clips (CC BY 4.0) without the `datasets` loading script.

`datasets` 4.x no longer runs loading scripts, so we download the raw TSV and audio tarball
from the google/fleurs dataset repo with huggingface_hub and extract only what we need.
"""

from __future__ import annotations

import csv
import tarfile
from dataclasses import dataclass
from pathlib import Path

REPO = "google/fleurs"
LANG = {
    "en": "en_us",
    "es": "es_419",
    "pt": "pt_br",
    "zh": "cmn_hans_cn",
    "hi": "hi_in",
    "ar": "ar_eg",
    "ta": "ta_in",
}


@dataclass
class Clip:
    id: str
    lang: str
    path: Path
    text: str


def _download(filename: str) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    try:
        return Path(hf_hub_download(REPO, filename, repo_type="dataset"))
    except Exception as exc:  # show what the repo actually contains, to fix the path quickly
        lang_dir = filename.split("/")[1] if filename.count("/") >= 2 else ""
        files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f"/{lang_dir}/" in f]
        raise RuntimeError(f"Could not download {REPO}/{filename}. Files for {lang_dir}: {files[:20]}") from exc


def revision(lang: str = "en", split: str = "test") -> str:
    """The google/fleurs commit the cached files come from (parsed from the Hugging Face cache path)."""
    import re

    m = re.search(r"snapshots[\\/]+([0-9a-f]{40})", str(_download(f"data/{LANG.get(lang, lang)}/{split}.tsv")))
    return m.group(1) if m else "unknown"


def rows(lang: str, split: str = "dev") -> list[tuple[str, str, str]]:
    """(sentence id, wav file name, raw transcription) for every recording in a split."""
    tsv = _download(f"data/{LANG.get(lang, lang)}/{split}.tsv")
    out = []
    with tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            wav_col = next((i for i, v in enumerate(row) if v.endswith(".wav")), None)
            if wav_col is None:
                continue
            text = row[wav_col + 1] if len(row) > wav_col + 1 else ""
            out.append((row[0], row[wav_col], text))
    return out


def fetch(lang: str, out_dir: Path, split: str = "dev", n: int = 40, ids: list[str] | None = None) -> list[Clip]:
    """Extract one recording per sentence id: the first n ids, or exactly `ids` (in that order) if given."""
    flang = LANG.get(lang, lang)
    first: dict[str, tuple[str, str]] = {}
    for rid, fname, text in rows(lang, split):
        first.setdefault(rid, (fname, text))
    order = [i for i in ids if i in first] if ids is not None else list(first)[:n]
    wanted = {first[rid][0]: (rid, first[rid][1]) for rid in order}

    dest = Path(out_dir) / flang / split
    dest.mkdir(parents=True, exist_ok=True)
    clips: list[Clip] = []
    missing = {f for f in wanted if not (dest / f).exists()}
    if missing:
        tar_path = _download(f"data/{flang}/audio/{split}.tar.gz")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                name = Path(member.name).name  # never trust archive paths
                if member.isfile() and name in missing:
                    src = tar.extractfile(member)
                    if src is not None:
                        (dest / name).write_bytes(src.read())
    for fname, (rid, text) in wanted.items():
        p = dest / fname
        if p.exists():
            clips.append(Clip(rid, lang, p, text))
    return clips
