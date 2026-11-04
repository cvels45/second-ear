"""Build the human-labeled text test set and its hash lock.

Clones two public datasets at pinned commits (sparse, about 90 MB), applies the fixed label rules in
second_ear/data/human_test.py, samples with a fixed seed, and splits 20% calib / 80% test.

Writes:
  data/test_human/pairs.jsonl            full records with text (gitignored; never committed)
  second-ear/eval/locks/human_test.json  IDs, labels, splits, and text hashes only (committed)
The printed items_sha256 is the lock: commit it before the pipeline sees any test item.

Usage (from second-ear/):  uv run python scripts/build_human_test.py
No API calls, no cost.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_ear.data.human_test import (  # noqa: E402
    assign_splits,
    load_indicmt,
    load_mqm,
    manifest,
    sample,
    to_jsonl,
)
from second_ear.settings import PKG_ROOT, load_settings  # noqa: E402

SEED = 2026
CALIB_SHARE = 0.2
REPOS = {
    "wmt-mqm": {
        "url": "https://github.com/google/wmt-mqm-human-evaluation",
        "commit": "29acd6999aaea3586378c9dcb58d8cc4820cede3",
        "paths": ["LICENSE", "README.md", "newstest2020/zhen/mqm_newstest2020_zhen.no-postedits.tsv",
                  "generalMT2024/mqm_generalMT2024_enes.tsv"],
        "license": "Apache-2.0",
    },
    "indicmt-eval": {
        "url": "https://github.com/ai4bharat/indicmt-eval",
        "commit": "d8a8666b7e79975d0c3490e433e3a2b98961e86e",
        "paths": ["README.md", "Dataset/Indic MT Eval"],
        "license": "CC-BY-SA-4.0 (per the AIKosh dataset listing; the repo has no top-level license file)",
    },
}
# (n_error, n_clean) per source before the calib/test split
SIZES = {"mqm20-zhen": (160, 160), "mqm24-enes": (160, 160), "indicmt-enhi": (100, 100), "indicmt-enta": (100, 100)}


def git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{out.stderr}")
    return out.stdout.strip()


def fetch(name: str, ext: Path) -> Path:
    spec, dest = REPOS[name], ext / name
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--filter=blob:none", "--no-checkout", spec["url"], str(dest))
        git("sparse-checkout", "set", "--no-cone", *spec["paths"], cwd=dest)
    git("checkout", "--quiet", spec["commit"], cwd=dest)
    head = git("rev-parse", "HEAD", cwd=dest)
    if head != spec["commit"]:
        raise SystemExit(f"{name}: expected {spec['commit']}, got {head}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ext-dir", type=Path, help="where the datasets are cloned (default: data/ext)")
    ap.add_argument("--out-dir", type=Path, help="default: data/test_human")
    args = ap.parse_args()
    s = load_settings()
    ext = args.ext_dir or s.data_dir / "ext"
    out_dir = args.out_dir or s.data_dir / "test_human"

    mqm, ind = fetch("wmt-mqm", ext), fetch("indicmt-eval", ext)
    pools = {
        "mqm20-zhen": load_mqm(
            mqm / "newstest2020/zhen/mqm_newstest2020_zhen.no-postedits.tsv", source_ds="mqm20-zhen", src_lang="zh",
            tgt_lang="en", systems={"Human-A.0", "Human-B.0"}, human_systems={"Human-A.0", "Human-B.0"},
            seg_col="seg_id",
            label_source="Google MQM, 3 professional translators per segment (Freitag et al. 2021)"),
        "mqm24-enes": load_mqm(
            mqm / "generalMT2024/mqm_generalMT2024_enes.tsv", source_ds="mqm24-enes", src_lang="en",
            tgt_lang="es", systems=None, human_systems={"refA"}, seg_col="docSegId",
            label_source="Google MQM, WMT24 metrics campaign, 1 professional rater per segment"),
        "indicmt-enhi": load_indicmt(sorted((ind / "Dataset/Indic MT Eval").glob("Hin_*.jsonl")), tgt_lang="hi"),
        "indicmt-enta": load_indicmt(sorted((ind / "Dataset/Indic MT Eval").glob("Tam_*.jsonl")), tgt_lang="ta"),
    }
    rng = random.Random(SEED)
    chosen = []
    for name, pool in pools.items():
        n_err, n_clean = SIZES[name]
        avail = (sum(p.label == "error" for p in pool), sum(p.label == "clean" for p in pool))
        picked = sample(pool, n_error=n_err, n_clean=n_clean, rng=rng,
                        prefer="test-en-speech" if name == "mqm24-enes" else None)
        print(f"{name:13} pool error={avail[0]:5} clean={avail[1]:5} -> picked "
              f"{sum(p.label == 'error' for p in picked)} error, {sum(p.label == 'clean' for p in picked)} clean")
        chosen += picked
    assign_splits(chosen, calib_share=CALIB_SHARE, rng=rng)

    sources = {k: {"url": v["url"], "commit": v["commit"], "license": v["license"]} for k, v in REPOS.items()}
    sources["rules"] = {"seed": SEED, "calib_share": CALIB_SHARE, "sizes": SIZES,
                        "module": "second_ear/data/human_test.py"}
    man = manifest(chosen, sources)
    to_jsonl(chosen, out_dir / "pairs.jsonl")
    lock = PKG_ROOT / "eval" / "locks" / "human_test.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\ncounts:", json.dumps(man["counts"], indent=1))
    print(f"\nWrote {out_dir / 'pairs.jsonl'} and {lock}")
    print(f"LOCK items_sha256 = {man['items_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
