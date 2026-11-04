"""Second, independent transcription with Meta's Omnilingual ASR (CTC 1B, Apache 2.0), on the local GPU.

Used on the short clips of each finding: if Voice Transcribe and Omnilingual disagree at the flagged
words, the finding may come from a misheard word rather than the interpreter. Both hypotheses go to
the audio adjudicator. Omnilingual covers 1,600+ languages, so the check works for low-resource
languages where a single ASR is least reliable.

Runs the Hugging Face (Wav2Vec2ForCTC) conversion of facebook/omniASR-CTC-1B. Gotchas from the model
discussion: normalize the waveform to zero mean and unit variance, and the CTC blank is token 0
(fairseq2's convention), not the wav2vec2 pad token. No punctuation in the output.

Self-test (from second-ear/):  uv run python -m second_ear.asr.omnilingual path\\to\\clip.wav
"""

from __future__ import annotations

import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

MODEL_ID = "aadel4/omniASR-CTC-1B"
MAX_S = 40.0  # the model was trained on clips up to about 40 s


class Omnilingual:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        self.id2tok = self._vocab(model_id, hf_hub_download)

    @staticmethod
    def _vocab(model_id: str, download) -> dict[int, str]:
        import json

        for name in ("vocab.json", "tokenizer.json"):
            try:
                path = download(model_id, name)
            except Exception:  # noqa: BLE001 - try the next file
                continue
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            if name == "tokenizer.json":
                d = d.get("model", {}).get("vocab", d)
                if isinstance(d, list):  # unigram: [[piece, score], ...]
                    return {i: p[0] for i, p in enumerate(d)}
            return {int(v): k for k, v in d.items()}
        raise FileNotFoundError(f"no vocab.json or tokenizer.json in {model_id}")

    def transcribe(self, x, sr: int = 16000) -> str:
        import numpy as np

        torch = self.torch
        x = np.asarray(x, dtype=np.float32)
        if np.abs(x).max(initial=0) > 1.5:
            x = x / 32768.0
        x = x[: int(MAX_S * sr)]
        x = (x - x.mean()) / (x.std() + 1e-7)  # layer norm over the waveform: without it the head emits blanks
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(x)[None].to(self.device, self.dtype)).logits[0]
        ids = logits.argmax(-1).tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i != 0:
                tok = self.id2tok.get(i, "")
                if not (tok.startswith("<") and tok.endswith(">")):
                    out.append(tok)
            prev = i
        text = "".join(out).replace("|", " ").replace("▁", " ")
        return re.sub(r"\s+", " ", text).strip()

    def transcribe_file(self, path: Path) -> str:
        import soundfile as sf

        x, sr = sf.read(str(path), dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        if sr != 16000:
            raise ValueError(f"{path}: need 16 kHz")
        return self.transcribe(x, sr)


@lru_cache(maxsize=1)
def load() -> Omnilingual | None:
    """The shared model, or None when torch/transformers or the weights are unavailable."""
    try:
        return Omnilingual()
    except Exception as exc:  # noqa: BLE001 - dual ASR is optional
        print(f"Omnilingual ASR unavailable ({type(exc).__name__}: {str(exc)[:160]}); single ASR only")
        return None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^\w\s]", " ", s)


def supports(words: str, hypothesis: str) -> float:
    """Share of the quoted words (characters for CJK) found in the other transcription."""
    w, h = _norm(words), _norm(hypothesis)
    if re.search(r"[一-鿿]", w):
        toks = [c for c in w if not c.isspace()]
        hs = set(h)
    else:
        toks = w.split()
        hs = set(h.split())
    if not toks:
        return 1.0
    return sum(t in hs for t in toks) / len(toks)


if __name__ == "__main__":
    m = load()
    if m is None:
        sys.exit(1)
    print(f"Omnilingual on {m.device}")
    for p in sys.argv[1:]:
        print(f"{Path(p).name}: {m.transcribe_file(Path(p))}")
