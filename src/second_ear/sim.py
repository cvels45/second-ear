"""Cross-lingual sentence similarity with LaBSE (Apache 2.0) for turn pairing.

Uses the `setu4993/LaBSE` port of Google's LaBSE, which keeps the original pooler: embeddings are the
pooler output, L2-normalized. Runs on the GPU when available (uv sync --group gpu). Similarity is the
cosine rescaled so unrelated sentences sit near 0 and translations near 1.
"""

from __future__ import annotations

from functools import lru_cache

MODEL_ID = "setu4993/LaBSE"


class LabseSim:
    def __init__(self, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device).eval()
        self._emb = lru_cache(maxsize=20000)(self._embed_one)

    def _embed_one(self, text: str):
        torch = self.torch
        enc = self.tok([text], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        with torch.inference_mode():
            out = self.model(**enc).pooler_output
        return torch.nn.functional.normalize(out, dim=-1)[0].float().cpu()

    def cosine(self, a: str, b: str) -> float:
        return float((self._emb(a) * self._emb(b)).sum())

    def __call__(self, a: str, b: str) -> float:
        return max(0.0, min(1.0, (self.cosine(a, b) - 0.2) / 0.6))
