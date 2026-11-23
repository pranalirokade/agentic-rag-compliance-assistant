"""Embedders. `LSAEmbedder` is an offline fallback (TF-IDF + SVD) so the repo and tests run
with no model downloads; use `SentenceTransformerEmbedder` for real semantic search."""
from __future__ import annotations

import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-9, None)


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dim: int = 128):
        self.dim = dim

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2))
        X = self.tfidf.fit_transform(texts)
        k = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(k, random_state=0).fit(X)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return _norm(self.svd.transform(self.tfidf.transform(texts))).astype("float32")


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model
        self._m = None

    def _load(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_name)
        return self._m

    def fit(self, texts):  # pretrained: nothing to fit
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._load().encode(texts, normalize_embeddings=True), dtype="float32")

    def __getstate__(self):  # don't pickle the model weights
        return {"model_name": self.model_name, "_m": None}


def make_embedder(kind: str):
    return SentenceTransformerEmbedder() if kind in ("st", "sentence-transformers") else LSAEmbedder()
