"""Hybrid (dense + BM25) index with Reciprocal Rank Fusion."""
from __future__ import annotations

import numpy as np

from .bm25 import BM25
from .chunking import Chunk


class DenseStore:
    """Cosine search over normalised vectors. Uses FAISS when installed, else numpy."""

    def __init__(self, vectors: np.ndarray, backend: str = "auto"):
        self.vectors = np.asarray(vectors, dtype="float32")
        self.faiss = None
        if backend in ("auto", "faiss"):
            try:
                import faiss
                self.faiss = faiss.IndexFlatIP(self.vectors.shape[1])
                self.faiss.add(self.vectors)
            except ImportError:
                if backend == "faiss":
                    raise

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.vectors))
        if self.faiss is not None:
            s, i = self.faiss.search(q.reshape(1, -1).astype("float32"), k)
            return [(int(a), float(b)) for a, b in zip(i[0], s[0]) if a >= 0]
        s = self.vectors @ q
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top]


class QdrantStore:
    """Same interface as DenseStore, backed by Qdrant (`:memory:` or a server URL)."""

    def __init__(self, vectors: np.ndarray, url: str = ":memory:", collection: str = "chunks"):
        from qdrant_client import QdrantClient, models
        self.client, self.collection = QdrantClient(url), collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE))
        self.client.upsert(collection, points=[
            models.PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(vectors)])

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        res = self.client.query_points(self.collection, query=q.tolist(), limit=k).points
        return [(int(p.id), float(p.score)) for p in res]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class HybridIndex:
    def __init__(self, embedder, backend: str = "auto"):
        self.embedder, self.backend = embedder, backend

    def build(self, chunks: list[Chunk]) -> "HybridIndex":
        self.chunks = chunks
        texts = [f"{c.section}. {c.text}" for c in chunks]
        self.embedder.fit(texts)
        self.vectors = self.embedder.encode(texts)
        self.bm25 = BM25(texts)
        self._init_store()
        return self

    def _init_store(self):
        self.store = (QdrantStore(self.vectors) if self.backend == "qdrant"
                      else DenseStore(self.vectors, self.backend))

    def search(self, query: str, k: int = 20, mode: str = "hybrid") -> list[tuple[Chunk, float]]:
        dense = self.store.search(self.embedder.encode([query])[0], k) if mode in ("dense", "hybrid") else []
        sparse = []
        if mode in ("bm25", "hybrid"):
            sc = self.bm25.scores(query)
            sparse = [(i, sc[i]) for i in np.argsort(sc)[::-1][:k] if sc[i] > 0]
        if mode == "dense":
            ranked = dense
        elif mode == "bm25":
            ranked = sparse
        else:
            ranked = rrf([[i for i, _ in dense], [i for i, _ in sparse]])[:k]
        return [(self.chunks[i], float(s)) for i, s in ranked]

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("store", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._init_store()
